"""
Implements a Gemma trainer class for Knowledge Distillation (KD)
while using Generative Adapters (Hyperformer).

This trainer:
1. Subclasses `Trainer`.
2. Adds KD logic (teacher model, alpha_kd, temp) to __init__.
3. Overrides `compute_loss` for combined CE + KLDiv loss.
4. Includes adapter-specific methods from the T5Trainer:
   - `evaluate`: To handle multi-task eval with `use_task_specific_params`.
   - `prediction_step`: To inject `task` and `task_embedding` into generate.
   - `_get_train_sampler`/`get_train_dataloader`: For `MultiTaskBatchSampler`.
   - `train`: To correctly load the best adapter-based model at the end.
"""

import collections
import math
import numpy as np
import os
import torch
from packaging import version
import torch.nn.functional as F
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel, logging
from transformers import Trainer
# from transformers.configuration_fsmt import FSMTConfig # From T5Trainer
# from transformers.utils.import_utils import is_torch_tpu_available
from typing import Any, Dict, Optional, Tuple, Union
from torch.utils.data.dataset import Dataset
from typing import List
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import (TrainOutput)
from transformers.trainer_utils import (set_seed)
from transformers.integrations import (hp_params)
from torch.optim import AdamW
# --- Dependencies from your T5Trainer ---
from adapters import MetaAdapterConfig
from utils import use_task_specific_params, reset_config
from data import MultiTaskBatchSampler
from tqdm.auto import tqdm
# We also need the optimizers and schedulers from the T5Trainer
from transformers.optimization import (
    Adafactor,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True


if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}
# --- End T5Trainer dependencies ---


if False:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class GemmaAdapterKDTrainer(Trainer):
    """
    Trainer for Knowledge Distillation with Gemma 3 AND Hyperformer Adapters.
    """
    def __init__(
        self,
        # --- KD Parameters ---
        teacher_model: PreTrainedModel,
        alpha_kd: float = 0.5,
        temperature: float = 2.0,
        
        # --- Adapter Parameters (from T5Trainer) ---
        config=None,
        data_args=None,
        dataset_sizes=None,
        adapter_config=None,
        multi_task_compute_metrics=None,
        
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # --- T5Trainer __init__ logic ---
        if config is None:
            assert isinstance(
                self.model, PreTrainedModel
            ), f"If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}"
            self.config = self._actual_model(self.model).config
        else:
            self.config = config

        self.adapter_config = adapter_config
        self.multi_task_compute_metrics = multi_task_compute_metrics
        self.dataset_sizes = dataset_sizes
        self.data_args = data_args
        
        # --- KD __init__ logic ---
        self.teacher_model = teacher_model
        self.alpha_kd = alpha_kd
        self.temperature = temperature

        # Ensure teacher model is on the same device and in eval mode
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.args.device)
            self.teacher_model.eval()

        # Handle loss padding for CausalLM
        self.pad_token_id = self.model.config.pad_token_id
        if self.pad_token_id is None:
             self.pad_token_id = self.model.config.eos_token_id
             logger.warn(f"pad_token_id not set, using eos_token_id: {self.pad_token_id}")
        
        # Standard Cross-Entropy loss for the "hard" labels
        # We use -100 as the ignore_index, which is standard for CausalLM
        self.loss_fn_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # KL Divergence loss for the "soft" teacher logits
        self.loss_fn_kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    
    # --- [Point 1 & 6: Compute Loss] ---
    # This is the new KD-aware loss function.
    # It assumes `inputs` contains `labels` (with -100) and
    # any adapter-specific keys (like `task`) from the dataloader.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the combined Knowledge Distillation loss.
        
        L_total = alpha * L_hard_CE + (1 - alpha) * L_distill_KL
        
        This function is called by `training_step` (with adapters)
        and `prediction_step` (with adapters).
        """
        
        # We need the labels for both loss calculations.
        labels = inputs.get("labels")
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        hard_loss = student_outputs.loss 
        
        ### Simple KD
        # --- Teacher Forward Pass ---
        with torch.no_grad():
            # Get teacher logits. The teacher is a *standard* Gemma model,
            # so we must pass only the inputs it understands.
            
            # Create a copy of inputs for the teacher
            teacher_inputs = inputs.copy()
            
            # --- IMPORTANT ---
            # Remove adapter-specific keys that the teacher won't understand
            teacher_inputs.pop("task", None)
            teacher_inputs.pop("task_embedding", None)
            teacher_inputs.pop("gemma_adapters", None)
            # Add any other adapter-specific keys here
            
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

        # 2. Distillation Loss (Student vs. Teacher)

        #clipping the last 64 tokens for dimension equivalance
        student_vocab_size = student_logits.shape[-1]
        if teacher_logits.shape[-1] > student_vocab_size:
            teacher_logits = teacher_logits[:, :, :student_vocab_size]

        # Apply temperature and log-softmax
        soft_student_logits = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_logits = F.log_softmax(teacher_logits / self.temperature, dim=-1)

        # We must mask the KLDiv loss just like the CE loss.
        # We only want to compute KL on tokens where labels != -100.
        if labels is not None:
            mask = (labels != -100).view(-1)
            
            flat_student_logits = soft_student_logits.view(-1, soft_student_logits.size(-1))
            flat_teacher_logits = soft_teacher_logits.view(-1, soft_teacher_logits.size(-1))

            masked_student_logits = flat_student_logits[mask]
            masked_teacher_logits = flat_teacher_logits[mask]
            
            # Ensure we have tokens to compute loss on
            if masked_student_logits.shape[0] > 0:
                distill_loss = self.temperature ** 2 * self.loss_fn_kl(
                    masked_student_logits,
                    masked_teacher_logits
                )
            else:
                distill_loss = torch.tensor(0.0, device=student_logits.device)
        else:
            # Fallback if no labels (not recommended)
            distill_loss = self.temperature ** 2 * self.loss_fn_kl(
                soft_student_logits,
                soft_teacher_logits
            )
            
        # 3. Combine Losses
        loss = self.alpha_kd * hard_loss + (1 - self.alpha_kd) * distill_loss

        return (loss, student_outputs) if return_outputs else loss

    def evaluate(self, eval_datasets: Optional[Dict[str, Dataset]] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        (Copied from T5Trainer)
        """
        results = {}
        if eval_datasets is None:
            eval_datasets = self.eval_dataset
        model_config = self.model.config
        
        task_iterator = tqdm(
            eval_datasets.items(),
            desc="Evaluating tasks",
            disable=not self.is_world_process_zero()  # only show on rank 0
        )   

        for eval_task, eval_dataset in task_iterator:
            dataset_size = len(eval_dataset)
            print(f"\n🔍 Evaluating task: {eval_task} | size = {dataset_size} examples\n")

            # Also update tqdm postfix with dataset size
            task_iterator.set_postfix({"task": eval_task, "size": dataset_size})

            self.compute_metrics = self.multi_task_compute_metrics[eval_task]
            
            # --- CRITICAL ADAPTER LOGIC ---
            use_task_specific_params(self.model, eval_task) ## Dicey ==> Update func?
            # ---
            print("hi1")
            if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
                raise ValueError("eval_dataset must implement __len__")
            print("hi2")
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            print("hi3")
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None
            )
            if self.args.tpu_metrics_debug or self.args.debug:
                xm.master_print(met.metrics_report())
            print("hi3")
            print(f"Output : {output}")
            tasks_metric = {eval_task + "_" + k: v for k, v in output.metrics.items()}
            for key in sorted(tasks_metric.keys()):
                print("hi3.1")
                logger.info(f"  {key} = {tasks_metric[key]}")
            print("hi4")
            results.update(tasks_metric)
            if len(tasks_metric) > 0:
                main_key = sorted(tasks_metric.keys())[0]
                task_iterator.set_postfix({
                    "task": eval_task,
                    "size": dataset_size,
                    main_key: tasks_metric[main_key]
                })
            # --- CRITICAL ADAPTER LOGIC ---
            reset_config(self.model, model_config)
            # ---
            
            break
        print("I am out of here")
        # Computes the average metrics across all the tasks without their corresponding losses.
        metrics = [results[key] for key in results.keys() if "loss" not in key]
        if len(metrics) > 0:
            results['eval_average_metrics'] = np.mean(metrics)
            
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, results)
        return results
    
    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.
        
        This is the full training loop from T5Trainer, modified to correctly
        load the best adapter-based model at the end.
        """
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self.current_gradient_accumulation_steps = self.args.gradient_accumulation_steps
        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.call_model_init(trial)
            self.model = model.to(self.args.device)
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        # Mixed precision training with apex (torch < 1.6)
        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )

        # Train!
        if False:
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", max_steps)
        # HF Trainer API >= 4.41 expects a start_time passed to _maybe_log_save_evaluate
        import time
        start_time = time.time()

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", self.state.global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(self.args.device)
        self._logging_loss_scalar = 0
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for epoch in range(epochs_trained, num_train_epochs):
            
            # --- CRITICAL MULTI-TASK LOGIC ---
            if isinstance(train_dataloader, DataLoader) and (
                isinstance(train_dataloader.sampler, DistributedSampler)
                or isinstance(train_dataloader.batch_sampler, MultiTaskBatchSampler) # This is the key
            ):
                if isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)
                else:
                    # This line is why we must copy the whole function
                    train_dataloader.batch_sampler.set_epoch(epoch)
            # ---

            if False:
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                #heyya
                grad_norm_for_log = None

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and _use_ddp_no_sync
                ):
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    # training_step calls compute_loss, which is our
                    # new overridden KD loss function.
                    tr_loss += self.training_step(model, inputs)
                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Return value of clip_grad_norm_ is the total norm; log it if available
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm_for_log = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        grad_norm_for_log = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        grad_norm_for_log = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if False:
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    print("I am here")
                    # New signature: (tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None)
                    self._maybe_log_save_evaluate(
                        tr_loss,
                        grad_norm_for_log,
                        model,
                        trial,
                        epoch,
                        self.args.ignore_keys_for_eval if hasattr(self.args, "ignore_keys_for_eval") else None,
                        start_time,
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
                # print(f"the model looks like this post training : {model}")
                break
            print("I am out of loop")
            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            # Epoch-end call (grad_norm not meaningful here → None)
            self._maybe_log_save_evaluate(
                tr_loss,
                None,
                model,
                trial,
                epoch,
                self.args.ignore_keys_for_eval if hasattr(self.args, "ignore_keys_for_eval") else None,
                start_time,
            )

            if self.args.tpu_metrics_debug or self.args.debug:
                if False:
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
            break
        print("I am fully out")
        if self.args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        # --- UNCOMMENTED AND FIXED ADAPTER LOGIC ---
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            # `self.model` is the unwrapped model instance.
            # We use `self.model.__class__` to call the *class method* `from_pretrained`.
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.__class__.from_pretrained(
                    self.state.best_model_checkpoint, 
                    adapter_config=self.adapter_config
                )
                self.model = self.model.to(self.args.device)
            else:
                # Fallback for non-PreTrainedModel
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, "pytorch_model.bin"))
                self.model.load_state_dict(state_dict)
        # ---
        
        if self._total_flos is not None:
            self.store_flos()
            self.log({"total_flos": self.state.total_flos})

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        return TrainOutput(
            global_step=self.state.global_step,
            training_loss=tr_loss.item() / max(self.state.global_step, 1),
            metrics={}
        )
    # --- [Point 4: Prediction Step] ---
    # This is the merged prediction_step. It handles adapter
    # kwargs *and* calls the KD-aware `compute_loss`.
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": self.model.config.max_length,  # FIXED
            "num_beams": 1,
        }

        if "task" in inputs:
            gen_kwargs["task"] = inputs["task"]

        if self.model.config.train_adapters and isinstance(self.adapter_config, MetaAdapterConfig):
            gen_kwargs["task_embedding"] = model.task_embedding_controller(inputs["task"])
        else:
            gen_kwargs["task_embedding"] = None

        # Generation
        generated_tokens = None
        if self.args.predict_with_generate and not prediction_loss_only:
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        # Compute loss
        labels = inputs.pop("labels", None)
        if labels is not None:
            inputs["labels"] = labels

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        # ---------- FIXED SECTION ----------
        if self.args.predict_with_generate:
            predictions = generated_tokens     # token IDs
        else:
            predictions = outputs.logits       # logits

        # Pad labels
        if labels is not None and labels.shape[-1] < gen_kwargs["max_new_tokens"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])

        return (loss, predictions, labels)

    # --- [Point 5: Other Functions] ---
    # These are required functions from your T5Trainer
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # (Copied from T5Trainer)
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.adafactor:
                self.optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                )
            else:
                self.optimizer = AdamW(
                    optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon
                )

        if self.lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler(num_training_steps)

    def _get_lr_scheduler(self, num_training_steps):
        # (Copied from T5Trainer)
        schedule_func = arg_to_scheduler[self.args.lr_scheduler]
        if self.args.lr_scheduler == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.args.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(self.optimizer, num_warmup_steps=self.args.warmup_steps)
        else:
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        # (Copied from T5Trainer)
        if self.dataset_sizes is None:
            return super()._get_train_sampler()
            
        if False and xm.xrt_world_size() > 1:
            num_replicas = xm.xrt_world_size()
            rank = xm.get_ordinal()
        elif self.args.local_rank != -1:
            num_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        else:
            num_replicas = 1
            rank = 0
        return MultiTaskBatchSampler(self.dataset_sizes, self.args.train_batch_size,
                                      self.args.temperature, rank=rank,
                                      num_replicas=num_replicas)

    def get_train_dataloader(self) -> DataLoader:
        # (Copied from T5Trainer)
        if self.dataset_sizes is None:
            return super().get_train_dataloader()
            
        multitask_sampler = self._get_train_sampler()
        return DataLoader(self.train_dataset, batch_sampler=multitask_sampler,
                          collate_fn=self.data_collator)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # (Copied from T5Trainer, but uses self.pad_token_id)
        if self.pad_token_id is None:
            raise ValueError("pad_token_id must be set in config to pad tensors.")

        padded_tensor = self.pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor