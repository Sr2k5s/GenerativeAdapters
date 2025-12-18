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
import json
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
from adapters import MetaAdapterConfig
from utils import use_task_specific_params, reset_config
from data import MultiTaskBatchSampler
from tqdm.auto import tqdm
from transformers.optimization import (
    Adafactor,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
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


GLOBAL_STEP = 0
if False:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)

from transformers import EvalPrediction

from metrics.metrics import build_compute_metrics_fn   



class GemmaAdapterKDTrainer(Trainer):
    """
    Trainer for Knowledge Distillation with Gemma 3 AND Hyperformer Adapters.
    """
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        alpha_kd: float = 0.5,
        temperature: float = 2.0,
        config=None,
        data_args=None,
        dataset_sizes=None,
        adapter_config=None,
        multi_task_compute_metrics=None,
        tokenizer = None,
        
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

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
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model
        self.alpha_kd = alpha_kd
        self.temperature = temperature
        self.step = 0
        self.set_model = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.set_model)
            self.teacher_model.eval()

        device = next(self.teacher_model.parameters()).device
        
        self.pad_token_id = self.model.config.pad_token_id
        if self.pad_token_id is None:
             self.pad_token_id = self.model.config.eos_token_id
             logger.warn(f"pad_token_id not set, using eos_token_id: {self.pad_token_id}")
        
        
        self.loss_fn_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)








    def _normalize_task_name(self, task: Any) -> Optional[str]:
        """
        Try to extract a simple lowercase task name ('dart', 'e2e_nlg', 'wiki_lingua')
        from whatever the dataloader put in `inputs["task"]`.
        """
        if task is None:
            return None

        if isinstance(task, str):
            return task.lower()

        if isinstance(task, (list, tuple)):
            if len(task) == 0:
                return None
            return str(task[0]).lower()

        return None

    def _get_gen_max_new_tokens(self, task: Any = None) -> int:
        """
        Choose a sane, TASK-SPECIFIC max_new_tokens instead of using config.max_length.

        Priority:
        1) Task-specific defaults:
           - dart       -> 64
           - e2e_nlg    -> 64
           - wiki_lingua-> 128
        2) args.generation_max_length (if set)
        3) data_args.eval_max_target_length (if available)
        4) fallback = 64
        """
        t = self._normalize_task_name(task)

        # ---- Task-specific defaults ----
        if t == "dart":
            return 128
        if t == "e2e_nlg":
            return 128
        if t == "wiki_lingua":
            return 128   # you can bump to 256 if your summaries are longer

        # ---- Generic overrides from training args ----
        gen_len = getattr(self.args, "generation_max_length", None)
        if gen_len is not None and gen_len > 0:
            return gen_len

        # ---- Fallback to data_args.eval_max_target_length, if available ----
        if getattr(self, "data_args", None) is not None:
            eval_max = getattr(self.data_args, "eval_max_target_length", None)
            if eval_max is not None and eval_max > 0:
                return eval_max

        # ---- Final fallback ----
        return 64

        
    
    
    def run_and_save_metrics(
        self,
        step: int,
        preds: np.ndarray,
        labels: np.ndarray,
        tokenizer,
        context_input_ids,
        task_name: str,
        output_path: str = "metrics.json"
        
        
    ):
        """
        preds   : numpy array of shape (batch, seq_len)
        labels  : numpy array of shape (batch, seq_len)
        """
        try:
            
            # Replace -100 with pad_token_id so operations don't crash
            preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
            labels = np.where(labels < 0, tokenizer.pad_token_id, labels)

            
            # We must remove the input prompt so metrics only evaluate the *new* tokens.
            trimmed_preds_list = []
            for i in range(preds.shape[0]):
                ctx_len = context_input_ids[i].shape[0]
                # Slice off the context
                trimmed = preds[i, ctx_len:]
                trimmed_preds_list.append(trimmed)

           
            max_len = max([len(t) for t in trimmed_preds_list])
            padded_preds = np.full((len(trimmed_preds_list), max_len), tokenizer.pad_token_id)
            
            for i, pred in enumerate(trimmed_preds_list):
                padded_preds[i, :len(pred)] = pred

           
            # Now we pass the TRIMMED, PADDED predictions
            eval_pred = EvalPrediction(
                predictions=padded_preds, 
                label_ids=labels
            )
            
            # print([task_name])
            metrics_fns = build_compute_metrics_fn([task_name], tokenizer)
            compute_metrics = metrics_fns[task_name]
            
            results = compute_metrics(eval_pred)
            # print(f"predictions--------->{results}")
            
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            
            pred_str = [tokenizer.decode(t, skip_special_tokens=True) for t in trimmed_preds_list]
            label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # print(f"labels string----->{label_str}")

            debug_dump = {
                "global_step": int(self.state.global_step),
                "batch_step": int(step),
                "task": task_name,
                "predictions": pred_str,
                "labels": label_str,
            }
            temp = "train_debug_"+task_name+"_" + str((int(step/100))%10) + ".json"
            with open(temp, "w") as f:
                json.dump(debug_dump, f, indent=2)

        except Exception as e:
            print(f"[DEBUG ERROR IN] Could not log predictions: {e}")
            import traceback
            traceback.print_exc() # Helps see exactly where it failed
        
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        
        for k, v in inputs.items():
            # Check if the value is a PyTorch tensor (and not the 'task' list/string)
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.set_model)
                    
        return inputs
    # --- [Point 1 & 6: Compute Loss] ---
    # This is the new KD-aware loss function
    # It assumes `inputs` contains `labels` (with -100) and
    # any adapter-specific keys (like `task`) from the dataloader.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the combined Knowledge Distillation loss.
        
        L_total = alpha * L_hard_CE + (1 - alpha) * L_distill_KL
        
        This function is called by `training_step` (with adapters)
        and `prediction_step` (with adapters).
        """
        # print(inputs.items())
        device = next(self.model.parameters()).device
        # print(f"======================================================Model Inside Compute Loss: {device}========================================")

        device = next(self.teacher_model.parameters()).device
        # print(f"======================================================Model Inside Compute Loss: {device}========================================")

        # if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
        #     # print(f"✅ Input IDs device is now: {inputs['input_ids'].device}")
    
        # if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor):
            # print(f"✅ Labels device is now: {inputs['labels'].device}")
        
        
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

        if self.step%100 == 0:
            teacher_preds = torch.argmax(soft_teacher_logits, dim=-1).detach().cpu().numpy()
            preds = np.where(teacher_preds < 0, self.tokenizer.pad_token_id, teacher_preds)
            trimmed_preds_list = []
            context_input_ids = inputs["input_ids"]
            for i in range(preds.shape[0]):
                ctx_len = context_input_ids[i].shape[0]
                # Slice off the context
                trimmed = preds[i, ctx_len:]
                trimmed_preds_list.append(trimmed)
            max_len = max([len(t) for t in trimmed_preds_list])
            padded_preds = np.full((len(trimmed_preds_list), max_len), self.tokenizer.pad_token_id)
            
            for i, pred in enumerate(trimmed_preds_list):
                padded_preds[i, :len(pred)] = pred
            label_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            pred_str = [self.tokenizer.decode(t, skip_special_tokens=True) for t in trimmed_preds_list]
            teacher_ids = torch.argmax(soft_teacher_logits, dim=-1).detach().cpu()  # (B, S)
            non_flat_mask = (inputs["labels"].detach().cpu() != -100)
            masked_teacher_tokens = []
            for i in range(teacher_ids.shape[0]):
                token_ids = teacher_ids[i][non_flat_mask[i]]
                masked_teacher_tokens.append(
                    self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
                )
            input_ids_cpu = inputs["input_ids"].detach().cpu()
            input_text = self.tokenizer.batch_decode(input_ids_cpu, skip_special_tokens=True)
            debug_dump = {
                "untrimmed_predictions": label_str,
                "trimmed_predictions": pred_str,
                "logits": preds.tolist(),
                "input_ids" : input_text,
                "masked_teacher_tokens": masked_teacher_tokens
            }
            temp = f"teacher_{inputs["task"]}_{(int(self.step/100))%10}.json"
            with open(temp, "w") as f:
                json.dump(debug_dump, f, indent=2)
            try:
                eval_input_ids = inputs["input_ids_eval"].to(self.set_model)
                eval_attention_mask = inputs["attention_mask_eval"].to(self.set_model)
                eval_labels = inputs["labels_eval"].detach().cpu().numpy()
    
                # Replace ignore_index (-100) with pad token so decoding works
                eval_labels = np.where(eval_labels < 0, self.tokenizer.pad_token_id, eval_labels)
                task_raw = inputs.get("task", "dart")
                task_name = self._normalize_task_name(task_raw) or "dart"
    
                max_new_tokens = self._get_gen_max_new_tokens(task_raw)
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "num_beams": getattr(self.args, "generation_num_beams", 1),
                }
    
                if "task" in inputs:
                    gen_kwargs["task"] = inputs["task"]
    
                if self.model.config.train_adapters and isinstance(self.adapter_config, MetaAdapterConfig):
                    gen_kwargs["task_embedding"] = model.task_embedding_controller(inputs["task"])
                else:
                    gen_kwargs["task_embedding"] = None
    
                with torch.no_grad():
                    generated_tokens = model.generate(
                        eval_input_ids,
                        attention_mask=eval_attention_mask,
                        **gen_kwargs,
                    )
    
                preds = generated_tokens.detach().cpu().numpy()
    
                
                self.run_and_save_metrics(
                    step=self.step,
                    preds=preds,
                    labels=eval_labels,
                    tokenizer=self.tokenizer,
                    context_input_ids = eval_input_ids,
                    task_name=inputs["task"],
                    output_path=f"metrics_step_{inputs["task"]}_{(int(self.step/100))%10}.json"
                    
                )
    
                # --- CLEANUP ---
                del generated_tokens
                del preds
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[DEBUG ERROR OUT] Could not log predictions: {e}")
            
            
        # 3. Combine Losses
        loss = self.alpha_kd * hard_loss + (1 - self.alpha_kd) * distill_loss

        return (loss, student_outputs) if return_outputs else loss

    def evaluate(self, eval_datasets: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None) -> Dict[str, float]:
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
            # print(f"\n🔍 Evaluating task: {eval_task} | size = {dataset_size} examples\n")

            # Also update tqdm postfix with dataset size
            task_iterator.set_postfix({"task": eval_task, "size": dataset_size})

            self.compute_metrics = self.multi_task_compute_metrics[eval_task]
            
            # --- CRITICAL ADAPTER LOGIC ---
            use_task_specific_params(self.model, eval_task) ## Dicey ==> Update func?
            # ---
            if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
                raise ValueError("eval_dataset must implement __len__")
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None
            )
            if self.args.tpu_metrics_debug or self.args.debug:
                xm.master_print(met.metrics_report())
            print(f"================================= Output : {output} ====================================")
            tasks_metric = {eval_task + "_" + k: v for k, v in output.metrics.items()}
            for key in sorted(tasks_metric.keys()):
                # print("hi3.3")
                logger.info(f"  {key} = {tasks_metric[key]}")
            # print("hi4")
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
            
            
        # print("I am out of here")
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
            self.model = model.to(self.set_model)
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        device = next(self.model.parameters()).device
        # print(f"======================================================Student Model is on: {device}========================================")
        
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
        # self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        # Mixed precision training with apex (torch < 1.6)
        model = self.model.to(self.set_model)
        # print(f"======================================================Student Model Copy is on: {next(model.parameters()).device}========================================")
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # print(f"======================================================Student Model 1 is on: {next(model.parameters()).device}========================================")   
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
        # print(f"======================================================Student Model 2 is on: {next(model.parameters()).device}========================================")
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



        # print(f"======================================================Student Model 3 is on: {next(model.parameters()).device}========================================")

        
        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", self.state.global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)



        # print(f"======================================================Student Model 4 is on: {next(model.parameters()).device}========================================")

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

        tr_loss = torch.tensor(0.0).to(self.set_model)
        self._logging_loss_scalar = 0
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)
        # print(f"======================================================Student Model 5 is on: {next(model.parameters()).device}========================================")
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
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.set_model]).per_device_loader(
                    self.set_model
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)



            # print(f"======================================================Student Model 5 is on: {next(model.parameters()).device}========================================")
            

            for step, inputs in enumerate(epoch_iterator):
                #heyya
                self.step = step
                grad_norm_for_log = None
                # inputs = self._prepare_inputs(inputs)


                # if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                #     print(f"✅ Input IDs device is now: {inputs['input_ids'].device}")
    
                # if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor):
                #     print(f"✅ Labels device is now: {inputs['labels'].device}")

                # print(f"======================================================Student Model 6 is on: {next(model.parameters()).device}========================================")
                
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
                    and hasattr(model, "no_sync")
                ):
                    # print(f"======================================================Student Model before train is on: {next(model.parameters()).device}========================================")
                    with model.no_sync():
                        step_loss = self.training_step(model, inputs)
                        tr_loss += step_loss.detach()
                else:
                    # training_step calls compute_loss, which is our
                    # new overridden KD loss function.
                    # print(f"======================================================Student Model before train is on: {next(model.parameters()).device}========================================")                    
                    step_loss = self.training_step(model, inputs)
                    tr_loss += step_loss.detach()
                    
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
                    # print("I am here")
                    # print(inputs["task"])
                    if ((step+1)/4 )% 100 == 0:
                        try:
                            # --------------------------
                            # Use EVAL-style inputs (prompt only)
                            # TaskCollator_gemma already put these in the batch
                            # --------------------------
                            eval_input_ids = inputs["input_ids_eval"].to(self.set_model)
                            eval_attention_mask = inputs["attention_mask_eval"].to(self.set_model)
                            eval_labels = inputs["labels_eval"].detach().cpu().numpy()

                            # Replace ignore_index (-100) with pad token so decoding works
                            eval_labels = np.where(eval_labels < 0, self.tokenizer.pad_token_id, eval_labels)

                            # --------------------------
                            # Task name normalization
                            # --------------------------
                            task_raw = inputs.get("task", "dart")
                            task_name = self._normalize_task_name(task_raw) or "dart"

                            # --------------------------
                            # Generation kwargs (same logic as prediction_step)
                            # --------------------------
                            max_new_tokens = self._get_gen_max_new_tokens(task_raw)
                            gen_kwargs = {
                                "max_new_tokens": max_new_tokens,
                                "num_beams": getattr(self.args, "generation_num_beams", 1),
                            }

                            if "task" in inputs:
                                gen_kwargs["task"] = inputs["task"]

                            if self.model.config.train_adapters and isinstance(self.adapter_config, MetaAdapterConfig):
                                gen_kwargs["task_embedding"] = model.task_embedding_controller(inputs["task"])
                            else:
                                gen_kwargs["task_embedding"] = None

                            # --------------------------
                            # Generate predictions (prompt-only)
                            # --------------------------
                            with torch.no_grad():
                                generated_tokens = model.generate(
                                    eval_input_ids,
                                    attention_mask=eval_attention_mask,
                                    **gen_kwargs,
                                )

                            preds = generated_tokens.detach().cpu().numpy()

                            # --------------------------
                            # Log metrics + debug dump
                            # --------------------------
                            self.run_and_save_metrics(
                                step=(step+1)/4,
                                preds=preds,
                                labels=eval_labels,
                                tokenizer=self.tokenizer,
                                context_input_ids = eval_input_ids,
                                task_name=inputs["task"],
                                output_path=f"metrics_step_{inputs["task"]}_{(int(step/100))%10}.json"
                                
                            )

                            # --- CLEANUP ---
                            del generated_tokens
                            del preds
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"[DEBUG ERROR OUT] Could not log predictions: {e}")
                            
                    self.control.should_evaluate = False
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
            # print("I am out of loop")
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
        # print("I am fully out")
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
                self.model = self.model.to(self.set_model)
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
        device = next(model.parameters()).device
        # print(f"======================================================Model Inside Compute Loss: {device}========================================")
        # if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
        #     print(f"✅ Input IDs device is now: {inputs['input_ids'].device}")
    
        # if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor):
            # print(f"✅ Labels device is now: {inputs['labels'].device}")
        
        model = model.to (self.set_model)
        # Generation kwargs
        task_for_len = inputs.get("task", None)
        max_new_tokens = self._get_gen_max_new_tokens(task_for_len)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": getattr(self.args, "generation_num_beams", 1),
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
            # predictions are token IDs from generate()
            predictions = generated_tokens
        else:
            # predictions are logits (for non-generate eval)
            predictions = outputs.logits

        # ----------------- FIXED LABEL PADDING -----------------
        # Only pad labels to match the *prediction length* when we're
        # returning token IDs (predict_with_generate = True).
        if self.args.predict_with_generate and labels is not None and predictions is not None:
            if predictions.dim() != 2:
                raise ValueError(
                    f"Expected generated token IDs with shape [batch, seq], "
                    f"got shape {tuple(predictions.shape)}"
                )
            pred_len = predictions.shape[-1]
            if labels.shape[-1] < pred_len:
                labels = self._pad_tensors_to_max_len(labels, pred_len)
        # -------------------------------------------------------

        
        # ==================== NEW CLEANUP SECTION ====================
        # 1. Move return values to CPU
        # This ensures the Trainer doesn't accumulate GPU tensors, which causes OOM.
        loss = loss.cpu()
        if predictions is not None:
            predictions = predictions.detach().cpu()
        if labels is not None:
            labels = labels.detach().cpu()
    
        # 2. Explicitly delete heavy GPU tensors
        # 'outputs' contains the full logits (vocab size), which is HUGE (~1GB).
        # If you used generate(), you don't need these logits anymore.
        if 'outputs' in locals():
            del outputs 
        
        # Delete inputs to free the input_ids and attention_mask
        del inputs
        
        # Optional: Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # =============================================================
    
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
    
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation dataloader but using evaluation tensors created
        by the TaskCollator_gemma:
            - input_ids_eval
            - attention_mask_eval
            - labels_eval

        NOTE:
        - We still call the original data_collator to build the batch.
        - We wrap the collator to override the training fields.
        """

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # ---------- WRAP ORIGINAL COLLATOR ----------
        def eval_collate(batch):
            # Run original TaskCollator_gemma => produces:
            #  input_ids, attention_mask, labels, task,
            #  input_ids_eval, attention_mask_eval, labels_eval
            batch_out = self.data_collator(batch)

            # Replace training tensors with evaluation ones
            batch_out["input_ids"] = batch_out["input_ids_eval"]
            batch_out["attention_mask"] = batch_out["attention_mask_eval"]
            batch_out["labels"] = batch_out["labels_eval"]
            # print("heloooooooooooooooooooooo")
            # print(batch_out["labels"].shape)
            # print(batch_out["input_ids"].shape)
            # task stays the same
            return batch_out
        
        # ---------- RETURN DATA LOADER ----------
        return DataLoader(
            eval_dataset,
            batch_size=8, ## Hard Code
            shuffle=False,
            collate_fn=eval_collate
        )


    def _pad_tensors_to_max_len(self, tensor, max_length):
        # (Copied from T5Trainer, but uses self.pad_token_id)
        if self.pad_token_id is None:
            raise ValueError("pad_token_id must be set in config to pad tensors.")

        padded_tensor = self.pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor