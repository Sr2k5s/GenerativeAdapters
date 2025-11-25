"""
Implements a Gemma trainer class for Knowledge Distillation (KD)
while using Generative Adapters (Hyperformer) on decoder-only Gemma 3.

This trainer:
1. Subclasses `Trainer`.
2. Adds KD logic (teacher model, alpha_kd, temp) to __init__.
3. Overrides `compute_loss` for combined CE + KLDiv loss.
4. Includes adapter-specific methods from the T5Trainer:
   - `evaluate`: To handle multi-task eval with `use_task_specific_params`.
   - `prediction_step`: To inject `task` and `task_embedding` into generate.
   - `_get_train_sampler`/`get_train_dataloader`: For `MultiTaskBatchSampler`.
   - `train`: To correctly load the best adapter-based model at the end.

COMBINED CHANGES / FIXES:
- FIX 1: compute_loss now strips eval-only keys before student/teacher forward.
- FIX 2: teacher forward does NOT receive labels or eval-only tensors.
- FIX 3: run_and_save_metrics trims generated tokens using REAL prompt length per sample
         (attention_mask sum), NOT padded prompt length.
- FIX 4: debug generation + prediction_step pass eos_token_id/pad_token_id to generate.
- FIX 5: debug generation runs model.generate in eval() mode and restores train().
- FIX 6: removed accidental use of teacher_inputs before it was defined.
- ROBUSTNESS: if teacher is None or KL fails, fall back to pure CE automatically.
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
from transformers import PreTrainedModel, logging, Trainer, EvalPrediction
from typing import Any, Dict, Optional, Tuple, Union, List
from torch.utils.data.dataset import Dataset
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import TrainOutput, set_seed
from transformers.integrations import hp_params
from torch.optim import AdamW

# --- Dependencies from your T5Trainer ---
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

# nltk punkt for some metrics (safe to keep)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

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

if False:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)
from metrics.metrics import build_compute_metrics_fn  # your file


class GemmaAdapterKDTrainer(Trainer):
    """Trainer for KD with Gemma 3 + Hyperformer adapters."""

    def __init__(
        self,
        # KD params
        teacher_model: Optional[PreTrainedModel],
        alpha_kd: float = 0.5,
        temperature: float = 2.0,
        # adapter/multitask params
        config=None,
        data_args=None,
        dataset_sizes=None,
        adapter_config=None,
        multi_task_compute_metrics=None,
        tokenizer=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # T5Trainer init logic
        if config is None:
            assert isinstance(self.model, PreTrainedModel), (
                "If no `config` is passed, model must be a PreTrainedModel, "
                f"but is {self.model.__class__}"
            )
            self.config = self._actual_model(self.model).config
        else:
            self.config = config

        self.adapter_config = adapter_config
        self.multi_task_compute_metrics = multi_task_compute_metrics
        self.dataset_sizes = dataset_sizes
        self.data_args = data_args
        self.tokenizer = tokenizer

        # KD init logic
        self.teacher_model = teacher_model
        self.alpha_kd = alpha_kd
        self.temperature = temperature

        self.set_model = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.set_model)
            self.teacher_model.eval()

        # pad / eos
        self.pad_token_id = self.model.config.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.model.config.eos_token_id
            logger.warning(f"pad_token_id not set, using eos_token_id: {self.pad_token_id}")

        self.loss_fn_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    # ==========================
    # Task name normalization
    # ==========================
    def _normalize_task_name(self, task: Any) -> Optional[str]:
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
        t = self._normalize_task_name(task)
        if t == "dart":
            return 64
        if t == "e2e_nlg":
            return 64
        if t == "wiki_lingua":
            return 128

        gen_len = getattr(self.args, "generation_max_length", None)
        if gen_len is not None and gen_len > 0:
            return int(gen_len)

        if getattr(self, "data_args", None) is not None:
            eval_max = getattr(self.data_args, "eval_max_target_length", None)
            if eval_max is not None and eval_max > 0:
                return int(eval_max)

        return 64

    # ======================================
    # Debug + metrics helper
    # ======================================
    def run_and_save_metrics(
        self,
        step: int,
        preds: np.ndarray,
        labels: np.ndarray,
        tokenizer,
        context_input_ids,
        context_attention_mask=None,
        task_name: str = "dart",
        output_path: str = "metrics.json",
    ):
        """
        preds   : np array (batch, generated_seq_len)
        labels  : np array (batch, target_seq_len or full masked)
        context_input_ids : prompt-only ids (padded)
        context_attention_mask : prompt-only attention mask (padded)
        """
        try:
            pad_id = tokenizer.pad_token_id

            # sanitize negatives
            preds = np.where(preds < 0, pad_id, preds)
            labels = np.where(labels < 0, pad_id, labels)

            # Trim predictions using REAL prompt length per sample
            trimmed_preds: List[np.ndarray] = []
            for i in range(preds.shape[0]):
                if context_attention_mask is not None:
                    ctx_len = int(context_attention_mask[i].sum())
                else:
                    ctx_len = int((context_input_ids[i] != pad_id).sum())

                trimmed = preds[i, ctx_len:]
                trimmed_preds.append(trimmed)

                if step < 5:
                    print(
                        f"[DEBUG TRIM] i={i} ctx_len(real)={ctx_len} "
                        f"pred_len={preds.shape[1]} trimmed_len={len(trimmed)}"
                    )

            # pad trimmed preds for EvalPrediction
            max_trim_len = max((len(t) for t in trimmed_preds), default=0)
            padded_trimmed_preds = np.full((preds.shape[0], max_trim_len), pad_id, dtype=preds.dtype)
            for i, t in enumerate(trimmed_preds):
                padded_trimmed_preds[i, : len(t)] = t

            # Compute metrics on trimmed preds
            eval_pred = EvalPrediction(predictions=padded_trimmed_preds, label_ids=labels)
            metrics_fns = build_compute_metrics_fn([task_name], tokenizer)
            compute_metrics = metrics_fns[task_name]
            results = compute_metrics(eval_pred)

            print(f"[DEBUG METRICS] results -----> {results}")

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            # Decode trimmed preds
            pred_str = [tokenizer.decode(t, skip_special_tokens=True) for t in trimmed_preds]
            label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

            if pred_str:
                print(f"[DEBUG DECODE] sample pred_str[0]: {pred_str[0][:200]}")
            if label_str:
                print(f"[DEBUG DECODE] sample label_str[0]: {label_str[0][:200]}")

            debug_dump = {
                "global_step": int(self.state.global_step),
                "batch_step": int(step),
                "task": task_name,
                "predictions": pred_str,
                "labels": label_str,
            }
            temp = f"train_debug_{step // 10}.json"
            with open(temp, "w") as f:
                json.dump(debug_dump, f, indent=2)

        except Exception as e:
            print(f"[DEBUG ERROR IN] Could not log predictions: {e}")

    # ======================================
    # Prepare inputs
    # ======================================
    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.set_model)
        return inputs

    # ======================================
    # KD-aware compute_loss
    # ======================================
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # FIX 1: strip eval-only keys before student forward
        student_inputs = inputs.copy()
        student_inputs.pop("input_ids_eval", None)
        student_inputs.pop("attention_mask_eval", None)
        student_inputs.pop("labels_eval", None)

        labels = student_inputs.get("labels", None)

        student_outputs = model(**student_inputs)
       
        student_logits = student_outputs.logits
        
        hard_loss = student_outputs.loss

        distill_loss = None
        alpha_used = 1.0  # default pure CE

        if self.teacher_model is not None:
            alpha_used = self.alpha_kd
            try:
                with torch.no_grad():
                    # teacher sees same seq as student, but no adapter-only keys, no labels
                    teacher_inputs = student_inputs.copy()
                    teacher_inputs.pop("task", None)
                    teacher_inputs.pop("task_embedding", None)
                    teacher_inputs.pop("gemma_adapters", None)
                    teacher_inputs.pop("labels", None)  # FIX 2

                    if self.state.global_step == 0 and self.is_world_process_zero():
                        print("[DEBUG TEACHER_INPUTS] keys:", teacher_inputs.keys())

                    teacher_outputs = self.teacher_model(**teacher_inputs)
                    teacher_logits = teacher_outputs.logits

                # vocab align
                student_vocab_size = student_logits.shape[-1]
                if teacher_logits.shape[-1] > student_vocab_size:
                    teacher_logits = teacher_logits[:, :, :student_vocab_size]

                soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
                soft_teacher = F.log_softmax(teacher_logits / self.temperature, dim=-1)

                if labels is not None:
                    mask = (labels != -100).view(-1)
                    flat_student = soft_student.view(-1, soft_student.size(-1))
                    flat_teacher = soft_teacher.view(-1, soft_teacher.size(-1))
                    masked_student = flat_student[mask]
                    masked_teacher = flat_teacher[mask]
                    if masked_student.numel() > 0:
                        distill_loss = (self.temperature ** 2) * self.loss_fn_kl(
                            masked_student, masked_teacher
                        )
                    else:
                        distill_loss = torch.tensor(0.0, device=student_logits.device)
                else:
                    distill_loss = (self.temperature ** 2) * self.loss_fn_kl(
                        soft_student, soft_teacher
                    )

            except Exception as e:
                # robustness fallback
                alpha_used = 1.0
                distill_loss = torch.tensor(0.0, device=student_logits.device)
                if self.is_world_process_zero():
                    print(f"[DEBUG KD FALLBACK] distill failed, using pure CE. Reason: {e}")

        # final loss
        if distill_loss is None:
            loss = hard_loss
        else:
            loss = alpha_used * hard_loss + (1.0 - alpha_used) * distill_loss

        # DEBUG balance
        if (self.state.global_step % 200 == 0) and self.is_world_process_zero():
            with torch.no_grad():
                print("\n================= DEBUG LOSS =================")
                print("global_step:", self.state.global_step)
                print("hard_loss (CE):", float(hard_loss))
                print("distill_loss (KL):", float(distill_loss) if distill_loss is not None else "None")
                print("alpha_kd(target):", self.alpha_kd, "alpha_used:", alpha_used, "temperature:", self.temperature)

                if labels is not None:
                    frac_supervised = (labels != -100).float().mean().item()
                    print("fraction supervised tokens:", frac_supervised)

                print(
                    "student logits mean/std:",
                    student_logits.mean().item(),
                    student_logits.std().item(),
                )
                if self.teacher_model is not None:
                    print(
                        "teacher logits mean/std:",
                        teacher_logits.mean().item(),
                        teacher_logits.std().item(),
                    )

                if labels is not None:
                    mask_dbg = (labels != -100).view(-1)
                    print(
                        "masked token count:",
                        int(mask_dbg.sum().item()),
                        "total tokens:",
                        int(mask_dbg.numel()),
                    )
                print("================================================\n")

        return (loss, student_outputs) if return_outputs else loss

    # ======================================
    # Multi-task evaluate
    # ======================================
    def evaluate(
        self,
        eval_datasets: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        results = {}
        if eval_datasets is None:
            eval_datasets = self.eval_dataset
        model_config = self.model.config

        task_iterator = tqdm(
            eval_datasets.items(),
            desc="Evaluating tasks",
            disable=not self.is_world_process_zero(),
        )

        for eval_task, eval_dataset in task_iterator:
            dataset_size = len(eval_dataset)
            print(f"\n🔍 Evaluating task: {eval_task} | size = {dataset_size} examples\n")
            task_iterator.set_postfix({"task": eval_task, "size": dataset_size})

            self.compute_metrics = self.multi_task_compute_metrics[eval_task]
            use_task_specific_params(self.model, eval_task)

            if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
                raise ValueError("eval_dataset must implement __len__")

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
            )

            tasks_metric = {eval_task + "_" + k: v for k, v in output.metrics.items()}
            for key in sorted(tasks_metric.keys()):
                logger.info(f"  {key} = {tasks_metric[key]}")
            results.update(tasks_metric)

            reset_config(self.model, model_config)
            break  # keep your existing one-task behavior

        metrics = [results[key] for key in results.keys() if "loss" not in key]
        if len(metrics) > 0:
            results["eval_average_metrics"] = float(np.mean(metrics))

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, results
        )
        return results

    # ======================================
    # Custom train loop (kept same, with fixed debug gen)
    # ======================================
    def train(
        self,
        model_path: Optional[str] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
    ):
        self._hp_search_setup(trial)
        self.current_gradient_accumulation_steps = self.args.gradient_accumulation_steps

        if self.model_init is not None:
            set_seed(self.args.seed)
            model = self.call_model_init(trial)
            self.model = model.to(self.set_model)
            self.optimizer, self.lr_scheduler = None, None

        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)
        train_dataloader = self.get_train_dataloader()

        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = (
                    self.args.max_steps // num_update_steps_per_epoch
                    + int(self.args.max_steps % num_update_steps_per_epoch > 0)
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state.is_hyper_param_search = trial is not None
        self._load_optimizer_and_scheduler(model_path)

        model = self.model.to(self.set_model)
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex to use fp16 training.")
            model, self.optimizer = amp.initialize(
                model, self.optimizer, opt_level=self.args.fp16_opt_level
            )

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

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

        if False:
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (
                    torch.distributed.get_world_size()
                    if torch.distributed.is_initialized()
                    else 1
                )
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.args.per_device_train_batch_size,
        )
        logger.info("  Total train batch size = %d", total_train_batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", max_steps)

        import time
        start_time = time.time()

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(
                os.path.join(model_path, "trainer_state.json")
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            steps_trained_in_current_epoch = (
                self.state.global_step % num_update_steps_per_epoch
            )
            logger.info("Continuing training from checkpoint")

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = (
            self.hp_name(trial) if self.hp_name is not None else None
        )
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

        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and (
                isinstance(train_dataloader.sampler, DistributedSampler)
                or isinstance(train_dataloader.batch_sampler, MultiTaskBatchSampler)
            ):
                if isinstance(train_dataloader.sampler, DistributedSampler):
                    train_dataloader.sampler.set_epoch(epoch)
                else:
                    train_dataloader.batch_sampler.set_epoch(epoch)

            if False:
                parallel_loader = pl.ParallelLoader(
                    train_dataloader, [self.set_model]
                ).per_device_loader(self.set_model)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(
                self.args, self.state, self.control
            )

            for step, inputs in enumerate(epoch_iterator):
                grad_norm_for_log = None

                # DEBUG batch inspection
                if step % 200 == 0 and self.is_world_process_zero():
                    print("\n================= DEBUG BATCH =================")
                    print("global_step:", self.state.global_step, "local_step:", step)
                    print("task raw:", inputs.get("task"))

                    keys_to_check = [
                        "input_ids",
                        "attention_mask",
                        "labels",
                        "input_ids_eval",
                        "attention_mask_eval",
                        "labels_eval",
                    ]
                    for k in keys_to_check:
                        v = inputs.get(k)
                        if isinstance(v, torch.Tensor):
                            print(
                                f"{k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device} "
                                f"min={v.min().item()} max={v.max().item()}"
                            )
                        else:
                            print(f"{k}: type={type(v)}")

                    if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor):
                        lab0 = inputs["labels"][0]
                        mask0 = lab0 != -100
                        if mask0.any():
                            first_lab_pos = mask0.nonzero(as_tuple=True)[0][0].item()
                            prompt_len_train = int(inputs["attention_mask"][0].sum().item())
                            print(
                                "first label pos:",
                                first_lab_pos,
                                "| prompt_len_train:",
                                prompt_len_train,
                            )

                    tok = self.tokenizer
                    if tok is not None and "input_ids_eval" in inputs:
                        am0 = inputs["attention_mask_eval"][0]
                        pl0 = int(am0.sum().item())
                        prompt0 = tok.decode(
                            inputs["input_ids_eval"][0][:pl0], skip_special_tokens=False
                        )
                        print("\nprompt[0] (eval prompt only):\n", prompt0[:500])

                    if tok is not None and "labels" in inputs:
                        lab_tokens0 = inputs["labels"][0][inputs["labels"][0] != -100][:200]
                        label0 = tok.decode(lab_tokens0, skip_special_tokens=False)
                        print("\nlabel[0] (first ~200 label toks):\n", label0[:500])
                    print("================================================\n")

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        self.args, self.state, self.control
                    )

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and _use_ddp_no_sync
                ):
                    with model.no_sync():
                        step_loss = self.training_step(model, inputs)
                        tr_loss += step_loss.detach()
                else:
                    step_loss = self.training_step(model, inputs)
                    tr_loss += step_loss.detach()

                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm_for_log = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.args.max_grad_norm
                        )
                    elif self.args.fp16 and _use_apex:
                        grad_norm_for_log = torch.nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer),
                            self.args.max_grad_norm,
                        )
                    else:
                        grad_norm_for_log = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.args.max_grad_norm
                        )

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
                    self.control = self.callback_handler.on_step_end(
                        self.args, self.state, self.control
                    )

                    # DEBUG generation block every 100 steps
                    if step % 100 == 0:
                        try:
                            tokenizer = self.tokenizer
                            eval_input_ids = inputs["input_ids_eval"].to(self.set_model)
                            eval_attention_mask = inputs["attention_mask_eval"].to(self.set_model)
                            eval_labels = inputs["labels_eval"].detach().cpu().numpy()

                            # sanitize labels for decode/metrics
                            eval_labels = np.where(
                                eval_labels < 0, tokenizer.pad_token_id, eval_labels
                            )

                            task_raw = inputs.get("task", "dart")
                            task_name = self._normalize_task_name(task_raw) or "dart"

                            max_new_tokens = self._get_gen_max_new_tokens(task_raw)
                            gen_kwargs = {
                                "max_new_tokens": max_new_tokens,
                                "num_beams": getattr(self.args, "generation_num_beams", 1),
                                "min_new_tokens": 5,
                                "eos_token_id": self.model.config.eos_token_id,
                                "pad_token_id": tokenizer.pad_token_id,
                            }

                            if "task" in inputs:
                                gen_kwargs["task"] = inputs["task"]

                            if (
                                self.model.config.train_adapters
                                and isinstance(self.adapter_config, MetaAdapterConfig)
                            ):
                                gen_kwargs["task_embedding"] = model.task_embedding_controller(
                                    inputs["task"]
                                )
                            else:
                                gen_kwargs["task_embedding"] = None

                            # FIX 5: temporarily eval mode for generation
                            was_training = model.training
                            model.eval()
                            with torch.no_grad():
                                generated_tokens = model.generate(
                                    eval_input_ids,
                                    attention_mask=eval_attention_mask,
                                    **gen_kwargs,
                                )
                            if was_training:
                                model.train()

                            preds = generated_tokens.detach().cpu().numpy()

                            self.run_and_save_metrics(
                                step=step,
                                preds=preds,
                                labels=eval_labels,
                                tokenizer=tokenizer,
                                context_input_ids=eval_input_ids.detach().cpu().numpy(),
                                context_attention_mask=eval_attention_mask.detach().cpu().numpy(),
                                task_name=task_name,
                                output_path=f"metrics_step_{self.state.global_step}.json",
                            )

                            del generated_tokens, preds
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
                        self.args.ignore_keys_for_eval
                        if hasattr(self.args, "ignore_keys_for_eval")
                        else None,
                        start_time,
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(
                self.args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss,
                None,
                model,
                trial,
                epoch,
                self.args.ignore_keys_for_eval
                if hasattr(self.args, "ignore_keys_for_eval")
                else None,
                start_time,
            )

            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("Training completed.")

        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} "
                f"(score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.__class__.from_pretrained(
                    self.state.best_model_checkpoint,
                    adapter_config=self.adapter_config,
                )
                self.model = self.model.to(self.set_model)
            else:
                state_dict = torch.load(
                    os.path.join(self.state.best_model_checkpoint, "pytorch_model.bin")
                )
                self.model.load_state_dict(state_dict)

        if self._total_flos is not None:
            self.store_flos()
            self.log({"total_flos": self.state.total_flos})

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

        return TrainOutput(
            global_step=self.state.global_step,
            training_loss=tr_loss.item() / max(self.state.global_step, 1),
            metrics={},
        )

    # ======================================
    # prediction_step (task-specific max_new_tokens)
    # ======================================
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        model = model.to(self.set_model)

        task_for_len = inputs.get("task", None)
        max_new_tokens = self._get_gen_max_new_tokens(task_for_len)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": getattr(self.args, "generation_num_beams", 1),
            "eos_token_id": self.model.config.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer is not None
            else self.model.config.eos_token_id,
        }

        if "task" in inputs:
            gen_kwargs["task"] = inputs["task"]

        if self.model.config.train_adapters and isinstance(
            self.adapter_config, MetaAdapterConfig
        ):
            gen_kwargs["task_embedding"] = model.task_embedding_controller(inputs["task"])
        else:
            gen_kwargs["task_embedding"] = None

        generated_tokens = None
        if self.args.predict_with_generate and not prediction_loss_only:
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        labels = inputs.pop("labels", None)
        if labels is not None:
            inputs["labels"] = labels

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        if self.args.predict_with_generate:
            predictions = generated_tokens
        else:
            predictions = outputs.logits

        if (
            self.args.predict_with_generate
            and labels is not None
            and predictions is not None
        ):
            if predictions.dim() != 2:
                raise ValueError(
                    f"Expected generated token IDs with shape [batch, seq], "
                    f"got shape {tuple(predictions.shape)}"
                )
            pred_len = predictions.shape[-1]
            if labels.shape[-1] < pred_len:
                labels = self._pad_tensors_to_max_len(labels, pred_len)

        loss = loss.cpu()
        if predictions is not None:
            predictions = predictions.detach().cpu()
        if labels is not None:
            labels = labels.detach().cpu()

        if "outputs" in locals():
            del outputs
        del inputs

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return (loss, predictions, labels)

    # ======================================
    # Optimizer / scheduler
    # ======================================
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
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
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    eps=self.args.adam_epsilon,
                )

        if self.lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler(num_training_steps)

    def _get_lr_scheduler(self, num_training_steps):
        schedule_func = arg_to_scheduler[self.args.lr_scheduler]
        if self.args.lr_scheduler == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.args.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.args.warmup_steps
            )
        else:
            scheduler = schedule_func(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        return scheduler

    # ======================================
    # Multi-task samplers/dataloaders
    # ======================================
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if self.dataset_sizes is None:
            return super()._get_train_sampler()

        if False and xm.xrt_world_size() > 1:
            num_replicas = xm.xrt_world_size()
            rank = xm.get_ordinal()
        elif self.args.local_rank != -1:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_initialized()
                else 1
            )
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
        else:
            num_replicas = 1
            rank = 0

        return MultiTaskBatchSampler(
            self.dataset_sizes,
            self.args.train_batch_size,
            self.args.temperature,
            rank=rank,
            num_replicas=num_replicas,
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.dataset_sizes is None:
            return super().get_train_dataloader()

        multitask_sampler = self._get_train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_sampler=multitask_sampler,
            collate_fn=self.data_collator,
        )

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        def eval_collate(batch):
            batch_out = self.data_collator(batch)
            # Use prompt-only inputs for model; labels are whatever collator returns for eval
            batch_out["input_ids"] = batch_out["input_ids_eval"]
            batch_out["attention_mask"] = batch_out["attention_mask_eval"]
            batch_out["labels"] = batch_out["labels_eval"]
            return batch_out

        return DataLoader(
            eval_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=eval_collate,
        )

    # ======================================
    # padding helper
    # ======================================
    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.pad_token_id is None:
            raise ValueError("pad_token_id must be set in config to pad tensors.")
        padded_tensor = self.pad_token_id * torch.ones(
            (tensor.shape[0], max_length),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
