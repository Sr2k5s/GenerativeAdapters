import json
from pathlib import Path

NEW_PARAMS = {
  "model_name_or_path": "google/gemma-3-1b-it",
  "tokenizer_name": "google/gemma-3-1b-it",
  "teacher_model_name_or_path": "google/gemma-3-12b-it",
  "alpha_kd": 0.47,
  "temperature_kd": 2.0,
  "learning_rate": 3e-4,
  "lr_scheduler": "linear",
  "output_dir": "outputs/gemma_kd_hyperformer_gem/",
  "max_source_length": 512,
  "max_target_length": 128,
  "val_max_target_length": 128,
  "test_max_target_length": 128,
  "num_train_epochs": 10,
  "warmup_steps": 100,
  "eval_steps": 2000,
  "overwrite_output_dir": True,
  "per_device_train_batch_size": 8,
  "per_device_eval_batch_size": 8,
  "save_steps": 2000,
  "logging_first_step": True,
  "logging_steps": 200,
  "save_total_limit": 1,
  "temperature": 3.33,
  "do_train": True,
  "do_test": True,
  "do_eval": True,
  "predict_with_generate": True,
  "split_validation_test": True,
  "load_best_model_at_end": True,
  "eval_strategy": "steps",
  "save_strategy": "steps",
  "evaluation_strategy": "steps", 
  "metric_for_best_model": "average_metrics",
  "greater_is_better": True,
  "max_steps": -1,
  "tasks": [
    "dart",
    "e2e_nlg",
    "squad_v2"
  ],
  "eval_tasks": [
    "dart",
    "e2e_nlg",
    "squad_v2"
  ],
  "train_adapters": True,
  "adapter_config_name": "meta-adapter",
  "reduction_factor": 32,
  "projected_task_embedding_dim": 64,
  "task_embedding_dim": 64,
  "conditional_layer_norm": True,
  "efficient_unique_hyper_net": True,
  "non_linearity": "gelu_new",
  "freeze_model": False,
  "unfreeze_layer_norms": True,
  "unfreeze_lm_head": False,
  "use_cpu":False,
  "gradient_accumulation_steps" :4
}

def load_and_set_params_json(file_path: str):
    file_path = Path(file_path)

    # Load existing JSON as dict
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Update / overwrite keys
    data.update(NEW_PARAMS)

    # Save back to same file
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data


# Example usage
if __name__ == "__main__":
    path = "gemma3_kd.json"   # <-- change this to your file
    config = load_and_set_params_json(path)
    print(f"✅ JSON updated and saved: {path}")