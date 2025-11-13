
#["dart", "web_nlg", "e2e_nlg", "totto", "cs_restaurants", "turku_hockey_data2text", "mlb_data_to_text", "xsum", "xlsum", "mlsum", "xwikis", "wiki_lingua", "wiki_cat_sum", "schema_guided_dialog", "conversational_weather", "opusparcus", "squad_v2", "wiki_auto_asset_turk"]


# hyperformer/data/tasks_Gem.py
# -*- coding: utf-8 -*-
"""
GEM benchmark tasks -> seq2seq format for Hyperformer-style pipelines.

- Uses Hugging Face datasets (script-backed) with datasets<4.0.0
- Converts each example to:
    {
        "src_texts": "<optional-prefix> <source>",
        "tgt_texts": "<target>",
        "task": "<task_name>"
    }
- Includes a verbose smoke test CLI.

Author: updated for GEM by ChatGPT
"""
from __future__ import annotations

import abc
import argparse
import functools
import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Mapping, Optional

import datasets
import numpy as np
import torch

from metrics import metrics  # your existing metric wrappers
from .utils import compute_task_max_decoding_length  # your existing helper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# -----------------------
# Abstract Base
# -----------------------
class AbstractTaskDataset(abc.ABC):
    """
    Base interface to expose:
      - name
      - task_specific_config (decoding knobs)
      - preprocessor(example, add_prefix=True) -> seq2seq fields
      - metrics (callables)
      - split_to_data_split (logical -> actual split names mapping)
    """
    name: str = NotImplemented
    task_specific_config: Dict = NotImplemented
    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    def __init__(self, seed: int = 42):
        self.seed = seed

    # ---- sampling helpers (kept compatible with your previous code style) ----
    def _check_n_obs(self, n_obs: Optional[int], total_size: int) -> Optional[int]:
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs clamped to dataset size: %s", n_obs)
        return n_obs

    def _select_dataset_samples(self, indices: List[int], dataset: datasets.Dataset, n_obs: Optional[int] = None):
        n_obs = self._check_n_obs(n_obs, len(indices))
        indices = indices[:n_obs] if n_obs is not None else indices
        return dataset.select(indices)

    def _shuffle_and_sample(self, split: str, n_obs: Optional[int] = None) -> datasets.Dataset:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(mapped_split)
        total = len(dataset)
        indices = torch.randperm(total, generator=generator).tolist()
        return self._select_dataset_samples(indices, dataset, n_obs=n_obs)

    # ---- main public entry point used by trainers ----
    def get_dataset(
        self,
        split: str,
        n_obs: Optional[int] = None,
        add_prefix: bool = True,
        split_validation_test: bool = False,  # retained for compatibility; not used for GEM
    ) -> datasets.Dataset:
        # GEM has proper splits; we simply shuffle + sample like before.
        dataset = self._shuffle_and_sample(split, n_obs)
        return dataset.map(
            functools.partial(self.preprocessor, add_prefix=add_prefix),
            remove_columns=dataset.column_names,
        )

    # ---- abstract ----
    @abc.abstractmethod
    def load_dataset(self, split: str) -> datasets.Dataset:
        ...


# -----------------------
# GEM Generic Task Wrapper
# -----------------------
class GEMGenericTask(AbstractTaskDataset):
    """
    A generic wrapper for most GEM tasks, which expose 'source' and 'target' (and often 'references').

    Args:
        hf_id: "GEM/<dataset_name>" on the Hub
        task_name: short name (used as prefix & 'task' field)
        task_type: string tag (e.g., "summarization", "data2text", ...)
        decoding: dict with decoding knobs (e.g., {'max_length': 128, 'num_beams': 4})
        metric_names: ['rouge', 'bleu', ...] -> resolved to your metrics.* callables
        split_overrides: optional mapping from logical split to actual HF split string
        prefix: optional natural-language or task-name prefix
        hf_config: optional config/subset name (e.g., 'en' for web_nlg)
    """
    def __init__(
        self,
        hf_id: str,
        task_name: str,
        task_type: str,
        decoding: Dict,
        metric_names: List[str],
        split_overrides: Optional[Mapping[str, str]] = None,
        prefix: Optional[str] = None,
        hf_config: Optional[str] = None,  # <-- MODIFIED: Added hf_config
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.hf_id = hf_id
        self.hf_config = hf_config  # <-- MODIFIED: Store hf_config
        self.name = task_name
        self.task_type = task_type
        self.task_specific_config = dict(decoding or {})
        self.prefix = prefix or task_name

        # resolve metrics by name using your existing metrics module
        metric_map = {
            "rouge": metrics.rouge,
            "bleu": metrics.bleu,
            "accuracy": metrics.accuracy,
            "f1": getattr(metrics, "f1_score_with_invalid", metrics.accuracy),  # fallback
            "pearson": getattr(metrics, "pearson_corrcoef", None),
            "spearman": getattr(metrics, "spearman_corrcoef", None),
        }
        resolved = []
        for m in metric_names:
            fn = metric_map.get(m)
            if fn is not None:
                resolved.append(fn)
            else:
                logger.warning("Unknown metric '%s' for task '%s' — skipping.", m, task_name)
        self.metrics = resolved or [metrics.accuracy]

        # optional per-dataset split mapping
        if split_overrides:
            self.split_to_data_split = dict(split_overrides)

    # robust loader that handles the “no plain test” case by falling back
    def load_dataset(self, split: str) -> datasets.Dataset:
        try:
            # <-- MODIFIED: Pass 'name=self.hf_config' to load_dataset
            return datasets.load_dataset(
                self.hf_id, 
                name=self.hf_config, 
                split=split, 
                trust_remote_code=True, 
                revision="main"
            )
        except Exception as e:
            # common GEM quirk: some tasks don't have 'test' → try an alternative
            if split == "test":
                for alt in ("validation", "test_wiki", "test_asset", "test_turk", "test_contract"):
                    try:
                        # <-- MODIFIED: Pass 'name=self.hf_config'
                        ds = datasets.load_dataset(
                            self.hf_id, name=self.hf_config, split=alt, trust_remote_code=True, revision="main"
                        )
                        logger.warning("[%s] falling back test→%s", self.name, alt)
                        return ds
                    except Exception:
                        continue
            if split == "validation":
                for alt in ("dev", "test"):
                    try:
                        # <-- MODIFIED: Pass 'name=self.hf_config'
                        ds = datasets.load_dataset(
                            self.hf_id, name=self.hf_config, split=alt, trust_remote_code=True, revision="main"
                        )
                        logger.warning("[%s] falling back validation→%s", self.name, alt)
                        return ds
                    except Exception:
                        continue
            raise e

    # universal GEM preprocessor: source → target
    def preprocessor(self, example, add_prefix: bool = True):
        # Most GEM tasks expose 'source' and 'target'
        src = example.get("source", None)
        tgt = example.get("target", None)

        # conservative fallbacks if a GEM variant deviates
        if src is None:
            # <-- MODIFIED: Added "document" fallback for canonical XSum
            src = example.get("input", example.get("inputs", example.get("document", "")))
        if tgt is None:
            # <-- MODIFIED: Added "summary" fallback for canonical XSum
            tgt = example.get("reference", example.get("references", example.get("summary", "")))
            if isinstance(tgt, list):
                # pick the first reference for training (others can be used for eval)
                tgt = tgt[0] if tgt else ""

        src_pieces = [self.prefix + ":", str(src)] if add_prefix else [str(src)]
        tgt_pieces = [str(tgt)]
        return {
            "src_texts": " ".join(src_pieces).strip(),
            "tgt_texts": " ".join(tgt_pieces).strip(),
            "task": self.name,
        }


# -----------------------
# Concrete Task Registry
# -----------------------
# Decoding defaults by type
DECODING_SMALL = {"max_length": 64, "num_beams": 4}
DECODING_MED = {"max_length": 128, "num_beams": 4}
DECODING_LONG = {"max_length": 256, "num_beams": 4}

# Build a broad, practical subset of GEM tasks.
# You can freely add more items here — they will auto-register.
_GEM_TASK_SPECS = [
    # ---------- Data-to-Text ----------
    dict(hf_id="GEM/dart",                    task_name="dart",         task_type="data2text",      decoding=DECODING_MED,  metrics=["bleu"]),
    # <-- MODIFIED: Added hf_config="en"
    dict(hf_id="GEM/web_nlg",                 task_name="web_nlg",      task_type="data2text",      decoding=DECODING_MED,  metrics=["bleu"], hf_config="en"),
    dict(hf_id="GEM/e2e_nlg",                 task_name="e2e_nlg",      task_type="data2text",      decoding=DECODING_MED,  metrics=["bleu"]),
    dict(hf_id="GEM/totto",                   task_name="totto",        task_type="data2text",      decoding=DECODING_MED,  metrics=["bleu"]),
    dict(hf_id="GEM/cs_restaurants",          task_name="cs_restaurants", task_type="data2text",    decoding=DECODING_MED,  metrics=["bleu"]),
    dict(hf_id="GEM/turku_hockey_data2text",  task_name="turku_hockey_data2text", task_type="data2text", decoding=DECODING_MED, metrics=["bleu"]),
    dict(hf_id="GEM/mlb_data_to_text",        task_name="mlb_data_to_text", task_type="data2text",  decoding=DECODING_MED,  metrics=["bleu"]),

    # ---------- Summarization ----------
    # <-- MODIFIED: Changed hf_id from "GEM/xsum" to "xsum"
    dict(hf_id="xsum",          task_name="xsum",          task_type="summarization", decoding=DECODING_MED,  metrics=["rouge"]),
    # <-- MODIFIED: Added hf_config="english"
    dict(hf_id="GEM/xlsum",         task_name="xlsum",         task_type="summarization", decoding=DECODING_LONG, metrics=["rouge"], hf_config="english"),
    # <-- MODIFIED: Added hf_config="de" (German, as an example)
    dict(hf_id="GEM/mlsum",         task_name="mlsum",         task_type="summarization", decoding=DECODING_LONG, metrics=["rouge"], hf_config="de"),
    dict(hf_id="GEM/xwikis",        task_name="xwikis",        task_type="summarization", decoding=DECODING_LONG, metrics=["rouge"]),
    dict(hf_id="GEM/wiki_lingua",   task_name="wiki_lingua",   task_type="summarization", decoding=DECODING_LONG, metrics=["rouge"]),
    dict(hf_id="GEM/wiki_cat_sum",  task_name="wiki_cat_sum",  task_type="summarization", decoding=DECODING_LONG, metrics=["rouge"]),

    # ---------- Dialog ----------
    dict(hf_id="GEM/schema_guided_dialog",    task_name="schema_guided_dialog", task_type="dialog", decoding=DECODING_SMALL, metrics=["bleu"]),
    dict(hf_id="GEM/conversational_weather",  task_name="conversational_weather", task_type="dialog", decoding=DECODING_SMALL, metrics=["bleu"]),

    # ---------- Paraphrase ----------
    # <-- MODIFIED: Added hf_config="en.100" (English, 100% threshold)
    dict(hf_id="GEM/opusparcus",    task_name="opusparcus",    task_type="paraphrase",    decoding=DECODING_SMALL, metrics=["bleu"], hf_config="en.100"),

    # ---------- QG ----------
    dict(hf_id="GEM/squad_v2",      task_name="squad_v2",      task_type="question_gen", decoding=DECODING_SMALL, metrics=["bleu"]),

    # ---------- Simplification ----------
    # NOTE: no plain "test" → we override to test_asset by default (common in GEM release)
    # NOTE: This dataset will still fail due to library incompatibility, which is correct.
    dict(
        hf_id="GEM/wiki_auto_asset_turk",
        task_name="wiki_auto_asset_turk",
        task_type="simplification",
        decoding=DECODING_MED,
        metrics=["rouge"],
        split_overrides={"train": "train", "validation": "validation", "test": "test_asset"},
    ),
]

# Instantiate registry
TASKS: OrderedDict[str, GEMGenericTask] = OrderedDict()
for spec in _GEM_TASK_SPECS:
    # This line (from your original) correctly renames 'metrics' to 'metric_names'
    spec['metric_names'] = spec.pop('metrics') 
    task = GEMGenericTask(**spec)
    TASKS[task.name] = task


class AutoTask:
    @classmethod
    def get(cls, task_name: str, seed: int = 42) -> AbstractTaskDataset:
        if task_name in TASKS:
            # (Optional) re-seed copy per request
            t = TASKS[task_name]
            t.seed = seed
            return t
        raise ValueError(
            f"Unrecognized task '{task_name}'. "
            f"Available: {', '.join(TASKS.keys())}"
        )

"""
# -----------------------
# Smoke test CLI
# -----------------------
def _preview(dataset: datasets.Dataset, task_name: str, add_prefix: bool, n: int = 2) -> None:
    # run the preprocessor just like get_dataset() does, but print a couple samples
    logger.info("Map: converting to seq2seq format …")
    dd = dataset.map(
        lambda ex: TASKS[task_name].preprocessor(ex, add_prefix=add_prefix),
        remove_columns=dataset.column_names,
    )
    for i in range(min(n, len(dd))):
        ex = dd[i]
        logger.info("    👉 src_texts: %s", (ex["src_texts"][:160] + "…") if len(ex["src_texts"]) > 160 else ex["src_texts"])
        logger.info("       tgt_texts: %s", (ex["tgt_texts"][:160] + "…") if len(ex["tgt_texts"]) > 160 else ex["tgt_texts"])

def smoke(
    n_obs_per_split: int = 2,
    add_prefix: bool = True,
    split_validation_test: bool = False,
    fast_only: bool = False,
    tasks_subset: Optional[List[str]] = None,
) -> None:
    logger.info("\n======== GEM data smoke test ========")
    logger.info("n_obs_per_split=%s | add_prefix=%s | split_validation_test=%s | fast_only=%s",
                n_obs_per_split, add_prefix, split_validation_test, fast_only)
    logger.info("=====================================\n")

    attempted = 0
    ok_tasks = 0
    failed_tasks = []

    names = tasks_subset if tasks_subset else list(TASKS.keys())
    for name in names:
        task = TASKS[name]
        attempted += 1
        logger.info("--- [%s] -------------------------------------------", name)
        logger.info("task_type=%s", task.task_type)
        logger.info("decoding=%s | metrics=%s", task.task_specific_config, [m.__name__ for m in task.metrics])
        
        has_failure = False

        # Try each split with mapped/fallback logic
        for split in ("train", "validation", "test"):
            try:
                # We load the raw dataset first to catch errors before mapping
                raw_ds = task.load_dataset(task.split_to_data_split[split])
                # Select a tiny sample *before* shuffling for speed
                sample_indices = list(range(min(n_obs_per_split, len(raw_ds))))
                sampled_raw = raw_ds.select(sample_indices)
                
                logger.info("  [%s] ok (raw load)", split)
                _preview(
                    sampled_raw,
                    task_name=name,
                    add_prefix=add_prefix,
                    n=n_obs_per_split,
                )
            except Exception as e:
                logger.info("  [%s] load FAILED: %s", split, str(e))
                has_failure = True
        
        if not has_failure:
            ok_tasks += 1
        else:
            failed_tasks.append(name)
            
        logger.info("")

    logger.info("============= SUMMARY =============")
    logger.info("Tasks attempted: %d", attempted)
    logger.info("Tasks loaded successfully: %d/%d", ok_tasks, attempted)
    if failed_tasks:
        logger.info("Tasks with failures: %s", ", ".join(failed_tasks))
    logger.info("===================================")


def main():
    parser = argparse.ArgumentParser(description="GEM tasks smoke test")
    parser.add_argument("--smoke", action="store_true", help="Run a quick data smoke test")
    # --- Start of Fix ---
    parser.add_argument("--n", type=int, default=2, help="Num examples per split")
    parser.add_argument("--no-prefix", action="store_true", help="Disable textual prefixing")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated subset of task names")
    # --- End of Fix ---
    args = parser.parse_args()

    if not args.smoke:
        logger.info("Nothing to do. Try: --smoke  or  --tasks xsum,web_nlg --smoke")
        return

    # --- Start of Fix 2 ---
    subset = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    # --- End of Fix 2 ---
    smoke(
        n_obs_per_split=args.n,
        add_prefix=not args.no_prefix,
        split_validation_test=False,
        fast_only=False,
        tasks_subset=subset,
    )

#if __name__ == "__main__":
#    main()
"""



