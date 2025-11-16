"""
Implements different GEM tasks and defines the processors to convert each dataset
to a sequence-to-sequence format for multi-task training.

This structure is based on the GLUE task processor and adapted for the
GEM (Generation, Evaluation, and Metrics) benchmark.
"""

from collections import OrderedDict
import abc
import datasets
import functools
import logging
import numpy as np
import torch
from typing import Callable, Dict, Mapping, List

# --- Metrics Stub ---
# You will need to replace this with your actual metrics module/functions
# (e.g., from `datasets.load_metric` or your own implementation)
class StubMetrics:
    def __getattr__(self, name):
        def stub_metric(*args, **kwargs):
            logging.warning(f"Metrics module not fully implemented. Called '{name}' as a stub.")
            return 0.0
        return stub_metric

metrics = StubMetrics()
# --------------------

logger = logging.getLogger(__name__)


# --- Helper Function Stubs ---
# This helper was used in the GLUE code for classification tasks.
def compute_task_max_decoding_length(label_list: List[str]) -> int:
    """Computes max decoding length for classification tasks."""
    # Simple heuristic: max length of label strings + a small buffer
    if not label_list:
        return 2
    return max(len(label) for label in label_list) + 2
# -----------------------------


class AbstractTaskDataset(abc.ABC):
    """
    Defines the abstract class for all the tasks.
    (This is the same base class you provided)
    """
    name = NotImplemented
    task_specific_config: Dict = NotImplemented
    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}

    # These lists are from the GLUE/SuperGLUE setup and may not be
    # relevant for GEM, but are kept for compatibility with the base class.
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "trec", "superglue-cb", "sick",
                                         "mrpc", "stsb", "imdb", "commonsense_qa", "superglue-boolq"]
    large_data_without_all_splits = ["yelp_polarity", "qqp", "qnli",
                                     "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "sst2"]

    def __init__(self, seed=42):
        self.seed = seed

    def get_sampled_split(self, split: int, n_obs: int = None):
        split = self.split_to_data_split[split]
        dataset = self.load_dataset(split)
        total_size = len(dataset)
        n_obs = self.check_n_obs(n_obs, total_size)
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def get_shuffled_sampled_split(self, split: int, n_obs: int = None):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(mapped_split)
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        dataset = self.select_dataset_samples(indices, dataset, n_obs=n_obs)
        return dataset

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def select_dataset_samples(self, indices, dataset, n_obs: int = None):
        n_obs = self.check_n_obs(n_obs, len(indices))
        indices = indices[:n_obs] if n_obs is not None else indices
        return dataset.select(indices)

    def load_dataset(self, split: int):
        # Default loader. Can be overridden by subclasses if needed (e.g., for GEM configs).
        # Most GEM tasks are configs of the 'gem' dataset.
        # This default loader assumes the task name IS the dataset name (like DART).
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def get_train_split_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["train"]
        dataset = self.load_dataset(mapped_split)
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        validation_size = 1000
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def get_half_validation_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["validation"]
        dataset = self.load_dataset(mapped_split)
        validation_size = len(dataset)
        indices = torch.randperm(validation_size, generator=generator).tolist()
        if split == "validation":
            return indices[:(validation_size // 2)]
        else:
            return indices[validation_size // 2:]

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # (Logic from base class, handles train/val/test splitting for GLUE)
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        
        # Apply the task-specific preprocessor
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix), remove_columns=dataset.column_names, load_from_cache_file=False)

    def seq2seq_format(self, src_strs: List[str], tgt_strs: List[str],
                       add_prefix: bool = False, prefix: str = None):
        src_prefix = self.name if prefix is None else prefix
        src_strs = src_prefix + ": " + ' '.join(src_strs) if add_prefix else src_strs
        return {"src_texts": src_strs,
                "tgt_texts": ' '.join(tgt_strs),
                "task": self.name}


# --- DART Helpers (from your example) ---

# Instructional prompt for Gemma model
INSTRUCTION_TEMPLATE = (
    "You are a helpful data-to-text assistant. "
    "Generate a coherent, natural-language sentence or paragraph that "
    "correctly describes all the information in the following triples:\n{triples}"
)

# --- Define the instruction template ---
E2E_NLG_INSTRUCTION = (
    "You are a data-to-text assistant. Convert the following meaning representation into ONE fluent sentence.\n"
    "Provide your final answer in a JSON block, marked with 'Final JSON:'.\n\n"
    "Meaning Representation:\n"
    "{data}\n\n"
    "Final JSON:\n"
    '{{"sentence": "<your sentence here>"}}\n'
)


def _linearize_triple(triple: List[str]) -> str:
    """Converts a DART triple (list of 3 strings) into a single string."""
    if not isinstance(triple, list) or len(triple) != 3:
        logging.warning(f"Malformed triple: {triple}. Skipping.")
        return ""
    # Ensure all parts are strings
    s, p, o = str(triple[0]), str(triple[1]), str(triple[2])
    return f"{s} | {p} | {o}"

# --- GEM Task Definitions ---



class E2ENLGTaskDataset(AbstractTaskDataset):
    """
    Task processor for the E2E_NLG dataset, using a specific
    JSON-based instructional prompt for Gemma.
    """
    name = "e2e_nlg"
    # E2E outputs are relatively short sentences
    task_specific_config = {'max_length': 128, 'num_beams': 1}
    metrics = [metrics.bleu] # Example metric

    def load_dataset(self, split: int):
        """Loads the standalone e2e_nlg dataset."""
        return datasets.load_dataset(
            "GEM/e2e_nlg", 
            split=split, 
            trust_remote_code=True
        )

    def preprocessor(self, example, add_prefix=True):
        """
        Applies the JSON-based instructional prompt.
        'add_prefix' is ignored as we use a full, custom prompt.
        """
        
        data_block = example.get("meaning_representation", "")
        src_text = E2E_NLG_INSTRUCTION.format(data=data_block)
        
        # --- 2. Target Text Logic (from _get_target_text) ---
        # Get the target text from the correct field
        tgt = example.get("human_reference", None) # 'e2e_nlg' specific field
        
        if tgt is None:
            tgt = example.get("target", None)
        if tgt is None:
            tgt = example.get("text", None)
        if tgt is None:
            refs = example.get("references", None)
            if isinstance(refs, list) and refs:
                tgt = refs[0] # Use the first reference
        if tgt is None:
            tgt = "" # Default to empty string
        
        # Handle cases where the target might be a list
        if isinstance(tgt, list) and tgt:
            tgt_text = str(tgt[0])
        else:
            tgt_text = str(tgt)

        # --- 3. Return the final format ---
        # Note: This trains the model to output the raw sentence,
        # not the JSON block mentioned in the prompt.
        return {
            "src_texts": src_text,
            "tgt_texts": tgt_text,
            "task": self.name
        }

class DARTTaskDataset(AbstractTaskDataset):
    """Task processor for the DART (data-to-text) dataset."""
    name = "dart"
    # GEM tasks are generative, so we set a reasonable max output length
    task_specific_config = {'max_length': 256, 'num_beams': 1}
    metrics = [metrics.bleu, metrics.rouge] # Example metrics
    
    def load_dataset(self, split: int):
        # DART is a standalone dataset on Hugging Face
        return datasets.load_dataset("dart", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        """
        Build src_texts from the tripleset using the Gemma instruction;
        target from target / text / references.
        (Adapted from your provided preprocessor function)
        """
        triples = example.get("tripleset", [])
        triple_strings: List[str] = [_linearize_triple(t) for t in triples if t]

        if triple_strings:
            triples_block = "\n".join(f"- {ts}" for ts in triple_strings)
        else:
            triples_block = "- (none)"

        # Apply the instructional prompt
        src_text = INSTRUCTION_TEMPLATE.format(triples=triples_block)
        
        # Logic to find the target text
        tgt = example.get("target", None)
        if tgt is None:
            tgt = " ".join(example.get('annotations', None).get('text',None))
        if tgt is None:
            refs = example.get("references", None)
            if isinstance(refs, list) and refs:
                tgt = refs[0] # Use the first reference
        if tgt is None:
            tgt = "" # Default to empty string if no target found
            logging.warning(f"No target found for DART example: {example}")

        # The base class's `seq2seq_format` expects lists of strings
        # But our prompt is already a single formatted string.
        # We also don't want to add the task-name prefix, as we have a full prompt.
        
        # Use custom formatting to override default prefixing
        return {
            "src_texts": str(src_text),
            "tgt_texts": str(tgt),
            "task": self.name
        }

class CommonGenTaskDataset(AbstractTaskDataset):
    """Task processor for the Common Gen dataset."""
    name = "common_gen"
    task_specific_config = {'max_length': 128, 'num_beams': 1}
    metrics = [metrics.bleu, metrics.rouge]

    def load_dataset(self, split: int):
        # Common Gen is also standalone
        return datasets.load_dataset("common_gen", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        # Input `example` has 'concepts' (list) and 'target' (string)
        concepts = example.get("concepts", [])
        target = example.get("target", "")

        src_texts = ["generate a sentence with concepts:", ", ".join(concepts)]
        tgt_texts = [target]
        
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

class XSumTaskDataset(AbstractTaskDataset):
    """Task processor for the XSum summarization dataset."""
    name = "xsum"
    task_specific_config = {'max_length': 64, 'num_beams': 1} # XSum summaries are short
    metrics = [metrics.rouge]

    def load_dataset(self, split: int):
        # XSum is standalone
        return datasets.load_dataset("xsum", split=split, trust_remote_code=True)

    def preprocessor(self, example, add_prefix=True):
        # Input `example` has 'document' and 'summary'
        document = example.get("document", "")
        summary = example.get("summary", "")

        src_texts = [document] # The document itself is the source
        tgt_texts = [summary]
        
        # Use a custom prefix for summarization tasks
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="summarize:")

class WebNLGTaskDataset(AbstractTaskDataset):
    """Task processor for the WebNLG (data-to-text) dataset."""
    name = "web_nlg"
    task_specific_config = {'max_length': 256, 'num_beams': 4}
    metrics = [metrics.bleu]

    def load_dataset(self, split: int):
        # WebNLG is a config of the 'gem' dataset
        # We must override load_dataset
        return datasets.load_dataset(
            "gem", 
            "web_nlg_en", 
            split=split, 
            trust_remote_code=True  # Added from our last conversation
        )

    def preprocessor(self, example, add_prefix=True):
        """
        --- MODIFIED PREPROCESSOR ---
        'target' in web_nlg is a list of strings (references).
        We take the first reference as the target for training.
        """
        src_input = example.get("input", "")
        
        # --- FIX IS HERE ---
        # Get the list of targets. Default to an empty list.
        target_list = example.get("target", [])
        
        # Select the first target if the list is not empty, otherwise use an empty string.
        target = target_list[0] if target_list else ""
        # --- END FIX ---

        src_texts = ["generate from triples:", src_input]
        tgt_texts = [target]  # This is now guaranteed to be [str]
        
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


# --- Task Collator Mapping ---

TASKS = OrderedDict([
    # Add your GEM tasks here
    ('dart', DARTTaskDataset),
    ('common_gen', CommonGenTaskDataset),
    ('xsum', XSumTaskDataset),
    ('e2e_nlg', E2ENLGTaskDataset),
    ('web_nlg', WebNLGTaskDataset),
    
    # You can add more tasks here by creating a new class
    # for each one and implementing its 'preprocessor'.
    # ('mlsum_de', MLSumDETaskDataset),
    # ('totto', TottoTaskDataset),
])


class AutoTask:
    """
    This is your "Task Collator" factory.
    It retrieves the correct task processor class based on the task name.
    (This class is the same as in your provided code)
    """
    @classmethod
    def get(self, task_name, seed=42):
        if task_name in TASKS:
            return TASKS[task_name](seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                task_name, list(TASKS.keys())
            )
        )


# --- Example Usage ---
if __name__ == "__main__":
    print("--- 🚀 Initializing GEM Task Collator ---")
    
    # 1. Get the task processor for 'dart'
    dart_task = AutoTask.get("dart")
    
    # 2. Load and preprocess the dataset
    # We get 5 validation samples for demonstration
    print("\n--- 1. Loading and processing 'dart' task ---")
    try:
        dart_dataset = dart_task.get_dataset(split="validation", n_obs=5)
        
        print(f"Loaded {len(dart_dataset)} 'dart' samples.")
        print("\nExample processed sample:")
        sample = dart_dataset[0]
        print(f"TASK: {sample['task']}")
        print(f"SOURCE (src_texts):\n{sample['src_texts'][:500]}...")
        print(f"\nTARGET (tgt_texts):\n{sample['tgt_texts']}")

    except Exception as e:
        print(f"Could not load 'dart' dataset. (Maybe network issue or 'fsspec' error?): {e}")


    # 3. Get the task processor for 'xsum'
    xsum_task = AutoTask.get("xsum")
    
    # 4. Load and preprocess the dataset
    print("\n--- 2. Loading and processing 'xsum' task ---")
    try:
        xsum_dataset = xsum_task.get_dataset(split="validation", n_obs=5, add_prefix=True)
        
        print(f"Loaded {len(xsum_dataset)} 'xsum' samples.")
        print("\nExample processed sample:")
        sample = xsum_dataset[0]
        print(f"TASK: {sample['task']}")
        print(f"SOURCE (src_texts):\n{sample['src_texts'][:500]}...")
        print(f"\nTARGET (tgt_texts):\n{sample['tgt_texts']}")
        
    except Exception as e:
        print(f"Could not load 'xsum' dataset. (Maybe network issue?): {e}")