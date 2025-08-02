from src.interactivelearning.logger import logger
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from trl.core import LengthSampler
from tqdm import tqdm
import re

def sanitize_dataset_name(name: str) -> str:
    """Replace filesystem‑unfriendly chars in a HF repo name."""
    return name.replace("/", "_").replace(" ", "_")



def is_not_empty(example, field_name="input_ids", pad_token_id=0):
    ids = example.get(field_name, [])
    return len(ids) > 0 and any(x != pad_token_id for x in ids)


class DatasetBuilder(ABC):
    """Base builder with optional batched tokenisation.

    `batch` > 0  → vectorised tokenisation via HuggingFace fast path.
    `batch` <= 0 → per‑row tokenisation (slower but minimal memory).
    """

    def __init__(
        self,
        config: Any,
        dataset_name: str,
        *,
        min_len: int = 2,
        max_len: int = 8,
        batch: int = 1024,  
        use_cache: bool = True,
        save_cache: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        self.config = config
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.input_size_sampler = LengthSampler(min_len, max_len)
        self.batch = max(int(batch), 0)  # ensure non‑negative

        self.token_limit: int | None = None  # word‑budget set later

        self.use_cache = use_cache
        self.save_cache = save_cache

        base_cache = cache_dir or "cached_datasets"
        self.cache_dir = os.path.join(base_cache, sanitize_dataset_name(dataset_name))
        os.makedirs(self.cache_dir, exist_ok=True)


    def _get_cache_path(self) -> str:
        return os.path.join(self.cache_dir, "tokenized")


    def set_token_limit(self, token_limit: int) -> None:
        self.token_limit = token_limit
        logger.info("Word‑limit set to %d", token_limit)


    @abstractmethod
    def load(self):
        ...

    @abstractmethod
    def _raw_split(self):
        """Return `(split_name, text_field)` pair."""
        ...


    def _tokenize_batch(self, batch: Dict[str, List[str]], text_key: str):
        """Vectorised tokenization for a batch of texts."""
        texts: List[str] = batch[text_key]
        ids_batch = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        out = {"input_ids": [], "query": []}
        for ids in ids_batch:
            ids = ids[: self.input_size_sampler()]
            out["input_ids"].append(ids)
            out["query"].append(self.tokenizer.decode(ids))
        return out

    def _tokenize_row(self, sample: Dict[str, str], text_key: str):
        """Row‑wise tokenization (fallback when `batch == 0`)."""
        ids = self.tokenizer.encode(sample[text_key])[: self.input_size_sampler()]
        return {"input_ids": ids, "query": self.tokenizer.decode(ids)}


    def _truncate_to_token_limit(self, dataset):
        if self.token_limit is None:
            return dataset
        total_words = 0
        selected = []
        for sample in dataset:
            words = len(sample["query"].split())
            if total_words + words > self.token_limit:
                break
            selected.append(sample)
            total_words += words
        logger.info("Dataset truncated to %d words (%d rows)", total_words, len(selected))
        return Dataset.from_list(selected)


class IMDBDatasetBuilder(DatasetBuilder):
    def __init__(self, config, **kwargs):
        super().__init__(config, "stanfordnlp/imdb", **kwargs)

    def _raw_split(self):
        return "train", "review"

    def load(self):
        split_name, text_key = self._raw_split()
        logger.info("Loading IMDB (%s)…", split_name)
        cache_path = self._get_cache_path()
        if self.use_cache and os.path.exists(cache_path):
            logger.info("→ using cache %s", cache_path)
            ds = load_from_disk(cache_path)
        else:
            ds = load_dataset(self.dataset_name, split=split_name)
            ds = ds.rename_columns({"text": text_key})
            ds = ds.filter(lambda x: len(x[text_key]) > 200, batched=False)
            if self.batch > 0:
                ds = ds.map(
                    lambda batch: self._tokenize_batch(batch, text_key),
                    batched=True,
                    batch_size=self.batch,
                    remove_columns=list(ds.column_names),
                )
            else:  # row‑wise
                ds = ds.map(
                    lambda sample: self._tokenize_row(sample, text_key),
                    batched=False,
                    remove_columns=list(ds.column_names),
                )
            if self.save_cache:
                logger.info("→ saving cache to %s", cache_path)
                ds.save_to_disk(cache_path)
        ds = self._truncate_to_token_limit(ds)                
        ds.set_format(type="torch")
        logger.info("IMDB ready: %d rows", len(ds))
        return ds

class TinyStoriesDatasetBuilder(DatasetBuilder):
    def __init__(self, config, **kwargs):
        super().__init__(config, "roneneldan/TinyStories", **kwargs)

    def _raw_split(self):
        return "train", "text"

    def load(self):
        split_name, text_key = self._raw_split()
        logger.info("Loading TinyStories (%s)…", split_name)
        cache_path = self._get_cache_path()
        if self.use_cache and os.path.exists(cache_path):
            logger.info("→ using cache %s", cache_path)
            ds = load_from_disk(cache_path)
        else:
            ds = load_dataset(self.dataset_name, split=split_name)
            if self.batch > 0:
                ds = ds.map(
                    lambda batch: self._tokenize_batch(batch, text_key),
                    batched=True,
                    batch_size=self.batch,
                    remove_columns=list(ds.column_names),
                )
            else:
                ds = ds.map(
                    lambda sample: self._tokenize_row(sample, text_key),
                    batched=False,
                    remove_columns=list(ds.column_names),
                )
            if self.save_cache:
                logger.info("→ saving cache to %s", cache_path)
                ds.save_to_disk(cache_path)

        ds = ds.filter(lambda ex: is_not_empty(ex, field_name="input_ids", pad_token_id=self.tokenizer.pad_token))
        ds = self._truncate_to_token_limit(ds)                
        ds.set_format(type="torch")
        logger.info("TinyStories ready: %d rows", len(ds))
        return ds


class WritingPromptsDatasetBuilder(DatasetBuilder):
    def __init__(self, config, **kwargs):
        super().__init__(config, "euclaise/writingprompts", **kwargs)

    def _raw_split(self):
        return "train", "prompt"


    def replace_bracketed_tag_and_add_story(self, example, text_key="prompt"):
        text = example[text_key]
        # Replace leading [ ... ] (with optional spaces) with 'Story idea: '
        text = re.sub(r'^\s*\[\s*.*?\s*\]\s*', 'Story idea: ', text, flags=re.IGNORECASE)
        # Add ' Story:' at the end if not already present
        if not text.rstrip().endswith("Story:"):
            text = text.rstrip() + " Story:"
        example[text_key] = text
        return example


    def load(self):
        split_name, text_key = self._raw_split()
        logger.info("Loading WritingPrompts (%s)…", split_name)
        cache_path = self._get_cache_path()
        if self.use_cache and os.path.exists(cache_path):
            logger.info("→ using cache %s", cache_path)
            ds = load_from_disk(cache_path)
        else:
            ds = load_dataset(self.dataset_name, split=split_name)
            ds = ds.filter(lambda x: len(x[text_key].split()) < 50, batched=False)
            ds = ds.map(lambda ex: self.replace_bracketed_tag_and_add_story(ex, text_key=text_key), batched=False)
            if self.batch > 0:
                ds = ds.map(
                    lambda batch: self._tokenize_batch(batch, text_key),
                    batched=True,
                    batch_size=self.batch,
                    remove_columns=list(ds.column_names),
                )
            else:
                ds = ds.map(
                    lambda sample: self._tokenize_row(sample, text_key),
                    batched=False,
                    remove_columns=list(ds.column_names),
                )
            if self.save_cache:
                logger.info("→ saving cache to %s", cache_path)
                ds.save_to_disk(cache_path)
        ds = self._truncate_to_token_limit(ds)
        ds.set_format(type="torch")
        logger.info("WritingPrompts ready: %d rows", len(ds))
        return ds
    

class DatasetCombiner:
    """Greedily fills the total word budget with datasets in given order.

    Example: token_limit=100M, builders=[IMDB, Tiny].
    → IMDB gets up to 100M words; if it outputs 60M, Tiny gets remaining 40M.
    """

    def __init__(self, builders):
        self.builders = builders
        self.total_word_budget: int | None = None


    def set_token_limit(self, token_limit: int):
        self.total_word_budget = token_limit
        logger.info("Total word budget set to %d (greedy allocation)", token_limit)

    def load(self):
        if self.total_word_budget is None:
            raise ValueError("set_token_limit() must be called before load().")

        remaining = self.total_word_budget
        datasets = []

        for builder in self.builders:
            if remaining <= 0:
                logger.info("Word budget exhausted; skipping remaining builders …")
                break

            builder.set_token_limit(remaining)
            ds = builder.load()
            datasets.append(ds)


            words_used = sum(len(sample["query"].split()) for sample in ds)
            remaining -= words_used
            logger.info(
                "Builder %s consumed %d words → %d remaining",
                builder.dataset_name,
                words_used,
                remaining,
            )

        logger.info("Concatenating %d datasets (final budget used: %d / %d)", len(datasets), self.total_word_budget - remaining, self.total_word_budget)
        return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
