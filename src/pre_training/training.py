#!/usr/bin/env python
"""
Train GPT‑2‑small (~124 M params) from scratch on the BabyLM Strict 100 M‑word
corpus.  Checkpoints are saved at the official BabyLM word‑count milestones and
pushed to the Hugging Face Hub on separate branches (e.g. chck_7M).

Usage
-----
python train_babylm_gpt2.py \
  --hub_repo YOUR_USERNAME/babylm-gpt2-small \
  --dataset_fraction 0.2 \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --fp16
"""

import math, os, argparse, shutil
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    GPT2Config, GPT2TokenizerFast, GPT2LMHeadModel,
    Trainer, TrainingArguments, TrainerCallback, set_seed
)
from huggingface_hub import HfApi

import time

def retry_hf(fn, max_retries=5, initial_wait=5, *args, **kwargs):
    """Retry Hugging Face Hub operations on network/server errors."""
    attempts = 0
    wait = initial_wait
    while attempts < max_retries:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempts += 1
            print(f"[WARN] HuggingFace Hub call failed: {e}. Retrying ({attempts}/{max_retries}) in {wait}s …")
            time.sleep(wait)
            wait *= 2  # exponential backoff
    raise RuntimeError(f"After {max_retries} retries, HuggingFace Hub call failed permanently.")

# ---------------------------------------------------------------------------
# 1  Command‑line arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train GPT‑2‑small on BabyLM Strict (nilq/babylm-100M)."
)
parser.add_argument("--hub_repo", type=str, required=True,
                    help="HF repo ID (username/repo) to push checkpoints & final model.")
parser.add_argument("--dataset_fraction", type=float, default=1.0,
                    help="Fraction of the corpus to use (1.0 = full 100 M words).")
parser.add_argument("--seq_length", type=int, default=512,
                    help="Context window in tokens.")
parser.add_argument("--output_dir", type=str, default="babylm_gpt2_small",
                    help="Local folder for checkpoints.")
parser.add_argument("--epochs", type=int, default=1,
                    help="Epochs over *your* chosen data fraction.")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Per‑device batch size (token sequences per step).")
parser.add_argument("--grad_accum_steps", type=int, default=1,
                    help="Gradient‑accumulation steps.")
parser.add_argument("--learning_rate", type=float, default=5e-5,
                    help="AdamW learning rate.")
parser.add_argument("--fp16", action="store_true", help="Enable FP16 training.")
parser.add_argument("--bf16", action="store_true", help="Enable BF16 training (Ampere+).")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

args = parser.parse_args()
set_seed(args.seed)

if not (0 < args.dataset_fraction <= 1.0):
    raise ValueError("--dataset_fraction must be within (0, 1].")

# ---------------------------------------------------------------------------
# 2  Hub repo setup
# ---------------------------------------------------------------------------

api = HfApi()
api.create_repo(repo_id=args.hub_repo, exist_ok=True)

# ---------------------------------------------------------------------------
# 3  Load BabyLM Strict
# ---------------------------------------------------------------------------

print("🡒 Loading BabyLM Strict 100 M‑word corpus (nilq/babylm-100M) …")
dataset = load_dataset("nilq/babylm-100M", split="train")

if args.dataset_fraction < 1.0:
    dataset = dataset.shuffle(seed=42)
    keep = int(len(dataset) * args.dataset_fraction)
    dataset = dataset.select(range(keep))
    print(f"   Using {keep:,} examples (~{args.dataset_fraction*100:.0f}% of data).")

# ---------------------------------------------------------------------------
# 4  Tokenizer
# ---------------------------------------------------------------------------

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.model_max_length = args.seq_length
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id

# Estimate token/word ratio for checkpoint schedule
sample = dataset.shuffle(args.seed).select(range(min(10_000, len(dataset))))["text"]
words = sum(len(t.split()) for t in sample)
toks = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in sample)
TOKS_PER_WORD = toks / words if words else 1.0
print(f"   ≈{TOKS_PER_WORD:.2f} tokens per word (sample).")

# ---------------------------------------------------------------------------
# 5  Tokenise and chunk into fixed‑length blocks
# ---------------------------------------------------------------------------

def tok_fn(ex):
    return tokenizer(ex["text"], add_special_tokens=False)

def group_fn(batch):
    concat = []
    for ids in batch["input_ids"]:
        concat.extend(ids + [eos_id])
    if concat and concat[-1] == eos_id:
        concat.pop()
    L = (len(concat) // args.seq_length) * args.seq_length
    if L == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    chunks = [concat[i:i+args.seq_length] for i in range(0, L, args.seq_length)]
    attn = [[1]*args.seq_length]*len(chunks)
    return {"input_ids": chunks, "attention_mask": attn, "labels": [c[:] for c in chunks]}

tokenised = dataset.map(tok_fn, batched=True, remove_columns=["text"])
lm_ds = tokenised.map(group_fn, batched=True, batch_size=1000)
lm_ds = lm_ds.filter(lambda x: len(x["input_ids"]) == args.seq_length)
print(f"   Final training sequences: {len(lm_ds):,} of length {args.seq_length}.")

# ---------------------------------------------------------------------------
# 6  Model
# ---------------------------------------------------------------------------

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=args.seq_length,
    n_ctx=args.seq_length,
    n_embd=768, n_layer=12, n_head=12,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
model = GPT2LMHeadModel(config)
print(f"🡒 New GPT‑2‑small initialised ({model.num_parameters():,} parameters).")

# ---------------------------------------------------------------------------
# 7  Training args
# ---------------------------------------------------------------------------

train_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum_steps,
    learning_rate=args.learning_rate,
    warmup_ratio=0.01,
    bf16=args.bf16,
    fp16=args.fp16,
    logging_strategy="steps", 
    logging_steps=100,
    save_strategy="no",
    report_to=["wandb"],
    seed=args.seed,
)

# ---------------------------------------------------------------------------
# 8  Milestone checkpoint callback
# ---------------------------------------------------------------------------

class WordMilestoneCB(TrainerCallback):
    """
    Save & push a checkpoint every time the running *word* counter
    crosses an official BabyLM milestone.

    Parameters
    ----------
    api          : HfApi        – handle to the Hub
    repo         : str          – repo ID (username/model)
    seq_len      : int          – tokens per training example
    tok_per_word : float        – ≈ tokens / word (sample estimate)
    """
    def __init__(self, api, repo, seq_len, tok_per_word,tokenizer):
        super().__init__()
        self.api, self.repo = api, repo
        self.seq_len = seq_len
        # 1 M..10 M, 20 M..100 M, 200 M..1 B
        ms = list(range(1, 11)) + [i * 10 for i in range(2, 11)] + [i * 100 for i in range(2, 11)]
        self.milestones = ms
        self.milestone_toks = [math.ceil(m * 1_000_000 * tok_per_word) for m in ms]
        self.i = 0               # next milestone index
        self.toks = 0            # running token counter
        self.tokenizer = tokenizer

    def on_step_end(self, args, state, control, **kw):
        # one *optimizer* step has just finished (i.e. after grad‑accum)
        tokens_per_update = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.seq_len
            * args.world_size      # all GPUs
        )
        self.toks += tokens_per_update

        while self.i < len(self.milestone_toks) and self.toks >= self.milestone_toks[self.i]:
            words = self.milestones[self.i] * 1_000_000
            branch = f"chck_{self.milestones[self.i]}M"
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{self.milestones[self.i]}M")
            os.makedirs(ckpt_dir, exist_ok=True)

            kw["model"].save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)

            print(f"\n★  Push checkpoint  ≈{words:,} words  →  {branch}")
            retry_hf(self.api.create_branch, repo_id=self.repo, branch=branch, exist_ok=True)
            retry_hf(self.api.upload_folder,
                     repo_id=self.repo, folder_path=ckpt_dir, repo_type="model",
                     revision=branch,
                     commit_message=f"Checkpoint at {words:,} words")
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            self.i += 1
        return control

milestone_cb = WordMilestoneCB(api, args.hub_repo, args.seq_length, TOKS_PER_WORD,tokenizer)

# ---------------------------------------------------------------------------
# 9  Trainer & launch
# ---------------------------------------------------------------------------

trainer = Trainer(
    model=model, args=train_args,
    train_dataset=lm_ds,
    tokenizer=tokenizer,
    callbacks=[milestone_cb]
)

trainer.train()

# ---------------------------------------------------------------------------
# 10  Push final model
# ---------------------------------------------------------------------------

print("\nTraining done – pushing final model to main branch …")
train_args.save(args.output_dir)
trainer.save_model(args.output_dir)  # saves model and tokenizer
retry_hf(trainer.push_to_hub, commit_message="Final model and tokenizer push")
print(f"Load with:  AutoModel.from_pretrained('{args.hub_repo}')")