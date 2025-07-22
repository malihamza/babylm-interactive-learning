import os
import csv
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from trl import PPOTrainer
from trl.core import LengthSampler
from tqdm import tqdm
from src.interactivelearning.logger import logger
import statistics
from datetime import datetime
from src.interactivelearning.schemas import PromptCompletionPair

def format_tokens(num):
    if num >= 1_000_000:
        return f"{num // 1_000_000}M"
    elif num >= 1_000:
        return f"{num // 1_000}K"
    return str(num)

def make_model_output_dir(base_dir, model_name, total_tokens):
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    dir_name = f"{model_name}_{format_tokens(total_tokens)}_tokens__{timestamp}"
    full_path = os.path.join(base_dir, dir_name)
    checkpoints_path = os.path.join(full_path, "checkpoints")
    metadata_path = os.path.join(full_path, "meta_data")
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)
    return checkpoints_path, metadata_path

class CustomPPOTrainer(PPOTrainer):
    def __init__(self, config, model, ref_model, tokenizer, dataset, reward_fn, save_base_dir="saved_models"):
        config.learning_rate = getattr(config, "learning_rate", 1.41e-5)
        super().__init__(
            config, model, ref_model, tokenizer,
            dataset=dataset, data_collator=self.collator
        )
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn

        min_len = getattr(config, "output_min_length", 4)
        max_len = getattr(config, "output_max_length", 16)
        logger.info(f" Minimum length received {min_len}")
        logger.info(max_len)
        self.output_length_sampler = LengthSampler(min_len, max_len)

        self._generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        self.checkpoint_interval = getattr(config, "checkpoint_interval", 10000)
        logger.info(f"Checkpoint interval recieved {self.checkpoint_interval}")

        self.total_token_nums = getattr(config, "token_limit", 0)
        
        self.total_tokens_processed = 0
        self.next_checkpoint_tokens = self.checkpoint_interval

        self.save_base_dir = save_base_dir
        self.checkpoint_dir = None
        self.meta_data_dir = None
        self.model_name = config.model_name.replace("/", "_")
        self.generated_data_log = []
        self.per_batch_logs = []


        logger.info("Initialized CustomPPOTrainer")

    def set_generation_kwargs(self, **kwargs):
        logger.info(f"Setting generation kwargs: {kwargs}")
        self._generation_kwargs.update(kwargs)

    def collator(self, data):
        if not data:
            return None
        try:
            return {
                "input_ids": [torch.tensor(d["input_ids"]) for d in data],
                "query": [d["query"] for d in data]
            }
        except Exception as e:
            logger.warning(f"Error in collator: {e}")
            return None

    def save_checkpoint(self):
        token_str = format_tokens(self.total_tokens_processed)
        save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{token_str}_tokens")
        logger.info(f"Saving checkpoint at {token_str} tokens -> {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def save_metadata_csv(self):
        if not self.generated_data_log:
            return

        output_file = os.path.join(self.meta_data_dir, "generated_outputs.csv")
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["response", "reward"])
            writer.writerows(self.generated_data_log)

        logger.info(f"Saved metadata CSV with {len(self.generated_data_log)} rows at {output_file}")

    def save_training_curve_csv(self):
        if not self.per_batch_logs:
            return
        output_file = os.path.join(self.meta_data_dir, "training_stats.csv")
        with open(output_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.per_batch_logs[0].keys())
            writer.writeheader()
            writer.writerows(self.per_batch_logs)

        logger.info(f"Saved training curve stats at {output_file}")

    def log_batch_stats(self, rewards, stats):
        rewards = [r.item() if isinstance(r, torch.Tensor) else r for r in rewards]
        avg = sum(rewards) / len(rewards)
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0

        kl = stats.get("kl", 0.0)
        entropy = stats.get("entropy", 0.0)
        policy_loss = stats.get("policy", 0.0)
        value_loss = stats.get("value", 0.0)

        logger.info(
            f"Batch Stats — Reward Avg: {avg:.4f}, Std: {std:.4f} | "
            f"KL: {kl:.4f}, Entropy: {entropy:.4f}, "
            f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}"
        )

        self.per_batch_logs.append({
            "avg_reward": avg,
            "std_reward": std,
            "kl": kl,
            "entropy": entropy,
            "policy_loss": policy_loss,
            "value_loss": value_loss
        })

    def log_training_summary(self):
        if not self.per_batch_logs:
            return
        avg_rewards = [entry["avg_reward"] for entry in self.per_batch_logs]
        overall_avg = sum(avg_rewards) / len(avg_rewards)
        overall_std = statistics.stdev(avg_rewards) if len(avg_rewards) > 1 else 0.0
        logger.info(f"Training Summary: {len(self.per_batch_logs)} batches | Overall Avg Reward = {overall_avg:.4f}, Std = {overall_std:.4f}")

    def run_training_loop(self, num_epochs=1, num_batches=2):
        logger.info(f"Starting training loop for {num_batches} batches")

        self.checkpoint_dir, self.meta_data_dir = make_model_output_dir(
            self.save_base_dir, self.model_name, self.total_token_nums
        )

        for _ in range(num_epochs):
            for batch in tqdm(self.dataloader):
                if batch is None:
                    logger.warning("Received None batch from dataloader, skipping...")
                    continue
                if not isinstance(batch, dict) or "input_ids" not in batch:
                    logger.warning("Malformed batch encountered, skipping...")
                    continue

                query_tensors = batch["input_ids"]
                gen_lens = [self.output_length_sampler() for _ in range(len(query_tensors))]

                responses = self.generate(
                    query_tensors,
                    max_new_tokens=max(gen_lens),
                    **self._generation_kwargs
                )

                response_tensors = [r[len(q):len(q) + l] for r, q, l in zip(responses, query_tensors, gen_lens)]

                batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]
                texts = [PromptCompletionPair(q,r) for q, r in zip(batch["query"], batch["response"])]
                rewards = self.reward_fn(texts)

                stats = self.step(query_tensors, response_tensors, rewards)
                self.log_stats(stats, batch, rewards)
                self.log_batch_stats(rewards, stats)

                processed_tokens = sum(len(q) for q in query_tensors)
                self.total_tokens_processed += processed_tokens

                self.generated_data_log.extend(
                    [(r, reward.item() if isinstance(reward, torch.Tensor) else reward)
                     for r, reward in zip(batch["response"], rewards)]
                )

                if self.total_tokens_processed >= self.next_checkpoint_tokens:
                    self.save_checkpoint()
                    self.next_checkpoint_tokens += self.checkpoint_interval

        logger.info("Training loop completed. Saving final checkpoint and metadata.")
        self.save_checkpoint()
        self.save_metadata_csv()
        self.save_training_curve_csv()
        self.log_training_summary()
