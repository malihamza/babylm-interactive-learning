import os
import csv
import statistics
from datetime import datetime
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import PPOTrainer
from trl.core import LengthSampler

from src.interactivelearning.logger import logger
from src.interactivelearning.schemas import PromptCompletionPair
import shutil
from huggingface_hub import HfApi
import wandb

def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def schedule_next_ckpt(words_so_far: int) -> int:
    """Compute the next checkpoint threshold based on words processed."""
    if words_so_far < 10_000_000:
        step = 1_000_000
    elif words_so_far < 100_000_000:
        step = 10_000_000
    else:
        step = 100_000_000
    return ((words_so_far // step) + 1) * step


class CustomPPOTrainer(PPOTrainer):
    """Train with per‑epoch word budgets and push checkpoints to the Hugging Face Hub."""

    def __init__(
        self,
        config,
        model,
        ref_model,
        tokenizer: AutoTokenizer,
        dataset,
        reward_fn,
        *,
        hf_base_repo: str | None = None,  
        hf_org: str | None = None,
        save_meta_dir: str | None = None,  
        word_budget: int = 100_000_000,
        gen_word_budget: int = 100_000_000,
        save_base_dir: str | None = None,

        wandb_project: str | None = None,
        wandb_tags: list[str] | None = None,
        watch_model: bool = False,
    ) -> None:
        super().__init__(
            config,
            model,
            ref_model,
            tokenizer,
            dataset=dataset,
            data_collator=self._collate,
        )

        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.word_budget = word_budget
        self.gen_word_budget = gen_word_budget


        self.api = HfApi()
        base_name = (hf_base_repo or config.model_name).replace("/", "-")
        name_with_budget = f"{base_name}_ppo_{fmt_tokens(word_budget)}"
        self.repo_id = f"{hf_org}/{name_with_budget}" if hf_org else name_with_budget

        self.gen_kwargs = {
            "min_new_tokens": 0,
            "max_new_tokens": 64,
            "min_length": -1,
            "top_k": 0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }


        self.batch_logs: List[dict] = []
        self.generated_log: List[tuple[str, float]] = []


        timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        model_dir_name = f"{name_with_budget}__{timestamp}"
        self.base_out_dir = os.path.join(save_base_dir, model_dir_name)
        self.meta_dir = os.path.join(self.base_out_dir, "meta_data") if save_meta_dir is None else save_meta_dir

        logger.info(self.meta_dir)
        
        os.makedirs(self.meta_dir, exist_ok=True)

        self.wandb_enabled = wandb_project is not None
        if self.wandb_enabled:
            run_name = (hf_base_repo or config.model_name).replace("/", "-")
            wandb.init(
                project=wandb_project,
                name=run_name,
                tags=wandb_tags,
                config={"word_budget": word_budget, "gen_word_budget": gen_word_budget, **config.__dict__},
            )
            if watch_model:
                wandb.watch(self.model, log="all", log_freq=1_000)


    def set_generation_kwargs(self, **kwargs):
        self.gen_kwargs.update(kwargs)

    @staticmethod
    def _collate(batch):
        return {
            "input_ids": [torch.tensor(b["input_ids"]) for b in batch],
            "query": [b["query"] for b in batch],
        }


    def _push_to_hub(self, branch: str, msg: str):
        """Upload model+tokenizer to a branch using a single HfApi handle."""

        self.api.create_repo(repo_id=self.repo_id, exist_ok=True, repo_type="model")
        if branch != "main":
            self.api.create_branch(repo_id=self.repo_id, branch=branch, exist_ok=True)


        ckpt_dir = os.path.join(self.meta_dir, "_ckpt_upload")
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=ckpt_dir,
            revision=branch,
            commit_message=msg,
        )
        shutil.rmtree(ckpt_dir)
        logger.info("Pushed → %s:%s", self.repo_id, branch)


    def _push_checkpoint(self, words_used: int):
        tag = fmt_tokens(words_used)
        branch = f"ckpt_{tag}_words"
        self._push_to_hub(branch, f"Checkpoint at {tag} words")

    def _push_final(self):
        self._push_to_hub("main", "Final model push")


    def _dump_logs(self):
        if self.generated_log:
            with open(os.path.join(self.meta_dir, "generated.csv"), "w", newline="") as f:
                csv.writer(f).writerows(self.generated_log)
        if self.batch_logs:
            with open(os.path.join(self.meta_dir, "batch_stats.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, self.batch_logs[0].keys()); w.writeheader(); w.writerows(self.batch_logs)
        logger.info("Logs saved → %s", self.meta_dir)


    def run_training_loop(self, num_epochs: int = 1):
        logger.info(
            "Start training w/ budgets: prompt=%d, gen=%d (per epoch=%d)",
            self.word_budget,
            self.gen_word_budget,
            num_epochs,
        )

        global_step         = 0
        total_prompt_words  = 0
        next_ckpt           = schedule_next_ckpt(0)

        for epoch in range(num_epochs):
            prompt_used = 0
            gen_used    = 0
            logger.info("Epoch %d/%d …", epoch + 1, num_epochs)

            for batch_idx, batch in enumerate(
                tqdm(self.dataloader, desc=f"epoch {epoch+1}/{num_epochs}")
            ):
                try:

                    if prompt_used >= self.word_budget or gen_used >= self.gen_word_budget:
                        logger.info("Budget hit → epoch done")
                        break

                    queries      = batch["input_ids"]
                    query_words  = sum(len(q.split()) for q in batch["query"])
                    if prompt_used + query_words > self.word_budget:
                        continue

                    gens = self.generate(queries, **self.gen_kwargs,)

                    resp_only    = [g[len(q):] for g, q in zip(gens, queries)]
                    dec_resp     = [self.tokenizer.decode(r) for r in resp_only]
                    resp_words   = sum(len(r.split()) for r in dec_resp)
                    if gen_used + resp_words > self.gen_word_budget:
                        continue

                    pairs        = [ PromptCompletionPair(q, q + r) for q, r in zip(batch["query"], dec_resp)]

                    rewards_dict = self.reward_fn(pairs)
                    rewards      = rewards_dict["rewards"]
                    raw_outputs  = rewards_dict["raw_outputs"]
                    reward_words = rewards_dict["total_length"]

                    if prompt_used + query_words + reward_words > self.word_budget:
                        continue


                    stats = self.step(queries, resp_only, rewards)
                    self._log_batch( rewards, stats, prompt_used + query_words + reward_words, gen_used   + resp_words, global_step,)
                    global_step += 1


                    delta_prompt = query_words + reward_words
                    prompt_used += delta_prompt
                    gen_used    += resp_words
                    total_prompt_words += delta_prompt

                    self.generated_log.extend(
                        (r, rew) for r, rew in zip(dec_resp, raw_outputs)
                    )

                    if total_prompt_words >= next_ckpt:
                        branch = f"ckpt_{fmt_tokens(next_ckpt)}"
                        self._push_to_hub(branch, f"Checkpoint at {fmt_tokens(next_ckpt)} words")
                        next_ckpt = schedule_next_ckpt(total_prompt_words)


                except Exception as e:
                    logger.exception(
                        "Error in epoch %d batch %d (global_step=%d). Skipping batch. %s", epoch + 1, batch_idx, global_step, str(e),)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            logger.info("Epoch %d finished: prompt=%d, gen=%d",epoch + 1,prompt_used,gen_used,)

        self._push_final()
        self._dump_logs()



    def _log_batch(self, rewards, stats, prompt_words, gen_words, global_step):
        """Record batch metrics locally and in Weights & Biases (wandb)."""
        rw = [float(r) for r in rewards]
        record = {
            "avg_reward": statistics.mean(rw),
            "std_reward": statistics.stdev(rw) if len(rw) > 1 else 0.0,
            "kl":             stats.get("objective/kl", 0.0),
            "entropy":        stats.get("objective/entropy", 0.0),
            "policy_loss":    stats.get("ppo/loss/policy", 0.0),
            "value_loss":     stats.get("ppo/loss/value", 0.0),
            "total_loss":     stats.get("ppo/loss/total", 0.0),
            "mean_value_pred":stats.get("ppo/val/vpred", 0.0),
            "prompt_words":   prompt_words,
            "gen_words":      gen_words,
            "learning_rate":  stats.get("ppo/learning_rate", 0.0),
        }


        self.batch_logs.append({**record, "global_step": global_step})

        if self.wandb_enabled:
            wandb.log(record, step=global_step)
