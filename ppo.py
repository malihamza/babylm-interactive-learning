import yaml
from src.interactivelearning.rewardmodel import RandomRewardModel, Llama3RewardModel
from src.interactivelearning.ppotrainer import CustomPPOTrainer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from src.interactivelearning.datasetbuilder import IMDBDatasetBuilder, DatasetCombiner, TinyStoriesDatasetBuilder, WritingPromptsDatasetBuilder, DeterministicPromptDatasetBuilder
from src.interactivelearning.utils import load_yaml_config
from src.interactivelearning.ppoconfig import CustomPPOConfig

import random
import numpy as np
import torch


def main(ppo_cfg, teacher_cfg):

    seed = teacher_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ppo_config = CustomPPOConfig(
        model_name=ppo_cfg["model_name"],
        revision_name=ppo_cfg.get("revision_name",None),
        learning_rate=float(ppo_cfg.get("learning_rate", 1e-5)),
        log_with=ppo_cfg.get("log_with", None),
        mini_batch_size=ppo_cfg.get("batch_size"),
        batch_size=ppo_cfg.get("batch_size"),
        output_min_length=ppo_cfg.get("output_min_length", 64),
        output_max_length=ppo_cfg.get("output_max_length", 128),
    )

    token_limit = ppo_cfg.get("token_limit")
    data_path = ppo_cfg.get("data_path")

    query_min_length = ppo_cfg.get("query_min_length")
    query_max_length = ppo_cfg.get("query_max_length")


    # Dataset builders
    # builder1 = WritingPromptsDatasetBuilder(ppo_config, 
    #                                         cache_dir=data_path,
    #                                     min_len=query_min_length, 
    #                                     max_len=query_max_length)


    #builder1 = TinyStoriesDatasetBuilder(ppo_config,
    #                                        cache_dir=data_path,
    #                                    min_len=query_min_length,
    #                                    max_len=query_max_length)

    builder1 = DeterministicPromptDatasetBuilder(
        ppo_config,
        prompt="Let me tell you a long, magical tale. Once upon a time, in a faraway land,",
        cache_dir=data_path
    )

    # Combine datasets
    combined_dataset = DatasetCombiner([builder1])
    combined_dataset.set_token_limit(token_limit=token_limit)
    combined_dataset = combined_dataset.load()


    # Reward model
    reward_model = Llama3RewardModel(config=teacher_cfg)
    

    model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg["model_name"], revision=ppo_cfg["revision_name"])
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_cfg["model_name"], revision=ppo_cfg["revision_name"])
    tokenizer = builder1.tokenizer

    trainer = CustomPPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=combined_dataset,
        reward_fn=reward_model,
        word_budget=token_limit,
        hf_org=ppo_cfg.get("hf_org", "llm-slice"),
        save_base_dir=ppo_cfg.get("save_base_dir", "saved_models"),
        wandb_project="ppo-rlhf",   
    )

    # Generation kwargs from config
    trainer.set_generation_kwargs(**ppo_cfg.get("generation_kwargs", {}))

    # Run training loop
    trainer.run_training_loop(
        num_epochs=ppo_cfg.get("num_epochs", 1),
        
    ) 


if __name__ == "__main__":
    
    ppo_config_path = "config/ppo.yaml"
    teacher_config_path = "config/teacher.yaml"
    ppo_cfg = load_yaml_config(ppo_config_path)
    teacher_cfg = load_yaml_config(teacher_config_path)
    main(ppo_cfg, teacher_cfg)
