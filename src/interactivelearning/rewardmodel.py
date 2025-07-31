from src.interactivelearning.logger import logger
import torch
from abc import ABC, abstractmethod
from src.interactivelearning.teacher import Llama3Teacher

class RewardModel(ABC):
    @abstractmethod
    def __call__(self, texts):
        pass


class RandomRewardModel(RewardModel):
    def __init__(self, config):
        super().__init__()
        
    def __call__(self, prompt_completion_pairs):
        logger.debug(f"Generating random reward scores for {len(prompt_completion_pairs)} samples")

        random_scores = torch.randint(0, 10, (len(prompt_completion_pairs),)) 
        normalized_rewards = [torch.tensor(float(x) / 9, dtype=torch.float32) for x in random_scores]

        raw_outputs = [str(x.item()) for x in random_scores]
        total_length = sum(len(s) for s in raw_outputs)
        return {
            "rewards": normalized_rewards,
            "raw_outputs": raw_outputs,
            "total_length": total_length,
        }


class Llama3RwardModel(RewardModel):
    def __init__(self, config):
        super().__init__()
        self.teacher_model = Llama3Teacher(config=config)

    def __call__(self, prompt_completion_pairs):
        rewards, raw_outputs, total_length = self.teacher_model.evaluate_batch(prompt_completion_pairs)
        
        reward_tensors = [torch.tensor(float(r), dtype=torch.float32) for r in rewards]
        return {
            "rewards": reward_tensors,
            "raw_outputs": raw_outputs,
            "total_length": total_length,
        }
