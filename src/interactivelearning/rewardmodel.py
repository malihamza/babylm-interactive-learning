from src.interactivelearning.logger import logger
import torch
from abc import ABC, abstractmethod
from src.interactivelearning.teacher import Llama3Teacher

class RewardModel(ABC):
    @abstractmethod
    def __call__(self, texts):
        pass


class RandomRewardModel(RewardModel):
    def __call__(self, texts):
        logger.debug(f"Generating reward scores for {len(texts)} samples")
        return [torch.tensor(float(x), dtype=torch.float32) for x in torch.randint(1, 10, (len(texts),))]

class Llama3RwardModel(RewardModel):
    def __init__(self, config):
        super().__init__()
        self.teacher_model = Llama3Teacher(config=config)

    def __call__(self, prompt_completion_pairs):

        scores = self.teacher_model.evaluate_batch(prompt_completion_pairs)
        return [torch.tensor(float(score),dtype=torch.float32) for score in scores]