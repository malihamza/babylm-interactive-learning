import matplotlib.pyplot as plt
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
from src.interactivelearning.logger import logger
from abc import ABC, abstractmethod
from jinja2 import Template
from src.interactivelearning.schemas import PromptCompletionPair
from typing import List, Tuple, Dict
import re



class Teacher(ABC):
    def __init__(self, config: dict):
        """
        Initializes the teacher model used to evaluate student completions.
        """
        self.model_name = config["model_name_or_path"]
        self.prompt_template = self.load_prompt(config["prompt_template_path"])
        self.sampling_params = SamplingParams(temperature=config["temperature"], max_tokens=config["max_tokens"])
        self.min_score = config.get("min_score", 1)
        self.max_score = config.get("max_score", 9)
        logger.info("Initializing teacher model: %s", self.model_name)
        self.load_model()

    def load_model(self):
        """
        Load the teacher model for scoring.
        Override this method to use your preferred backend.
        """
        raise NotImplementedError("Implement model loading logic here")

    def parse_score(self, text: str) -> int:
        """
        Extracts the first integer in the model's output using regex.
        Enforces the score to be within the configured range.
        """
        match = re.search(r"\b(\d+)\b", text)
        if match:
            score = int(match.group(1))
            bounded_score = max(self.min_score, min(score, self.max_score))
            logger.debug("Parsed score: %d (bounded to %d)", score, bounded_score)
            return bounded_score
        else:
            logger.error("No valid score found in model output: '%s'", text)
            raise ValueError(f"No valid score found in model output: '{text}'")

    @staticmethod
    def load_prompt(prompt_template_path: str) -> Template:
        try:
            with open(prompt_template_path, "r") as file:
                logger.info("Loaded prompt template from %s", prompt_template_path)
                return Template(file.read())
        except FileNotFoundError:
            logger.error("Prompt template file not found: %s", prompt_template_path)
            raise RuntimeError(f"Prompt template file not found: {prompt_template_path}")
        except Exception as e:
            logger.error("Failed to load prompt template: %s", e)
            raise RuntimeError(f"Failed to load prompt template: {e}")       

    def evaluate(self, prompt: str, completion: str) -> int:
        """
        Evaluate the given prompt-completion pair using the teacher model.
        """
        raise NotImplementedError("Implement scoring logic (1 to 9 scale)")

    def prepare_llm_input(self, sample: PromptCompletionPair) -> List[Dict]:
        """
        Fills the prompt template with the given prompt-completion pair.
        """
        logger.debug("Preparing input for LLM using sample: %s", sample)
        user_prompt = self.prompt_template.render(
            context=sample.prompt,
            continuation=sample.completion,
            min_score=self.min_score,
            max_score=self.max_score
        )

        return [{"role": "user", "content": user_prompt}]

    def evaluate_batch(self, batch: List[PromptCompletionPair]) -> List[int]:
        logger.debug("Evaluating batch of %d samples", len(batch))

        messages_batch = [self.prepare_llm_input(sample) for sample in batch]

        results = self.model.chat(messages_batch, self.sampling_params, use_tqdm=False)
        outputs = [r.outputs[0].text.strip() for r in results]
        scores = [self.parse_score(output) for output in outputs]

        logger.debug("Outputs: %s", outputs)
        logger.debug("Scores: %s", scores)

        return scores


class Llama3Teacher(Teacher):
    def __init__(self, config: Dict):
        super().__init__(config=config)

    def load_model(self):
        try:
            self.model = LLM(
                model=self.model_name,
                dtype="bfloat16",
                enable_lora=False,
                gpu_memory_utilization=0.3,                
                max_seq_len_to_capture = 1024,
                max_model_len = 1024,
                
            )
            logger.info("Llama LLM model initialized ✓")
        except Exception as e:
            logger.error("✗ Error initializing model: %s", e)
            raise
