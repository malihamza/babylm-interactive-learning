from vllm import LLM, SamplingParams
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
        self.sampling_params = SamplingParams(temperature=config["temperature"], max_tokens=config["max_tokens"], seed=config["seed"])
        self.min_score = config.get("min_score", 0)
        self.max_score = config.get("max_score", 3)
        self.default_score = config.get("default_score", 0.5)
        logger.info("Initializing teacher model: %s", self.model_name)
        self.load_model()

    def load_model(self):
        """
        Load the teacher model for scoring.
        Override this method to use your preferred backend.
        """
        raise NotImplementedError("Implement model loading logic here")

    def parse_reward(self, text: str) -> float:
        """
        Parses the output string for all integers in [0,3], sums, and normalizes to [0,1].
        """
        # Extract all numbers (as strings)
        numbers = re.findall(r"\b\d+\b", text)
        if not numbers:
            logger.warning("No valid scores found in output: '%s', returning 0.5", text)
            return self.default_score  
        # Clamp to [0,3] and convert to int
        #logger.info(f"Raw teacher output: {text}")
        scores = [max(0, min(int(n), self.max_score)) for n in numbers]
        weights = [1, 1, 1] # weight for each of the three categories
        total = sum(w * s for w, s in zip(weights, scores))
        total_max_score = sum(weights) * self.max_score # self.max_score is per category
        normalized = total / total_max_score
        return normalized

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
            story_prompt=sample.prompt,
            student_completion=sample.completion,
        )

        return [{"role": "user", "content": user_prompt}]

    def evaluate_batch(self, batch: List[PromptCompletionPair]):
        logger.debug("Evaluating batch of %d samples", len(batch))
        messages_batch = [self.prepare_llm_input(sample) for sample in batch]
        
        try:
            results = self.model.chat(messages_batch, self.sampling_params, use_tqdm=False)
        except Exception as e:
            logger.error("LLM chat failed for batch of %d samples: %s", len(batch), str(e))
            logger.debug("Returning default values for the entire batch")

            return (
                [self.default_score] * len(batch),  
                [""] * len(batch),              
                0                               
            )

        raw_outputs = []
        rewards = []
        total_length = 0

        for r in results:
            try:
                text = r.outputs[0].text.strip()
            except Exception as e:
                logger.warning("Failed to extract output text: %s", str(e))
                text = ""
            raw_outputs.append(text)
            try:
                reward = self.parse_reward(text)
            except Exception as e:
                logger.warning("Score parsing failed for output '%s': %s", text, str(e))
                reward = self.default_score
            rewards.append(reward)
            total_length += len(text)

        logger.debug("Raw outputs: %s", raw_outputs)
        logger.debug("Rewards: %s", rewards)
        logger.debug("Total char length: %s", total_length)

        return rewards, raw_outputs, total_length



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
