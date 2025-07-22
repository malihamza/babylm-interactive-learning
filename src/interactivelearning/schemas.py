from dataclasses import dataclass

@dataclass
class PromptCompletionPair:
    prompt: str
    completion: str