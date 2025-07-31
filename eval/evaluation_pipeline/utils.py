from __future__ import annotations
from typing import Any, TYPE_CHECKING
from transformers.modeling_outputs import ModelOutput

if TYPE_CHECKING:
    import torch


def get_logits(outputs: Any) -> torch.Tensor:
    """This helper function, checks the type passed outputs,
    and extracts the logits from them.

    Args:
        outputs(Any): The outputs of a HuggingFace model.

    Returns:
        torch.Tensor: The logits of the model.
    """
    if type(outputs) is tuple:
        encoding: torch.Tensor = outputs[0]
    elif isinstance(outputs, ModelOutput):
        if hasattr(outputs, "logits"):
            encoding = outputs.logits
        elif hasattr(outputs, "last_hidden_state"):
            encoding = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states"):
            encoding = outputs.hidden_states[-1]
        else:
            print("Unknown name for output of the model!")
            exit()
    else:
        print(f"Add support for output type: {type(outputs)}!")
        exit()

    return encoding
