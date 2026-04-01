# General
from typing import Literal, Union

# Torch
import torch

# Wordle



DeviceType = Literal["auto", "cpu", "cuda", "mps"]
def resolve_device(device: DeviceType | str) -> torch.device:
    """Pick the best available torch.device given a preference string."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)

def get_device(object) -> torch.device:
    """Get the device of a general object."""
    if isinstance(object, torch.nn.Module):
        return next(object.parameters()).device
    if isinstance(object, torch.Tensor):
        return object.device
    raise ValueError(f"Cannot get device of object of type {type(object)}")
