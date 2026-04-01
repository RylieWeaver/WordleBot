# General

# Torch
import torch

# Wordle



def move_to(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to(v, device) for v in batch)
    else:
        return batch
