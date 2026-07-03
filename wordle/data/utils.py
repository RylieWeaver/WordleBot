# General

# Torch
import torch

# Wordle



def move_to(batch, device, non_blocking=False):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    elif isinstance(batch, dict):
        return {k: move_to(v, device, non_blocking=non_blocking) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to(v, device, non_blocking=non_blocking) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to(v, device, non_blocking=non_blocking) for v in batch)
    else:
        return batch
