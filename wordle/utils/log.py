# General

# Torch
import torch

# Wordle



def clear_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("No GPU available, cannot clear cache.")
