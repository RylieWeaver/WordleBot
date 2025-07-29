# General

# Torch
import torch

def expand_var(x, dim, size):
    """
    Expand a tensor in a designated dimension while keeping
    the original shape in the other dimensions.
    """
    x = x.unsqueeze(dim)
    target_shape = [-1] * x.dim()
    target_shape[dim] = size
    return x.expand(*target_shape)
