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


def op_except(tensor, except_dims, type='sum', keepdim=False):
    """
    Perform an operation (e.g. sum or mean) over all dimensions 
    except the given one.
    """
    ndim = tensor.dim()
    # Make as list
    if isinstance(except_dims, int):
        except_dims = [except_dims]
    except_dims = [(d % ndim) for d in except_dims]  # Handle negatives
    op_dims = [d for d in range(ndim) if d not in except_dims]  # Range except
    # Perform operation
    if type == 'sum':
        return tensor.sum(dim=op_dims, keepdim=keepdim)
    elif type == 'mean':
        return tensor.mean(dim=op_dims, keepdim=keepdim)
    else:
        raise NotImplementedError(f'Operation {type} not implemented in op_except')
