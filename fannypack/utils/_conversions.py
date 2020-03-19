"""
.. module:: fannypack

blah blah
"""

import numpy as np
import torch


def to_device(x, device, detach=True):
    """Copies a tensor, list of tensors, or dict of tensors to a different
    device.
    """
    if type(x) == torch.Tensor:
        # Convert plain arrays
        if detach:
            x = x.detach()
        output = x.to(device)
    elif type(x) == dict:
        # Convert dictionaries of values
        output = {}
        for key, value in x.items():
            output[key] = to_device(value, device, detach)
    elif type(x) in (list, tuple):
        # Convert lists of values
        output = []
        for value in x:
            output.append(to_device(value, device, detach))
    else:
        assert False, f"Invalid datatype {type(x)}!"
    return output


def to_torch(x, device="cpu"):
    """Converts a numpy array, list of numpy arrays, or dict of numpy arrays
    for use in PyTorch.
    """
    if type(x) == np.ndarray:
        # Convert plain arrays
        output = torch.from_numpy(x).float().to(device)
    elif type(x) == dict:
        # Convert dictionaries of values
        output = {}
        for key, value in x.items():
            output[key] = to_torch(value, device)
    elif type(x) in (list, tuple):
        # Convert lists of values
        output = []
        for value in x:
            output.append(to_torch(value, device))
    else:
        assert False, f"Invalid datatype {type(x)}!"

    return output


def to_numpy(x):
    """Converts a tensor, list of tensors, or dict of tensors for use in Numpy.
    """
    if type(x) == torch.Tensor:
        # Convert plain tensors
        output = x.detach().cpu().numpy()
    elif type(x) == dict:
        # Convert dictionaries of values
        output = {}
        for key, value in x.items():
            output[key] = to_numpy(value)
    elif type(x) in (list, tuple):
        # Convert lists of values
        output = []
        for value in x:
            output.append(to_numpy(value))
    else:
        assert False, f"Invalid datatype {type(x)}!"

    return output
