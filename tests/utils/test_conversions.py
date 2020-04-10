import pytest

import torch
import numpy as np
import fannypack


def test_conversion_nested():
    """Check numpy <=> torch conversions.
    """
    X = {"data": [np.zeros((10, 1, 5, 1, 10))]}
    X_torch = fannypack.utils.to_torch(X)
    X_numpy = fannypack.utils.to_numpy(X_torch)

    assert X_torch["data"][0].shape == X_numpy["data"][0].shape
    assert type(X_numpy["data"][0]) == np.ndarray
    assert type(X_torch["data"][0]) == torch.Tensor


def test_to_device():
    """Sanity check for to_device.
    """
    X = {"data": [np.zeros((10, 1, 5, 1, 10))]}
    X_torch = fannypack.utils.to_torch(X)

    X_new = fannypack.utils.to_device(
        X_torch, torch.device("cpu"), detach=True
    )
    assert X_torch["data"][0].shape == X_new["data"][0].shape
    assert type(X_new["data"][0]) == torch.Tensor
