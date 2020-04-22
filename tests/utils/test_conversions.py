import fannypack
import numpy as np
import pytest
import torch


@pytest.fixture
def nested_conversion():
    X = {"data": ([np.zeros((10, 1, 5, 1, 10))], np.eye(3))}
    X_torch = fannypack.utils.to_torch(X)
    X_numpy = fannypack.utils.to_numpy(X_torch)
    return X, X_torch, X_numpy


def test_conversion_shapes(nested_conversion):
    """Check numpy <=> torch conversions.
    """
    X, X_torch, X_numpy = nested_conversion
    assert X_torch["data"][0][0].shape == X_numpy["data"][0][0].shape
    assert X_torch["data"][1].shape == X_numpy["data"][1].shape


def test_conversion_types(nested_conversion):
    """Check numpy <=> torch conversions.
    """
    X, X_torch, X_numpy = nested_conversion
    assert type(X_numpy["data"][0][0]) == np.ndarray
    assert type(X_torch["data"][0][0]) == torch.Tensor
    assert type(X_numpy["data"][1]) == np.ndarray
    assert type(X_torch["data"][1]) == torch.Tensor


def test_conversion_values(nested_conversion):
    """Check numpy <=> torch conversions.
    """
    X, X_torch, X_numpy = nested_conversion
    assert np.allclose(X["data"][0][0], X_numpy["data"][0][0])
    assert np.allclose(X["data"][1], X_numpy["data"][1])


def test_conversion_failure():
    """Sanity check for invalid input.
    """
    with pytest.raises(AssertionError):
        fannypack.utils.to_torch(None)


def test_to_device():
    """Sanity check for to_device.
    """
    X = {"data": [np.zeros((10, 1, 5, 1, 10))]}
    X_torch = fannypack.utils.to_torch(X)

    X_new = fannypack.utils.to_device(X_torch, torch.device("cpu"), detach=True)
    assert X_torch["data"][0].shape == X_new["data"][0].shape
    assert type(X_new["data"][0]) == torch.Tensor
