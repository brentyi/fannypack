from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

import fannypack


@dataclass
class Dataclass:
    value: Any
    string: str = "True"
    boolean: bool = False


@pytest.fixture
def nested_conversion():
    X = Dataclass(value={"data": ([np.zeros((10, 1, 5, 1, 10))], np.eye(3))})
    X_torch = fannypack.utils.to_torch(X)
    X_numpy = fannypack.utils.to_numpy(X_torch)
    return X, X_torch, X_numpy


def test_dataclass_checks(nested_conversion):
    """Check for dataclass field issues.
    """
    X, X_torch, X_numpy = nested_conversion

    assert X is not X_torch
    assert X is not X_numpy

    assert X.string == X_torch.string
    assert X.string == X_numpy.string
    assert X.boolean == X_torch.boolean
    assert X.boolean == X_numpy.boolean


def test_conversion_shapes(nested_conversion):
    """Check numpy <=> torch conversions.
    """
    X, X_torch, X_numpy = nested_conversion
    assert X_torch.value["data"][0][0].shape == X_numpy.value["data"][0][0].shape
    assert X_torch.value["data"][1].shape == X_numpy.value["data"][1].shape


def test_conversion_types(nested_conversion):
    """Check numpy <=> torch conversions.
    """
    X, X_torch, X_numpy = nested_conversion
    assert type(X_numpy.value["data"][0][0]) == np.ndarray
    assert type(X_torch.value["data"][0][0]) == torch.Tensor
    assert type(X_numpy.value["data"][1]) == np.ndarray
    assert type(X_torch.value["data"][1]) == torch.Tensor


def test_conversion_values(nested_conversion):
    """Check numpy <=> torch conversions.
    """
    X, X_torch, X_numpy = nested_conversion
    assert np.allclose(X.value["data"][0][0], X_numpy.value["data"][0][0])
    assert np.allclose(X.value["data"][1], X_numpy.value["data"][1])


def test_conversion_failure():
    """Smoke test for an unsupported input type.
    """
    with pytest.raises(AssertionError):
        fannypack.utils.to_torch(None)


def test_to_device():
    """Smoke test for to_device.
    """
    X = {"data": [np.zeros((10, 1, 5, 1, 10))]}
    X_torch = fannypack.utils.to_torch(X)

    X_new = fannypack.utils.to_device(X_torch, torch.device("cpu"), detach=True)
    assert X_torch["data"][0].shape == X_new["data"][0].shape
    assert type(X_new["data"][0]) == torch.Tensor
