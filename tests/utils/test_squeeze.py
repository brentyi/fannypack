import numpy as np
import pytest

import fannypack


def test_squeeze_simple():
    X = np.zeros((10, 1, 5, 1, 10))
    X = fannypack.utils.squeeze(X)
    assert X.shape == (10, 5, 10)


def test_squeeze_axis():
    X = np.zeros((10, 1, 5, 1, 10))
    X = fannypack.utils.squeeze(X, axis=1)
    assert X.shape == (10, 5, 1, 10)


def test_squeeze_bad_axis():
    X = np.zeros((10, 1, 5, 1, 10))
    with pytest.raises(AssertionError):
        X = fannypack.utils.squeeze(X, axis=0)
