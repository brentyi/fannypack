import pytest

import numpy as np
import fannypack


@pytest.fixture
def iterator():
    raw = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
    iterator = fannypack.utils.SliceWrapper(raw)
    return iterator


def test_fixture(iterator):
    """Fixture sanity check.
    """
    assert type(iterator.data["a"]) == list
    iterator.convert_to_numpy()
    assert type(iterator.data["a"]) == np.ndarray


def test_shape(iterator):
    """Check `shape` property.
    """
    assert iterator.shape == (4, )
    iterator.convert_to_numpy()
    assert iterator.shape == (4, )

def test_read_slice(iterator):
    """Check that we can read slices of iterators.
    """
    assert iterator[::1]["a"] == [1, 2, 3, 4]
    assert iterator[::1]["b"] == [5, 6, 7, 8]

    assert iterator[::2]["a"] == [1, 3]
    assert iterator[::2]["b"] == [5, 7]

    assert iterator[:2:]["a"] == [1, 2]
    assert iterator[1:3:]["b"] == [6, 7]


def test_read_slice_numpy(iterator):
    """Check that we can read slices of our iterators (numpy).
    """
    iterator.convert_to_numpy()

    assert np.allclose(iterator[::1]["a"], [1, 2, 3, 4])
    assert np.allclose(iterator[::1]["b"], [5, 6, 7, 8])

    assert np.allclose(iterator[::2]["a"], [1, 3])
    assert np.allclose(iterator[::2]["b"], [5, 7])

    assert np.allclose(iterator[:2:]["a"], [1, 2])
    assert np.allclose(iterator[1:3:]["b"], [6, 7])


def test_write_slice(iterator):
    """Check that writing does nothing for iterators containing raw lists.
    """
    iterator[::1]["a"][0] = 1000
    test_read_slice(iterator)


def test_write_slice_numpy(iterator):
    """Check that we can write to slices for iterators containing numpy arrays.
    """
    iterator.convert_to_numpy()

    iterator[::1]["a"][::2] = 0

    assert np.allclose(iterator[::1]["a"], [0, 2, 0, 4])
    assert np.allclose(iterator[::1]["b"], [5, 6, 7, 8])

    assert np.allclose(iterator[::2]["a"], [0, 0])
    assert np.allclose(iterator[::2]["b"], [5, 7])

    assert np.allclose(iterator[:2:]["a"], [0, 2])
    assert np.allclose(iterator[1:3:]["b"], [6, 7])
