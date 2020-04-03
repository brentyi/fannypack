import numpy as np
import torch

_mutable_iterables = (list, np.ndarray, torch.tensor)
_valid_iterables = _mutable_iterables + (tuple,)


class SliceWrapper:
    """Wrapper for manipulating the contents of dictionaries that contain
    same-length iterables.

    Nice for slicing/indexing into/appending to groups of Python lists, numpy
    arrays, torch tensors, etc.
    """

    def __init__(self, data):
        self._data = data
        self._type = type(data)

        # Sanity checks
        if type(data) == dict:
            # Every value in the dict should have the same length
            length = None
            for value in data.values():
                assert type(value) in _valid_iterables
                l = len(value)
                if length is None:
                    length = l
                else:
                    assert length == l
        else:
            # Non-dictionary inputs
            assert type(data) in _valid_iterables, "Invalid datatype!"

    def __getitem__(self, index):
        if self._type == dict:
            # Check that the index is sane
            assert type(index) in (int, slice, tuple)
            if type(index) == int and index >= len(self):
                # For use as a standard Python iterator
                raise IndexError

            output = {}
            for key, value in self._data.items():
                output[key] = value[index]
            return output
        elif self._type in _valid_iterables:
            return self._data[index]
        else:
            assert False, "Invalid operation!"

    def __len__(self):
        if self._type == dict:
            # Compute length of first value in dictionary
            return len(next(self._data.values()))
        else:
            # Compute length of valid iterable
            return len(self._data)

    def append(self, other):
        if self._type == dict:
            assert type(other) == dict
            for key, value in other.items():
                if key in self._data.keys():
                    self._data[key].append(value)
                else:
                    self._data[key] = [value]
        elif self._type in _mutable_iterables:
            self._data.append(other)
        else:
            assert False, "Invalid operation!"

    def extend(self, other):
        if self._type == dict:
            assert type(other) == dict
            for key, value in other.items():
                if key in self._data.keys():
                    assert type(self._data[key]) == type(value)

                    # Handle numpy arrays (inefficient)
                    if type(value) == np.ndarray:
                        self._data[key] = np.concatenate(
                            (self._data[key], value), axis=0
                        )

                    # Handle standard Python lists
                    else:
                        self._data[key].extend(value)
                else:
                    self._data[key] = value
        elif self._type in _mutable_iterables:
            self._data.extend(other)
        else:
            assert False, "Invalid operation!"

    def convert_to_numpy(self):
        if self._type == dict:
            # Convert elements in dictionary to numpy
            for key, value in self._data.items():
                self._data[key] = np.asarray(value)
        else:
            # Convert contents (list, tuple, etc) to numpy
            self._data = np.asarray(self._data)

    @property
    def shape(self):
        if self._type == dict:
            output = None
            for value in self._data.values():
                shape = self._shape_helper(value)
                if output == None:
                    output = shape
                else:
                    for i in range(min(len(output), len(shape))):
                        if output[i] != shape[i]:
                            output = output[:i]
                            break
            return tuple(output)
        else:
            return self._shape_helper(self._data)

    @staticmethod
    def _shape_helper(data):
        if type(data) in (torch.tensor, np.ndarray):
            # Return full shape
            return data.shape
        elif type(data) in (list, tuple):
            # Return 1D shape
            return (len(data),)
        else:
            assert False, "Invalid operation!"

    @property
    def data(self):
        return self._data
