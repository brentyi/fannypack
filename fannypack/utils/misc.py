import sys
import numpy as np
import torch


class DictIterator:
    """
    Wrapper for manipulating the contents of dictionaries that contain
    same-length iterables

    Nice for slicing/indexing into/appending to groups of Python lists, numpy
    arrays, torch tensors, etc
    """

    def __init__(self, data):
        assert type(data) == dict

        # Every value in the dict should have the same length
        self._length = None
        for value in data.values():
            length = len(value)
            if self._length is None:
                self._length = length
            else:
                assert length == self._length

        self._data = data

    def __getitem__(self, index):
        # Check that the index is sane
        assert type(index) in (int, slice, tuple)
        if type(index) == int and index >= len(self):
            # For use as a standard Python iterator
            raise IndexError

        output = {}
        for key, value in self._data.items():
            output[key] = value[index]
        return output

    def __len__(self):
        return self._length

    def append(self, other):
        for key, value in other.items():
            if key in self._data.keys():
                self._data[key].append(value)
            else:
                self._data[key] = [value]

    def extend(self, other):
        for key, value in other.items():
            if key in self._data.keys():
                self._data[key].extend(value)
            else:
                self._data[key] = [value]

    def convert_to_numpy(self):
        for key, value in self._data.items():
            self._data[key] = np.asarray(value)


def to_device(x, device, detach=True):
    """
    Copies a tensor, list of tensors, or dictionary of tensors to a different device.
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
        assert False, "Invalid datatype {}!".format(type(x))
    return output


def to_torch(x, device='cpu'):
    """
    Converts a numpy array, list of np arrays, or dictionary of np arrays for use in PyTorch.
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
        assert False, "Invalid datatype {}!".format(type(x))

    return output


def to_numpy(x):
    """
    Converts a tensor, list of tensors, or dictionary of tensors for use in Numpy.
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
        assert False, "Invalid datatype {}!".format(type(x))

    return output
