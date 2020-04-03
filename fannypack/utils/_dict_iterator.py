import numpy as np


class DictIterator:
    """Wrapper for manipulating the contents of dictionaries that contain
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

    def convert_to_numpy(self):
        for key, value in self._data.items():
            self._data[key] = np.asarray(value)

    @property
    def data(self):
        return self._data
