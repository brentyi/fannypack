import sys
import numpy as np
from IPython.display import clear_output


def progress_bar(progress):
    """
    Simple progress bar for Jupyter notebooks.
    TODO: switch to tqdm?
    """
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


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
