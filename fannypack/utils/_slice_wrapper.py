# TODO: this class has grown somewhat organically and is pretty messy; could use a more
# intentional rewrite

from typing import Any, Dict, Iterable, List, Tuple, Union, cast

import numpy as np
import torch

_mutable_iterables = (list, np.ndarray, torch.Tensor)
_valid_iterables = _mutable_iterables + (tuple,)


class SliceWrapper:
    """Wrapper for manipulating the contents of dictionaries that contain
    same-length iterables.

    Nice for slicing/indexing into/appending to groups of Python lists, numpy
    arrays, torch tensors, etc.
    """

    def __init__(
        self,
        data: Union[
            np.ndarray,
            torch.Tensor,
            Iterable,
            Dict[Any, Union[np.ndarray, torch.Tensor, Iterable]],
        ],
    ) -> None:
        self._data = data
        self._type: type = type(data)

        # Sanity checks
        if type(data) == dict:
            # Every value in the dict should have the same length
            length = None
            for value in cast(Dict, data).values():
                assert type(value) in _valid_iterables
                l = len(value)
                if length is None:
                    length = l
                else:
                    assert length == l
        else:
            # Non-dictionary inputs
            assert type(data) in _valid_iterables, "Invalid datatype!"

    def __getitem__(self, index: int) -> Union[np.ndarray, torch.Tensor, Iterable]:
        if self._type == dict:
            # Check that the index is sane
            assert type(index) in (int, slice, tuple)
            if type(index) == int and cast(int, index) >= len(self):
                # For use as a standard Python iterator
                raise IndexError

            output = {}
            for key, value in cast(Dict, self._data).items():
                output[key] = value[index]
            return output
        elif self._type in _valid_iterables:
            return cast(Dict, self._data)[index]
        else:
            assert False, "Invalid operation!"

    def __len__(self) -> int:
        return self.shape[0]

    def append(self, other: Any) -> None:
        if self._type == dict:
            self._data = cast(dict, self._data)
            assert type(other) == dict
            for key, value in other.items():
                if key in self._data.keys():
                    # TODO: add support for torch tensors
                    # Handle numpy arrays (inefficient)
                    if type(self._data[key]) == np.ndarray:
                        self._data[key] = np.append(self._data[key], value)
                    # Handle standard Python lists
                    elif type(self._data[key]) == list:
                        cast(Dict, self._data)[key].append(value)
                else:
                    self._data[key] = [value]
        elif self._type == list:
            cast(List, self._data).append(other)
        else:
            assert False, "Invalid operation!"

    def extend(
        self,
        other: Union[
            np.ndarray,
            torch.Tensor,
            Iterable,
            Dict[Any, Union[np.ndarray, torch.Tensor, Iterable]],
        ],
    ):
        if self._type == dict:
            assert type(other) == dict
            for key, value in cast(Dict, other).items():
                if key in cast(Dict, self._data).keys():
                    # TODO: add support for torch tensors
                    # Handle numpy arrays (inefficient)
                    if type(cast(Dict, self._data)[key]) == np.ndarray:
                        cast(Dict, self._data)[key] = np.concatenate(
                            (cast(Dict, self._data)[key], value), axis=0
                        )
                    # Handle standard Python lists
                    else:
                        cast(Dict, self._data)[key].extend(value)
                else:
                    cast(Dict, self._data)[key] = value
        elif self._type == list:
            cast(List, self._data).extend(other)
        else:
            assert False, "Unsupported operation!"

    def convert_to_numpy(self) -> None:
        if self._type == dict:
            # Convert elements in dictionary to numpy
            for key, value in cast(Dict, self._data).items():
                cast(Dict, self._data)[key] = np.asarray(value)
        else:
            # Convert contents (list, tuple, etc) to numpy
            self._data = np.asarray(self._data)

    @property
    def shape(self) -> Tuple:
        if self._type == dict:
            output: Tuple
            first = True
            for value in cast(Dict, self._data).values():
                shape = self._shape_helper(value)
                if first:
                    output = shape
                    first = False
                else:
                    for i in range(min(len(output), len(shape))):
                        if output[i] != shape[i]:
                            output = output[:i]
                            break
            return tuple(output)
        else:
            return self._shape_helper(self._data)

    @staticmethod
    def _shape_helper(data) -> Tuple:
        if type(data) in (torch.Tensor, np.ndarray):
            # Return full shape
            return data.shape
        elif type(data) in (list, tuple):
            # Return 1D shape
            return (len(data),)
        else:
            assert False, "Invalid operation!"

    @property
    def data(
        self,
    ) -> Union[
        np.ndarray,
        torch.Tensor,
        Iterable,
        Dict[Any, Union[np.ndarray, torch.Tensor, Iterable]],
    ]:
        return self._data
