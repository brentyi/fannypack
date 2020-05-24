import dataclasses
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union, cast, overload

import numpy as np
import torch

# General Container template; used by all conversion functions
Container = Union[List, Tuple, Dict, Any]


# Private conversion helper: recursively calls a conversion function on all inputs
# within any nested set of containers
def _convert(
    x: Union[Container, torch.Tensor, np.ndarray], convert: Callable
) -> Union[Container, torch.Tensor, np.ndarray]:
    if type(x) in (torch.Tensor, np.ndarray):
        # Convert plain arrays
        output = convert(x)
    elif type(x) == dict:
        # Convert dictionaries of values
        x = cast(dict, x)
        output = {}
        for key, value in x.items():
            output[key] = _convert(value, convert)
    elif type(x) == list:
        # Convert lists of values
        x = cast(list, x)
        output = [_convert(value, convert) for value in x]
    elif type(x) == tuple:
        # Convert tuples of values
        x = cast(tuple, x)
        output = tuple(_convert(value, convert) for value in x)
    elif dataclasses.is_dataclass(x):
        # Convert dataclass of values
        changes = {}
        for field in dataclasses.fields(x):
            value = getattr(x, field.name)
            try:
                changes[field.name] = _convert(value, convert)
            except AssertionError as e:
                pass

        output = dataclasses.replace(x, **changes)
    else:
        assert False, f"Unsupported datatype {type(x)}!"
    return output


def to_device(
    x: Union[Container, torch.Tensor], device: torch.device, detach: bool = False
) -> Union[Container, torch.Tensor]:
    """Move a torch tensor, list of tensors, dict, or dataclass of tensors to a
    different device.

    Args:
        x: (torch.Tensor, list, tuple, dict, or dataclass) Tensor or container of
            tensors to move.
        device (torch.device): Target device.
        detach (bool, optional): If set, detaches tensors after moving. Defaults to
            False.

    Returns:
        torch.Tensor, list, tuple, dict, or dataclass: Output, type will mirror input.
    """

    def convert(x):
        if detach:
            x = x.detach()
        return x.to(device)

    return _convert(x, convert)


def to_torch(
    x: Union[Container, np.ndarray],
    device: str = "cpu",
    convert_doubles_to_floats: bool = True,
) -> Union[Container, torch.Tensor]:
    """Converts a numpy array, list of numpy arrays, dict, or dataclass of numpy arrays
    for use in PyTorch.

    Args:
        x: (np.ndarray, list, tuple, dict, or dataclass) Array or container of arrays to
            convert to torch tensors.
        device (torch.device, optional): Torch device to create tensors on. Defaults to
            `"cpu"`.
        convert_doubles_to_floats (bool, optional): If set, converts 64-bit floats to
            32-bit. Defaults to True.

    Returns:
        torch.Tensor, list, tuple, dict, or dataclass: Output, type will mirror input.
    """

    def convert(x: np.ndarray):
        output = torch.from_numpy(x)
        if x.dtype == np.float64 and convert_doubles_to_floats:
            output = output.float()
        output = output.to(device)
        return output

    return _convert(x, convert)


def to_numpy(x: Union[torch.Tensor, Container]) -> Union[np.ndarray, Container]:
    """Converts a tensor, list of tensors, dict, or dataclass of tensors for use in
    Numpy.

    Args:
        x: (torch.Tensor, list, tuple, dict, or dataclass) Tensor or container of
            tensors to convert to numpy.

    Returns:
        np.ndarray, list, tuple, dict, or dataclass: Output, type will mirror input.
    """

    def convert(x: torch.Tensor):
        output = x.detach().cpu().numpy()
        return output

    return _convert(x, convert)
