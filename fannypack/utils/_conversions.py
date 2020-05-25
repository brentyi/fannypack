import dataclasses
import functools
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union, cast, overload

import numpy as np
import torch

# Type aliases, variables
# > We'd ideally be able to use a TypeVar for Container, but this causes an overlapping
#   function signatures error in mypy
Container = Any
InputType = TypeVar("InputType", np.ndarray, torch.Tensor)
OutputType = TypeVar("OutputType", np.ndarray, torch.Tensor)


@overload
def _convert_recursive(
    x: InputType, convert: Callable[[InputType], OutputType], input_type: type,
) -> OutputType:
    pass


@overload
def _convert_recursive(
    x: Container, convert: Callable[[InputType], OutputType], input_type: type,
) -> Container:
    pass


def _convert_recursive(x, convert, input_type):
    """Private conversion helper. Recursively calls a conversion function on inputs
    within a nested set of containers.
    """

    # Conversion base case
    if type(x) == input_type:
        x = cast(InputType, x)
        return convert(x)

    # Convert containers: bind arguments to helper function
    convert_recursive = functools.partial(
        _convert_recursive, convert=convert, input_type=input_type
    )

    # Convert dictionaries of values
    if type(x) == dict:
        x = cast(dict, x)
        return {k: convert_recursive(v) for k, v in x.items()}

    # Convert lists of values
    elif type(x) == list:
        x = cast(list, x)
        return list(map(convert_recursive, x))

    # Convert tuples of values
    elif type(x) == tuple:
        x = cast(tuple, x)
        return tuple(map(convert_recursive, x))

    # Convert dataclass containing values
    elif dataclasses.is_dataclass(x):
        changes = {}
        for field in dataclasses.fields(x):
            value = getattr(x, field.name)
            try:
                changes[field.name] = convert_recursive(value)
            except AssertionError as e:
                # For dataclasses, we leave unsupported types alone
                # May want to rethink this?
                pass
        return dataclasses.replace(x, **changes)

    # Unsupported input types
    else:
        assert False, f"Unsupported datatype {type(x)}!"


@overload
def to_device(
    x: torch.Tensor, device: torch.device, detach: bool = False
) -> torch.Tensor:
    pass


@overload
def to_device(x: Container, device: torch.device, detach: bool = False) -> Container:
    pass


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

    return _convert_recursive(x, convert=convert, input_type=torch.Tensor)


@overload
def to_torch(
    x: np.ndarray, device: str = "cpu", convert_doubles_to_floats: bool = True,
) -> torch.Tensor:
    pass


@overload
def to_torch(
    x: Container, device: str = "cpu", convert_doubles_to_floats: bool = True,
) -> Container:
    pass


def to_torch(
    x, device="cpu", convert_doubles_to_floats=True,
):
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

    def convert(x: np.ndarray) -> torch.Tensor:
        output = torch.from_numpy(x)
        if x.dtype == np.float64 and convert_doubles_to_floats:
            output = output.float()
        output = output.to(device)
        return output

    return _convert_recursive(x, convert=convert, input_type=np.ndarray)


@overload
def to_numpy(x: torch.Tensor) -> np.ndarray:
    pass


@overload
def to_numpy(x: Container) -> Container:
    pass


def to_numpy(x):
    """Converts a tensor, list of tensors, dict, or dataclass of tensors for use in
    Numpy.

    Args:
        x: (torch.Tensor, list, tuple, dict, or dataclass) Tensor or container of
            tensors to convert to numpy.

    Returns:
        np.ndarray, list, tuple, dict, or dataclass: Output, type will mirror input.
    """

    def convert(x: torch.Tensor) -> np.ndarray:
        output = x.detach().cpu().numpy()
        return output

    return _convert_recursive(x, convert=convert, input_type=torch.Tensor)
