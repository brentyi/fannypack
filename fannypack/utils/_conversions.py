import functools
from typing import Callable, Dict, List, Tuple, Type, TypeVar, Union, cast, overload

import numpy as np
import torch

Container = TypeVar("Container", bound=Union[Dict, List, Tuple])
_InputType = TypeVar("_InputType", np.ndarray, torch.Tensor)
_OutputType = TypeVar("_OutputType", np.ndarray, torch.Tensor)


@overload
def _convert_recursive(
    x: _InputType, convert: Callable[[_InputType], _OutputType], input_type: Type,
) -> _OutputType:
    ...


@overload
def _convert_recursive(
    x: Container, convert: Callable[[_InputType], _OutputType], input_type: Type,
) -> Container:
    ...


def _convert_recursive(x, convert, input_type):
    """Private conversion helper. Recursively calls a conversion function on inputs
    within a nested set of containers.
    """

    # Conversion base case
    if isinstance(x, input_type):
        x = cast(_InputType, x)
        return convert(x)

    # Convert containers: bind arguments to helper function
    convert_recursive = functools.partial(
        _convert_recursive, convert=convert, input_type=input_type
    )

    # Convert dictionaries of values
    if isinstance(x, dict):
        x = cast(dict, x)
        return dict(zip(x.keys(), map(convert_recursive, x.values())))

    # Convert lists of values
    if isinstance(x, list):
        x = cast(list, x)
        return list(map(convert_recursive, x))

    # Convert tuples of values
    if isinstance(x, tuple):
        x = cast(tuple, x)
        if hasattr(x, "_fields"):  # NamedTuple
            return type(x)(*map(convert_recursive, x))
        else:
            return tuple(map(convert_recursive, x))

    # Unsupported input types
    assert False, f"Unsupported datatype {type(x)}!"


@overload
def to_device(
    x: torch.Tensor, device: torch.device, detach: bool = False
) -> torch.Tensor:
    ...


@overload
def to_device(x: Container, device: torch.device, detach: bool = False) -> Container:
    ...


def to_device(
    x: Union[Container, torch.Tensor], device: torch.device, detach: bool = False
) -> Union[Container, torch.Tensor]:
    """Move a torch tensor, list, tuple (standard or named), or dict of tensors to a
    different device. Recursively applied for nested containers.

    Args:
        x (torch.Tensor, list, tuple (standard or named), or dict): Tensor or container
            of tensors to move.
        device (torch.device): Target device.
        detach (bool, optional): If set, detaches tensors after moving. Defaults to
            False.

    Returns:
        torch.Tensor, list, tuple (standard or named), or dict: Output, type will mirror
        input.
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
    ...


@overload
def to_torch(
    x: Container, device: str = "cpu", convert_doubles_to_floats: bool = True,
) -> Container:
    ...


def to_torch(
    x, device="cpu", convert_doubles_to_floats=True,
):
    """Converts a NumPy array, list, tuple (standard or named), or dict of NumPy arrays
    for use in PyTorch.  Recursively applied for nested containers.

    Args:
        x (np.ndarray, list, tuple (standard or named), or dict): Array or container of
            arrays to convert to torch tensors.
        device (torch.device, optional): Torch device to create tensors on. Defaults to
            `"cpu"`.
        convert_doubles_to_floats (bool, optional): If set, converts 64-bit floats to
            32-bit. Defaults to True.

    Returns:
        torch.Tensor, list, tuple (standard or named), or dict: Output, type will mirror input.
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
    ...


@overload
def to_numpy(x: Container) -> Container:
    ...


def to_numpy(x):
    """Converts a tensor, list, tuple (standard or named), or dict of tensors for use in
    Numpy. Recursively applied for nested containers.

    Args:
        x (torch.Tensor, list, tuple (standard or named), or dict): Tensor or container
            of tensors to convert to NumPy.

    Returns:
        np.ndarray, list, tuple (standard or named), or dict: Output, type will mirror input.
    """

    def convert(x: torch.Tensor) -> np.ndarray:
        output = x.detach().cpu().numpy()
        return output

    return _convert_recursive(x, convert=convert, input_type=torch.Tensor)
