import dataclasses
import functools
from typing import Callable, Dict, List, Tuple, TypeVar, Union, cast, overload

import numpy as np
import torch

# Type aliases, variables
Container = TypeVar("Container", Dict, List, Tuple)
InputType = TypeVar("InputType", np.ndarray, torch.Tensor)
OutputType = TypeVar("OutputType", np.ndarray, torch.Tensor)


@overload
def _convert_recursive(
    x: InputType, convert: Callable[[InputType], OutputType], input_type: type,
) -> OutputType:
    ...


@overload
def _convert_recursive(
    x: Container, convert: Callable[[InputType], OutputType], input_type: type,
) -> Container:
    ...


def _convert_recursive(x, convert, input_type):
    """Private conversion helper. Recursively calls a conversion function on inputs
    within a nested set of containers.
    """

    # Conversion base case
    if isinstance(x, input_type):
        x = cast(InputType, x)
        return convert(x)

    # Convert containers: bind arguments to helper function
    convert_recursive = functools.partial(
        _convert_recursive, convert=convert, input_type=input_type
    )

    # Convert dictionaries of values
    if isinstance(x, dict):
        x = cast(dict, x)
        return {k: convert_recursive(v) for k, v in x.items()}

    # Convert lists of values
    if isinstance(x, list):
        x = cast(list, x)
        return list(map(convert_recursive, x))

    # Convert tuples of values
    if isinstance(x, tuple):
        x = cast(tuple, x)
        return tuple(map(convert_recursive, x))

    # Convert dataclass containing values
    if dataclasses.is_dataclass(x):
        changes = {}
        for field in dataclasses.fields(x):
            value = getattr(x, field.name)
            try:
                changes[field.name] = convert_recursive(value)
            except AssertionError:
                # For dataclasses, we leave unsupported types alone
                # May want to rethink this?
                pass
        return dataclasses.replace(x, **changes)

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
    """Move a torch tensor, list of tensors, dict, or dataclass of tensors to a
    different device. Recursively applied for nested containers.

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
    ...


@overload
def to_torch(
    x: Container, device: str = "cpu", convert_doubles_to_floats: bool = True,
) -> Container:
    ...


def to_torch(
    x, device="cpu", convert_doubles_to_floats=True,
):
    """Converts a numpy array, list of numpy arrays, dict, or dataclass of numpy arrays
    for use in PyTorch. Recursively applied for nested containers.

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
    ...


@overload
def to_numpy(x: Container) -> Container:
    ...


def to_numpy(x):
    """Converts a tensor, list of tensors, dict, or dataclass of tensors for use in
    Numpy. Recursively applied for nested containers.

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
