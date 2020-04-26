from typing import Callable, Union, cast, overload

import numpy as np
import torch


@overload
def _convert(x: list, convert: Callable) -> list:
    pass


@overload
def _convert(x: tuple, convert: Callable) -> tuple:
    pass


@overload
def _convert(x: dict, convert: Callable) -> dict:
    pass


@overload
def _convert(x: torch.Tensor, convert: Callable) -> torch.Tensor:
    pass


@overload
def _convert(x: np.ndarray, convert: Callable) -> np.ndarray:
    pass


def _convert(x, convert):
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
    else:
        assert False, f"Invalid datatype {type(x)}!"
    return output


@overload
def to_device(x: tuple, device: torch.device, detach: bool = False,) -> tuple:
    pass


@overload
def to_device(x: list, device: torch.device, detach: bool = False,) -> list:
    pass


@overload
def to_device(x: dict, device: torch.device, detach: bool = False,) -> dict:
    pass


@overload
def to_device(
    x: torch.Tensor, device: torch.device, detach: bool = False,
) -> torch.Tensor:
    pass


def to_device(
    x, device, detach=False,
):
    """Copies a tensor, list of tensors, or dict of tensors to a different
    device.
    """

    def convert(x):
        if detach:
            x = x.detach()
        return x.to(device)

    return _convert(x, convert)


@overload
def to_torch(x: tuple, device: str = "cpu") -> tuple:
    pass


@overload
def to_torch(x: list, device: str = "cpu") -> list:
    pass


@overload
def to_torch(x: dict, device: str = "cpu") -> dict:
    pass


@overload
def to_torch(x: np.ndarray, device: str = "cpu") -> np.ndarray:
    pass


def to_torch(x, device="cpu"):
    """Converts a numpy array, list of numpy arrays, or dict of numpy arrays
    for use in PyTorch.
    """

    def convert(x: np.ndarray):
        output = torch.from_numpy(x)
        if x.dtype == np.float64:
            # This is maybe sketchy? Undocumented behavior?
            output = output.float()
        output = output.to(device)
        return output

    return _convert(x, convert)


@overload
def to_numpy(x: tuple) -> tuple:
    pass


@overload
def to_numpy(x: list) -> list:
    pass


@overload
def to_numpy(x: dict) -> dict:
    pass


@overload
def to_numpy(x: np.ndarray) -> np.ndarray:
    pass


def to_numpy(x):
    """Converts a tensor, list of tensors, or dict of tensors for use in Numpy.
    """

    def convert(x: torch.Tensor):
        output = x.detach().cpu().numpy()
        return output

    return _convert(x, convert)
