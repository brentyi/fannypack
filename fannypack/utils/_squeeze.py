from typing import Any, List, Tuple, Union, cast

__all__ = ["squeeze"]


def squeeze(x: Any, axis: Union[int, Tuple[int, ...]] = None) -> Any:
    """Generic squeeze function.

    Args:
        x (Any): Object to squeeze. Must have a `shape` attribute and be indexable with
            slices.
        axis (Union[int, Tuple[int, ...]], optional): Axis or axes to squeeze along.
            If None (default), squeezes all dimensions with value `1`.

    Returns:
        Any: Squeeze object.
    """
    if type(axis) == int:
        axis = cast(Tuple[int, ...], (axis,))
    else:
        axis = cast(Tuple[int, ...], axis)

    slices: List[Union[int, slice]] = []
    for i, dim in enumerate(x.shape):
        if dim == 1 and (axis is None or i in axis):
            slices.append(0)
        elif axis is not None and i in axis:
            assert False, "Desired axis can't be squeezed"
        else:
            slices.append(slice(None))

    return x[tuple(slices)]
