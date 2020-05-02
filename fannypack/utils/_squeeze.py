from typing import Iterable, List, Tuple, Union, cast


def squeeze(x, axis: Union[int, Tuple[int, ...]] = None):
    """Generic squeeze function.
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
