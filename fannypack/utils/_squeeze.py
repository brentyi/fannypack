def squeeze(x, axis=None):
    """Generic squeeze function.
    """
    slices = []
    for i, dim in enumerate(x.shape):
        if dim == 1 and (axis is None or axis == i):
            slices.append(0)
        elif axis == i:
            assert False, "Desired axis can't be squeezed"
        else:
            slices.append(slice(None))

    slices = tuple(slices)
    return x[slices]
