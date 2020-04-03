_freeze_restore_values = {}


def freeze_module(module, recurse=True):
    """Freeze the weights of a PyTorch module.

    Args:
        module (torch.nn.Module): Module to freeze.
        recurse (bool, optional): If True, then recursively freezes children.
            Otherwise, only freezes immediate parameters.
    """

    global _freeze_restore_values

    # Do nothing if module is already frozen
    if module in _freeze_restore_values:
        return

    # Recursively call on children
    if recurse:
        for child in module.children():
            freeze_module(child)

    # Freeze parameters
    restore_values = {}
    for name, parameter in module.named_parameters(recurse=False):
        restore_values[name] = parameter.requires_grad
        parameter.requires_grad = False
    _freeze_restore_values[module] = restore_values


def unfreeze_module(module, recurse=True):
    """Freeze the weights of a PyTorch module.

    Args:
        module (torch.nn.Module): Module to freeze.
        recurse (bool, optional): If True, then recursively freezes children.
            Otherwise, only freezes immediate parameters.
    """

    global _freeze_restore_values

    # Do nothing if module is already unfrozen
    if module not in _freeze_restore_values:
        return

    # Recursively call on children
    if recurse:
        for child in module.children():
            freeze_module(child)

    # Freeze parameters
    restore_values = _freeze_restore_values.pop(module)
    for name, parameter in module.named_parameters(recurse=False):
        parameter.requires_grad = restore_values[name]
