import pytest

import fannypack
import torch

from ..fixtures import simple_net


def test_freeze_unfreeze(simple_net):
    """Check to make sure freezing/unfreezing a module behaves as expected.
    """

    # All parameters should be initially unfrozen
    _check_frozen(simple_net, False)

    # Freeze the module
    fannypack.utils.freeze_module(simple_net)

    # All parameters should be frozen
    _check_frozen(simple_net, True)

    # Unfreeze the module
    fannypack.utils.unfreeze_module(simple_net)

    # All parameters should be unfrozen
    _check_frozen(simple_net, False)


def test_unfreeze_children(simple_net):
    """Check to make sure unfreezing child modules behaves as expected.
    """

    # All parameters should be initially unfrozen
    _check_frozen(simple_net, False)

    # Freeze the module
    fannypack.utils.freeze_module(simple_net)

    # All parameters should be frozen
    _check_frozen(simple_net, True)

    # Unfreeze the module
    fannypack.utils.unfreeze_module(simple_net.layer1)
    fannypack.utils.unfreeze_module(simple_net.layer2)
    fannypack.utils.unfreeze_module(simple_net.layer3)
    fannypack.utils.unfreeze_module(simple_net.layer4)

    # All parameters should be unfrozen
    _check_frozen(simple_net, False)


def test_freeze_children(simple_net):
    """Check to make sure freezing child modules behaves as expected.
    """

    # All parameters should be initially unfrozen
    _check_frozen(simple_net, False)

    # Freeze a child module
    fannypack.utils.freeze_module(simple_net.layer1)

    # All parameters for layer1 should be frozen
    _check_frozen(simple_net.layer1, True)
    _check_frozen(simple_net.layer2, False)
    _check_frozen(simple_net.layer3, False)
    _check_frozen(simple_net.layer4, False)

    # Unfreeze the entire module
    fannypack.utils.unfreeze_module(simple_net)

    # All parameters should be unfrozen
    _check_frozen(simple_net, False)


# Helper to check whether a module is frozen or not
def _check_frozen(module, is_frozen):
    for param in module.parameters():
        assert param.requires_grad == (not is_frozen)
