from typing import Dict

import numpy as np
import torch.nn.functional as F

import fannypack

from ..fixtures import simple_buddy_temporary_data, simple_net


def test_freeze_integration(simple_buddy_temporary_data):
    """Integration test for module freezing + minimizing with Buddy."""
    model, buddy, data, labels = simple_buddy_temporary_data

    # Minimize loss
    loss = F.mse_loss(model(data), labels)
    buddy.minimize(loss)

    # Get NumPy representation of parameters, then copy
    layer4_params_before: Dict[str, np.ndarray] = fannypack.utils.SliceWrapper(
        fannypack.utils.to_numpy(dict(model.layer4.named_parameters()))
    ).map(np.array)

    print(layer4_params_before)

    # Freeze layer4, then minimize
    fannypack.utils.freeze_module(model.layer4)
    for _ in range(20):
        loss = F.mse_loss(model(data), labels)
        buddy.minimize(loss)

    layer4_params_after: Dict[str, np.ndarray] = fannypack.utils.to_numpy(
        dict(model.layer4.named_parameters())
    )

    # Compare (frozen) layer4 parameters before/after second minimize
    assert set(layer4_params_before.keys()) == set(layer4_params_after.keys())
    for key in layer4_params_after.keys():
        np.testing.assert_allclose(layer4_params_before[key], layer4_params_after[key])


def test_freeze_unfreeze(simple_net):
    """Check to make sure freezing/unfreezing a module behaves as expected."""

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
    """Check to make sure unfreezing child modules behaves as expected."""

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
    """Check to make sure freezing child modules behaves as expected."""

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
