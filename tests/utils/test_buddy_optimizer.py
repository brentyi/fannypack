import pytest
import fannypack
import numpy as np
import os
import torch
import torch.nn.functional as F

from ..fixtures import simple_buddy


def test_buddy_train(simple_buddy):
    """Make sure Buddy losses go down.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)
    buddy.set_learning_rate(1e-3)
    for _ in range(200):
        loss = F.mse_loss(simple_net(data), labels)
        buddy.minimize(loss)

    # Loss should at least have halved
    final_loss = F.mse_loss(simple_net(data), labels)
    assert final_loss < initial_loss / 2.0


def test_buddy_train_multiloss_unstable(simple_buddy):
    """Training should be less happy if we (a) use a single optimizer and (b)
    switch abruptly between loss functions with very different scales.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)
    buddy.set_learning_rate(1e-3)
    for _ in range(50):
        loss = F.mse_loss(simple_net(data), labels) * 1000
        buddy.minimize(loss)
    for _ in range(100):
        loss = F.mse_loss(simple_net(data), labels)
        buddy.minimize(loss)
    for _ in range(50):
        loss = F.mse_loss(simple_net(data), labels) * 1000
        buddy.minimize(loss)

    # Loss will not have halved
    final_loss = F.mse_loss(simple_net(data), labels)
    assert final_loss > initial_loss / 2.0


def test_buddy_train_multiloss_stable(simple_buddy):
    """Training should stabilize if we use separate optimizers for our two
    different loss functions.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)
    buddy.set_learning_rate(1e-3, "big_loss")
    buddy.set_learning_rate(1e-3, "little_loss")
    for _ in range(50):
        loss = F.mse_loss(simple_net(data), labels) * 1000
        buddy.minimize(loss, optimizer_name="big_loss")
    for _ in range(100):
        loss = F.mse_loss(simple_net(data), labels)
        buddy.minimize(loss, optimizer_name="little_loss")
    for _ in range(50):
        loss = F.mse_loss(simple_net(data), labels) * 1000
        buddy.minimize(loss, optimizer_name="big_loss")

    final_loss = F.mse_loss(simple_net(data), labels)

    # Loss should at least have halved
    assert final_loss < initial_loss / 2.0
