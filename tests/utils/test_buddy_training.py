import pytest
import fannypack
import numpy as np
import os
import torch
import torch.nn.functional as F

from ..fixtures import simple_buddy_temporary_data


def test_buddy_train(simple_buddy_temporary_data):
    """Make sure Buddy losses go down.
    """
    model, buddy, data, labels = simple_buddy_temporary_data
    model.train()

    initial_loss = F.mse_loss(model(data), labels)

    # Try using LR scheduler interface
    buddy.set_learning_rate(lambda steps: 1e-3)
    buddy.set_learning_rate(1e-3)
    buddy.set_learning_rate(lambda steps: 1e-3)

    for _ in range(200):
        # Optimize
        loss = F.mse_loss(model(data), labels)
        buddy.minimize(loss)

        # Log for tensorboard
        with buddy.log_scope("scope"):
            buddy.log("loss", loss)

    # Loss should at least have halved
    final_loss = F.mse_loss(model(data), labels)
    assert final_loss < initial_loss / 2.0


def test_buddy_train_multiloss_unstable(simple_buddy_temporary_data):
    """Training should be less happy if we (a) use a single optimizer and (b)
    switch abruptly between loss functions with very different scales.
    """
    model, buddy, data, labels = simple_buddy_temporary_data
    model.train()

    initial_loss = F.mse_loss(model(data), labels)
    buddy.set_learning_rate(1e-3)
    for _ in range(50):
        loss = F.mse_loss(model(data), labels) * 1000
        buddy.minimize(loss)
    for _ in range(100):
        loss = F.mse_loss(model(data), labels)
        buddy.minimize(loss)
    for _ in range(50):
        loss = F.mse_loss(model(data), labels) * 1000
        buddy.minimize(loss)

    # Loss will not have halved
    final_loss = F.mse_loss(model(data), labels)
    assert final_loss > initial_loss / 2.0


def test_buddy_train_multiloss_stable(simple_buddy_temporary_data):
    """Training should stabilize if we use separate optimizers for our two
    different loss functions.
    """
    model, buddy, data, labels = simple_buddy_temporary_data
    model.train()

    initial_loss = F.mse_loss(model(data), labels)
    buddy.set_learning_rate(1e-3, "big_loss")
    buddy.set_learning_rate(1e-3, "little_loss")
    for _ in range(50):
        loss = F.mse_loss(model(data), labels) * 1000
        buddy.minimize(loss, optimizer_name="big_loss")
    for _ in range(100):
        loss = F.mse_loss(model(data), labels)
        buddy.minimize(loss, optimizer_name="little_loss")
    for _ in range(50):
        loss = F.mse_loss(model(data), labels) * 1000
        buddy.minimize(loss, optimizer_name="big_loss")

    final_loss = F.mse_loss(model(data), labels)

    # Loss should at least have halved
    assert final_loss < initial_loss / 2.0
