import numpy as np
import pytest
import torch
import torch.nn.functional as F

import fannypack

from ..fixtures import simple_buddy_temporary_data


def test_buddy_no_model():
    """Check that errors are raised if a Buddy is used without a model attached."""
    buddy = fannypack.utils.Buddy("no_model")

    with pytest.raises(AssertionError):
        buddy.model

    with pytest.raises(AssertionError):
        buddy.save_checkpoint()

    with pytest.raises(AssertionError):
        buddy.load_checkpoint()

    with pytest.raises(AssertionError):
        fake_loss = torch.Tensor([0.0]).to(buddy.device)
        buddy.minimize(fake_loss)


def test_buddy_log_scopes(simple_buddy_temporary_data):
    """Check that log scope functions as expected."""
    model, buddy, data, labels = simple_buddy_temporary_data

    assert buddy.log_scope_prefix() == ""
    with buddy.log_scope("scope0"):
        buddy.log_scope_push("scope1")
        with buddy.log_scope("scope2"):
            assert buddy.log_scope_prefix("name") == "scope0/scope1/scope2/name"
            buddy.log_image("garbage_image", np.zeros((3, 32, 32), dtype=np.float32))
        buddy.log_scope_pop("scope1")


def test_buddy_log_histograms_no_grads(simple_buddy_temporary_data):
    """Check behavior of histogram logging when no gradients exist."""
    model, buddy, data, labels = simple_buddy_temporary_data

    # If we log parameters with no gradients, nothing should happen
    buddy.log_parameters_histogram(ignore_zero_grad=True)

    # If we log gradients with no gradients... throw an error
    with pytest.raises(AssertionError):
        buddy.log_grad_histogram()


def test_buddy_learning_rates(simple_buddy_temporary_data):
    """Check that we can set learning rates. (scalar)"""
    model, buddy, data, labels = simple_buddy_temporary_data

    buddy.set_learning_rate(1e-5)
    assert buddy.get_learning_rate() == 1e-5

    buddy.set_learning_rate(lambda steps: 1e-3)
    assert buddy.get_learning_rate() == 1e-3

    buddy.set_learning_rate(1e-5)
    assert buddy.get_learning_rate() == 1e-5


def test_buddy_learning_rates_lambda(simple_buddy_temporary_data):
    """Check that we can set learning rates. (lambda)"""
    model, buddy, data, labels = simple_buddy_temporary_data

    buddy.set_learning_rate(lambda s: 1e-2)
    assert buddy.get_learning_rate() == 1e-2

    buddy.set_learning_rate(lambda steps: 1e-5)
    assert buddy.get_learning_rate() == 1e-5


def test_buddy_train(simple_buddy_temporary_data):
    """Make sure Buddy losses go down, and that we can write log files without errors."""
    model, buddy, data, labels = simple_buddy_temporary_data
    assert buddy.optimizer_steps == 0
    model.train()

    initial_loss = F.mse_loss(model(data), labels)

    # Try using default learning rate
    buddy.set_default_learning_rate(lambda steps: 1e-3)

    for _ in range(200):
        # Optimize
        loss = F.mse_loss(model(data), labels)
        buddy.minimize(loss)

        # Log for tensorboard
        with buddy.log_scope("scope"):
            buddy.log_scalar("loss", loss)
        buddy.log_parameters_histogram()
        buddy.log_grad_histogram()

    # Flush logs
    buddy.log_writer.flush()

    assert buddy.get_learning_rate() == 1e-3
    assert buddy.optimizer_steps == 200

    # Loss should at least have halved
    final_loss = F.mse_loss(model(data), labels)
    assert final_loss < initial_loss / 2.0
    buddy.save_checkpoint()


def test_buddy_gradient_clipping(simple_buddy_temporary_data):
    """Test that setting the max gradient norm correctly clips the gradients."""

    def _gradient_norm(f: torch.nn.Module) -> float:
        """Compute the gradient norm.

        Args:
            f (torch.nn.Module): Module to check gradients of..

        Returns:
            float: The gradient norm.
        """
        total_norm = 0.0
        for p in f.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    model, buddy, data, labels = simple_buddy_temporary_data
    assert buddy.optimizer_steps == 0
    model.train()

    # Use an extremely small norm value
    max_norm_value = 0.01

    for _ in range(5):
        # Optimize without gradient clipping
        loss = F.mse_loss(model(data), labels)
        buddy.minimize(loss)
        norm_value = _gradient_norm(model)
        assert norm_value > max_norm_value

    for _ in range(5):
        # Optimize with gradient clipping
        loss = F.mse_loss(model(data), labels)
        buddy.minimize(loss, clip_grad_max_norm=max_norm_value)
        norm_value = _gradient_norm(model)

        # All gradients should be clipped
        assert abs(norm_value - max_norm_value) < 1e-5


def test_buddy_train_multiloss_unstable(simple_buddy_temporary_data):
    """Training should be less happy if we (a) use a single optimizer and (b)
    switch abruptly between loss functions with very different scales.
    """
    model, buddy, data, labels = simple_buddy_temporary_data
    assert buddy.optimizer_steps == 0
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

    assert buddy.optimizer_steps == 200

    # Loss will not have halved
    final_loss = F.mse_loss(model(data), labels)
    assert final_loss > initial_loss / 2.0


def test_buddy_train_multiloss_stable(simple_buddy_temporary_data):
    """Training should stabilize if we use separate optimizers for our two
    different loss functions.
    """
    model, buddy, data, labels = simple_buddy_temporary_data
    assert buddy.optimizer_steps == 0
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

    assert buddy.optimizer_steps == 200

    # Loss should at least have halved
    final_loss = F.mse_loss(model(data), labels)
    assert final_loss < initial_loss / 2.0
