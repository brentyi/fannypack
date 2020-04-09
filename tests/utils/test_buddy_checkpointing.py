import pytest
import fannypack
import numpy as np
import os
import torch
import torch.nn.functional as F

from ..fixtures import simple_buddy, simple_buddy_temporary_data


def test_buddy_checkpoint_labels(simple_buddy):
    """Make sure Buddy can list the existing checkpoint labels.
    """
    simple_net, buddy, data, labels = simple_buddy
    assert set(buddy.checkpoint_labels) == set(["new", "legacy"])


def test_buddy_load_checkpoint_new_format_path(simple_buddy):
    """Make sure Buddy can load checkpoints via a label specifier.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)

    buddy.load_checkpoint(label="new")
    final_loss = F.mse_loss(simple_net(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_checkpoint_new_format_path(simple_buddy):
    """Make sure Buddy can load checkpoints directly by path.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)

    buddy.load_checkpoint(
        path=os.path.join(
            # Use path relative to this test
            os.path.dirname(__file__),
            "../data/checkpoints/simple_net-new.ckpt",
        )
    )
    final_loss = F.mse_loss(simple_net(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_checkpoint_legacy_format_label(simple_buddy):
    """Make sure Buddy is backward-compatible with an old checkpoint
    format when we load it using a label specifier.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)

    buddy.load_checkpoint(label="legacy")
    final_loss = F.mse_loss(simple_net(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_checkpoint_legacy_format_path(simple_buddy):
    """Make sure Buddy is backward-compatible with an old checkpoint
    format when we load it directly by path.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)

    buddy.load_checkpoint(
        path=os.path.join(
            # Use path relative to this test
            os.path.dirname(__file__),
            "../data/checkpoints/simple_net-legacy.ckpt",
        )
    )
    final_loss = F.mse_loss(simple_net(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_missing_checkpoint_path(simple_buddy):
    """Make sure Buddy raises an error if we load a nonexistent checkpoint.
    """
    simple_net, buddy, data, labels = simple_buddy

    with pytest.raises(FileNotFoundError):
        buddy.load_checkpoint(
            path=os.path.join(
                os.path.dirname(__file__),
                "checkpoints/this_file_doesnt_exist.ckpt",
            )
        )


def test_buddy_load_missing_checkpoint_label(simple_buddy):
    """Make sure Buddy raises an error if we load a nonexistent checkpoint.
    """
    simple_net, buddy, data, labels = simple_buddy

    with pytest.raises(FileNotFoundError):
        buddy.load_checkpoint(label="this_checkpoint_label_doesnt_exist")


def test_buddy_save_checkpoint(simple_buddy_temporary_data):
    """Make sure Buddy can save/load a checkpoint.
    """
    simple_net, buddy, data, labels = simple_buddy_temporary_data

    # Save initial state
    assert buddy.optimizer_steps == 0
    buddy.save_checkpoint()

    # Step once via ADAM
    loss = F.mse_loss(simple_net(data), labels) * 1000
    buddy.minimize(loss)
    assert buddy.optimizer_steps == 1

    # Restore initial state and verify
    buddy.load_checkpoint()
    assert buddy.optimizer_steps == 0


def test_buddy_save_checkpoint_label(simple_buddy_temporary_data):
    """Make sure Buddy can save/load a checkpoint with a label specifier.
    """
    simple_net, buddy, data, labels = simple_buddy_temporary_data

    # Save initial state
    assert buddy.optimizer_steps == 0
    buddy.save_checkpoint(label="memorable_name")

    # Step once via ADAM
    loss = F.mse_loss(simple_net(data), labels)
    buddy.minimize(loss)
    assert buddy.optimizer_steps == 1

    # Save current state
    buddy.save_checkpoint()

    # Restore initial state and verify
    buddy.load_checkpoint(label="memorable_name")
    assert buddy.optimizer_steps == 0


def test_buddy_save_checkpoint_labels(simple_buddy_temporary_data):
    """Make sure Buddy's list of existing checkpoints is correctly updated.
    """
    simple_net, buddy, data, labels = simple_buddy_temporary_data

    # No initial checkpoints
    assert len(buddy.checkpoint_labels) == 0

    # Save initial state
    buddy.save_checkpoint()

    # Check checkpoint_labels property
    assert len(buddy.checkpoint_labels) == 1
    assert int(buddy.checkpoint_labels[0]) == buddy.optimizer_steps

    # Save labeled state
    buddy.save_checkpoint(label="memorable_name")

    # Check checkpoint_labels property
    assert len(buddy.checkpoint_labels) == 2
    assert "memorable_name" in buddy.checkpoint_labels
