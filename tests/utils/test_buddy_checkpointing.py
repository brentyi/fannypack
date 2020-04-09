import pytest
import fannypack
import numpy as np
import os
import torch
import torch.nn.functional as F

from ..fixtures import simple_buddy


def test_buddy_load_checkpoint_new_format(simple_buddy):
    """Make sure Buddy can load checkpoints.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)

    buddy.load_checkpoint(
        path=os.path.join(
            os.path.dirname(__file__), "checkpoints/simple_net_new.ckpt"
        )
    )
    final_loss = F.mse_loss(simple_net(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_checkpoint_legacy_format(simple_buddy):
    """Make sure Buddy is backward-compatible with an old checkpoint
    format.
    """
    simple_net, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(simple_net(data), labels)

    buddy.load_checkpoint(
        path=os.path.join(
            os.path.dirname(__file__), "checkpoints/simple_net_legacy.ckpt"
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
