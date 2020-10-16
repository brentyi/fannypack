import os

import pytest
import torch
import torch.nn.functional as F

from ..fixtures import resblock_buddy_temporary_data, simple_buddy


def test_buddy_device(simple_buddy):
    """Make sure Buddy can list the existing checkpoint labels."""
    model, buddy, data, labels = simple_buddy
    assert isinstance(buddy.device, torch.device)


def test_buddy_checkpoint_labels(simple_buddy):
    """Make sure Buddy can list the existing checkpoint labels."""
    model, buddy, data, labels = simple_buddy
    assert set(buddy.checkpoint_labels) == set(["0000000000000200", "new", "legacy"])


def test_buddy_load_checkpoint_new_format_label(simple_buddy):
    """Make sure Buddy can load checkpoints via a label specifier."""
    model, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(model(data), labels)

    buddy.load_checkpoint(label="new")
    final_loss = F.mse_loss(model(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_checkpoint_new_format_path(simple_buddy):
    """Make sure Buddy can load checkpoints directly by path."""
    model, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(model(data), labels)

    buddy.load_checkpoint(
        path=os.path.join(
            # Use path relative to this test
            os.path.dirname(__file__),
            "../assets/checkpoints/simple_net-new.ckpt",
        )
    )
    final_loss = F.mse_loss(model(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_checkpoint_legacy_format_label(simple_buddy):
    """Make sure Buddy is backward-compatible with an old checkpoint
    format when we load it using a label specifier.
    """
    model, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(model(data), labels)

    buddy.load_checkpoint(label="legacy")
    final_loss = F.mse_loss(model(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_checkpoint_legacy_format_path(simple_buddy):
    """Make sure Buddy is backward-compatible with an old checkpoint
    format when we load it directly by path.
    """
    model, buddy, data, labels = simple_buddy
    initial_loss = F.mse_loss(model(data), labels)

    buddy.load_checkpoint(
        path=os.path.join(
            # Use path relative to this test
            os.path.dirname(__file__),
            "../assets/checkpoints/simple_net-legacy.ckpt",
        )
    )
    final_loss = F.mse_loss(model(data), labels)

    assert final_loss < initial_loss / 2.0
    assert buddy.optimizer_steps == 200


def test_buddy_load_missing_checkpoint_path(simple_buddy):
    """Make sure Buddy raises an error if we load a nonexistent checkpoint."""
    model, buddy, data, labels = simple_buddy

    with pytest.raises(FileNotFoundError):
        buddy.load_checkpoint(
            path=os.path.join(
                os.path.dirname(__file__),
                "checkpoints/this_file_doesnt_exist.ckpt",
            )
        )


def test_buddy_load_missing_checkpoint_label(simple_buddy):
    """Make sure Buddy raises an error if we load a nonexistent checkpoint."""
    model, buddy, data, labels = simple_buddy

    with pytest.raises(FileNotFoundError):
        buddy.load_checkpoint(label="this_checkpoint_label_doesnt_exist")


def test_buddy_load_missing_checkpoint(simple_buddy):
    """Make sure Buddy raises an error if we load a nonexistent checkpoint."""
    model, buddy, data, labels = simple_buddy

    with pytest.raises(FileNotFoundError):
        buddy.load_checkpoint(experiment_name="invalid_experiment")


def test_buddy_load_checkpoint_experiment_name(simple_buddy):
    """Make sure Buddy can save/load a checkpoint."""
    model, buddy, data, labels = simple_buddy

    buddy.load_checkpoint(experiment_name="simple_net")
    assert buddy.optimizer_steps == 200


def test_buddy_save_checkpoint(resblock_buddy_temporary_data):
    """Make sure Buddy can save/load a checkpoint."""
    model, buddy, data, labels = resblock_buddy_temporary_data

    # Save initial state
    assert buddy.optimizer_steps == 0
    buddy.save_checkpoint()

    # Step once via ADAM
    loss = F.mse_loss(model(data), labels) * 1000
    buddy.minimize(loss)
    assert buddy.optimizer_steps == 1

    # Restore initial state and verify
    buddy.load_checkpoint()
    assert buddy.optimizer_steps == 0


def test_buddy_save_checkpoint_label(resblock_buddy_temporary_data):
    """Make sure Buddy can save/load a checkpoint with a label specifier."""
    model, buddy, data, labels = resblock_buddy_temporary_data

    # Save initial state
    assert buddy.optimizer_steps == 0
    buddy.save_checkpoint(label="memorable_name")

    # Step once via ADAM
    loss = F.mse_loss(model(data), labels)
    buddy.minimize(loss)
    assert buddy.optimizer_steps == 1

    # Save current state
    buddy.save_checkpoint()

    # Restore initial state and verify
    buddy.load_checkpoint(label="memorable_name")
    assert buddy.optimizer_steps == 0


def test_buddy_save_checkpoint_labels(resblock_buddy_temporary_data):
    """Make sure Buddy's list of existing checkpoints is correctly updated."""
    model, buddy, data, labels = resblock_buddy_temporary_data

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


def test_buddy_checkpoint_modules(resblock_buddy_temporary_data):
    """Test that Buddy can load individual modules from checkpoint files."""
    model, buddy, data, labels = resblock_buddy_temporary_data

    # Check that layer parameters are different
    layer2_params = dict(model.layer2.named_parameters())
    layer4_params = dict(model.layer4.named_parameters())
    assert set(layer2_params.keys()) == set(layer4_params.keys())
    for key in layer2_params.keys():
        if "bias" in key:
            continue
        assert not torch.allclose(layer2_params[key].data, layer4_params[key].data)

    # Save a checkpoint
    buddy.save_checkpoint()

    # Load module parameters from our checkpoint file
    buddy.load_checkpoint_module(source="layer2", target="layer4")

    # If we pass in just a single argument, we should be able to infer the target module
    # Note that this is loading from a checkpoint we just saved, so this operation won't
    # actually modify any parameters
    buddy.load_checkpoint_module("layer2")

    # Check that layer parameters are now identical
    layer2_params = dict(model.layer2.named_parameters())
    layer4_params = dict(model.layer4.named_parameters())
    assert set(layer2_params.keys()) == set(layer4_params.keys())
    for key in layer2_params.keys():
        if "bias" in key:
            continue
        assert torch.allclose(layer2_params[key].data, layer4_params[key].data)


def test_buddy_checkpoint_optimizers(resblock_buddy_temporary_data):
    """Sanity-check for Buddy's optimizer loading interface."""
    model, buddy, data, labels = resblock_buddy_temporary_data

    # Create some optimizer parameters
    buddy.set_learning_rate(1e-3)
    assert buddy.get_learning_rate() == 1e-3

    # Save a checkpoint
    buddy.save_checkpoint()

    # Overwrite optimizer parameter
    buddy.set_learning_rate(1e-5)
    assert buddy.get_learning_rate() == 1e-5
    # Restore optimizer parameters
    buddy.load_checkpoint_optimizers()
    assert buddy.get_learning_rate() == 1e-3

    # Overwrite optimizer parameter
    buddy.set_learning_rate(1e-5)
    assert buddy.get_learning_rate() == 1e-5
    # Restore optimizer parameters
    buddy.load_checkpoint_optimizer("primary")
    assert buddy.get_learning_rate() == 1e-3

    # Overwrite optimizer parameter
    buddy.set_learning_rate(1e-5)
    assert buddy.get_learning_rate() == 1e-5
    # Restore optimizer parameters
    buddy.load_checkpoint_optimizer(source="primary", target="primary")
    assert buddy.get_learning_rate() == 1e-3

    # Load nonexistent optimizer parameter: these should fail
    with pytest.raises(AssertionError):
        buddy.load_checkpoint_optimizer(
            source="this_optimizer_does_not_exist", target="primary"
        )
    with pytest.raises(AssertionError):
        buddy.load_checkpoint_optimizer("this_optimizer_does_not_exist")


def test_buddy_checkpointed_train(resblock_buddy_temporary_data):
    """Do some checks on rapid continuous checkpointing."""
    model, buddy, data, labels = resblock_buddy_temporary_data
    model.train()

    # Save an initial checkpoint
    buddy.save_checkpoint()

    # This should do nothing
    buddy.save_checkpoint()

    assert len(buddy.checkpoint_labels) == 1
    orig_checkpoint = buddy.checkpoint_labels[0]

    # Do some training
    initial_loss = F.mse_loss(model(data), labels)
    buddy.set_learning_rate(1e-3)
    for _ in range(10):
        loss = F.mse_loss(model(data), labels)
        buddy.minimize(loss)
        buddy.save_checkpoint()

    # Checkpoint count should be truncated
    assert len(buddy.checkpoint_labels) == 5
    assert orig_checkpoint not in buddy.checkpoint_labels

    # Loss should at least have halved
    final_loss = F.mse_loss(model(data), labels)
    assert final_loss < initial_loss / 2.0
