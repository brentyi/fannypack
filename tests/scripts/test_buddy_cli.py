import os
import subprocess
from typing import AnyStr, List, Tuple, Union

import torch

import fannypack


def _run_command(command: Union[str, List[str]]) -> Tuple[AnyStr, AnyStr, int]:
    """Helper for running a command & returning results.
    """
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.join(os.path.dirname(__file__), "../assets"),
    )
    out, err = proc.communicate()
    return out, err, proc.returncode


def test_buddy_no_args():
    """Make sure that `buddy` fails without arguments.
    """
    out, err, exitcode = _run_command(["buddy"])
    assert exitcode == 2


def test_buddy_delete_no_args():
    """Make sure that `buddy delete` fails without arguments.
    """
    out, err, exitcode = _run_command(["buddy", "delete"])
    assert exitcode == 2


def test_buddy_info_no_args():
    """Make sure that `buddy info` fails without arguments.
    """
    out, err, exitcode = _run_command(["buddy", "info"])
    assert exitcode == 2


def test_buddy_list():
    """Check that we can list experiments.
    """
    out, err, exitcode = _run_command(["buddy", "list"])
    assert exitcode == 0
    assert out.startswith(b"Found 3 experiments")
    assert out


def test_buddy_rename_no_args():
    """Make sure that `buddy rename` fails without arguments.
    """
    out, err, exitcode = _run_command(["buddy", "rename"])
    assert exitcode == 2


def test_buddy_info():
    """Make sure that `buddy info` gives us sane results.
    """

    out, err, exitcode = _run_command(["buddy", "info", "simple_net"])
    assert exitcode == 0
    assert b"(steps: 200)" in out


def test_buddy_rename():
    """Make sure that we can rename experiments.
    """
    # Pre-condition
    out, err, exitcode = _run_command(["buddy", "list"])
    assert exitcode == 0
    assert out.startswith(b"Found 3 experiments")
    assert b"simple_net" in out

    # Rename experiment
    out, err, exitcode = _run_command(["buddy", "rename", "simple_net", "blah"])
    assert exitcode == 0

    # Post-condition
    out, err, exitcode = _run_command(["buddy", "list"])
    assert exitcode == 0
    assert out.startswith(b"Found 3 experiments")
    assert b"simple_net" not in out

    # Revert changes
    out, err, exitcode = _run_command(["buddy", "rename", "blah", "simple_net"])
    assert exitcode == 0


def test_buddy_rename():
    """Make sure that we can delete experiments.
    """

    # Create experiment
    buddy = fannypack.utils.Buddy(
        "temporary_net",
        model=torch.nn.Linear(10, 20),
        # Use directories relative to this fixture
        checkpoint_dir=os.path.join(
            os.path.dirname(__file__), "../assets/checkpoints/"
        ),
        metadata_dir=os.path.join(os.path.dirname(__file__), "../assets/metadata/"),
        log_dir=os.path.join(os.path.dirname(__file__), "../assets/logs/"),
        verbose=True,
        # Disable auto-checkpointing
        optimizer_checkpoint_interval=0,
        cpu_only=True,
    )

    # Pre-condition
    out, err, exitcode = _run_command(["buddy", "list"])
    assert exitcode == 0
    assert out.startswith(b"Found 3 experiments")
    assert b"temporary_net" not in out

    # Save some files
    buddy.add_metadata({"blah": "blah"})
    buddy.save_checkpoint()

    # Pre-condition
    out, err, exitcode = _run_command(["buddy", "list"])
    assert exitcode == 0
    assert out.startswith(b"Found 4 experiments")
    assert b"temporary_net" in out

    # Delete buddy
    del buddy

    # Delete experiment
    out, err, exitcode = _run_command(["buddy", "delete", "temporary_net", "--forever"])
    assert exitcode == 0

    # Post-condition
    out, err, exitcode = _run_command(["buddy", "list"])
    assert exitcode == 0
    assert out.startswith(b"Found 3 experiments")
    assert b"temporary_net" not in out
