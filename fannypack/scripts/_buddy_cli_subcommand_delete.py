import argparse
import os
import shutil

from ._buddy_cli_subcommand import Subcommand
from ._buddy_cli_utils import BuddyPaths, find_checkpoints, find_experiments

_TRASH_DIR = "./_trash/"


def _delete(path, forever) -> None:
    assert os.path.exists(path)

    if forever:
        # Delete file/directory forever
        if os.path.isdir(path):
            print(f"Deleting {path} (recursive)")
            shutil.rmtree(path)
        elif os.path.isfile(path):
            print(f"Deleting {path}")
            os.remove(path)
        else:
            assert False, "Something went wrong"
    else:
        # Move files/directory to a new path
        new_path = os.path.join(_TRASH_DIR, path)

        # Create trash directory if it doesn't exist yet
        directory = os.path.dirname(new_path)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        # Move file/directory to trash
        print(f"Moving {path} to {new_path}")
        os.rename(path, new_path)


class DeleteSubcommand(Subcommand):
    """Delete a Buddy experiment.
    """

    subcommand: str = "delete"

    @classmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        parser.add_argument(
            "experiment_name",
            type=str,
            help="Name of experiment, as printed by `$ buddy list`.",
            metavar="EXPERIMENT_NAME",  # Set metavar => don't show choices in help menu
            choices=find_experiments(paths).experiment_names,
        )
        parser.add_argument(
            "--forever",
            action="store_true",
            help=f"Delete experiment forever: if unset, move files into `{_TRASH_DIR}`.",
        )

    @classmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        # Get experiment name
        experiment_name = args.experiment_name

        # If we're just moving an experiment, check that it doesn't exist already
        if not args.forever:
            new_checkpoint_files = find_checkpoints(
                experiment_name, path=os.path.join(_TRASH_DIR, paths.checkpoint_dir)
            )
            if len(new_checkpoint_files) != 0:
                raise RuntimeError(
                    "Checkpoints for matching experiment name already exist in trash; "
                    "rename experiment before deleting."
                )
            if os.path.exists(
                os.path.join(_TRASH_DIR, paths.log_dir, f"{experiment_name}")
            ):
                raise RuntimeError(
                    "Logs for matching experiment name already exist in trash; "
                    "rename experiment before deleting."
                )
            if os.path.exists(
                os.path.join(_TRASH_DIR, paths.metadata_dir, f"{experiment_name}.yaml")
            ):
                raise RuntimeError(
                    "Metadata for matching experiment name already exist in trash; "
                    "rename experiment before deleting."
                )

        # Delete checkpoint files
        checkpoint_paths = find_checkpoints(experiment_name, paths.checkpoint_dir)
        print(f"Found {len(checkpoint_paths)} checkpoint files")
        for path in checkpoint_paths:
            _delete(path, args.forever)

        # Delete metadata
        metadata_path = os.path.join(paths.metadata_dir, f"{experiment_name}.yaml")
        if os.path.exists(metadata_path):
            _delete(metadata_path, args.forever)
        else:
            print("No metadata found")

        # Delete logs
        log_path = os.path.join(paths.log_dir, f"{experiment_name}")
        if os.path.exists(log_path):
            _delete(log_path, args.forever)
        else:
            print("No logs found")
