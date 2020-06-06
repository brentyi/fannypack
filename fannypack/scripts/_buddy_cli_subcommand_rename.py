import argparse
import glob
import os

import argcomplete

from ._buddy_cli_subcommand import Subcommand
from ._buddy_cli_utils import BuddyPaths, find_experiments


class RenameSubcommand(Subcommand):
    """Rename a Buddy experiment.
    """

    subcommand: str = "rename"

    @classmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        parser.add_argument(
            "source",
            type=str,
            help="Current name of experiment, as printed by `$ buddy list`.",
            metavar="SOURCE",  # Set metavar => don't show choices in help menu
            choices=find_experiments(paths).experiment_names,
        )
        parser.add_argument("dest", type=str, help="New name of experiment.")

    @classmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        # Get old, new experiment names
        old_experiment_name = args.source
        new_experiment_name = args.dest

        # Validate that new experiment name doesn't exist
        new_checkpoint_files = glob.glob(
            os.path.join(
                paths.checkpoint_dir, f"{glob.escape(new_experiment_name)}-*.ckpt"
            )
        )
        if len(new_checkpoint_files) != 0:
            raise RuntimeError(
                f"Checkpoints already exist for destination name: {new_experiment_name}"
            )
        if os.path.exists(os.path.join(paths.log_dir, f"{new_experiment_name}")):
            raise RuntimeError(
                f"Logs already exist for destination name: {new_experiment_name}"
            )
        if os.path.exists(
            os.path.join(paths.metadata_dir, f"{new_experiment_name}.yaml")
        ):
            raise RuntimeError(
                f"Metadata already exist for destination name: {new_experiment_name}"
            )

        # Move checkpoint files
        checkpoint_paths = [
            path
            for path in glob.glob(
                os.path.join(
                    paths.checkpoint_dir, f"{glob.escape(old_experiment_name)}-*.ckpt"
                )
            )
            if path.rpartition("-")[2].endswith(old_experiment_name)
        ]
        print(f"Found {len(checkpoint_paths)} checkpoint files")
        for path in checkpoint_paths:
            # Get new checkpoint path
            prefix = os.path.join(paths.checkpoint_dir, f"{old_experiment_name}-")
            suffix = ".ckpt"
            assert path.startswith(prefix)
            assert path.endswith(suffix)
            label = path[len(prefix) : -len(suffix)]
            new_path = os.path.join(
                paths.checkpoint_dir, f"{new_experiment_name}-{label}.ckpt"
            )

            # Move checkpoint
            print(f"> Moving {path} to {new_path}")
            os.rename(path, new_path)

        # Move metadata
        metadata_path = os.path.join(paths.metadata_dir, f"{old_experiment_name}.yaml")
        if os.path.exists(metadata_path):
            new_path = os.path.join(paths.metadata_dir, f"{new_experiment_name}.yaml")
            print(f"Moving {metadata_path} to {new_path}")
            os.rename(metadata_path, new_path)
        else:
            print("No metadata found")

        # Move logs
        metadata_path = os.path.join(paths.log_dir, f"{old_experiment_name}")
        if os.path.exists(metadata_path):
            new_path = os.path.join(paths.log_dir, f"{new_experiment_name}")
            print(f"Moving {metadata_path} to {new_path}")
            os.rename(metadata_path, new_path)
        else:
            print("No logs found")
