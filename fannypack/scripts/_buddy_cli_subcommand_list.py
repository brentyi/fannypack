import argparse
import datetime
import os
from typing import Dict, List

import prettytable

from ._buddy_cli_subcommand import Subcommand


def _listdir(path: str) -> List[str]:
    """Helper for listing files in a directory
    """
    try:
        return os.listdir(path)
    except FileNotFoundError:
        print(f"Couldn't find {path} -- skipping")
        return []


class ListSubcommand(Subcommand):
    """Get & summarize existing Buddy experiments.
    """

    subcommand: str = "list"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        # No arguments
        pass

    @classmethod
    def main(cls, args: argparse.Namespace) -> None:
        # Last modified: checkpoints and metadata only
        # > We could also do logs, but seems high effort?
        timestamps: Dict[str, float] = {}

        # Count checkpoints for each experiment
        checkpoint_counts: Dict[str, int] = {}
        for file in _listdir(args.checkpoint_dir):
            # Remove .ckpt suffix
            if file[-5:] != ".ckpt":
                print(f"Skipping malformed checkpoint filename: {file}")
                continue
            trimmed = file[:-5]

            # Get experiment name
            parts = trimmed.split("-")
            if len(parts) != 2:
                print(f"Skipping malformed checkpoint filename: {file}")
                continue
            name = parts[0]

            # Update tracker
            if name not in checkpoint_counts.keys():
                checkpoint_counts[name] = 0
            checkpoint_counts[name] += 1

            # Update timestamp
            mtime = os.path.getmtime(os.path.join(args.checkpoint_dir, file))
            if name not in timestamps.keys() or mtime > timestamps[name]:
                timestamps[name] = mtime

        # Get experiment names from metadata files
        metadata_experiments = set()
        for file in _listdir(args.metadata_dir):
            # Remove .yaml suffix
            if file[-5:] != ".yaml":
                print(f"Skipping malformed metadata filename: {file}")
                continue
            name = file[:-5]
            metadata_experiments.add(name)

            # Update timestamp
            mtime = os.path.getmtime(os.path.join(args.metadata_dir, file))
            if name not in timestamps.keys() or mtime > timestamps[name]:
                timestamps[name] = mtime

        # Get experiment names from log directories
        log_experiments = set(_listdir(args.log_dir))

        # Generate table
        experiment_names = (
            set(checkpoint_counts.keys()) | log_experiments | metadata_experiments
        )
        table = prettytable.PrettyTable(
            field_names=["Name", "Checkpoints", "Metadata", "Logs", "Last Modified"]
        )

        table.horizontal_char = "─"
        table.vertical_char = "│"
        table.junction_char = "┼"

        table.sortby = "Name"

        for name in experiment_names:
            # Get checkpoint count
            checkpoint_count = 0
            if name in checkpoint_counts:
                checkpoint_count = checkpoint_counts[name]

            # Get timestamp
            timestamp = ""
            if name in timestamps:
                timestamp = datetime.datetime.fromtimestamp(timestamps[name]).strftime(
                    "%B %d, %Y @ %-H:%M:%S"
                )

            # Add row for experiment
            yes_no = {True: "Yes", False: ""}
            table.add_row(
                [
                    name,
                    checkpoint_count,
                    yes_no[name in metadata_experiments],
                    yes_no[name in log_experiments],
                    timestamp,
                ]
            )

        # Print table
        print(f"Found {len(experiment_names)} experiments!")
        print(table)
