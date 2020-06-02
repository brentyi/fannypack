"""CLI interface for experiment management.

To print options, install `fannypack` via pip and run:
```
$ buddy --help
```
"""

import abc
import argparse
import datetime
import os
from typing import Dict

import prettytable


def listdir(path: str):
    """Helper for listing files in a directory
    """
    try:
        return os.listdir(path)
    except FileNotFoundError:
        print(f"Couldn't find {path} -- skipping")
        return []


def main():
    parser = argparse.ArgumentParser(
        prog="buddy",
        description="CLI interface for Buddy, a tool for managing PyTorch experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shared configuration flags
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/",
        help="Path to checkpoints; should match Buddy's `checkpoint_dir` argument.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="metadata/",
        help="Path to metadata files; should match Buddy's `metadata_dir` argument.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/",
        help="Path to Tensorboard logs; should match Buddy's `log_dir` argument.",
    )

    # Separate parsers for subcommands
    subparsers = parser.add_subparsers(required=True, dest="subcommand", help="blah")

    # Add subcommands
    subcommand_types = [ListSubcommand]
    subcommand_map = {}
    for S in subcommand_types:
        subparser = subparsers.add_parser(S.subcommand, help=S.__doc__)
        S.add_arguments(subparser)
        subcommand_map[S.subcommand] = S
    args = parser.parse_args()

    # Run subcommand
    subcommand_map[args.subcommand].main(args)


class Subcommand(abc.ABC):
    """Subcommand interface: defines arguments, runtime routine.
    """

    subcommand: str
    helptext: str

    @abc.abstractclassmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        raise NotImplementedError

    @abc.abstractclassmethod
    def main(cls, args: argparse.Namespace):
        raise NotImplementedError


class ListSubcommand:
    """Get & summarize existing Buddy experiments.
    """

    subcommand: str = "list"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        # No arguments
        pass

    @classmethod
    def main(cls, args: argparse.Namespace):
        # Last modified: checkpoints and metadata only
        # > We could also do logs, but seems high effort?
        timestamps: Dict[str, float] = {}

        # Count checkpoints for each experiment
        checkpoint_counts: Dict[str, int] = {}
        for file in listdir(args.checkpoint_dir):
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
        for file in listdir(args.metadata_dir):
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
        log_experiments = set(listdir(args.log_dir))

        # Generate table
        experiment_names = (
            set(checkpoint_counts.keys()) | log_experiments | metadata_experiments
        )
        table = prettytable.PrettyTable(
            field_names=["Name", "Checkpoints", "Metadata", "Logs", "Last Modified"]
        )
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
                    "%B %d, %Y @ %-H:%M:%S "
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
