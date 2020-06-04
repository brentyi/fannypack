import argparse
import datetime
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import beautifultable
import termcolor

from ._buddy_cli_subcommand import BuddyPaths, Subcommand


@dataclass
class FindOutput:
    experiment_names: Set[str]
    checkpoint_counts: Dict[str, int]
    log_experiments: Set[str]
    metadata_experiments: Set[str]
    timestamps: Dict[str, float]


def find_experiments(paths: BuddyPaths, verbose: bool = False) -> FindOutput:
    """Helper for listing experiments
    """

    def _print(*args, **kwargs):
        if not verbose:
            return
        print(*args, **kwargs)

    def _listdir(path: str) -> List[str]:
        """Helper for listing files in a directory
        """
        try:
            return os.listdir(path)
        except FileNotFoundError:
            return []

    # Last modified: checkpoints and metadata only
    # > We could also do logs, but seems high effort?
    timestamps: Dict[str, float] = {}

    # Count checkpoints for each experiment
    checkpoint_counts: Dict[str, int] = {}
    for file in _listdir(paths.checkpoint_dir):
        # Remove .ckpt suffix
        if file[-5:] != ".ckpt":
            _print(f"Skipping malformed checkpoint filename: {file}")
            continue
        trimmed = file[:-5]

        # Get experiment name
        parts = trimmed.split("-")
        if len(parts) != 2:
            _print(f"Skipping malformed checkpoint filename: {file}")
            continue
        name = parts[0]

        # Update tracker
        if name not in checkpoint_counts.keys():
            checkpoint_counts[name] = 0
        checkpoint_counts[name] += 1

        # Update timestamp
        mtime = os.path.getmtime(os.path.join(paths.checkpoint_dir, file))
        if name not in timestamps.keys() or mtime > timestamps[name]:
            timestamps[name] = mtime

    # Get experiment names from metadata files
    metadata_experiments = set()
    for file in _listdir(paths.metadata_dir):
        # Remove .yaml suffix
        if file[-5:] != ".yaml":
            _print(f"Skipping malformed metadata filename: {file}")
            continue
        name = file[:-5]
        metadata_experiments.add(name)

        # Update timestamp
        mtime = os.path.getmtime(os.path.join(paths.metadata_dir, file))
        if name not in timestamps.keys() or mtime > timestamps[name]:
            timestamps[name] = mtime

    # Get experiment names from log directories
    log_experiments = set(_listdir(paths.log_dir))

    # Get all experiments
    experiment_names = (
        set(checkpoint_counts.keys()) | log_experiments | metadata_experiments
    )

    return FindOutput(
        experiment_names=experiment_names,
        checkpoint_counts=checkpoint_counts,
        log_experiments=log_experiments,
        metadata_experiments=metadata_experiments,
        timestamps=timestamps,
    )


class ListSubcommand(Subcommand):
    """Get & summarize existing Buddy experiments.
    """

    subcommand: str = "list"

    @classmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        # No arguments
        pass

    @classmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        results = find_experiments(paths, verbose=True)

        # Generate dynamic-width table
        try:
            terminal_columns = int(os.popen("stty size", "r").read().split()[1])
        except IndexError:
            # stty size fails when run from outside proper terminal (eg in tests)
            terminal_columns = 100
        table = beautifultable.BeautifulTable(max_width=min(100, terminal_columns))
        table.set_style(beautifultable.STYLE_BOX_ROUNDED)
        table.row_separator_char = ""

        # Add bolded headers
        column_headers = [
            "Name",
            "Checkpoints",
            "Logs",
            "Metadata",
            "Last Modified",
        ]
        table.column_headers = [
            termcolor.colored(h, attrs=["bold"]) for h in column_headers
        ]

        for name in results.experiment_names:
            # Get checkpoint count
            checkpoint_count = 0
            if name in results.checkpoint_counts:
                checkpoint_count = results.checkpoint_counts[name]

            # Get timestamp
            timestamp = ""
            if name in results.timestamps:
                timestamp = datetime.datetime.fromtimestamp(
                    results.timestamps[name]
                ).strftime(
                    "%b %d, %Y @ %-H:%M" if terminal_columns > 100 else "%Y-%m-%d"
                )

            # Add row for experiment
            yes_no = {
                True: termcolor.colored("Yes", "green"),
                False: termcolor.colored("No", "red"),
            }
            table.append_row(
                [
                    name,
                    checkpoint_count,
                    yes_no[name in results.log_experiments],
                    yes_no[name in results.metadata_experiments],
                    timestamp,
                ]
            )

        # Print table, sorted by name
        print(f"Found {len(results.experiment_names)} experiments!")
        table.sort(table.column_headers[0])
        print(table)
