import argparse
import datetime
import os

import beautifultable
import termcolor

from ._buddy_cli_subcommand import Subcommand
from ._buddy_cli_utils import BuddyPaths, find_experiments


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
