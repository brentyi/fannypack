import argparse
import glob
import os
import shutil

import argcomplete
import beautifultable
import termcolor

import fannypack

from ._buddy_cli_subcommand import Subcommand
from ._buddy_cli_utils import BuddyPaths, find_checkpoints, find_experiments


def _get_size(path):
    if os.path.isfile(path):
        return os.stat(path).st_size
    elif os.path.isdir(path):
        return sum(
            [
                os.stat(p).st_size
                for p in glob.glob(os.path.join(path, "**/*"), recursive=True)
            ]
        )
    else:
        return 0


def _format_size(size):
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    unit_index = 0
    while size > 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f}{units[unit_index]}"


class InfoSubcommand(Subcommand):
    """Print info about a Buddy experiment: checkpoints, metadata, etc.
    """

    subcommand: str = "info"

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

    @classmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        # Get experiment name
        experiment_name = args.experiment_name
        print(experiment_name)

        # Generate dynamic-width table
        try:
            terminal_columns = int(os.popen("stty size", "r").read().split()[1])
        except IndexError:
            # stty size fails when run from outside proper terminal (eg in tests)
            terminal_columns = 100
        table = beautifultable.BeautifulTable(
            max_width=min(100, terminal_columns),
            default_alignment=beautifultable.ALIGN_LEFT,
        )
        table.set_style(beautifultable.STYLE_BOX_ROUNDED)

        def add_table_row(label, value):
            table.append_row([termcolor.colored(label, attrs=["bold"]), value])

        # Constant for "not applicable" fields
        NA = termcolor.colored("N/A", "red")

        # Find checkpoint files
        checkpoint_paths = find_checkpoints(experiment_name, paths.checkpoint_dir)

        # Display size, labels of checkpoints
        if len(checkpoint_paths) > 0:
            checkpoint_total_size = 0
            checkpoint_labels = []
            buddy = fannypack.utils.Buddy(experiment_name, verbose=False)
            checkpoint_paths, steps = buddy._find_checkpoints(
                paths.checkpoint_dir, args.experiment_name
            )
            for checkpoint_path in checkpoint_paths:
                prefix = os.path.join(paths.checkpoint_dir, f"{experiment_name}-")
                suffix = ".ckpt"
                assert checkpoint_path.startswith(prefix)
                assert checkpoint_path.endswith(suffix)
                label = checkpoint_path[len(prefix) : -len(suffix)]

                checkpoint_labels.append(f"{label} (steps: {steps[checkpoint_path]})")
                checkpoint_total_size += _get_size(checkpoint_path)

            add_table_row("Total checkpoint size", _format_size(checkpoint_total_size))
            add_table_row(
                "Average checkpoint size",
                _format_size(checkpoint_total_size / len(checkpoint_paths)),
            )
            add_table_row("Checkpoint labels", "\n".join(checkpoint_labels))
        else:
            add_table_row("Total checkpoint size", NA)
            add_table_row("Average checkpoint size", NA)
            add_table_row("Checkpoint labels", "")

        # Display log file size
        log_path = os.path.join(paths.log_dir, f"{experiment_name}")
        if os.path.exists(log_path):
            #  _delete(log_path, args.forever)
            add_table_row("Log size", _format_size(_get_size(log_path)))
        else:
            add_table_row("Log size", NA)

        # Display metadata + metadata size
        metadata_path = os.path.join(paths.metadata_dir, f"{experiment_name}.yaml")
        if os.path.exists(metadata_path):
            add_table_row("Metadata size", _format_size(_get_size(metadata_path)))
            with open(metadata_path, "r") as f:
                add_table_row("Metadata", f.read().strip())
        else:
            add_table_row("Metadata size", NA)
            add_table_row("Metadata", NA)

        # Print table
        print(table)
