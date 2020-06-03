import argparse
import glob
import os
import shutil

import prettytable

from ._buddy_cli_subcommand import Subcommand


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
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "experiment_name",
            type=str,
            help="Name of experiment, as printed by `$ buddy list`.",
        )

    @classmethod
    def main(cls, args: argparse.Namespace) -> None:
        # Get experiment name
        experiment_name = args.experiment_name

        # Set up & format table
        table = prettytable.PrettyTable(field_names=["Name", "Value"])

        table.align = "l"
        table.header = False
        table.hrules = prettytable.ALL
        table.right_padding_width = 3

        table.horizontal_char = "─"
        table.vertical_char = "│"
        table.junction_char = "┼"

        # Find checkpoint files
        checkpoint_paths = glob.glob(
            os.path.join(args.checkpoint_dir, f"{glob.escape(experiment_name)}-*.ckpt")
        )

        # Display size, labels of checkpoints
        if len(checkpoint_paths) > 0:
            checkpoint_total_size = 0
            checkpoint_labels = []
            for path in checkpoint_paths:
                prefix = os.path.join(args.checkpoint_dir, f"{experiment_name}-")
                suffix = ".ckpt"
                assert path.startswith(prefix)
                assert path.endswith(suffix)
                label = path[len(prefix) : -len(suffix)]
                checkpoint_labels.append(label)
                checkpoint_total_size += _get_size(path)

            table.add_row(
                ["Total checkpoint size", _format_size(checkpoint_total_size)]
            )
            table.add_row(
                [
                    "Average checkpoint size",
                    _format_size(checkpoint_total_size / len(checkpoint_paths)),
                ]
            )
            table.add_row(["Checkpoint labels", "\n".join(checkpoint_labels)])
        else:
            table.add_row(["Total checkpoint size", "N/A"])
            table.add_row(["Average checkpoint size", "N/A"])
            table.add_row(["Checkpoint labels", ""])

        # Display log file size
        log_path = os.path.join(args.log_dir, f"{experiment_name}")
        if os.path.exists(log_path):
            #  _delete(log_path, args.forever)
            table.add_row(["Log size", _format_size(_get_size(log_path))])
        else:
            table.add_row(["Log size", "N/A"])

        # Display metadata + metadata size
        metadata_path = os.path.join(args.metadata_dir, f"{experiment_name}.yaml")
        if os.path.exists(metadata_path):
            table.add_row(["Metadata size", _format_size(_get_size(metadata_path))])
            with open(metadata_path, "r") as f:
                table.add_row(["Metadata", f.read().strip()])
        else:
            table.add_row(["Metadata size", "N/A"])
            table.add_row(["Metadata", "N/A"])

        print(experiment_name)
        print(table)
