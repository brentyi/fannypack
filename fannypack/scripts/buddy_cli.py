"""CLI interface for experiment management.

To print options, install `fannypack` via pip and run:
```
$ buddy --help
```
"""

import argparse
from typing import Dict, List

from ._buddy_cli_subcommand import Subcommand
from ._buddy_cli_subcommand_delete import DeleteSubcommand
from ._buddy_cli_subcommand_list import ListSubcommand
from ._buddy_cli_subcommand_rename import RenameSubcommand


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
    subparsers = parser.add_subparsers(
        required=True,
        dest="subcommand",
        help="Get help by running `$ buddy {subcommand} --help`.",
    )

    # Add subcommands
    subcommand_types: List[Callable[..., Subcommand]] = [
        ListSubcommand,
        RenameSubcommand,
        DeleteSubcommand,
    ]
    subcommand_map: Dict[str, Callable] = {}
    for S in subcommand_types:
        subparser = subparsers.add_parser(
            S.subcommand, help=S.__doc__, description=S.__doc__
        )
        S.add_arguments(subparser)
        subcommand_map[S.subcommand] = S
    args = parser.parse_args()

    # Run subcommand
    subcommand_map[args.subcommand].main(args)


if __name__ == "__main__":
    main()
