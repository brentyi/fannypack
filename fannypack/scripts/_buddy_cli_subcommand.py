import abc
import argparse

from ._buddy_cli_utils import BuddyPaths


class Subcommand(abc.ABC):
    """Subcommand interface: defines arguments, runtime routine.
    """

    subcommand: str
    helptext: str

    @classmethod
    @abc.abstractclassmethod
    def add_arguments(
        cls, *, parser: argparse.ArgumentParser, paths: BuddyPaths
    ) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractclassmethod
    def main(cls, *, args: argparse.Namespace, paths: BuddyPaths) -> None:
        raise NotImplementedError
