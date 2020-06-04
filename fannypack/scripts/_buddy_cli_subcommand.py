import abc
import argparse
from dataclasses import dataclass


@dataclass
class BuddyPaths:
    checkpoint_dir: str
    log_dir: str
    metadata_dir: str


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
