import abc
import argparse


class Subcommand(abc.ABC):
    """Subcommand interface: defines arguments, runtime routine.
    """

    subcommand: str
    helptext: str

    @classmethod
    @abc.abstractclassmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @classmethod
    @abc.abstractclassmethod
    def main(cls, args: argparse.Namespace) -> None:
        raise NotImplementedError
