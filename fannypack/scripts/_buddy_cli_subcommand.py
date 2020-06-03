import abc
import argparse


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
