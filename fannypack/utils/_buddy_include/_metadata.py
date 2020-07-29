import abc
import os
from typing import TYPE_CHECKING, Any, Dict, cast

import yaml

if TYPE_CHECKING:
    from .._buddy import Buddy


class _BuddyMetadata(abc.ABC):
    """Buddy's experiment metadata management interface.
    """

    def __init__(self, metadata_dir: str) -> None:
        """Metadata-specific setup.
        """

        # Attempt to read existing metadata
        self._metadata_dir = metadata_dir
        try:
            self.load_metadata(_write=True)
        except FileNotFoundError:
            self._metadata: Dict[str, Any] = {}

    def load_metadata(
        self,
        experiment_name: str = None,
        metadata_dir: str = None,
        path: str = None,
        _write=True,
    ) -> None:
        """Read existing metadata file. Note that metadata is loaded automatically: this
        only needs to be called if loading across experiments.

        Overwrites existing metadata.
        """
        if path is None:
            if experiment_name is None:
                experiment_name = cast("Buddy", self)._experiment_name
            if metadata_dir is None:
                metadata_dir = self._metadata_dir
            path = os.path.join(metadata_dir, f"{experiment_name}.yaml")
        else:
            assert experiment_name is None and metadata_dir is None

        # Read from disk
        with open(path, "r") as file:
            self._metadata = yaml.load(file, Loader=yaml.Loader)
            cast("Buddy", self)._print("Loaded metadata:", self._metadata)

        # Write to disk
        if _write:
            self._write_metadata()

    def add_metadata(self, content: Dict[str, Any]) -> None:
        """Add human-readable metadata for this experiment. Input should be a
        dictionary that is merged with existing metadata.
        """
        assert type(content) == dict

        # Merge input metadata with current metadata
        for key, value in content.items():
            self._metadata[key] = value

        # Write to disk
        self._write_metadata()

    def set_metadata(self, content: Dict[str, Any]) -> None:
        """Assign human-readable metadata for this experiment. Input should be
        a dictionary that replaces existing metadata.
        """
        assert type(content) == dict

        # Replace input metadata with current metadata
        self._metadata = content

        # Write to disk
        self._write_metadata()

    def _write_metadata(self) -> None:
        # Create metadata directory if needed
        if not os.path.isdir(self._metadata_dir):
            os.makedirs(self._metadata_dir)
            cast("Buddy", self)._print("Created directory:", self._metadata_dir)

        # Write metadata to file
        metadata_path = "{}/{}.yaml".format(
            self._metadata_dir, cast("Buddy", self)._experiment_name,
        )
        with open(metadata_path, "w") as file:
            yaml.dump(self._metadata, file, default_flow_style=False)
            cast("Buddy", self)._print("Wrote metadata to:", metadata_path)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Read-only interface for experiment metadata.
        """
        return self._metadata

    @property
    def metadata_path(self) -> str:
        """Read-only path to my metadata file.
        """
        return os.path.join(
            self._metadata_dir, f"{cast('Buddy', self)._experiment_name}.yaml"
        )
