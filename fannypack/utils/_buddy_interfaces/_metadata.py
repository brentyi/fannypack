import yaml
import os


class _BuddyMetadata:
    """Buddy's experiment metadata management interface.
    """

    def __init__(self, metadata_dir):
        """Metadata-specific setup.
        """

        # Attempt to read existing metadata
        self._metadata_dir = metadata_dir
        try:
            self.load_metadata()
        except FileNotFoundError:
            self._metadata = {}

    def load_metadata(self, experiment_name=None, metadata_dir=None, path=None):
        """Read existing metadata file.
        """
        if path is None:
            if experiment_name is None:
                experiment_name = self._experiment_name
            if metadata_dir is None:
                metadata_dir = self._metadata_dir
            path = "{}/{}.yaml".format(metadata_dir, experiment_name)
        else:
            assert experiment_name is None and metadata_dir is None

        with open(path, "r") as file:
            self._metadata = yaml.load(file, Loader=yaml.SafeLoader)
            self._print("Loaded metadata:", self._metadata)

    def add_metadata(self, content):
        """Add human-readable metadata for this experiment. Input should be a
        dictionary.
        """
        assert type(content) == dict

        # Merge input metadata with current metadata
        for key, value in content.items():
            self._metadata[key] = value

        # Create metadata directory if needed
        if not os.path.isdir(self._metadata_dir):
            os.makedirs(self._metadata_dir)
            self._print("Created directory:", self._metadata_dir)

        # Write metadata to file
        metadata_path = "{}/{}.yaml".format(
            self._metadata_dir, self._experiment_name,
        )
        with open(metadata_path, "w") as file:
            yaml.dump(self._metadata, file, default_flow_style=False)
            self._print("Wrote metadata to:", metadata_path)

    @property
    def metadata(self):
        """Read-only interface for experiment metadata.
        """
        return self._metadata
