import yaml
import os


class _BuddyMetadata:
    """Private mixin for encapsulating experiment metadata management.
    """

    def __init__(self):
        """Metadata-specific setup.
        """
        super().__init__()

        # Attempt to read existing metadata
        metadata_path = "{}/{}.yaml".format(
            self._config['metadata_dir'],
            self._experiment_name)
        try:
            with open(metadata_path, "r") as file:
                self._metadata = yaml.load(file, Loader=yaml.SafeLoader)
                self._print("Loaded metadata:", self._metadata)
        except FileNotFoundError:
            self._metadata = {}

    def add_metadata(self, content):
        """Add human-readable metadata for this experiment. Input should be a
        dictionary.
        """
        assert type(content) == dict

        # Merge input metadata with current metadata
        for key, value in content.items():
            self._metadata[key] = value

        # Create metadata directory if needed
        if not os.path.isdir(self._config['metadata_dir']):
            os.makedirs(self._config['metadata_dir'])
            self._print("Created directory:", self._config['metadata_dir'])

        # Write metadata to file
        metadata_path = "{}/{}.yaml".format(
            self._config['metadata_dir'],
            self._experiment_name)
        with open(metadata_path, "w") as file:
            yaml.dump(
                self._metadata,
                file,
                default_flow_style=False
            )
            self._print("Wrote metadata to:", metadata_path)

    @property
    def metadata():
        """Read-only interface for experiment metadata.
        """
        return self._metadata
