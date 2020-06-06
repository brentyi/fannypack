import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, cast


def find_checkpoints(experiment_name: str, path: str) -> List[str]:
    """Finds checkpoints associated with an experiment.
    """

    # Glob for checkpoint files
    checkpoint_files = glob.glob(
        os.path.join(path, f"{glob.escape(experiment_name)}-*.ckpt")
    )

    # Filter further with rpartition (for handling experiment names with hyphens)
    return list(filter(
        lambda path: path.rpartition("-")[0].endswith(experiment_name), checkpoint_files
    ))


@dataclass
class BuddyPaths:
    """Dataclass for storing paths to experiment files.
    """

    checkpoint_dir: str
    log_dir: str
    metadata_dir: str


def _listdir(path: str) -> List[str]:
    """Helper for listing files in a directory
    """
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return []


@dataclass
class FindOutput:
    """Output of `find_experiments(...)`.
    """

    experiment_names: Set[str]
    checkpoint_counts: Dict[str, int]
    log_experiments: Set[str]
    metadata_experiments: Set[str]
    timestamps: Dict[str, float]


find_output_cache: Optional[FindOutput] = None


def find_experiments(paths: BuddyPaths, verbose: bool = False) -> FindOutput:
    """Helper for listing experiments
    """

    # Return cached results
    global find_output_cache
    if find_output_cache is not None:
        return cast(FindOutput, find_output_cache)

    # Print helper
    def _print(*args, **kwargs):
        if not verbose:
            return
        print(*args, **kwargs)

    # Last modified: checkpoints and metadata only
    # > We could also do logs, but seems high effort?
    timestamps: Dict[str, float] = {}

    # Count checkpoints for each experiment
    checkpoint_counts: Dict[str, int] = {}
    for file in _listdir(paths.checkpoint_dir):
        # Remove .ckpt suffix
        if file[-5:] != ".ckpt":
            _print(f"Skipping malformed checkpoint filename: {file}")
            continue
        trimmed = file[:-5]

        # Get experiment name
        name, hyphen, _label = trimmed.rpartition("-")
        if hyphen != "-":
            _print(f"Skipping malformed checkpoint filename: {file}")
            continue

        # Update tracker
        if name not in checkpoint_counts.keys():
            checkpoint_counts[name] = 0
        checkpoint_counts[name] += 1

        # Update timestamp
        mtime = os.path.getmtime(os.path.join(paths.checkpoint_dir, file))
        if name not in timestamps.keys() or mtime > timestamps[name]:
            timestamps[name] = mtime

    # Get experiment names from metadata files
    metadata_experiments = set()
    for file in _listdir(paths.metadata_dir):
        # Remove .yaml suffix
        if file[-5:] != ".yaml":
            _print(f"Skipping malformed metadata filename: {file}")
            continue
        name = file[:-5]
        metadata_experiments.add(name)

        # Update timestamp
        mtime = os.path.getmtime(os.path.join(paths.metadata_dir, file))
        if name not in timestamps.keys() or mtime > timestamps[name]:
            timestamps[name] = mtime

    # Get experiment names from log directories
    log_experiments = set(_listdir(paths.log_dir))

    # Get all experiments
    experiment_names = (
        set(checkpoint_counts.keys()) | log_experiments | metadata_experiments
    )

    # Update global find_output cache
    find_output_cache = FindOutput(
        experiment_names=experiment_names,
        checkpoint_counts=checkpoint_counts,
        log_experiments=log_experiments,
        metadata_experiments=metadata_experiments,
        timestamps=timestamps,
    )
    return find_output_cache
