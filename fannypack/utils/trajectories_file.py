"""
Helper for logging observations & writing them to an hdf5 file.
"""

import numpy as np
import h5py


class TrajectoriesFile:
    def __init__(self, path, convert_doubles=True,
                 compress=True, diagnostics=False):
        """Constructor for the TrajectoriesFile class, which provides a simple
        interface for reading from/writing to hdf5 files.

        Args:
            path (str): file path for this trajectory file
            convert_doubles (bool): automatically convert doubles to floats
            compress (bool): enable gzip compression for hdf5 files
        """
        assert path[-5:] == ".hdf5", "Missing file extension!"

        # Meta
        self._path = path
        self._convert_doubles = convert_doubles
        self._compress = compress
        self._diagnostics = diagnostics

        # Maps observation key => observation list
        self._obs_dict = {}

        # Count the number of trajectories that already exist
        self._trajectory_prefix = "trajectory"
        with self._h5py_file() as f:
            self._print("Loading trajectory from file:", f)
            if len(f.keys()) > 0:
                prefix_length = len(self._trajectory_prefix)
                ids = [int(k[prefix_length:]) for k in f.keys()]
                self._trajectory_count = max(ids) + 1
            else:
                self._trajectory_count = 0

            self._print("Existing trajectory count:", self._trajectory_count)

        assert type(self._trajectory_count) == int

        self._file = None

    def _print(self, *args, **kwargs):
        if self._diagnostics:
            assert type(args[0]) == str
            args = list(args)
            args[0] = f"[TrajectoriesFile: {self._path}] {args[0]}"
            print(*args, **kwargs)

    def __enter__(self):
        """Automatic file opening, for use in `with` statements.
        """
        if self._file is None:
            self._print("Opening file...")
            self._file = self._h5py_file()
        return self

    def __exit__(self, *unused):
        """Automatic file closing, for use in `with` statements.
        """
        if self._file is not None:
            self._print("Closing file...")
            self._file.close()
            self._file = None

    def __getitem__(self, index):
        """Accessor for individual trajectories held by this file.
        """
        assert self._file is not None, "Not called in with statement!"

        # Check that the index is sane
        assert type(index) == int
        if index >= len(self):
            # For use as a standard Python iterator
            raise IndexError

        traj_key = self._trajectory_prefix + str(index)
        assert traj_key in self._file.keys()

        # Copy values to numpy array
        output = {}
        for key, value in self._file[traj_key].items():
            output[key] = value[:]
            assert type(output[key]) == np.ndarray

        return output

    def __len__(self):
        """Returns the number of recorded trajectories.
        """
        return self._trajectory_count

    def add_timestep(self, obs):
        """Add a timestep to the current trajectory.

        Args:
            obs (dict): map from observation names (str) to values (np.ndarray)
        """
        for key, value in obs.items():
            if key not in self._obs_dict:
                self._obs_dict[key] = []

            assert type(self._obs_dict[key]) == list
            self._obs_dict[key].append(np.copy(value))

    def clear_trajectory(self):
        """Abandon the current trajectory.
        """
        self._print("Clearing trajectory!")
        self._obs_dict = {}

    def end_trajectory(self):
        """Write the current trajectory to disk, and mark the start of a new
        trajectory.

        The next call to `add_timestep()` will be time 0 of the next trajectory.
        """
        assert self._file is not None, "Not called in with statement!"

        length = len(list(self._obs_dict.values())[0])
        self._print(f"Ending trajectory! (length={length})")

        # Put all pushed observations into a new group
        trajectory_name = self._trajectory_prefix + \
            str(self._trajectory_count)
        group = self._file.create_group(trajectory_name)
        for key, obs_list in self._obs_dict.items():
            # Convert list of observations to a numpy array
            data = np.array(obs_list)

            # Compress floats
            if data.dtype == np.float64 and self._convert_doubles:
                data = data.astype(np.float32)

            if self._compress:
                group.create_dataset(
                    key, data=data, chunks=True, compression="gzip")
            else:
                group.create_dataset(key, data=data, chunks=True)

        self._obs_dict = {}
        self._trajectory_count += 1

        self._print("Existing trajectory count:", self._trajectory_count)

    def reencode(self, target_path):
        """Re-encode contents into a new hdf5 file.

        Mostly used for re-encoding trajectory files generated with older
        versions of this class.

        Returns:
            new TrajectoriesFile
        """
        source = self._h5py_file()
        target = TrajectoriesFile(target_path)
        with source, target:
            for name, trajectory in source.items():
                keys = trajectory.keys()
                for obs_step in zip(*trajectory.values()):
                    target.add_timestep(dict(zip(keys, obs_step)))
                target.end_trajectory()
                print("Wrote ", name)
        return target

    def _h5py_file(self, mode='a'):
        """Private helper for creating h5py file objects.
        """
        return h5py.File(self._path, mode)
