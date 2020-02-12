import numpy as np
import h5py


class TrajectoriesFile:
    """An interface for reading/writing trajectory files using hdf5 files.

    Each TrajectoriesFile represents an iterable list of trajectories, where
    each trajectory is stored as a dictionary that maps `str` keys to
    `np.ndarray` contents. The first dimension of each n-dimensional content
    array is always time.

    Example usage (read):
    ```
    with TrajectoriesFile('test.hdf5') as traj_file:

        for traj in traj_file:
            print(traj.keys()) # list of keys
            print(traj['some-key-name']) # numpy array
    ```

    Example usage (write):
    ```
    traj_file = TrajectoriesFile('test.hdf5', read_only=False)

    traj_file.add_timestep({'a': 1, 'b': 2})
    traj_file.add_timestep({'a': 3, 'b': 4})

    with traj_file:
        traj_file.end_trajectory()

    print(len(traj_file)) # 1 trajectory!

    with traj_file:
        print(traj_file[0]['a']) # [1, 3]
        print(traj_file[0]['b']) # [2, 4]
    ```

    Note that some operations -- ones that require interfacing with the
    filesytem -- need to be called within a `with` statement.

    """

    def __init__(self, path, convert_doubles=True, read_only=True,
                 compress=True, verbose=False):
        """Constructs an interface for reading from/writing to hdf5 files.

        Args:
            path (str): File path for this trajectory file.
            convert_doubles (bool): Convert doubles to floats to shrink files.
            read_only (bool, optional): Open file in read-only mode.
            compress (bool, optional): Reduce filesize w/ gzip.
            verbose (bool, optional): Enable debug prints.
        """
        assert path[-5:] == ".hdf5", "Missing file extension!"

        # Meta
        self._path = path
        self._convert_doubles = convert_doubles
        self._read_only = read_only
        self._compress = compress
        self._verbose = verbose

        # Maps content key => content list
        self._content_dict = {}

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
        Must be called with the TrajectoriesFile object in a `with` statement.

        Args:
            index (int): Trajectory #.

        Returns:
            dict: A (str->np.ndarray) map containing data collected at each
                timestep of our trajectory.
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

    def __setitem__(self, index, item):
        """Assignment operation for modifying or mutating trajectories.
        Must be called with the TrajectoriesFile object in a `with` statement.

        Args:
            index (int): Trajectory #.
            item (dict): A (str->np.ndarray) map, as would be returned by
                __getitem__().
        """
        assert self._file is not None, "Not called in with statement!"

        # Check that the inputs are sane
        assert type(index) == int
        assert type(item) == dict

        if index >= len(self):
            raise IndexError

        traj_key = self._trajectory_prefix + str(index)
        for key, value in item.items():
            self._file[traj_key][key][...] = value

    def __len__(self):
        """Returns the number of recorded trajectories.
        """
        return self._trajectory_count

    def add_timestep(self, content):
        """Add a timestep to the current trajectory.

        Args:
            content (dict): Map from timestep keys (str) to values (np.ndarray).
        """
        for key, value in content.items():
            if key not in self._content_dict:
                self._content_dict[key] = []

            assert type(self._content_dict[key]) == list
            self._content_dict[key].append(np.copy(value))

    def clear_trajectory(self):
        """Abandon the current trajectory.
        """
        self._print("Clearing trajectory!")
        self._content_dict = {}

    def end_trajectory(self):
        """Write the current trajectory to disk, and mark the start of a new
        trajectory.
        Must be called with the TrajectoriesFile object in a `with` statement.

        The next call to `add_timestep()` will be time 0 of the next trajectory.
        """
        assert self._file is not None, "Not called in with statement!"

        if not self._content_dict:
            self._print("Empty observation dictionary; skipping trajectory end")
            return

        length = len(list(self._content_dict.values())[0])

        self._print(f"Ending trajectory! (length={length})")

        # Put all pushed contents into a new group
        trajectory_name = self._trajectory_prefix + \
            str(self._trajectory_count)
        group = self._file.create_group(trajectory_name)
        for key, content_list in self._content_dict.items():
            # Convert list of contents to a numpy array
            data = np.array(content_list)

            # Compress floats
            if data.dtype == np.float64 and self._convert_doubles:
                data = data.astype(np.float32)

            if self._compress:
                group.create_dataset(
                    key, data=data, chunks=True, compression="gzip")
            else:
                group.create_dataset(key, data=data, chunks=True)

        self._content_dict = {}
        self._trajectory_count += 1

        self._print("Existing trajectory count:", self._trajectory_count)

    def reencode(self, target_path):
        """Re-encode contents into a new hdf5 file.

        Mostly used for re-encoding trajectory files generated with older
        versions of this class.

        Must be called with the TrajectoriesFile object in a `with` statement.

        Args:
            target_path (str): Destination to write contents to.

        Returns:
            TrajectoriesFile: New file object.
        """
        assert self._file is not None, "Not called in with statement!"

        source = self._file
        target = TrajectoriesFile(target_path)
        with source, target:
            for name, trajectory in source.items():
                keys = trajectory.keys()
                for content_step in zip(*trajectory.values()):
                    target.add_timestep(dict(zip(keys, content_step)))
                target.end_trajectory()
                print("Wrote ", name)
        return target

    def _h5py_file(self, mode=None):
        """Private helper for creating h5py file objects.
        """
        if mode is None:
            mode = "r" if self._read_only else "a"

        return h5py.File(self._path, mode)

    def _print(self, *args, **kwargs):
        """Private helper for logging.
        """
        # Only print in verbose mode
        if not self._verbose:
            return

        args = list(args)
        args[0] = f"[TrajectoriesFile-{self._path}] {args[0]}"
        print(*args, **kwargs)
