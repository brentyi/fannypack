import pytest

import fannypack
import os
import numpy as np


@pytest.fixture()
def trajectories_file_write():
    """Fixture for setting up a trajectories file to write to.
    """

    # Create object
    path = os.path.join(os.path.dirname(__file__), "temporary_trajectory.hdf5")
    trajectories_file = fannypack.utils.TrajectoriesFile(path, read_only=False)

    # Yield
    yield trajectories_file

    # Delete temporary file
    if os.path.isfile(path):
        os.remove(path)


@pytest.fixture()
def trajectories_file_read():
    """Fixture for reading from an existing trajectories file.
    """

    path = os.path.join(
        os.path.dirname(__file__), "../data/trajectories/test.hdf5"
    )
    trajectories_file = fannypack.utils.TrajectoriesFile(path, read_only=True)
    return trajectories_file


def test_write(trajectories_file_write):
    """Checks that all of our write operations function as expected.
    """
    # Populate our trajectories file
    with trajectories_file_write as traj_file:
        # Add some trajectories
        for i in range(4):
            # Add some trajectory metadata
            traj_file.add_meta({"trajectory_index": i})
            traj_file.add_meta({"trajectory_index_string": str(i)})

            # Add some timesteps to our trajectory
            for j in range(i * 2 + 1):
                traj_file.add_timestep({"five": 5.0, "timestep": j})

            # Save what we have so far!
            traj_file.complete_trajectory()
            assert len(traj_file) == i + 1

        # Do some stuff & abandon before completion
        # This should result in no change to our file
        traj_file.add_meta({"trajectory_index": 8})
        traj_file.add_meta({"trajectory_index_string": str(8)})
        traj_file.add_timestep({"five": 5.0, "timestep": 8})
        traj_file.abandon_trajectory()
        traj_file.complete_trajectory()
        assert len(traj_file) == 4

        # Expand!
        traj_file.resize(len(traj_file) + 2)

        # Copy over initial values & make them consistent with formulas above
        i = 4

        # Give some bad initial values
        values = traj_file[3]
        traj_file[i] = values

        # Overwrite
        values["trajectory_index"] = i
        values["trajectory_index_string"] = str(i)
        values["five"] = [5 for _ in range(i * 2 + 1)]
        values["timestep"] = [j for j in range(i * 2 + 1)]
        traj_file[i] = values

        # Write some bad initial values to i + 1
        traj_file[i + 1] = values

        # Contract (removes last trajectory)
        traj_file.resize(len(traj_file) - 1)

    # Open the file we just wrote to, and run standard read tests
    traj_file = fannypack.utils.TrajectoriesFile(
        trajectories_file_write._path, read_only=True
    )
    test_read(traj_file)
    test_read_backward(traj_file)
    test_get_all_timestepped(traj_file)
    test_get_all_metadata(traj_file)
    test_get_all_string(traj_file)

    # Clear the original file
    with trajectories_file_write as traj_file:
        traj_file.clear()

    # Open the file we just wrote to, and run standard read tests
    traj_file = fannypack.utils.TrajectoriesFile(
        trajectories_file_write._path, read_only=True
    )
    assert len(traj_file) == 0


def test_read(trajectories_file_read):
    """Read an existing trajectories file, and check that its content match
    what we wrote to it.
    """
    traj_file = trajectories_file_read
    with traj_file:
        # We should have added 5 trajectories
        assert len(traj_file) == 5

        # Iterate over each trajectory
        counter = 0
        for i, traj in enumerate(traj_file):
            assert type(traj) == dict
            assert traj["trajectory_index"] == i
            assert traj["trajectory_index_string"] == str(i)
            assert len(traj["five"]) == i * 2 + 1
            assert len(traj["timestep"]) == i * 2 + 1
            counter += 1
        assert len(traj_file) == counter


def test_read_backward(trajectories_file_read):
    """Same as `test_read()`, but uses negative trajectory indexing.
    """
    traj_file = trajectories_file_read
    with traj_file:
        # We should have added 5 trajectories
        assert len(traj_file) == 5

        # Iterate over each trajectory
        counter = 0
        for i in range(len(traj_file)):
            traj = traj_file[i - len(traj_file)]
            assert type(traj) == dict
            assert traj["trajectory_index"] == i
            assert traj["trajectory_index_string"] == str(i)
            assert len(traj["five"]) == i * 2 + 1
            assert len(traj["timestep"]) == i * 2 + 1
            counter += 1
        assert len(traj_file) == counter


def test_get_all_timestepped(trajectories_file_read):
    """Checks our `get_all()` method with time series data.
    """
    with trajectories_file_read as traj_file:
        counter = 0
        for fives in traj_file.get_all("five"):
            for five in fives:
                assert five == 5.0
            counter += 1
        assert counter == 5


def test_get_all_metadata(trajectories_file_read):
    """Checks our `get_all()` method with per-trajectory metadata.
    """
    with trajectories_file_read as traj_file:
        counter = 0
        for index in traj_file.get_all("trajectory_index"):
            assert index == counter
            counter += 1
        assert counter == 5


def test_get_all_string(trajectories_file_read):
    """Checks our `get_all()` method with a string-type per-trajectory metadata.
    """
    with trajectories_file_read as traj_file:
        counter = 0
        for index in traj_file.get_all("trajectory_index_string"):
            assert index == str(counter)
            counter += 1
        assert counter == 5


def test_out_of_bounds_index_read(trajectories_file_read):
    """Check that we throw an appropriate error when passed an invalid index.
    (read, positive)
    """
    with trajectories_file_read as traj_file, pytest.raises(IndexError):
        trajectories_file_read[777]


def test_out_of_bounds_index_read_negative(trajectories_file_read):
    """Check that we throw an appropriate error when passed an invalid index.
    (read, negative)
    """
    with trajectories_file_read as traj_file, pytest.raises(IndexError):
        trajectories_file_read[-777]


def test_out_of_bounds_index_write(trajectories_file_read):
    """Check that we throw an appropriate error when passed an invalid index.
    (write, positive)
    """
    with trajectories_file_read as traj_file, pytest.raises(IndexError):
        trajectories_file_read[777] = {}


def test_out_of_bounds_index_write_negative(trajectories_file_read):
    """Check that we throw an appropriate error when passed an invalid index.
    (write, negative)
    """
    with trajectories_file_read as traj_file, pytest.raises(IndexError):
        trajectories_file_read[-777] = {}
