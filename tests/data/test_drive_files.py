import os
import shutil
import time

import pytest

import fannypack


def test_cached_drive_file():

    # Move the cache to a temporary directory
    fannypack.data.set_cache_path("tmp")

    # Load file for the first time
    start_time = time.time()
    path0 = fannypack.data.cached_drive_file(
        "secret_key.pem",
        "https://drive.google.com/file/d/1AsY9Cs3xE0RSlr0FKlnSKHp6zIwFSvXe/view",
    )
    elapsed0 = time.time() - start_time

    # Load it a second
    start_time = time.time()
    path1 = fannypack.data.cached_drive_file(
        "secret_key.pem",
        "https://drive.google.com/file/d/1AsY9Cs3xE0RSlr0FKlnSKHp6zIwFSvXe/view",
    )
    elapsed1 = time.time() - start_time

    # Make sure second time was faster
    assert elapsed0 > elapsed1
    print(path0)

    # Check path values
    assert path0 == path1
    assert "secret_key.pem" in path0
    assert os.path.exists(path0)

    # Delete temporary files when done
    path = os.path.join("tmp/")
    if os.path.isdir(path):
        shutil.rmtree(path)
