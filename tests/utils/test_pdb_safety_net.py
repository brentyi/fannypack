import os
import signal

import fannypack


class FakePDB:
    def __init__(self):
        self.called = False

    def set_trace(self):
        self.called = True


def test_sigint():
    """Check that we open PDB when a SIGINT event is handled."""
    fake_pdb = FakePDB()

    fannypack.utils.pdb_safety_net()
    fannypack.utils._pdb_safety_net.pdb = fake_pdb

    assert not fake_pdb.called
    os.kill(os.getpid(), signal.SIGINT)
    assert fake_pdb.called
