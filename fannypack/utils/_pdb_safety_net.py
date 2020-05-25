import pdb
import signal
import sys
import traceback as tb


def pdb_safety_net():
    """Helper for opening PDB when either (a) the user hits Ctrl+C or (b) we encounter
    an uncaught exception.
    """

    # Open PDB on Ctrl+C
    def handler(sig, frame):
        pdb.set_trace()

    signal.signal(signal.SIGINT, handler)

    # Open PDB when we encounter an uncaught exception
    def excepthook(type_, value, traceback):  # pragma: no cover (impossible to test)
        tb.print_exception(type_, value, traceback, limit=100)
        pdb.post_mortem(traceback)

    sys.excepthook = excepthook
