import fannypack


def test_stopwatch():
    """Check that stopwatch helper runs without errors."""
    with fannypack.utils.stopwatch():
        pass
    with fannypack.utils.stopwatch("with label"):
        pass
