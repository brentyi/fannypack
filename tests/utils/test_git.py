import fannypack


def test_get_git_commit_hash():
    """Checks that we can read this file's commit hash."""
    assert len(fannypack.utils.get_git_commit_hash(__file__)) == 40
