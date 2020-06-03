import subprocess


def _capture(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
    out, err = proc.communicate()
    return out, err, proc.returncode


def test_buddy_no_args():
    command = ["buddy"]
    out, err, exitcode = _capture(command)
    assert exitcode == 2


def test_buddy_delete_no_args():
    command = ["buddy", "delete"]
    out, err, exitcode = _capture(command)
    assert exitcode == 2


def test_buddy_info_no_args():
    command = ["buddy", "info"]
    out, err, exitcode = _capture(command)
    assert exitcode == 2


def test_buddy_list_no_args():
    command = ["buddy", "list"]
    out, err, exitcode = _capture(command)
    assert exitcode == 0


def test_buddy_rename_no_args():
    command = ["buddy", "rename"]
    out, err, exitcode = _capture(command)
    assert exitcode == 2
