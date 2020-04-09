"""Parses a coverage report from pytest-cov, and ensures that coverage for all
files is above a provided threshold.

A sample report might look something like this:

```
============================= test session starts ==============================
platform linux -- Python 3.7.3, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /home/brent/fannypack
plugins: cov-2.8.1
collected 22 items

tests/utils/test_buddy_checkpointing.py .........                        [ 40%]
tests/utils/test_buddy_optimizer.py ...                                  [ 54%]
tests/utils/test_module_freezing.py ...                                  [ 68%]
tests/utils/test_slice_wrapper.py .......                                [100%]

----------- coverage: platform linux, python 3.7.3-final-0 -----------
Name                                                  Stmts   Miss  Cover
-------------------------------------------------------------------------
fannypack/__init__.py                                     2      0   100%
fannypack/nn/__init__.py                                  1      0   100%
fannypack/nn/_resblocks.py                               35     25    29%
fannypack/utils/__init__.py                               8      0   100%
fannypack/utils/_buddy.py                                35      5    86%
fannypack/utils/_buddy_interfaces/__init__.py             0      0   100%
fannypack/utils/_buddy_interfaces/_checkpointing.py     167     47    72%
fannypack/utils/_buddy_interfaces/_logging.py            27     17    37%
fannypack/utils/_buddy_interfaces/_metadata.py           33     14    58%
fannypack/utils/_buddy_interfaces/_optimizer.py          53      7    87%
fannypack/utils/_conversions.py                          43     38    12%
fannypack/utils/_deprecation.py                           8      2    75%
fannypack/utils/_module_freezing.py                      21      0   100%
fannypack/utils/_slice_wrapper.py                        80     31    61%
fannypack/utils/_squeeze.py                              10      9    10%
fannypack/utils/_trajectories_file.py                   135    116    14%
-------------------------------------------------------------------------
TOTAL                                                   658    311    53%
```

"""

import sys
import os
import argparse

# Accept argument for required coverage percentage
parser = argparse.ArgumentParser()
parser.add_argument("--percentage-threshold", type=int, required=True)
args = parser.parse_args()

# Read coverage report (from pytest-cov)
lines = sys.stdin.readlines()
success = True
for line in lines:
    # Strip line breaks, whitespace funniness
    line = line.strip()

    # We only care about lines that end with a percentage
    if len(line) == 0 or line[-1] != "%":
        continue

    # We only care about individual files :)
    source_path = line[: line.index(" ")]
    if source_path == "TOTAL":
        continue

    # Get coverage percentage
    percentage = int(line[line.rindex(" ") + 1 : -1])
    print(f"{source_path}: {percentage}%")

    # Check coverage percentage
    if percentage < args.percentage_threshold:
        print("    Failed!")
        success = False

assert success
