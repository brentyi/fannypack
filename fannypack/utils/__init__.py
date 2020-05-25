from ..data import _trajectories_file
from ._buddy import *
from ._conversions import *
from ._deprecation import *
from ._module_freezing import *
from ._slice_wrapper import *
from ._squeeze import *

DictIterator = new_name_wrapper(
    "fannypack.utils.DictIterator", "fannypack.utils.SliceWrapper", SliceWrapper,
)

TrajectoriesFile = new_name_wrapper(
    "fannypack.utils.TrajectoriesFiles",
    "fannypack.data.TrajectoriesFile",
    _trajectories_file.TrajectoriesFile,
)
