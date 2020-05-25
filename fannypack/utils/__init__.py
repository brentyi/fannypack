from ..data import _trajectories_file
from ._buddy import Buddy
from ._conversions import to_device, to_numpy, to_torch
from ._deprecation import deprecation_wrapper, new_name_wrapper
from ._module_freezing import freeze_module, unfreeze_module
from ._pdb_safety_net import pdb_safety_net
from ._slice_wrapper import SliceWrapper
from ._squeeze import squeeze

DictIterator = new_name_wrapper(
    "fannypack.utils.DictIterator", "fannypack.utils.SliceWrapper", SliceWrapper,
)

TrajectoriesFile = new_name_wrapper(
    "fannypack.utils.TrajectoriesFiles",
    "fannypack.data.TrajectoriesFile",
    _trajectories_file.TrajectoriesFile,
)
