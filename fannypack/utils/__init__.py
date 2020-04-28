from ._buddy import Buddy
from ._conversions import to_device, to_numpy, to_torch
from ._deprecation import deprecation_wrapper, new_name_wrapper
from ._module_freezing import freeze_module, unfreeze_module
from ._slice_wrapper import SliceWrapper
from ._squeeze import squeeze
from ._trajectories_file import TrajectoriesFile

DictIterator = new_name_wrapper(
    "fannypack.utils.DictIterator", "fannypack.utils.SliceWrapper", SliceWrapper,
)
