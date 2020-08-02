# Register subcommands
from . import _subcommand_delete as _
from . import _subcommand_info as _
from . import _subcommand_list as _
from . import _subcommand_rename as _

# Expose important stuff for CLI script
from ._subcommand import Subcommand, subcommand_registry
from ._utils import BuddyPaths

# Erase `_`
_ = None
del _
