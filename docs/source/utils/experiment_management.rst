Experiment Management
==========================================

.. contents:: :local:

******************************************
Overview
******************************************

.. autoclass:: fannypack.utils.Buddy
   :members:

******************************************
Checkpointing
******************************************

.. autoclass:: fannypack.utils._buddy_include._checkpointing._BuddyCheckpointing
   :members:

******************************************
Optimization
******************************************

.. autoclass:: fannypack.utils._buddy_include._optimizer._BuddyOptimizer
   :members:

******************************************
Tensorboard Logging
******************************************

.. autoclass:: fannypack.utils._buddy_include._logging._BuddyLogging
   :members:

******************************************
Experiment Metadata
******************************************

.. autoclass:: fannypack.utils._buddy_include._metadata._BuddyMetadata
   :members:

******************************************
Command-line Interface
******************************************

Buddy's CLI currently supports four primary functions:

- ``buddy delete [experiment_name]``: Delete an existing experiment. Displays a
  selection menu with metadata preview if no experiment name is passed in.
- ``buddy info {experiment_name}``: Print summary + metadata of an existing
  experiment.
- ``buddy list``: Print table of existing experiments + basic information.
- ``buddy rename {source} {dest}``: Rename an existing experiment.

For more details and a full list of options, run ``buddy {subcommand} --help``.

---

The Buddy CLI also has full support for autcompleting experiment names. This
needs to be registered in your `.bashrc` to be enabled:

.. code-block:: bash

   # Append to .bashrc
   eval "$(register-python-argcomplete buddy)"

Alternatively, for zsh:

.. code-block:: bash

   # Append to .zshrc
   autoload -U +X compinit && compinit
   autoload -U +X bashcompinit && bashcompinit
   eval "$(register-python-argcomplete buddy)"

