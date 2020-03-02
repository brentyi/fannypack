import torch
import torch.nn as nn

from ._buddy_mixins._checkpointing import _BuddyCheckpointing
from ._buddy_mixins._logging import _BuddyLogging
from ._buddy_mixins._metadata import _BuddyMetadata
from ._buddy_mixins._optimizer import _BuddyOptimizer


class Buddy(
        _BuddyCheckpointing,
        _BuddyLogging,
        _BuddyMetadata,
        _BuddyOptimizer):

    """Buddy is a model manager that abstracts away PyTorch boilerplate.

    Helps with:
        - Creating/using/managing optimizers
        - Checkpointing (models + optimizers)
        - Namespaced/scoped Tensorboard logging
        - Saving human-readable metadata files
    """

    def __init__(
            self,
            experiment_name,
            model,
            *,
            verbose=True,
            checkpoint_dir="checkpoints",
            checkpoint_max_to_keep=5,
            metadata_dir="metadata",
            log_dir="logs",
            **optimizer_config):
        """Constructor
        """
        # Validate and assign core parameters.
        assert type(experiment_name) == str
        assert isinstance(model, nn.Module)
        assert type(verbose) == bool

        self._experiment_name = experiment_name
        self._model = model
        self._verbose = True

        # Use GPU for training if available.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            model.cuda()
        else:
            self._device = torch.device("cpu")
        self._print("Using device:", self._device)

        # Call constructors for each of our mixins.
        # This sets up logging, checkpointing, and optimization-specific state.
        #
        # State within each mixin should be encapsulated. (exception:
        # checkpointing automatically saves optimizer state)
        _BuddyCheckpointing.__init__(
            self, checkpoint_dir, checkpoint_max_to_keep)
        _BuddyMetadata.__init__(self, metadata_dir)
        _BuddyLogging.__init__(self, log_dir)
        _BuddyOptimizer.__init__(self, optimizer_config)

        # Print available checkpoints
        self._print("Available checkpoint labels:", self.checkpoint_labels)

    @property
    def device(self):
        """Read-only interface for the active torch device.
        """
        return self._device

    def _print(self, *args, **kwargs):
        """Private helper for logging.
        """
        # Only print in verbose mode
        if not self._verbose:
            return

        args = list(args)
        args[0] = f"[buddy-{self._experiment_name}] {args[0]}"
        print(*args, **kwargs)
