import torch
import torch.nn as nn

from ._buddy_mixins._checkpointing import _BuddyCheckpointing
from ._buddy_mixins._logging import _BuddyLogging
from ._buddy_mixins._optimizer import _BuddyOptimizer


class Buddy(_BuddyCheckpointing, _BuddyLogging, _BuddyOptimizer):
    """Buddy is a model manager that abstracts away PyTorch boilerplate.

    Helps with:
        - Creating/using/managing optimizers
        - Checkpointing (models + optimizers)
        - Namespaced/scoped Tensorboard logging
    """

    # Default configuration parameters
    DEFAULT_CONFIG = dict(
        optimizer_type="adam",
        optimizer_names=["primary"],
        log_dir="logs",
        checkpoint_dir="checkpoints",
        checkpoint_max_to_keep=5,
        learning_rate_schedulers={},
    )

    def __init__(self, experiment_name, model,
                 verbose=True, load_checkpoint=True, **config):
        """Constructor
        """
        # Validate and assign core parameters.
        assert type(experiment_name) == str
        assert isinstance(model, nn.Module)
        assert type(verbose) == bool

        self._experiment_name = experiment_name
        self._model = model
        self._verbose = True

        # Validate and assign our training configuration.
        self._config = self.DEFAULT_CONFIG.copy()
        for key, value in config.items():
            assert key in self.DEFAULT_CONFIG.keys()
            assert type(value) == type(self.DEFAULT_CONFIG[key])
            self._config[key] = config[key]

        # Use GPU for training if available.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            model.cuda()
        else:
            self._device = torch.device("cpu")
        self._print("Using device:", self._device)

        # Enable autograd anomaly detection by default
        torch.autograd.set_detect_anomaly(True)

        # Call constructors for each of our three mixins.
        # This sets up logging, checkpointing, and optimization-specific state.
        #
        # State within each mixin should be encapsulated. (exception:
        # checkpointing automatically saves optimizer state)
        super().__init__()

        # Automatically load latest checkpoint
        # Note that this is called _after_ checkpointing is set up above
        if load_checkpoint:
            self.load_checkpoint()

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
