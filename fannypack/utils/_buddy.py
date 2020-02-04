import torch
import torch.nn as nn

from ._buddy_mixins._checkpointing import _BuddyCheckpointing
from ._buddy_mixins._logging import _BuddyLogging
from ._buddy_mixins._optimization import _BuddyOptimization


class Buddy(_BuddyCheckpointing, _BuddyLogging, _BuddyOptimization):
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

    def __init__(self, experiment_name, model, load_checkpoint=True, **config):
        """Constructor
        """
        # Validate and assign core parameters.
        assert type(experiment_name) == str
        assert isinstance(model, nn.Module)

        self._experiment_name = experiment_name
        self._model = model

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
        print("Using device:", self._device)

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
