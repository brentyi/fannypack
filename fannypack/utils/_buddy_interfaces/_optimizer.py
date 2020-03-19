import torch.optim


class _BuddyOptimizer:
    """Buddy's optimization interface.
    """

    # Supported optimizer types
    _OPTIMIZER_TYPES = {
        "adam": torch.optim.Adam,
        "adadelta": torch.optim.Adadelta,
    }

    # Default learning rates
    _OPTIMIZER_DEFAULT_LEARNING_RATES = {
        "adam": 1e-4,
        "adadelta": 1,
    }

    def __init__(self, optimizer_type, optimizer_names):
        """Optimizer-specific setup.
        """
        # Assign our training configuration.
        self._optimizer_config = {
            "optimizer_type": optimizer_type,
            "optimizer_names": optimizer_names,
            "learning_rate_schedulers": {},
        }

        # Instantiate optimizers, step count -- note that these may be
        # overriden by a loaded checkpoint
        self._optimizer_dict = _BuddyOptimizer._instantiate_optimizers(
            model=self._model,
            optimizer_type=optimizer_type,
            optimizer_names=optimizer_names,
        )
        self._optimizer_steps = 0

    def minimize(
        self,
        loss,
        optimizer_name="primary",
        retain_graph=False,
        checkpoint_interval=1000,
    ):
        """Compute gradients and use them to minimize a loss function.
        """

        assert optimizer_name in self._optimizer_dict.keys()

        # Update learning rate using scheduler if possible
        schedulers = self._optimizer_config["learning_rate_schedulers"]
        if optimizer_name in schedulers:
            self._set_learning_rate(
                schedulers[optimizer_name](self._optimizer_steps),
                optimizer_name,
            )

        # Take gradient step
        self._optimizer_dict[optimizer_name].zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._optimizer_dict[optimizer_name].step()

        # Update step & checkpoint
        self._optimizer_steps += 1
        if self._optimizer_steps % checkpoint_interval == 0:
            self.save_checkpoint()

    def set_learning_rate(self, value, optimizer_name="primary"):
        """Sets an optimizer learning rate. Accepts either a floating point
        learning rate or a schedule function (int steps -> float LR).
        """

        schedulers = self._optimizer_config["learning_rate_schedulers"]
        if callable(value):
            # Store a scheduler
            assert type(value(0)) == float
            schedulers[optimizer_name] = value
        else:
            # Set learning rate to a float
            assert type(value) == float
            # Delete scheduler
            if optimizer_name in schedulers:
                schedulers.pop(optimizer_name)

            # Set scalar learning rate
            self._set_learning_rate(value, optimizer_name)

    @property
    def optimizer_steps(self):
        """Read-only interface for # of steps taken by optimizer.
        """
        return self._optimizer_steps

    def _set_learning_rate(self, value, optimizer_name):
        """(Private) Sets an optimizer's learning rate.
        """

        assert optimizer_name in self._optimizer_dict.keys()

        optimizer = self._optimizer_dict[optimizer_name]
        for param_group in optimizer.param_groups:
            param_group["lr"] = value

    @classmethod
    def _instantiate_optimizers(cls, model, optimizer_type, optimizer_names):
        """(Private) Instantiates optimizer objects and sets default learning
        rates.
        """

        # Make sure we're creating a valid optimizer
        optimizer_type = optimizer_type
        assert optimizer_type in cls._OPTIMIZER_TYPES

        # Instantiate an optimizer for each optimizer name
        #
        # Note that if we're loading from a checkpoint, the initial learning
        # rate may be immediately overwritten
        Optimizer = cls._OPTIMIZER_TYPES[optimizer_type]
        initial_learning_rate = cls._OPTIMIZER_DEFAULT_LEARNING_RATES[
            optimizer_type
        ]
        optimizer_instances = {}
        for name in optimizer_names:
            optimizer_instances[name] = Optimizer(
                model.parameters(), lr=initial_learning_rate
            )

        return optimizer_instances
