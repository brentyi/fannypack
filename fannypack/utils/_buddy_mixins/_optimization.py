import torch


class _BuddyOptimization:
    """Private mixin for encapsulating optimizer functions.
    """

    # Supported optimizer types
    OPTIMIZER_TYPES = {
        'adam': torch.optim.Adam,
        'adadelta': torch.optim.Adadelta,
    }
    DEFAULT_LEARNING_RATES = {
        'adam': 1e-4,
        'adadelta': 1
    }

    def __init__(self):
        super().__init__()

        # Instantiate optimizers, step count -- note that these may be
        # overriden by a loaded checkpoint
        self._optimizers = self._instantiate_optimizers()
        self._steps = 0

    def minimize(self, loss, optimizer_name="primary",
                 retain_graph=False, checkpoint_interval=1000):
        """Compute gradients and use them to minimize a loss function.
        """

        assert optimizer_name in self._optimizers.keys()

        # Update learning rate using scheduler if possible
        schedulers = self._config['learning_rate_schedulers']
        if optimizer_name in schedulers:
            self._set_learning_rate(
                schedulers[optimizer_name](self._steps), optimizer_name)

        # Take gradient step
        self._optimizers[optimizer_name].zero_grad()
        loss.backward(retain_graph=retain_graph)
        self._optimizers[optimizer_name].step()

        # Update step & checkpoint
        self._steps += 1
        if self._steps % checkpoint_interval == 0:
            self.save_checkpoint()

    def set_learning_rate(self, value, optimizer_name="primary"):
        """Sets an optimizer learning rate. Accepts either a floating point
        learning rate or a schedule function (int steps -> float LR).
        """

        schedulers = self._config['learning_rate_schedulers']
        if callable(value):
            # Store a scheduler
            assert type(value(0)) == float
            schedulers[optimizer_name] = value
        else:
            # Delete scheduler
            if optimizer_name in schedulers:
                schedulers.pop(optimizer_name)

            # Set scalar learning rate
            self._set_learning_rate(value, optimizer_name)

    def _set_learning_rate(self, value, optimizer_name):
        """(Private) Sets an optimizer's learning rate.
        """

        assert optimizer_name in self._optimizers.keys()

        optimizer = self._optimizers[optimizer_name]
        for param_group in optimizer.param_groups:
            param_group['lr'] = value

    def _instantiate_optimizers(self):
        """(Private) Instantiates optimizer objects and sets default learning rates.
        """

        # Make sure we're creating a valid optimizer
        optimizer_type = self._config['optimizer_type']
        assert optimizer_type in self.OPTIMIZER_TYPES

        # Instantiate an optimizer for each value in _config['optimizer_names']
        #
        # Note that if we're loading from a checkpoint, the initial learning
        # rate may be immediately overwritten
        Optimizer = self.OPTIMIZER_TYPES[optimizer_type]
        initial_learning_rate = self.DEFAULT_LEARNING_RATES[optimizer_type]
        optimizer_instances = {}
        for name in self._config['optimizer_names']:
            optimizer_instances[name] = Optimizer(
                self._model.parameters(), lr=initial_learning_rate)

        return optimizer_instances
