import torch.utils.tensorboard


class _BuddyLogging:
    """Private mixin for encapsulating logging functions.
    """

    def __init__(self):
        """Logging-specific setup.
        """
        super().__init__()

        # Create some misc state variables for tensorboard
        # The writer is lazily instantiated in TrainingBuddy.log()
        self._writer = None
        self._log_scopes = []

    def log_scope(self, scope):
        """Returns a scope to log tensors in.

        Example usage:

        ```
            with buddy.log_scope("scope"):
                # Logs to scope/loss
                buddy.log("loss", loss_tensor)
        ```
        """
        class _Namespace:
            def __enter__(unused_self):
                self._log_scopes.append(scope)
                return unused_self

            def __exit__(*unused):
                self._log_scopes.pop()
                return

        return _Namespace()

    def log(self, name, value):
        """Log a tensor for visualization in Tensorboard. Currently only supports scalars.
        """
        if len(self._log_scopes) > 0:
            name = "{}/{}".format("/".join(self._log_scopes), name)
        if self._writer is None:
            self._writer = torch.utils.tensorboard.SummaryWriter(
                self._config['log_dir'] + "/" + self._experiment_name)

        self._writer.add_scalar(name, value, global_step=self._steps)
