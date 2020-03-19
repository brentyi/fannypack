import torch.utils.tensorboard


class _BuddyLogging:
    """Buddy's TensorBoard logging interface.
    """

    def __init__(self, log_dir):
        """Logging-specific setup.
        """
        self._log_dir = log_dir

        # State variables for TensorBoard
        # Note that the writer is lazily instantiated in TrainingBuddy.log()
        self._log_writer = None
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
                self.log_scope_push(scope)
                return unused_self

            def __exit__(*unused):
                self.log_scope_pop(scope)
                return

        return _Namespace()

    def log_scope_push(self, scope):
        """Push a scope to log tensors into.

        Example usage:
        ```
            buddy.log_scope_push("scope")

            # Logs to scope/loss
            buddy.log("loss", loss_tensor)

            buddy.log_scope_pop("scope") # name parameter is optional
        ```
        """
        self._log_scopes.append(scope)

    def log_scope_pop(self, scope=None):
        """Pop a scope we logged tensors into. See `log_scope_push()`.
        """
        popped = self._log_scopes.pop()
        if scope is not None:
            assert popped == scope

    def log(self, name, value):
        """Log a tensor for visualization in TensorBoard. Currently only
        supports scalars.
        """
        # Add scope prefixes
        if len(self._log_scopes) > 0:
            name = "{}/{}".format("/".join(self._log_scopes), name)

        # Lazy instantiation for tensorboard writer
        if self._log_writer is None:
            self._log_writer = torch.utils.tensorboard.SummaryWriter(
                self._log_dir + "/" + self._experiment_name
            )

        # Log scalar
        self._log_writer.add_scalar(
            name, value, global_step=self._optimizer_steps
        )
