from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import fannypack
import torch.utils.tensorboard


@dataclass
class _LogNamespace:
    buddy: _BuddyLogging
    scope: str

    def __enter__(self):
        self.buddy.log_scope_push(self.scope)
        return self

    def __exit__(self, *unused):
        self.buddy.log_scope_pop(self.scope)
        return


class _BuddyLogging:
    """Buddy's TensorBoard logging interface.
    """

    def __init__(self, log_dir: str) -> None:
        """Logging-specific setup.
        """
        self._log_dir = log_dir

        # State variables for TensorBoard
        # Note that the writer is lazily instantiated in TrainingBuddy.log()
        self._log_writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
        self._log_scopes: List[str] = []

    def log_scope(self, scope: str) -> _LogNamespace:
        """Returns a scope to log tensors in.

        Example usage:

        ```
            with buddy.log_scope("scope"):
                # Logs to scope/loss
                buddy.log("loss", loss_tensor)
        ```
        """
        return _LogNamespace(self, scope)

    def log_scope_push(self, scope: str) -> None:
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

    def log_scope_pop(self, scope: str = None) -> None:
        """Pop a scope we logged tensors into. See `log_scope_push()`.
        """
        popped = self._log_scopes.pop()
        if scope is not None:
            assert popped == scope

    def log(self, name: str, value) -> None:
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
        self._log_writer.add_scalar(name, value, global_step=self.optimizer_steps)
