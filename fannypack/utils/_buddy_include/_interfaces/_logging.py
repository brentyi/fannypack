from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch.utils.tensorboard

from ... import _deprecation
from .._forward_declarations import _BuddyForwardDeclarations


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


class _BuddyLogging(_BuddyForwardDeclarations):
    """Buddy's TensorBoard logging interface.
    """

    def __init__(self, log_dir: str) -> None:
        """Logging-specific setup.

        Args:
            log_dir (str): Path to save Tensorboard logs to.
        """
        self._log_dir = log_dir

        # Backwards-compatibility for deprecated API
        self.log = _deprecation.new_name_wrapper(
            "Buddy.log()", "Buddy.log_scalar()", self.log_scalar
        )

        # State variables for TensorBoard
        # Note that the writer is lazily instantiated; see below
        self._log_writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
        self._log_scopes: List[str] = []

    def log_scope(self, scope: str) -> _LogNamespace:
        """Returns a scope to log tensors in.

        Example usage:

        ```
            with buddy.log_scope("scope"):
                # Logs to scope/loss
                buddy.log_scalar("loss", loss_tensor)
        ```

        Args:
            scope (str): Name of scope.

        Returns:
            _LogNamespace: Object for automatically pushing/popping scope.
        """
        return _LogNamespace(self, scope)

    def log_scope_push(self, scope: str) -> None:
        """Push a scope to log tensors into.

        Example usage:
        ```
            buddy.log_scope_push("scope")

            # Logs to scope/loss
            buddy.log_scalar("loss", loss_tensor)

            buddy.log_scope_pop("scope") # name parameter is optional

        Args:
            scope (str): Name of scope.
        ```
        """
        self._log_scopes.append(scope)

    def log_scope_pop(self, scope: str = None) -> None:
        """Pop a scope we logged tensors into. See `log_scope_push()`.

        Args:
            scope (str, optional): Name of scope. Needs to be the top one in the stack.
        """
        popped = self._log_scopes.pop()
        if scope is not None:
            assert popped == scope

    def log_image(
        self,
        name: str,
        image: Union[torch.Tensor, np.ndarray],
        dataformats: str = "CHW",
    ) -> None:
        """Log an image tensor for visualization in TensorBoard.

        Args:
            name (str): Identifier for Tensorboard.
            image (torch.Tensor or np.ndarray): Image to log.
            dataformats (str, optional): Dimension ordering. Defaults to "CHW".
        """
        # Add scope prefixes
        if len(self._log_scopes) > 0:
            name = "{}/{}".format("/".join(self._log_scopes), name)

        # Log scalar
        self._lazy_log_writer.add_image(
            name, image, global_step=self.optimizer_steps, dataformats=dataformats
        )

    def log_scalar(
        self, name: str, value: Union[torch.Tensor, np.ndarray, float]
    ) -> None:
        """Log a scalar for visualization in TensorBoard.

        Args:
            name (str): Identifier for Tensorboard.
            value (torch.Tensor, np.ndarray, or float): Value to log.
        """
        # Add scope prefixes
        if len(self._log_scopes) > 0:
            name = "{}/{}".format("/".join(self._log_scopes), name)

        # Log scalar
        self._lazy_log_writer.add_scalar(name, value, global_step=self.optimizer_steps)

    @property
    def _lazy_log_writer(self):
        """ Lazy instantiation for Tensorboard writer.
        """
        if self._log_writer is None:
            self._log_writer = torch.utils.tensorboard.SummaryWriter(
                self._log_dir + "/" + self._experiment_name
            )
        return self._log_writer
