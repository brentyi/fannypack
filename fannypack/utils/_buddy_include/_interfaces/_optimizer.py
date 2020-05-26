from __future__ import annotations

import abc
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union, cast

import torch

from .._forward_declarations import _BuddyForwardDeclarations

if TYPE_CHECKING:
    from ._checkpointing import _BuddyCheckpointing


class _BuddyOptimizer(_BuddyForwardDeclarations, abc.ABC):
    """Buddy's optimization interface.
    """

    # Supported optimizer types
    # Note that torch (as of 1.5) has some stub issues with optim.Adadelta,
    # optim.Optimizer
    _OPTIMIZER_TYPES: Dict[str, torch.optim.Optimizer] = {  # type: ignore
        "adam": torch.optim.Adam,
        "adadelta": torch.optim.Adadelta,  # type: ignore
    }

    # Default learning rates
    _OPTIMIZER_DEFAULT_LEARNING_RATES: Dict[str, float] = {
        "adam": 1e-4,
        "adadelta": 1,
    }

    def __init__(
        self, optimizer_type: str, optimizer_checkpoint_interval: float
    ) -> None:
        """Optimizer-specific setup.
        """
        # Assign our training configuration.
        self._optimizer_config: Dict[str, Any] = {
            "global_steps": 0,
            "optimizer_type": optimizer_type,
            "learning_rate_schedulers": {},
        }

        # Map from optimizer name to optimizers
        # These are constructed lazily!
        self._optimizer_dict: Dict[str, torch.optim.Optimizer] = {}  # type: ignore

        # Autocheckpoint variables
        self._optimizer_checkpoint_interval: float = optimizer_checkpoint_interval
        self._optimizer_last_checkpoint_time: Optional[float] = None

        # Default learning rate
        self._optimizer_default_learning_rate: Union[
            float, Callable[[int], float]
        ] = self._OPTIMIZER_DEFAULT_LEARNING_RATES[optimizer_type]

    def minimize(
        self,
        loss: torch.Tensor,
        optimizer_name: str = "primary",
        *,
        retain_graph: bool = False,
        checkpoint_interval: float = None,
    ) -> None:
        """Compute gradients and use them to minimize a loss function.
        """
        assert self._model is not None, "No model attached!"
        self._instantiate_optimizer(optimizer_name)

        # Update learning rate using scheduler if possible
        schedulers = self._optimizer_config["learning_rate_schedulers"]
        if optimizer_name in schedulers:
            self._set_learning_rate(
                schedulers[optimizer_name](self._optimizer_config["global_steps"]),
                optimizer_name,
            )

        # Take gradient step
        self._optimizer_dict[optimizer_name].zero_grad()
        loss.backward(retain_graph=retain_graph)  # type: ignore
        self._optimizer_dict[optimizer_name].step()

        # Update global step count
        self._optimizer_config["global_steps"] += 1

        # Autocheckpoint procedure
        if checkpoint_interval is None:
            checkpoint_interval = self._optimizer_checkpoint_interval

        # Disable autocheckpoint if interval is 0
        if checkpoint_interval == 0:
            return

        if self._optimizer_last_checkpoint_time is None:
            # First iteration
            self._optimizer_last_checkpoint_time = time.time()
        elif (
            time.time() - cast(float, self._optimizer_last_checkpoint_time)
            > self._optimizer_checkpoint_interval
        ):  # pragma: no cover
            # Checkpoint!
            cast("_BuddyCheckpointing", self).save_checkpoint()
            self._optimizer_last_checkpoint_time = time.time()

    def get_learning_rate(self, optimizer_name: str = "primary") -> float:
        """Gets an optimizer learning rate.
        """
        assert self._model is not None, "No model attached!"
        assert optimizer_name in self._optimizer_dict

        # Return scheduled learning rate
        schedulers = self._optimizer_config["learning_rate_schedulers"]
        if optimizer_name in schedulers:
            return schedulers[optimizer_name](self.optimizer_steps)

        # Return raw learning rate
        # Currently, only one parameter group is supported
        optimizer = self._optimizer_dict[optimizer_name]
        assert len(optimizer.param_groups) == 1
        return optimizer.param_groups[0]["lr"]

    def set_learning_rate(
        self,
        value: Union[float, Callable[[int], float]],
        optimizer_name: str = "primary",
    ) -> None:
        """Sets an optimizer learning rate. Accepts either a floating point
        learning rate or a schedule function (int steps -> float LR).
        """
        assert self._model is not None, "No model attached!"
        schedulers = self._optimizer_config["learning_rate_schedulers"]
        if callable(value):
            # Store a scheduler
            assert type(value(0)) == float
            schedulers[optimizer_name] = value
        else:
            # Set learning rate to a float
            assert type(value) == float
            # Delete scheduler
            if optimizer_name in schedulers.keys():
                schedulers.pop(optimizer_name)

            # Set scalar learning rate
            self._set_learning_rate(value, optimizer_name)

    def set_default_learning_rate(
        self, value: Union[float, Callable[[int], float]]
    ) -> None:
        """Sets a default learning rate for new optimizers.
        """
        self._optimizer_default_learning_rate = value

    @property
    def optimizer_steps(self) -> int:
        """Read-only interface for # of steps taken by optimizer.
        """
        return self._optimizer_config["global_steps"]

    def _set_learning_rate(self, value: float, optimizer_name: str) -> None:
        """(Private) Sets an optimizer's learning rate.
        """

        self._instantiate_optimizer(optimizer_name)

        # Currently, only one parameter group is supported
        optimizer = self._optimizer_dict[optimizer_name]
        assert len(optimizer.param_groups) == 1
        optimizer.param_groups[0]["lr"] = value

    def _instantiate_optimizer(self, optimizer_name: str) -> None:
        """(Private) Instantiates an optimizer. Returns immediately if
        optimizer already exists.
        """
        assert self._model is not None, "No model attached!"
        if optimizer_name in self._optimizer_dict.keys():
            # Optimizer already exists: do nothing!
            return

        self._print("Instantiating optimizer: ", optimizer_name)

        # Make sure we're creating a valid optimizer
        optimizer_type = self._optimizer_config["optimizer_type"]
        assert optimizer_type in self._OPTIMIZER_TYPES

        # Parameters
        Optimizer = self._OPTIMIZER_TYPES[optimizer_type]

        # Construct optimizer
        self._optimizer_dict[optimizer_name] = Optimizer(self._model.parameters())
        self.set_learning_rate(
            self._optimizer_default_learning_rate, optimizer_name=optimizer_name
        )
