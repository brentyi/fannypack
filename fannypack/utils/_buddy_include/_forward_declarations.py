import abc
from typing import Any, Dict

import torch
import torch.nn as nn


class _BuddyForwardDeclarations(abc.ABC):
    """Abstract class for forward-declaring attributes & functions that are shared
    laterally across interface mixins. Enables static type-checking.
    """

    # Primary attributes
    _experiment_name: str
    _model: nn.Module
    _device: torch.device

    # Shared-access optimizer attributes
    optimizer_steps: int
    _optimizer_dict: Dict[str, Any]
    _optimizer_config: Dict[str, Any]

    # Shared functions
    @abc.abstractproperty
    def device(self) -> torch.device:
        """Read-only interface for the active torch device.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _print(self, *args, **kwargs) -> None:
        """Private helper for logging.
        """
        raise NotImplementedError
