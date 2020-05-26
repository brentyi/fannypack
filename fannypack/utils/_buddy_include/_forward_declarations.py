import abc
from typing import Optional

import torch
import torch.nn as nn


class _BuddyForwardDeclarations(abc.ABC):
    """Abstract class for forward-declaring Buddy's attributes. Enables static
    type-checking in mixins.
    """

    _experiment_name: str
    _model: Optional[nn.Module]
    _device: torch.device

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
