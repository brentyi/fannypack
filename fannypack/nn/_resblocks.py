import abc

import torch
import torch.nn as nn


class Base(nn.Module, abc.ABC):
    def __init__(self, activation: str = "relu", activations_inplace: bool = False):
        super().__init__()

        self.activations_inplace = activations_inplace

        self.block1: nn.Module
        self.block2: nn.Module
        self.activation = self._activation_func(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ResBlock forward pass.
        """
        residual = x
        x = self.block1(x)
        x = self.activation(x)
        x = self.block2(x)
        assert x.shape[0] == residual.shape[0]
        x += residual
        x = self.activation(x)
        return x

    def _activation_func(self, activation: str) -> nn.Module:
        return nn.ModuleDict(
            {
                "relu": nn.ReLU(inplace=self.activations_inplace),
                "leaky_relu": nn.LeakyReLU(inplace=self.activations_inplace),
                "selu": nn.SELU(inplace=self.activations_inplace),
                "none": nn.Identity(),
            }
        )[activation]


class Linear(Base):
    def __init__(self, units: int, bottleneck_units: int = None, **resblock_base_args):
        super().__init__(**resblock_base_args)

        if bottleneck_units is None:
            bottleneck_units = units
        self.block1 = nn.Linear(units, bottleneck_units)
        self.block2 = nn.Linear(bottleneck_units, units)


class Conv2d(Base):
    def __init__(
        self,
        channels: int,
        bottleneck_channels: int = None,
        kernel_size: int = 3,
        **resblock_base_args
    ):
        super().__init__(**resblock_base_args)

        if bottleneck_channels is None:
            bottleneck_channels = channels

        self.block1 = nn.Conv2d(
            channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.block2 = nn.Conv2d(
            bottleneck_channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
