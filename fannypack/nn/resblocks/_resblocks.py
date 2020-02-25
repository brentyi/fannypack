import abc
import torch.nn as nn


class _AbstractResBlock(nn.Module, abc.ABC):
    default_activation = "relu"

    def __init__(self, activation=None, activations_inplace=True):
        super().__init__()

        if activation is None:
            activation = self.default_activation
        self.activations_inplace = activations_inplace

        self.block1 = None
        self.block2 = None
        self.activation = self._activation_func(activation)

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.activation(x)
        x = self.block2(x)
        assert x.shape[0] == residual.shape[0]
        x += residual
        x = self.activation(x)
        return x

    def _activation_func(self, activation):
        return nn.ModuleDict([
            ['relu', nn.ReLU(inplace=self.activations_inplace)],
            ['leaky_relu', nn.LeakyReLU(inplace=self.activations_inplace)],
            ['selu', nn.SELU(inplace=self.activations_inplace)],
            ['none', nn.Identity()],
        ])[activation]


class Linear(_AbstractResBlock):
    default_activation = "relu"

    def __init__(self, units, bottleneck_units=None, **kwargs):
        super().__init__(**kwargs)

        if bottleneck_units is None:
            bottleneck_units = units
        self.block1 = nn.Linear(units, bottleneck_units)
        self.block2 = nn.Linear(bottleneck_units, units)


class Conv2d(_AbstractResBlock):
    default_activation = "relu"
    default_kernel_size = 3

    def __init__(self, channels, bottleneck_channels=None,
                 kernel_size=None, **kwargs):
        super().__init__(**kwargs)

        if bottleneck_channels is None:
            bottleneck_channels = channels
        if kernel_size is None:
            kernel_size = self.default_kernel_size

        conv2d_args = {
            'kernel_size': kernel_size,
            'padding': kernel_size // 2
        }

        self.block1 = nn.Conv2d(channels, bottleneck_channels, **conv2d_args)
        self.block2 = nn.Conv2d(bottleneck_channels, channels, **conv2d_args)
