from fannypack.nn import resblocks


def test_resblock_smoke_test():
    """Make sure we can build all resblocks.
    """

    for inplace in (True, False):
        for activation in resblocks.Base._activation_types.keys():
            resblocks.Linear(20, 3, activation=activation, activations_inplace=inplace)
            resblocks.Conv2d(
                channels=20,
                bottleneck_channels=3,
                kernel_size=5,
                activation=activation,
                activations_inplace=inplace,
            )
