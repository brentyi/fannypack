import pytest
import torch.nn as nn


class SimpleNet(nn.Module):
    """ Simple PyTorch model. Scalar input, scalar output.
    """

    def __init__(self):
        super().__init__()

        # Define layers
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 1)

        # Compose layers w/ nonlinearities
        self.layers = nn.Sequential(
            self.layer1,
            nn.ReLU(inplace=True),
            self.layer2,
            nn.ReLU(inplace=True),
            self.layer3,
            nn.ReLU(inplace=True),
            self.layer4,
        )

    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def simple_net():
    """Constructs an MLP implemented in PyTorch.
    """
    return SimpleNet()
