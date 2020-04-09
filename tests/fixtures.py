import pytest

import torch
import torch.nn as nn
import numpy as np
import os
import shutil

import fannypack


class SimpleNet(nn.Module):
    """ Simple PyTorch model. Scalar input, scalar output.
    """

    def __init__(self):
        super().__init__()

        # Define layers
        self.layer1 = nn.Linear(1, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 8)
        self.layer4 = nn.Linear(8, 1)

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
    # Deterministic tests are nice..
    torch.manual_seed(0)
    return SimpleNet()


@pytest.fixture()
def simple_buddy():
    """Fixture for setting up a Buddy, as well as some dummy training data.
    """
    # Deterministic tests are nice..
    np.random.seed(0)
    torch.manual_seed(0)

    # Construct neural net, training buddy
    simple_net = SimpleNet()
    buddy = fannypack.utils.Buddy(
        "simple_net",
        simple_net,
        # Use directories relative to this fixture
        checkpoint_dir=os.path.join(
            os.path.dirname(__file__), "data/checkpoints/"
        ),
        metadata_dir=os.path.join(os.path.dirname(__file__), "data/metadata/"),
        log_dir=os.path.join(os.path.dirname(__file__), "data/log/"),
        verbose=True,
    )

    # Batch size
    N = 20

    # Learn to regress a constant
    data = torch.FloatTensor(np.random.normal(size=(N, 1)))
    labels = torch.FloatTensor(np.random.normal(loc=3, size=(1, 1))).expand(
        (N, 1)
    )
    return simple_net, buddy, data, labels


@pytest.fixture()
def simple_buddy_temporary_data():
    """Fixture for setting up a Buddy, as well as some dummy training data.

    This is identical to `simple_buddy`, but uses temporary directories for
    checkpointing, metadata saving, and logging. Saved files are deleted
    automatically when the test exits.
    """
    # Deterministic tests are nice..
    np.random.seed(0)
    torch.manual_seed(0)

    # Construct neural net, training buddy
    simple_net = SimpleNet()
    buddy = fannypack.utils.Buddy(
        "simple_net",
        simple_net,
        # Use directories relative to this fixture
        checkpoint_dir=os.path.join(
            os.path.dirname(__file__), "tmp/data/checkpoints/"
        ),
        metadata_dir=os.path.join(
            os.path.dirname(__file__), "tmp/data/metadata/"
        ),
        log_dir=os.path.join(os.path.dirname(__file__), "tmp/data/log/"),
        verbose=True,
    )

    # Batch size
    N = 20

    # Learn to regress a constant
    data = torch.FloatTensor(np.random.normal(size=(N, 1)))
    labels = torch.FloatTensor(np.random.normal(loc=3, size=(1, 1))).expand(
        (N, 1)
    )
    yield simple_net, buddy, data, labels

    # Delete temporary files when done
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "tmp/"))
