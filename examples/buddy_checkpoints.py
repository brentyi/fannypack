import torch.nn as nn

import fannypack


class SimpleNet(nn.Module):
    """Simple PyTorch model. Scalar input, scalar output."""

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


# Create a simple network
model = SimpleNet()
buddy = fannypack.utils.Buddy("experiment-1", model)

# Save a checkpoint (automatically labeled with step count)
buddy.save_checkpoint()

# Save a labeled checkpoint
buddy.save_checkpoint(label="memorable_name")

# Load a checkpoint (latest)
buddy.load_checkpoint()

# Load a labeled checkpoint
buddy.load_checkpoint(label="memorable_name")

# Load a checkpoint from a specific path
buddy.load_checkpoint(path="checkpoints/specific_path.ckpt")

# Load a specific module from a checkpoint
# Note that this takes the same optional label, path args as `load_checkpoint`
buddy.load_checkpoint_module("layer2", label="memorable_name")

# Load a specfic module, which might have a different name in our checkpoint
# Helpful if a module's been renamed!
#
# Note that this takes the same optional label, path args as `load_checkpoint`
buddy.load_checkpoint_module(source="layer3", target="layer2", label="memorable_name")

# Similar API for loading just optimizer state!
buddy.load_checkpoint_optimizer("primary", label="memorable_name")
buddy.load_checkpoint_optimizer(
    source="primary", target="primary", label="memorable_name"
)
