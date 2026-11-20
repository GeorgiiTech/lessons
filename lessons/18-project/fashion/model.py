"""The model: no data loading, no training loop, just the network."""

from torch import nn


def make_mlp(hidden: int = 256, n_classes: int = 10) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden),
                         nn.ReLU(), nn.Linear(hidden, n_classes))
