"""Lesson 14 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/14-transfer/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch
from torch import nn
from torchvision import models


def freeze_all_but(model: nn.Module, prefix: str) -> int:
    """Freeze every parameter whose name does not start with `prefix`
    and return the number of trainable values left."""
    # TODO: loop over model.named_parameters(), set requires_grad
    raise NotImplementedError


def two_speed_groups(model: nn.Module, head: str, backbone_lr: float,
                     head_lr: float) -> list[dict]:
    """Optimizer parameter groups: the head learns at head_lr, the rest
    of the network (the backbone) at the smaller backbone_lr."""
    # TODO: two dicts {"params": [...], "lr": ...}, backbone first
    raise NotImplementedError


if __name__ == "__main__":
    model = models.resnet18(weights=None)           # no download needed
    model.fc = nn.Linear(512, 10)
    assert freeze_all_but(model, "fc") == 5130, "512 × 10 + 10"
    assert not model.conv1.weight.requires_grad, "backbone is frozen"

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    groups = two_speed_groups(model, "fc", 1e-4, 1e-3)
    assert [g["lr"] for g in groups] == [1e-4, 1e-3]
    sizes = [sum(p.numel() for p in g["params"]) for g in groups]
    assert sizes == [11_176_512, 5130], sizes
    opt = torch.optim.Adam(groups)
    assert opt.param_groups[1]["lr"] == 1e-3
    print("All checks passed")
