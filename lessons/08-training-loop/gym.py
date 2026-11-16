"""Lesson 08 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/08-training-loop/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch
from torch import nn


def split(x: torch.Tensor, y: torch.Tensor, val_fraction: float, seed: int):
    """Shuffle with torch.randperm (seeded), then return x_train, y_train, x_val, y_val."""
    # TODO
    raise NotImplementedError


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean squared error in evaluation mode, without building a graph. Return a float."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    x = torch.arange(50.0).unsqueeze(1)
    y = 2 * x
    x_tr, y_tr, x_val, y_val = split(x, y, val_fraction=0.2, seed=0)
    assert x_tr.shape == (40, 1) and x_val.shape == (10, 1), "40 train, 10 val"
    assert set(x_tr.flatten().tolist()).isdisjoint(x_val.flatten().tolist()), "no example in both sets"
    assert torch.equal(y_val, 2 * x_val), "x and y must be shuffled together"
    assert not torch.equal(x_tr[:5], x[:5]), "shuffle before splitting"
    x_tr2, *_ = split(x, y, val_fraction=0.2, seed=0)
    assert torch.equal(x_tr, x_tr2), "same seed, same split"

    model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1))
    model.train()
    a, b = evaluate(model, x_val, y_val), evaluate(model, x_val, y_val)
    assert isinstance(a, float), "return a Python float"
    assert a == b, "evaluation must be deterministic: switch dropout off"
    assert not model.training, "evaluate should leave the model in eval mode"
    print("All checks passed")
