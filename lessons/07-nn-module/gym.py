"""Lesson 07 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/07-nn-module/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch
from torch import nn


class TwoFeatureModel(nn.Module):
    """score = w1 * hours + w2 * sleep + b: one nn.Linear with 2 inputs and 1 output."""

    def __init__(self) -> None:
        super().__init__()
        # TODO: create the layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def fit(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int) -> float:
    """Train with nn.MSELoss and torch.optim.Adam(lr=0.1). Return the final loss as a float."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(200, 2) * torch.tensor([10.0, 8.0])     # hours studied, hours slept
    y = x @ torch.tensor([[4.0], [3.0]]) + 20               # the true rule, no noise

    model = TwoFeatureModel()
    assert sum(p.numel() for p in model.parameters()) == 3, "2 weights + 1 bias"
    loss = fit(model, x, y, epochs=3000)
    print(f"loss {loss:.4f}")
    assert loss < 1e-2, f"loss should be near 0, got {loss}"
    print("All checks passed")
