"""Lesson 15 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/15-debugging/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch
import torch.nn.functional as F
from torch import nn


def can_overfit(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                steps: int = 200) -> bool:
    """Train on this one batch with Adam (lr 1e-2) and return True
    if the final cross-entropy loss is below 0.05."""
    # TODO: the usual loop: forward, loss, zero_grad, backward, step
    raise NotImplementedError


def grad_norms(model: nn.Module) -> dict[str, float | None]:
    """After backward(): the norm of each parameter's gradient by name,
    or None if a parameter got no gradient."""
    # TODO: loop over model.named_parameters(), use p.grad.norm()
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    x, y = torch.randn(32, 20), torch.randint(0, 5, (32,))
    good = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
    assert can_overfit(good, x, y), "a correct model memorizes one batch"
    broken = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5),
                           nn.Softmax(dim=1))       # softmax before CE
    assert not can_overfit(broken, x, y), "softmax + CE gets stuck"

    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 5))
    model[0].weight.requires_grad_(False)            # frozen on purpose
    F.cross_entropy(model(x), y).backward()
    norms = grad_norms(model)
    assert list(norms) == ["0.weight", "0.bias", "2.weight", "2.bias"]
    assert norms["0.weight"] is None and norms["2.weight"] > 0
    print("All checks passed")
