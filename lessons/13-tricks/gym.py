"""Lesson 13 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/13-tricks/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch
from torch import nn


def my_batch_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Train-mode BatchNorm1d without gamma and beta: for each feature
    (column), subtract its batch mean and divide by sqrt(var + eps)."""
    # TODO: mean and biased variance over dim 0 (unbiased=False)
    raise NotImplementedError


def my_clip_(grads: list[torch.Tensor], max_norm: float) -> float:
    """Clip in place like clip_grad_norm_: if the total norm of all
    gradients is above max_norm, scale every one of them down."""
    # TODO: total = sqrt(sum of squares of all elements); scale if needed
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(32, 5) * 3 + 7
    bn = nn.BatchNorm1d(5, affine=False)
    assert torch.allclose(my_batch_norm(x), bn(x), atol=1e-5), \
        "same result as nn.BatchNorm1d in train mode"

    grads = [torch.randn(3, 4) * 10, torch.randn(4) * 10]
    params = [g.clone().requires_grad_() for g in grads]
    for p, g in zip(params, grads):
        p.grad = g.clone()
    expected = nn.utils.clip_grad_norm_(params, max_norm=1.0)
    total = my_clip_(grads, max_norm=1.0)
    assert abs(total - expected.item()) < 1e-4, "returns the norm before"
    for p, g in zip(params, grads):
        assert torch.allclose(p.grad, g, atol=1e-6), "scaled like PyTorch"
    assert my_clip_([torch.ones(4) * 0.1], 1.0) < 1.0, "small: untouched"
    print("All checks passed")
