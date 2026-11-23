"""Lesson 22 · Solution to gym.py.

Run:  uv run python lessons/22-lora/solution.py
No download needed: plain tensors and a tiny nn.Linear.
"""

import torch
from torch import nn


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(parameters with requires_grad, all parameters)."""
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, sum(p.numel() for p in model.parameters())


def lora_forward(x, W, A, B, alpha: float, r: int) -> torch.Tensor:
    """y = x W^T + (alpha / r) * x A^T B^T
    W: (out, in) frozen, A: (r, in), B: (out, r)."""
    return x @ W.T + (alpha / r) * (x @ A.T @ B.T)


if __name__ == "__main__":
    net = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 2))
    net[0].requires_grad_(False)
    assert count_trainable(net) == (42, 262), count_trainable(net)

    torch.manual_seed(0)
    x, W = torch.randn(4, 16), torch.randn(32, 16)
    A, B = torch.randn(8, 16), torch.zeros(32, 8)
    assert torch.allclose(lora_forward(x, W, A, B, 16, 8), x @ W.T), \
        "with B = 0 the output must not change"
    B = torch.randn(32, 8)
    merged = W + (16 / 8) * B @ A                 # what merging does
    assert torch.allclose(lora_forward(x, W, A, B, 16, 8), x @ merged.T,
                          atol=1e-4)
    print("All checks passed")
