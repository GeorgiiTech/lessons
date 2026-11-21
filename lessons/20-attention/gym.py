"""Lesson 20 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/20-attention/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import math

import torch
import torch.nn.functional as F


def my_attention(q: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor) -> torch.Tensor:
    """softmax(q @ k^T / sqrt(d)) @ v, for (..., T, d) tensors."""
    # TODO: scores, scale by sqrt of the last dim, softmax over the last dim
    raise NotImplementedError


def causal_mask(T: int) -> torch.Tensor:
    """(T, T) bool tensor, True where position t must NOT look
    (every column after the row: the future)."""
    # TODO: torch.triu of ones, diagonal=1
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, 6, 8).unbind(0)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(my_attention(q, k, v), ref, atol=1e-5)

    m = causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m.tolist()[0] == [False, True, True, True]
    assert not m.diagonal().any(), "a token may look at itself"
    scores = (q @ k.transpose(-2, -1) / math.sqrt(8)).masked_fill(
        causal_mask(6), float("-inf"))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(scores.softmax(-1) @ v, ref, atol=1e-5)
    print("All checks passed")
