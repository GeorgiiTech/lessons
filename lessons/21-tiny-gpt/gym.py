"""Lesson 21 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/21-tiny-gpt/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch


def next_char_pairs(ids: torch.Tensor, context: int, start: int):
    """Input x = ids[start : start+context], target y = the same window
    shifted one character to the right."""
    # TODO: two slices of ids
    raise NotImplementedError


def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """logits (B, T, vocab) -> the most likely NEXT token, shape (B, 1).
    Only the last position predicts the next token."""
    # TODO: take position -1, argmax over the vocabulary, keep a dim
    raise NotImplementedError


if __name__ == "__main__":
    ids = torch.arange(10, 20)
    x, y = next_char_pairs(ids, context=4, start=2)
    assert x.tolist() == [12, 13, 14, 15], x
    assert y.tolist() == [13, 14, 15, 16], y

    logits = torch.zeros(2, 3, 5)
    logits[0, -1, 4] = 1.0
    logits[1, -1, 2] = 1.0
    logits[1, 0, 0] = 9.0                      # not the last position
    out = sample_greedy(logits)
    assert out.shape == (2, 1), out.shape
    assert out.flatten().tolist() == [4, 2], out
    print("All checks passed")
