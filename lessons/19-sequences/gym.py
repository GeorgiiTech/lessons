"""Lesson 19 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/19-sequences/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch
from torch import nn


def lookup(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """What nn.Embedding does: pick row `id` of `weight` for every id.
    ids can have any shape; the result gets one extra last dimension."""
    # TODO: plain tensor indexing does it in one expression
    raise NotImplementedError


def pad_batch(seqs: list[list[int]], pad_id: int = 0):
    """Sequences of different lengths -> one (B, T_max) tensor filled
    with pad_id, plus a tensor with the real lengths."""
    # TODO: torch.full((B, T_max), pad_id), copy each sequence in
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    emb = nn.Embedding(10, 4)
    ids = torch.tensor([[1, 5, 5], [0, 9, 2]])
    assert torch.equal(lookup(ids, emb.weight), emb(ids)), "same as nn"

    x, lengths = pad_batch([[4, 2, 7], [3], [5, 1]])
    assert x.tolist() == [[4, 2, 7], [3, 0, 0], [5, 1, 0]]
    assert lengths.tolist() == [3, 1, 2]
    x, _ = pad_batch([[1], [2, 3]], pad_id=-1)
    assert x.tolist() == [[1, -1], [2, 3]]
    print("All checks passed")
