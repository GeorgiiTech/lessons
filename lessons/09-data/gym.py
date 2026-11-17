"""Lesson 09 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/09-data/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch
from torch.utils.data import DataLoader, Dataset


class Squares(Dataset):
    """n examples: example i is (tensor([i]), tensor([i * i])), both float32."""

    def __init__(self, n: int):
        self.n = n

    # TODO: __len__ and __getitem__


def make_loader(dataset: Dataset, batch_size: int, seed: int) -> DataLoader:
    """A shuffled DataLoader whose order depends only on `seed` (use a torch.Generator)."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    data = Squares(50)
    assert len(data) == 50, "__len__ gives the number of examples"
    x, y = data[7]
    assert x.tolist() == [7.0] and y.tolist() == [49.0], "example 7 is (7, 49)"
    assert x.dtype == torch.float32 and y.dtype == torch.float32, "float32 tensors"

    loader = make_loader(data, batch_size=16, seed=0)
    batches = list(loader)
    assert len(batches) == 4, "50 examples in batches of 16 → 4 batches"
    assert [len(b[0]) for b in batches] == [16, 16, 16, 2], "the last batch is smaller"
    seen = sorted(int(v) for xb, _ in batches for v in xb.flatten())
    assert seen == list(range(50)), "every example exactly once per epoch"
    assert seen != [int(v) for xb, _ in batches for v in xb.flatten()], "shuffle the order"
    again = [int(v) for xb, _ in make_loader(data, 16, seed=0) for v in xb.flatten()]
    assert again == [int(v) for xb, _ in batches for v in xb.flatten()], "same seed, same order"
    for xb, yb in batches:
        assert torch.equal(yb, xb * xb), "x and y stay paired after shuffling"
    print("All checks passed")
