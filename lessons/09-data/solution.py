"""Lesson 09 · Solution to gym.py."""

import torch
from torch.utils.data import DataLoader, Dataset


class Squares(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        x = torch.tensor([float(i)])
        return x, x * x


def make_loader(dataset: Dataset, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)


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
