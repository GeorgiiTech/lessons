"""Lesson 01 · Gym solution."""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_grid() -> torch.Tensor:
    return torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)


def row_sums(t: torch.Tensor) -> torch.Tensor:
    return t.sum(dim=1)


if __name__ == "__main__":
    device = get_device()
    grid = make_grid()
    assert grid.shape == (3, 4) and grid.dtype == torch.float32
    assert row_sums(grid.to(device)).cpu().tolist() == [10.0, 26.0, 42.0]
    print("All checks passed")
