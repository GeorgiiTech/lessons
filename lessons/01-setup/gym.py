"""Lesson 01 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/01-setup/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch


def get_device() -> torch.device:
    """cuda if available, else mps if available, else cpu."""
    # TODO: write it without looking at lesson.py
    raise NotImplementedError


def make_grid() -> torch.Tensor:
    """A float32 tensor of shape (3, 4): the numbers 1..12, row by row.
    Hint: torch.arange and .reshape (more about reshape in lesson 04)."""
    # TODO
    raise NotImplementedError


def row_sums(t: torch.Tensor) -> torch.Tensor:
    """The sum of each row of a 2-D tensor, without a Python loop.
    Hint: .sum(dim=...)."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    device = get_device()
    assert isinstance(device, torch.device) and device.type in {"cpu", "cuda", "mps"}

    grid = make_grid()
    assert grid.shape == (3, 4), f"shape should be (3, 4), got {tuple(grid.shape)}"
    assert grid.dtype == torch.float32, f"dtype should be float32, got {grid.dtype}"
    assert grid[0, 0] == 1 and grid[2, 3] == 12

    sums = row_sums(grid.to(device))
    assert sums.device.type == device.type, "the result should stay on the same device"
    assert sums.cpu().tolist() == [10.0, 26.0, 42.0], f"got {sums.tolist()}"
    print("All checks passed")
