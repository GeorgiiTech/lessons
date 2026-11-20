"""Lesson 18 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/18-project/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch


def to_flag(name: str) -> str:
    """A config field name as a command-line flag:
    "batch_size" -> "--batch-size"."""
    # TODO: "--" + name with "_" replaced by "-"
    raise NotImplementedError


def check_batch(x: torch.Tensor, y: torch.Tensor) -> None:
    """Fail early with a clear message: x must be (B, 1, 28, 28) float,
    y must be (B,) int64 with the same B. Raise ValueError otherwise."""
    # TODO: compare x.shape[1:], x.dtype, y.shape, y.dtype, raise ValueError
    raise NotImplementedError


if __name__ == "__main__":
    assert to_flag("batch_size") == "--batch-size"
    assert to_flag("lr") == "--lr"

    check_batch(torch.zeros(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
    bad = [(torch.zeros(4, 28, 28), torch.zeros(4, dtype=torch.long)),
           (torch.zeros(4, 1, 28, 28), torch.zeros(4)),
           (torch.zeros(4, 1, 28, 28), torch.zeros(5, dtype=torch.long))]
    for x, y in bad:
        try:
            check_batch(x, y)
        except ValueError:
            continue
        raise AssertionError(f"should fail: {tuple(x.shape)} {y.dtype}")
    print("All checks passed")
