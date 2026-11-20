"""Lesson 18 · Solution to gym.py.

Run:  uv run python lessons/18-project/solution.py
"""

import torch


def to_flag(name: str) -> str:
    """A config field name as a command-line flag:
    "batch_size" -> "--batch-size"."""
    return "--" + name.replace("_", "-")


def check_batch(x: torch.Tensor, y: torch.Tensor) -> None:
    """Fail early with a clear message: x must be (B, 1, 28, 28) float,
    y must be (B,) int64 with the same B. Raise ValueError otherwise."""
    if x.dim() != 4 or tuple(x.shape[1:]) != (1, 28, 28):
        raise ValueError(f"x: expected (B, 1, 28, 28), got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise ValueError(f"x: expected a float tensor, got {x.dtype}")
    if y.shape != (x.shape[0],) or y.dtype != torch.long:
        raise ValueError(f"y: expected ({x.shape[0]},) int64, "
                         f"got {tuple(y.shape)} {y.dtype}")


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
