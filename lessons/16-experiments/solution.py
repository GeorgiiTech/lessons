"""Lesson 16 · Solution to gym.py.

Run:  uv run python lessons/16-experiments/solution.py
"""

import tempfile
from pathlib import Path

import torch
from torch import nn


def run_name(config: dict) -> str:
    """A folder name that says what changed: keys sorted, "key=value"
    joined by "_", e.g. {"lr": 0.001, "seed": 0} -> "lr=0.001_seed=0"."""
    return "_".join(f"{k}={v}" for k, v in sorted(config.items()))


def save(path: Path, model: nn.Module, optimizer, epoch: int) -> None:
    """Save the weights, the optimizer state and the epoch in one file."""
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch}, path)


def load(path: Path, model: nn.Module, optimizer) -> int:
    """Restore model and optimizer from `path`, return the saved epoch."""
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]


if __name__ == "__main__":
    assert run_name({"seed": 0, "lr": 0.001}) == "lr=0.001_seed=0"

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    model(torch.randn(8, 4)).sum().backward()
    opt.step()                                  # Adam now has state
    path = Path(tempfile.mkdtemp()) / "ckpt.pt"
    save(path, model, opt, epoch=3)

    fresh = nn.Linear(4, 2)
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=0.1)
    assert load(path, fresh, fresh_opt) == 3, "returns the epoch"
    assert torch.equal(fresh.weight, model.weight), "same weights"
    state = fresh_opt.state_dict()["state"]
    assert state and state[0]["step"] == 1, "optimizer state restored too"
    print("All checks passed")
