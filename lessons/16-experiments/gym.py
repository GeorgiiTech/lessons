"""Lesson 16 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/16-experiments/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import tempfile
from pathlib import Path

import torch
from torch import nn


def run_name(config: dict) -> str:
    """A folder name that says what changed: keys sorted, "key=value"
    joined by "_", e.g. {"lr": 0.001, "seed": 0} -> "lr=0.001_seed=0"."""
    # TODO: sorted(config.items()), f"{k}={v}", "_".join(...)
    raise NotImplementedError


def save(path: Path, model: nn.Module, optimizer, epoch: int) -> None:
    """Save the weights, the optimizer state and the epoch in one file."""
    # TODO: torch.save({...}, path) with three keys
    raise NotImplementedError


def load(path: Path, model: nn.Module, optimizer) -> int:
    """Restore model and optimizer from `path`, return the saved epoch."""
    # TODO: torch.load, load_state_dict twice, return the epoch
    raise NotImplementedError


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
