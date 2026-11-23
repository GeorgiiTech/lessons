"""Lesson 26 · Solution to gym.py.

Run:  uv run python lessons/26-serve/solution.py
"""

from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn


def roundtrip(make_model, model: nn.Module, path: Path) -> nn.Module:
    """Save model's state_dict to path, build a fresh model with
    make_model(), load the weights into it and switch it to eval mode."""
    torch.save(model.state_dict(), path)
    fresh = make_model()
    fresh.load_state_dict(torch.load(path, weights_only=True))
    return fresh.eval()


def top_class(logits: np.ndarray) -> tuple[int, float]:
    """One row of logits -> (index of the best class, its softmax
    probability). Subtract the max before exp, so large logits are safe."""
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top = int(probs.argmax())
    return top, float(probs[top])


if __name__ == "__main__":
    def make():
        return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))

    torch.manual_seed(0)
    model, x = make(), torch.randn(5, 3)
    with tempfile.TemporaryDirectory() as d:
        new = roundtrip(make, model, Path(d) / "m.pt")
        assert new is not model and not new.training
        assert torch.equal(new(x), model(x))

    i, p = top_class(np.array([1.0, 3.0, 2.0]))
    assert i == 1 and abs(p - 0.6652) < 1e-3, (i, p)
    i, p = top_class(np.array([1000.0, 0.0]))
    assert i == 0 and abs(p - 1.0) < 1e-6, "subtract the max first"
    print("All checks passed")
