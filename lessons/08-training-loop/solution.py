"""Lesson 08 · Solution to gym.py."""

import torch
from torch import nn


def split(x: torch.Tensor, y: torch.Tensor, val_fraction: float, seed: int):
    torch.manual_seed(seed)
    order = torch.randperm(len(x))
    n_val = int(len(x) * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        return nn.functional.mse_loss(model(x), y).item()


if __name__ == "__main__":
    x = torch.arange(50.0).unsqueeze(1)
    y = 2 * x
    x_tr, y_tr, x_val, y_val = split(x, y, val_fraction=0.2, seed=0)
    assert x_tr.shape == (40, 1) and x_val.shape == (10, 1)
    assert set(x_tr.flatten().tolist()).isdisjoint(x_val.flatten().tolist())
    assert torch.equal(y_val, 2 * x_val)
    assert not torch.equal(x_tr[:5], x[:5])
    x_tr2, *_ = split(x, y, val_fraction=0.2, seed=0)
    assert torch.equal(x_tr, x_tr2)

    model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1))
    model.train()
    a, b = evaluate(model, x_val, y_val), evaluate(model, x_val, y_val)
    assert isinstance(a, float) and a == b and not model.training
    print(f"train {tuple(x_tr.shape)} | val {tuple(x_val.shape)} | eval loss twice: {a:.1f} = {b:.1f}")
    print("All checks passed")
