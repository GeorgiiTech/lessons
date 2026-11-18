"""Lesson 11 · Gym solution."""

import torch


def my_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training:
        return x
    keep = (torch.rand_like(x) >= p).float()
    return x * keep / (1 - p)          # inverted dropout: the mean stays


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.bad = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best:
            self.best, self.bad = val_loss, 0
        else:
            self.bad += 1
        return self.bad >= self.patience


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.ones(100_000)
    y = my_dropout(x, p=0.3, training=True)
    zeros = (y == 0).float().mean().item()
    assert abs(zeros - 0.3) < 0.01, "about 30% of the values are zero"
    assert abs(y.mean().item() - 1.0) < 0.02, "the mean stays the same"
    assert torch.equal(my_dropout(x, 0.3, training=False), x), "eval: no-op"

    stopper = EarlyStopping(patience=3)
    losses = [0.9, 0.7, 0.6, 0.65, 0.61, 0.5, 0.55, 0.56, 0.57]
    stops = [stopper.step(v) for v in losses]
    assert stops.index(True) == 8, "stop after 3 epochs without a new best"
    assert not any(stops[:8]), "0.5 at epoch 6 resets the count"
    print("All checks passed")
