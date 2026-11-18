"""Lesson 11 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/11-overfitting/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch


def my_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Zero each element with probability p, scale the rest by 1 / (1 - p).
    In eval mode (training=False) return x unchanged."""
    # TODO
    raise NotImplementedError


class EarlyStopping:
    """step(val_loss) returns True when it is time to stop: `patience`
    epochs in a row without a new best (lowest) validation loss."""

    def __init__(self, patience: int):
        self.patience = patience
        # TODO: remember the best loss so far and the bad epochs in a row

    def step(self, val_loss: float) -> bool:
        # TODO
        raise NotImplementedError


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
