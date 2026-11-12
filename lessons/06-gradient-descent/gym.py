"""Lesson 06 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/06-gradient-descent/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch


def mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error: one number."""
    # TODO
    raise NotImplementedError


def fit(x: torch.Tensor, y: torch.Tensor, lr: float, epochs: int) -> tuple[float, float]:
    """Fit y = w * x + b with plain gradient descent (start at w = b = 0).
    Return (w, b) as Python floats."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    assert abs(mse(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 0.0])).item() - 5.0) < 1e-6, "mse([1, 3], [0, 0]) = 5"

    temperature = torch.linspace(0, 5, 20)       # a hot day -> more ice cream
    sales = 3 * temperature - 2                  # no noise: the answer is exact
    w, b = fit(temperature, sales, lr=0.05, epochs=2000)
    assert abs(w - 3) < 1e-2 and abs(b + 2) < 1e-2, f"expected w=3, b=-2, got w={w:.3f}, b={b:.3f}"
    print("All checks passed")
