"""Lesson 05 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/05-autograd/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch


def grads_at(a: float, b: float) -> tuple[float, float]:
    """Return (df/da, df/db) for f = a*b + b**2, using autograd."""
    # TODO
    raise NotImplementedError


def descend(w0: float, lr: float, steps: int) -> float:
    """Minimize loss = (w - 4)**2 with plain gradient descent.
    Start at w0, do `steps` updates, return the final w.
    Remember: update inside torch.no_grad(), and reset the grad."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    da, db = grads_at(2.0, 3.0)
    assert abs(da - 3.0) < 1e-6, f"df/da should be b = 3, got {da}"
    assert abs(db - 8.0) < 1e-6, f"df/db should be a + 2b = 8, got {db}"

    w = descend(0.0, 0.1, 50)
    assert abs(w - 4.0) < 1e-3, f"w should approach 4, got {w}"
    print("All checks passed")
