"""Lesson 05 · Gym solution."""

import torch


def grads_at(a: float, b: float) -> tuple[float, float]:
    ta = torch.tensor(a, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    f = ta * tb + tb ** 2
    f.backward()
    return ta.grad.item(), tb.grad.item()


def descend(w0: float, lr: float, steps: int) -> float:
    w = torch.tensor(w0, requires_grad=True)
    for _ in range(steps):
        loss = (w - 4.0) ** 2
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad = None
    return w.item()


if __name__ == "__main__":
    assert grads_at(2.0, 3.0) == (3.0, 8.0)
    assert abs(descend(0.0, 0.1, 50) - 4.0) < 1e-3
    print("All checks passed")
