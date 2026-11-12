"""Lesson 06 · Solution to gym.py."""

import torch


def mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).mean()


def fit(x: torch.Tensor, y: torch.Tensor, lr: float, epochs: int) -> tuple[float, float]:
    w = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    for _ in range(epochs):
        loss = mse(w * x + b, y)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None
        b.grad = None
    return w.item(), b.item()


if __name__ == "__main__":
    assert abs(mse(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 0.0])).item() - 5.0) < 1e-6

    temperature = torch.linspace(0, 5, 20)
    sales = 3 * temperature - 2
    w, b = fit(temperature, sales, lr=0.05, epochs=2000)
    assert abs(w - 3) < 1e-2 and abs(b + 2) < 1e-2
    print(f"w = {w:.3f}, b = {b:.3f}")
    print("All checks passed")
