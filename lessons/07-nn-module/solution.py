"""Lesson 07 · Solution to gym.py."""

import torch
from torch import nn


class TwoFeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int) -> float:
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(epochs):
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(200, 2) * torch.tensor([10.0, 8.0])
    y = x @ torch.tensor([[4.0], [3.0]]) + 20

    model = TwoFeatureModel()
    assert sum(p.numel() for p in model.parameters()) == 3
    loss = fit(model, x, y, epochs=3000)
    w = model.linear.weight.detach().flatten().tolist()
    print(f"loss {loss:.4f} | w = {w[0]:.2f}, {w[1]:.2f} | b = {model.linear.bias.item():.2f}")
    assert loss < 1e-2
    print("All checks passed")
