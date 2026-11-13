"""Lesson 07 · nn.Module, Loss Functions and Optimizers.

Run:  uv run python lessons/07-nn-module/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import torch
from torch import nn


# %% [1] The same data as lesson 06, as columns: (100, 1).
def make_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    hours = torch.rand(100) * 10
    scores = 5 * hours + 40 + torch.randn(100) * 5
    x = (hours / 10).unsqueeze(1)              # scaled, shape (100, 1)
    y = scores.unsqueeze(1)                    # shape (100, 1)
    print("x:", tuple(x.shape), "y:", tuple(y.shape))
    return x, y


# %% [2] nn.Linear: w and b, created and tracked for you.
def linear_layer() -> None:
    torch.manual_seed(1)
    layer = nn.Linear(in_features=1, out_features=1)
    for name, p in layer.named_parameters():
        print(f"{name}: shape {tuple(p.shape)}, requires_grad={p.requires_grad}")


# %% [3] Your own model: a class with __init__ and forward.
class ScoreModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def show_model() -> None:
    model = ScoreModel()
    print(model)
    print("parameters:", sum(p.numel() for p in model.parameters()))


# %% [4] Loss functions: nn.MSELoss is our mse from lesson 06.
def loss_check(x: torch.Tensor, y: torch.Tensor) -> None:
    pred = torch.zeros_like(y)
    ours = ((pred - y) ** 2).mean()
    theirs = nn.MSELoss()(pred, y)
    print(f"ours {ours:.2f} | nn.MSELoss {theirs:.2f}")


# %% [5] The optimizer does the update. The loop is now five lines.
def train(x, y, opt_name: str, lr: float, epochs: int, log_every: int) -> nn.Module:
    torch.manual_seed(1)
    model = ScoreModel()
    loss_fn = nn.MSELoss()
    opt_cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[opt_name]
    optimizer = opt_cls(model.parameters(), lr=lr)
    for epoch in range(epochs + 1):
        pred = model(x)                        # forward
        loss = loss_fn(pred, y)                # loss
        optimizer.zero_grad()                  # reset
        loss.backward()                        # gradients
        optimizer.step()                       # update
        if epoch % log_every == 0:
            print(f"{opt_name} epoch {epoch:3d}: loss {loss.item():8.2f}")
    return model


# %% [6] Read the learned numbers back.
def learned(model: nn.Module, x_scale: float = 10) -> None:
    w = model.linear.weight.item() / x_scale   # back to points per hour
    b = model.linear.bias.item()
    print(f"w {w:.2f} per hour, b {b:.2f}")


# %% [7] Save and load: the state_dict.
def save_load(model: nn.Module, x: torch.Tensor) -> None:
    torch.save(model.state_dict(), "score_model.pt")
    fresh = ScoreModel()
    fresh.load_state_dict(torch.load("score_model.pt"))
    with torch.no_grad():
        same = torch.equal(model(x), fresh(x))
    print("loaded model gives the same predictions:", same)


if __name__ == "__main__":
    x, y = make_data()
    linear_layer()
    show_model()
    loss_check(x, y)
    model = train(x, y, "sgd", lr=0.5, epochs=100, log_every=25)
    learned(model)
    train(x, y, "adam", lr=0.5, epochs=100, log_every=50)      # same lr: too slow for Adam
    adam = train(x * 10, y, "adam", lr=2.0, epochs=100, log_every=50)   # raw hours, no scaling
    learned(adam, x_scale=1)
    save_load(model, x)
