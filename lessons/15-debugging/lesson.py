"""Lesson 15 · Debugging neural networks: a checklist with real bugs.

Run:  uv run python lessons/15-debugging/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
Runs on the CPU in about a minute (FashionMNIST is in ./data).
"""

import math
import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

QUICK = os.environ.get("LESSONS_QUICK") == "1"


def make_mlp() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 10))


def load(n: int, train: bool = True):
    data = datasets.FashionMNIST("data", train=train, download=True,
                                 transform=transforms.ToTensor())
    x = torch.stack([data[i][0] for i in range(n)])
    y = torch.tensor([data[i][1] for i in range(n)])
    return x, y


# %% [1] Check 1: the loss before training. 10 classes → ln(10) ≈ 2.303.
def first_loss(x, y) -> None:
    loss = F.cross_entropy(make_mlp()(x), y)
    print(f"first loss {loss.item():.3f}  expected {math.log(10):.3f}")


# %% [2] Check 2: overfit one batch. A correct setup reaches ~0 loss.
def overfit_one_batch(x, y, bug: str = "", steps: int = 200,
                      name: str = "") -> list[float]:
    model = make_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        logits = model(x)
        if bug == "softmax":
            logits = logits.softmax(dim=1)         # BUG: CE does it again
        loss = F.cross_entropy(logits, y)
        if bug != "no zero_grad":
            optimizer.zero_grad()                  # BUG when skipped
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    shown = "  ".join(f"{losses[i]:.3f}" for i in (0, 49, steps - 1))
    label = name or bug or "correct"
    print(f"{label:12s} loss at steps 1/50/{steps}: {shown}")
    return losses


# %% [3] Check 3: look at the data. Shuffled labels still overfit a batch!
def shuffled_labels(x, y, x_val, y_val) -> None:
    for name, labels in (("right labels", y),
                         ("shuffled", y[torch.randperm(len(y))])):
        model = make_mlp()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2 if QUICK else 5):
            for i in range(0, len(x), 64):
                loss = F.cross_entropy(model(x[i:i + 64]), labels[i:i + 64])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            acc = (model(x_val).argmax(1) == y_val).float().mean().item()
        print(f"{name:12s} train loss {loss.item():.3f}  val acc {acc:.3f}")


# %% [4] Check 4: shapes. (100, 1) - (100,) broadcasts to (100, 100).
def broadcasting_bug() -> None:
    torch.manual_seed(0)
    x = torch.randn(100, 1)
    y = 3 * x[:, 0] + 1 + 0.1 * torch.randn(100)     # y = 3x + 1, (100,)
    for name in ("bug", "fixed"):
        torch.manual_seed(0)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(200):
            pred = model(x)                          # shape (100, 1)
            if name == "fixed":
                pred = pred.squeeze(1)               # shape (100,)
            loss = ((pred - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        w, b = model.weight.item(), model.bias.item()
        print(f"{name:5s} {tuple((pred - y).shape)}  w {w:.2f}  b {b:.2f}")


# %% [5] Check 5: are all the weights registered? A list hides them.
class ListNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [nn.Linear(784, 256), nn.Linear(256, 10)]   # BUG


class ModuleListNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(784, 256),
                                     nn.Linear(256, 10)])


def registered() -> None:
    for net in (ListNet(), ModuleListNet()):
        n = sum(p.numel() for p in net.parameters())
        print(f"{type(net).__name__:13s} parameters: {n:,}")
    try:
        torch.optim.Adam(ListNet().parameters())
    except ValueError as e:
        print("Adam:", e)


# %% [6] Check 6: NaN and infinity. log(softmax) breaks, log_softmax not.
def nan_hunt() -> None:
    logits = torch.tensor([[1000.0, 0.0]])          # very confident
    target = torch.tensor([1])                       # ...and wrong
    naive = -torch.log(logits.softmax(dim=1))[0, target]
    print("log(softmax):", naive.item())
    print("cross_entropy:", F.cross_entropy(logits, target).item())
    loss = naive * 0.0                               # inf * 0 = nan
    print("inf * 0 =", loss.item(), " finite?", torch.isfinite(loss).item())


if __name__ == "__main__":
    x, y = load(1000 if QUICK else 10_000)
    x_val, y_val = load(500 if QUICK else 2000, train=False)
    batch_x, batch_y = x[:64], y[:64]
    first_loss(batch_x, batch_y)
    for bug in ("", "softmax", "no zero_grad"):
        overfit_one_batch(batch_x, batch_y, bug)
    torch.manual_seed(1)
    overfit_one_batch(batch_x, batch_y[torch.randperm(64)], name="shuffled")
    shuffled_labels(x, y, x_val, y_val)
    broadcasting_bug()
    registered()
    nan_hunt()
