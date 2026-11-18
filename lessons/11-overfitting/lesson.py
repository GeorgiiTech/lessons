"""Lesson 11 · Overfitting and how to fight it.

Run:  uv run python lessons/11-overfitting/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
Uses FashionMNIST from lesson 10 (downloaded into ./data on the first run).
"""

import copy
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Small sizes when tests run this file (LESSONS_QUICK=1).
QUICK = os.environ.get("LESSONS_QUICK") == "1"
EPOCHS = 3 if QUICK else 60


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# %% [1] Few examples, a big model: the classic recipe for overfitting.
def small_split(root: str = "data", n_train: int = 1000, n_val: int = 5000):
    to_tensor = transforms.ToTensor()
    full = datasets.FashionMNIST(root, train=True, download=True,
                                 transform=to_tensor)
    test = datasets.FashionMNIST(root, train=False, download=True,
                                 transform=to_tensor)
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(full), generator=g)
    val = Subset(full, order[:n_val].tolist())       # the same val set always
    train = Subset(full, order[n_val:n_val + n_train].tolist())
    return train, val, test


def make_model(p_drop: float = 0.0) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p_drop),
        nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p_drop),
        nn.Linear(512, 10),
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_fn(logits, y).item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total / n, correct / n


# %% [2] One training run; optional weight decay and early stopping.
def fit(train_loader, val_loader, device, p_drop=0.0, weight_decay=0.0,
        patience=None, epochs=EPOCHS, seed=0):
    torch.manual_seed(seed)
    model = make_model(p_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_loss, train_acc = evaluate(model, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append((epoch, train_loss, val_loss, train_acc, val_acc))
        if val_loss < best:                          # %% [5] early stopping
            best, bad = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def report(name, history):
    b = min(history, key=lambda h: h[2])
    print(f"{name:12s} epochs {history[-1][0]:2d}  best val loss {b[2]:.3f} "
          f"(ep {b[0]:2d}, acc {b[4]:.3f})  last {history[-1][2]:.3f}")


if __name__ == "__main__":
    device = pick_device()
    train, val, test = small_split()
    if QUICK:
        val, test = Subset(val, range(500)), Subset(test, range(500))
    g = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train, batch_size=64, shuffle=True, generator=g)
    val_loader = DataLoader(val, batch_size=500)
    test_loader = DataLoader(test, batch_size=500)
    print("train:", len(train), "val:", len(val), "test:", len(test))
    print("parameters:", sum(p.numel() for p in make_model().parameters()))

    # %% [3] Plain training: the train loss falls, the val loss turns back up.
    model, plain = fit(train_loader, val_loader, device)
    for e, tl, vl, ta, va in plain:
        if e in (1, 5, 10, 15, 20, 30, 40, 50, 60):
            print(f"epoch {e:2d}  train loss {tl:.3f}  val loss {vl:.3f}  "
                  f"train acc {ta:.3f}  val acc {va:.3f}")
    report("plain", plain)

    # %% [4] Dropout and weight decay.
    _, drop = fit(train_loader, val_loader, device, p_drop=0.5)
    report("dropout 0.5", drop)
    _, decay = fit(train_loader, val_loader, device, weight_decay=0.5)
    report("weight decay", decay)

    # %% [5] Early stopping: stop after `patience` epochs without progress.
    model, early = fit(train_loader, val_loader, device, p_drop=0.5,
                       patience=5)
    report("drop + early", early)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"test loss {test_loss:.3f}  test acc {test_acc:.3f}")

    # %% [6] The best cure of all: more data (10 times more examples).
    big, _, _ = small_split(n_train=10_000 if not QUICK else 2000)
    big_loader = DataLoader(big, batch_size=64, shuffle=True, generator=g)
    _, more = fit(big_loader, val_loader, device)
    report("10x data", more)
    if not QUICK:                              # curves for the video
        torch.save({"plain": plain, "dropout": drop, "decay": decay,
                    "early": early, "more": more}, "data/lesson11_history.pt")
