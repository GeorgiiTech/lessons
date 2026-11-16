"""Lesson 08 · The Training Loop Done Right.

Run:  uv run python lessons/08-training-loop/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import copy

import torch
from torch import nn


# %% [1] One line that picks the best device on any machine.
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# %% [2] Seeds: the same run gives the same numbers.
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)                    # CPU, CUDA and MPS generators


# %% [3] Hold some data back: train / validation split.
def make_split(seed: int = 0):
    set_seed(seed)
    hours = torch.rand(100, 1) * 10
    scores = 5 * hours + 40 + torch.randn(100, 1) * 5
    x = hours / 10
    order = torch.randperm(100)                # shuffle before splitting
    train_idx, val_idx = order[:80], order[80:]
    return x[train_idx], scores[train_idx], x[val_idx], scores[val_idx]


# %% [4] A small network with dropout: it behaves differently in train and eval mode.
def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(), nn.Dropout(p=0.1),
        nn.Linear(64, 1),
    )


def modes_demo(device: torch.device) -> None:
    set_seed(0)
    model = make_model().to(device)
    x = torch.ones(1, 1, device=device)
    model.train()
    a, b = model(x).item(), model(x).item()
    print(f"train mode: {a:.3f} vs {b:.3f}")
    model.eval()
    with torch.inference_mode():
        a, b = model(x).item(), model(x).item()
    print(f"eval mode:  {a:.3f} vs {b:.3f}")


# %% [5] One training epoch and one evaluation, as functions.
def train_one_epoch(model, x, y, loss_fn, optimizer) -> float:
    model.train()                              # dropout on
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, x, y, loss_fn) -> float:
    model.eval()                               # dropout off
    with torch.inference_mode():               # no graph, less memory
        return loss_fn(model(x), y).item()


# %% [6] The full loop: train, validate, keep the best weights.
def fit(seed: int, device: torch.device, epochs: int = 400, log_every: int = 100):
    x_tr, y_tr, x_val, y_val = (t.to(device) for t in make_split())
    set_seed(seed)
    model = make_model().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    best_val, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, x_tr, y_tr, loss_fn, optimizer)
        val_loss = evaluate(model, x_val, y_val, loss_fn)
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
        if epoch % log_every == 0:
            print(f"epoch {epoch}: train {train_loss:7.2f} | val {val_loss:7.2f}")
    model.load_state_dict(best_state)
    print(f"best val loss: {best_val:.2f}")
    return best_val


if __name__ == "__main__":
    device = pick_device()
    print("device:", device)
    x_tr, y_tr, x_val, y_val = make_split()
    print("train:", tuple(x_tr.shape), "val:", tuple(x_val.shape))
    modes_demo(device)
    first = fit(seed=1, device=device)
    second = fit(seed=1, device=device, log_every=10_000)
    print("same seed, same result:", first == second)
