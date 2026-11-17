"""Lesson 09 · Datasets and DataLoaders.

Run:  uv run python lessons/09-data/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import os
import time

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

# Small sizes when tests run this file (LESSONS_QUICK=1).
QUICK = os.environ.get("LESSONS_QUICK") == "1"


# %% [1] A Dataset answers two questions: how many examples, and give me number i.
class StudentScores(Dataset):
    def __init__(self, n: int = 100, transform=None, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.hours = torch.rand(n, 1, generator=g) * 10
        self.scores = 5 * self.hours + 40 + torch.randn(n, 1, generator=g) * 5
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hours)

    def __getitem__(self, i: int):
        x, y = self.hours[i], self.scores[i]
        if self.transform:
            x = self.transform(x)
        return x, y


# %% [2] A transform runs on every example as it is read.
def scale_hours(x: torch.Tensor) -> torch.Tensor:
    return x / 10


# %% [3] A slow dataset: every example takes 10 ms to "read from disk".
class SlowDataset(Dataset):
    def __len__(self) -> int:
        return 64

    def __getitem__(self, i: int):
        time.sleep(0.01)
        return torch.tensor([float(i)])


def time_loader(workers: int, epochs: int = 3) -> list[float]:
    loader = DataLoader(SlowDataset(), batch_size=16, num_workers=workers,
                        persistent_workers=workers > 0)
    times = []
    for _ in range(epochs):
        start = time.perf_counter()
        for _ in loader:
            pass
        times.append(round(time.perf_counter() - start, 2))
    return times


# %% [4] Mini-batch training: many small steps per epoch.
def fit(train_set, val_set, batch_size: int, epochs: int, seed: int = 1) -> float:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:                    # one step per batch
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        total = sum(loss_fn(model(x), y).item() * len(x) for x, y in val_loader)
    return total / len(val_set)


# %% [5] Examples of different lengths: a custom collate_fn pads them.
def pad_collate(batch):
    lengths = torch.tensor([len(seq) for seq in batch])
    return pad_sequence(batch, batch_first=True), lengths


if __name__ == "__main__":
    torch.manual_seed(0)
    data = StudentScores(transform=scale_hours)
    print("len:", len(data))
    x, y = data[0]
    print("data[0]:", round(x.item(), 3), round(y.item(), 2))

    # TensorDataset: the shortcut when everything already fits in tensors.
    quick = TensorDataset(data.hours / 10, data.scores)
    print("TensorDataset[0]:", [round(t.item(), 2) for t in quick[0]])

    g = torch.Generator().manual_seed(0)
    train_set, val_set = random_split(data, [80, 20], generator=g)
    print("train:", len(train_set), "val:", len(val_set))

    loader = DataLoader(train_set, batch_size=16, shuffle=True)
    xb, yb = next(iter(loader))
    print("batches:", len(loader), "| x:", tuple(xb.shape), "y:", tuple(yb.shape))
    print("batch sizes (32):", [len(b[0]) for b in DataLoader(train_set, batch_size=32)])
    print("drop_last:", [len(b[0]) for b in DataLoader(train_set, batch_size=32, drop_last=True)])

    small = DataLoader(range(10), batch_size=5, shuffle=True)
    for epoch in range(2):
        print(f"epoch {epoch}:", [b.tolist() for b in small])

    for bs in (80, 16):
        val = fit(train_set, val_set, batch_size=bs, epochs=20)
        print(f"batch {bs:2d}: {20 * (80 // bs):3d} steps in 20 epochs, val loss {val:6.2f}")

    if not QUICK:
        for w in (0, 4):
            print(f"num_workers={w}: seconds per epoch {time_loader(w)}")

    sentences = [torch.tensor([5, 8, 2]), torch.tensor([7]), torch.tensor([3, 9])]
    padded, lengths = next(iter(DataLoader(sentences, batch_size=3, collate_fn=pad_collate)))
    print("padded:", padded.tolist(), "lengths:", lengths.tolist())
