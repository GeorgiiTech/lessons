"""Lesson 16 · Reproducible experiments: seeds, configs, checkpoints, logs.

Run:  uv run python lessons/16-experiments/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
Runs on the CPU (bit-for-bit repeatable) in under a minute.
Logs go to runs/ (open them with: uv run tensorboard --logdir runs).
"""

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

QUICK = os.environ.get("LESSONS_QUICK") == "1"
RUNS = Path("runs")


# %% [1] One config object instead of numbers scattered through the code.
@dataclass
class Config:
    lr: float = 1e-3
    epochs: int = 6
    batch_size: int = 64
    hidden: int = 128
    seed: int | None = 0                     # None = no seed (bad idea)
    name: str = "baseline"


# %% [2] Seeds: Python, NumPy and PyTorch each have their own generator.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                  # CPU and all GPUs


def make_model(cfg: Config) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(784, cfg.hidden), nn.ReLU(),
                         nn.Dropout(0.2), nn.Linear(cfg.hidden, 10))


def load_data(n_train: int, n_val: int):
    tf = transforms.ToTensor()
    train = datasets.FashionMNIST("data", train=True, download=True,
                                  transform=tf)
    x = torch.stack([train[i][0] for i in range(n_train + n_val)])
    y = torch.tensor([train[i][1] for i in range(n_train + n_val)])
    return (TensorDataset(x[:n_train], y[:n_train]),
            TensorDataset(x[n_train:], y[n_train:]))


# %% [3] Checkpoints: everything needed to continue, not just the weights.
def save_checkpoint(path: Path, model, optimizer, epoch: int,
                    cfg: Config) -> None:
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": asdict(cfg),
                "rng": torch.get_rng_state()}, path)


def load_checkpoint(path: Path, model, optimizer) -> int:
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng"])
    return ckpt["epoch"]


# %% [4] One run: config in, metrics + checkpoint + TensorBoard log out.
def run(cfg: Config, train_set, val_set, stop_after: int | None = None,
        resume: Path | None = None) -> float:
    if cfg.seed is not None:
        set_seed(cfg.seed)
    model = make_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    start = 0
    if resume is not None:
        start = load_checkpoint(resume, model, optimizer)
    folder = RUNS / cfg.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    writer = SummaryWriter(folder)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    for epoch in range(start + 1, cfg.epochs + 1):
        model.train()
        for x, y in loader:
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = accuracy(model, val_set)
        writer.add_scalar("loss/train", loss.item(), epoch)
        writer.add_scalar("accuracy/val", acc, epoch)
        save_checkpoint(folder / "last.pt", model, optimizer, epoch, cfg)
        if stop_after == epoch:
            print(f"{cfg.name}: stopped after epoch {epoch}")
            break
    writer.close()
    return acc


@torch.no_grad()
def accuracy(model, dataset) -> float:
    model.eval()
    x, y = dataset.tensors
    return (model(x).argmax(1) == y).float().mean().item()


if __name__ == "__main__":
    torch.set_num_threads(1)                 # one thread: exact repeats
    train_set, val_set = load_data(*((1000, 500) if QUICK else (10_000,
                                                                2_000)))
    epochs = 2 if QUICK else 6

    # Two runs without a seed, then two runs with seed 0.
    for seed in (None, None, 0, 0):
        acc = run(Config(seed=seed, epochs=epochs, name=f"seed-{seed}"),
                  train_set, val_set)
        print(f"seed {seed}:  val acc {acc:.4f}")

    # The same config, five seeds: how big is pure luck?
    accs = [run(Config(seed=s, epochs=epochs, name=f"spread-{s}"),
                train_set, val_set) for s in range(5)]
    print("5 seeds:", " ".join(f"{a:.4f}" for a in accs),
          f" mean {np.mean(accs):.4f}  std {np.std(accs):.4f}")

    # Interrupt after epoch 3, resume from the checkpoint, compare.
    full = run(Config(epochs=epochs, name="full"), train_set, val_set)
    cfg = Config(epochs=epochs, name="resumed")
    run(cfg, train_set, val_set, stop_after=epochs // 2)
    resumed = run(cfg, train_set, val_set, resume=RUNS / "resumed/last.pt")
    print(f"full run {full:.4f}  stopped + resumed {resumed:.4f}")

    # A tiny sweep: one config change per run, each in its own folder.
    for lr in (1e-2, 1e-3, 1e-4):
        acc = run(Config(lr=lr, epochs=epochs, name=f"lr-{lr:g}"),
                  train_set, val_set)
        print(f"lr {lr:g}:  val acc {acc:.4f}")
