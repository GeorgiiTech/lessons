"""Training entry point:  python -m fashion.train --epochs 3 --lr 1e-3"""

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from fashion.config import Config, parse_args
from fashion.data import loaders
from fashion.model import make_mlp


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = sum((model(x).argmax(1) == y).sum().item() for x, y in loader)
    return correct / len(loader.dataset)


def main(cfg: Config) -> float:
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = loaders(cfg.n_train, cfg.batch_size, cfg.data)
    model = make_mlp(cfg.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer)
        acc = accuracy(model, val_loader)
        print(f"epoch {epoch}  loss {loss:.3f}  val acc {acc:.3f}")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return acc


if __name__ == "__main__":
    main(parse_args())
