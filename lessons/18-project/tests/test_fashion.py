"""Fast tests for the fashion package: run with  pytest lessons/18-project"""

import torch
import torch.nn.functional as F

from fashion.config import parse_args
from fashion.model import make_mlp


def test_output_shape() -> None:
    model = make_mlp(hidden=32)
    assert model(torch.randn(5, 1, 28, 28)).shape == (5, 10)


def test_first_loss_is_ln10() -> None:
    torch.manual_seed(0)
    x, y = torch.randn(256, 1, 28, 28) * 0.1, torch.randint(0, 10, (256,))
    loss = F.cross_entropy(make_mlp()(x), y).item()
    assert abs(loss - 2.303) < 0.1


def test_overfits_one_batch() -> None:
    torch.manual_seed(0)
    x, y = torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))
    model = make_mlp(hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_cli_parses_numbers() -> None:
    cfg = parse_args(["--epochs", "5", "--lr", "0.01", "--n-train", "100"])
    assert (cfg.epochs, cfg.lr, cfg.n_train) == (5, 0.01, 100)
