"""Lesson 13 · Training tricks: init, normalization, schedules, clipping.

Run:  uv run python lessons/13-tricks/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
The first run downloads FashionMNIST (about 30 MB) into ./data.
"""

import math
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

QUICK = os.environ.get("LESSONS_QUICK") == "1"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# %% [1] Initialization: push random data through 20 layers, watch the std.
def deep_mlp(depth: int, init: str = "default", norm: bool = False):
    layers = [nn.Flatten()]
    for i in range(depth):
        linear = nn.Linear(784 if i == 0 else 256, 256)
        if init == "small":
            nn.init.normal_(linear.weight, std=0.01)
        elif init == "big":
            nn.init.normal_(linear.weight, std=1.0)
        elif init == "kaiming":
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
        if init != "default":
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(256, 10))
    return nn.Sequential(*layers)


@torch.no_grad()
def activation_stds(model: nn.Module, x: torch.Tensor) -> list[float]:
    stds = []
    for layer in model:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            stds.append(x.std().item())
    return stds


def init_experiment() -> None:
    x = torch.randn(512, 784)
    for init in ("big", "small", "default", "kaiming"):
        torch.manual_seed(0)
        stds = activation_stds(deep_mlp(20, init), x)
        shown = "  ".join(f"{stds[i]:.2g}" for i in (0, 4, 9, 19))
        print(f"{init:8s} std after layers 1/5/10/20: {shown}")


# %% [2] Normalization: BatchNorm averages over the batch, LayerNorm
#        over the features of each sample.
def norm_demo() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 3) * 5 + 10             # 4 samples, 3 features
    bn, ln = nn.BatchNorm1d(3), nn.LayerNorm(3)
    print("batch norm, mean per feature:", bn(x).mean(dim=0).abs().max()
          .item() < 1e-5)
    print("layer norm, mean per sample: ", ln(x).mean(dim=1).abs().max()
          .item() < 1e-5)
    try:
        bn(x[:1])                              # train mode, batch of 1
    except ValueError as e:
        print("batch of 1:", str(e).split(",")[0])
    bn.eval()                                  # uses running averages
    print("eval mode, batch of 1:", tuple(bn(x[:1]).shape))


# %% [3] The CNN from lesson 12, with BatchNorm after each convolution.
def make_cnn(norm: bool = False) -> nn.Module:
    def block(c_in: int, c_out: int) -> list[nn.Module]:
        conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=not norm)
        bn = [nn.BatchNorm2d(c_out)] if norm else []
        return [conv, *bn, nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(
        *block(1, 32),                   # (B, 32, 14, 14)
        *block(32, 64),                  # (B, 64, 7, 7)
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


# %% [4] One training loop with every trick as an option.
def train_epoch(model, loader, device, loss_fn, optimizer,
                scheduler=None, clip=None) -> tuple[float, float]:
    model.train()
    total, max_norm = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(),
                                        clip or math.inf)
        max_norm = max(max_norm, norm.item())
        optimizer.step()
        if scheduler is not None:
            scheduler.step()                 # OneCycleLR: every batch
        total += loss.item() * len(y)
    return total / len(loader.dataset), max_norm


@torch.inference_mode()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
    return correct / len(loader.dataset)


def fit(model, train_loader, val_loader, device, epochs, optimizer,
        scheduler=None, clip=None, log=True) -> float:
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        loss, norm = train_epoch(model, train_loader, device, loss_fn,
                                 optimizer, scheduler, clip)
        acc = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        if log:
            print(f"epoch {epoch:2d}  loss {loss:.3f}  val {acc:.3f}  "
                  f"max grad norm {norm:.3g}  lr {lr:.2g}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_acc


if __name__ == "__main__":
    device = pick_device()
    init_experiment()
    norm_demo()

    to_tensor = transforms.ToTensor()
    train_full = datasets.FashionMNIST("data", train=True, download=True,
                                       transform=to_tensor)
    test = datasets.FashionMNIST("data", train=False, download=True,
                                 transform=to_tensor)
    g = torch.Generator().manual_seed(0)
    train_set, val_set = random_split(train_full, [50_000, 10_000],
                                      generator=g)
    if QUICK:
        train_set = Subset(train_set, range(2000))
        val_set = Subset(val_set, range(500))
        test = Subset(test, range(500))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test, batch_size=256)

    # A 10-layer MLP, plain SGD: default init vs Kaiming vs BatchNorm.
    deep_epochs = 1 if QUICK else 3
    for name, init, norm in (("default", "default", False),
                             ("kaiming", "kaiming", False),
                             ("batchnorm", "default", True)):
        torch.manual_seed(0)
        model = deep_mlp(10, init, norm).to(device)
        sgd = torch.optim.SGD(model.parameters(), lr=0.01)
        print(f"10 layers, {name}:")
        fit(model, train_loader, val_loader, device, deep_epochs, sgd)

    # Gradient clipping: the same net, SGD with momentum and lr 0.1.
    for clip in (None, 1.0):
        torch.manual_seed(0)
        model = deep_mlp(10, "kaiming").to(device)
        sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        print(f"10 layers, lr 0.1, clip {clip}:")
        fit(model, train_loader, val_loader, device, deep_epochs, sgd,
            clip=clip)

    # The CNN: lesson 12 as it was vs BatchNorm + OneCycleLR.
    epochs = 2 if QUICK else 10
    results = {}
    for name, norm in (("CNN", False), ("CNN+tricks", True)):
        torch.manual_seed(0)
        model = make_cnn(norm).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None
        if norm:                         # the schedule takes over the lr
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1e-2,
                total_steps=epochs * len(train_loader))
        print(f"{name}:")
        fit(model, train_loader, val_loader, device, epochs, optimizer,
            scheduler)
        results[name] = evaluate(model, test_loader, device)

    for name, acc in results.items():
        print(f"{name:10s} test {acc:.3f}")
