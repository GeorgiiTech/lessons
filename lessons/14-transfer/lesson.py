"""Lesson 14 · Transfer learning: borrow a network trained on ImageNet.

Run:  uv run python lessons/14-transfer/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
The first run downloads CIFAR-10 (about 170 MB) into ./data and the
ResNet-18 weights (about 45 MB) into the torch cache.
"""

import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

QUICK = os.environ.get("LESSONS_QUICK") == "1"
N_TRAIN = 200 if QUICK else 5000
N_VAL = 100 if QUICK else 1000


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# %% [1] A pretrained model: ResNet-18, trained on ImageNet (1000 classes).
def look_at_resnet() -> None:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    print("parameters:", f"{n_params(model):,}")
    print("ImageNet top-1:", weights.meta["_metrics"]["ImageNet-1K"]["acc@1"])
    print("last layer:", model.fc)
    print(weights.transforms())


def n_params(model: nn.Module, trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable)


# %% [2] Three ways to use it, plus a small CNN trained from scratch.
def small_cnn() -> nn.Module:                # lesson 13's tricks, RGB
    def block(c_in: int, c_out: int) -> list[nn.Module]:
        return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128),
                         nn.Flatten(), nn.Linear(128 * 4 * 4, 128),
                         nn.ReLU(), nn.Linear(128, 10))


def resnet(pretrained: bool, freeze: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False          # the backbone stays as it is
    model.fc = nn.Linear(model.fc.in_features, 10)   # new head, trainable
    return model


# %% [3] Training: lesson 13's loop with OneCycleLR, only trainable params.
def fit(model, train_loader, val_loader, device, epochs, lr) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            loss = loss_fn(model(x.to(device)), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}  val {acc:.3f}  {time.time() - start:.0f} s")
    return acc


@torch.inference_mode()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(loader.dataset)


def cifar(transform):
    train_full = datasets.CIFAR10("data", train=True, download=True,
                                  transform=transform)
    test = datasets.CIFAR10("data", train=False, download=True,
                            transform=transform)
    order = torch.randperm(50_000, generator=torch.Generator().manual_seed(0))
    train = Subset(train_full, order[:N_TRAIN].tolist())   # only 5,000
    val = Subset(train_full, order[-N_VAL:].tolist())
    if QUICK:
        test = Subset(test, range(100))
    workers = 0 if QUICK else 4
    return (DataLoader(train, batch_size=64, shuffle=True,
                       num_workers=workers),
            DataLoader(val, batch_size=256, num_workers=workers),
            DataLoader(test, batch_size=256, num_workers=workers))


if __name__ == "__main__":
    device = pick_device()
    look_at_resnet()

    # The same ImageNet preprocessing the model was trained with: 224 × 224.
    imagenet = models.ResNet18_Weights.DEFAULT.transforms()
    small = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))])
    runs = (("from scratch", small_cnn, small, 1e-2),
            ("ResNet, random", lambda: resnet(False, False), imagenet, 1e-3),
            ("ResNet, frozen", lambda: resnet(True, True), imagenet, 1e-2),
            ("ResNet, fine-tuned", lambda: resnet(True, False), imagenet,
             1e-3))
    results = {}
    for name, make, transform, lr in runs:
        torch.manual_seed(0)
        model = make().to(device)
        train_loader, val_loader, test_loader = cifar(transform)
        print(f"{name}: {n_params(model, trainable=True):,} trainable")
        epochs = 1 if QUICK else (15 if name == "from scratch" else 5)
        fit(model, train_loader, val_loader, device, epochs, lr)
        results[name] = evaluate(model, test_loader, device)

    for name, acc in results.items():
        print(f"{name:20s} test {acc:.3f}")
