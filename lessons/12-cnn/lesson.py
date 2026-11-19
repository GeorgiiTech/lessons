"""Lesson 12 · Convolutional neural networks: a CNN on FashionMNIST.

Run:  uv run python lessons/12-cnn/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
The first run downloads FashionMNIST (about 30 MB) into ./data.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

QUICK = os.environ.get("LESSONS_QUICK") == "1"
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# %% [1] A convolution by hand: slide a 3 × 3 kernel over a tiny image.
def tiny_convolution() -> None:
    image = torch.tensor([[0., 0., 1., 1., 1.],
                          [0., 0., 1., 1., 1.],
                          [0., 0., 1., 1., 1.],
                          [0., 0., 1., 1., 1.],
                          [0., 0., 1., 1., 1.]])
    kernel = torch.tensor([[-1., 0., 1.],
                           [-1., 0., 1.],
                           [-1., 0., 1.]])       # finds vertical edges
    out = F.conv2d(image[None, None], kernel[None, None])
    print("image", tuple(image.shape), "→ output", tuple(out.shape[2:]))
    print(out[0, 0])
    patch = image[0:3, 1:4]                      # top row, middle column
    print("one output by hand:", (patch * kernel).sum().item())


# %% [2] nn.Conv2d and nn.MaxPool2d: what they do to the shape.
def shapes() -> None:
    x = torch.randn(64, 1, 28, 28)               # a batch of 64 pictures
    conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    pool = nn.MaxPool2d(2)
    print("input  ", tuple(x.shape))
    print("conv   ", tuple(conv(x).shape))
    print("pool   ", tuple(pool(conv(x)).shape))
    print("weights", tuple(conv.weight.shape), "+ bias", conv.bias.numel())


# %% [3] The CNN: two conv blocks find patterns, a small head classifies.
def make_cnn() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),                 # (B, 32, 14, 14)
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),                 # (B, 64, 7, 7)
        nn.Flatten(),                    # (B, 3136)
        nn.Linear(64 * 7 * 7, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def make_mlp() -> nn.Module:             # lesson 10's model, to compare
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 10))


# %% [4] Training and evaluation: the same loop as lessons 08-11.
def train_epoch(model, loader, device, loss_fn, optimizer) -> None:
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    matrix = torch.zeros(10, 10, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
        n += len(y)
        for t, p in zip(y, pred):
            matrix[t, p] += 1
    return correct / n, matrix


def fit(model, train_loader, val_loader, device, epochs: int):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, device, loss_fn, optimizer)
        val_acc, _ = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}  val acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    device = pick_device()
    tiny_convolution()
    shapes()

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

    epochs = 2 if QUICK else 10
    results = {}
    for name, make in (("MLP", make_mlp), ("CNN", make_cnn)):
        torch.manual_seed(0)
        model = make().to(device)
        print(f"{name}: {n_params(model):,} parameters")
        fit(model, train_loader, val_loader, device, epochs)
        test_acc, matrix = evaluate(model, test_loader, device)
        shirt = (matrix[6, 6] / matrix[6].sum()).item()
        results[name] = (test_acc, shirt)
        print(f"{name}: test acc {test_acc:.3f}  Shirt {shirt:.3f}")
        if name == "CNN":
            torch.save(model.state_dict(), "data/lesson12_cnn.pt")

    for name, (acc, shirt) in results.items():
        print(f"{name:4s} test {acc:.3f}  Shirt {shirt:.3f}")
