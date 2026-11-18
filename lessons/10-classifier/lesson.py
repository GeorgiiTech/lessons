"""Lesson 10 · First real classifier: an MLP on FashionMNIST.

Run:  uv run python lessons/10-classifier/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
The first run downloads FashionMNIST (about 30 MB) into ./data.
"""

import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# Small sizes when tests run this file (LESSONS_QUICK=1).
QUICK = os.environ.get("LESSONS_QUICK") == "1"
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# %% [1] FashionMNIST: 70,000 small grey pictures of clothes, 10 classes.
def load_data(root: str = "data"):
    to_tensor = transforms.ToTensor()          # 0..255 pixels → 0..1 floats
    train = datasets.FashionMNIST(root, train=True, download=True,
                                  transform=to_tensor)
    test = datasets.FashionMNIST(root, train=False, download=True,
                                 transform=to_tensor)
    return train, test


# %% [2] The model: flatten 28 × 28 pixels, one hidden layer, 10 scores.
def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),                  # (B, 1, 28, 28) → (B, 784)
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),            # one score (logit) per class
    )


# %% [3] Accuracy: the class with the highest score vs the true label.
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.inference_mode()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


# %% [4] One epoch of training: the loop from lesson 08.
def train_epoch(model, loader, device, loss_fn, optimizer) -> None:
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# %% [5] The confusion matrix: row = true class, column = predicted class.
@torch.inference_mode()
def confusion_matrix(model, loader, device, k: int = 10) -> torch.Tensor:
    model.eval()
    matrix = torch.zeros(k, k, dtype=torch.long)
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()
        for t, p in zip(y, pred):
            matrix[t, p] += 1
    return matrix


if __name__ == "__main__":
    torch.manual_seed(0)
    device = pick_device()
    train_full, test = load_data()
    print("train:", len(train_full), "test:", len(test))
    image, label = train_full[0]
    print("one image:", tuple(image.shape), image.dtype,
          "min", image.min().item(), "max", round(image.max().item(), 2),
          "label", label, CLASSES[label])

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

    model = make_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters:", n_params)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x, y = next(iter(train_loader))
    with torch.inference_mode():
        logits = model(x.to(device))
    print("logits shape:", tuple(logits.shape))
    print("untrained loss:", round(loss_fn(logits, y.to(device)).item(), 3),
          "accuracy:", round(accuracy(logits, y.to(device)), 3))

    best_acc, best_state = 0.0, None
    for epoch in range(1, 3 if QUICK else 11):
        train_epoch(model, train_loader, device, loss_fn, optimizer)
        train_loss, train_acc = evaluate(model, train_loader, device, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        print(f"epoch {epoch:2d}  train acc {train_acc:.3f}  "
              f"val loss {val_loss:.3f}  val acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
    print(f"test loss {test_loss:.3f}  test acc {test_acc:.3f}")

    m = confusion_matrix(model, test_loader, device)
    print("confusion matrix (rows = true):")
    print(m)
    per_class = (m.diag() / m.sum(dim=1)).tolist()
    for name, acc in sorted(zip(CLASSES, per_class), key=lambda t: t[1]):
        print(f"  {name:9s} {acc:.3f}")
    off = m.clone().fill_diagonal_(0)
    t, p = divmod(off.argmax().item(), 10)
    print("most common mistake:", CLASSES[t], "→", CLASSES[p],
          off[t, p].item())

    image, label = test[0]
    with torch.inference_mode():
        probs = model(image.unsqueeze(0).to(device)).softmax(dim=1)[0].cpu()
    top = probs.topk(3)
    print("test[0] true:", CLASSES[label], "top-3:",
          [(CLASSES[i], round(v, 3)) for v, i in zip(top.values.tolist(),
                                                    top.indices.tolist())])
