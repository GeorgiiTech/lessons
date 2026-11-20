"""Loading the data: the only file that knows where pictures come from."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def load(n: int, train: bool = True, root: str = "data") -> TensorDataset:
    ds = datasets.FashionMNIST(root, train=train, download=True,
                               transform=transforms.ToTensor())
    x = torch.stack([ds[i][0] for i in range(n)])
    y = torch.tensor([ds[i][1] for i in range(n)])
    return TensorDataset(x, y)


def loaders(n_train: int, batch_size: int,
            root: str = "data") -> tuple[DataLoader, DataLoader]:
    train = load(n_train, root=root)
    val = load(2000, train=False, root=root)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=500))
