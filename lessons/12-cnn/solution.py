"""Lesson 12 · Solution to gym.py."""

import torch
import torch.nn.functional as F
from torch import nn


def my_conv2d(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """One channel, stride 1, no padding: slide `kernel` (k × k) over
    `image` (H × W) and return the (H - k + 1) × (W - k + 1) output."""
    k = kernel.shape[0]
    h, w = image.shape[0] - k + 1, image.shape[1] - k + 1
    out = torch.zeros(h, w)
    for i in range(h):
        for j in range(w):
            out[i, j] = (image[i:i + k, j:j + k] * kernel).sum()
    return out


def out_size(size: int, kernel: int,
             padding: int = 0, stride: int = 1) -> int:
    """Output height (or width) of a convolution or pooling layer."""
    return (size + 2 * padding - kernel) // stride + 1


if __name__ == "__main__":
    torch.manual_seed(0)
    image, kernel = torch.randn(6, 7), torch.randn(3, 3)
    expected = F.conv2d(image[None, None], kernel[None, None])[0, 0]
    assert torch.allclose(my_conv2d(image, kernel), expected, atol=1e-5), \
        "same result as F.conv2d"

    assert out_size(28, 3, padding=1) == 28, "padding 1 keeps 28 × 28"
    assert out_size(28, 2, stride=2) == 14, "MaxPool2d(2) halves it"
    assert out_size(32, 5) == 28, "a 5 × 5 kernel loses 4 pixels"
    x = torch.randn(1, 3, 32, 32)
    y = nn.Conv2d(3, 8, 5, stride=2, padding=2)(x)
    assert y.shape[-1] == out_size(32, 5, padding=2, stride=2)
    print("All checks passed")
