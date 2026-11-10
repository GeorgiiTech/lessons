"""Lesson 04 · Gym solution."""

import torch


def flatten_images(batch: torch.Tensor) -> torch.Tensor:
    return batch.reshape(batch.shape[0], -1)


def to_channels_last(img: torch.Tensor) -> torch.Tensor:
    return img.permute(1, 2, 0)


def predict_one(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) @ w


if __name__ == "__main__":
    assert flatten_images(torch.rand(8, 3, 4, 4)).shape == (8, 48)
    assert to_channels_last(torch.rand(3, 5, 7)).shape == (5, 7, 3)
    assert predict_one(torch.rand(6), torch.rand(6, 10)).shape == (1, 10)
    print("All checks passed")
