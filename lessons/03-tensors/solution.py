"""Lesson 03 · Gym solution."""

import torch


def identity_like(n: int) -> torch.Tensor:
    return torch.eye(n)


def average(scores: list[int]) -> float:
    return torch.tensor(scores, dtype=torch.float32).mean().item()


def seeded_noise(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(2, 2)


if __name__ == "__main__":
    assert identity_like(3).sum().item() == 3
    assert abs(average([1, 2, 4]) - 7 / 3) < 1e-6
    assert torch.equal(seeded_noise(42), seeded_noise(42))
    print("All checks passed")
