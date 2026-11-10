"""Lesson 03 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/03-tensors/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch


def identity_like(n: int) -> torch.Tensor:
    """An n x n float32 tensor: ones on the diagonal, zeros elsewhere.
    Hint: torch.eye."""
    # TODO
    raise NotImplementedError


def average(scores: list[int]) -> float:
    """Mean of integer scores as a Python float, using a tensor."""
    # TODO
    raise NotImplementedError


def seeded_noise(seed: int) -> torch.Tensor:
    """Shape (2, 2) standard-normal noise that is the same for the
    same seed. Hint: torch.manual_seed, torch.randn."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    eye = identity_like(3)
    assert eye.shape == (3, 3) and eye.dtype == torch.float32
    assert eye.sum().item() == 3 and eye[0, 0] == 1 and eye[0, 1] == 0

    avg = average([1, 2, 4])
    assert isinstance(avg, float), "return a Python float (.item())"
    assert abs(avg - 7 / 3) < 1e-6

    assert torch.equal(seeded_noise(42), seeded_noise(42))
    assert not torch.equal(seeded_noise(42), seeded_noise(7))
    print("All checks passed")
