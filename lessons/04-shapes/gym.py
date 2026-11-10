"""Lesson 04 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/04-shapes/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import torch


def flatten_images(batch: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*H*W), without hard-coding the sizes.
    Hint: reshape with -1."""
    # TODO
    raise NotImplementedError


def to_channels_last(img: torch.Tensor) -> torch.Tensor:
    """(C, H, W) -> (H, W, C). Hint: permute."""
    # TODO
    raise NotImplementedError


def predict_one(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x is ONE sample of shape (features,), w is (features, classes).
    Return shape (1, classes): add a batch dim, then matmul."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    b = torch.rand(8, 3, 4, 4)
    f = flatten_images(b)
    assert f.shape == (8, 48), f"expected (8, 48), got {tuple(f.shape)}"
    assert torch.equal(f[0], b[0].flatten())

    img = torch.rand(3, 5, 7)
    assert to_channels_last(img).shape == (5, 7, 3)
    assert to_channels_last(img)[2, 4, 1] == img[1, 2, 4]

    out = predict_one(torch.rand(6), torch.rand(6, 10))
    assert out.shape == (1, 10), f"expected (1, 10), got {tuple(out.shape)}"
    print("All checks passed")
