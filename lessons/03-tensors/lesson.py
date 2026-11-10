"""Lesson 03 · Tensors 101.

Run:  uv run python lessons/03-tensors/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import numpy as np
import torch


# %% [1] Many ways to make a tensor.
def creation() -> None:
    print("tensor :", torch.tensor([[1.0, 2.0], [3.0, 4.0]]).tolist())
    print("zeros  :", torch.zeros(2, 3).tolist())
    print("ones   :", torch.ones(3).tolist())
    print("full   :", torch.full((2, 2), 7.0).tolist())
    print("arange :", torch.arange(0, 10, 2).tolist())
    print("linspace:", torch.linspace(0, 1, 5).tolist())


# %% [2] Random tensors: always with a seed.
def random_tensors() -> None:
    torch.manual_seed(0)
    a = torch.rand(2, 3)       # uniform in [0, 1)
    torch.manual_seed(0)
    b = torch.rand(2, 3)
    print("same seed, same numbers:", torch.equal(a, b))
    g = torch.randn(10_000)    # normal: mean 0, std 1
    print(f"randn mean {g.mean():.3f}, std {g.std():.3f}")


# %% [3] Shape, size, dimensions.
def attributes() -> None:
    x = torch.zeros(2, 3, 4)
    print("shape:", tuple(x.shape), "| ndim:", x.ndim,
          "| numel:", x.numel())
    print("x[0].shape:", tuple(x[0].shape))


# %% [4] dtype: what Python numbers become.
def dtypes() -> None:
    print("ints   ->", torch.tensor([1, 2, 3]).dtype)
    print("floats ->", torch.tensor([1.0, 2.0]).dtype)
    print("bools  ->", torch.tensor([True, False]).dtype)
    mixed = torch.tensor([1, 2]) + torch.tensor([0.5, 0.5])
    print("int + float ->", mixed.dtype, mixed.tolist())


# %% [5] The classic mistake: an integer tensor.
def int_tensor_trap() -> None:
    scores = torch.tensor([1, 2, 4])
    try:
        scores.mean()
    except RuntimeError as err:
        print("error:", str(err).split(".")[0])
    print("fixed :", scores.float().mean().item())
    print("7 / 2 =", (torch.tensor(7) / 2).item(),
          "| 7 // 2 =", (torch.tensor(7) // 2).item())


# %% [6] Getting numbers out: item, tolist, numpy.
def conversions() -> None:
    t = torch.tensor([1.5, 2.5])
    print("item  :", t.sum().item(), type(t.sum().item()).__name__)
    print("tolist:", t.tolist())
    print("numpy :", t.numpy(), t.numpy().dtype)
    print("back  :", torch.from_numpy(np.ones(2)).dtype)


# %% [7] In-place operations end with an underscore.
def in_place() -> None:
    x = torch.ones(3)
    y = x.add(1)                # new tensor
    print("after add :", x.tolist(), y.tolist())
    x.add_(1)                   # changes x itself
    print("after add_:", x.tolist())


if __name__ == "__main__":
    creation()
    random_tensors()
    attributes()
    dtypes()
    int_tensor_trap()
    conversions()
    in_place()
