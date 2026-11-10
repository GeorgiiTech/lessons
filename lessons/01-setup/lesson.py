"""Lesson 01 · Why PyTorch and How to Set It Up.

Run:  uv run python lessons/01-setup/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import os
import platform
import random
import time

import torch

# Smaller sizes when the tests run this file (LESSONS_QUICK=1); the video uses the full sizes.
QUICK = os.environ.get("LESSONS_QUICK") == "1"


# %% [1] Which versions are we running?
def show_versions() -> None:
    print(f"Python  {platform.python_version()}")
    print(f"PyTorch {torch.__version__}")


# %% [2] Pick the fastest device: NVIDIA GPU (cuda), Apple GPU (mps) or the CPU.
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """GPUs run asynchronously: wait until the work is really done before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# %% [3] Your first tensor: values + shape + dtype + device.
def first_tensor(device: torch.device) -> torch.Tensor:
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    print(x)
    print("shape:", x.shape, "| dtype:", x.dtype, "| device:", x.device)
    x = x.to(device)                      # move it to the GPU (if we have one)
    print("now on:", x.device)
    return x


# %% [4] Math on whole tensors at once: no loops.
def tensor_math(x: torch.Tensor) -> None:
    print("x * 2   =", (x * 2).tolist())
    print("x.sum() =", x.sum().item())
    print("x @ x.T =", (x @ x.T).tolist())   # (2x3) @ (3x2) -> (2x2)


# %% [5] Why not plain Python? Multiply two n x n matrices both ways.
def python_matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n, m, p = len(a), len(b), len(b[0])
    return [[sum(a[i][k] * b[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def compare_python_vs_torch(n: int) -> tuple[float, float]:
    random.seed(0)
    a = [[random.random() for _ in range(n)] for _ in range(n)]
    b = [[random.random() for _ in range(n)] for _ in range(n)]

    start = time.perf_counter()
    c_python = python_matmul(a, b)
    t_python = time.perf_counter() - start

    ta, tb = torch.tensor(a), torch.tensor(b)
    start = time.perf_counter()
    c_torch = ta @ tb
    t_torch = time.perf_counter() - start

    assert torch.allclose(torch.tensor(c_python), c_torch, atol=1e-3)   # same answer
    print(f"{n}x{n} matmul: Python {t_python * 1000:.1f} ms, PyTorch {t_torch * 1000:.3f} ms "
          f"-> {t_python / t_torch:,.0f}x faster")
    return t_python, t_torch


# %% [6] CPU vs GPU on a big matrix.
def compare_cpu_vs_device(n: int, device: torch.device, repeats: int = 5) -> dict[str, float]:
    torch.manual_seed(0)
    times = {}
    for dev in dict.fromkeys([torch.device("cpu"), device]):    # skips the duplicate when device is cpu
        a = torch.randn(n, n, device=dev)
        b = torch.randn(n, n, device=dev)
        a @ b                                                    # warm-up (first call sets things up)
        sync(dev)
        start = time.perf_counter()
        for _ in range(repeats):
            a @ b
        sync(dev)
        times[dev.type] = (time.perf_counter() - start) / repeats
        print(f"{n}x{n} matmul on {dev.type}: {times[dev.type] * 1000:.1f} ms")
    return times


# %% [7] The other superpower: PyTorch computes gradients for you (lesson 05).
def autograd_teaser() -> None:
    w = torch.tensor(3.0, requires_grad=True)
    y = w ** 2                  # y = w²
    y.backward()                # dy/dw = 2w
    print("dy/dw at w=3:", w.grad.item())


if __name__ == "__main__":
    show_versions()
    device = get_device()
    print("device:", device)
    x = first_tensor(device)
    tensor_math(x)
    compare_python_vs_torch(60 if QUICK else 200)
    compare_cpu_vs_device(256 if QUICK else 4096, device)
    autograd_teaser()
