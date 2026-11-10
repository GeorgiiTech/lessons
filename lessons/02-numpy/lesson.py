"""Lesson 02 · Thinking in Arrays: NumPy and Vectorization.

Run:  uv run python lessons/02-numpy/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import os
import time

import numpy as np
import torch

# Small sizes when tests run this file (LESSONS_QUICK=1).
QUICK = os.environ.get("LESSONS_QUICK") == "1"
N = 100_000 if QUICK else 10_000_000


# %% [1] The same job twice: sum of squares of N numbers.
def loop_sum_squares(values: list[float]) -> float:
    total = 0.0
    for v in values:
        total += v * v
    return total


def numpy_sum_squares(values: np.ndarray) -> float:
    return float(np.sum(values * values))


def best_time(fn, *args, repeats: int = 3) -> float:
    """Run fn several times, return the fastest run (seconds)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def compare_loop_vs_numpy(n: int) -> None:
    rng = np.random.default_rng(0)
    arr = rng.random(n)
    lst = arr.tolist()

    a = loop_sum_squares(lst)
    b = numpy_sum_squares(arr)
    assert abs(a - b) / b < 1e-9          # same answer

    t_loop = best_time(loop_sum_squares, lst)
    t_numpy = best_time(numpy_sum_squares, arr)
    print(f"{n:,} numbers: loop {t_loop * 1000:.0f} ms, "
          f"NumPy {t_numpy * 1000:.1f} ms "
          f"-> {t_loop / t_numpy:.0f}x faster")


# %% [2] An array: one block of memory, one dtype, a shape.
def array_basics() -> np.ndarray:
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    print("shape:", x.shape, "| dtype:", x.dtype,
          "| ndim:", x.ndim, "| size:", x.size)
    return x


# %% [3] Operations work on every element at once.
def elementwise(x: np.ndarray) -> None:
    print("x + 10 =", (x + 10).tolist())
    print("x ** 2 =", (x ** 2).tolist())
    print("x > 3  =", (x > 3).tolist())
    print("x[x > 3] =", x[x > 3].tolist())   # boolean mask


# %% [4] Aggregations and the axis argument.
def aggregations(x: np.ndarray) -> None:
    print("sum all     :", x.sum())
    print("sum axis=0  :", x.sum(axis=0).tolist())  # down columns
    print("sum axis=1  :", x.sum(axis=1).tolist())  # along rows
    print("mean axis=0 :", x.mean(axis=0).tolist())


# %% [5] Broadcasting: a smaller array is stretched to fit.
def broadcasting(x: np.ndarray) -> None:
    row = np.array([10.0, 20.0, 30.0])        # shape (3,)
    print("x + row =", (x + row).tolist())
    col = np.array([[100.0], [200.0]])        # shape (2, 1)
    print("x + col =", (x + col).tolist())


# %% [6] Real use: standardize each column (mean 0, std 1).
def standardize(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0)                  # shape (features,)
    std = data.std(axis=0)
    return (data - mean) / std


def standardize_demo() -> None:
    # 4 students x 3 measurements: hours studied, sleep, score
    data = np.array([[2.0, 7.0, 55.0],
                     [4.0, 6.0, 62.0],
                     [6.0, 8.0, 71.0],
                     [8.0, 5.0, 80.0]])
    z = standardize(data)
    print("means:", z.mean(axis=0).round(6).tolist())
    print("stds :", z.std(axis=0).round(6).tolist())


# %% [7] The classic mistake: shapes (3,) and (3, 1).
def broadcasting_trap() -> None:
    a = np.array([1.0, 2.0, 3.0])             # shape (3,)
    b = np.array([[1.0], [2.0], [3.0]])       # shape (3, 1)
    print("a - b shape:", (a - b).shape)      # (3, 3)!
    print("a - b.ravel() shape:", (a - b.ravel()).shape)


# %% [8] Slices are views: they share memory with the array.
def views_not_copies() -> None:
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    first_row = x[0]                          # a view
    first_row[0] = 100.0
    print("x after editing the view:", x.tolist())
    safe = x[1].copy()                        # a real copy
    safe[0] = -1.0
    print("x after editing the copy:", x.tolist())


# %% [9] Vectorizing an if: maximum and where.
def relu_loop(values: list[float]) -> list[float]:
    return [v if v > 0 else 0.0 for v in values]


def vectorized_if() -> None:
    z = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    print("relu loop :", relu_loop(z.tolist()))
    print("maximum   :", np.maximum(z, 0.0).tolist())
    print("where     :", np.where(z > 0, z, 0.1 * z).tolist())


# %% [10] NumPy and PyTorch talk to each other.
def numpy_and_torch() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    shared = torch.from_numpy(arr)            # same memory
    copied = torch.tensor(arr)                # a copy
    arr[0] = 99.0
    print("from_numpy:", shared.tolist(), shared.dtype)
    print("tensor    :", copied.tolist(), copied.dtype)
    print("as float32:", shared.float().dtype)


if __name__ == "__main__":
    compare_loop_vs_numpy(N)
    x = array_basics()
    elementwise(x)
    aggregations(x)
    broadcasting(x)
    standardize_demo()
    broadcasting_trap()
    views_not_copies()
    vectorized_if()
    numpy_and_torch()
