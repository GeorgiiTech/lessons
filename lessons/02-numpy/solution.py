"""Lesson 02 · Gym solution."""

import numpy as np


def standardize(data: np.ndarray) -> np.ndarray:
    return (data - data.mean(axis=0)) / data.std(axis=0)


def count_above(data: np.ndarray, threshold: float) -> int:
    return int((data > threshold).sum())


def pairwise_sq_dist(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]   # (n, n, 2)
    return (diff ** 2).sum(axis=-1)


if __name__ == "__main__":
    data = np.array([[2.0, 7.0], [4.0, 6.0], [6.0, 8.0], [8.0, 5.0]])
    z = standardize(data)
    assert np.allclose(z.mean(axis=0), 0) and np.allclose(z.std(axis=0), 1)
    assert count_above(data, 5.5) == 5
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    assert pairwise_sq_dist(pts).tolist() == [[0, 25, 100], [25, 0, 25], [100, 25, 0]]
    print("All checks passed")
