"""Lesson 02 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/02-numpy/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
No Python loops allowed: use whole-array operations.
"""

import numpy as np


def standardize(data: np.ndarray) -> np.ndarray:
    """Each column → mean 0 and standard deviation 1."""
    # TODO
    raise NotImplementedError


def count_above(data: np.ndarray, threshold: float) -> int:
    """How many values are greater than threshold?
    Hint: a boolean mask, then .sum()."""
    # TODO
    raise NotImplementedError


def pairwise_sq_dist(points: np.ndarray) -> np.ndarray:
    """points has shape (n, 2). Return an (n, n) array where
    [i, j] is the squared distance between point i and j.
    Hint: points[:, None, :] - points[None, :, :]"""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    data = np.array([[2.0, 7.0], [4.0, 6.0], [6.0, 8.0], [8.0, 5.0]])
    z = standardize(data)
    assert z.shape == (4, 2), f"shape should stay (4, 2), got {z.shape}"
    assert np.allclose(z.mean(axis=0), 0), "column means should be 0"
    assert np.allclose(z.std(axis=0), 1), "column stds should be 1"

    assert count_above(data, 5.5) == 5, "5 values are above 5.5"

    pts = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    d = pairwise_sq_dist(pts)
    assert d.shape == (3, 3), f"shape should be (3, 3), got {d.shape}"
    assert d.tolist() == [[0, 25, 100], [25, 0, 25], [100, 25, 0]]
    print("All checks passed")
