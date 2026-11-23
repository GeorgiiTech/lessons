"""Lesson 24 · Solution to gym.py.

Run:  uv run python lessons/24-dqn/solution.py
"""

import numpy as np
import torch


def bellman_target(reward, next_q, terminated, gamma: float = 0.99):
    """reward (B,), next_q (B, actions) from the target network,
    terminated (B,) of 0/1. A terminated transition has no future;
    a truncated one still does, so only `terminated` stops the sum."""
    return reward + gamma * next_q.max(dim=1).values * (1 - terminated)


def epsilon_greedy(q_values: torch.Tensor, eps: float, rng) -> int:
    """With probability eps a random action, otherwise the argmax."""
    if rng.random() < eps:
        return int(rng.integers(len(q_values)))
    return int(q_values.argmax())


if __name__ == "__main__":
    r = torch.tensor([1.0, 1.0, 1.0])
    nq = torch.tensor([[2.0, 5.0], [3.0, 1.0], [9.0, 9.0]])
    term = torch.tensor([0.0, 0.0, 1.0])
    t = bellman_target(r, nq, term, gamma=0.5)
    assert torch.allclose(t, torch.tensor([3.5, 2.5, 1.0])), t

    rng = np.random.default_rng(0)
    q = torch.tensor([0.1, 0.9])
    assert all(epsilon_greedy(q, 0.0, rng) == 1 for _ in range(50))
    picks = [epsilon_greedy(q, 1.0, rng) for _ in range(1000)]
    assert 400 < sum(picks) < 600, "eps = 1 should be a fair coin"
    print("All checks passed")
