"""Lesson 25 · Solution to gym.py.

Run:  uv run python lessons/25-policy-gradients/solution.py
"""

import torch


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    Hint: walk the rewards backwards."""
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return torch.tensor(out[::-1])


def reinforce_loss(log_probs: torch.Tensor, returns: torch.Tensor):
    """Minimizing this loss raises the log-probability of actions
    with a positive (normalized) return."""
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return -(log_probs * returns).sum()


if __name__ == "__main__":
    g = discounted_returns([1.0, 1.0, 1.0], gamma=0.5)
    assert torch.allclose(g, torch.tensor([1.75, 1.5, 1.0])), g
    g = discounted_returns([0.0, 0.0, 10.0], gamma=0.9)
    assert torch.allclose(g, torch.tensor([8.1, 9.0, 10.0])), g

    logits = torch.zeros(3, requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=0)
    loss = reinforce_loss(log_probs, torch.tensor([3.0, 1.0, 2.0]))
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1], \
        "the action with the best return must become more likely"
    print("All checks passed")
