"""Lesson 23 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/23-gymnasium/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import gymnasium as gym
import numpy as np


def episode_return(env, policy, seed: int) -> float:
    """Reset with the seed, step with policy(obs) and add up the rewards
    until the episode ends: terminated OR truncated."""
    # TODO: the reset / step loop
    raise NotImplementedError


def my_policy(obs) -> int:
    """CartPole: obs = [cart position, cart velocity, pole angle,
    pole angular velocity]. Return 0 (push left) or 1 (push right).
    Goal: an average of at least 195 steps."""
    # TODO: look at the pole, not at the cart
    raise NotImplementedError


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    assert episode_return(env, lambda obs: 0, seed=0) < 20, \
        "always pushing left should fall quickly"
    assert episode_return(env, lambda obs: 0, seed=0) == \
        episode_return(env, lambda obs: 0, seed=0), "use the seed"
    scores = [episode_return(env, my_policy, seed=s) for s in range(20)]
    print(f"my_policy: mean {np.mean(scores):.1f}")
    assert np.mean(scores) >= 195, "keep the pole up for longer"
    assert max(scores) <= 500, "CartPole-v1 truncates at 500 steps"
    print("All checks passed")
