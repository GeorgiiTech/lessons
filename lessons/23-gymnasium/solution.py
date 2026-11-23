"""Lesson 23 · Solution to gym.py.

Run:  uv run python lessons/23-gymnasium/solution.py
"""

import gymnasium as gym
import numpy as np


def episode_return(env, policy, seed: int) -> float:
    """Reset with the seed, step with policy(obs) and add up the rewards
    until the episode ends: terminated OR truncated."""
    obs, info = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        if terminated or truncated:
            return total


def my_policy(obs) -> int:
    """CartPole: obs = [cart position, cart velocity, pole angle,
    pole angular velocity]. Return 0 (push left) or 1 (push right).
    Goal: an average of at least 195 steps."""
    angle, angular_velocity = obs[2], obs[3]
    return int(angle + 0.5 * angular_velocity > 0)


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
