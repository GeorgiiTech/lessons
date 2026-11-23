"""Lesson 23 · Gymnasium: the environment API, and CartPole.

Run:  uv run python lessons/23-gymnasium/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
No download needed. CPU, about a minute.
"""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn

QUICK = os.environ.get("LESSONS_QUICK") == "1"


# %% [1] An environment: spaces, reset, step.
def tour() -> None:
    env = gym.make("CartPole-v1")
    print("observation space:", env.observation_space.shape,
          env.observation_space.dtype, " action space:", env.action_space)
    obs, info = env.reset(seed=0)
    print("first observation:", np.round(obs, 3))
    obs, reward, terminated, truncated, info = env.step(1)   # push right
    print("after one step:", np.round(obs, 3), " reward", reward)
    print("terminated", terminated, " truncated", truncated)
    env.close()


# %% [2] One episode: loop until terminated OR truncated.
def run_episode(env, policy, seed: int, why: bool = False):
    obs, info = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        if terminated or truncated:
            return (total, terminated, truncated) if why else total


def evaluate(policy, episodes: int = 100) -> tuple[float, float]:
    env = gym.make("CartPole-v1")
    scores = [run_episode(env, policy, seed=s) for s in range(episodes)]
    env.close()
    return float(np.mean(scores)), float(np.max(scores))


# %% [3] Three policies: random, a rule, a tiny linear network.
rng = np.random.default_rng(0)


def random_policy(obs) -> int:
    return int(rng.integers(2))


def lean_policy(obs) -> int:
    """Push the cart toward the side the pole is falling to."""
    angle, angular_velocity = obs[2], obs[3]
    return int(angle + 0.5 * angular_velocity > 0)


class LinearPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)          # 4 numbers in, 2 actions out

    @torch.no_grad()
    def __call__(self, obs) -> int:
        logits = self.net(torch.as_tensor(obs, dtype=torch.float32))
        return int(logits.argmax())


# %% [4] The simplest learning: try random weights, keep the best.
def random_search(tries: int) -> tuple[float, int]:
    torch.manual_seed(0)
    best, best_at = 0.0, 0
    for i in range(1, tries + 1):
        policy = LinearPolicy()             # fresh random weights
        mean, _ = evaluate(policy, episodes=10)
        if mean > best:
            best, best_at = mean, i
            print(f"try {i:3d}: mean {mean:6.1f}  (new best)")
    return best, best_at


if __name__ == "__main__":
    tour()
    env = gym.make("CartPole-v1")
    start = time.perf_counter()
    steps = 0
    obs, info = env.reset(seed=0)
    for _ in range(10_000):
        obs, r, term, trunc, info = env.step(random_policy(obs))
        steps += 1
        if term or trunc:
            obs, info = env.reset()
    us = (time.perf_counter() - start) / steps * 1e6
    print(f"{steps:,} steps, {us:.1f} microseconds per step")

    episodes = 20 if QUICK else 100
    for name, policy in (("random", random_policy),
                         ("lean rule", lean_policy)):
        mean, best = evaluate(policy, episodes)
        print(f"{name:10s} mean {mean:6.1f}  best {best:5.0f}")
    for name, policy in (("random", random_policy),
                         ("lean rule", lean_policy)):
        total, term, trunc = run_episode(env, policy, seed=0, why=True)
        print(f"{name:10s} ended after {total:3.0f} steps:"
              f" terminated={term}  truncated={trunc}")
    best, at = random_search(tries=10 if QUICK else 100)
    print(f"random search: best mean {best:.1f} found at try {at}")
