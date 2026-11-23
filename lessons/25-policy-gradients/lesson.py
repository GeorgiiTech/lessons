"""Lesson 25 · Policy gradients: REINFORCE by hand, then PPO.

Run:  uv sync --extra rl
      uv run python lessons/25-policy-gradients/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
CPU, a few minutes.
"""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

QUICK = os.environ.get("LESSONS_QUICK") == "1"
torch.set_num_threads(4)
GAMMA = 0.99


# %% [1] A policy network outputs probabilities, not values.
def policy_network() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                         nn.Linear(64, 2), nn.Softmax(dim=-1))


def act(policy, obs):
    probs = policy(torch.as_tensor(obs))
    dist = Categorical(probs)
    action = dist.sample()                     # sample, don't argmax
    return int(action), dist.log_prob(action)


# %% [2] Returns-to-go: what each action was followed by.
def returns_to_go(rewards: list[float], gamma: float = GAMMA):
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return torch.tensor(out[::-1])


# %% [3] REINFORCE: make actions that led to big returns more likely.
def reinforce(episodes: int, normalize: bool = True, seed: int = 0,
              log: bool = True) -> list[float]:
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    policy = policy_network()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    lengths = []
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, rewards, done = [], [], False
        while not done:
            action, log_prob = act(policy, obs)
            obs, r, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(r)
            done = terminated or truncated
        g = returns_to_go(rewards)
        if normalize:                          # a simple baseline
            g = (g - g.mean()) / (g.std() + 1e-8)
        loss = -(torch.stack(log_probs) * g).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        lengths.append(len(rewards))
        if log and ep % (episodes // 6) == 0:
            print(f"episode {ep:4d}  mean of last 50:"
                  f" {np.mean(lengths[-50:]):6.1f}")
    return lengths


# %% [4] PPO in three lines with Stable-Baselines3.
def ppo(steps: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    model = PPO("MlpPolicy", "CartPole-v1", seed=0, verbose=0)
    start = time.time()
    model.learn(total_timesteps=steps)
    mean, std = evaluate_policy(model, model.get_env(),
                                n_eval_episodes=20)
    return mean, std, time.time() - start


if __name__ == "__main__":
    p = policy_network()
    print("probabilities:", p(torch.zeros(4)).tolist())
    print("returns-to-go of 4 rewards of 1:",
          [round(v, 3) for v in returns_to_go([1, 1, 1, 1]).tolist()])
    episodes = 60 if QUICK else 600
    start = time.time()
    lengths = reinforce(episodes)
    print(f"REINFORCE: {time.time() - start:.0f} s, {sum(lengths):,}"
          f" steps, last 50 episodes {np.mean(lengths[-50:]):.1f}")
    raw = reinforce(episodes, normalize=False, log=False)
    print(f"without normalizing returns: last 50 {np.mean(raw[-50:]):.1f}")
    mean, std, sec = ppo(5_000 if QUICK else 100_000)
    print(f"PPO (Stable-Baselines3): mean {mean:.1f} +- {std:.1f}"
          f" after {5_000 if QUICK else 100_000:,} steps, {sec:.0f} s")
