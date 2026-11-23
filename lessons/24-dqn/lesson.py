"""Lesson 24 · Deep Q-learning (DQN) on CartPole.

Run:  uv run python lessons/24-dqn/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
CPU on purpose: the network is tiny, so the GPU would only add overhead.
About 3 minutes for the three runs.
"""

import copy
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

QUICK = os.environ.get("LESSONS_QUICK") == "1"
torch.set_num_threads(4)
GAMMA = 0.99


# %% [1] Q(s, a): one network, one output per action.
def q_network() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, 2))


# %% [2] A replay buffer: remember transitions, sample them at random.
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.obs = torch.zeros(capacity, 4)
        self.next_obs = torch.zeros(capacity, 4)
        self.action = torch.zeros(capacity, dtype=torch.long)
        self.reward = torch.zeros(capacity)
        self.terminated = torch.zeros(capacity)
        self.capacity, self.size, self.i = capacity, 0, 0

    def push(self, obs, action, reward, next_obs, terminated) -> None:
        self.obs[self.i] = torch.as_tensor(obs)
        self.next_obs[self.i] = torch.as_tensor(next_obs)
        self.action[self.i], self.reward[self.i] = action, reward
        self.terminated[self.i] = float(terminated)
        self.i = (self.i + 1) % self.capacity        # overwrite the oldest
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = torch.randint(self.size, (batch,))
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.next_obs[idx], self.terminated[idx])


# %% [3] The DQN update: Bellman target from a frozen copy of the network.
def dqn_loss(q, target_q, batch) -> torch.Tensor:
    obs, action, reward, next_obs, terminated = batch
    q_taken = q(obs).gather(1, action[:, None]).squeeze(1)
    with torch.no_grad():
        best_next = target_q(next_obs).max(dim=1).values
        target = reward + GAMMA * best_next * (1 - terminated)
    return F.smooth_l1_loss(q_taken, target)


def epsilon(step: int, decay: int = 10_000) -> float:
    return max(0.05, 1.0 - step / decay)      # 1.0 -> 0.05, then flat


@torch.no_grad()
def evaluate(q, episodes: int = 10) -> float:
    env = gym.make("CartPole-v1")
    scores = []
    for seed in range(1000, 1000 + episodes):
        obs, _ = env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            action = int(q(torch.as_tensor(obs)).argmax())
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        scores.append(total)
    return float(np.mean(scores))


# %% [4] The training loop: act, remember, learn, sync the target.
def train(steps: int, use_target: bool = True, buffer_size: int = 50_000,
          seed: int = 0, log: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    q = q_network()
    target_q = q_network()
    target_q.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buffer = ReplayBuffer(buffer_size)
    obs, _ = env.reset(seed=seed)
    curve, best, best_state = [], -1.0, None
    for step in range(1, steps + 1):
        if rng.random() < epsilon(step):
            action = int(rng.integers(2))                   # explore
        else:
            with torch.no_grad():
                action = int(q(torch.as_tensor(obs)).argmax())  # exploit
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.push(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
        if buffer.size >= min(1_000, buffer_size):
            batch = buffer.sample(min(128, buffer.size))
            loss = dqn_loss(q, target_q if use_target else q, batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
        if step % 1_000 == 0:
            target_q.load_state_dict(q.state_dict())        # sync
        if step % (steps // 8) == 0:
            score = evaluate(q)
            curve.append(score)
            if score > best:                        # keep the best weights
                best, best_state = score, copy.deepcopy(q.state_dict())
            if log:
                print(f"step {step:6d}  epsilon {epsilon(step):.2f}"
                      f"  greedy score {score:6.1f}")
    q.load_state_dict(best_state)
    return curve, q


if __name__ == "__main__":
    q = q_network()
    print("Q(s) for one state:", tuple(q(torch.zeros(1, 4)).shape),
          " parameters:", sum(p.numel() for p in q.parameters()))
    steps = 2_000 if QUICK else 60_000
    start = time.time()
    curve, q = train(steps)
    print(f"DQN: {time.time() - start:.0f} s, best greedy score"
          f" {max(curve):.1f}, last {curve[-1]:.1f}")
    print(f"best weights, 100 new episodes: {evaluate(q, 100):.1f}")
    with torch.no_grad():
        env = gym.make("CartPole-v1")
        s0, _ = env.reset(seed=1000)
        print("Q(start) =", [round(v, 1) for v in
                             q(torch.as_tensor(s0)).tolist()],
              f" (most a +1 per step reward can sum to:"
              f" {1 / (1 - GAMMA):.0f})")
    for name, kwargs in (("no target network", {"use_target": False}),
                         ("no replay (buffer 128)", {"buffer_size": 128})):
        c, _ = train(steps, log=False, **kwargs)
        print(f"{name:22s} best {max(c):6.1f}  last {c[-1]:6.1f}")
