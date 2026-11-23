# Lesson 25 · Policy gradients: REINFORCE, then PPO

Needs the optional extra for the PPO part: `uv sync --extra rl` (Stable-Baselines3).

- A policy network 4 → 64 → 2 with a softmax: probabilities, sampled with `torch.distributions.Categorical`.
- Discounted returns-to-go (gamma 0.99), normalized as a simple baseline; loss = −Σ log π(a|s) · G.
- REINFORCE on CartPole, 600 episodes (Adam 1e-2, 20 s on the CPU): mean of the last 50 episodes 165.0 at episode 100,
  482.0 at 300, 450.4 at the end. Without normalizing the returns: 426.1.
- PPO from Stable-Baselines3 in three lines: 500.0 ± 0.0 on 20 evaluation episodes after 100,000 steps (19 s).
- Gym: discounted returns, and the REINFORCE loss.

```bash
uv sync --extra rl
uv run python lessons/25-policy-gradients/lesson.py
uv run python lessons/25-policy-gradients/gym.py
uv run python lessons/25-policy-gradients/solution.py
```
