# Lesson 23 · Gymnasium: the environment API and CartPole

- `gym.make("CartPole-v1")`: observation space Box (4,) float32 (cart position, cart velocity, pole angle, pole angular
  velocity), action space Discrete(2) (push left / right).
- `reset(seed=...)` → `(obs, info)`; `step(action)` → `(obs, reward, terminated, truncated, info)`: reward 1 per step.
- An episode ends when `terminated` (the pole fell or the cart left the track) OR `truncated` (the 500-step time limit).
- 10,000 random steps: about 4 microseconds per step.
- 100 episodes each: random policy mean 24.1 steps (best 73); a one-line rule (push toward the side the pole falls to)
  500.0, every episode truncated at the limit.
- A linear PyTorch policy (`nn.Linear(4, 2)`) found by random search over 100 sets of random weights: best mean 252.1 at try 38.
- Gym: the episode loop with `terminated or truncated`, and your own rule that averages ≥ 195 steps.

```bash
uv run python lessons/23-gymnasium/lesson.py
uv run python lessons/23-gymnasium/gym.py
uv run python lessons/23-gymnasium/solution.py
```
