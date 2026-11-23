# Lesson 24 · Deep Q-learning (DQN) on CartPole

- Q-network: 4 → 128 → 128 → 2 (17,410 parameters), one Q-value per action.
- Replay buffer (50,000 transitions, overwrite the oldest, random batches of 128), a target network synced every 1,000 steps,
  the Bellman target `r + 0.99 · max Q_target(s')` cut only by `terminated` (not by `truncated`), Huber loss, gradient clipping 10,
  epsilon-greedy 1.0 → 0.05 over the first 10,000 steps. CPU on purpose (a tiny network).
- 60,000 steps in 28 s: greedy score 137.8 → 368.9 at step 15,000, then down to 64-146 (DQN is unstable); the best weights,
  kept like in lesson 8, score 360.0 on 100 new episodes.
- Same run without a target network: collapses (9.0). With a 128-transition buffer instead of 50,000: best 366.9, last 177.3.
- Gym: the Bellman target, and epsilon-greedy.

```bash
uv run python lessons/24-dqn/lesson.py
uv run python lessons/24-dqn/gym.py
uv run python lessons/24-dqn/solution.py
```
