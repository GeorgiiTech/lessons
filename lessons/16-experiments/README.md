# Lesson 16 · Reproducible experiments and checkpoints

- One `Config` dataclass per run, saved as `config.json` next to its results in `runs/<name>/`.
- Seeds: two unseeded runs of the same config gave 87.05% and 85.80% validation accuracy; seed 0 gave 86.05% twice.
  Five seeds: 86.05, 85.75, 86.55, 86.60, 86.20 (mean 86.23%, std 0.32): differences below ~1 point can be pure luck.
- Checkpoints with model, optimizer, epoch, config and RNG state: stopped after epoch 3 and resumed → exactly the same
  86.05% as the uninterrupted run.
- TensorBoard: `SummaryWriter(...).add_scalar(...)`; view with `uv run tensorboard --logdir runs`.
- A tiny learning-rate sweep: 1e-2 → 85.00%, 1e-3 → 86.05%, 1e-4 → 78.65%.
- Gym: a run name from a config, and save/load a full checkpoint.

```bash
uv run python lessons/16-experiments/lesson.py
uv run python lessons/16-experiments/gym.py
uv run python lessons/16-experiments/solution.py
```
