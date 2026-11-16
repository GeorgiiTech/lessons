# Lesson 08 · The Training Loop Done Right

- Device-agnostic code: cuda, then mps, then cpu.
- Seeds: the same run gives the same numbers.
- A train / validation split (shuffle first).
- model.train() and model.eval(): dropout on and off; torch.inference_mode() for evaluation.
- train_one_epoch and evaluate as functions; keeping the best weights by validation loss.

```bash
uv run python lessons/08-training-loop/lesson.py
uv run python lessons/08-training-loop/gym.py
uv run python lessons/08-training-loop/solution.py
```

Videos: EN — soon · UA — скоро
