# Lesson 02 · Thinking in Arrays: NumPy and Vectorization

- A Python loop vs one NumPy call on millions of numbers (measured).
- Arrays: one block of memory, one dtype, a shape. NumPy's default float64 vs PyTorch's float32.
- Element-wise math, boolean masks, aggregations with `axis`.
- Broadcasting, standardizing features, and the classic `(3,)` vs `(3, 1)` trap.
- NumPy ↔ PyTorch: `torch.from_numpy` shares memory, `torch.tensor` copies.

```bash
uv run python lessons/02-numpy/lesson.py      # the code from the video
uv run python lessons/02-numpy/gym.py         # your exercise (fill in the TODOs)
uv run python lessons/02-numpy/solution.py    # one possible solution
```

Videos: [EN](https://youtu.be/I8gNEusabGU) · [UA](https://youtu.be/0pfXluBHkoA)
