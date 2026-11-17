# Lesson 09 · Datasets and DataLoaders

- A `Dataset` answers two questions: `__len__` (how many) and `__getitem__` (give me example i).
- Transforms run on every example as it is read; `TensorDataset` when everything already fits in tensors.
- `random_split` with a seeded generator for train / validation.
- `DataLoader`: batch_size, shuffle (a new order every epoch), drop_last.
- Mini-batch training: more steps per epoch, a better model in the same number of epochs.
- `num_workers` (+ `persistent_workers`) for slow data; a custom `collate_fn` pads examples of different lengths.

```bash
uv run python lessons/09-data/lesson.py
uv run python lessons/09-data/gym.py
uv run python lessons/09-data/solution.py
```

Videos: EN — soon · UA — скоро
