# Lesson 18 · From notebook to project

Lesson 10's FashionMNIST classifier as a small package:

```
lessons/18-project/
  fashion/config.py   Config dataclass + argparse (every field becomes a --flag)
  fashion/data.py     loading the data, nothing else
  fashion/model.py    the network, nothing else
  fashion/train.py    the training loop and main(): python -m fashion.train
  tests/              pytest: output shape, first loss = ln(10), overfit one batch, CLI parsing
```

From `lessons/18-project/`:

```bash
uv run python -m fashion.train --help
uv run python -m fashion.train --epochs 3 --n-train 10000 --data ../../data
uv run python -m pytest tests -q
```

Real run: 3 epochs on 10,000 pictures → validation accuracy 80.7%, 82.6%, 84.8%; the 4 tests pass in about 2 seconds.
Gym: turn a field name into a flag, and a `check_batch` that fails early with a clear message.
