# Lesson 11 · Overfitting and how to fight it

- Overfitting: 1,000 FashionMNIST pictures and a 669,706-parameter MLP. The training loss falls to almost zero,
  the validation loss turns back up after about 10 epochs.
- `nn.Dropout(p)`: zeroes random activations in `model.train()`, does nothing in `model.eval()`.
- Weight decay with `torch.optim.AdamW(..., weight_decay=...)`: pulls every weight a little towards zero.
- Early stopping: keep the best validation weights, stop after `patience` epochs without a new best.
- The best cure: more data (10 times more pictures: best validation loss 0.564 → 0.397).

```bash
uv run python lessons/11-overfitting/lesson.py
uv run python lessons/11-overfitting/gym.py
uv run python lessons/11-overfitting/solution.py
```

Videos: [EN](https://youtu.be/agYHwPB_zxQ) · [UA](https://youtu.be/vr6NP9uVKx4)
