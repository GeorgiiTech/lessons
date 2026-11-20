# Lesson 15 · Debugging neural networks

Six checks, each with a real planted bug (FashionMNIST, CPU, a few seconds):

1. The first loss: 2.302 for 10 classes, expected ln(10) = 2.303.
2. Overfit one batch of 64: correct 2.302 → 0.002; softmax before cross-entropy stuck at 1.627; no `zero_grad()` ends at 2.505.
3. Look at the data: shuffled labels still overfit one batch (0.004), but on 10,000 pictures train loss 2.229 and validation 9.1%
   (right labels: 84.9%).
4. Shapes: `(100, 1) - (100,)` broadcasts to `(100, 100)`; a linear fit of y = 3x + 1 learns w 0.00, b 1.13 (fixed: 2.99, 1.02).
5. Registered weights: layers in a Python list → 0 parameters ("optimizer got an empty parameter list"); `nn.ModuleList` → 203,530.
6. NaN and infinity: `-log(softmax)` of logits [1000, 0] is inf, `F.cross_entropy` gives 1000.0; inf × 0 = nan; `torch.isfinite`.

Gym: an overfit-one-batch check (`can_overfit`) and gradient norms per parameter (`grad_norms`).

```bash
uv run python lessons/15-debugging/lesson.py
uv run python lessons/15-debugging/gym.py
uv run python lessons/15-debugging/solution.py
```
