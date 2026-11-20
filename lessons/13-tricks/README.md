# Lesson 13 · Training tricks: init, normalization, schedules, clipping

- Initialization: the activation std through 20 linear + ReLU layers explodes with std 1 (1.9e+21), vanishes with std 0.01
  (1.9e-19), shrinks to ~0.02 with PyTorch's default, and stays ~0.9 with `nn.init.kaiming_normal_`.
- A 10-layer MLP, SGD lr 0.01, 3 epochs: default init is stuck at loss 2.303 (9.5%), Kaiming reaches 82.7%,
  BatchNorm1d with the default init 86.1% (best validation accuracy).
- BatchNorm vs LayerNorm; a batch of 1 fails in train mode, works in eval mode (running averages).
- Gradient clipping: SGD with momentum, lr 0.1: without clipping the gradient norm hits 1.6e+07 and the loss is NaN;
  `clip_grad_norm_(..., 1.0)` trains to 83.2%.
- Lesson 12's CNN (test 91.3%) with BatchNorm2d (`bias=False` in the convs) + OneCycleLR (max lr 1e-2): test 92.3%.
- Gym: write BatchNorm (train mode) and gradient-norm clipping yourself.

```bash
uv run python lessons/13-tricks/lesson.py
uv run python lessons/13-tricks/gym.py
uv run python lessons/13-tricks/solution.py
```
