# Lessons · Python for ML with PyTorch

Code for the video series by **Georgii Kharlampiiev**: 26 lessons from tensors to a tiny GPT and reinforcement
learning agents in Gymnasium. Every lesson has:

- `lesson.py` — the code from the video, in the same order (run it top to bottom),
- `gym.py` — a practice exercise: fill in the `TODO`s, run the file, the asserts tell you if you're right,
- `solution.py` — one possible solution.

All code is tested on CPU (and runs on NVIDIA CUDA and Apple Silicon MPS when available).

## Setup

```bash
git clone https://github.com/GeorgiiTech/lessons.git
cd lessons
uv sync                      # installs Python 3.12 + the pinned versions
uv run python lessons/01-setup/lesson.py
uv run pytest                # runs every lesson and solution
```

Without uv: create a Python 3.12 virtual environment and run `pip install -r requirements.txt`.

Extras for later lessons: `uv sync --extra nlp` (21-22), `--extra rl` (25), `--extra serve` (26).

Versions used in the whole series: Python 3.12, PyTorch 2.15.0, torchvision 0.29.2, NumPy 2.5.4, Gymnasium 1.4.0.

## Lessons

| # | Lesson | Code | Video (EN) | Відео (UA) |
|---|---|---|---|---|
| 01 | Why PyTorch and How to Set It Up | [01-setup](lessons/01-setup) | [EN](https://youtu.be/L3bsJqpwbwk) | [UA](https://youtu.be/TmXBpPfG_gQ) |
| 02 | Thinking in Arrays: NumPy and Vectorization | [02-numpy](lessons/02-numpy) | [EN](https://youtu.be/I8gNEusabGU) | [UA](https://youtu.be/0pfXluBHkoA) |
| 03 | Tensors 101 | [03-tensors](lessons/03-tensors) | [EN](https://youtu.be/MxjEdMns93g) | [UA](https://youtu.be/yzwDuoy1J0Q) |
| 04 | Tensor Shapes: Reshape, Broadcast, Matmul | [04-shapes](lessons/04-shapes) | [EN](https://youtu.be/3Q9q5C1KC9o) | [UA](https://youtu.be/ppfM7sTQbJc) |
| 05 | Autograd: How PyTorch Computes Gradients | [05-autograd](lessons/05-autograd) | [EN](https://youtu.be/HCdEWG4co2g) | [UA](https://youtu.be/8KlW4jxTWBU) |
| 06 | Gradient Descent From Scratch | [06-gradient-descent](lessons/06-gradient-descent) | [EN](https://youtu.be/-N6InAF1K6g) | [UA](https://youtu.be/7mdYferU-m8) |
| 07 | nn.Module, Loss Functions and Optimizers | [07-nn-module](lessons/07-nn-module) | [EN](https://youtu.be/VO3UTFaqGjw) | [UA](https://youtu.be/A0oo-k7S4kE) |

Telegram: [EN](https://t.me/kharlampiiev_ai) · [UA](https://t.me/kharlampiiev_ai_ua)
