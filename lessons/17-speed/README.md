# Lesson 17 · Making training fast: GPU, mixed precision, torch.compile

Measured on an M4 Pro Mac (MPS); your numbers will differ: compare them, don't copy them.

- A wider CNN, batch 256, 32×32 RGB: CPU 158.3 ms/step, MPS 24.7 ms/step (×6.4).
- Timing without `torch.mps.synchronize()` / `torch.cuda.synchronize()`: 10 big matmuls look like 24.4 ms, really 276.2 ms.
- float16 vs bfloat16: 70000 overflows to inf in float16, 1e-8 becomes 0; bfloat16 keeps the range, loses precision.
- `torch.autocast` (float16) on MPS: ×0.97 here (no gain for this small model); NVIDIA reports up to 3× on Tensor Core GPUs.
- Batch size: 9,843 → 10,302 → 10,243 pictures/s for batch 32 / 128 / 512.
- `torch.compile` on MPS: ×0.84 here, 2.9 s to compile; PyTorch reports 43% faster training on 163 models on an A100.
- `torch.profiler`: the operations that take the most time in one training step.
- Gym: time a function correctly (warm-up), parameter memory in float32 vs float16.

```bash
uv run python lessons/17-speed/lesson.py
uv run python lessons/17-speed/gym.py
uv run python lessons/17-speed/solution.py
```
