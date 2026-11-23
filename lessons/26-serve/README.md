# Lesson 26 · Save, export, serve

Needs the optional extra: `uv sync --extra serve` (onnx, onnxscript, onnxruntime, fastapi, httpx).

- Lesson 10's FashionMNIST MLP, 3 epochs: 87.1% on 2,000 test pictures.
- `state_dict`: 816 KB; a fresh model + `load_state_dict(torch.load(..., weights_only=True))` gives identical outputs.
- `torch.export.export` with a dynamic batch dimension: a graph of 4 operations (flatten, linear, relu, linear), saved as
  `model.pt2` and loaded without the Python class.
- `torch.onnx.export(..., dynamo=True)`: `model.onnx` (820 KB), checked with `onnx.checker`, run with ONNX Runtime:
  max difference vs PyTorch 8.6e-06; one picture 9 µs (PyTorch) vs 7 µs (ONNX Runtime) on the CPU.
- A FastAPI endpoint `POST /predict` (784 pixels in, class + probability out), tested with `TestClient`: an ankle boot is
  predicted "Ankle boot" (0.842); 100 pixels instead of 784 → 422 (validation error).
- Gym: a save/load roundtrip, and a numerically safe softmax top class.

```bash
uv sync --extra serve
uv run python lessons/26-serve/lesson.py
uv run python lessons/26-serve/gym.py
uv run python lessons/26-serve/solution.py
```
