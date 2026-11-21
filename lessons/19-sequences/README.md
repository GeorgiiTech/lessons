# Lesson 19 · Embeddings and sequences

- `nn.Embedding` is a lookup table: ids [3, 1, 3] → three rows; the same as one-hot @ weight; only used rows get a gradient.
- An RNN by hand (h = tanh(W x + U h + b)) gives exactly `nn.RNN`'s last output.
- Why they struggle: the gradient that reaches a token 50 steps back is 8e-17 (RNN) and 9e-14 (LSTM) at initialization.
- Memory task (read T random tokens, name the first one; 2000 steps, Adam 1e-2):
  T=5: all 100%. T=20: RNN 68.1%, LSTM 9.4%, LSTM with forget-gate bias 1: 100%. T=50: RNN 49.6%, LSTMs at chance (≈10%).
- One step after another: an LSTM forward takes 4.9 / 20.4 / 100.8 ms for 50 / 200 / 800 tokens (CPU).
- Gym: an embedding lookup by indexing, and padding a batch of sequences.

```bash
uv run python lessons/19-sequences/lesson.py
uv run python lessons/19-sequences/gym.py
uv run python lessons/19-sequences/solution.py
```
