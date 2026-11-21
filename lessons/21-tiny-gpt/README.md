# Lesson 21 · Build a tiny GPT

- Tiny Shakespeare (1,115,394 characters, downloaded on the first run) with a character tokenizer: vocabulary 65.
- A transformer block = LayerNorm → causal multi-head attention (`F.scaled_dot_product_attention(is_causal=True)`) → residual,
  LayerNorm → MLP (4× wider, GELU) → residual.
- TinyGPT: token + position embeddings, 4 blocks, 128 dims, 4 heads, context 128 → 826,433 parameters;
  logits (B, T, 65): every position predicts the next character.
- First loss 4.286 ≈ ln 65 = 4.174 (a random guess). 3,000 AdamW steps (batch 64, lr 1e-3) on MPS in about 9 minutes:
  validation loss 1.888 → 1.555.
- Generation one character at a time; temperature 0.5 (safe, repetitive), 1.0, 1.5 (made-up words).
- Gym: next-character pairs, and greedy sampling from the last position.

```bash
uv run python lessons/21-tiny-gpt/lesson.py
uv run python lessons/21-tiny-gpt/gym.py
uv run python lessons/21-tiny-gpt/solution.py
```
