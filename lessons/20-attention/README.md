# Lesson 20 · Attention from scratch

- One head: scores = q @ kᵀ / √d → (B, T, T), softmax per row (rows sum to 1), weights @ v → (B, T, C);
  identical to `F.scaled_dot_product_attention`.
- A causal mask (`torch.triu(..., diagonal=1)` → −inf): token t only looks at tokens 0..t.
- Multi-head attention: (B, T, C) → (B, heads, T, C/heads) → attend → merge; 4,224 parameters for C = 32, 4 heads.
- Lesson 19's memory task (name the first of T tokens) with one attention layer + positional embeddings, 300 steps:
  100% at T = 20, 50 and 200 (the LSTMs were at chance at T = 50).
- The price: the attention weights grow with T²: 0.6 M numbers at T = 50, 163.8 M at T = 800 (batch 64, 4 heads);
  forward 3.5 / 17.7 / 195.3 ms on the CPU for 50 / 200 / 800 tokens.
- Gym: attention in two lines, and the causal mask.

```bash
uv run python lessons/20-attention/lesson.py
uv run python lessons/20-attention/gym.py
uv run python lessons/20-attention/solution.py
```
