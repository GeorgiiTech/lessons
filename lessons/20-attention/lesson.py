"""Lesson 20 · Attention from scratch, with the shapes at every step.

Run:  uv run python lessons/20-attention/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
No dataset needed. CPU, about a minute.
"""

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

QUICK = os.environ.get("LESSONS_QUICK") == "1"
torch.set_num_threads(4)


# %% [1] One head of attention, step by step: (B, T, C) in and out.
def attention(q, k, v, causal: bool = False):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)      # (B, T, T)
    if causal:
        T = q.shape[-2]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)                      # rows sum to 1
    return weights @ v, weights                           # (B, T, C)


def shapes_demo() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)                 # 2 sequences, 5 tokens, 8 dims
    wq, wk, wv = (nn.Linear(8, 8, bias=False) for _ in range(3))
    q, k, v = wq(x), wk(x), wv(x)
    out, w = attention(q, k, v)
    print("x", tuple(x.shape), " q k v", tuple(q.shape))
    sums = [round(v, 3) for v in w[0].sum(-1).tolist()]
    print("weights", tuple(w.shape), " row sums", sums)
    print("out", tuple(out.shape))
    ref = F.scaled_dot_product_attention(q, k, v)
    print("same as F.scaled_dot_product_attention:",
          torch.allclose(out, ref, atol=1e-6))


# %% [2] A causal mask: token t may only look at tokens 0..t.
def causal_demo() -> None:
    torch.manual_seed(0)
    q = k = v = torch.randn(1, 4, 8)
    _, w = attention(q, k, v, causal=True)
    for row in w[0]:
        print("  ".join(f"{p:.2f}" for p in row.tolist()))


# %% [3] Multi-head attention: split C into heads, attend, merge.
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, causal: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, heads, T, C // heads)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2)
                   for t in (q, k, v))
        y, _ = attention(q, k, v, causal)
        y = y.transpose(1, 2).reshape(B, T, C)            # merge heads
        return self.out(y)


# %% [4] Lesson 19's memory task, now with attention.
class AttnNet(nn.Module):
    def __init__(self, length: int, vocab: int = 10, dim: int = 32) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(length, dim)              # where am I?
        self.attn = MultiHeadAttention(dim, heads=4)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T))
        h = h + self.attn(h)
        return self.head(h[:, -1])                        # last position


def memory_task(length: int, steps: int) -> float:
    torch.manual_seed(0)
    model = AttnNet(length)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x = torch.randint(0, 10, (64, length))
        loss = F.cross_entropy(model(x), x[:, 0])
        opt.zero_grad()
        loss.backward()
        opt.step()
    x = torch.randint(0, 10, (2000, length))
    with torch.no_grad():
        return (model(x).argmax(1) == x[:, 0]).float().mean().item()


# %% [5] All positions at once: time vs length.
def time_per_length(length: int) -> float:
    model = AttnNet(length)
    x = torch.randint(0, 10, (64, length))
    with torch.no_grad():
        model(x)
        start = time.perf_counter()
        for _ in range(5):
            model(x)
    return (time.perf_counter() - start) / 5 * 1000


if __name__ == "__main__":
    shapes_demo()
    causal_demo()
    mha = MultiHeadAttention(32, heads=4)
    n = sum(p.numel() for p in mha.parameters())
    print("multi-head out", tuple(mha(torch.randn(2, 5, 32)).shape),
          f" parameters {n:,}")
    steps = 100 if QUICK else 300
    for length in (20, 50, 200):
        acc = memory_task(length, steps)
        print(f"T={length:3d}  attention acc {acc:.3f}")
    for length in (50, 200, 800):
        ms = time_per_length(length)
        cells = 64 * 4 * length * length
        print(f"T={length:4d}  forward {ms:6.1f} ms  "
              f"attention weights {cells / 1e6:6.1f} M numbers")
