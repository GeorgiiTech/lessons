"""Lesson 21 · Build a tiny GPT and train it on Shakespeare.

Run:  uv run python lessons/21-tiny-gpt/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
The first run downloads Tiny Shakespeare (about 1 MB) into ./data.
"""

import math
import os
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

QUICK = os.environ.get("LESSONS_QUICK") == "1"
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# %% [1] Characters as tokens: the simplest possible tokenizer.
def load_text(path: Path = Path("data/tinyshakespeare.txt")) -> str:
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text()


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self.index[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


# %% [2] One transformer block: attention + MLP, each with a residual.
class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


# %% [3] The GPT: embeddings, a stack of blocks, a head over the vocabulary.
class TinyGPT(nn.Module):
    def __init__(self, vocab: int, context: int = 128, dim: int = 128,
                 layers: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.Sequential(*[Block(dim, heads)
                                      for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        return self.head(self.ln(self.blocks(x)))      # (B, T, vocab)


# %% [4] Training data: every position predicts the next character.
def get_batch(data: torch.Tensor, context: int, batch: int, device):
    starts = torch.randint(len(data) - context - 1, (batch,))
    x = torch.stack([data[s:s + context] for s in starts])
    y = torch.stack([data[s + 1:s + context + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def val_loss(model, data, context, device, batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, context, 64, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1),
                                      y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


# %% [5] Generation: sample one character at a time, with a temperature.
@torch.no_grad()
def generate(model, idx: torch.Tensor, steps: int,
             temperature: float = 1.0) -> torch.Tensor:
    model.eval()
    for _ in range(steps):
        logits = model(idx[:, -model.context:])[:, -1] / temperature
        probs = logits.softmax(dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


if __name__ == "__main__":
    torch.manual_seed(0)
    device = pick_device()
    text = load_text()
    tok = CharTokenizer(text)
    print(f"{len(text):,} characters, vocabulary {len(tok.chars)}")
    print("encode('Hello'):", tok.encode("Hello"))
    data = torch.tensor(tok.encode(text))
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]

    model = TinyGPT(len(tok.chars)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {params:,}")
    x, y = get_batch(train, model.context, 64, device)
    print("x", tuple(x.shape), "-> logits", tuple(model(x).shape))
    first = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    print(f"first loss {first:.3f}  ln(vocab) {math.log(len(tok.chars)):.3f}")

    steps = 50 if QUICK else 3000
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train, model.context, 64, device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % (steps // 5) == 0:
            v = val_loss(model, val, model.context, device)
            print(f"step {step:5d}  train {loss.item():.3f}  val {v:.3f}"
                  f"  {time.time() - start:5.0f} s")

    prompt = torch.tensor([tok.encode("ROMEO:\n")], device=device)
    for temperature in (0.5, 1.0, 1.5):
        torch.manual_seed(1)
        out = generate(model, prompt, 160, temperature)
        print(f"--- temperature {temperature}")
        print(tok.decode(out[0].tolist()))
