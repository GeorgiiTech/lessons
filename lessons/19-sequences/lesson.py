"""Lesson 19 · Embeddings and sequences: nn.Embedding, RNN, LSTM.

Run:  uv run python lessons/19-sequences/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
No dataset needed: the sequences are random tokens. CPU, about 2 minutes.
"""

import os
import time

import torch
import torch.nn.functional as F
from torch import nn

QUICK = os.environ.get("LESSONS_QUICK") == "1"
torch.set_num_threads(4)


# %% [1] An embedding is a lookup table: token id -> row of numbers.
def embedding_demo() -> None:
    torch.manual_seed(0)
    emb = nn.Embedding(num_embeddings=10, embedding_dim=4)
    ids = torch.tensor([3, 1, 3])
    vectors = emb(ids)
    print("ids", ids.tolist(), "-> vectors", tuple(vectors.shape))
    one_hot = F.one_hot(ids, num_classes=10).float()
    print("same as one-hot @ weight:",
          torch.allclose(vectors, one_hot @ emb.weight))
    vectors.sum().backward()
    rows = emb.weight.grad.abs().sum(dim=1).nonzero().flatten().tolist()
    print("rows with a gradient:", rows)


# %% [2] An RNN by hand: one hidden state, updated token by token.
def rnn_by_hand() -> None:
    torch.manual_seed(0)
    rnn = nn.RNN(input_size=4, hidden_size=3, batch_first=True)
    x = torch.randn(1, 5, 4)                  # 1 sequence, 5 steps
    h = torch.zeros(3)
    for t in range(5):                        # h = tanh(W x + U h + b)
        h = torch.tanh(rnn.weight_ih_l0 @ x[0, t] + rnn.bias_ih_l0
                       + rnn.weight_hh_l0 @ h + rnn.bias_hh_l0)
    out, h_n = rnn(x)
    print("output", tuple(out.shape), " last state", tuple(h_n.shape))
    print("by hand == nn.RNN:", torch.allclose(h, out[0, -1], atol=1e-6))


class SeqNet(nn.Module):
    """Embedding -> RNN or LSTM -> a prediction from the last state."""

    def __init__(self, kind: str, vocab: int = 10, dim: int = 16,
                 hidden: int = 64, forget_bias: float = 0.0) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        rnn_class = nn.RNN if kind == "rnn" else nn.LSTM
        self.rnn = rnn_class(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
        if kind == "lstm" and forget_bias:    # gates: input, forget, ...
            with torch.no_grad():
                self.rnn.bias_hh_l0[hidden:2 * hidden] = forget_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.emb(x))        # (B, T, hidden)
        return self.head(out[:, -1])          # last step only


# %% [3] Why they struggle: how much gradient reaches early tokens?
def gradient_reach(kind: str, length: int = 100) -> list[float]:
    torch.manual_seed(0)
    model = SeqNet(kind)
    x = torch.randint(0, 10, (32, length))
    e = model.emb(x)
    e.retain_grad()
    out, _ = model.rnn(e)
    F.cross_entropy(model.head(out[:, -1]), x[:, 0]).backward()
    norms = e.grad.norm(dim=2).mean(dim=0)    # one number per position
    return [norms[length - 1 - d].item() for d in (0, 10, 20, 50)]


# %% [4] A memory task: read T random tokens, then name the first one.
def memory_task(kind: str, length: int, steps: int,
                forget_bias: float = 0.0) -> float:
    torch.manual_seed(0)
    model = SeqNet(kind, forget_bias=forget_bias)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x = torch.randint(0, 10, (64, length))
        loss = F.cross_entropy(model(x), x[:, 0])
        opt.zero_grad()
        loss.backward()
        opt.step()
    x = torch.randint(0, 10, (2000, length))
    with torch.no_grad():
        return (model(x).argmax(1) == x[:, 0]).float().mean().item()


# %% [5] One step after another: time grows with the sequence length.
def time_per_length(length: int) -> float:
    model = SeqNet("lstm")
    x = torch.randint(0, 10, (64, length))
    with torch.no_grad():
        model(x)                              # warm-up
        start = time.perf_counter()
        for _ in range(5):
            model(x)
    return (time.perf_counter() - start) / 5 * 1000


if __name__ == "__main__":
    embedding_demo()
    rnn_by_hand()
    for kind in ("rnn", "lstm"):
        g = gradient_reach(kind)
        print(f"{kind:4s} gradient 0/10/20/50 steps back: "
              + "  ".join(f"{v:.0e}" for v in g))
    steps = 100 if QUICK else 2000
    for length in (5, 20, 50):
        rnn = memory_task("rnn", length, steps)
        lstm = memory_task("lstm", length, steps)
        lstm_fb = memory_task("lstm", length, steps, forget_bias=1.0)
        print(f"T={length:3d}  rnn {rnn:.3f}  lstm {lstm:.3f}  "
              f"lstm+forget bias {lstm_fb:.3f}")
    for length in (50, 200, 800):
        ms = time_per_length(length)
        print(f"T={length:4d}  lstm forward {ms:6.1f} ms")
