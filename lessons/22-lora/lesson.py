"""Lesson 22 · Hugging Face + fine-tuning a small model with LoRA.

Run:  uv sync --extra nlp
      uv run python lessons/22-lora/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
The first run downloads DistilGPT-2 (about 350 MB) and Tiny Shakespeare.
"""

import os
import time
import urllib.request
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

QUICK = os.environ.get("LESSONS_QUICK") == "1"
NAME = "distilbert/distilgpt2"
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_text(path: Path = Path("data/tinyshakespeare.txt")) -> str:
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text()


# %% [1] A pretrained model and its tokenizer, in two lines.
def load(device):
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME).to(device)
    return tok, model


@torch.no_grad()
def sample(model, tok, prompt: str, device, n: int = 60) -> str:
    model.eval()
    torch.manual_seed(3)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=n, do_sample=True,
                         temperature=0.8, no_repeat_ngram_size=3,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0])


# %% [2] Loss per CHARACTER, so we can compare with lesson 21 (1.555).
@torch.no_grad()
def loss_per_char(model, ids: torch.Tensor, chars: int, device,
                  block: int = 128) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(ids) - 1, block):
        x = ids[i:i + block + 1].to(device)
        if len(x) < 2:
            break
        logits = model(x[:-1][None]).logits[0]
        total += torch.nn.functional.cross_entropy(
            logits, x[1:], reduction="sum").item()
    return total / chars


# %% [3] LoRA: freeze the model, learn a small low-rank update.
def add_lora(model):
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                        target_modules=["c_attn"], task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def count(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total


# %% [4] Fine-tune on Shakespeare: the training loop from lesson 8.
def train(model, ids: torch.Tensor, device, steps: int, block: int = 128):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    start = time.time()
    for step in range(1, steps + 1):
        starts = torch.randint(len(ids) - block - 1, (8,))
        x = torch.stack([ids[s:s + block + 1] for s in starts]).to(device)
        loss = model(x[:, :-1], labels=x[:, :-1]).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % (steps // 5) == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}"
                  f"  {time.time() - start:4.0f} s")


if __name__ == "__main__":
    torch.manual_seed(0)
    device = pick_device()
    tok, model = load(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}"
          f"  vocabulary: {len(tok):,}")
    print("tok('Hello'):", tok("Hello").input_ids,
          " tok('ROMEO:'):", tok("ROMEO:").input_ids)

    text = load_text()
    n = int(0.9 * len(text))
    val_text = text[n:n + (20_000 if QUICK else 60_000)]
    train_ids = torch.tensor(tok(text[:n]).input_ids)
    val_ids = torch.tensor(tok(val_text).input_ids)
    print(f"train tokens {len(train_ids):,}  "
          f"({n / len(train_ids):.2f} characters per token)")

    before = loss_per_char(model, val_ids, len(val_text), device)
    print(f"before: {before:.3f} per character")
    print(sample(model, tok, "ROMEO:", device))

    model = add_lora(model)
    t, total = count(model)
    print(f"trainable {t:,} of {total:,}  ({100 * t / total:.2f}%)")
    a = model.base_model.model.transformer.h[0].attn.c_attn
    print("lora_A", tuple(a.lora_A.default.weight.shape),
          " lora_B", tuple(a.lora_B.default.weight.shape),
          " B starts at zero:", bool((a.lora_B.default.weight == 0).all()))

    train(model, train_ids, device, steps=50 if QUICK else 600)
    after = loss_per_char(model, val_ids, len(val_text), device)
    print(f"after:  {after:.3f} per character  (lesson 21 from scratch:"
          f" 1.555)")
    print(sample(model, tok, "ROMEO:", device))

    out = Path("runs/l22/adapter")
    model.save_pretrained(out)
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f"adapter on disk: {size / 1e6:.2f} MB"
          f"  (full model: {4 * total / 1e6:.0f} MB in float32)")
