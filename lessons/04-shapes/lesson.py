"""Lesson 04 · Tensor Shapes: Reshape, Broadcast, Matmul.

Run:  uv run python lessons/04-shapes/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import torch


def first_line(err: Exception) -> str:
    return str(err).split("\n")[0].split(". ")[0]


# %% [1] reshape: same numbers, new shape.
def reshape_basics() -> None:
    x = torch.arange(6.0)                 # shape (6,)
    print("x          :", x.tolist())
    print("(2, 3)     :", x.reshape(2, 3).tolist())
    print("(3, 2)     :", x.reshape(3, 2).tolist())
    print("(-1, 2)    :", tuple(x.reshape(-1, 2).shape))


# %% [2] The classic mistake: reshape is not transpose.
def reshape_vs_transpose() -> None:
    x = torch.arange(6.0).reshape(2, 3)
    print("x.reshape(3, 2):", x.reshape(3, 2).tolist())
    print("x.T            :", x.T.tolist())


# %% [3] view needs contiguous memory; reshape copies if needed.
def view_vs_reshape() -> None:
    x = torch.arange(6.0).reshape(2, 3)
    t = x.T
    print("x.T contiguous:", t.is_contiguous())
    try:
        t.view(6)
    except RuntimeError as err:
        print("view error    :", first_line(err)[:52])
    print("reshape works :", t.reshape(6).tolist())


# %% [4] permute: reorder dimensions (an image: C, H, W).
def permute_image() -> None:
    img = torch.rand(3, 224, 224)         # channels first
    hwc = img.permute(1, 2, 0)            # channels last
    print("CHW:", tuple(img.shape), "-> HWC:", tuple(hwc.shape))


# %% [5] unsqueeze / squeeze: add or remove a size-1 dim.
def batch_dim() -> None:
    sample = torch.rand(3, 224, 224)
    batch = sample.unsqueeze(0)           # one image as a batch
    print("unsqueeze(0):", tuple(batch.shape))
    print("squeeze(0)  :", tuple(batch.squeeze(0).shape))


# %% [6] Broadcasting in PyTorch: add a bias to every row.
def bias_broadcast() -> None:
    scores = torch.zeros(64, 10)          # 64 samples, 10 classes
    bias = torch.arange(10.0)             # shape (10,)
    out = scores + bias
    print("(64, 10) + (10,) ->", tuple(out.shape))


# %% [7] Matrix multiplication: inner sizes must match.
def matmul_rules() -> None:
    a = torch.rand(2, 3)
    b = torch.rand(3, 4)
    print("(2, 3) @ (3, 4) ->", tuple((a @ b).shape))
    try:
        a @ a
    except RuntimeError as err:
        print("(2, 3) @ (2, 3) ->", first_line(err))


# %% [8] Batched matmul: the batch dims come along.
def batched_matmul() -> None:
    xb = torch.rand(64, 5, 3)             # 64 samples of (5, 3)
    w = torch.rand(3, 4)
    print("(64, 5, 3) @ (3, 4) ->", tuple((xb @ w).shape))


# %% [9] Joining tensors: cat vs stack.
def cat_vs_stack() -> None:
    a = torch.zeros(2, 3)
    b = torch.ones(2, 3)
    print("cat dim=0  :", tuple(torch.cat([a, b], dim=0).shape))
    print("cat dim=1  :", tuple(torch.cat([a, b], dim=1).shape))
    print("stack dim=0:", tuple(torch.stack([a, b], dim=0).shape))


# %% [10] Indexing can drop a dimension.
def index_keeps_dim() -> None:
    x = torch.rand(4, 3)
    print("x[0]   :", tuple(x[0].shape))      # dim dropped
    print("x[0:1] :", tuple(x[0:1].shape))    # dim kept
    print("x[None]:", tuple(x[None].shape))   # like unsqueeze(0)


# %% [11] Best practice: check shapes where tensors meet.
def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == w.shape[0], f"{x.shape} @ {w.shape}"
    return x @ w


if __name__ == "__main__":
    torch.manual_seed(0)
    reshape_basics()
    reshape_vs_transpose()
    view_vs_reshape()
    permute_image()
    batch_dim()
    bias_broadcast()
    matmul_rules()
    batched_matmul()
    cat_vs_stack()
    index_keeps_dim()
    print("linear:", tuple(linear(torch.rand(8, 3), torch.rand(3, 2)).shape))
