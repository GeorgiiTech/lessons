"""Lesson 05 · Autograd: How PyTorch Computes Gradients.

Run:  uv run python lessons/05-autograd/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import torch


def first_line(err: Exception) -> str:
    return str(err).split("\n")[0][:70]


# %% [1] requires_grad: ask PyTorch to watch a tensor.
def first_gradient() -> None:
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2 + 3 * x            # y = x² + 3x
    print("y =", y.item(), "| grad_fn:", type(y.grad_fn).__name__)
    y.backward()                  # dy/dx = 2x + 3
    print("dy/dx at x=2:", x.grad.item())


# %% [2] A tiny model: y = w·x + b, loss = (y - target)².
def chain_rule() -> None:
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x, target = torch.tensor(2.0), torch.tensor(10.0)
    y = w * x + b                 # 7
    loss = (y - target) ** 2      # 9
    loss.backward()
    print("y =", y.item(), "| loss =", loss.item())
    print("dloss/dw =", w.grad.item(), "| dloss/db =", b.grad.item())


# %% [3] Gradients ADD UP across backward calls.
def accumulation() -> None:
    w = torch.tensor(3.0, requires_grad=True)
    for step in range(1, 3):
        loss = (w * 2.0 - 10.0) ** 2
        loss.backward()
        print(f"after backward #{step}: w.grad =", w.grad.item())
    w.grad = None                 # what optimizer.zero_grad() does
    print("after reset: w.grad =", w.grad)


# %% [4] One gradient-descent step by hand.
def one_step() -> None:
    w = torch.tensor(3.0, requires_grad=True)
    lr = 0.05
    for step in range(3):
        loss = (w * 2.0 + 1.0 - 10.0) ** 2
        loss.backward()
        with torch.no_grad():     # the update is not part of the model
            w -= lr * w.grad
        print(f"step {step}: loss {loss.item():.2f}, w -> {w.item():.3f}")
        w.grad = None


# %% [5] Vectors: the gradient has the shape of the input.
def vector_grad() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    f = (x ** 2).sum()            # f = x1² + x2² + x3²
    f.backward()
    print("x.grad =", x.grad.tolist())   # 2x


# %% [6] Turning tracking off: no_grad and detach.
def no_tracking() -> None:
    w = torch.tensor(3.0, requires_grad=True)
    with torch.no_grad():
        y = w * 2
    print("inside no_grad, requires_grad:", y.requires_grad)
    z = (w * 2).detach()
    print("after detach, requires_grad:", z.requires_grad)


# %% [7] The classic mistake: backward() on a non-scalar.
def non_scalar() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2                     # 3 numbers, not one
    try:
        y.backward()
    except RuntimeError as err:
        print("error:", first_line(err))
    y.sum().backward()            # reduce to one number first
    print("fixed, x.grad =", x.grad.tolist())


# %% [8] Trust, but check: compare with a numerical slope.
def numeric_check() -> None:
    f = lambda t: t ** 3          # f'(t) = 3t²
    t = torch.tensor(2.0, requires_grad=True)
    f(t).backward()
    h = 1e-3
    numeric = (f(torch.tensor(2.0 + h)) - f(torch.tensor(2.0 - h))) / (2 * h)
    print(f"autograd {t.grad.item():.4f} | numeric {numeric.item():.4f}")


if __name__ == "__main__":
    first_gradient()
    chain_rule()
    accumulation()
    one_step()
    vector_grad()
    no_tracking()
    non_scalar()
    numeric_check()
