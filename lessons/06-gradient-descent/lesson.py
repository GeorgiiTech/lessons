"""Lesson 06 · Gradient Descent From Scratch.

Run:  uv run python lessons/06-gradient-descent/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
"""

import torch


# %% [1] The data: study hours -> exam score (true line: 5 * hours + 40, plus noise).
def make_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    hours = torch.rand(100) * 10                 # 100 students, 0..10 hours
    noise = torch.randn(100) * 5                 # real scores are never exact
    scores = 5 * hours + 40 + noise
    return hours, scores


def show_data(hours: torch.Tensor, scores: torch.Tensor) -> None:
    print("shape:", tuple(hours.shape), tuple(scores.shape))
    for h, s in zip(hours[:3], scores[:3]):
        print(f"{h:.1f} h -> {s:.0f} points")


# %% [2] The model: a line. And a loss: mean squared error.
def predict(w: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return w * x + b


def mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).mean()        # one number for the whole dataset


def start_loss(hours: torch.Tensor, scores: torch.Tensor) -> None:
    w, b = torch.tensor(0.0), torch.tensor(0.0)
    print(f"w=0, b=0 -> loss {mse(predict(w, b, hours), scores):.0f}")


# %% [3] Training: forward, loss, backward, step, reset. Repeat.
def train(x: torch.Tensor, y: torch.Tensor, lr: float, epochs: int, log_every: int):
    w = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    for epoch in range(epochs + 1):
        loss = mse(predict(w, b, x), y)        # 1. forward + loss
        loss.backward()                        # 2. gradients
        with torch.no_grad():                  # 3. step
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None                          # 4. reset
        b.grad = None
        if epoch % log_every == 0:
            print(f"epoch {epoch:4d}: loss {loss.item():8.2f}  w {w.item():5.2f}  b {b.item():6.2f}")
    return w.detach(), b.detach()


# %% [4] Too big a step: the loss explodes.
def too_big(hours: torch.Tensor, scores: torch.Tensor) -> None:
    train(hours, scores, lr=0.05, epochs=4, log_every=1)


# %% [5] Check against the exact answer (least squares).
def exact(hours: torch.Tensor, scores: torch.Tensor) -> None:
    X = torch.stack([hours, torch.ones_like(hours)], dim=1)   # (100, 2)
    sol = torch.linalg.lstsq(X, scores.unsqueeze(1)).solution
    print(f"exact: w {sol[0, 0]:.2f}  b {sol[1, 0]:.2f}")


# %% [6] Use the model: predict a score.
def use(w: torch.Tensor, b: torch.Tensor) -> None:
    print(f"7 hours -> {predict(w, b, torch.tensor(7.0)):.0f} points")


# %% [7] Scale the input: same answer, 10x fewer epochs.
def scaled(hours: torch.Tensor, scores: torch.Tensor) -> None:
    w, b = train(hours / 10, scores, lr=0.5, epochs=100, log_every=50)
    print(f"back in hours: w {w / 10:.2f}  b {b:.2f}")


if __name__ == "__main__":
    hours, scores = make_data()
    show_data(hours, scores)
    start_loss(hours, scores)
    w, b = train(hours, scores, lr=0.02, epochs=1000, log_every=250)
    too_big(hours, scores)
    exact(hours, scores)
    use(w, b)
    scaled(hours, scores)
