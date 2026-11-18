"""Lesson 10 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/10-classifier/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
No dataset needed: the checks use small hand-made tensors.
"""

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Share of rows where the highest logit is the true label."""
    # TODO
    raise NotImplementedError


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor,
                     k: int) -> torch.Tensor:
    """k × k int64 matrix: rows = true class, columns = predicted class."""
    # TODO (bonus: no Python loop, try torch.bincount)
    raise NotImplementedError


def per_class_accuracy(matrix: torch.Tensor) -> torch.Tensor:
    """Correct / total for every true class (diagonal / row sums)."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    logits = torch.tensor([[2.0, 0.1, 0.3],
                           [0.2, 1.5, 0.1],
                           [0.1, 0.2, 3.0],
                           [1.0, 2.0, 0.0]])
    labels = torch.tensor([0, 1, 2, 0])
    assert abs(accuracy(logits, labels) - 0.75) < 1e-6, "3 of 4 right"

    preds = torch.tensor([0, 1, 2, 1, 2, 2])
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    m = confusion_matrix(preds, labels, k=3)
    assert m.dtype == torch.int64, "counts are integers"
    assert m.tolist() == [[1, 1, 0], [0, 1, 1], [0, 0, 2]], "rows = true"
    assert m.sum().item() == 6, "every example in exactly one cell"

    acc = per_class_accuracy(m)
    assert torch.allclose(acc, torch.tensor([0.5, 0.5, 1.0])), "diag / rows"
    print("All checks passed")
