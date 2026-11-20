"""Lesson 17 · Gym: practice what you just watched.

Fill in every TODO, then run:  uv run python lessons/17-speed/gym.py
If all asserts pass you'll see "All checks passed". Stuck? See solution.py.
"""

import time

import torch
from torch import nn


def ms_per_call(fn, warmup: int = 3, repeats: int = 20) -> float:
    """Average milliseconds per call of fn(): call it `warmup` times
    first (not timed), then time `repeats` calls with perf_counter."""
    # TODO: warm-up loop, start = time.perf_counter(), timed loop
    raise NotImplementedError


def param_megabytes(model: nn.Module) -> float:
    """Memory of all parameters in MB (1 MB = 1024 * 1024 bytes):
    numel() times element_size() for every parameter."""
    # TODO: sum(p.numel() * p.element_size() for ...) / 2**20
    raise NotImplementedError


if __name__ == "__main__":
    calls = []
    ms = ms_per_call(lambda: (calls.append(1), time.sleep(0.01)),
                     warmup=2, repeats=5)
    assert len(calls) == 7, "2 warm-up calls + 5 timed calls"
    assert 9 < ms < 30, f"about 10 ms per call, got {ms:.1f}"

    model = nn.Linear(1024, 1024)            # 1,049,600 parameters
    mb32 = param_megabytes(model)
    assert abs(mb32 - 4.004) < 0.01, mb32
    assert abs(param_megabytes(model.half()) - mb32 / 2) < 0.01, \
        "float16 needs half the memory"
    print("All checks passed")
