"""Lesson 17 · Fast training: device, mixed precision, compile, profiler.

Run:  uv run python lessons/17-speed/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
Times are from the machine you run it on: compare them, don't copy them.
"""

import os
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import ProfilerActivity, profile

QUICK = os.environ.get("LESSONS_QUICK") == "1"
STEPS = 5 if QUICK else 50


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_cnn() -> nn.Module:                 # lesson 13's CNN, a bit wider
    def block(c_in: int, c_out: int) -> list[nn.Module]:
        return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*block(3, 64), *block(64, 128), *block(128, 256),
                         nn.Flatten(), nn.Linear(256 * 4 * 4, 10))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# %% [1] Measure correctly: warm up, then synchronize before reading time.
def time_steps(model, device, batch=256, amp=False, steps=STEPS) -> float:
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(batch, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch,), device=device)

    def step():
        with torch.autocast(device.type, dtype=torch.float16 if
                            device.type != "cpu" else torch.bfloat16,
                            enabled=amp):
            loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(3):                       # warm-up: kernels, caches
        step()
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        step()
    sync(device)                             # wait for the GPU to finish
    return (time.perf_counter() - start) / steps * 1000   # ms per step


# %% [2] The trap: timing without synchronize measures almost nothing.
def unsynced(device: torch.device) -> tuple[float, float]:
    a = torch.randn(4096, 4096, device=device)
    sync(device)
    start = time.perf_counter()
    for _ in range(10):
        b = a @ a
    queued = time.perf_counter() - start     # GPU may still be working
    sync(device)
    done = time.perf_counter() - start
    return queued * 1000, done * 1000


# %% [3] Mixed precision: half the bytes per number.
def precision_demo() -> None:
    x = torch.tensor([1.0001, 65504.0, 70000.0, 1e-8])
    print("float32 :", x.tolist())
    print("float16 :", x.half().tolist())
    print("bfloat16:", x.bfloat16().tolist())


# %% [4] The profiler: where does the time go?
def profile_one_step(model, device) -> None:
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        time_steps(model, device, steps=1)
    events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total,
                    reverse=True)
    total = sum(e.self_cpu_time_total for e in events)
    for e in events[:5]:
        share = e.self_cpu_time_total / total * 100
        print(f"{e.key[:30]:30s} {share:5.1f} %  x{e.count}")


if __name__ == "__main__":
    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    cpu_ms = time_steps(make_cnn(), torch.device("cpu"),
                        steps=2 if QUICK else 10)
    dev_ms = time_steps(make_cnn(), device)
    print(f"cpu    {cpu_ms:7.1f} ms/step")
    print(f"{device.type:6s} {dev_ms:7.1f} ms/step   x{cpu_ms / dev_ms:.1f}")

    queued, done = unsynced(device)
    print(f"10 matmuls: {queued:.1f} ms queued, {done:.1f} ms finished")

    precision_demo()
    amp_ms = time_steps(make_cnn(), device, amp=True)
    print(f"{device.type} + autocast fp16 {amp_ms:7.1f} ms/step   "
          f"x{dev_ms / amp_ms:.2f}")

    for batch in (32, 128, 512):
        ms = time_steps(make_cnn(), device, batch=batch)
        print(f"batch {batch:4d}: {ms:6.1f} ms/step  "
              f"{batch / ms * 1000:8.0f} pictures/s")

    compiled = torch.compile(make_cnn())
    start = time.perf_counter()
    comp_ms = time_steps(compiled, device)
    total = time.perf_counter() - start
    print(f"compiled {comp_ms:7.1f} ms/step   x{dev_ms / comp_ms:.2f}   "
          f"(first call incl. compile: {total:.1f} s)")

    profile_one_step(make_cnn(), device)
