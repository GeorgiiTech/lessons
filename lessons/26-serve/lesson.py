"""Lesson 26 · Save, export, serve: state_dict, torch.export, ONNX, FastAPI.

Run:  uv sync --extra serve
      uv run python lessons/26-serve/lesson.py
Each "# %% [N]" section is one part of the video, in the same order.
Uses FashionMNIST from ./data (downloaded on the first run). CPU, ~1 min.
"""

import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from torch import nn
from torch.export import Dim
from torchvision import datasets, transforms

QUICK = os.environ.get("LESSONS_QUICK") == "1"
OUT = Path("runs/l26")
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# %% [1] A model worth shipping: lesson 10's classifier.
def make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 10))


def train_quick(epochs: int):
    torch.manual_seed(0)
    tf = transforms.ToTensor()
    train = datasets.FashionMNIST("data", train=True, download=True,
                                  transform=tf)
    test = datasets.FashionMNIST("data", train=False, download=True,
                                 transform=tf)
    if QUICK:
        train = torch.utils.data.Subset(train, range(5_000))
    loader = torch.utils.data.DataLoader(train, batch_size=128,
                                         shuffle=True)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in loader:
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    x = torch.stack([test[i][0] for i in range(2_000)])
    y = torch.tensor([test[i][1] for i in range(2_000)])
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    return model, x, y, acc


# %% [2] Save the weights, not the object.
def save_and_load(model, x) -> nn.Module:
    torch.save(model.state_dict(), OUT / "model.pt")
    fresh = make_model()                        # the code builds the shape
    fresh.load_state_dict(torch.load(OUT / "model.pt", weights_only=True))
    fresh.eval()
    with torch.no_grad():
        same = torch.equal(fresh(x), model(x))
    kb = (OUT / "model.pt").stat().st_size / 1e3
    print(f"state_dict: {kb:.0f} KB, reloaded outputs identical: {same}")
    return fresh


# %% [3] torch.export: the model as a graph, no Python class needed.
def export_graph(model, x):
    batch = Dim("batch")
    ep = torch.export.export(model, (x[:2],),
                             dynamic_shapes=({0: batch},))
    ops = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
    print(f"exported graph: {len(ops)} operations:",
          ", ".join(o.split(".")[1] for o in ops))
    torch.export.save(ep, OUT / "model.pt2")
    loaded = torch.export.load(OUT / "model.pt2").module()
    with torch.no_grad():
        print("loaded .pt2 matches:", torch.allclose(loaded(x), model(x)))


# %% [4] ONNX: run the same model without PyTorch.
def export_onnx(model, x) -> ort.InferenceSession:
    program = torch.onnx.export(model, (x[:2],), dynamo=True,
                                dynamic_shapes=({0: Dim("batch")},),
                                verbose=False)
    program.save(str(OUT / "model.onnx"))
    onnx.checker.check_model(onnx.load(OUT / "model.onnx"))
    session = ort.InferenceSession(OUT / "model.onnx")
    name = session.get_inputs()[0].name
    got = session.run(None, {name: x.numpy()})[0]
    with torch.no_grad():
        diff = np.abs(got - model(x).numpy()).max()
    kb = (OUT / "model.onnx").stat().st_size / 1e3
    print(f"model.onnx: {kb:.0f} KB, input '{name}',"
          f" max difference vs PyTorch {diff:.1e}")
    return session


# %% [5] A web API with FastAPI: JSON in, prediction out.
class Picture(BaseModel):
    pixels: list[float] = Field(min_length=784, max_length=784)


def make_app(session: ort.InferenceSession) -> FastAPI:
    app = FastAPI()
    name = session.get_inputs()[0].name

    @app.post("/predict")
    def predict(picture: Picture) -> dict:
        x = np.asarray(picture.pixels, np.float32).reshape(1, 1, 28, 28)
        logits = session.run(None, {name: x})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top = int(probs.argmax())
        return {"class": CLASSES[top],
                "probability": round(float(probs[top]), 3)}

    return app


def latency(fn, runs: int = 200) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs * 1e6


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model, x, y, acc = train_quick(epochs=1 if QUICK else 3)
    print(f"test accuracy on 2,000 pictures: {acc:.3f}")
    model = save_and_load(model, x)
    export_graph(model, x)
    session = export_onnx(model, x)
    one = x[:1]
    name = session.get_inputs()[0].name
    with torch.no_grad():
        t_torch = latency(lambda: model(one))
    t_ort = latency(lambda: session.run(None, {name: one.numpy()}))
    print(f"one picture: PyTorch {t_torch:.0f} us,"
          f" ONNX Runtime {t_ort:.0f} us")

    client = TestClient(make_app(session))
    pixels = x[0].flatten().tolist()
    r = client.post("/predict", json={"pixels": pixels})
    print("POST /predict ->", r.status_code, r.json(),
          " true:", CLASSES[int(y[0])])
    r = client.post("/predict", json={"pixels": pixels[:100]})
    print("100 pixels instead of 784 ->", r.status_code)
