# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "numpy==2.5.4"]
# ///
"""Trains MLX's MNIST example and nuvec's port on the same machine and compares them.

Downloads MNIST into `target/mnist` (once), runs the MLX example model (MLP 784-32-32-10,
ReLU, SGD lr 0.1, batch 256) uncompiled and with `mx.compile`, then runs
`cargo run --release -p nuvec-nn --example mnist`. Prints test accuracy after the last
epoch and the median epoch time (epochs after the first).

Usage: `just mnist` (or `uv run scripts/mnist_vs_mlx.py [--epochs N]`).
"""

from __future__ import annotations

import argparse
import gzip
import re
import statistics
import subprocess
import time
import urllib.request
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "target" / "mnist"
MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]


def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DATA / name
        if not target.exists():
            with urllib.request.urlopen(MIRROR + name + ".gz") as r:
                target.write_bytes(gzip.decompress(r.read()))


def read_idx(name: str) -> np.ndarray:
    raw = (DATA / name).read_bytes()
    ndim = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8).reshape(shape)


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [nn.Linear(i, o) for i, o in zip(sizes[:-1], sizes[1:])]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)


def train_mlx(compiled: bool, epochs: int, asynchronous: bool = False) -> tuple[float, float]:
    mx.set_default_device(mx.gpu)
    mx.random.seed(0)
    np.random.seed(0)
    x = mx.array(read_idx(FILES[0]).reshape(-1, 784).astype(np.float32) / 255.0)
    y = mx.array(read_idx(FILES[1]).astype(np.uint32))
    tx = mx.array(read_idx(FILES[2]).reshape(-1, 784).astype(np.float32) / 255.0)
    ty = mx.array(read_idx(FILES[3]).astype(np.uint32))
    model = MLP(2, 784, 32, 10)
    mx.eval(model.parameters())
    optimizer = optim.SGD(learning_rate=0.1)
    loss_and_grad = nn.value_and_grad(
        model, lambda m, a, b: nn.losses.cross_entropy(m(a), b, reduction="mean"))

    def step(a, b):
        loss, grads = loss_and_grad(model, a, b)
        optimizer.update(model, grads)
        return loss

    if compiled:
        step = partial(mx.compile, inputs=model.state, outputs=model.state)(step)
    times, accuracy = [], 0.0
    for _ in range(epochs):
        tic = time.perf_counter()
        perm = mx.array(np.random.permutation(y.size))
        for s in range(0, y.size, 256):
            ids = perm[s: s + 256]
            step(x[ids], y[ids])
            (mx.async_eval if asynchronous else mx.eval)(model.state)
        accuracy = mx.mean(mx.argmax(model(tx), axis=1) == ty).item()
        times.append(time.perf_counter() - tic)
    return accuracy, statistics.median(times[1:] or times)


def train_nuvec(epochs: int, asynchronous: bool = False) -> tuple[float, float]:
    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "nuvec-nn", "--example", "mnist", "--",
         str(DATA), "--epochs", str(epochs)] + (["--async"] if asynchronous else []),
        cwd=ROOT, check=True, capture_output=True, text=True).stdout
    m = re.search(r"accuracy=([\d.]+) epoch_seconds=([\d.]+)", out)
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    epochs = parser.parse_args().epochs
    download()
    rows = [
        ("nuvec (GPU, eval per step)", *train_nuvec(epochs)),
        ("MLX (GPU, uncompiled, eval per step)", *train_mlx(False, epochs)),
        ("nuvec (GPU, async_eval per step)", *train_nuvec(epochs, True)),
        ("MLX (GPU, uncompiled, async_eval per step)", *train_mlx(False, epochs, True)),
        ("MLX (GPU, compiled, eval per step)", *train_mlx(True, epochs)),
    ]
    print(f"MNIST MLP 784-32-32-10, SGD lr 0.1, batch 256, {epochs} epochs; MLX {mx.__version__}")
    print("| run | test accuracy | median epoch (s) |\n|---|---:|---:|")
    for name, acc, sec in rows:
        print(f"| {name} | {acc:.4f} | {sec:.3f} |")


if __name__ == "__main__":
    main()
