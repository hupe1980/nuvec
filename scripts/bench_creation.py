# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""Creation ops on the GPU, nuvec against MLX: the creating call alone and creation plus
evaluation of zeros, ones and full from 4 KB to 256 MB, and 1,000 small creations added
to a GPU array. Runs nuvec's `creation` bench, then MLX's same loops, and prints both.

    uv run scripts/bench_creation.py
"""

import subprocess
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent


def median(v):
    v = sorted(v)
    return v[len(v) // 2]


def mlx_rows():
    rows = {}
    make = {"zeros": lambda n, d: mx.zeros((n,), d), "ones": lambda n, d: mx.ones((n,), d),
            "full": lambda n, d: mx.full((n,), 1.5, d)}
    for _ in range(300):  # warm-up: the GPU's clocks rise under load
        mx.eval(mx.ones((1 << 22,)))
    for dname, d in [("float32", mx.float32), ("bfloat16", mx.bfloat16)]:
        for op in ["zeros", "ones", "full"]:
            for b in [4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20]:
                n = b // d.size
                reps = 10 if b >= 16 << 20 else 100
                mx.eval(make[op](n, d))
                create, both = [], []
                for _ in range(reps):
                    t = time.perf_counter()
                    a = make[op](n, d)
                    create.append((time.perf_counter() - t) * 1e6)
                    mx.eval(a)
                    both.append((time.perf_counter() - t) * 1e6)
                    del a
                rows[(op, dname, b)] = (median(create), median(both))
    small = {}
    for label, shape in [("0-d", ()), ("16", (16,))]:
        x = mx.ones(shape)
        mx.eval(x)
        runs = []
        for _ in range(7):
            t = time.perf_counter()
            y = x
            for _ in range(1000):
                y = y + mx.zeros(shape)
            mx.eval(y)
            runs.append((time.perf_counter() - t) * 1e6)
        small[label] = median(runs)
    return rows, small


def nuvec_rows():
    out = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "creation"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    rows, small = {}, {}
    for line in out.splitlines():
        f = line.split("\t")
        if f[0] == "small":
            small[f[1]] = float(f[2])
        elif len(f) == 5:
            rows[(f[0], f[1], int(f[2]))] = (float(f[3]), float(f[4]))
    return rows, small


def main():
    nv, nvs = nuvec_rows()
    ml, mls = mlx_rows()
    print(f"creation on the GPU, MLX {mx.__version__}; µs (median)")
    print("| op | dtype | bytes | create nuvec / MLX | create+eval nuvec / MLX | ratio |")
    print("|---|---|---:|---:|---:|---:|")
    for k in nv:
        a, b = nv[k], ml[k]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {a[0]:.1f} / {b[0]:.1f} | {a[1]:.1f} / {b[1]:.1f} "
              f"| {a[1] / b[1]:.2f}× |")
    for k in nvs:
        print(f"| small loop | float32 | {k} | | {nvs[k]:.1f} / {mls[k]:.1f} | {nvs[k] / mls[k]:.2f}× |")


if __name__ == "__main__":
    main()
