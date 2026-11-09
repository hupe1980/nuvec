# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""Decompositions on the CPU, nuvec (faer) against MLX (Accelerate's LAPACK), float32.

Runs `cargo bench -p nuvec-core --bench linalg` and the same cases in MLX on its CPU
stream (the best of five timed runs after a warm-up), then prints milliseconds and the
nuvec/MLX ratio.

Usage: `uv run scripts/bench_linalg.py [--accelerate]` (`--accelerate`: nuvec with Accelerate's
LAPACK, the feature `accelerate`).
"""

import os
import platform
import sys
import subprocess
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent


def mlx_results() -> dict[tuple[str, int], float]:
    out = {}
    L = mx.linalg
    for n in [64, 256, 1024]:
        mx.random.seed(n)
        a = mx.random.normal((n, n))
        spd = a @ a.T + n * mx.eye(n)
        b = mx.random.normal((n, 16))
        mx.eval(a, spd, b)
        cases = {
            "qr": lambda: L.qr(a, stream=mx.cpu),
            "svd": lambda: L.svd(a, stream=mx.cpu),
            "eigh": lambda: L.eigh(spd, stream=mx.cpu),
            "inv": lambda: L.inv(a, stream=mx.cpu),
            "solve": lambda: L.solve(a, b, stream=mx.cpu),
            "cholesky": lambda: L.cholesky(spd, stream=mx.cpu),
            "cholesky_upper": lambda: L.cholesky(spd, upper=True, stream=mx.cpu),
            "lu": lambda: L.lu_factor(a, stream=mx.cpu),
        }
        for name, f in cases.items():
            mx.eval(f())
            best = float("inf")
            for _ in range(5):
                t = time.perf_counter()
                mx.eval(f())
                best = min(best, time.perf_counter() - t)
            out[(name, n)] = best * 1e3
    return out


def nuvec_results() -> dict[tuple[str, int], float]:
    features = ["--features", "accelerate"] if "--accelerate" in sys.argv else []
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "linalg", *features],
                          cwd=ROOT, capture_output=True, text=True, check=True, env=os.environ)
    out = {}
    for line in proc.stdout.splitlines():
        name, n, ms = line.split("\t")
        out[(name, int(n))] = float(ms)
    return out


def main() -> None:
    runs = [(nuvec_results(), mlx_results()) for _ in range(2)]
    best = lambda side, key: min(r[side][key] for r in runs)  # noqa: E731
    print(f"linalg on the CPU, {platform.machine()} {platform.mac_ver()[0]}; MLX {mx.__version__}; "
          "best of 5 runs x 2 alternating rounds; float32; nuvec "
          + ("with Accelerate" if "--accelerate" in sys.argv else "pure Rust (faer)"))
    print("| op | n | nuvec ms | MLX ms | nuvec / MLX |")
    print("|---|---:|---:|---:|---:|")
    for key in sorted(runs[0][0], key=lambda k: (k[1], k[0])):
        o, t = best(0, key), best(1, key)
        print(f"| {key[0]} | {key[1]} | {o:.3f} | {t:.3f} | {o / t:.2f}× |")


if __name__ == "__main__":
    main()
