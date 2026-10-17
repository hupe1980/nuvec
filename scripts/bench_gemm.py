# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""GPU GEMM shape matrix: nuvec vs MLX on the same machine.

Runs `cargo bench -p nuvec-core --bench gemm` and the same shapes in MLX, with the same
method (sustained throughput: products submitted back to back with `async_eval` after a
warm-up, best of three bursts of about 0.1 s), then prints TFLOP/s, the nuvec/MLX time
ratio and its geometric mean per dtype.

Usage: `just bench-gemm`.
"""

from __future__ import annotations

import math
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
SHAPES = [
    ("sq256", 1, 256, 256, 256, False),
    ("sq512", 1, 512, 512, 512, False),
    ("sq1024", 1, 1024, 1024, 1024, False),
    ("sq2048", 1, 2048, 2048, 2048, False),
    ("sq4096", 1, 4096, 4096, 4096, False),
    ("nt2048", 1, 2048, 2048, 2048, True),
    ("skinny16", 1, 16, 4096, 4096, True),
    ("tallk", 1, 256, 8192, 256, False),
    ("wide", 1, 4096, 512, 4096, False),
    ("batch8", 8, 512, 512, 512, False),
    ("mnist", 1, 256, 784, 32, True),
    ("vec4096", 1, 1, 4096, 4096, True),
    ("vecup", 1, 1, 4096, 11008, True),
    ("vecdown", 1, 1, 11008, 4096, True),
    ("matvec", 1, 4096, 4096, 1, False),
]
DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def mlx_results() -> dict[tuple[str, str], float]:
    mx.set_default_device(mx.gpu)
    out = {}
    for label, dtype in DTYPES.items():
        for name, batch, m, k, n, bt in SHAPES:
            a_shape = (batch, m, k) if batch > 1 else (m, k)
            b_shape = (batch, k, n) if batch > 1 else ((n, k) if bt else (k, n))
            a = mx.random.uniform(-1, 1, a_shape).astype(dtype)
            b = mx.random.uniform(-1, 1, b_shape).astype(dtype)
            mx.eval(a, b)
            if bt and batch == 1:
                b = b.T
            flops = 2 * batch * m * n * k
            reps = min(max(2_000_000_000_000 // 10 // max(flops, 1), 20), 2000)

            def burst(count: int) -> float:
                t = time.perf_counter()
                last = None
                for _ in range(count):
                    last = a @ b
                    mx.async_eval(last)
                mx.eval(last)
                return (time.perf_counter() - t) / count

            burst(reps)  # warm-up: GPU clocks ramp up
            out[(name, label)] = min(burst(reps) for _ in range(3)) * 1e6
    return out


def nuvec_results() -> dict[tuple[str, str], float]:
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "gemm"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        name, label, us = line.split("\t")
        out[(name, label)] = float(us)
    return out


def main() -> None:
    # Throughput drifts over minutes (thermals, background load): alternate the two
    # programs twice and keep each side's best time per case.
    runs = [(nuvec_results(), mlx_results()) for _ in range(2)]
    ours = {key: min(r[0][key] for r in runs) for key in runs[0][0]}
    theirs = {key: min(r[1][key] for r in runs) for key in runs[0][1]}
    print(f"GPU GEMM, {platform.machine()} {platform.mac_ver()[0]}; MLX {mx.__version__}; "
          "sustained throughput (async_eval bursts; best of 3 bursts x 2 alternating runs)")
    print("| shape | dtype | nuvec TFLOP/s | MLX TFLOP/s | nuvec / MLX time |")
    print("|---|---|---:|---:|---:|")
    for label in DTYPES:
        ratios = []
        for name, batch, m, k, n, _ in SHAPES:
            flops = 2 * batch * m * n * k
            o, t = ours[(name, label)], theirs[(name, label)]
            ratios.append(o / t)
            print(f"| {name} | {label} | {flops / o / 1e6:.2f} | {flops / t / 1e6:.2f} | {o / t:.2f}× |")
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print(f"| **geomean** | {label} | | | **{gm:.2f}×** |")


if __name__ == "__main__":
    main()
