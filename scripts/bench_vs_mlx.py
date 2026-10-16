# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "numpy==2.5.4"]
# ///
"""Compares nuvec and MLX on the same machine.

Times identical workloads in MLX (CPU device, pinned version), runs
`cargo bench -p nuvec-core --bench vs_mlx`, and prints a markdown table with median
times and the nuvec/MLX ratio (lower is better for nuvec).

Usage: `just bench-mlx` (or `uv run scripts/bench_vs_mlx.py`).
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent


def median_us(samples: list[float]) -> float:
    return sorted(samples)[len(samples) // 2] * 1e6


def bench(reps: int, run) -> float:
    for _ in range(3):
        run()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return median_us(samples)


def mlx_results(device=mx.cpu, prefix: str = "") -> dict[str, float]:
    mx.set_default_device(device)
    out: dict[str, float] = {}
    one = mx.array(1.0, dtype=mx.float32)

    def dispatch() -> None:
        x = mx.array(0.0, dtype=mx.float32)
        for _ in range(10_000):
            x = x + one
        mx.eval(x)

    out[prefix + "dispatch_10k"] = bench(20 if not prefix else 10, dispatch)
    a = mx.arange(0, 1e6, 1, dtype=mx.float32)
    b = mx.ones((1_000_000,), dtype=mx.float32)
    mx.eval(a, b)
    out[prefix + "add_1m"] = bench(50, lambda: mx.eval(a + b))
    out[prefix + "sum_1m"] = bench(50, lambda: mx.eval(mx.sum(a)))
    m, n = a.reshape(1000, 1000), b.reshape(1000, 1000)
    mx.eval(m, n)
    out[prefix + "add_strided_1m"] = bench(50, lambda: mx.eval(m.T + n))
    out[prefix + "sum_axis_1k"] = bench(50, lambda: mx.eval(mx.sum(m, axis=1)))
    for size in (256, 1024):
        x, y = mx.ones((size, size)), mx.ones((size, size))
        mx.eval(x, y)
        out[f"{prefix}matmul_{size}"] = bench(20 if prefix else (10 if size > 512 else 50), lambda: mx.eval(x @ y))
    return out


def nuvec_results() -> dict[str, float]:
    proc = subprocess.run(
        ["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "vs_mlx"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    results = {}
    for line in proc.stdout.splitlines():
        name, _, value = line.partition("\t")
        if value:
            results[name] = float(value)
    return results


def machine() -> str:
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = platform.processor()
    return f"{cpu}, {platform.system()} {platform.release()}"


def main() -> None:
    theirs = mlx_results()
    theirs.update(mlx_results(mx.gpu, "gpu_"))
    ours = nuvec_results()
    print(f"Machine: {machine()}")
    print(f"MLX {mx.__version__}; rows without a prefix use the CPU device, `gpu_` rows the GPU.")
    print("nuvec release build. Medians over repeated runs; MLX timings include Python overhead.\n")
    print("| workload | nuvec (µs) | MLX (µs) | nuvec / MLX |")
    print("|---|---:|---:|---:|")
    for name, mlx_us in theirs.items():
        ours_us = ours.get(name)
        if ours_us is None:
            print(f"| {name} | n/a | {mlx_us:.1f} | n/a |")
        else:
            print(f"| {name} | {ours_us:.1f} | {mlx_us:.1f} | {ours_us / mlx_us:.2f}× |")


if __name__ == "__main__":
    main()
