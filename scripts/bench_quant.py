# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""GPU quantized matrix products: nuvec vs MLX on the same machine.

Runs `cargo bench -p nuvec-core --bench quant` and the same cases in MLX with the same
method (products submitted back to back with `async_eval` after a warm-up, best of three
bursts), alternating the two programs twice, then prints microseconds per product, the
nuvec/MLX time ratio and its geometric mean per dtype.

Each case is measured twice: one product per `async_eval` (a command buffer each, so the
per-submission cost counts) and eight per `async_eval` (closer to kernel time, as in a
model step with many operations per evaluation).

Usage: `just bench-quant`.
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
# (name, M, K, N, transpose, bits, group size), as in crates/nuvec-core/benches/quant.rs
SHAPES = [
    ("qmv4096", 1, 4096, 4096, True, 4, 64),
    ("qmvup", 1, 4096, 11008, True, 4, 64),
    ("qmvdown", 1, 11008, 4096, True, 4, 64),
    ("qmv4096-b8", 1, 4096, 4096, True, 8, 64),
    ("qmv4096-b3", 1, 4096, 4096, True, 3, 64),
    ("qmv4096-g32", 1, 4096, 4096, True, 4, 32),
    ("qmv-m4", 4, 4096, 4096, True, 4, 64),
    ("qmm-m32", 32, 4096, 4096, True, 4, 64),
    ("qmm-m512", 512, 4096, 4096, True, 4, 64),
    ("qvm4096", 1, 4096, 4096, False, 4, 64),
    ("qmm-n128", 128, 4096, 4096, False, 4, 64),
    ("qmm-961x8192", 961, 2048, 8192, True, 4, 64),
    ("qmm-961x2048", 961, 2048, 2048, True, 4, 64),
    ("gqmm-up", 7688, 2048, 512, True, 4, 64),
    ("gqmm-down", 7688, 512, 2048, True, 4, 64),
]
EXPERTS = 256
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""
SHAPES = [s for s in SHAPES if FILTER in s[0]]
DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}


def mlx_results() -> dict[tuple[str, str], float]:
    mx.set_default_device(mx.gpu)
    out = {}
    for label, dtype in DTYPES.items():
        for name, m, k, n, transpose, bits, gs in SHAPES:
            gathered = name.startswith("gqmm")
            shape = (EXPERTS, n, k) if gathered else (n, k) if transpose else (k, n)
            w = mx.random.uniform(-1, 1, shape).astype(dtype)
            wq, s, b = mx.quantize(w, group_size=gs, bits=bits)
            x = mx.random.uniform(-1, 1, (m, 1, k) if gathered else (m, k)).astype(dtype)
            experts = mx.array([i * EXPERTS // m for i in range(m)], dtype=mx.uint32)
            mx.eval(wq, s, b, x, experts)
            nbytes = k * n * bits // 8
            reps = min(max(40_000_000_000 // 10 // max(nbytes + 2 * m * k * n, 1), 20), 4000)

            def product() -> mx.array:
                if gathered:
                    return mx.gather_qmm(x, wq, s, b, rhs_indices=experts, transpose=True,
                                         group_size=gs, bits=bits, sorted_indices=True)
                return mx.quantized_matmul(x, wq, s, b, transpose=transpose, group_size=gs,
                                           bits=bits)

            def burst(count: int, per_eval: int) -> float:
                t = time.perf_counter()
                pending = []
                for _ in range(count):
                    pending.append(product())
                    if len(pending) == per_eval:
                        mx.async_eval(pending)
                        pending = []
                mx.eval(pending)
                mx.eval(product())
                return (time.perf_counter() - t) / count

            def best(per_eval: int) -> float:
                burst(reps, per_eval)  # warm-up: GPU clocks ramp up
                return min(burst(reps, per_eval) for _ in range(3)) * 1e6

            out[(name, label)] = (best(1), best(8))
    return out


def nuvec_results() -> dict[tuple[str, str], float]:
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "quant"],
                          cwd=ROOT, capture_output=True, text=True, check=True,
                          env={**os.environ, "BENCH_FILTER": FILTER})
    out = {}
    for line in proc.stdout.splitlines():
        name, label, single, grouped = line.split("\t")
        out[(name, label)] = (float(single), float(grouped))
    return out


def main() -> None:
    runs = [(nuvec_results(), mlx_results()) for _ in range(2)]
    best = lambda side, key, i: min(r[side][key][i] for r in runs)  # noqa: E731
    print(f"GPU quantized_matmul, {platform.machine()} {platform.mac_ver()[0]}; "
          f"MLX {mx.__version__}; sustained throughput (best of 3 bursts x 2 alternating runs); "
          "µs per product with 1 and 8 products per async_eval")
    print("| case | M×K×N | dtype | nuvec µs (1 / 8) | MLX µs (1 / 8) | nuvec / MLX (1 / 8) |")
    print("|---|---|---|---:|---:|---:|")
    for label in DTYPES:
        ratios: list[list[float]] = [[], []]
        for name, m, k, n, *_ in SHAPES:
            o = [best(0, (name, label), i) for i in range(2)]
            t = [best(1, (name, label), i) for i in range(2)]
            r = [o[i] / t[i] for i in range(2)]
            ratios[0].append(r[0])
            ratios[1].append(r[1])
            print(f"| {name} | {m}×{k}×{n} | {label} | {o[0]:.1f} / {o[1]:.1f} | "
                  f"{t[0]:.1f} / {t[1]:.1f} | {r[0]:.2f}× / {r[1]:.2f}× |")
        gm = [math.exp(sum(math.log(x) for x in rs) / len(rs)) for rs in ratios]
        print(f"| **geomean** | | {label} | | | **{gm[0]:.2f}× / {gm[1]:.2f}×** |")


if __name__ == "__main__":
    main()
