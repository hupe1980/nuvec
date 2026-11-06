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
    # Gemma 4 E4B's decode (one row): MLP, attention, per-layer input, output head.
    ("e4b-up", 1, 2560, 10240, True, 4, 64),
    ("e4b-down", 1, 10240, 2560, True, 4, 64),
    ("e4b-q", 1, 2560, 2048, True, 4, 64),
    ("e4b-kv", 1, 2560, 512, True, 4, 64),
    ("e4b-o", 1, 2048, 2560, True, 4, 64),
    ("e4b-pli", 1, 256, 2560, True, 4, 64),
    ("e4b-head", 1, 2560, 262144, True, 4, 64),
    # Gemma 4 26B-A4B's decode at one row (hidden 2,816: the generic `qmv`), weights in the
    # system cache, then out of it (N = 32,768, against an aligned K).
    ("g26-q1", 1, 2816, 4096, True, 4, 64),
    ("g26-mlp1", 1, 2816, 2112, True, 4, 64),
    ("g26-down1", 1, 2112, 2816, True, 4, 64),
    ("g26-dram1", 1, 2816, 32768, True, 4, 64),
    ("g26-dram1-k3072", 1, 3072, 32768, True, 4, 64),
    # At 8 requests: dense, then experts (`g26g`).
    ("g26-q", 8, 2816, 4096, True, 4, 64),
    ("g26-kv", 8, 2816, 2048, True, 4, 64),
    ("g26-o", 8, 4096, 2816, True, 4, 64),
    ("g26-mlp", 8, 2816, 2112, True, 4, 64),
    ("g26-down", 8, 2112, 2816, True, 4, 64),
    ("g26g-up", 64, 2816, 704, True, 4, 64),
    ("g26g-down", 64, 704, 2816, True, 4, 64),
    # Dense prefill (Qwen3-8B, 2,048 tokens).
    ("qmm-2048x4096", 2048, 4096, 4096, True, 4, 64),
    ("qmm-2048x12288", 2048, 4096, 12288, True, 4, 64),
    ("qmm-2048-down", 2048, 12288, 4096, True, 4, 64),
    # MoE prefill at 2,048-token chunks (Qwen3.6-35B-A3B): attention and linear-attention
    # projections, the shared expert, then the routed experts (16,384 rows, 8 per token).
    ("q36-8192", 2048, 2048, 8192, True, 4, 64),
    ("q36-4096", 2048, 2048, 4096, True, 4, 64),
    ("q36-out", 2048, 4096, 2048, True, 4, 64),
    ("q36-512", 2048, 2048, 512, True, 4, 64),
    ("q36-shared-down", 2048, 512, 2048, True, 4, 64),
    # Qwen3-0.6B prefill (mlx-lm's 2,047-row chunk): q, k/v, o, gate/up, down.
    ("q06-q", 2047, 1024, 2048, True, 4, 64),
    ("q06-kv", 2047, 1024, 1024, True, 4, 64),
    ("q06-o", 2047, 2048, 1024, True, 4, 64),
    ("q06-up", 2047, 1024, 3072, True, 4, 64),
    ("q06-down", 2047, 3072, 1024, True, 4, 64),
    ("q06n-q", 2047, 2048, 1024, False, 4, 64),
    ("q06n-kv", 2047, 1024, 1024, False, 4, 64),
    ("q06n-o", 2047, 1024, 2048, False, 4, 64),
    ("q06n-up", 2047, 3072, 1024, False, 4, 64),
    ("q06n-down", 2047, 1024, 3072, False, 4, 64),
    ("q06-head", 2047, 1024, 151936, True, 4, 64),
    ("q06n-head", 2047, 151936, 1024, False, 4, 64),
    # Qwen3-0.6B's prefill of a 25-token prompt (a 24-row chunk).
    ("q06s-q", 24, 1024, 2048, True, 4, 64),
    ("q06s-kv", 24, 1024, 1024, True, 4, 64),
    ("q06s-o", 24, 2048, 1024, True, 4, 64),
    ("q06s-up", 24, 1024, 3072, True, 4, 64),
    ("q06s-down", 24, 3072, 1024, True, 4, 64),
    ("gqmm-2048-up", 16384, 2048, 512, True, 4, 64),
    ("gqmm-2048-down", 16384, 512, 2048, True, 4, 64),
]
# Floating-point modes, as FP_SHAPES in the bench: (name, M, K, N, transpose) per mode,
# named `mode-name` (gpt-oss's dense shapes are `oss-*`, K = N = 2880).
FP_SHAPES = [
    ("qmv4096", 1, 4096, 4096, True),
    ("qmvup", 1, 4096, 11008, True),
    ("qmvdown", 1, 11008, 4096, True),
    ("qmv-m4", 4, 4096, 4096, True),
    ("qmv-m8", 8, 4096, 4096, True),
    ("qmm-m32", 32, 4096, 4096, True),
    ("qmm-m512", 512, 4096, 4096, True),
    ("qvm4096", 1, 4096, 4096, False),
    ("qmv-k128", 1, 128, 4096, True),
    ("oss-qmv2880", 1, 2880, 2880, True),
    ("oss-dram", 1, 2880, 32768, True),
    ("oss-dram-k3072", 1, 3072, 32768, True),
    ("oss-qmv2880-m4", 4, 2880, 2880, True),
    ("oss-qmm2880", 512, 2880, 2880, True),
]
# Floating-point gathers, as FP_GATHERS in the bench: (name, rows, K, N, experts) per mode,
# rows sorted by expert; nvfp4 also with per-expert global scales (`nvfp4-name-gs`).
FP_GATHERS = [
    ("gather-decode", 8, 2048, 768, 128),
    ("gather-sorted", 512, 2048, 768, 128),
    ("oss-gather-decode", 4, 2880, 2880, 32),
    ("oss-gather-sorted", 512, 2880, 2880, 32),
]
FP_MODES = {"mxfp4": (32, 4), "mxfp8": (32, 8), "nvfp4": (16, 4)}
EXPERTS = 256
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""
# (name, M, K, N, transpose, bits, group size, mode, experts, global scale), in the
# bench's order
SHAPES = [(*s, "affine", EXPERTS if s[0].startswith("gqmm")
           else 128 if s[0].startswith("g26g") else 0, False) for s in SHAPES]
for mode, (gs, bits) in FP_MODES.items():
    SHAPES += [(f"{mode}-{name}", m, k, n, t, bits, gs, mode, 0, False)
               for name, m, k, n, t in FP_SHAPES]
    for name, m, k, n, e in FP_GATHERS:
        SHAPES.append((f"{mode}-{name}", m, k, n, True, bits, gs, mode, e, False))
        if mode == "nvfp4":
            SHAPES.append((f"{mode}-{name}-gs", m, k, n, True, bits, gs, mode, e, True))
SHAPES = [s for s in SHAPES if FILTER in s[0]]
DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}


def mlx_results() -> dict[tuple[str, str], float]:
    mx.set_default_device(mx.gpu)
    out = {}
    for label, dtype in DTYPES.items():
        for name, m, k, n, transpose, bits, gs, mode, experts, global_scale in SHAPES:
            gathered = experts > 0
            shape = (experts, n, k) if gathered else (n, k) if transpose else (k, n)
            w = mx.random.uniform(-1, 1, shape).astype(dtype)
            q = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
            wq, s, b = (q[0], q[1], q[2] if mode == "affine" else None)
            x = mx.random.uniform(-1, 1, (m, 1, k) if gathered else (m, k)).astype(dtype)
            indices = mx.array([i * max(experts, 1) // m for i in range(m)], dtype=mx.uint32)
            scale = mx.random.uniform(0.5, 2.0, (max(experts, 1),))
            gs_arg = {"global_scale": scale} if global_scale else {}
            mx.eval(wq, s, x, indices, scale)
            nbytes = k * n * bits // 8
            reps = min(max(40_000_000_000 // 10 // max(nbytes + 2 * m * k * n, 1), 20), 4000)

            def product() -> mx.array:
                if gathered:
                    return mx.gather_qmm(x, wq, s, b, rhs_indices=indices, transpose=True,
                                         group_size=gs, bits=bits, mode=mode,
                                         sorted_indices=True, **gs_arg)
                return mx.quantized_matmul(x, wq, s, b, transpose=transpose, group_size=gs,
                                           bits=bits, mode=mode)

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
        # Geometric means per quantization mode.
        ratios: dict[str, list[list[float]]] = {}
        for name, m, k, n, _, _, _, mode, *_ in SHAPES:
            o = [best(0, (name, label), i) for i in range(2)]
            t = [best(1, (name, label), i) for i in range(2)]
            r = [o[i] / t[i] for i in range(2)]
            per = ratios.setdefault(mode, [[], []])
            per[0].append(r[0])
            per[1].append(r[1])
            print(f"| {name} | {m}×{k}×{n} | {label} | {o[0]:.1f} / {o[1]:.1f} | "
                  f"{t[0]:.1f} / {t[1]:.1f} | {r[0]:.2f}× / {r[1]:.2f}× |")
        for mode, rs in ratios.items():
            gm = [math.exp(sum(math.log(x) for x in r) / len(r)) for r in rs]
            print(f"| **geomean {mode}** | | {label} | | | **{gm[0]:.2f}× / {gm[1]:.2f}×** |")


if __name__ == "__main__":
    main()
