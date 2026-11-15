# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""Decode attention over a quantized KV cache: one query row over 4K–64K keys (head 128,
32 query and 8 key heads), bf16 vector attention against mlx-lm's quantized-cache attention
(`quantized_scaled_dot_product_attention`) at 8 and 4 bits, nuvec against MLX on the same
machine. Prints microseconds per call (32 calls evaluated together, median of 9) and each
variant's time against bf16.

Usage: `just bench-kv-attention`.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import quantized_scaled_dot_product_attention

ROOT = Path(__file__).resolve().parent.parent
CALLS, D, HQ, HKV = 32, 128, 32, 8
KEYS = (4096, 16384, 65536)


def nuvec() -> dict[tuple[str, int], float]:
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "kv_attention"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        name, lk, us = line.split("\t")
        out[(name, int(lk))] = float(us)
    return out


def mlx_times(lk: int) -> dict[str, float]:
    k = mx.random.normal((1, HKV, lk, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, lk, D)).astype(mx.bfloat16)
    qs = [mx.random.normal((1, HQ, 1, D)).astype(mx.bfloat16) for _ in range(CALLS)]
    scale = D ** -0.5
    caches = {bits: (mx.quantize(k, 64, bits), mx.quantize(v, 64, bits)) for bits in (8, 4)}
    mx.eval(k, v, *qs, caches)
    variants = {
        "bf16": lambda q: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale),
        "q8": lambda q: quantized_scaled_dot_product_attention(
            q, *caches[8], scale=scale, mask=None, group_size=64, bits=8),
        "q4": lambda q: quantized_scaled_dot_product_attention(
            q, *caches[4], scale=scale, mask=None, group_size=64, bits=4),
    }
    out = {}
    for name, f in variants.items():
        def run() -> float:
            t = time.perf_counter()
            mx.eval([f(q) for q in qs])
            return (time.perf_counter() - t) * 1e6 / CALLS
        run()
        out[name] = sorted(run() for _ in range(9))[4]
    return out


def main() -> None:
    ours = nuvec()
    print("| keys | variant | nuvec µs | MLX µs | nuvec/MLX | nuvec / bf16 | MLX / bf16 |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for lk in KEYS:
        m = mlx_times(lk)
        for name in ("bf16", "q8", "q4"):
            n = ours[(name, lk)]
            print(f"| {lk} | {name} | {n:.1f} | {m[name]:.1f} | {n / m[name]:.2f}× "
                  f"| {n / ours[('bf16', lk)]:.2f} | {m[name] / m['bf16']:.2f} |")


if __name__ == "__main__":
    main()
