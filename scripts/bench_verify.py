# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5"]
# ///
"""Verify attention (speculative decoding): 1, 2, 4 and 8 causal query rows over a key
cache, nuvec against MLX on the same machine. Prints microseconds per call (32 calls
evaluated together, median of 9) and each row count's time against one row.

Usage: `just bench-verify [filter]`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""
CALLS = 32
SHAPES = [(128, 32, 8), (256, 16, 2)]


def nuvec() -> dict[tuple[int, int, int], float]:
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "verify"],
                          cwd=ROOT, capture_output=True, text=True, check=True,
                          env={**os.environ, "BENCH_FILTER": FILTER})
    out = {}
    for line in proc.stdout.splitlines():
        d, _, lq, lk, us = line.split("\t")
        out[(int(d), int(lq), int(lk))] = float(us)
    return out


def mlx_time(d: int, hq: int, hkv: int, lq: int, lk: int) -> float:
    k = mx.zeros((1, hkv, lk, d), dtype=mx.bfloat16) + 0.02
    v = mx.zeros((1, hkv, lk, d), dtype=mx.bfloat16) + 0.03
    qs = [mx.zeros((1, hq, lq, d), dtype=mx.bfloat16) + (0.01 + i * 1e-3) for i in range(CALLS)]
    mx.eval(k, v, *qs)
    mask = "causal" if lq > 1 else None

    def run() -> float:
        t = time.perf_counter()
        mx.eval([mx.fast.scaled_dot_product_attention(q, k, v, scale=0.088, mask=mask) for q in qs])
        return (time.perf_counter() - t) * 1e6 / CALLS

    run()
    return sorted(run() for _ in range(9))[4]


def main() -> None:
    ours = nuvec()
    print("| head | gqa | keys | rows | nuvec µs | MLX µs | nuvec/MLX | nuvec rows/1 | MLX rows/1 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d, hq, hkv in SHAPES:
        for lk in (4096, 16384, 32768):
            base = {}
            for lq in (1, 2, 4, 8):
                if f"{d}-{lq}-{lk}" not in f"{d}-{lq}-{lk}" or (d, lq, lk) not in ours:
                    continue
                n, m = ours[(d, lq, lk)], mlx_time(d, hq, hkv, lq, lk)
                base.setdefault("n", n)
                base.setdefault("m", m)
                print(f"| {d} | {hq // hkv} | {lk} | {lq} | {n:.1f} | {m:.1f} | {n / m:.2f}× "
                      f"| {n / base['n']:.2f}× | {m / base['m']:.2f}× |")


if __name__ == "__main__":
    main()
