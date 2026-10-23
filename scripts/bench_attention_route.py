# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""Attention under a score-memory cap at mrun's chunk shapes: the measurements behind
nuvec's capped routing rule (`capped_tiled` in the attention planner) and its uncapped
head-size-256 rule.

Runs `cargo bench -p nuvec-core --bench attention_route` three times (uncapped routing,
and under a 512 MiB score cap with the tiled kernel or the composition in query blocks
forced) and MLX's `mx.fast.scaled_dot_product_attention` (uncapped; skipped where its
composition would materialise more than 4 GiB of scores) at the same causal shapes, then
prints milliseconds per call, the faster capped variant per cell and its margin.

Usage: `just bench-attention-route [filter]` (cells whose `d-lq-lk-dtype` contains it).
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
DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def nuvec(variant: str) -> dict[tuple, tuple[float, str]]:
    env = {**os.environ, "BENCH_VARIANT": variant, "BENCH_FILTER": FILTER}
    if variant != "uncapped":
        env["NUVEC_ATTENTION_CAPPED_KERNEL"] = variant
    proc = subprocess.run(
        ["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "attention_route"],
        cwd=ROOT, capture_output=True, text=True, check=True, env=env)
    out = {}
    for line in proc.stdout.splitlines():
        _, d, lq, lk, dt, ms, plan = line.split("\t")
        out[(int(d), int(lq), int(lk), dt)] = (float(ms), plan)
    return out


def mlx(cells) -> dict[tuple, float]:
    mx.set_default_device(mx.gpu)
    heads = {256: (16, 2), 128: (32, 8)}
    out = {}
    for (d, lq, lk, dt) in cells:
        hq, hkv = heads[d]
        size = {"float32": 4}.get(dt, 2)
        if hq * lq * lk * size > 4 << 30:
            continue
        q = (mx.zeros((1, hq, lq, d)) + 0.01).astype(DTYPES[dt])
        k = (mx.zeros((1, hkv, lk, d)) + 0.02).astype(DTYPES[dt])
        mx.eval(q, k)

        def run() -> float:
            t = time.perf_counter()
            mx.eval(mx.fast.scaled_dot_product_attention(q, k, k, scale=0.0625, mask="causal"))
            return (time.perf_counter() - t) * 1e3

        run()
        reps = 3 if lq * lk > 1 << 26 else 7
        times = sorted(run() for _ in range(reps))
        out[(d, lq, lk, dt)] = times[len(times) // 2]
        del q, k
        mx.clear_cache()
    return out


def main() -> None:
    uncapped = nuvec("uncapped")
    tiled = nuvec("tiled")
    blocks = nuvec("blocks")
    ref = mlx(sorted(tiled))
    print(f"{'d':>4} {'lq':>5} {'lk':>7} {'dtype':9} {'uncapped':>9} {'tiled':>9} {'blocks':>9}"
          f" {'mlx':>9}  capped winner (margin)  blocks plan")
    for cell in sorted(tiled):
        t, b = tiled[cell][0], blocks.get(cell, (float("nan"), ""))[0]
        u = uncapped.get(cell, (float("nan"), ""))[0]
        m = ref.get(cell, float("nan"))
        win = "tiled" if t <= b else "blocks"
        margin = max(t, b) / min(t, b)
        d, lq, lk, dt = cell
        print(f"{d:>4} {lq:>5} {lk:>7} {dt:9} {u:9.3f} {t:9.3f} {b:9.3f} {m:9.3f}  "
              f"{win:6} ({margin:.2f}x)  {blocks.get(cell, (0, ''))[1]}")


if __name__ == "__main__":
    main()
