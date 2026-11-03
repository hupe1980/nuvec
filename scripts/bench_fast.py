# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""GPU fused operations (normalisation, RoPE, softmax, log-sum-exp, cross entropy): nuvec
vs MLX on the same machine.

Runs `cargo bench -p nuvec-core --bench fast` and the same cases in MLX with the same
method (operations submitted back to back with `async_eval` after a warm-up, best of three
bursts, one or eight operations per `async_eval`), alternating the two programs twice,
then prints microseconds per operation, the nuvec/MLX time ratio and its geometric mean.

Usage: `just bench-fast [filter]` (cases whose name contains `filter`; `a|b` for either).
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
# (name, shape), as in crates/nuvec-core/benches/fast.rs
CASES = [
    ("rms_norm-decode", (1, 4096)),
    ("rms_norm-e4b", (1, 2560)),
    ("rms_norm-prefill", (2048, 4096)),
    ("layer_norm-prefill", (2048, 4096)),
    ("rope-decode", (1, 32, 1, 128)),
    ("rope-prefill", (1, 32, 2048, 128)),
    ("softmax-attn", (32, 512, 512)),
    ("logsumexp-vocab", (64, 32000)),
    ("cross_entropy-vocab", (64, 32000)),
    ("cross_entropy-mnist", (256, 10)),
    ("sdpa-decode512", (1, 32, 8, 1, 512, 128)),
    ("sdpa-decode4096", (1, 32, 8, 1, 4096, 128)),
    ("sdpa-decode32k", (1, 32, 8, 1, 32768, 128)),
    ("sdpa-prefill", (1, 32, 8, 1024, 1024, 128)),
    ("sdpa-prefill64", (1, 16, 16, 2048, 2048, 64)),
    ("sdpa-chunk2048", (1, 16, 8, 2048, 2048, 128)),
    ("sdpa-chunk1793", (1, 16, 8, 1793, 3841, 128)),
    ("sdpa-chunk-q36-32k", (1, 16, 2, 2048, 32768, 256)),
    ("sdpa-verify2-1k", (1, 32, 8, 2, 1025, 128)),
    ("sdpa-verify4-4k", (1, 32, 8, 4, 4096, 128)),
    ("sdpa-verify8-4k", (1, 32, 8, 8, 4096, 128)),
    ("sdpa-verify4-q36", (1, 16, 2, 4, 8194, 256)),
    ("sdpa-verify8-q36", (1, 16, 2, 8, 32768, 256)),
    # Sliding windows (`-w<keys>`): gpt-oss (D 64, W 128) and Gemma 4 sliding layers
    # (D 256, W 1024), MLX with the equivalent boolean mask (as mlx-lm builds it).
    ("sdpa-wdecode-w128", (1, 64, 8, 1, 4096, 64)),
    ("sdpa-wprefill64-w128", (1, 64, 8, 2048, 2175, 64)),
    ("sdpa-wdecode256-w1024", (1, 16, 8, 1, 8192, 256)),
    ("sdpa-wprefill256-w1024", (1, 16, 8, 2048, 3071, 256)),
    # Decode over exactly the window's keys: MLX without a mask, as mlx-lm's rotating cache
    # calls it (the baseline); nuvec with its window option.
    ("sdpa-wsdecode-w128", (1, 64, 8, 1, 128, 64)),
    ("sdpa-wsdecode256-w1024", (1, 16, 8, 1, 1024, 256)),
    # The same without a window (plain causal decode at head size 256).
    ("sdpa-decode256-1k", (1, 16, 8, 1, 1024, 256)),
    # Gemma 4 global layers (D 512): decode, and MLX's blocked causal prefill.
    ("sdpa-d512-decode4k", (1, 32, 4, 1, 4096, 512)),
    ("sdpa-d512-decode32k", (1, 32, 4, 1, 32768, 512)),
    ("sdpa-d512-prefill", (1, 16, 2, 2048, 2048, 512)),
    # Gemma 4 E4B's decode: sliding layers (D 256, 512 keys) and global layers (D 512 with
    # 4 query heads per key head: the composition in both engines).
    ("sdpa-e4b-d256-512", (1, 8, 2, 1, 512, 256)),
    ("sdpa-e4b-d512-1k", (1, 8, 2, 1, 1024, 512)),
    ("sdpa-e4b-d512-4k", (1, 8, 2, 1, 4096, 512)),
]
# The boolean window masks, built once per shape (mlx-lm's `create_causal_mask`).
WINDOWS: dict[tuple[int, int, int], mx.array] = {}


def window_mask(lq: int, lk: int, window: int) -> mx.array:
    pos = mx.arange(lk - lq, lk)[:, None]
    keys = mx.arange(lk)[None]
    return (pos >= keys) & (pos < keys + window)


DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""
CASES = [c for c in CASES if any(f in c[0] for f in FILTER.split("|"))]


def op(name: str, x: mx.array, w: mx.array, targets: mx.array) -> mx.array:
    kind = name.split("-")[0]
    if kind == "sdpa":
        mask = "causal" if x.shape[2] > 1 else None
        if "-w" in name and not name.startswith("sdpa-ws"):
            window = int(name.rsplit("-w", 1)[1])
            mask = WINDOWS[(x.shape[2], w.shape[2], window)]
        return mx.fast.scaled_dot_product_attention(x, w, w, scale=0.088, mask=mask)
    if kind == "rms_norm":
        return mx.fast.rms_norm(x, w, 1e-5)
    if kind == "layer_norm":
        return mx.fast.layer_norm(x, w, w, 1e-5)
    if kind == "rope":
        return mx.fast.rope(x, 128, traditional=False, base=10000.0, scale=1.0, offset=7)
    if kind == "softmax":
        return mx.softmax(x, axis=-1, precise=True)
    if kind == "logsumexp":
        return mx.logsumexp(x, axis=-1)
    return mx.fast.cross_entropy(x, targets)


def mlx_results() -> dict[tuple[str, str], tuple[float, float]]:
    mx.set_default_device(mx.gpu)
    out = {}
    for label, dtype in DTYPES.items():
        for name, shape in CASES:
            if name.startswith("sdpa"):
                b, hq, hkv, lq, lk, d = shape
                x = mx.random.uniform(-1, 1, (b, hq, lq, d)).astype(dtype)
                w = mx.random.uniform(-1, 1, (b, hkv, lk, d)).astype(dtype)
                shape = (b * hq * lq, d)
                if "-w" in name and not name.startswith("sdpa-ws"):
                    window = int(name.rsplit("-w", 1)[1])
                    WINDOWS[(lq, lk, window)] = window_mask(lq, lk, window)
                    mx.eval(WINDOWS[(lq, lk, window)])
            else:
                x = mx.random.uniform(-1, 1, shape).astype(dtype)
                w = mx.random.uniform(-1, 1, shape[-1:]).astype(dtype)
            rows = shape[:-1]
            n = math.prod(rows)
            targets = mx.array([i * 7919 % shape[-1] for i in range(n)], dtype=mx.int32)
            targets = targets.reshape(rows)
            mx.eval(x, w, targets)
            reps = min(max(200_000_000 // math.prod(shape), 20), 4000)

            def burst(count: int, per_eval: int) -> float:
                t = time.perf_counter()
                pending = []
                for _ in range(count):
                    pending.append(op(name, x, w, targets))
                    if len(pending) == per_eval:
                        mx.async_eval(pending)
                        pending = []
                mx.eval(pending)
                mx.eval(op(name, x, w, targets))
                return (time.perf_counter() - t) / count

            def best(per_eval: int) -> float:
                burst(reps, per_eval)  # warm-up: GPU clocks ramp up
                return min(burst(reps, per_eval) for _ in range(3)) * 1e6

            out[(name, label)] = (best(1), best(8))
    return out


def nuvec_results() -> dict[tuple[str, str], tuple[float, float]]:
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "fast"],
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
    print(f"GPU fused operations, {platform.machine()} {platform.mac_ver()[0]}; "
          f"MLX {mx.__version__}; sustained throughput (best of 3 bursts x 2 alternating runs); "
          "µs per operation with 1 and 8 operations per async_eval")
    print("| case | shape | dtype | nuvec µs (1 / 8) | MLX µs (1 / 8) | nuvec / MLX (1 / 8) |")
    print("|---|---|---|---:|---:|---:|")
    for label in DTYPES:
        ratios: list[list[float]] = [[], []]
        for name, shape in CASES:
            o = [best(0, (name, label), i) for i in range(2)]
            t = [best(1, (name, label), i) for i in range(2)]
            r = [o[i] / t[i] for i in range(2)]
            ratios[0].append(r[0])
            ratios[1].append(r[1])
            dims = "×".join(map(str, shape))
            print(f"| {name} | {dims} | {label} | {o[0]:.1f} / {o[1]:.1f} | "
                  f"{t[0]:.1f} / {t[1]:.1f} | {r[0]:.2f}× / {r[1]:.2f}× |")
        gm = [math.exp(sum(math.log(x) for x in rs) / len(rs)) for rs in ratios]
        print(f"| **geomean** | | {label} | | | **{gm[0]:.2f}× / {gm[1]:.2f}×** |")


if __name__ == "__main__":
    main()
