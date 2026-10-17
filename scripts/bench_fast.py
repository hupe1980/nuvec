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

Usage: `just bench-fast`.
"""

from __future__ import annotations

import math
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
# (name, shape), as in crates/nuvec-core/benches/fast.rs
CASES = [
    ("rms_norm-decode", (1, 4096)),
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
    ("sdpa-prefill", (1, 32, 8, 1024, 1024, 128)),
    ("sdpa-prefill64", (1, 16, 16, 2048, 2048, 64)),
]
DTYPES = {"float32": mx.float32, "float16": mx.float16}


def op(name: str, x: mx.array, w: mx.array, targets: mx.array) -> mx.array:
    kind = name.split("-")[0]
    if kind == "sdpa":
        mask = "causal" if x.shape[2] > 1 else None
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
                          cwd=ROOT, capture_output=True, text=True, check=True)
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
