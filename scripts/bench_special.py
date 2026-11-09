# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""MLX's special products and transforms on the GPU, nuvec against MLX.

Runs `cargo bench -p nuvec-core --bench special` and the same cases in MLX on its default
(GPU) stream: milliseconds per call, ten independent calls per evaluation, the best of ten
timed evaluations after a warm-up; best of two alternating rounds.

Usage: `uv run scripts/bench_special.py [filter]`.
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
REPS = 10
FILTER = next((a for a in sys.argv[1:] if not a.startswith("-")), "")


def mlx_results() -> dict[str, float]:
    bf = mx.bfloat16
    n = lambda *s, dt=bf: mx.random.normal(s).astype(dt)  # noqa: E731
    h1, h2, h3 = n(256, 4096, dt=mx.float16), n(64, 16384, dt=mx.float16), n(256, 28 * 128, dt=mx.float16)
    e1, e2, e3 = n(64, 2048), n(2048, 2048), n(2048, 16)
    sa, sb = n(256, 4096), n(4096, 256)
    segments = mx.array([[i * 512, (i + 1) * 512] for i in range(8)], dtype=mx.int32)
    ma, mb = n(1024, 1024), n(1024, 1024)
    mask = mx.random.bernoulli(0.5, (32, 32))
    qx, qx1, qw = n(256, 4096), n(1, 4096), n(4096, 4096)
    wq, ws = mx.quantize(qw, mode="nvfp4")
    mx.eval(h1, h2, h3, e1, e2, e3, sa, sb, segments, ma, mb, mask, qx, qx1, wq, ws)
    cases = {
        "hadamard-4096": lambda: mx.hadamard_transform(h1),
        "hadamard-16384": lambda: mx.hadamard_transform(h2),
        "hadamard-28x128": lambda: mx.hadamard_transform(h3),
        "einsum-chain": lambda: mx.einsum("ij,jk,kl->il", e1, e2, e3),
        "segmented_mm": lambda: mx.segmented_mm(sa, sb, segments),
        "block_masked_mm": lambda: mx.block_masked_mm(ma, mb, 32, mask),
        "qqmm-256": lambda: mx.qqmm(qx, wq, ws, mode="nvfp4"),
        "qqmm-1": lambda: mx.qqmm(qx1, wq, ws, mode="nvfp4"),
    }
    out = {}
    for name, f in cases.items():
        if FILTER not in name:
            continue
        batch = lambda: [f() for _ in range(REPS)]  # noqa: E731
        mx.eval(batch())
        best = float("inf")
        for _ in range(10):
            t = time.perf_counter()
            mx.eval(batch())
            best = min(best, time.perf_counter() - t)
        out[name] = best * 1e3 / REPS
    return out


def nuvec_results() -> dict[str, float]:
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "special", "--", FILTER],
                          cwd=ROOT, capture_output=True, text=True, check=True, env=os.environ)
    return {name: float(ms) for name, ms in (line.split("\t") for line in proc.stdout.splitlines())}


def main() -> None:
    runs = [(nuvec_results(), mlx_results()) for _ in range(2)]
    best = lambda side, key: min(r[side][key] for r in runs)  # noqa: E731
    print(f"special products on the GPU, {platform.machine()} {platform.mac_ver()[0]}; MLX {mx.__version__}; "
          "ms per call, 10 per evaluation, best of 10 x 2 alternating rounds")
    print("| case | nuvec ms | MLX ms | nuvec / MLX |")
    print("|---|---:|---:|---:|")
    for key in runs[0][0]:
        o, t = best(0, key), best(1, key)
        print(f"| {key} | {o:.3f} | {t:.3f} | {o / t:.2f}× |")


if __name__ == "__main__":
    main()
