# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""Convolutions on the GPU, nuvec against MLX, float16 and float32.

Runs `cargo bench -p nuvec-core --bench conv` and the same cases in MLX on its default
(GPU) stream: milliseconds per convolution, ten independent convolutions per evaluation,
the best of ten timed evaluations after a warm-up. Prints both and the nuvec/MLX ratio,
best of two alternating rounds.

Usage: `uv run scripts/bench_conv.py [filter]`.
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

CASES = {
    "patch16": ((1, 224, 224, 3), (768, 16, 16, 3), lambda x, w: mx.conv2d(x, w, stride=16)),
    "resnet3x3": ((8, 56, 56, 64), (64, 3, 3, 64), lambda x, w: mx.conv2d(x, w, padding=1)),
    "resnet3x3-c128": ((8, 28, 28, 128), (128, 3, 3, 128), lambda x, w: mx.conv2d(x, w, padding=1)),
    "resnet3x3-c256": ((8, 14, 14, 256), (256, 3, 3, 256), lambda x, w: mx.conv2d(x, w, padding=1)),
    "resnet3x3-c512": ((8, 7, 7, 512), (512, 3, 3, 512), lambda x, w: mx.conv2d(x, w, padding=1)),
    "unet3x3-c256": ((1, 64, 64, 256), (256, 3, 3, 256), lambda x, w: mx.conv2d(x, w, padding=1)),
    "stem7x7-s2": ((8, 224, 224, 3), (64, 7, 7, 3), lambda x, w: mx.conv2d(x, w, stride=2, padding=3)),
    "unaligned3x3": ((8, 56, 56, 24), (32, 3, 3, 24), lambda x, w: mx.conv2d(x, w, padding=1)),
    "resnet3x3-s2": ((8, 56, 56, 64), (128, 3, 3, 64), lambda x, w: mx.conv2d(x, w, stride=2, padding=1)),
    "conv1x1": ((8, 56, 56, 256), (64, 1, 1, 256), lambda x, w: mx.conv2d(x, w)),
    "depthwise3x3": ((8, 56, 56, 128), (128, 3, 3, 1), lambda x, w: mx.conv2d(x, w, padding=1, groups=128)),
    "depthwise7x7": ((8, 56, 56, 96), (96, 7, 7, 1), lambda x, w: mx.conv2d(x, w, padding=3, groups=96)),
    "transpose4x4-s2": ((4, 32, 32, 64), (32, 4, 4, 64),
                        lambda x, w: mx.conv_transpose2d(x, w, stride=2, padding=1)),
    "conv3d": ((1, 8, 32, 32, 32), (32, 3, 3, 3, 32), lambda x, w: mx.conv3d(x, w, padding=1)),
}


def mlx_results() -> dict[tuple[str, str], float]:
    out = {}
    for name, (xs, ws, f) in CASES.items():
        if FILTER not in name:
            continue
        for dtype, label in [(mx.float16, "float16"), (mx.float32, "float32")]:
            x = mx.random.normal(xs).astype(dtype)
            w = (0.1 * mx.random.normal(ws)).astype(dtype)
            batch = lambda: [f(x, w) for _ in range(REPS)]  # noqa: E731
            mx.eval(x, w, batch())
            best = float("inf")
            for _ in range(10):
                t = time.perf_counter()
                mx.eval(batch())
                best = min(best, time.perf_counter() - t)
            out[(name, label)] = best * 1e3 / REPS
    return out


def nuvec_results() -> dict[tuple[str, str], float]:
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "conv", "--", FILTER],
                          cwd=ROOT, capture_output=True, text=True, check=True, env=os.environ)
    out = {}
    for line in proc.stdout.splitlines():
        name, dtype, ms = line.split("\t")
        out[(name, dtype)] = float(ms)
    return out


def main() -> None:
    runs = [(nuvec_results(), mlx_results()) for _ in range(2)]
    best = lambda side, key: min(r[side][key] for r in runs)  # noqa: E731
    print(f"convolutions on the GPU, {platform.machine()} {platform.mac_ver()[0]}; MLX {mx.__version__}; "
          "ms per convolution, 10 per evaluation, best of 10 x 2 alternating rounds")
    print("| case | dtype | nuvec ms | MLX ms | nuvec / MLX |")
    print("|---|---|---:|---:|---:|")
    for key in runs[0][0]:
        o, t = best(0, key), best(1, key)
        print(f"| {key[0]} | {key[1]} | {o:.3f} | {t:.3f} | {o / t:.2f}× |")


if __name__ == "__main__":
    main()
