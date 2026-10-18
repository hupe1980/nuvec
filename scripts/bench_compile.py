# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""Compiled element-wise functions (GELU, SwiGLU gating, an Adam step): nuvec vs MLX on
the same machine, uncompiled and compiled.

Runs `cargo bench -p nuvec-compile --bench compile` and the same cases in MLX with the
same method (one call per `async_eval`, sustained after a warm-up, best of three bursts),
alternating the two programs twice, then prints microseconds per call and the nuvec/MLX
time ratios with their geometric means.

Usage: `just bench-compile`.
"""

from __future__ import annotations

import math
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
SIZES = [1 << 12, 1 << 20, 1 << 24]
DTYPES = {"float32": mx.float32, "float16": mx.float16}


# As in crates/nuvec-compile/benches/compile.rs.
def gelu(x):
    inner = (x + (x * x * x) * 0.044715) * 0.7978846
    return [(x * 0.5) * (mx.tanh(inner) + 1.0)]


def swiglu(g, u):
    return [g * mx.sigmoid(g) * u]


def adam(p, m, v, g):
    m = m * 0.9 + g * 0.1
    v = v * 0.999 + (g * g) * 0.001
    step = m / (mx.sqrt(v) + 1e-8)
    return [p - step * 1e-3, m, v]


CASES = [("gelu", 1, gelu), ("swiglu", 2, swiglu), ("adam", 4, adam)]


def mlx_results() -> dict[tuple[str, int, str], tuple[float, float]]:
    mx.set_default_device(mx.gpu)
    out = {}
    for label, dtype in DTYPES.items():
        for name, arity, f in CASES:
            compiled = mx.compile(f)
            for n in SIZES:
                xs = [mx.random.uniform(0, 1, (n,)).astype(dtype) for _ in range(arity)]
                mx.eval(xs)
                reps = min(max(100_000_000 // n, 20), 4000)

                def burst(run) -> float:
                    t = time.perf_counter()
                    for _ in range(reps):
                        mx.async_eval(run())
                    mx.eval(run())
                    return (time.perf_counter() - t) / reps

                def best(run) -> float:
                    burst(run)
                    return min(burst(run) for _ in range(3)) * 1e6

                plain = best(lambda: f(*xs))
                fused = best(lambda: compiled(*xs))
                out[(name, n, label)] = (plain, fused)
    return out


def nuvec_results() -> dict[tuple[str, int, str], tuple[float, float]]:
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-compile", "--bench", "compile"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        name, n, label, plain, fused = line.split("\t")
        out[(name, int(n), label)] = (float(plain), float(fused))
    return out


def main() -> None:
    runs = [(nuvec_results(), mlx_results()) for _ in range(2)]
    best = lambda side, key, i: min(r[side][key][i] for r in runs)  # noqa: E731
    print(f"GPU compiled element-wise functions, {platform.machine()} {platform.mac_ver()[0]}; "
          f"MLX {mx.__version__}; µs per call (best of 3 bursts x 2 alternating runs)")
    print("| case | elements | dtype | nuvec µs (plain / compiled) | MLX µs (plain / compiled) "
          "| nuvec / MLX (plain / compiled) |")
    print("|---|---:|---|---:|---:|---:|")
    for label in DTYPES:
        ratios: list[list[float]] = [[], []]
        for name, _, _ in CASES:
            for n in SIZES:
                o = [best(0, (name, n, label), i) for i in range(2)]
                t = [best(1, (name, n, label), i) for i in range(2)]
                r = [o[i] / t[i] for i in range(2)]
                ratios[0].append(r[0])
                ratios[1].append(r[1])
                print(f"| {name} | {n} | {label} | {o[0]:.1f} / {o[1]:.1f} | "
                      f"{t[0]:.1f} / {t[1]:.1f} | {r[0]:.2f}× / {r[1]:.2f}× |")
        gm = [math.exp(sum(math.log(x) for x in rs) / len(rs)) for rs in ratios]
        print(f"| **geomean** | | {label} | | | **{gm[0]:.2f}× / {gm[1]:.2f}×** |")


if __name__ == "__main__":
    main()
