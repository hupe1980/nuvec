# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""Attention forward and backward on the GPU, nuvec against MLX: `value_and_grad` of
`sum(attention(q, k, v) * w)` for q, k and v (B 4, 16 query heads, 8 key heads, L 512 /
2,048 / 4,096, D 64 / 128, causal and not, f16 and bf16), best of 5 after a warm-up, and
peak memory. Runs nuvec's `attention_vjp` bench, then MLX's same loop.

    uv run scripts/bench_attention_vjp.py [filter]
"""

import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""


def mlx_rows():
    rows = {}
    for dname, dt in [("float16", mx.float16), ("bfloat16", mx.bfloat16)]:
        for l in [512, 2048, 4096]:
            for d in [64, 128]:
                for causal in [True, False]:
                    name = f"b4-h16-kv8-l{l}-d{d}-{'causal' if causal else 'full'}"
                    if FILTER not in name:
                        continue
                    u = lambda s: mx.random.uniform(-1, 1, s).astype(dt)
                    q, k, v, w = u((4, 16, l, d)), u((4, 8, l, d)), u((4, 8, l, d)), u((4, 16, l, d))
                    mx.eval(q, k, v, w)
                    scale = d ** -0.5
                    mask = "causal" if causal else None

                    def f(q, k, v):
                        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
                        return (o.astype(mx.float32) * w).sum()

                    g = mx.value_and_grad(f, argnums=(0, 1, 2))

                    def step():
                        t = time.perf_counter()
                        loss, grads = g(q, k, v)
                        mx.eval(loss, grads)
                        return (time.perf_counter() - t) * 1e6

                    step()
                    mx.reset_peak_memory()
                    best = min(step() for _ in range(5))
                    rows[(name, dname)] = (best, mx.get_peak_memory())
    return rows


def nuvec_rows():
    out = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "attention_vjp"],
                         cwd=ROOT, capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "BENCH_FILTER": FILTER}).stdout
    rows = {}
    for line in out.splitlines():
        f = line.split("\t")
        if len(f) == 4:
            rows[(f[0], f[1])] = (float(f[2]), int(f[3]))
    return rows


def main():
    nv, ml = nuvec_rows(), mlx_rows()
    print(f"attention forward + backward, MLX {mx.__version__}; µs (best of 5), peak MB")
    print("| case | dtype | nuvec / MLX µs | ratio | peak nuvec / MLX MB | ratio |")
    print("|---|---|---:|---:|---:|---:|")
    for k in nv:
        a, b = nv[k], ml[k]
        print(f"| {k[0]} | {k[1]} | {a[0]:.0f} / {b[0]:.0f} | {a[0] / b[0]:.2f}× | "
              f"{a[1] / 1e6:.0f} / {b[1] / 1e6:.0f} | {a[1] / b[1]:.2f}× |")


if __name__ == "__main__":
    main()
