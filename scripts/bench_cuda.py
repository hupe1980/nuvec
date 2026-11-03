# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx[cuda13]==0.32.4"]
# ///
"""MLX's side of the CUDA spike: the shapes of nuvec-gpu's `cuda_bench` example, on
MLX's CUDA backend, as JSON lines. Run on the same GPU (scripts/cuda_remote.py). TF32 for
float32 products follows MLX_ENABLE_TF32 (MLX's default: on)."""

import json
import os
import time

import mlx.core as mx


def bench(f, iters):
    for _ in range(3):
        mx.eval(f())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(f())
    mx.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    dev = mx.device_info() if hasattr(mx, "device_info") else {}
    print(json.dumps({"tool": "mlx", "version": mx.__version__, "tf32": os.environ.get("MLX_ENABLE_TF32", "1"),
                      "device": str(dev.get("device_name", ""))}))
    n = 1 << 26
    for dt, name in [(mx.float32, "float32"), (mx.float16, "float16"), (mx.bfloat16, "bfloat16")]:
        a = mx.ones((n,), dtype=dt)
        b = mx.ones((n,), dtype=dt)
        mx.eval(a, b)
        for op, f, arity in [("Add", lambda: a + b, 2), ("Exp", lambda: mx.exp(a), 1)]:
            s = bench(f, 20)
            size = a.itemsize
            print(json.dumps({"op": op, "dtype": name, "n": n, "ms": s * 1e3, "gbps": (arity + 1) * n * size / s / 1e9}))
        x = mx.ones((8192, 8192), dtype=dt)
        mx.eval(x)
        s = bench(lambda: mx.sum(x, axis=-1), 20)
        print(json.dumps({"op": "RowSum", "dtype": name, "shape": [8192, 8192], "ms": s * 1e3,
                          "gbps": 8192 * 8192 * x.itemsize / s / 1e9}))
    for dt, name in [(mx.float16, "float16"), (mx.bfloat16, "bfloat16"), (mx.float32, "float32")]:
        for size in [1024, 2048, 4096, 8192]:
            a = mx.ones((size, size), dtype=dt)
            b = mx.ones((size, size), dtype=dt)
            mx.eval(a, b)
            s = bench(lambda: a @ b, 5 if size >= 8192 else 20)
            print(json.dumps({"op": "Gemm", "dtype": name, "size": size, "ms": s * 1e3,
                              "tflops": 2 * size**3 / s / 1e12}))


if __name__ == "__main__":
    main()
