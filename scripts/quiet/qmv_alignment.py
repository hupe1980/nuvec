# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx==0.32.4"]
# ///
"""MLX's quantized GEMV at aligned and unaligned K (research/11 §4, opportunity 2): K that is
not a multiple of 512 (4-bit) takes MLX's generic `qmv`, not `qmv_fast`. Prints µs per call
and GB/s. Decision: unaligned ≤ 0.90× aligned GB/s → build nuvec's fast unaligned GEMV."""

import json
import os
import time

import mlx.core as mx


def bench(f, per=8, rounds=3, iters=200):
    if os.environ.get("QUICK"):
        rounds, iters = 1, 2
    best = float("inf")
    for _ in range(rounds):
        mx.eval(f())
        t = time.perf_counter()
        for _ in range(iters):
            mx.eval([f() for _ in range(per)])
        best = min(best, (time.perf_counter() - t) / (iters * per))
    return best


def main():
    mx.random.seed(0)
    cases = [("affine", 4, 64), ("affine", 8, 64), ("mxfp4", 4, 32)]
    shapes = [(4096, 2560), (4096, 2816), (4096, 2880), (4096, 3072), (2048, 768), (2048, 1024)]
    for mode, bits, gs in cases:
        for n, k in shapes:
            if k % gs:
                continue
            w = mx.random.normal((n, k)).astype(mx.bfloat16)
            q = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
            wq, scales = q[0], q[1]
            biases = q[2] if mode == "affine" else None
            x = mx.random.normal((1, k)).astype(mx.bfloat16)
            mx.eval(wq, scales, x, *( [biases] if biases is not None else []))
            f = lambda: mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=gs, bits=bits, mode=mode)
            s = bench(f)
            nbytes = wq.nbytes + scales.nbytes + (biases.nbytes if biases is not None else 0)
            print(json.dumps({"mode": mode, "bits": bits, "group": gs, "n": n, "k": k,
                              "aligned_512": k % 512 == 0, "us": s * 1e6, "gbps": nbytes / s / 1e9}))


if __name__ == "__main__":
    main()
