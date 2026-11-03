# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx==0.32.4"]
# ///
"""The upper bound of fusing MoE decode glue, on MLX with
Qwen3-30B-A3B's shapes: 48 layers, 128 experts, top-8, D = 2048, I = 768, 4-bit experts.
(a) mlx-lm's path (router, softmax, argpartition, gather, normalise, gate/up/down gathers,
combine); (b) the three gathers alone with fixed indices; (c) (b) with gate and up as one
gather. Decision: (a) − (b) ≥ 0.8 ms per 48 layers → build the fused routing and combine;
(c) saving ≥ 0.3 ms more → fuse gate and up."""

import json
import os
import time

import mlx.core as mx

L, E, K, D, I, GS, BITS = (2 if os.environ.get("QUICK") else 48), 128, 8, 2048, 768, 64, 4


def q(shape):
    return mx.quantize(mx.random.normal(shape).astype(mx.bfloat16) * 0.02, group_size=GS, bits=BITS)


def gq(x, w, idx):
    return mx.gather_qmm(x, *w, rhs_indices=idx, transpose=True, group_size=GS, bits=BITS)


def main():
    mx.random.seed(0)
    layers = []
    for _ in range(L):
        layers.append({"router": q((E, D)), "gate": q((E, I, D)), "up": q((E, I, D)),
                       "gate_up": q((E, 2 * I, D)), "down": q((E, D, I))})
    mx.eval(layers)
    x0 = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    fixed = mx.arange(K, dtype=mx.uint32).reshape(1, 1, K)

    def full():
        x = x0
        for p in layers:
            logits = mx.quantized_matmul(x, *p["router"], transpose=True, group_size=GS, bits=BITS)
            gates = mx.softmax(logits, axis=-1, precise=True)
            top = mx.argpartition(-gates, kth=K - 1, axis=-1)[..., :K]
            scores = mx.take_along_axis(gates, top, axis=-1)
            scores = scores / scores.sum(-1, keepdims=True)
            xe = x[..., None, None, :]
            h = nn_silu(gq(xe, p["gate"], top)) * gq(xe, p["up"], top)
            y = gq(h, p["down"], top).squeeze(-2)
            x = x + (y * scores[..., None]).sum(-2)
        return x

    def gathers(fused):
        def run():
            x = x0
            for p in layers:
                xe = x[..., None, None, :]
                if fused:
                    gu = gq(xe, p["gate_up"], fixed)
                    h = nn_silu(gu[..., :I]) * gu[..., I:]
                else:
                    h = nn_silu(gq(xe, p["gate"], fixed)) * gq(xe, p["up"], fixed)
                x = x + gq(h, p["down"], fixed).squeeze(-2).sum(-2)
            return x
        return run

    variants = {"a_mlx_lm": full, "b_gathers": gathers(False), "c_gate_up": gathers(True)}
    times = {k: [] for k in variants}
    for _ in range(3):
        for name, f in variants.items():
            for _ in range(5):
                mx.eval(f())
            t = time.perf_counter()
            for _ in range(50):
                mx.eval(f())
            times[name].append((time.perf_counter() - t) / 50 * 1e3)
    out = {k: min(v) for k, v in times.items()}
    out["glue_ms"] = out["a_mlx_lm"] - out["b_gathers"]
    out["gate_up_saving_ms"] = out["b_gathers"] - out["c_gate_up"]
    print(json.dumps(out))


def nn_silu(x):
    return x * mx.sigmoid(x)


if __name__ == "__main__":
    main()
