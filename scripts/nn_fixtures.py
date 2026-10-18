# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "numpy==2.5.4"]
# ///
"""Records `mlx.nn` and `mlx.optimizers` results as test fixtures for nuvec-nn.

Runs MLX on its CPU device with fixed seeds and writes every input and expected output to
one safetensors file, which `crates/nuvec-nn/tests/mlx.rs` replays (names are
`<case>.<tensor>`). The LoRA layer is mlx-lm's `LoRALinear`, restated here so that mlx-lm
is not a dependency.

Usage: `just nn-fixtures` (or `uv run scripts/nn_fixtures.py`).
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

mx.set_default_device(mx.cpu)
OUT = Path(__file__).resolve().parent.parent / "crates/nuvec-nn/tests/fixtures/mlx_nn.safetensors"
rng = np.random.default_rng(20261018)
out: dict[str, mx.array] = {}


def rand(*shape: int) -> mx.array:
    return mx.array(rng.normal(size=shape).astype(np.float32))


def put(name: str, **tensors: mx.array) -> None:
    for k, v in tensors.items():
        out[f"{name}.{k}"] = v


# ── Initialisation (seeded draws in MLX's order) ────────────────────────────

mx.random.seed(1)
put("init_embedding", weight=nn.Embedding(10, 4).weight)

mx.random.seed(2)
base = nn.Linear(6, 5)
# mlx-lm's LoRALinear.from_base: a fresh nn.Linear(6, 5, bias=False), then lora_a.
nn.Linear(6, 5, bias=False)
put("init_lora", lora_a=mx.random.uniform(low=-1 / math.sqrt(6), high=1 / math.sqrt(6), shape=(6, 3)))

# ── Layers and activations ──────────────────────────────────────────────────

emb = nn.Embedding(7, 3)
emb.weight = rand(7, 3)
ids = mx.array([[0, 6, 2], [2, 2, 5]], dtype=mx.uint32)
x3 = rand(2, 3)
put("embedding", weight=emb.weight, ids=ids, out=emb(ids), x=x3, as_linear=emb.as_linear(x3))

x = rand(2, 3, 8)
rms = nn.RMSNorm(8, eps=1e-5)
rms.weight = rand(8)
put("rms_norm", x=x, weight=rms.weight, out=rms(x))
ln = nn.LayerNorm(8, eps=1e-5)
ln.weight, ln.bias = rand(8), rand(8)
put("layer_norm", x=x, weight=ln.weight, bias=ln.bias, out=ln(x),
    plain=nn.LayerNorm(8, eps=1e-5, affine=False)(x))

q = rand(1, 2, 5, 12)
put("rope", x=q, out=nn.RoPE(8)(q, offset=3), traditional=nn.RoPE(12, traditional=True)(q, offset=0))

a = rand(4, 6) * 3
put("activations", x=a, silu=nn.silu(a), gelu_approx=nn.gelu_approx(a),
    gelu_fast_approx=nn.gelu_fast_approx(a), relu=nn.relu(a))

w, b, la, lb = rand(5, 6), rand(5), rand(6, 3), rand(3, 5)
xl = rand(2, 6)
y = (xl @ w.T + b) + (20.0 * ((xl @ la) @ lb)).astype(xl.dtype)
fused = w + (20.0 * lb.T @ la.T).astype(w.dtype)
put("lora", weight=w, bias=b, lora_a=la, lora_b=lb, x=xl, out=y, fused_weight=fused)

# ── Optimizers: three steps from the same parameters and gradients ─────────

params = {"matrix": rand(3, 4), "vector": rand(4)}
grads = [{"matrix": rand(3, 4), "vector": rand(4)} for _ in range(3)]
put("optim", matrix=params["matrix"], vector=params["vector"],
    **{f"g{i}_{k}": g[k] for i, g in enumerate(grads) for k in g})
OPTIMIZERS = {
    "sgd": lambda: optim.SGD(0.1, momentum=0.9, weight_decay=0.01, nesterov=True),
    "adam": lambda: optim.Adam(0.1, bias_correction=True),
    "adamw": lambda: optim.AdamW(0.1, weight_decay=0.1),
    "lion": lambda: optim.Lion(0.1, weight_decay=0.1),
    "muon": lambda: optim.Muon(0.1),
    "adamw_cosine": lambda: optim.AdamW(optim.cosine_decay(0.1, 4, 0.01)),
}
for name, make in OPTIMIZERS.items():
    opt, p = make(), dict(params)
    for i, g in enumerate(grads):
        p = opt.apply_gradients(g, p)
        mx.eval(p)
        put(f"optim_{name}", **{f"s{i}_{k}": p[k] for k in p})

# ── Schedules and clipping ──────────────────────────────────────────────────

SCHEDULES = {
    "exponential": optim.exponential_decay(0.1, 0.9),
    "step": optim.step_decay(0.1, 0.5, 3),
    "cosine": optim.cosine_decay(0.1, 10, 0.01),
    "linear": optim.linear_schedule(0.0, 0.1, 5),
    "joined": optim.join_schedules([optim.linear_schedule(0.0, 0.1, 5), optim.cosine_decay(0.1, 10)], [5]),
}
steps = mx.arange(0, 20, dtype=mx.uint64)
for name, s in SCHEDULES.items():
    put("schedule", **{name: mx.stack([mx.array(s(t)).astype(mx.float32) for t in steps])})

gs = [rand(3, 4), rand(5).astype(mx.float16) * 4]
(clipped, norm) = optim.clip_grad_norm(gs, 1.0)
put("clip", g0=gs[0], g1=gs[1], c0=clipped[0], c1=clipped[1], norm=norm)

mx.eval(out)
OUT.parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(OUT), out)
print(f"wrote {len(out)} tensors -> {OUT.name}")
