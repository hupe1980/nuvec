# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "numpy==2.5.4"]
# ///
"""Records `mlx.nn` and `mlx.optimizers` results as test fixtures for nuvec-nn.

Runs MLX on its CPU device with fixed seeds and writes every input and expected output to
one safetensors file, which `crates/nuvec-nn/tests/mlx.rs` replays (names are
`<case>.<tensor>`). The LoRA layer is mlx-lm's `LoRALinear`, restated here so that mlx-lm
is not a dependency.

Also records the parameter names and counts of every layer (`mlx_nn_names.json`),
an MLX `save_weights` file of a small model (`mlx_save_weights.safetensors`), optimizer state
(`step`, `learning_rate`), AdamW's defaults, `MultiOptimizer`, a freeze/unfreeze sequence and
dropout outputs.

Usage: `just nn-fixtures` (or `uv run scripts/nn_fixtures.py`);
`uv run scripts/nn_fixtures.py --verify <file>` loads a nuvec-written safetensors file with
MLX's `model.load_weights` (the model of `mlx_save_weights.safetensors`) and prints its
names, dtypes and checksums.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

mx.set_default_device(mx.cpu)
FIXTURES = Path(__file__).resolve().parent.parent / "crates/nuvec-nn/tests/fixtures"
OUT = FIXTURES / "mlx_nn.safetensors"
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

# ── Optimizer state, AdamW defaults, MultiOptimizer, freezing ─────

for name, make in {
    "sgd": lambda: optim.SGD(0.1, momentum=0.9, weight_decay=0.01, nesterov=True),
    "adamw_cosine": lambda: optim.AdamW(optim.cosine_decay(0.1, 4, 0.01)),
    "sgd_joined": lambda: optim.SGD(optim.join_schedules(
        [optim.linear_schedule(0.0, 0.1, 2), optim.cosine_decay(0.1, 3)], [2])),
}.items():
    opt, p = make(), dict(params)
    lrs, stepz = [], []
    for g in grads + grads[:2]:
        p = opt.apply_gradients(g, p)
        mx.eval(p, opt.state)
        lrs.append(opt.state["learning_rate"].astype(mx.float32))
        stepz.append(opt.state["step"])
    put(f"state_{name}", learning_rate=mx.stack(lrs), step=mx.stack(stepz).astype(mx.uint64))

opt, p = optim.AdamW(1e-3), dict(params)
for i, g in enumerate(grads):
    p = opt.apply_gradients(g, p)
    mx.eval(p)
    put("optim_adamw_default", **{f"s{i}_{k}": p[k] for k in p})


multi_params = {
    "layer": {"weight": rand(4, 6), "lora_a": rand(6, 2), "lora_b": rand(2, 4)},
    "out": {"weight": rand(3, 4)},
}
multi_grads = [{k: {n: rand(*v.shape) for n, v in sub.items()} for k, sub in multi_params.items()}
               for _ in range(3)]
put("multi", **{f"p_{k}.{n}": v for k, sub in multi_params.items() for n, v in sub.items()})
for i, g in enumerate(multi_grads):
    put("multi", **{f"g{i}_{k}.{n}": v for k, sub in g.items() for n, v in sub.items()})
multi = optim.MultiOptimizer(
    [optim.Adam(1e-3), optim.SGD(1e-2, weight_decay=1e-4)],
    [lambda path, _: "lora_a" in path],
)
p = multi_params
for i, g in enumerate(multi_grads):
    p = multi.apply_gradients(g, p)
    mx.eval(p)
    put("multi", **{f"s{i}_{k}.{n}": v for k, sub in p.items() for n, v in sub.items()})

# Adam over two layers: 3 steps, `a` frozen for 2 steps, unfrozen for 1 (constant gradients
# by path, as nuvec's test).
mx.random.seed(5)
pair = nn.Module()
pair.a, pair.b = nn.Linear(3, 2), nn.Linear(2, 2)
put("freeze_init", **dict(tree_flatten(pair.parameters())))
SCALE = {"a.weight": 0.1, "a.bias": -0.2, "b.weight": 0.3, "b.bias": -0.4}
def constant_grads(m):
    flat = tree_flatten(m.trainable_parameters())
    return tree_unflatten([(k, mx.full(v.shape, SCALE[k])) for k, v in flat])
adam = optim.Adam(0.01)
for step in range(6):
    if step == 3:
        pair.a.freeze()
    if step == 5:
        pair.a.unfreeze()
    adam.update(pair, constant_grads(pair))
    mx.eval(pair.parameters(), adam.state)
    put("freeze", **{f"s{step}_{k}": v for k, v in tree_flatten(pair.parameters())})
put("freeze_state", **{f"{k}": v for k, v in tree_flatten(adam.state) if k != "learning_rate"})

# ── Dropout ───────────────────────────────────────────────────────

xd = rand(4, 8)
put("dropout", x=xd)
for seed in (0, 1):
    for prob in (0.1, 0.5):
        mx.random.seed(seed)
        put("dropout", **{f"s{seed}_p{int(prob * 10)}": nn.Dropout(prob)(xd)})

mx.eval(out)
OUT.parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(str(OUT), out)
print(f"wrote {len(out)} tensors -> {OUT.name}")

# ── Parameter names and counts, save_weights ──────────────────────


def names(module) -> list[str]:
    return [k for k, _ in tree_flatten(module.parameters())]


def count(tree) -> int:
    return sum(v.size for _, v in tree_flatten(tree))


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = [nn.Linear(i, o) for i, o in zip(sizes[:-1], sizes[1:])]


mx.random.seed(0)
ql = nn.QuantizedLinear.from_linear(nn.Linear(64, 8))
ql.unfreeze()
mlp = MLP([784, 32, 32, 10])
lora_base = nn.Linear(6, 5)


class LoRALinear(nn.Module):
    # mlx-lm's tuner/lora.py LoRALinear (names only).
    def __init__(self, linear, r):
        super().__init__()
        self.linear = linear
        out_dims, in_dims = linear.weight.shape
        self.lora_a = mx.zeros((in_dims, r))
        self.lora_b = mx.zeros((r, out_dims))


lora = LoRALinear(lora_base, 3)
lora.freeze()
lora.unfreeze(keys=["lora_a", "lora_b"], strict=False)
NAMES = {
    "Linear": names(nn.Linear(4, 3)),
    "Linear_no_bias": names(nn.Linear(4, 3, bias=False)),
    "QuantizedLinear": names(nn.QuantizedLinear.from_linear(nn.Linear(64, 8))),
    "QuantizedLinear_trainable_after_unfreeze": [k for k, _ in tree_flatten(ql.trainable_parameters())],
    "Embedding": names(nn.Embedding(10, 4)),
    "QuantizedEmbedding": names(nn.QuantizedEmbedding.from_embedding(nn.Embedding(10, 64))),
    "RMSNorm": names(nn.RMSNorm(8)),
    "LayerNorm": names(nn.LayerNorm(8)),
    "LayerNorm_no_bias": names(nn.LayerNorm(8, bias=False)),
    "LayerNorm_no_affine": names(nn.LayerNorm(8, affine=False)),
    "RoPE": names(nn.RoPE(8)),
    "LoRALinear": names(lora),
    "LoRALinear_trainable": [k for k, _ in tree_flatten(lora.trainable_parameters())],
    "MLP": names(mlp),
    "MLP_total": count(mlp.parameters()),
    "MLP_trainable": count(mlp.trainable_parameters()),
}
(FIXTURES / "mlx_nn_names.json").write_text(json.dumps(NAMES, indent=1) + "\n")
print(f"wrote {len(NAMES)} name lists -> mlx_nn_names.json")


class Small(nn.Module):
    # The model of mlx_save_weights.safetensors.
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(8, 4)
        self.quantized = nn.QuantizedLinear.from_linear(nn.Linear(64, 4))
        self.norm = nn.LayerNorm(4)


mx.random.seed(3)
small = Small()
small.norm.weight = rand(4)
mx.eval(small.parameters())
small.save_weights(str(FIXTURES / "mlx_save_weights.safetensors"))
print("wrote mlx_save_weights.safetensors")

if len(sys.argv) == 3 and sys.argv[1] == "--verify":
    check = Small()
    check.load_weights(sys.argv[2])
    for k, v in tree_flatten(check.parameters()):
        print(k, v.dtype, v.shape, float(mx.sum(v.astype(mx.float32)).item()))
    print("verified: MLX loaded", sys.argv[2])
