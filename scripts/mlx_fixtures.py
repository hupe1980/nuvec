# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "numpy==2.5.4"]
# ///
"""Records MLX outputs as test fixtures for nuvec.

Runs MLX on its CPU device with fixed seeds and writes every case to one binary file
(format "BSMXFIX1", documented at the bottom of this file). The Rust test
`crates/nuvec-core/tests/mlx.rs` replays the cases without Python.

Usage: `just fixtures` (or `uv run scripts/mlx_fixtures.py`).
"""

from __future__ import annotations

import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)  # deterministic, and the device nuvec is compared with

SEED = 20261016
OUT = Path(__file__).resolve().parent.parent / "crates/nuvec-core/tests/fixtures/mlx.bin"

# nuvec DType order -> (code, MLX dtype)
DTYPES = [
    ("bool", mx.bool_), ("uint8", mx.uint8), ("uint16", mx.uint16), ("uint32", mx.uint32),
    ("uint64", mx.uint64), ("int8", mx.int8), ("int16", mx.int16), ("int32", mx.int32),
    ("int64", mx.int64), ("float16", mx.float16), ("bfloat16", mx.bfloat16),
    ("float32", mx.float32), ("float64", mx.float64), ("complex64", mx.complex64),
]
CODE = {name: i for i, (name, _) in enumerate(DTYPES)}
MX = dict(DTYPES)
NAME_OF = {dt: name for name, dt in DTYPES}

rng = np.random.default_rng(SEED)


# ── Inputs ──────────────────────────────────────────────────────────────────

def make(dtype: str, shape: tuple[int, ...], edge: bool = False) -> mx.array:
    """Random values of `dtype`; with `edge`, include NaN/inf or integer limits."""
    n = int(np.prod(shape)) if shape else 1
    if dtype == "bool":
        data = rng.integers(0, 2, n).astype(np.bool_)
        return mx.array(data.reshape(shape))
    if dtype.startswith(("uint", "int")):
        info = np.iinfo(np.dtype(dtype))
        data = rng.integers(info.min, info.max, n, dtype=np.dtype(dtype), endpoint=True)
        if edge and n >= 3:
            data[:3] = [info.min, info.max, 0]
        return mx.array(data.reshape(shape))
    if dtype == "complex64":
        data = (rng.normal(size=n) * 4 + 1j * rng.normal(size=n) * 4).astype(np.complex64)
        return mx.array(data.reshape(shape))
    data = (rng.normal(size=n) * 4).astype(np.float32)
    if edge and n >= 4:
        data[:4] = [np.nan, np.inf, -np.inf, -0.0]
    if dtype == "float64":
        return mx.array(data.astype(np.float64).reshape(shape))
    return mx.array(data.reshape(shape)).astype(MX[dtype])


def ints(dtype: str, shape: tuple[int, ...], lo: int, hi: int) -> mx.array:
    n = int(np.prod(shape)) if shape else 1
    return mx.array(rng.integers(lo, hi, n).reshape(shape)).astype(MX[dtype])


# ── The operation table (mirrored by the Rust replayer) ─────────────────────

def nums(s: str) -> list[int]:
    return [int(x) for x in s.split(",")] if s else []


def pre_view(x: mx.array, pre: str | None) -> mx.array:
    if pre == "t":
        return mx.transpose(x)
    if pre == "r":
        return x[..., ::-1]
    return x


def parse_slice(spec: str) -> tuple[slice, ...]:
    out = []
    for part in spec.split(";"):
        a, b, c = part.split(":")
        out.append(slice(int(a) if a else None, int(b) if b else None, int(c)))
    return tuple(out)


def scalar(spec: str) -> bool | int | float | complex:
    """A Python scalar written as `<kind>:<value>` (complex values as `re,im`)."""
    kind, value = spec.split(":", 1)
    if kind == "complex":
        re, im = value.split(",")
        return complex(float(re), float(im))
    return {"bool": lambda v: v == "True", "int": int, "float": float}[kind](value)


def qparams(p: dict[str, str]) -> dict[str, object]:
    """MLX's quantization keywords from a case's parameters (mode defaults when absent)."""
    out: dict[str, object] = {"mode": p.get("mode", "affine")}
    for key in ("group_size", "bits"):
        if key in p:
            out[key] = int(p[key])
    return out


def apply(op: str, p: dict[str, str], xs: list[mx.array]) -> mx.array:
    if xs:
        xs = [pre_view(xs[0], p.get("pre"))] + xs[1:]
    binary = {
        "add": mx.add, "subtract": mx.subtract, "multiply": mx.multiply, "divide": mx.divide,
        "maximum": mx.maximum, "minimum": mx.minimum, "logaddexp": mx.logaddexp,
        "floor_divide": mx.floor_divide, "power": mx.power, "equal": mx.equal,
        "not_equal": mx.not_equal, "less": mx.less, "less_equal": mx.less_equal,
        "greater": mx.greater, "greater_equal": mx.greater_equal, "matmul": mx.matmul,
    }
    unary = {
        "negative": mx.negative, "abs": mx.abs, "exp": mx.exp, "log": mx.log, "sqrt": mx.sqrt,
        "sin": mx.sin, "cos": mx.cos, "tanh": mx.tanh, "sigmoid": mx.sigmoid,
        "conjugate": mx.conjugate, "erfinv": mx.erfinv,
    }
    reductions = {
        "sum": mx.sum, "prod": mx.prod, "max": mx.max, "min": mx.min, "mean": mx.mean,
        "all": mx.all, "any": mx.any,
    }
    if op in binary and "scalar" in p:
        v = scalar(p["scalar"])
        return binary[op](v, xs[0]) if p.get("side") == "left" else binary[op](xs[0], v)
    if op in binary:
        return binary[op](xs[0], xs[1])
    if op in unary:
        return unary[op](xs[0])
    if op == "addmm":
        return mx.addmm(xs[2], xs[0], xs[1], alpha=float(p["alpha"]), beta=float(p["beta"]))
    if op == "rms_norm":
        return mx.fast.rms_norm(xs[0], xs[1] if len(xs) > 1 else None, float(p["eps"]))
    if op == "layer_norm":
        w, b = (xs + [None, None])[1:3]
        return mx.fast.layer_norm(xs[0], w, b, float(p["eps"]))
    if op == "rope":
        base = float(p["base"]) if "base" in p else None
        return mx.fast.rope(xs[0], int(p["dims"]), traditional=p["traditional"] == "1", base=base,
                            scale=float(p["scale"]), offset=xs[1],
                            freqs=xs[2] if len(xs) > 2 else None)
    if op == "cross_entropy":
        return mx.fast.cross_entropy(xs[0], xs[1])
    if op == "sdpa":
        # `cap` (nuvec's max_score_bytes) changes how nuvec runs, not the result: MLX
        # computes the uncapped attention. A `window` is mlx-lm's sliding causal mask.
        mode = p.get("mask", "none")
        rest = xs[3:]
        mask = "causal" if mode == "causal" else (rest.pop(0) if mode == "array" else None)
        if "window" in p:
            lq, lk, w = xs[0].shape[2], xs[1].shape[2], int(p["window"])
            pos = mx.arange(lk - lq, lk)[:, None]
            keys = mx.arange(lk)[None]
            mask = (pos >= keys) & (pos < keys + w)
        sinks = rest[0] if rest else None
        return mx.fast.scaled_dot_product_attention(xs[0], xs[1], xs[2], scale=float(p["scale"]),
                                                    mask=mask, sinks=sinks)
    if op == "quantize":
        return list(mx.quantize(xs[0], **qparams(p)))
    if op == "dequantize":
        biases = xs[2] if len(xs) > 2 else None
        dtype = DTYPES[int(p["dtype"])][1] if "dtype" in p else None
        return mx.dequantize(xs[0], xs[1], biases, dtype=dtype, **qparams(p))
    if op in ("quantized_matmul", "quantized_matmul_xsb"):
        if op == "quantized_matmul":
            x, w, s = xs[:3]
            b = xs[3] if len(xs) > 3 else None
        else:
            x, w, s, b = xs[0], xs[3], xs[1], xs[2]
        return mx.quantized_matmul(x, w, s, b, transpose=p["transpose"] == "1", **qparams(p))
    if op in ("logsumexp", "softmax"):
        axes = nums(p.get("axes", "")) or None
        if op == "softmax":
            return mx.softmax(xs[0], axis=axes)
        return mx.logsumexp(xs[0], axis=axes, keepdims=p.get("keepdims") == "1")
    if op == "random":
        dtype = DTYPES[int(p.get("dtype", CODE["float32"]))][1]
        shape = nums(p.get("shape", ""))
        if p.get("seq") == "1":
            mx.random.seed(int(p["seed"]))
            key = None
            for _ in range(int(p.get("skip", "0"))):
                mx.random.uniform(shape=shape)
        else:
            key = mx.random.key(int(p["seed"]))
        fn = p["fn"]
        if fn == "bits":
            return mx.random.bits(shape, width=int(p["width"]), key=key)
        if fn == "split":
            return mx.random.split(key, num=int(p["num"]))
        if fn == "uniform":
            return mx.random.uniform(float(p["low"]), float(p["high"]), shape, dtype, key=key)
        if fn == "normal":
            loc = float(p["loc"]) if "loc" in p else None
            scale = float(p["scale"]) if "scale" in p else None
            kw = {k: v for k, v in (("loc", loc), ("scale", scale)) if v is not None}
            return mx.random.normal(shape, dtype, key=key, **kw)
        if fn == "gumbel":
            return mx.random.gumbel(shape, dtype, key=key)
        if fn == "categorical":
            return mx.random.categorical(xs[0], axis=int(p["axis"]), key=key)
        if fn == "randint":
            return mx.random.randint(int(p["low"]), int(p["high"]), shape, dtype, key=key)
        if fn == "bernoulli":
            return mx.random.bernoulli(float(p["p"]), shape, key=key)
        raise ValueError(fn)
    if op == "where":
        return mx.where(xs[0], xs[1], xs[2])
    if op == "astype":
        return xs[0].astype(DTYPES[int(p["dtype"])][1])
    if op == "reshape":
        return mx.reshape(xs[0], nums(p["shape"]))
    if op == "transpose":
        axes = nums(p.get("axes", ""))
        return mx.transpose(xs[0], axes) if axes else mx.transpose(xs[0])
    if op == "slice":
        return xs[0][parse_slice(p["args"])]
    if op == "broadcast_to":
        return mx.broadcast_to(xs[0], nums(p["shape"]))
    if op == "conv1d":
        return mx.conv1d(xs[0], xs[1], stride=int(p["stride"]), padding=int(p["padding"]),
                         dilation=int(p["dilation"]), groups=int(p["groups"]))
    if op == "gather_mm":
        # Inputs: a, b, then the indices that `idx` names ("l" lhs, "r" rhs).
        rest = iter(xs[2:])
        lhs = next(rest) if "l" in p["idx"] else None
        rhs = next(rest) if "r" in p["idx"] else None
        b = mx.swapaxes(xs[1], -1, -2) if p.get("tb") == "1" else xs[1]
        return mx.gather_mm(xs[0], b, lhs, rhs, sorted_indices=p["sorted"] == "1")
    if op == "gather_qmm":
        # Inputs: x, w, scales, a global scale if `gs`, then the indices that `idx` names.
        rest = iter(xs[3:])
        gs = next(rest) if p.get("gs") == "1" else None
        lhs = next(rest) if "l" in p["idx"] else None
        rhs = next(rest) if "r" in p["idx"] else None
        return mx.gather_qmm(xs[0], xs[1], xs[2], None, lhs, rhs, transpose=p["transpose"] == "1",
                             global_scale=gs, sorted_indices=p["sorted"] == "1", **qparams(p))
    if op == "to_fp8":
        return mx.to_fp8(xs[0])
    if op == "from_fp8":
        return mx.from_fp8(xs[0], DTYPES[int(p["dtype"])][1])
    if op == "view":
        return xs[0].view(DTYPES[int(p["dtype"])][1])
    if op == "repeat":
        axis = int(p["axis"]) if p.get("axis", "") != "" else None
        return mx.repeat(xs[0], int(p["repeats"]), axis)
    if op == "contiguous":
        return mx.contiguous(xs[0].T if p.get("pre") == "t" else xs[0])
    if op == "expand_dims":
        return mx.expand_dims(xs[0], int(p["axis"]))
    if op == "squeeze":
        axes = nums(p.get("axes", ""))
        return mx.squeeze(xs[0], axes) if axes else mx.squeeze(xs[0])
    if op in reductions:
        axes = nums(p.get("axes", ""))
        keep = p.get("keepdims") == "1"
        return reductions[op](xs[0], axis=axes or None, keepdims=keep)
    if op in ("argmax", "argmin"):
        fn = mx.argmax if op == "argmax" else mx.argmin
        axis = p.get("axis", "")
        return fn(xs[0], axis=int(axis) if axis else None, keepdims=p.get("keepdims") == "1")
    if op == "arange":
        return mx.arange(float(p["start"]), float(p["stop"]), float(p["step"]),
                         dtype=DTYPES[int(p["dtype"])][1])
    if op in ("sort", "argsort"):
        return (mx.sort if op == "sort" else mx.argsort)(xs[0], axis=int(p["axis"]))
    if op == "topk":
        # MLX leaves the order of the k elements unspecified: compare them sorted.
        return mx.sort(mx.topk(xs[0], int(p["k"]), axis=int(p["axis"])), axis=int(p["axis"]))
    if op in ("cumsum", "cumprod", "cummax", "cummin", "logcumsumexp"):
        fn = getattr(mx, op)
        axis = p.get("axis", "")
        return fn(xs[0], axis=int(axis) if axis else None, reverse=p.get("reverse") == "1",
                  inclusive=p.get("inclusive", "1") == "1")
    if op == "take":
        axis = p.get("axis", "")
        return mx.take(xs[0], xs[1], axis=int(axis) if axis else None)
    if op == "take_along_axis":
        return mx.take_along_axis(xs[0], xs[1], axis=int(p["axis"]))
    if op == "put_along_axis":
        a, i, v = (xs[0], xs[2], xs[1]) if p.get("order") == "avi" else xs[:3]
        return mx.put_along_axis(a, i, v, axis=int(p["axis"]))
    if op == "concatenate":
        return mx.concatenate(xs, axis=int(p["axis"]))
    if op == "stack":
        return mx.stack(xs, axis=int(p["axis"]))
    if op == "split":
        at = p.get("at")
        return list(mx.split(xs[0], nums(at) if at else int(p["parts"]), axis=int(p["axis"])))
    if op in ("zeros", "ones"):
        fn = mx.zeros if op == "zeros" else mx.ones
        return fn(nums(p["shape"]), dtype=DTYPES[int(p["dtype"])][1])
    raise ValueError(op)


# ── Cases ───────────────────────────────────────────────────────────────────

cases: list[tuple[str, str, list[mx.array], list[mx.array]]] = []


def case(name: str, op: str, inputs: list[mx.array], **params: object) -> None:
    p = {k: str(v) for k, v in params.items()}
    spelled = op + "".join(f"|{k}={v}" for k, v in p.items())
    out = apply(op, p, list(inputs))
    outs = out if isinstance(out, list) else [out]
    mx.eval(outs)
    cases.append((name, spelled, list(inputs), outs))


ALL = [name for name, _ in DTYPES]
INTS = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"]
FLOATS = ["float16", "bfloat16", "float32", "float64"]

# Element-wise binary over every dtype, with edge values and a broadcast.
for dt in ALL:
    for op in ["add", "subtract", "multiply", "divide", "equal", "less"]:
        if op == "less" and dt == "complex64":
            continue
        case(f"{op}-{dt}", op, [make(dt, (3, 4), edge=True), make(dt, (4,))])
    if dt != "complex64":
        case(f"maximum-{dt}", "maximum", [make(dt, (2, 5), edge=True), make(dt, (2, 5))])
        case(f"minimum-{dt}-t", "minimum", [make(dt, (5, 2), edge=True), make(dt, (2, 5))], pre="t")

# logaddexp: every float type with NaN/inf edges (both infinite included), integers promoted.
for dt in ["float32", "float16", "bfloat16", "float64"]:
    case(f"logaddexp-{dt}", "logaddexp", [make(dt, (3, 4), edge=True), make(dt, (4,))])
INF = mx.array([np.inf, -np.inf, np.inf, -np.inf, np.nan, 0.0, -30.0, 80.0])
case("logaddexp-infinities", "logaddexp",
     [INF, mx.array([np.inf, -np.inf, -np.inf, 1.0, 2.0, -30.0, 0.0, 80.0])])
case("logaddexp-int", "logaddexp", [ints("int32", (2, 3), -5, 5), ints("int32", (3,), -5, 5)])

# Promotion between mixed dtypes.
for a, b in [("int8", "uint8"), ("uint64", "int64"), ("int32", "float16"), ("float16", "bfloat16"),
             ("bool", "float32"), ("uint32", "int16"), ("int64", "complex64"), ("bool", "bool")]:
    case(f"add-{a}-{b}", "add", [make(a, (2, 3)), make(b, (2, 3))])
    if "complex64" not in (a, b):
        case(f"greater_equal-{a}-{b}", "greater_equal", [make(a, (3, 1)), make(b, (1, 3))])
case("multiply-broadcast-3d", "multiply", [make("float32", (2, 1, 4)), make("float32", (3, 1))])
case("not_equal-rev", "not_equal", [make("int16", (3, 4)), make("int16", (3, 4))], pre="r")

# Unary.
for dt in FLOATS + ["int32"]:
    for op in ["exp", "log", "sqrt", "sin", "cos", "tanh", "sigmoid", "abs", "negative"]:
        case(f"{op}-{dt}", op, [make(dt, (4, 3), edge=dt in FLOATS)])
for dt in ["int8", "uint16", "complex64"]:
    case(f"abs-{dt}", "abs", [make(dt, (6,), edge=True)])
    case(f"negative-{dt}", "negative", [make(dt, (6,), edge=True)])
case("exp-t", "exp", [make("float32", (3, 5))], pre="t")

# where.
case("where-broadcast", "where", [make("bool", (3, 1)), make("float32", (3, 4)), make("int32", (4,))])
case("where-int-cond", "where", [ints("int32", (2, 3), -1, 2), make("float16", (2, 3)), make("float16", ())])

# astype: edge values to every dtype. Values stay inside every target's range: MLX's
# out-of-range float->int casts are C++ undefined behaviour (nuvec saturates).
EDGE_F32 = mx.array(np.array([0.0, -0.0, 1.5, -1.5, 2.5, 100.25, -100.75, 127.0], dtype=np.float32))
for dt in ALL:
    case(f"astype-float32-{dt}", "astype", [EDGE_F32], dtype=CODE[dt])
    case(f"astype-int32-{dt}", "astype", [ints("int32", (8,), -300, 300)], dtype=CODE[dt])
    case(f"astype-bool-{dt}", "astype", [make("bool", (4,))], dtype=CODE[dt])
for src in ["uint64", "int64", "float64", "complex64", "bfloat16", "float16"]:
    for dt in ["float32", "int32", "bool"]:
        case(f"astype-{src}-{dt}", "astype", [make(src, (6,))], dtype=CODE[dt])

# Views.
case("reshape", "reshape", [make("float32", (2, 3))], shape="3,2")
case("reshape-infer", "reshape", [make("int16", (2, 3, 4))], shape="-1,4")
case("reshape-copy", "reshape", [make("float32", (4, 3))], shape="12", pre="t")
case("transpose-default", "transpose", [make("uint8", (2, 3, 4))])
case("transpose-axes", "transpose", [make("float64", (2, 3, 4))], axes="1,2,0")
case("slice-basic", "slice", [make("float32", (5, 6))], args="1:4:1;::2")
case("slice-neg", "slice", [make("int32", (7, 4))], args="::-2;-1::-1")
case("slice-clamp", "slice", [make("float32", (5,))], args="-100:100:3")
case("slice-empty", "slice", [make("float32", (5,))], args="3:1:1")
case("slice-t", "slice", [make("float32", (4, 6))], args="1::2;::-1", pre="t")
case("broadcast_to", "broadcast_to", [make("float32", (3, 1))], shape="2,3,4")
case("expand_dims", "expand_dims", [make("float32", (2, 3))], axis=-1)
case("squeeze-all", "squeeze", [make("float32", (1, 3, 1))])
case("squeeze-axis", "squeeze", [make("float32", (1, 3, 1))], axes="0")

# Creation.
case("arange-int", "arange", [], start=0, stop=10, step=3, dtype=CODE["int32"])
case("arange-neg", "arange", [], start=1, stop=0, step=-0.25, dtype=CODE["float32"])
# Exact steps only: MLX accumulates `start += step` in the target type, nuvec computes
# `start + i*step` once, so inexact steps differ in the last bits.
case("arange-f16", "arange", [], start=-2, stop=2, step=0.25, dtype=CODE["float16"])
case("arange-empty", "arange", [], start=5, stop=1, step=1, dtype=CODE["float32"])
case("zeros-bf16", "zeros", [], shape="2,3", dtype=CODE["bfloat16"])
case("ones-c64", "ones", [], shape="3", dtype=CODE["complex64"])

# Reductions.
for dt in ["int8", "uint16", "int64", "bool", "float16", "float32", "float64"]:
    x = make(dt, (3, 4, 5))
    for op in ["sum", "prod", "max", "min", "mean", "all", "any"]:
        if op == "prod" and dt in FLOATS:
            x_p = make(dt, (3, 4, 5)) * 0.1 + 1 if dt != "bool" else x
            case(f"{op}-{dt}-all", op, [x_p.astype(MX[dt])])
            continue
        case(f"{op}-{dt}-all", op, [x])
        case(f"{op}-{dt}-axes", op, [x], axes="0,2", keepdims=1)
        case(f"{op}-{dt}-last", op, [x], axes="-1", keepdims=0)
case("sum-f32-nan", "sum", [make("float32", (4, 3), edge=True)], axes="0", keepdims=0)
case("max-f32-nan", "max", [make("float32", (4, 3), edge=True)], axes="1", keepdims=1)
case("sum-t", "sum", [make("float32", (5, 4))], axes="1", keepdims=0, pre="t")
case("mean-int32", "mean", [ints("int32", (4, 6), -50, 50)], axes="1", keepdims=0)
case("sum-c64", "sum", [make("complex64", (3, 3))], axes="0", keepdims=0)

# argmax / argmin with ties and NaN.
for op in ["argmax", "argmin"]:
    case(f"{op}-ties", op, [ints("int32", (4, 6), 0, 3)], axis=1, keepdims=0)
    case(f"{op}-flat", op, [make("float32", (3, 4))], keepdims=0)
    case(f"{op}-keep", op, [make("uint8", (3, 4))], axis=0, keepdims=1)
    case(f"{op}-nan", op, [make("float32", (2, 5), edge=True)], axis=1, keepdims=0)

# matmul.
case("matmul-2d", "matmul", [make("float32", (4, 5)), make("float32", (5, 3))])
case("matmul-batch", "matmul", [make("float32", (2, 1, 3, 4)), make("float32", (3, 4, 2))])
case("matmul-vec-left", "matmul", [make("float32", (5,)), make("float32", (5, 3))])
case("matmul-vec-right", "matmul", [make("float64", (4, 5)), make("float64", (5,))])
case("matmul-t", "matmul", [make("float32", (6, 4)), make("float32", (6, 3))], pre="t")
case("matmul-f16", "matmul", [make("float16", (3, 8)), make("float16", (8, 2))])
case("matmul-bf16", "matmul", [make("bfloat16", (3, 8)), make("bfloat16", (8, 2))])
case("matmul-c64", "matmul", [make("complex64", (2, 3)), make("complex64", (3, 2))])
case("matmul-mixed", "matmul", [make("int32", (2, 3)), make("float32", (3, 2))])

# Python scalars are weak: they adapt to the array operand (on either side).
SCALAR_CASES = [
    ("multiply", "float16", "float:2.5", "right"),
    ("add", "bfloat16", "int:1", "right"),
    ("subtract", "float32", "float:1.0", "left"),
    ("divide", "int32", "int:1", "left"),
    ("divide", "float16", "int:3", "right"),
    ("add", "int8", "int:100", "right"),
    ("add", "int16", "int:-32768", "right"),
    ("multiply", "uint8", "int:3", "left"),
    ("subtract", "uint64", "int:4000000000", "left"),
    ("subtract", "int64", f"int:{1 << 40}", "right"),
    ("add", "bool", "int:1", "right"),
    ("add", "bool", "bool:True", "right"),
    ("multiply", "int32", "float:0.5", "right"),
    ("add", "float64", "float:0.1", "right"),
    ("multiply", "float32", "complex:0.0,1.0", "right"),
    ("maximum", "float16", "float:0.0", "right"),
    ("minimum", "float32", "float:1.0", "left"),
    ("greater", "int32", "int:0", "right"),
    ("less_equal", "uint16", "int:7", "left"),
    ("equal", "float32", "float:2.0", "right"),
    ("not_equal", "bool", "bool:False", "left"),
]
for op, dt, sc, side in SCALAR_CASES:
    case(f"{op}-{dt}-scalar-{side}", op, [make(dt, (2, 3), edge=dt in FLOATS)], scalar=sc, side=side)

# Derivatives: `vjp` cases record (primals..., cotangent) -> one gradient per primal;
# `jvp` cases record (primals..., tangents...) -> the output tangent. `fn` names the op.
# `consts` are further inputs that are not differentiated; they are recorded after the
# primals (and before the cotangent or tangents) and passed to `fn` after the primals.
def derivative(kind: str, name: str, fn: str, primals: list[mx.array],
               consts: list[mx.array] | None = None, **params: object) -> None:
    consts = consts or []
    p = {"fn": fn, **{k: str(v) for k, v in params.items()}}
    if consts:
        p["consts"] = str(len(consts))
    spelled = kind + "".join(f"|{k}={v}" for k, v in p.items())

    def f(*xs: mx.array) -> list[mx.array]:
        return [apply(fn, p, list(xs) + consts)]

    out = f(*primals)[0]
    if kind == "vjp":
        extra = [make(NAME_OF[out.dtype], out.shape)]
        _, results = mx.vjp(f, primals, extra)
    else:
        extra = [make(NAME_OF[x.dtype], x.shape) for x in primals]
        _, results = mx.jvp(f, primals, extra)
    mx.eval(results)
    cases.append((name, spelled, primals + consts + extra, list(results)))


def positive(dtype: str, shape: tuple[int, ...]) -> mx.array:
    return (mx.abs(make("float32", shape)) + 0.5).astype(MX[dtype])


for op in ["negative", "abs", "exp", "log", "sqrt", "sin", "cos", "tanh", "sigmoid"]:
    x = positive("float32", (2, 3)) if op in ("log", "sqrt") else make("float32", (2, 3))
    derivative("vjp", f"vjp-{op}", op, [x])
derivative("vjp", "vjp-exp-f16", "exp", [make("float16", (5,))])
derivative("vjp", "vjp-tanh-f64", "tanh", [make("float64", (4,))])
for op in ["add", "subtract", "multiply", "divide", "maximum", "minimum"]:
    derivative("vjp", f"vjp-{op}-broadcast", op, [make("float32", (2, 3)), positive("float32", (3,))])
tie = ints("float32", (6,), -2, 3)
derivative("vjp", "vjp-logaddexp", "logaddexp", [make("float32", (2, 3)), make("float32", (3,))])
derivative("jvp", "jvp-logaddexp", "logaddexp", [make("float32", (4,)), make("float32", (4,))])
derivative("vjp", "vjp-maximum-ties", "maximum", [tie, ints("float32", (6,), -2, 3)])
derivative("vjp", "vjp-minimum-ties", "minimum", [tie, ints("float32", (6,), -2, 3)])
derivative("vjp", "vjp-multiply-t", "multiply", [make("float32", (4, 3)), make("float32", (3, 4))], pre="t")
derivative("vjp", "vjp-where", "where", [make("bool", (2, 3)), make("float32", (2, 3)), make("float32", (3,))])
derivative("vjp", "vjp-astype-f16", "astype", [make("float32", (5,))], dtype=CODE["float16"])
derivative("vjp", "vjp-reshape", "reshape", [make("float32", (2, 3))], shape="3,2")
derivative("vjp", "vjp-conv1d", "conv1d", [make("float32", (2, 8, 4)), make("float32", (6, 3, 2))],
           stride=2, padding=1, dilation=1, groups=2)
derivative("vjp", "vjp-conv1d-depthwise", "conv1d", [make("float32", (1, 7, 3)), make("float32", (3, 4, 1))],
           stride=1, padding=3, dilation=1, groups=3)
derivative("vjp", "vjp-gather_mm", "gather_mm", [make("float32", (3, 2, 5)), make("float32", (4, 5, 3))],
           [mx.array([2, 2, 0], dtype=mx.uint32), mx.array([1, 3, 1], dtype=mx.uint32)],
           idx="lr", sorted=0)
derivative("vjp", "vjp-gather_mm-sorted", "gather_mm",
           [make("float32", (6, 1, 4)), make("float32", (3, 4, 5))],
           [mx.array([0, 0, 1, 2, 2, 2], dtype=mx.uint32)], idx="r", sorted=1)
derivative("vjp", "vjp-repeat", "repeat", [make("float32", (2, 3))], repeats=2, axis=0)
derivative("vjp", "vjp-transpose", "transpose", [make("float32", (2, 3, 4))], axes="1,2,0")
derivative("vjp", "vjp-slice", "slice", [make("float32", (5, 6))], args="1:4:1;::-2")
derivative("vjp", "vjp-broadcast_to", "broadcast_to", [make("float32", (3, 1))], shape="2,3,4")
derivative("vjp", "vjp-expand_dims", "expand_dims", [make("float32", (2, 3))], axis=-1)
derivative("vjp", "vjp-squeeze", "squeeze", [make("float32", (1, 3, 1))])
derivative("vjp", "vjp-sum", "sum", [make("float32", (2, 3, 4))], axes="0,2", keepdims=0)
derivative("vjp", "vjp-mean", "mean", [make("float32", (2, 3, 4))], axes="1", keepdims=1)
derivative("vjp", "vjp-prod-zeros", "prod", [ints("float32", (4, 5), -1, 2)], axes="1", keepdims=0)
derivative("vjp", "vjp-max-ties", "max", [ints("float32", (4, 5), 0, 3)], axes="1", keepdims=0)
derivative("vjp", "vjp-min-all", "min", [ints("float32", (3, 4), 0, 3)])
derivative("vjp", "vjp-matmul", "matmul", [make("float32", (4, 5)), make("float32", (5, 3))])
derivative("vjp", "vjp-matmul-batch", "matmul", [make("float32", (2, 1, 3, 4)), make("float32", (3, 4, 2))])
derivative("jvp", "jvp-sin", "sin", [make("float32", (2, 3))])
derivative("jvp", "jvp-divide", "divide", [make("float32", (2, 3)), positive("float32", (3,))])
derivative("jvp", "jvp-where", "where", [make("bool", (2, 3)), make("float32", (2, 3)), make("float32", (3,))])
derivative("jvp", "jvp-slice", "slice", [make("float32", (5, 6))], args="::2;-1::-1")
derivative("jvp", "jvp-sum", "sum", [make("float32", (3, 4))], axes="1", keepdims=1)
derivative("jvp", "jvp-prod-zeros", "prod", [ints("float32", (4, 5), -1, 2)], axes="1", keepdims=0)
derivative("jvp", "jvp-max-ties", "max", [ints("float32", (4, 5), 0, 3)], axes="0", keepdims=0)
derivative("jvp", "jvp-matmul", "matmul", [make("float32", (3, 4)), make("float32", (4, 2))])

# vmap: `in_axes` per input ("n" for unbatched), one `out_axes`; inputs are the batched
# arrays, the output is the batched result.
def batched(name: str, fn: str, inputs: list[mx.array], in_axes: str, out_axes: int = 0,
            **params: object) -> None:
    p = {"fn": fn, "in_axes": in_axes, "out_axes": str(out_axes),
         **{k: str(v) for k, v in params.items()}}
    spelled = "vmap" + "".join(f"|{k}={v}" for k, v in p.items())
    axes = tuple(None if a == "n" else int(a) for a in in_axes.split(","))
    out = mx.vmap(lambda *xs: apply(fn, p, list(xs)), in_axes=axes, out_axes=out_axes)(*inputs)
    mx.eval(out)
    cases.append((name, spelled, list(inputs), [out]))


batched("vmap-sin", "sin", [make("float32", (2, 3, 4))], "1")
batched("vmap-exp-out1", "exp", [make("float32", (3, 2))], "0", out_axes=1)
batched("vmap-abs-neg", "abs", [make("float32", (2, 3))], "-1")
batched("vmap-add-shared", "add", [make("float32", (3, 4)), make("float32", (4,))], "0,n")
batched("vmap-multiply-axes", "multiply", [make("float32", (2, 3)), make("float32", (3, 2))], "1,0")
batched("vmap-multiply-f16", "multiply", [make("float16", (3, 4)), make("float16", (3, 4))], "0,0")
batched("vmap-divide", "divide", [make("float32", (3, 2)), positive("float32", (2, 3))], "0,1", out_axes=1)
batched("vmap-logaddexp", "logaddexp", [make("float32", (4,)), make("float32", (4, 2))], "n,1")
batched("vmap-maximum", "maximum", [make("float32", (4,)), make("float32", (4, 2))], "n,1")
batched("vmap-less", "less", [make("float32", (3, 2)), make("float32", (3, 2))], "0,0")
batched("vmap-where", "where", [make("bool", (2, 3)), make("float32", (4, 2, 3)), make("float32", (3,))], "n,0,n")
batched("vmap-astype", "astype", [make("float32", (3, 4))], "0", dtype=CODE["float16"])
batched("vmap-reshape", "reshape", [make("float32", (2, 3, 4))], "1", shape="4,2")
batched("vmap-transpose", "transpose", [make("float32", (2, 3, 4, 5))], "2", axes="2,0,1")
batched("vmap-slice", "slice", [make("float32", (3, 5, 6))], "0", args="1:4:1;::-2")
batched("vmap-broadcast_to", "broadcast_to", [make("float32", (3, 1, 2))], "2", shape="2,3,4")
batched("vmap-expand_dims", "expand_dims", [make("float32", (2, 3))], "1", axis=0)
batched("vmap-squeeze", "squeeze", [make("float32", (1, 3, 2, 1))], "2")
batched("vmap-sum", "sum", [make("float32", (2, 3, 4))], "1", axes="1", keepdims=0)
batched("vmap-prod-keep", "prod", [make("float32", (3, 2, 4))], "0", axes="0", keepdims=1)
batched("vmap-max-all", "max", [make("float32", (3, 4, 2))], "2")
batched("vmap-argmax", "argmax", [make("float32", (2, 5, 3))], "2", axis=1, keepdims=0)
batched("vmap-matmul", "matmul", [make("float32", (2, 3, 4)), make("float32", (2, 4, 5))], "0,0")
batched("vmap-matmul-shared-left", "matmul", [make("float32", (3, 4)), make("float32", (4, 2, 5))], "n,1")
batched("vmap-matmul-shared-right", "matmul", [make("float32", (3, 2, 4)), make("float32", (4, 5))], "1,n")
batched("vmap-matmul-vector", "matmul", [make("float32", (4, 3)), make("float32", (4, 2))], "1,n")

# Complex numbers: conjugation, and MLX's conjugation convention for derivatives.
case("conjugate-complex64", "conjugate", [make("complex64", (2, 3))])
case("conjugate-float32", "conjugate", [make("float32", (4,), edge=True)])
case("conjugate-t", "conjugate", [make("complex64", (3, 2))], pre="t")
c = lambda *s: make("complex64", s)
derivative("vjp", "vjp-c-negative", "negative", [c(2, 3)])
derivative("vjp", "vjp-c-conjugate", "conjugate", [c(2, 3)])
derivative("vjp", "vjp-c-abs", "abs", [c(2, 3)])
derivative("jvp", "jvp-c-abs", "abs", [c(2, 3)])
for op in ["add", "subtract", "multiply", "divide"]:
    derivative("vjp", f"vjp-c-{op}", op, [c(2, 3), c(3)])
    derivative("jvp", f"jvp-c-{op}", op, [c(2, 3), c(3)])
derivative("vjp", "vjp-c-multiply-real", "multiply", [c(2, 3), make("float32", (3,))])
derivative("vjp", "vjp-c-where", "where", [make("bool", (2, 3)), c(2, 3), c(3)])
derivative("vjp", "vjp-c-astype-real", "astype", [c(4)], dtype=CODE["float32"])
derivative("vjp", "vjp-c-astype-complex", "astype", [make("float32", (4,))], dtype=CODE["complex64"])
derivative("vjp", "vjp-c-slice", "slice", [c(5, 4)], args="1:4:1;::-2")
derivative("vjp", "vjp-c-sum", "sum", [c(2, 3)], axes="1", keepdims=0)
derivative("vjp", "vjp-c-prod", "prod", [c(2, 3)], axes="1", keepdims=0)
derivative("jvp", "jvp-c-prod", "prod", [c(2, 3)], axes="1", keepdims=0)
derivative("vjp", "vjp-c-matmul", "matmul", [c(2, 3), c(3, 2)])
derivative("jvp", "jvp-c-matmul", "matmul", [c(2, 3), c(3, 2)])
batched("vmap-c-conjugate", "conjugate", [c(3, 2)], "1")

# erfinv, logsumexp, softmax.
case("erfinv-f32", "erfinv", [mx.array(np.linspace(-0.999, 0.999, 41, dtype=np.float32))])
case("erfinv-edge", "erfinv", [mx.array(np.array([-1, 1, 0, -0.0, 0.9999999, 2.0], dtype=np.float32))])
case("erfinv-f16", "erfinv", [mx.array(np.linspace(-0.99, 0.99, 9, dtype=np.float32)).astype(mx.float16)])
case("logsumexp-all", "logsumexp", [make("float32", (3, 4))])
case("logsumexp-axis", "logsumexp", [make("float32", (3, 4, 5))], axes="1", keepdims=0)
case("logsumexp-keep", "logsumexp", [make("float32", (3, 4))], axes="-1", keepdims=1)
case("logsumexp-inf", "logsumexp",
     [mx.array(np.array([[-np.inf, -np.inf], [np.inf, 1.0], [1000.0, 1000.0]], dtype=np.float32))],
     axes="1", keepdims=0)
case("logsumexp-int", "logsumexp", [ints("int32", (2, 5), -3, 4)], axes="1", keepdims=0)
case("softmax-last", "softmax", [make("float32", (3, 5))], axes="-1")
case("softmax-axes", "softmax", [make("float32", (2, 3, 4))], axes="0,2")
case("softmax-f16", "softmax", [make("float16", (4, 6))], axes="1")
derivative("vjp", "vjp-erfinv", "erfinv", [mx.array(np.linspace(-0.9, 0.9, 7, dtype=np.float32))])
derivative("vjp", "vjp-logsumexp", "logsumexp", [make("float32", (3, 4))], axes="1", keepdims=0)
derivative("jvp", "jvp-logsumexp", "logsumexp", [make("float32", (3, 4))], axes="1", keepdims=0)
derivative("vjp", "vjp-softmax", "softmax", [make("float32", (3, 4))], axes="1")
derivative("jvp", "jvp-softmax", "softmax", [make("float32", (3, 4))], axes="1")
batched("vmap-softmax", "softmax", [make("float32", (3, 2, 4))], "1", axes="-1")
batched("vmap-logsumexp", "logsumexp", [make("float32", (3, 4))], "1", axes="0", keepdims=0)

# Random numbers: bit for bit from explicit keys and from the seeded per-thread sequence.
# (`bits` is not exposed in Python; uniform samples are exact functions of 32-bit bits, and
# odd sizes exercise the odd-word tail of the layout.)
for seed in [0, 1, (1 << 40) + 5]:
    for shape in ["1", "4", "5", "7", "2,3"]:
        case(f"random-bits-{seed}-{shape}", "random", [], fn="uniform", seed=seed, shape=shape,
             low=0.0, high=1.0, dtype=CODE["float32"])
case("random-split-2", "random", [], fn="split", seed=0, num=2)
case("random-split-5", "random", [], fn="split", seed=1234, num=5)
for dt in ["float32", "float16", "bfloat16"]:
    case(f"random-uniform-{dt}", "random", [], fn="uniform", seed=7, shape="3,4", low=-2.0, high=3.0,
         dtype=CODE[dt])
    case(f"random-normal-{dt}", "random", [], fn="normal", seed=11, shape="64", dtype=CODE[dt])
case("random-normal-loc-scale", "random", [], fn="normal", seed=3, shape="2,8", loc=1.5, scale=0.25,
     dtype=CODE["float32"])
case("random-seq-first", "random", [], fn="uniform", seq=1, seed=0, shape="5", low=0.0, high=1.0,
     dtype=CODE["float32"])
case("random-seq-third", "random", [], fn="uniform", seq=1, seed=0, skip=2, shape="5", low=0.0,
     high=1.0, dtype=CODE["float32"])

for dt in ["float32", "bfloat16"]:
    case(f"random-gumbel-{dt}", "random", [], fn="gumbel", seed=5, shape="3,7", dtype=CODE[dt])
case("random-categorical", "random", [make("float32", (4, 9))], fn="categorical", seed=9, axis=-1)
case("random-categorical-axis0", "random", [make("bfloat16", (6, 3))], fn="categorical", seed=2,
     axis=0)
for dt in ["int32", "uint8", "int64"]:
    case(f"random-randint-{dt}", "random", [], fn="randint", seed=4, shape="10", low=-3 if dt != "uint8" else 2,
         high=17, dtype=CODE[dt])
case("random-bernoulli", "random", [], fn="bernoulli", seed=6, shape="2,9", p=0.3)
# conv1d: input [N, L, C_in], weight [C_out, K, C_in / groups].
for name, (n, l, ci, co, k, st, pad, dil, g) in {
    "basic": (2, 9, 3, 4, 3, 1, 0, 1, 1),
    "stride-pad": (1, 10, 4, 2, 3, 2, 1, 1, 1),
    "dilated": (1, 12, 2, 3, 3, 1, 2, 2, 1),
    "groups": (2, 8, 6, 4, 2, 1, 0, 1, 2),
    "depthwise": (1, 11, 5, 5, 4, 1, 3, 1, 5),
}.items():
    for dt in ["float32", "bfloat16"]:
        case(f"conv1d-{name}-{dt}", "conv1d", [make(dt, (n, l, ci)), make(dt, (co, k, ci // g))],
             stride=st, padding=pad, dilation=dil, groups=g)
# gather_mm: a [..., M, K], b [..., K, N], indices into their flattened batches (MLX's CPU
# kernel supports float32 only).
for dt in ["float32"]:
    case(f"gather_mm-lr-{dt}", "gather_mm", [make(dt, (2, 3, 3, 8)), make(dt, (5, 8, 4)),
         ints("uint32", (2, 1), 0, 6), ints("uint32", (1, 3), 0, 5)], idx="lr", sorted=0)
    case(f"gather_mm-r-{dt}", "gather_mm", [make(dt, (1, 4, 8)), make(dt, (3, 8, 6)),
         ints("uint32", (5,), 0, 3)], idx="r", sorted=0)
    case(f"gather_mm-sorted-{dt}", "gather_mm", [make(dt, (7, 1, 16)), make(dt, (4, 16, 9)),
         mx.array([0, 1, 1, 1, 3, 3, 3], dtype=mx.uint32)], idx="r", sorted=1)
case("gather_mm-default", "gather_mm", [make("float32", (3, 2, 4)), make("float32", (3, 4, 2))],
     idx="", sorted=0)
case("gather_mm-tb", "gather_mm", [make("float32", (4, 2, 6)), make("float32", (3, 5, 6)),
     ints("uint32", (4,), 0, 3)], idx="r", sorted=0, tb=1)
# One row per output batch (nuvec's `gather_mv`): unsorted, sorted, transposed b, and a
# lhs gather over batched rows.
case("gather_mv-r", "gather_mm", [make("float32", (16, 1, 64)), make("float32", (8, 64, 40)),
     ints("uint32", (16,), 0, 8)], idx="r", sorted=0)
case("gather_mv-sorted", "gather_mm", [make("float32", (12, 1, 48)), make("float32", (5, 48, 33)),
     mx.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4], dtype=mx.uint32)], idx="r", sorted=1)
case("gather_mv-tb", "gather_mm", [make("float32", (9, 1, 40)), make("float32", (6, 2100, 40)),
     ints("uint32", (9,), 0, 6)], idx="r", sorted=0, tb=1)
case("gather_mv-lr", "gather_mm", [make("float32", (3, 4, 1, 32)), make("float32", (5, 32, 7)),
     ints("uint32", (6,), 0, 12), ints("uint32", (6,), 0, 5)], idx="lr", sorted=0)
derivative("vjp", "vjp-gather_mv", "gather_mm", [make("float32", (6, 1, 8)), make("float32", (3, 8, 5))],
           [mx.array([2, 0, 1, 2, 2, 0], dtype=mx.uint32)], idx="r", sorted=0)
case("repeat-axis", "repeat", [make("float32", (2, 3))], repeats=3, axis=1)
case("repeat-flat", "repeat", [make("int32", (2, 2))], repeats=2, axis="")
case("contiguous-transposed", "contiguous", [make("float32", (3, 4))], pre="t")

# addmm: alpha * (a @ b) + beta * c, with c broadcast (inputs: a, b, c).
case("addmm-bias", "addmm", [make("float32", (5, 7)), make("float32", (7, 3)), make("float32", (3,))],
     alpha=1.0, beta=1.0)
case("addmm-axpby", "addmm", [make("float32", (4, 6)), make("float32", (6, 5)), make("float32", (4, 5))],
     alpha=0.5, beta=-2.0)
case("addmm-batch", "addmm", [make("float32", (2, 3, 4)), make("float32", (4, 5)), make("float32", (5,))],
     alpha=2.0, beta=1.0)
case("addmm-vector", "addmm", [make("float32", (4,)), make("float32", (4, 3)), make("float32", (3,))],
     alpha=1.0, beta=0.5)
case("addmm-f16", "addmm", [make("float16", (8, 16)), make("float16", (16, 4)), make("float16", (4,))],
     alpha=1.0, beta=1.0)
case("addmm-t", "addmm", [make("float32", (6, 4)), make("float32", (6, 3)), make("float32", (3,))],
     alpha=1.0, beta=1.0, pre="t")
derivative("vjp", "vjp-addmm", "addmm", [make("float32", (3, 4)), make("float32", (4, 2)), make("float32", (2,))],
           alpha=1.5, beta=0.5)
# alpha = beta = 1: MLX 0.32.4's AddMM jvp ignores alpha and beta (a bug; its vjp is
# correct, and nuvec keeps `alpha (dA B + A dB) + beta dC`, see the decision log).
derivative("jvp", "jvp-addmm", "addmm", [make("float32", (3, 4)), make("float32", (4, 2)), make("float32", (3, 2))],
           alpha=1.0, beta=1.0)
batched("vmap-addmm", "addmm", [make("float32", (3, 2, 4)), make("float32", (4, 5)), make("float32", (5,))],
        "0,n,n", alpha=1.0, beta=1.0)

# Affine quantization: bit-exact codes, scales and biases for every dtype, group size and
# bit width; groups that are constant, one-signed or contain zero.
QBITS = [1, 2, 3, 4, 5, 6, 8]
for dt in ["float32", "float16", "bfloat16"]:
    for gs in [32, 64, 128]:
        for bits in QBITS:
            case(f"quantize-{dt}-g{gs}-b{bits}", "quantize", [make(dt, (3, 2 * gs))],
                 group_size=gs, bits=bits)
EDGE_Q = np.concatenate([np.full(32, 0.75), -1 - np.arange(32) / 8, np.arange(32) / 31 - 0.25,
                         np.arange(32) * 3.0]).astype(np.float32).reshape(4, 32)
for bits in QBITS:
    case(f"quantize-edges-b{bits}", "quantize", [mx.array(EDGE_Q)], group_size=32, bits=bits)


def quantized(dt: str, shape: tuple[int, ...], gs: int, bits: int) -> list[mx.array]:
    wq, s, b = mx.quantize(make(dt, shape), group_size=gs, bits=bits)
    mx.eval(wq, s, b)
    return [wq, s, b]


for dt, gs in [("float32", 32), ("float16", 64), ("bfloat16", 128)]:
    for bits in QBITS:
        case(f"dequantize-{dt}-g{gs}-b{bits}", "dequantize", quantized(dt, (2, 2 * gs), gs, bits),
             group_size=gs, bits=bits)

# quantized_matmul: (x, w, scales, biases).
for name, xs, ws, t, gs, bits, dt in [
    ("qmm-vec", (64,), (8, 64), 1, 32, 4, "float32"),
    ("qmm-rows", (5, 128), (12, 128), 1, 64, 4, "float32"),
    ("qmm-3bit", (3, 96), (6, 96), 1, 32, 3, "float32"),
    ("qmm-8bit-f16", (4, 128), (16, 128), 1, 128, 8, "float16"),
    ("qmm-n", (3, 16), (16, 64), 0, 32, 4, "float32"),
    ("qmm-n-6bit", (2, 32), (32, 128), 0, 64, 6, "float32"),
    ("qmm-batch", (2, 3, 64), (2, 10, 64), 1, 32, 2, "float32"),
    ("qmm-shared", (2, 3, 64), (10, 64), 1, 64, 5, "bfloat16"),
]:
    case(name, "quantized_matmul", [make(dt, xs)] + quantized(dt, ws, gs, bits),
         transpose=t, group_size=gs, bits=bits)
# Few-row products (2–16 rows, nuvec's multi-row kernels) for every bit width.
for i, bits in enumerate([1, 2, 3, 5, 6, 8]):
    for rows in [2, 7, 16]:
        dt, gs = [("float16", 32), ("bfloat16", 64), ("float16", 128)][(i + rows) % 3]
        case(f"qmm-rows{rows}-b{bits}-{dt}", "quantized_matmul",
             [make(dt, (rows, 512))] + quantized(dt, (72, 512), gs, bits),
             transpose=1, group_size=gs, bits=bits)
        case(f"qmm-n-rows{rows}-b{bits}-{dt}", "quantized_matmul",
             [make(dt, (rows, 256))] + quantized(dt, (256, 128), gs, bits),
             transpose=0, group_size=gs, bits=bits)
wq, s, b = quantized("float32", (6, 64), 32, 4)
derivative("vjp", "vjp-qmm", "quantized_matmul_xsb", [make("float32", (3, 64)), s, b], [wq],
           transpose=1, group_size=32, bits=4)
wq, s, b = quantized("float32", (32, 64), 64, 4)
derivative("vjp", "vjp-qmm-n", "quantized_matmul_xsb", [make("float32", (2, 32)), s, b], [wq],
           transpose=0, group_size=64, bits=4)
wq, s, b = quantized("float32", (6, 64), 32, 8)
derivative("jvp", "jvp-qmm", "quantized_matmul", [make("float32", (2, 3, 64))], [wq, s, b],
           transpose=1, group_size=32, bits=8)

# Fused operations (`mx.fast`; MLX's CPU runs their fallback graphs).
for dt in ["float32", "float16", "bfloat16"]:
    case(f"rms_norm-{dt}", "rms_norm", [make(dt, (3, 4, 16)), make(dt, (16,))], eps=1e-5)
    case(f"rms_norm-bare-{dt}", "rms_norm", [make(dt, (5, 33))], eps=1e-6)
    case(f"layer_norm-{dt}", "layer_norm", [make(dt, (3, 24)), make(dt, (24,)), make(dt, (24,))],
         eps=1e-5)
    case(f"layer_norm-bare-{dt}", "layer_norm", [make(dt, (2, 3, 7))], eps=1e-5)
    case(f"rope-{dt}", "rope", [make(dt, (2, 3, 5, 8)), mx.array(3, dtype=mx.int32)], dims=8,
         traditional=0, base=10000.0, scale=1.0)
    case(f"rope-trad-{dt}", "rope", [make(dt, (2, 4, 12)), mx.array([1, 7], dtype=mx.int32)],
         dims=8, traditional=1, base=500.0, scale=0.5)
    case(f"rope-freqs-{dt}", "rope", [make(dt, (1, 2, 3, 8)), mx.array(0, dtype=mx.int32),
                                      mx.array([1.0, 4.0, 9.0, 30.0])], dims=8, traditional=0,
         scale=1.0)
    case(f"cross_entropy-{dt}", "cross_entropy", [make(dt, (4, 11)), ints("int32", (4,), -11, 11)])
derivative("vjp", "vjp-rms_norm", "rms_norm", [make("float32", (3, 16)), make("float32", (16,))],
           eps=1e-5)
derivative("vjp", "vjp-layer_norm", "layer_norm",
           [make("float32", (3, 16)), make("float32", (16,)), make("float32", (16,))], eps=1e-5)
derivative("vjp", "vjp-rope", "rope", [make("float32", (2, 3, 4, 8))],
           [mx.array(2, dtype=mx.int32)], dims=8, traditional=0, base=10000.0, scale=1.0)
derivative("vjp", "vjp-cross_entropy", "cross_entropy", [make("float32", (5, 7))],
           [ints("int32", (5,), 0, 7)])
# Scaled dot-product attention: (q, k, v[, mask][, sinks]). Inputs are N(0, 16): the
# scales keep the scores about N(0, 1), away from float32's rounding amplification.
for dt in ["float32", "float16"]:
    case(f"sdpa-{dt}", "sdpa", [make(dt, (2, 4, 6, 16)), make(dt, (2, 4, 9, 16)),
                                make(dt, (2, 4, 9, 16))], scale=0.0625)
    case(f"sdpa-causal-gqa-{dt}", "sdpa", [make(dt, (1, 8, 5, 8)), make(dt, (1, 2, 7, 8)),
                                           make(dt, (1, 2, 7, 12))], scale=0.088, mask="causal")
    case(f"sdpa-bool-{dt}", "sdpa", [make(dt, (1, 2, 3, 8)), make(dt, (1, 2, 4, 8)),
                                     make(dt, (1, 2, 4, 8)), make("bool", (3, 4))], scale=0.088,
         mask="array")
    case(f"sdpa-additive-sinks-{dt}", "sdpa",
         [make(dt, (1, 4, 2, 8)), make(dt, (1, 2, 5, 8)), make(dt, (1, 2, 5, 8)),
          make(dt, (1, 4, 1, 5)), make(dt, (4,))], scale=0.088, mask="array")
# Capped attention (the composition in query blocks, or the tiled kernel) and sliding
# windows, replayed against MLX's uncapped attention; sliced key caches and transposed
# queries are built from these in tests/mlx.rs.
for dt in ["float32", "float16", "bfloat16"]:
    for (b, hq, hkv, lq, lk, d) in [(1, 4, 2, 33, 45, 64), (2, 2, 2, 20, 20, 80), (1, 8, 1, 18, 60, 256)]:
        row = b * hq * lk * (4 if dt == "float32" else 2)
        # Inputs are N(0, 16): scale the scores to about N(0, 1).
        sc = round(1 / (16 * d ** 0.5), 6)
        shapes = [make(dt, (b, hq, lq, d)), make(dt, (b, hkv, lk, d)), make(dt, (b, hkv, lk, d))]
        for blocks in [1, 2, 3, -(-lq // 9)]:
            cap = row * -(-lq // blocks)
            case(f"sdpa-cap{blocks}-{dt}-{lq}x{lk}x{d}", "sdpa", shapes, scale=sc, mask="causal",
                 cap=cap)
        case(f"sdpa-cap-sinks-{dt}-{lq}x{lk}x{d}", "sdpa", shapes + [make(dt, (hq,))], scale=sc,
             mask="causal", cap=b * hq * lk * 4 * 4)
        case(f"sdpa-cap-bool-{dt}-{lq}x{lk}x{d}", "sdpa", shapes + [make("bool", (lq, lk))],
             scale=sc, mask="array", cap=row * 5)
        case(f"sdpa-window-{dt}-{lq}x{lk}x{d}", "sdpa", shapes, scale=sc, mask="causal",
             window=7)
        case(f"sdpa-window-cap-{dt}-{lq}x{lk}x{d}", "sdpa", shapes + [make(dt, (hq,))],
             scale=sc, mask="causal", window=7, cap=b * hq * 12 * 4 * 3)
# Shapes the GPU kernels take (vector for one query, tiled for 40), replayed on both devices.
case("sdpa-vector-kernel", "sdpa", [make("float32", (1, 4, 1, 64)), make("float32", (1, 2, 40, 64)),
                                    make("float32", (1, 2, 40, 64))], scale=0.125)
case("sdpa-full-kernel", "sdpa", [make("float16", (1, 2, 40, 64)), make("float16", (1, 2, 40, 64)),
                                  make("float16", (1, 2, 40, 64))], scale=0.125, mask="causal")
derivative("vjp", "vjp-sdpa", "sdpa", [make("float32", (1, 2, 4, 8)), make("float32", (1, 2, 5, 8)),
                                       make("float32", (1, 2, 5, 8))], scale=0.0625, mask="causal")
derivative("jvp", "jvp-rms_norm", "rms_norm", [make("float32", (2, 8)), make("float32", (8,))],
           eps=1e-5)

# Indexing and joining: negative indices, broadcasting, every element size.
for dt in ["float32", "float16", "bfloat16", "int64", "bool", "complex64", "uint8"]:
    case(f"take-{dt}", "take", [make(dt, (5, 4)), ints("int32", (3,), -5, 5)], axis=0)
case("take-axis1-2d", "take", [make("float32", (3, 6)), ints("uint8", (2, 2), 0, 6)], axis=1)
case("take-flat", "take", [make("float32", (3, 4)), ints("int64", (5,), -12, 12)])
case("take-scalar", "take", [make("float32", (4, 3)), mx.array(2, dtype=mx.int32)], axis=0)
case("take-t", "take", [make("float32", (4, 5)), ints("int16", (3,), 0, 4)], axis=1, pre="t")
case("take_along_axis", "take_along_axis", [make("float32", (3, 4)), ints("int32", (3, 2), -4, 4)],
     axis=1)
case("take_along_axis-bcast", "take_along_axis",
     [make("float32", (3, 4)), ints("uint32", (1, 2), 0, 4)], axis=1)
case("take_along_axis-axis0", "take_along_axis",
     [make("int32", (4, 3)), ints("int64", (2, 3), 0, 4)], axis=0)
PERM = mx.array(rng.permutation(5)[:3].reshape(3, 1)).astype(mx.int32)  # unique rows
case("put_along_axis", "put_along_axis", [make("float32", (5, 2)), PERM, make("float32", (3, 2))],
     axis=0)
case("put_along_axis-bcast", "put_along_axis",
     [make("int16", (2, 6)), mx.array([[0, -1, 3]]).astype(mx.int32), make("int16", (3,))], axis=1)
case("concatenate", "concatenate",
     [make("float32", (2, 3)), make("float32", (4, 3)), make("float32", (1, 3))], axis=0)
case("concatenate-promote", "concatenate", [make("int32", (2, 1)), make("float16", (2, 3))],
     axis=-1)
case("concatenate-t", "concatenate", [make("float32", (3, 2)), make("float32", (2, 2))], axis=1,
     pre="t")
case("stack", "stack", [make("float32", (2, 3)), make("float32", (2, 3))], axis=-1)
case("split-parts", "split", [make("float32", (2, 6))], axis=1, parts=3)
case("split-at", "split", [make("int32", (7, 2))], axis=0, at="1,3,3")
DUP = mx.array([1, 3, 1, 1, 0]).astype(mx.int32)  # repeated rows accumulate gradients
derivative("vjp", "vjp-take-dup", "take", [make("float32", (4, 3))], consts=[DUP], axis=0)
derivative("vjp", "vjp-take_along_axis-dup", "take_along_axis", [make("float32", (3, 4))],
           consts=[mx.array([[0, 0, 2], [3, 3, 3], [1, 2, 1]]).astype(mx.int32)], axis=1)
derivative("vjp", "vjp-put_along_axis", "put_along_axis",
           [make("float32", (5, 2)), make("float32", (3, 2))], consts=[PERM], axis=0, order="avi")
derivative("vjp", "vjp-concatenate", "concatenate", [make("float32", (2, 3)), make("float32", (2, 1))],
           axis=1)
derivative("jvp", "jvp-take", "take", [make("float32", (4, 3))], consts=[DUP], axis=0)
derivative("jvp", "jvp-concatenate", "concatenate",
           [make("float32", (2, 3)), make("float32", (1, 3))], axis=0)
batched("vmap-take_along_axis", "take_along_axis",
        [make("float32", (2, 3, 4)), ints("int32", (2, 3, 2), 0, 4)], "0,0", axis=1)
batched("vmap-take-shared-table", "take", [make("float32", (5, 3)), ints("int32", (4, 2), 0, 5)],
        "n,0", axis=0)
batched("vmap-concatenate", "concatenate", [make("float32", (3, 2, 2)), make("float32", (2, 1))],
        "0,n", axis=1)

# Sorting and scans: NaN last, ties stable, every direction and inclusiveness.
SORTABLE = make("float32", (4, 9), edge=True)
for op in ["sort", "argsort"]:
    case(f"{op}-f32-nan", op, [SORTABLE], axis=1)
    case(f"{op}-ties", op, [ints("int32", (3, 10), 0, 4)], axis=1)
    case(f"{op}-axis0-t", op, [make("float16", (5, 3))], axis=0, pre="t")
    case(f"{op}-u8", op, [make("uint8", (2, 7))], axis=-1)
    case(f"{op}-bf16", op, [make("bfloat16", (6,))], axis=0)
case("topk", "topk", [make("float32", (3, 8))], axis=1, k=3)
for op in ["cumsum", "cumprod", "cummax", "cummin", "logcumsumexp"]:
    for rev in ["0", "1"]:
        for inc in ["1", "0"]:
            case(f"{op}-r{rev}-i{inc}", op, [make("float32", (3, 5))], axis=1, reverse=rev, inclusive=inc)
    case(f"{op}-axis0", op, [make("float32", (4, 3))], axis=0)
    case(f"{op}-flat", op, [make("float32", (2, 3))])
case("cumsum-int", "cumsum", [ints("int32", (2, 6), -5, 5)], axis=1)
case("cumsum-bool", "cumsum", [make("bool", (3, 4))], axis=1)
case("cumprod-int", "cumprod", [ints("int16", (2, 5), -3, 3)], axis=1)
case("cummax-nan", "cummax", [make("float32", (2, 6), edge=True)], axis=1)
case("cummin-u8", "cummin", [make("uint8", (2, 6))], axis=1)
case("cumsum-f16", "cumsum", [make("float16", (2, 40))], axis=1)
case("logcumsumexp-int", "logcumsumexp", [ints("int32", (2, 4), -3, 3)], axis=1)
derivative("vjp", "vjp-sort", "sort", [make("float32", (3, 5))], axis=1)
derivative("jvp", "jvp-sort", "sort", [make("float32", (3, 5))], axis=0)
for rev in ["0", "1"]:
    for inc in ["1", "0"]:
        derivative("vjp", f"vjp-cumsum-r{rev}-i{inc}", "cumsum", [make("float32", (2, 5))], axis=1, reverse=rev, inclusive=inc)
        derivative("vjp", f"vjp-cumprod-zeros-r{rev}-i{inc}", "cumprod", [ints("float32", (3, 6), -1, 2)], axis=1, reverse=rev, inclusive=inc)
        derivative("vjp", f"vjp-cummax-ties-r{rev}-i{inc}", "cummax", [ints("float32", (3, 6), 0, 3)], axis=1, reverse=rev, inclusive=inc)
        derivative("vjp", f"vjp-cummin-r{rev}-i{inc}", "cummin", [make("float32", (2, 5))], axis=1, reverse=rev, inclusive=inc)
        derivative("vjp", f"vjp-logcumsumexp-r{rev}-i{inc}", "logcumsumexp", [make("float32", (2, 5))], axis=1, reverse=rev, inclusive=inc)
derivative("jvp", "jvp-cumsum", "cumsum", [make("float32", (2, 5))], axis=1, reverse=1)
derivative("jvp", "jvp-logcumsumexp", "logcumsumexp", [make("float32", (2, 5))], axis=1)
batched("vmap-sort", "sort", [make("float32", (3, 2, 5))], "1", axis=1)
batched("vmap-cumsum", "cumsum", [make("float32", (2, 4, 3))], "2", axis=0, reverse=1)


# ── Array API cases: indexing, updates, var/std, empty axes, scans ──
# Written to a second file, `mlx_api.bin`, replayed by `tests/indexing.rs`, `tests/axes.rs`
# and `tests/reduce.rs`. Each case evaluates a Python expression over `x` (the first input,
# after `pre`), `a1`, `a2`, … (index arrays) and `v` (the last input for updates); the Rust
# tests hold the same expressions, keyed by the `expr` parameter.

OUT_API = OUT.with_name("mlx_api.bin")
api_cases: list[tuple[str, str, list[mx.array], list[mx.array]]] = []
api_rng = np.random.default_rng(SEED + 62)


def api_make(dtype: str, shape: tuple[int, ...]) -> mx.array:
    n = int(np.prod(shape)) if shape else 1
    if dtype == "bool":
        return mx.array(api_rng.integers(0, 2, n).astype(np.bool_).reshape(shape))
    if dtype.startswith(("uint", "int")):
        return mx.array(api_rng.integers(-50, 50, n).reshape(shape)).astype(MX[dtype])
    return mx.array((api_rng.normal(size=n) * 4).astype(np.float32).reshape(shape)).astype(MX[dtype])


def api_ints(shape: tuple[int, ...], lo: int, hi: int) -> mx.array:
    n = int(np.prod(shape)) if shape else 1
    return mx.array(api_rng.integers(lo, hi, n).reshape(shape)).astype(mx.int32)


def api_eval(kind: str, expr: str, xs: list[mx.array], pre: str | None) -> mx.array:
    env = {"mx": mx, "x": pre_view(xs[0], pre)}
    for k, a in enumerate(xs[1:], start=1):
        env[f"a{k}"] = a
    env["v"] = xs[-1]
    if kind == "set":
        target, value = expr.split("=", 1)
        y = mx.array(env["x"])
        exec(f"y{target[1:]} = {value}", env | {"y": y})
        return y
    return eval(expr, env)


def api_case(kind: str, expr: str, inputs: list[mx.array], pre: str | None = None) -> None:
    dt = NAME_OF[inputs[0].dtype]
    name = f"{kind}/{expr}/{dt}/{pre or 'c'}"
    out = api_eval(kind, expr, inputs, pre)
    mx.eval(out)
    spelled = f"{kind}|expr={expr}" + (f"|pre={pre}" if pre else "")
    api_cases.append((name, spelled, list(inputs), [out]))


API_DTYPES = ["float32", "float16", "bfloat16", "int32", "bool"]
BASIC = [
    "x[0]", "x[-1]", "x[1:3]", "x[:, 1:3]", "x[..., -1]", "x[:, None, :]", "x[::-1]",
    "x[1:, :3, 0:6:2]", "x[None]", "x[..., None]", "x[2, ..., 1:4]", "x[-3:-1, ::2]",
    "x[:, ::-2, 4]", "x[1:100]", "x[3:1]", "x[..., ::-1, None, 0]", "x[:3]", "x[1, 2, 3]",
    "x[...]", "x[:, -2:]", "x[::2, 1::2, ::-3]",
    # nuvec's stepped ranges select the range, then walk it (backwards for a negative step):
    # `(1..4).step(-1)` is `x[3:0:-1]`, `(..2).step(-1)` is `x[1::-1]`, `(1..).step(-1)` is
    # `x[:0:-1]`, `(0..4).step(-2)` is `x[3::-2]`.
    "x[3:0:-1]", "x[1::-1]", "x[:0:-1]", "x[3::-2]",
]
for dt in API_DTYPES:
    for expr in BASIC:
        for pre in (None, "t", "r"):
            api_case("index", expr, [api_make(dt, (4, 5, 6) if pre != "t" else (6, 5, 4))], pre)

# Advanced (integer arrays): MLX's placement rules, broadcasting of index arrays.
ADVANCED = [
    ("x[a1]", [(3,)]), ("x[a1]", [(2, 3)]), ("x[a1, a2]", [(3,), (3,)]),
    ("x[a1, :, a2]", [(3,), (3,)]), ("x[:, a1, a2]", [(3,), (3,)]),
    ("x[a1, a2]", [(2, 1), (3,)]), ("x[a1, 2]", [(3,)]), ("x[1, :, a1]", [(2,)]),
    ("x[a1, 1:4]", [(3,)]), ("x[..., a1]", [(2,)]), ("x[None, a1]", [(2,)]),
    ("x[a1, None]", [(2,)]), ("x[::-1, a1]", [(4,)]), ("x[a1, a2, a1]", [(2, 2), (2,)]),
]
for dt in ["float32", "int32", "float16"]:
    for expr, idx_shapes in ADVANCED:
        for pre in (None, "t"):
            x = api_make(dt, (4, 5, 6) if pre != "t" else (6, 5, 4))
            idx = [api_ints(s, -4, 4) for s in idx_shapes]
            api_case("index", expr, [x] + idx, pre)

# Functional updates: `x.at[...]` (add) and copies with `y[...] = v` (set).
UPDATES = [
    ("set", "x[:, 1:3]=v", [], (4, 2, 6)), ("set", "x[0]=v", [], (5, 6)),
    ("set", "x[..., -1]=v", [], (5,)), ("set", "x[::2, :, 1:4]=v", [], (1, 3)),
    ("set", "x[:, None, 2]=v", [], (4, 1, 6)), ("set", "x[a1]=v", [(3,)], (3, 5, 6)),
    ("set", "x[a1, :, a2]=v", [(2,), (2,)], (2, 5)),
    ("add", "x.at[1:3].add(v)", [], (5, 6)), ("add", "x.at[..., 0].add(v)", [], (4, 5)),
    ("add", "x.at[a1].add(v)", [(7,)], (7, 5, 6)), ("add", "x.at[a1, a2].add(v)", [(6,), (6,)], (6, 6)),
    ("add", "x.at[:, a1].add(v)", [(4,)], (4, 4, 6)),
]
for dt in ["float32", "int32"]:
    for kind, expr, idx_shapes, vshape in UPDATES:
        x = api_make(dt, (4, 5, 6))
        lo, hi = (-4, 4)
        idx = [api_ints(s, lo, hi) for s in idx_shapes]
        if kind == "set" and idx:  # unique indices: MLX leaves repeated `set` unspecified
            idx = [mx.array(api_rng.permutation(4)[: s[0]] - 4 * api_rng.integers(0, 2)).astype(mx.int32)
                   for s in idx_shapes]
        api_case(kind, expr, [x] + idx + [api_make(dt, vshape)])

# var/std (ddof, keepdims), empty-axes reductions (MLX C++: no axis reduced), scans.
for dt in ["float32", "float16", "int32"]:
    x = api_make(dt, (3, 4, 5))
    for expr in ["mx.var(x)", "mx.var(x, axis=-1)", "mx.var(x, axis=[0, 2], keepdims=True)",
                 "mx.var(x, axis=1, ddof=1)", "mx.std(x)", "mx.std(x, axis=0, ddof=1, keepdims=True)",
                 "mx.std(x, axis=[1, 2])", "mx.var(x, axis=-1, ddof=5)"]:
        api_case("reduce", expr, [x])
        api_case("reduce", expr, [api_make(dt, (5, 4, 3))], "t")
for dt in ["float32", "int8", "int32", "bool", "float16"]:
    x = api_make(dt, (2, 3, 4))
    for fn in ["sum", "prod", "max", "min", "mean", "all", "any", "logsumexp", "var", "std"]:
        if dt == "bool" and fn in ("logsumexp", "var", "std", "mean"):
            continue
        api_case("reduce", f"mx.{fn}(x, axis=[])", [x])
        api_case("reduce", f"mx.{fn}(x, axis=[], keepdims=True)", [x])
for dt in ["float32", "int32"]:
    x = api_make(dt, (3, 6))
    for fn in ["cumsum", "cumprod", "cummax", "cummin"] + (["logcumsumexp"] if dt == "float32" else []):
        for rev in (False, True):
            for inc in (True, False):
                api_case("reduce", f"mx.{fn}(x, axis=-1, reverse={rev}, inclusive={inc})", [x])
        api_case("reduce", f"mx.{fn}(x, reverse=True)", [x], "t")

# ── GPU cases: floating-point quantization ──────────────────────────────────
#
# MLX's Metal kernels are the reference for the microscaling modes (its CPU writes +0 where
# Metal writes -0, and rounds some scales differently), so these cases run on MLX's GPU
# stream and nuvec compares them bit for bit on both of its devices.

OUT_GPU = OUT.with_name("mlx_gpu.bin")
gpu_cases: list[tuple[str, str, list[mx.array], list[mx.array]]] = []


def case_gpu(name: str, op: str, inputs: list[mx.array], **params: object) -> None:
    p = {k: str(v) for k, v in params.items()}
    spelled = op + "".join(f"|{k}={v}" for k, v in p.items())
    with mx.stream(mx.gpu):
        out = apply(op, p, list(inputs))
        outs = out if isinstance(out, list) else [out]
        mx.eval(outs)
    gpu_cases.append((name, spelled, list(inputs), outs))


FP_MODES = {"mxfp4": 32, "mxfp8": 32, "nvfp4": 16}


def fp_edges(mode: str) -> np.ndarray:
    """Rows of 64 values (whole groups in every mode) that reach each rounding branch."""
    big = 448.0 if mode == "mxfp8" else 6.0
    # With a group maximum of `big`, the scale is exactly 1 and values encode as written.
    if mode == "mxfp8":
        ties = [1.0625, 1.1875, 0.0029296875, 0.001953125, 3.25, 240.0, 464.0 / 2, 13.0]
    else:
        ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.1]
    tied = np.array([big] + ties + [-t for t in ties] + [0.0] * (64 - 1 - 2 * len(ties)))
    rows = [
        np.zeros(64),                                    # all-zero groups: scale code 0
        -np.abs(rng.normal(size=64)) - 0.01,             # one-signed (negative) groups
        tied,                                            # exact ties at scale 1
        np.concatenate([np.full(32, 1e30), rng.normal(size=32)]),  # beyond f16's range
        np.concatenate([[np.nan], rng.normal(size=31), [np.inf], rng.normal(size=31)]),
        np.concatenate([[-np.inf], rng.normal(size=63)]),
        np.concatenate([np.array([0xFFC00000], dtype=np.uint32).view(np.float32),  # -NaN
                        rng.normal(size=63)]),
        rng.normal(size=64) * 1e-40,                     # f32 subnormals
        np.where(np.arange(64) % 2 == 0, -0.0, 1e-3),     # -0 beside tiny values
        rng.normal(size=64) * 1e5,
        rng.normal(size=64) * 1e-5,
    ]
    return np.stack(rows).astype(np.float32)


for mode, gs in FP_MODES.items():
    for dt in ["float32", "float16", "bfloat16"]:
        for shape in [(3, 2 * gs), (5, 7 * gs), (2, 3, 4 * gs)]:
            name = f"fp-quantize-{mode}-{dt}-{'x'.join(map(str, shape))}"
            case_gpu(name, "quantize", [make(dt, shape)], mode=mode)
        case_gpu(f"fp-quantize-edges-{mode}-{dt}", "quantize",
                 [mx.array(fp_edges(mode)).astype(MX[dt])], mode=mode)
        # Strided inputs: a transposed matrix, and the last axis reversed.
        case_gpu(f"fp-quantize-t-{mode}-{dt}", "quantize", [make(dt, (4 * gs, 3))], mode=mode,
                 pre="t")
        case_gpu(f"fp-quantize-r-{mode}-{dt}", "quantize", [make(dt, (3, 4 * gs))], mode=mode,
                 pre="r")
        for out in [None, "float16", "float32", "bfloat16"]:
            with mx.stream(mx.gpu):
                wq, scales = mx.quantize(make(dt, (4, 4 * gs)), mode=mode)
                mx.eval(wq, scales)
            extra = {} if out is None else {"dtype": CODE[out]}
            case_gpu(f"fp-dequantize-{mode}-{dt}-{out or 'default'}", "dequantize", [wq, scales],
                     mode=mode, **extra)
        with mx.stream(mx.gpu):
            wq, scales = mx.quantize(mx.array(fp_edges(mode)).astype(MX[dt]), mode=mode)
            mx.eval(wq, scales)
        case_gpu(f"fp-dequantize-edges-{mode}-{dt}", "dequantize", [wq, scales], mode=mode,
                 dtype=CODE["float32"])

# Products on MLX's paths (where nuvec runs MLX's kernel with MLX's dispatch): one row
# (qmv_fast, qmv at K = 2880, qmv_quad, partial rows), qmv_wide between R = 8 and MLX's
# vector limit (13 on the M4 Pro at these sizes), qmv_quad above R, split-K, batched qmm_t
# and qmv, untransposed qvm (fewer than 4 rows) and qmm_n, a 1-D x.
FP_PRODUCTS = [
    ("qmv-fast", (1, 4096), (32, 4096), True),
    ("qmv-k2880", (1, 2880), (40, 2880), True),
    ("qmv-quad128", (1, 128), (40, 128), True),
    ("qmv-quad64", (1, 64), (24, 64), True),
    ("qmv-partial", (1, 4096), (30, 4096), True),
    ("qmv-wide", (11, 1024), (40, 1024), True),
    ("qmv-wide-k2880", (10, 2880), (24, 2880), True),
    ("quad-rows", (12, 128), (64, 128), True),
    ("splitk", (40, 512), (64, 512), True),
    ("splitk-k2880", (16, 2880), (32, 2880), True),
    ("qmm-t-batched", (2, 20, 512), (2, 24, 512), True),
    ("qmv-batched", (2, 1, 512), (2, 24, 512), True),
    ("qvm", (1, 64), (64, 128), False),
    ("qvm-3", (3, 96), (96, 64), False),
    ("qmm-n", (32, 64), (64, 96), False),
    ("vector", (512,), (24, 512), True),
]
for mode, gs in FP_MODES.items():
    for dt in ["float32", "float16", "bfloat16"]:
        for name, xs, ws, t in FP_PRODUCTS:
            if not t and ws[-1] % gs:
                continue
            with mx.stream(mx.gpu):
                wq, scales = mx.quantize(make(dt, ws) / 4, mode=mode)
                mx.eval(wq, scales)
            case_gpu(f"fp-qmm-{mode}-{dt}-{name}", "quantized_matmul", [make(dt, xs) / 4, wq, scales],
                     mode=mode, transpose=int(t))

# Gathers on MLX's paths: per-row gather_qmv(_fast) (unsorted, or sorted below MLX's grouping
# rule), the grouped gather_qmm_rhs (more than 64 sorted rows, 4 or more per expert),
# gather_qmm_t from the vector limit on, untransposed gather_qvm (fewer than 4 rows) and
# gather_qmm_n (more than 8 rows); nvfp4 also with a global scale per expert. gpt-oss's
# inner size (K = 2880, not fast-aligned) in bfloat16 only, with fewer experts and outputs
# than the model, to keep the file small.
FP_GATHERS = [
    ("qmv-1", 8, 32, 512, 1, 1, False, True),
    ("qmv-2", 8, 32, 512, 1, 2, False, True),
    ("qmv-5-k2880", 4, 12, 2880, 1, 5, False, True),
    ("qmv-16", 8, 32, 512, 1, 16, False, True),
    ("sorted-few", 8, 32, 512, 1, 20, True, True),
    ("grouped", 8, 32, 256, 1, 100, True, True),
    ("qmm-t", 4, 32, 256, 40, 3, False, True),
    ("qvm", 8, 64, 128, 1, 3, False, False),
    ("qmm-n", 8, 64, 128, 12, 2, False, False),
]
OSS_GATHERS = [
    ("oss-decode", 4, 16, 2880, 1, 4, False, True),
    ("oss-grouped", 4, 16, 2880, 1, 72, True, True),
]
for mode, gs in FP_MODES.items():
    for dt in ["float32", "float16", "bfloat16"]:
        for name, e, n, k, m, batches, srt, t in FP_GATHERS + (OSS_GATHERS if dt == "bfloat16" else []):
            with mx.stream(mx.gpu):
                wq, scales = mx.quantize(make(dt, (e, n, k) if t else (e, k, n)) / 4, mode=mode)
                mx.eval(wq, scales)
            ids = np.array([(i * 7 + 3) % e for i in range(batches)], dtype=np.uint32)
            if srt:
                ids.sort()
            rhs = mx.array(ids)
            x = make(dt, (batches, m, k)) / 4
            for global_scale in [False] + ([True] if mode == "nvfp4" else []):
                inputs = [x, wq, scales]
                extra = {}
                if global_scale:
                    inputs.append(mx.array((0.25 + np.arange(e) / 8).astype(np.float32)))
                    extra["gs"] = 1
                case_gpu(f"fp-gather-{mode}-{dt}-{name}{'-gs' if global_scale else ''}", "gather_qmm",
                         inputs + [rhs], mode=mode, transpose=int(t), sorted=int(srt), idx="r",
                         **extra)

# E4M3 conversions: every float16 and bfloat16 bit pattern, a float32 sweep of 2^20 values
# (random magnitudes over the whole range, ties, saturation, subnormals, NaN and infinities),
# and every code back in each float type.
every16 = np.arange(65536, dtype=np.uint16)
case_gpu("fp8-to-float16-all", "to_fp8", [mx.array(every16).view(mx.float16)])
case_gpu("fp8-to-bfloat16-all", "to_fp8", [mx.array(every16).view(mx.bfloat16)])
sweep = (rng.standard_normal(1 << 20) * np.exp2(rng.integers(-30, 12, 1 << 20))).astype(np.float32)
sweep[:16] = [0.0, -0.0, np.inf, -np.inf, np.nan, 448.0, 464.0, 465.0, 1e-40, -1e-40, 2.0**-9,
              2.0**-10, 3 * 2.0**-11, 1.0625, 1.1875, -240.0]
sweep[16] = np.array([0xFFC00000], dtype=np.uint32).view(np.float32)[0]  # -NaN
case_gpu("fp8-to-float32-sweep", "to_fp8", [mx.array(sweep)])
codes = mx.array(np.arange(256, dtype=np.uint8))
for dt in ["float16", "bfloat16", "float32"]:
    case_gpu(f"fp8-from-{dt}", "from_fp8", [codes], dtype=CODE[dt])

# Untransposed products from K = 1024 (MLX's qvm_split_k: 8 partitions, 32 above K = 8192,
# summed in the output dtype; K = 2880 does not split evenly; float32 sums are the standard
# reduction, so only one split there), after every other case so the earlier inputs keep
# their random values.
FP_SPLIT_K = [
    ("qvm-splitk", (1, 2048), (2048, 64)),
    ("qvm-splitk-k2880", (3, 2880), (2880, 32)),
    ("qvm-splitk32", (2, 8320), (8320, 32)),
    ("qvm-splitk-batched", (2, 1, 1024), (2, 1024, 32)),
]
for mode in FP_MODES:
    for dt in ["float32", "float16", "bfloat16"]:
        for name, xs, ws in FP_SPLIT_K:
            if dt == "float32" and name in ("qvm-splitk32", "qvm-splitk-batched"):
                continue
            with mx.stream(mx.gpu):
                wq, scales = mx.quantize(make(dt, ws) / 4, mode=mode)
                mx.eval(wq, scales)
            case_gpu(f"fp-qmm-{mode}-{dt}-{name}", "quantized_matmul", [make(dt, xs) / 4, wq, scales],
                     mode=mode, transpose=0)

# The same in affine mode (scales and biases), 4 bits in groups of 64 and 8 bits in 32.
AFFINE_SPLIT_K = [
    ("qvm-splitk", (1, 2048), (2048, 128)),
    ("qvm-splitk-k2880", (3, 2880), (2880, 64)),
    ("qvm-splitk32", (2, 8320), (8320, 64)),
    ("qvm-splitk-batched", (2, 1, 1024), (2, 1024, 64)),
]
for gs, bits in [(64, 4), (32, 8)]:
    for dt in ["float32", "float16", "bfloat16"]:
        for name, xs, ws in AFFINE_SPLIT_K:
            if dt == "float32" and name in ("qvm-splitk32", "qvm-splitk-batched"):
                continue
            with mx.stream(mx.gpu):
                wq, scales, biases = mx.quantize(make(dt, ws) / 4, group_size=gs, bits=bits)
                mx.eval(wq, scales, biases)
            case_gpu(f"affine-qmm-b{bits}-{dt}-{name}", "quantized_matmul",
                     [make(dt, xs) / 4, wq, scales, biases], group_size=gs, bits=bits,
                     transpose=0)

# floor_divide (added after every other case, with inputs of its own, so earlier cases keep
# their random values): every integer type with mixed signs and both remainders (no zero
# divisors: MLX leaves them undefined), MIN // 1, floats with ties, infinities, NaN and zero
# divisors, a broadcast and an integer/float promotion.
fd = np.random.default_rng(SEED + 55)
for dt in INTS:
    lo, hi = (0, 100) if dt.startswith("u") else (-100, 100)
    a = fd.integers(lo, hi, (4, 6)).astype(dt)
    b = fd.integers(1, 9, (6,)).astype(dt)
    if not dt.startswith("u"):
        b = b * np.where(fd.integers(0, 2, (6,)) == 0, -1, 1).astype(dt)
    case(f"floor_divide-{dt}", "floor_divide", [mx.array(a), mx.array(b)])
case("floor_divide-int32-min", "floor_divide",
     [mx.array(np.array([np.iinfo(np.int32).min, -7, 7], np.int32)), mx.array(np.array([1, 7, -7], np.int32))])
special = np.array([7.5, -7.5, 6.0, -6.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 0.0, 1e30, -3.0], np.float32)
divisor = np.array([2.0, 2.0, -2.0, 4.0, 0.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1e-30, 0.1], np.float32)
for dt in FLOATS:
    case(f"floor_divide-{dt}-special", "floor_divide",
         [mx.array(special).astype(getattr(mx, dt)), mx.array(divisor).astype(getattr(mx, dt))])
    case(f"floor_divide-{dt}", "floor_divide",
         [mx.array(fd.standard_normal((3, 5)) * 10).astype(getattr(mx, dt)),
          mx.array(fd.standard_normal((5,)) + 0.5).astype(getattr(mx, dt))])
case("floor_divide-int-float", "floor_divide",
     [mx.array(np.array([7, -7, 9], np.int32)), mx.array(np.array([2.5, 2.5, -4.0], np.float32))])

# power (appended, inputs of its own): integer squaring with wrapping and zero exponents
# (no negative ones: MLX leaves them undefined), floats with negative bases, zeros,
# fractional and negative exponents, infinities and NaN, and an integer/float promotion.
pw = np.random.default_rng(SEED + 59)
for dt in ["int8", "uint8", "int32", "uint32", "int64"]:
    lo = 0 if dt.startswith("u") else -5
    case(f"power-{dt}", "power", [mx.array(pw.integers(lo, 6, (12,)).astype(dt)),
                                  mx.array(pw.integers(0, 9, (12,)).astype(dt))])
pbase = np.array([2.0, -2.0, 0.0, -0.0, 1.5, -1.5, np.inf, -np.inf, np.nan, 1.0, 0.5, 10.0], np.float32)
pexp = np.array([0.5, 3.0, -1.0, 2.0, -2.5, 0.5, 0.0, 3.0, 1.0, np.nan, np.inf, -3.0], np.float32)
for dt in FLOATS:
    case(f"power-{dt}-special", "power",
         [mx.array(pbase).astype(getattr(mx, dt)), mx.array(pexp).astype(getattr(mx, dt))])
    case(f"power-{dt}", "power", [mx.array(np.abs(pw.standard_normal((3, 5))) * 4).astype(getattr(mx, dt)),
                                  mx.array(pw.standard_normal((5,)) * 2).astype(getattr(mx, dt))])
    if dt != "float64":
        case_gpu(f"power-gpu-{dt}", "power",
                 [mx.array(np.abs(pw.standard_normal((4, 64))) * 4).astype(getattr(mx, dt)),
                  mx.array(pw.standard_normal((4, 64)) * 3).astype(getattr(mx, dt))])
case("power-int-float", "power",
     [mx.array(np.array([2, -3, 4], np.int32)), mx.array(np.array([0.5, 2.0, -1.5], np.float32))])

# Affine dequantize on MLX's GPU (compiled with contraction: up to one rounding from MLX's
# CPU, whose results mlx.bin holds): every bit width and group size, three dtypes, with
# inputs of their own (appended last). Quantize is MLX's CPU's on both devices.
aq = np.random.default_rng(SEED + 57)
for dt in ["float32", "float16", "bfloat16"]:
    for bits in [2, 3, 4, 5, 6, 8]:
        for gs in [32, 64, 128]:
            w = mx.array((aq.standard_normal((6, 256)) * 2).astype(np.float32)).astype(getattr(mx, dt))
            with mx.stream(mx.gpu):
                q, s, b = mx.quantize(w, group_size=gs, bits=bits)
                mx.eval(q, s, b)
            case_gpu(f"affine-dequantize-{dt}-b{bits}-g{gs}", "dequantize", [q, s, b], group_size=gs,
                     bits=bits)

# MLX's blocked head-512 prefill (`use_blocked_d512`: half precision, causal, 1,024 queries
# or more): grouped queries over a longer key cache, and a block count that is not whole.
dq = np.random.default_rng(SEED + 58)
for dt, (hq, hkv, lq, lk) in [("bfloat16", (2, 1, 1024, 1030)), ("float16", (1, 1, 1100, 1100))]:
    q, k, v = (mx.array(dq.standard_normal((1, h, n, 512)).astype(np.float32)).astype(getattr(mx, dt))
               for h, n in [(hq, lq), (hkv, lk), (hkv, lk)])
    case_gpu(f"sdpa-d512-{dt}-{lq}x{lk}", "sdpa", [q, k, v], scale=round(512 ** -0.5, 6),
             mask="causal")

# ── Writer ──────────────────────────────────────────────────────────────────

def string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<H", len(raw)) + raw


def tensor(a: mx.array) -> bytes:
    a = mx.contiguous(a)
    mx.eval(a)
    name = NAME_OF[a.dtype]
    if name == "bfloat16":
        data = np.array(a.view(mx.uint16)).tobytes()
    else:
        data = np.ascontiguousarray(np.array(a)).tobytes()
    head = struct.pack("<BB", CODE[name], a.ndim) + b"".join(struct.pack("<Q", d) for d in a.shape)
    return head + struct.pack("<Q", len(data)) + data


def write(out: Path, cases: list[tuple[str, str, list[mx.array], list[mx.array]]],
          device: str = "cpu") -> None:
    blob = bytearray(b"BSMXFIX1")
    blob += string(f"mlx={mx.__version__} device={device} seed={SEED}")
    blob += struct.pack("<I", len(cases))
    for name, op, inputs, outputs in cases:
        blob += string(name) + string(op)
        blob += struct.pack("<B", len(inputs)) + b"".join(tensor(x) for x in inputs)
        blob += struct.pack("<B", len(outputs)) + b"".join(tensor(y) for y in outputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))
    print(f"wrote {len(cases)} cases, {len(blob)} bytes -> {out.name}")


def main() -> None:
    write(OUT, cases)
    write(OUT_API, api_cases)
    write(OUT_GPU, gpu_cases, device="gpu")


if __name__ == "__main__":
    main()

# Format "BSMXFIX1" (little-endian):
#   magic "BSMXFIX1" | meta:str | count:u32 | case*
#   case   := name:str op:str n_in:u8 tensor* n_out:u8 tensor*
#   tensor := dtype:u8 ndim:u8 dims:u64* nbytes:u64 data (row-major, contiguous)
#   str    := len:u16 utf8
