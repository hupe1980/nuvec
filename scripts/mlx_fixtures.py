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


def apply(op: str, p: dict[str, str], xs: list[mx.array]) -> mx.array:
    if xs:
        xs = [pre_view(xs[0], p.get("pre"))] + xs[1:]
    binary = {
        "add": mx.add, "subtract": mx.subtract, "multiply": mx.multiply, "divide": mx.divide,
        "maximum": mx.maximum, "minimum": mx.minimum, "equal": mx.equal,
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
    mx.eval(out)
    cases.append((name, spelled, list(inputs), [out]))


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
# out-of-range float->int casts are C++ undefined behaviour (nuvec saturates, D-021).
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
# `start + i*step` once (D-021), so inexact steps differ in the last bits.
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
def derivative(kind: str, name: str, fn: str, primals: list[mx.array], **params: object) -> None:
    p = {"fn": fn, **{k: str(v) for k, v in params.items()}}
    spelled = kind + "".join(f"|{k}={v}" for k, v in p.items())

    def f(*xs: mx.array) -> list[mx.array]:
        return [apply(fn, p, list(xs))]

    out = f(*primals)[0]
    if kind == "vjp":
        extra = [make(NAME_OF[out.dtype], out.shape)]
        _, results = mx.vjp(f, primals, extra)
    else:
        extra = [make(NAME_OF[x.dtype], x.shape) for x in primals]
        _, results = mx.jvp(f, primals, extra)
    mx.eval(results)
    cases.append((name, spelled, primals + extra, list(results)))


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
derivative("vjp", "vjp-maximum-ties", "maximum", [tie, ints("float32", (6,), -2, 3)])
derivative("vjp", "vjp-minimum-ties", "minimum", [tie, ints("float32", (6,), -2, 3)])
derivative("vjp", "vjp-multiply-t", "multiply", [make("float32", (4, 3)), make("float32", (3, 4))], pre="t")
derivative("vjp", "vjp-where", "where", [make("bool", (2, 3)), make("float32", (2, 3)), make("float32", (3,))])
derivative("vjp", "vjp-astype-f16", "astype", [make("float32", (5,))], dtype=CODE["float16"])
derivative("vjp", "vjp-reshape", "reshape", [make("float32", (2, 3))], shape="3,2")
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


def main() -> None:
    blob = bytearray(b"BSMXFIX1")
    blob += string(f"mlx={mx.__version__} device=cpu seed={SEED}")
    blob += struct.pack("<I", len(cases))
    for name, op, inputs, outputs in cases:
        blob += string(name) + string(op)
        blob += struct.pack("<B", len(inputs)) + b"".join(tensor(x) for x in inputs)
        blob += struct.pack("<B", len(outputs)) + b"".join(tensor(y) for y in outputs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(bytes(blob))
    print(f"wrote {len(cases)} cases, {len(blob)} bytes -> {OUT.name}")


if __name__ == "__main__":
    main()

# Format "BSMXFIX1" (little-endian):
#   magic "BSMXFIX1" | meta:str | count:u32 | case*
#   case   := name:str op:str n_in:u8 tensor* n_out:u8 tensor*
#   tensor := dtype:u8 ndim:u8 dims:u64* nbytes:u64 data (row-major, contiguous)
#   str    := len:u16 utf8
