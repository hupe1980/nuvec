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
    }
    reductions = {
        "sum": mx.sum, "prod": mx.prod, "max": mx.max, "min": mx.min, "mean": mx.mean,
        "all": mx.all, "any": mx.any,
    }
    if op in binary:
        return binary[op](xs[0], xs[1])
    if op in unary:
        return unary[op](xs[0])
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
