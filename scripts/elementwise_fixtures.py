# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5", "numpy"]
# ///
"""Element-wise fixtures from MLX: every half-precision value, 2^20 float32 patterns.

`uv run scripts/elementwise_fixtures.py` writes
`crates/nuvec-core/tests/fixtures/elementwise.txt`, MLX's GPU and CPU outputs as FNV-1a hashes
of their bits in 16 chunks (NaNs canonical):

- float16 and bfloat16: each unary op over all 65,536 bit patterns, each binary op over 2^20
  pairs (`x = i & 0xffff`, `y = ((i · 40503 + 12345) >> 4) & 0xffff`);
- float32: 2^20 bit patterns `i · 2654435761 mod 2^32` (every exponent, both signs, NaNs and
  infinities), paired with `x · 40503 + 12345 mod 2^32` for binary ops.

Then fused and reduction ops (`FAST`) over deterministic inputs `v(i) = ((i · 2654435761 mod
2^32) >> 8) · 2^-21 − 4` in each dtype: norms, rotations, softmax, reductions, scans, the
Hadamard transform and a depthwise convolution.

`crates/nuvec-core/tests/elementwise.rs` checks nuvec against it.
"""

import argparse
import json
import platform
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "crates/nuvec-core/tests/fixtures/elementwise.txt"

UNARY = [
    "abs", "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctanh", "ceil", "cos",
    "cosh", "erf", "erfinv", "exp", "expm1", "floor", "log", "log2", "log10", "log1p",
    "negative", "round", "sigmoid", "sign", "sin", "sinh", "square", "sqrt", "rsqrt", "tan",
    "tanh",
]
BINARY = [
    "add", "subtract", "multiply", "divide", "floor_divide", "remainder", "logaddexp",
    "maximum", "minimum", "power", "arctan2",
]
FAST = [
    "rms_norm", "layer_norm", "rope", "rope_traditional", "softmax", "logsumexp", "sum",
    "mean", "max", "cumsum", "logcumsumexp", "hadamard", "conv_depthwise", "sum_columns",
    "sum_long_columns",
]
CHUNKS = 16
NAN = {"float16": 0x7E00, "bfloat16": 0x7FC0, "float32": 0x7FC00000}
WORD = {"float16": mx.uint16, "bfloat16": mx.uint16, "float32": mx.uint32}


def fnv(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def result_words(out: mx.array, dtype: str) -> np.ndarray:
    bits = np.array(mx.contiguous(out).view(WORD[dtype])).reshape(-1).copy()
    bits[np.array(mx.isnan(out)).reshape(-1)] = NAN[dtype]
    return bits


def inputs(dtype: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unary inputs and the binary pairs, as bit patterns."""
    i = np.arange(1 << 20, dtype=np.uint64)
    if dtype == "float32":
        x = ((i * 2654435761) & 0xFFFFFFFF).astype(np.uint32)
        y = ((x.astype(np.uint64) * 40503 + 12345) & 0xFFFFFFFF).astype(np.uint32)
        return x, x, y
    xs = (i & 0xFFFF).astype(np.uint16)
    ys = (((i * 40503 + 12345) >> 4) & 0xFFFF).astype(np.uint16)
    return np.arange(65536, dtype=np.uint16), xs, ys


def values(n: int, start: int = 0) -> mx.array:
    i = np.arange(start, start + n, dtype=np.uint64)
    k = ((i * 2654435761) & 0xFFFFFFFF) >> 8
    return mx.array((k.astype(np.float32) * np.float32(2.0**-21)) - np.float32(4.0))


def fast(op: str, dt, stream) -> mx.array:
    x = values(32 * 4096).reshape(32, 4096).astype(dt)
    w = (values(4096, 7) * 0.25 + 1).astype(dt)
    b = (values(4096, 13) * 0.1).astype(dt)
    f = {
        "rms_norm": lambda: mx.fast.rms_norm(x, w, 1e-6, stream=stream),
        "layer_norm": lambda: mx.fast.layer_norm(x, w, b, 1e-5, stream=stream),
        "rope": lambda: mx.fast.rope(
            x.reshape(1, 16, 64, 128), 64, traditional=False, base=10000.0, scale=1.0,
            offset=7, stream=stream),
        "rope_traditional": lambda: mx.fast.rope(
            x.reshape(1, 16, 64, 128), 128, traditional=True, base=1e6, scale=1.0, offset=300,
            stream=stream),
        "softmax": lambda: mx.softmax(x, axis=-1, stream=stream),
        "logsumexp": lambda: mx.logsumexp(x, axis=-1, stream=stream),
        "sum": lambda: mx.sum(x, axis=-1, stream=stream),
        "mean": lambda: mx.mean(x, axis=-1, stream=stream),
        "max": lambda: mx.max(x, axis=-1, stream=stream),
        "cumsum": lambda: mx.cumsum(x, axis=-1, stream=stream),
        "logcumsumexp": lambda: mx.logcumsumexp(x, axis=-1, stream=stream),
        "hadamard": lambda: mx.hadamard_transform(x.reshape(128, 1024), stream=stream),
        "sum_columns": lambda: mx.sum(x.reshape(4096, 32), axis=0, stream=stream),
        "sum_long_columns": lambda: mx.sum(
            values(1100 * 32).reshape(1100, 32).astype(dt), axis=0, stream=stream),
        "conv_depthwise": lambda: mx.conv1d(
            x.reshape(1, 256, 512), (values(512 * 4, 3) * 0.5).reshape(512, 4, 1).astype(dt),
            groups=512, stream=stream),
    }[op]
    return f()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--dtype", choices=tuple(WORD))
    parser.add_argument("--op", choices=[*UNARY, *BINARY, *("fast:" + op for op in FAST)])
    parser.add_argument("--words", type=Path, help="save canonical output words for one case")
    args = parser.parse_args()
    if (args.device or args.dtype or args.op) and args.out is None:
        parser.error("filtered fixtures require --out")
    if args.words and not (args.device and args.dtype and args.op):
        parser.error("--words requires --device, --dtype and --op")
    destination = args.out or OUT
    lines = [f"# mlx={mx.__version__} chunks={CHUNKS}"]
    for device in [args.device] if args.device else ["gpu", "cpu"]:
        stream = mx.gpu if device == "gpu" else mx.cpu
        for dtype in [args.dtype] if args.dtype else WORD:
            dt = getattr(mx, dtype)
            u, xs, ys = inputs(dtype)
            x, a, b = (mx.array(v).view(dt) for v in (u, xs, ys))
            for op in [args.op] if args.op else [*UNARY, *BINARY, *("fast:" + f for f in FAST)]:
                if op.startswith("fast:"):
                    out = fast(op[5:], dt, stream)
                elif op in BINARY:
                    out = getattr(mx, op)(a, b, stream=stream)
                else:
                    out = getattr(mx, op)(x, stream=stream)
                bits = result_words(out, dtype)
                if args.words:
                    args.words.write_bytes(bits.tobytes())
                chunks = [str(fnv(c.tobytes())) for c in np.array_split(bits, CHUNKS)]
                lines.append(" ".join([device, dtype, op, *chunks]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    if args.metadata:
        info = mx.device_info()
        args.metadata.write_text(json.dumps({
            "mlx": mx.__version__, "macos": platform.mac_ver()[0],
            "chip": info.get("device_name"), "architecture": info.get("architecture"),
        }, indent=2) + "\n")
    print(f"wrote {destination} ({len(lines) - 1} cases)")


if __name__ == "__main__":
    main()
