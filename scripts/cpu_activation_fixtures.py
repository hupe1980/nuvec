# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5", "numpy==2.5.4"]
# ///
"""Generate the CPU view fixtures, or diagnose one full-layout element-wise case.

uv run scripts/cpu_activation_fixtures.py --out crates/nuvec-core/tests/fixtures/cpu-activations.txt
uv run scripts/cpu_activation_fixtures.py --witness float16 sigmoid
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import elementwise_fixtures as corpus
import mlx.core as mx


def layouts():
    # Store the backing operands and slice geometry so the Rust check replays the same view.
    yield "scalar", 1, 1, 1
    yield "empty", 0, 0, 1
    for size in (7, 8, 9):
        yield "tail", size, 0, 1
    yield "offset", 8, 1, 1
    yield "strided", 8, 1, 2
    yield "broadcast", 9, 0, 1
    yield "transpose", 9, 0, 1
    yield "row-broadcast", 18, 0, 1


def generate():
    lines = ["# MLX CPU 0.32.5: dtype op layout size start step left-bits right-bits expected-bits"]
    for dtype in ("float16", "bfloat16"):
        dt = getattr(mx, dtype)
        # Include negative zero, underflow, saturating inputs, infinities and NaNs.
        a = mx.array([-0.0, -2.3, -8.1, 0.1, -88, 1.35, 6.71, 2**-20, 0,
                      -float("inf"), float("inf"), -0.125, 1.88, float("nan"),
                      -16, 4.2, 8.3, 0.5], dtype=dt)
        b = mx.array([0.0, 0.63, 0.17, -2.7, -90, -2.2, -50, -2**-18, -0.0,
                      -float("inf"), 0, 2.64, float("nan"), 1, 16, -3.6, 10, -1],
                     dtype=dt)
        def pack(arr, dtype=dtype):
            return ",".join(f"{value:04x}" for value in corpus.result_words(arr, dtype)) or "-"

        for layout, size, start, step in layouts():
            x, y = a[start : start + size * step : step], b[start : start + size * step : step]
            if layout == "scalar":
                x, y = x.reshape(()), y.reshape(())
            if layout == "broadcast":
                y = b[1]
            if layout == "transpose":
                x, y = x.reshape(3, 3).T, y.reshape(3, 3).T
            if layout == "row-broadcast":
                x, y = x.reshape(2, 9), b[:9].reshape(1, 9)
            for op, out in [("sigmoid", mx.sigmoid(x)), ("logaddexp", mx.logaddexp(x, y))]:
                lines.append(f"{dtype} {op} {layout} {size} {start} {step} {pack(a)} {pack(b)} {pack(out)}")
    return "\n".join(lines) + "\n"


def witness(dtype, op):
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "cpu-reference.bin"
        subprocess.run([sys.executable, corpus.__file__, "--device", "cpu", "--dtype", dtype,
                        "--op", op, "--out", str(Path(tmp) / "case.txt"), "--words", str(ref)],
                       check=True)
        env = dict(os.environ, NUVEC_WITNESS_OP=op, NUVEC_WITNESS_DTYPE=dtype, NUVEC_WITNESS_REF=str(ref))
        subprocess.run(["cargo", "test", "--locked", "--release", "-p", "nuvec-core",
                        "--test", "elementwise", "cpu::selected_case_witness", "--",
                        "--ignored", "--exact", "--nocapture"], cwd=corpus.ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--witness", nargs=2, metavar=("DTYPE", "OP"))
    args = parser.parse_args()
    if mx.__version__ != "0.32.5":
        parser.error(f"expected MLX 0.32.5, found {mx.__version__}")
    mx.set_default_device(mx.cpu)
    if args.witness:
        dtype, op = args.witness
        if dtype not in corpus.WORD or op not in corpus.UNARY + corpus.BINARY:
            parser.error("expected a corpus dtype and an element-wise operation")
        witness(dtype, op)
    else:
        args.out.write_text(generate())


if __name__ == "__main__":
    main()
