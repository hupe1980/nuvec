# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "numpy==2.5.4"]
# ///
"""Writes files with MLX and NumPy for nuvec-io's tests (`crates/nuvec-io/tests/fixtures`).

The values are simple formulas the Rust tests recompute: `arange(n)` scaled per dtype.

Usage: `just io-fixtures` (or `uv run scripts/io_fixtures.py`).
"""

from pathlib import Path

import mlx.core as mx
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "crates/nuvec-io/tests/fixtures"
OUT.mkdir(parents=True, exist_ok=True)

tensors = {
    "bool": (mx.arange(6) % 2 == 0).reshape(2, 3),
    "u8": mx.arange(5).astype(mx.uint8),
    "u32": mx.arange(4).astype(mx.uint32),
    "i16": (mx.arange(6) - 3).astype(mx.int16),
    "i64": (mx.arange(6) - 3).astype(mx.int64),
    "f16": (mx.arange(8) * 0.25).astype(mx.float16),
    "bf16": (mx.arange(8) * 0.5).astype(mx.bfloat16),
    "f32": (mx.arange(12) * 0.125).reshape(3, 4),
    "scalar": mx.array(2.5),
    "empty": mx.zeros((0, 3)),
}
mx.save_safetensors(str(OUT / "mlx.safetensors"), tensors, metadata={"format": "mlx", "note": "ü \"q\""})

np.save(OUT / "c_f32.npy", (np.arange(24, dtype=np.float32) * 0.5).reshape(2, 3, 4))
np.save(OUT / "fortran_i16.npy", np.asfortranarray(np.arange(12, dtype=np.int16).reshape(3, 4)))
np.save(OUT / "bool.npy", np.arange(5) % 2 == 1)
np.save(OUT / "c64.npy", (np.arange(4) + 1j * np.arange(4) * 2).astype(np.complex64))
np.save(OUT / "scalar_f64.npy", np.float64(-1.5))
mx.save(str(OUT / "mlx_f16.npy"), (mx.arange(6) * 0.5).astype(mx.float16).reshape(2, 3))
print(f"wrote fixtures to {OUT}")
