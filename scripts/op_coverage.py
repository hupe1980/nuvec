# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""`mlx.core`'s and `mlx.core.linalg`'s functions without a nuvec counterpart.

Reads nuvec's public functions from the sources (`pub fn`s and the operation macros'
lists) and prints MLX's array functions that are missing, minus the runtime controls
nuvec has elsewhere (memory, streams, devices, files, compile) and those in `LATER`
(listed as out of scope). Exits non-zero when anything else is missing.

Usage: `uv run scripts/op_coverage.py`.
"""

import inspect
import re
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent

# Elsewhere in nuvec, or not array operations.
RUNTIME = set("""
clear_cache clear_streams compile disable_compile enable_compile device_count device_info
default_device default_stream get_active_memory get_array_buffer_size get_cache_memory
get_memory_limit get_peak_memory get_printoptions is_available new_stream
new_thread_local_stream new_thread_unsafe_stream printoptions reset_peak_memory
set_cache_limit set_default_device set_default_stream set_memory_limit set_printoptions
set_wired_limit stream synchronize load save save_gguf save_safetensors savez
savez_compressed from_dlpack asarray export_function export_to_dot exporter import_function
eval async_eval checkpoint custom_function grad value_and_grad jvp vjp vmap
""".split())
# Out of scope: listed, not required.
LATER: set[str] = set()
# MLX names nuvec spells otherwise.
ALIASES = {
    "concat": "concatenate", "pow": "power", "permute_dims": "transpose",
    "cumulative_sum": "cumsum", "cumulative_prod": "cumprod", "where": "where_",
    "left_shift": "left_shift", "conjugate": "conjugate",
}


def nuvec_names() -> set[str]:
    names: set[str] = set()
    for f in (ROOT / "crates/nuvec-core/src").rglob("*.rs"):
        text = f.read_text()
        names |= set(re.findall(r"pub fn ([a-z0-9_]+)", text))
        for macro in ("binary_ops!", "binary_fn!", "unary_fn!", "reductions!", "scans!"):
            for block in re.findall(re.escape(macro) + r" \{(.*?)\n\}", text, re.S):
                names |= set(re.findall(r"(\w+)(?:, \w+)? =>", block))
        for block in re.findall(r"unary_ops!\((.*?)\);", text, re.S):
            names |= set(re.findall(r"(\w+)", block))
    return names


def linalg_names() -> set[str]:
    text = (ROOT / "crates/nuvec-core/src/linalg.rs").read_text()
    names = set(re.findall(r"pub fn ([a-z0-9_]+)", text))
    # MLX's `svd(a, compute_uv=False)` is nuvec's `svd_values`.
    return names


def main() -> None:
    have = nuvec_names()
    la = linalg_names()
    la_missing = [n for n in dir(mx.linalg) if not n.startswith("_") and n not in la]
    print(f"{len(dir(mx.linalg))} mx.linalg names; missing: {' '.join(la_missing) or 'none'}")
    mlx = sorted(n for n in dir(mx) if not n.startswith("_") and callable(getattr(mx, n))
                 and not inspect.isclass(getattr(mx, n)))
    missing = [n for n in mlx if n not in RUNTIME | LATER and n not in have
               and ALIASES.get(n, "~") not in have]
    later = [n for n in mlx if n in LATER and n not in have]
    print(f"{len(mlx)} mlx.core functions; out of scope: {' '.join(later) or 'none'}")
    if missing or la_missing:
        print(f"missing ({len(missing)}): {' '.join(missing)}")
        sys.exit(1)
    print("every other function has a nuvec counterpart")


if __name__ == "__main__":
    main()
