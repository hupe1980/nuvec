# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""Checkpoint loading: nuvec (memory-mapped, no copy) vs MLX (pread into new buffers).

Writes a safetensors file of 2 GiB of bfloat16 weights (128 tensors of 4096 x 2048) under
`target/`, then times opening it and summing every tensor on the GPU, best of 5, in both
libraries. The file is in the page cache after it is written (a warm reload).

Usage: `just bench-load`.
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "target" / "bench-load.safetensors"


def main() -> None:
    if not PATH.exists():
        PATH.parent.mkdir(exist_ok=True)
        weights = {f"layer{i}.weight": mx.random.normal((4096, 2048)).astype(mx.bfloat16)
                   for i in range(128)}
        mx.save_safetensors(str(PATH), weights)
    size = PATH.stat().st_size / 2**30
    mx.set_default_device(mx.gpu)
    mx.reset_peak_memory()
    best = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        weights = mx.load(str(PATH))
        mx.eval([mx.sum(w) for w in weights.values()])
        best = min(best, time.perf_counter() - t)
        del weights
    proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-io", "--bench", "load", "--",
                           str(PATH)], cwd=ROOT, capture_output=True, text=True, check=True)
    _, ours, our_peak = proc.stdout.strip().split("\t")
    ours, our_peak = float(ours), int(our_peak)
    theirs, their_peak = best * 1e3, mx.get_peak_memory()
    print(f"{size:.1f} GiB safetensors (128 bf16 tensors), {platform.machine()} "
          f"{platform.mac_ver()[0]}; MLX {mx.__version__}; warm page cache; best of 5")
    print("| | nuvec | MLX | MLX / nuvec |")
    print("|---|---:|---:|---:|")
    print(f"| load + sum every tensor (ms) | {ours:.1f} | {theirs:.1f} | {theirs / ours:.2f}× |")
    print(f"| peak GPU allocations (GiB) | {our_peak / 2**30:.2f} | {their_peak / 2**30:.2f} | |")


if __name__ == "__main__":
    main()
