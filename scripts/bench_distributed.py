# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""`all_sum` over the TCP ring: nuvec vs MLX's ring backend on the same machine, with 2 and
4 ranks over loopback.

Builds `cargo bench -p nuvec-distributed --bench all_sum`, then for each rank count starts
the ranks of nuvec's bench and of the same loop in MLX (`mx.distributed.init(backend=
"ring")`, `all_sum` on the CPU stream, one evaluation per sum, the best of three bursts
after a warm-up) with one hostfile each, alternating the two programs twice. Prints
microseconds per sum, throughput and the nuvec/MLX time ratios.

Usage: `just bench-distributed`.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = [4 << 10, 1 << 20, 16 << 20, 256 << 20]


def mlx_rank() -> None:
    import mlx.core as mx

    group = mx.distributed.init(backend="ring", strict=True)
    for nbytes in SIZES:
        x = mx.ones((nbytes // 4,), dtype=mx.float32)
        mx.eval(x)
        reps = max(3, min(200, (64 << 20) // nbytes))
        best = float("inf")
        for burst in range(4):
            mx.eval(mx.distributed.all_sum(mx.ones((1,)), stream=mx.cpu))
            t = time.perf_counter()
            for _ in range(reps):
                mx.eval(mx.distributed.all_sum(x, stream=mx.cpu))
            us = (time.perf_counter() - t) / reps * 1e6
            if burst > 0:
                best = min(best, us)
        if group.rank() == 0:
            print(f"{nbytes}\t{best:.1f}", flush=True)


def free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def run(cmd: list[str], n: int) -> dict[int, float]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([[f"127.0.0.1:{p}"] for p in free_ports(n)], f)
        hostfile = f.name
    procs = [
        subprocess.Popen(
            cmd,
            env={**os.environ, "MLX_HOSTFILE": hostfile, "MLX_RANK": str(r)},
            stdout=subprocess.PIPE,
            text=True,
        )
        for r in range(n)
    ]
    out = procs[0].communicate(timeout=1800)[0]
    for p in procs[1:]:
        p.wait(timeout=60)
    os.unlink(hostfile)
    if any(p.returncode for p in procs):
        sys.exit(f"{cmd[0]}: a rank failed")
    return {int(b): float(us) for b, us in (line.split("\t") for line in out.split("\n") if line)}


def main() -> None:
    if os.environ.get("MLX_RANK") is not None:
        mlx_rank()
        return
    built = subprocess.run(
        ["cargo", "bench", "-q", "-p", "nuvec-distributed", "--bench", "all_sum", "--no-run",
         "--message-format=json"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    exe = next(
        json.loads(line)["executable"]
        for line in built.splitlines()
        if '"executable":"' in line and "all_sum" in line
    )
    nuvec_cmd = [exe, "--bench"]
    mlx_cmd = [sys.executable, __file__]
    import mlx.core as mx

    print(f"all_sum over a TCP ring on loopback, MLX {mx.__version__}; µs per sum (best of 3 bursts x 2 alternating runs)")
    print("| ranks | bytes | nuvec µs | MLX µs | nuvec GB/s | MLX GB/s | nuvec / MLX time |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for n in (2, 4):
        nv: dict[int, float] = {}
        ml: dict[int, float] = {}
        for _ in range(2):
            for k, v in run(nuvec_cmd, n).items():
                nv[k] = min(nv.get(k, float("inf")), v)
            for k, v in run(mlx_cmd, n).items():
                ml[k] = min(ml.get(k, float("inf")), v)
        for b in SIZES:
            print(f"| {n} | {b} | {nv[b]:.1f} | {ml[b]:.1f} | {b / nv[b] / 1e3:.2f} | "
                  f"{b / ml[b] / 1e3:.2f} | {nv[b] / ml[b]:.2f}× |")


if __name__ == "__main__":
    main()
