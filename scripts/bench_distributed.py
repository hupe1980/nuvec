# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4"]
# ///
"""`all_sum`: nuvec vs MLX on the same machines. By default over the TCP ring with 2 and 4
ranks on loopback; with `--hostfile` on several Macs through `mlx.launch` (`--backend ring`,
`jaccl` or `jaccl-ring`; JACCL needs Thunderbolt 5 cables and RDMA enabled).

Builds `cargo bench -p nuvec-distributed --bench all_sum`, then for each rank count starts
the ranks of nuvec's bench and of the same loop in MLX (`mx.distributed.init(backend=
"ring")`, `all_sum` on the CPU stream, one evaluation per sum, the best of three bursts
after a warm-up) with one hostfile each, alternating the two programs twice. Prints
microseconds per sum, throughput and the nuvec/MLX time ratios, for nuvec unencrypted and
encrypted (a key set), and a 32 MB pipeline hop (`send` from rank 0 to rank 1) for each.

Usage: `just bench-distributed`; on several Macs
`uv run scripts/bench_distributed.py --hostfile hosts.json --backend jaccl` (MLX's hostfile;
the repository at the same path on every Mac, so the built bench runs there too).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = [4 << 10, 1 << 20, 4 << 20, 16 << 20, 256 << 20]
# `mlx.launch` (its module has no `__main__` guard, so `python -m` would do nothing).
LAUNCH = [sys.executable, "-c", "from mlx._distributed_utils.launch import main; main()"]


def mlx_decode(group) -> None:
    """Decode's shape: 56 GPU ops each followed by an all_sum of [1, 1024] bfloat16 per
    evaluation, on the GPU stream (the best of five bursts of 20 evaluations);
    NUVEC_BENCH_ROWS rows instead of 1 (2,048: prefill's shape)."""
    import mlx.core as mx

    x = mx.ones((int(os.environ.get("NUVEC_BENCH_ROWS", "1")), 1024), dtype=mx.bfloat16)
    mx.eval(x)
    best = float("inf")
    for burst in range(6):
        mx.eval(mx.distributed.all_sum(mx.ones((1,))))
        t = time.perf_counter()
        for _ in range(20):
            y = x
            for _ in range(56):
                y = mx.distributed.all_sum(y * 0.5)
            mx.eval(y)
        us = (time.perf_counter() - t) / (20 * 56) * 1e6
        if burst > 0:
            best = min(best, us)
    if group.rank() == 0:
        print(f"decode\t{best:.1f}", flush=True)


def mlx_hop(group) -> None:
    """32 MB float32 from rank 0 to rank 1 on the CPU stream: µs per send, the best of five
    bursts of five ending in a barrier."""
    import mlx.core as mx

    x = mx.ones(((32 << 20) // 4,), dtype=mx.float32)
    mx.eval(x)
    best = float("inf")
    for burst in range(6):
        mx.eval(mx.distributed.all_sum(mx.ones((1,)), stream=mx.cpu))
        t = time.perf_counter()
        for _ in range(5):
            if group.rank() == 0:
                mx.eval(mx.distributed.send(x, 1, stream=mx.cpu))
            elif group.rank() == 1:
                mx.eval(mx.distributed.recv_like(x, 0, stream=mx.cpu))
        mx.eval(mx.distributed.all_sum(mx.ones((1,)), stream=mx.cpu))
        us = (time.perf_counter() - t) / 5 * 1e6
        if burst > 0:
            best = min(best, us)
    if group.rank() == 0:
        print(f"hop\t{best:.1f}", flush=True)


def mlx_rank() -> None:
    import mlx.core as mx

    group = mx.distributed.init(backend=os.environ.get("NUVEC_BENCH_BACKEND", "ring"), strict=True)
    if os.environ.get("NUVEC_BENCH_DECODE"):
        mlx_decode(group)
        return
    if os.environ.get("NUVEC_BENCH_HOP"):
        mlx_hop(group)
        return
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


KEY = "bench-secret-0123456789abcdef"


def run(cmd: list[str], n: int, env: dict[str, str] | None = None) -> dict[int, float]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([[f"127.0.0.1:{p}"] for p in free_ports(n)], f)
        hostfile = f.name
    procs = [
        subprocess.Popen(
            cmd,
            env={**os.environ, **(env or {}), "MLX_HOSTFILE": hostfile, "MLX_RANK": str(r)},
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
    rows = (line.split("\t") for line in out.split("\n") if line)
    return {int(b) if b.isdigit() else b: float(us) for b, us in rows}


def launched(cmd: list[str], hostfile: str, backend: str) -> dict[int, float]:
    """Rank 0's `bytes<TAB>us` lines of `cmd` started on every host by `mlx.launch`."""
    mlx_backend = "jaccl" if backend.startswith("jaccl") else backend
    out = subprocess.run(
        [*LAUNCH, "--backend", backend, "--hostfile", hostfile,
         "--env", f"NUVEC_BENCH_BACKEND={mlx_backend}", "--", *cmd],
        check=True, capture_output=True, text=True, timeout=3600,
    ).stdout
    rows = (re.match(r"^\s*(\d+)\t([\d.]+)\s*$", line) for line in out.splitlines())
    return {int(m[1]): float(m[2]) for m in rows if m}


def main() -> None:
    if os.environ.get("MLX_RANK") is not None:
        mlx_rank()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostfile", help="MLX's hostfile: run on its Macs through mlx.launch")
    parser.add_argument("--backend", default="ring", choices=["ring", "jaccl", "jaccl-ring"])
    parser.add_argument("--decode", action="store_true",
                        help="decode's shape on the GPU stream (2 ranks on loopback)")
    args = parser.parse_args()
    features = ["--features", "jaccl"] if args.backend.startswith("jaccl") else []
    built = subprocess.run(
        ["cargo", "bench", "-q", "-p", "nuvec-distributed", "--bench", "all_sum", "--no-run",
         *features, "--message-format=json"],
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

    if args.hostfile:
        print(f"all_sum over {args.backend} on {args.hostfile}, MLX {mx.__version__}; µs per sum (best of 3 bursts x 2 alternating runs)")
        print("| bytes | nuvec µs | MLX µs | nuvec GB/s | MLX GB/s | nuvec / MLX time |")
        print("|---:|---:|---:|---:|---:|---:|")
        nv: dict[int, float] = {}
        ml: dict[int, float] = {}
        for _ in range(2):
            for k, v in launched(nuvec_cmd, args.hostfile, args.backend).items():
                nv[k] = min(nv.get(k, float("inf")), v)
            for k, v in launched(mlx_cmd, args.hostfile, args.backend).items():
                ml[k] = min(ml.get(k, float("inf")), v)
        for b in SIZES:
            print(f"| {b} | {nv[b]:.1f} | {ml[b]:.1f} | {b / nv[b] / 1e3:.2f} | "
                  f"{b / ml[b] / 1e3:.2f} | {nv[b] / ml[b]:.2f}× |")
        return

    if args.decode:
        os.environ["NUVEC_BENCH_DECODE"] = "1"
        nv, ml = [], []
        for _ in range(3):
            nv.append(run([*nuvec_cmd, "--decode"], 2)["decode"])
            ml.append(run(mlx_cmd, 2)["decode"])
        print(f"decode-shaped all_sum, 2 ranks on loopback, MLX {mx.__version__}; µs per all_sum")
        print(f"nuvec {min(nv):.1f} ({', '.join(f'{v:.1f}' for v in nv)})")
        print(f"MLX   {min(ml):.1f} ({', '.join(f'{v:.1f}' for v in ml)})")
        print(f"MLX / nuvec time: {min(ml) / min(nv):.2f}x")
        return

    sealed = {"NUVEC_DISTRIBUTED_KEY": KEY}
    print(f"all_sum over a TCP ring on loopback, MLX {mx.__version__}; µs per sum (best of 3 bursts x 2 alternating runs)")
    print("| ranks | bytes | nuvec µs | encrypted µs | MLX µs | nuvec GB/s | encrypted GB/s | nuvec / MLX | encrypted / MLX |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in (2, 4):
        nv: dict[int, float] = {}
        ne: dict[int, float] = {}
        ml: dict[int, float] = {}
        for _ in range(2):
            for k, v in run(nuvec_cmd, n).items():
                nv[k] = min(nv.get(k, float("inf")), v)
            for k, v in run(nuvec_cmd, n, sealed).items():
                ne[k] = min(ne.get(k, float("inf")), v)
            for k, v in run(mlx_cmd, n).items():
                ml[k] = min(ml.get(k, float("inf")), v)
        for b in SIZES:
            print(f"| {n} | {b} | {nv[b]:.1f} | {ne[b]:.1f} | {ml[b]:.1f} | {b / nv[b] / 1e3:.2f} | "
                  f"{b / ne[b] / 1e3:.2f} | {nv[b] / ml[b]:.2f}× | {ne[b] / ml[b]:.2f}× |")
    hops = {"nuvec": [], "encrypted": [], "MLX": []}
    for _ in range(3):
        hops["nuvec"].append(run([*nuvec_cmd, "--hop"], 2)["hop"])
        hops["encrypted"].append(run([*nuvec_cmd, "--hop"], 2, sealed)["hop"])
        hops["MLX"].append(run(mlx_cmd, 2, {"NUVEC_BENCH_HOP": "1"})["hop"])
    print("\n32 MB hop (send from rank 0 to rank 1), 2 ranks on loopback; µs (best of 3 runs)")
    for name, v in hops.items():
        print(f"{name}: {min(v):.0f} µs, {(32 << 20) / min(v) / 1e3:.2f} GB/s ({', '.join(f'{x:.0f}' for x in v)})")


if __name__ == "__main__":
    main()
