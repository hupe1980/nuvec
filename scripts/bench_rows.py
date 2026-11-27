# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5"]
# ///
"""Few-row products (B = 1..16 rows: dense, affine quantized, gathered) on the GPU: nuvec's
default routing and its row-exact mode (`with_row_exact`, lanes that must equal their solo
runs) against MLX, and against nuvec 0.3.0's routing when `BASE` names a checkout of v0.3.0
whose `crates/nuvec-core/benches/rows.rs` holds the same cases (in the 0.3 API).

Runs `cargo bench -p nuvec-core --bench rows` in both modes (and in `BASE`) twice, and the
same cases in MLX, then prints microseconds per product, the row-exact time against one
row and against MLX, and the largest row count up to which every case keeps at least 0.98×
of 0.3.0 (the candidate for `ROW_INDEPENDENT_ROWS`).

Usage: `just bench-rows [filter]`, `BASE=../nuvec-0.3.0 just bench-rows`.
Use `BENCH_ROWS=5 just bench-rows affine8` for a single row count. The exact/B1
column requires a one-row control and is unavailable when selecting a different count.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
FILTER = sys.argv[1] if len(sys.argv) > 1 else ""
DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def nuvec(root: Path, exact: bool = False) -> dict[tuple[str, str, int], float]:
    out: dict[tuple[str, str, int], float] = {}
    for _ in range(2):
        proc = subprocess.run(["cargo", "bench", "-q", "-p", "nuvec-core", "--bench", "rows"],
                              cwd=root, capture_output=True, text=True, check=True,
                              env={**os.environ, "BENCH_FILTER": FILTER,
                                   "BENCH_EXACT": "1" if exact else "0"})
        for line in proc.stdout.splitlines():
            name, dt, b, us = line.split("\t")
            key = (name, dt, int(b))
            out[key] = min(out.get(key, float("inf")), float(us))
    return out


def mlx_op(name: str, dt, b: int):
    dims = [int(v) for v in name.rsplit("-", 1)[1].split("x")]
    u = lambda *s: (mx.sin(mx.arange(int(mx.prod(mx.array(s)).item())) * 0.7381) * 0.5).reshape(s).astype(dt)
    if name.startswith("dense"):
        k, n = dims
        x, w = u(b, k), u(n, k)
        mx.eval(x, w)
        return lambda: x @ w.T
    if name.startswith("affine4stream"):
        n, k = dims
        ws = [mx.quantize(u(n, k) + i * 1e-3, group_size=64, bits=4) for i in range(16)]
        x = u(b, k)
        mx.eval(ws, x)
        turn = [0]

        def op():
            wq, s, bi = ws[turn[0]]
            turn[0] = (turn[0] + 1) % len(ws)
            return mx.quantized_matmul(x, wq, s, bi, transpose=True, group_size=64, bits=4)
        return op
    if name.startswith("affine"):
        bits = int(name[len("affine")])
        n, k = dims
        wq, s, bi = mx.quantize(u(n, k), group_size=64, bits=bits)
        x = u(b, k)
        mx.eval(wq, s, bi, x)
        return lambda: mx.quantized_matmul(x, wq, s, bi, transpose=True, group_size=64, bits=bits)
    if name.startswith("gather_qmm"):
        n, k = dims
        wq, s, bi = mx.quantize(u(128, n, k), group_size=64, bits=4)
        ids = [(i * 37 + i // 8 * 11) % 128 for i in range(b * 8)]
        if "sorted" in name:
            x, idx = u(b * 8, 1, k), mx.array(sorted(ids), dtype=mx.uint32)
            mx.eval(wq, s, bi, x, idx)
            return lambda: mx.gather_qmm(x, wq, s, bi, rhs_indices=idx, transpose=True,
                                         group_size=64, bits=4, sorted_indices=True)
        x, idx = u(b, 1, 1, k), mx.array(ids, dtype=mx.uint32).reshape(b, 8)
        mx.eval(wq, s, bi, x, idx)
        return lambda: mx.gather_qmm(x, wq, s, bi, rhs_indices=idx, transpose=True,
                                     group_size=64, bits=4)
    n, k = dims
    w, x = u(64, k, n), u(b * 8, 1, k)
    idx = mx.array(sorted((i * 13) % 64 for i in range(b * 8)), dtype=mx.uint32)
    mx.eval(w, x, idx)
    return lambda: mx.gather_mm(x, w, rhs_indices=idx, sorted_indices=True)


def burst(op, count: int) -> float:
    t = time.perf_counter()
    pending = []
    for _ in range(count):
        pending.append(op())
        if len(pending) == 8:
            mx.async_eval(pending)
            pending = []
    mx.eval(pending)
    mx.eval(op())
    return (time.perf_counter() - t) / count


def mlx(keys) -> dict[tuple[str, str, int], float]:
    mx.set_default_device(mx.gpu)
    out = {}
    for name, dt, b in keys:
        op = mlx_op(name, DTYPES[dt], b)

        burst(op, 200)
        out[(name, dt, b)] = min(burst(op, 200) for _ in range(3)) * 1e6
    return out


def main() -> None:
    now = nuvec(ROOT)
    exact = nuvec(ROOT, exact=True)
    base = nuvec(Path(os.environ["BASE"])) if os.environ.get("BASE") else {}
    ref = mlx(sorted(now))
    nan = float("nan")
    print(f"{'case':38} {'dtype':9} {'B':>3} {'nuvec':>9} {'exact':>9} {'mlx':>9} "
          f"{'exact/B1':>9} {'exact/mlx':>9} {'0.3.0/now':>9}")
    ok_up_to = 16
    for key in sorted(now):
        name, dt, b = key
        r = base[key] / now[key] if key in base else nan
        if key in base and b >= 2 and r < 0.98:
            ok_up_to = min(ok_up_to, b - 1)
        e, one = exact.get(key, nan), exact.get((name, dt, 1), nan)
        print(f"{name:38} {dt:9} {b:>3} {now[key]:9.1f} {e:9.1f} {ref.get(key, nan):9.1f} "
              f"{e / one:9.2f} {e / ref.get(key, nan):9.2f} {r:9.2f}")
    if base:
        print(f"every case ≥ 0.98× of 0.3.0 up to B = {ok_up_to}")


if __name__ == "__main__":
    main()
