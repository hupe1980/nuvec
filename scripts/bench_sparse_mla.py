# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5", "mlx-lm==0.32.2"]
# ///
"""Single DeepSeek-V3.2 attention layer, release and pinned upstream peers on the same Mac.

Use just bench-sparse-mla --output report.json. All processes consume identical
safetensors bytes. The timers cover an append and evaluation. Prefix construction,
host reads, Python startup, and fixture generation stay outside the timed region.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from bench_parity import write_atomic

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "6fe0ce48c90cec11f2c305830ceba461051c33b2"
CONFIG = {
    "hidden_size": 7168,
    "num_attention_heads": 128,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
    "max_position_embeddings": 163840,
    "rope_scaling": {
        "type": "yarn",
        "factor": 40.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
}


def make_layer():
    try:
        from mlx_lm.models.deepseek_v32.config import ModelArgs
        from mlx_lm.models.deepseek_v32.layers import DeepseekV32Attention
    except ModuleNotFoundError:
        from mlx_lm.models.deepseek_v32 import DeepseekV32Attention, ModelArgs
    return DeepseekV32Attention(ModelArgs(**CONFIG))


def fixture(path, length):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.random.seed(119)
    layer = make_layer()
    layer.set_dtype(mx.bfloat16)
    data = dict(tree_flatten(layer.parameters()))
    data["hidden"] = mx.random.uniform(-0.75, 0.75, (1, length, 7168)).astype(mx.bfloat16)
    mx.save_safetensors(str(path), data)


def mlx_worker(path, prefix, rows):
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    layer = make_layer()
    data = mx.load(str(path))
    hidden = data.pop("hidden")
    layer.load_weights(list(data.items()))
    del data
    # Build both histories directly from the same prefix hidden states. No attention
    # output from the old prefix is needed for an isolated layer's new queries.
    prev, x = hidden[:, :prefix], hidden[:, prefix : prefix + rows]
    packed = layer.kv_a_proj_with_mqa(prev)
    lat, pos = mx.split(packed, [CONFIG["kv_lora_rank"]], axis=-1)
    lat = layer.kv_a_layernorm(lat)[:, None]
    pos = layer.rope(pos[:, None], offset=0)
    index = layer.indexer.rope(layer.indexer.k_norm(layer.indexer.wk(prev))[:, None], offset=0)
    mask = (
        None if rows == 1 else mx.arange(prefix + rows) <= mx.arange(prefix, prefix + rows)[:, None]
    )
    mx.eval(lat, pos, index, x, *([] if mask is None else [mask]))
    mx.clear_cache()
    print("READY", flush=True)
    for line in sys.stdin:
        if line.strip() != "measure":
            break
        # Same retained prefix for each append; allocate any capacity expansion inside
        # the timer, as for the immutable Rust session. Disposable wrappers isolate runs.
        cache = [KVCache(), KVCache()]
        cache[0].keys, cache[0].values, cache[0].offset = lat, pos, prefix
        cache[1].keys, cache[1].values, cache[1].offset = index, index, prefix
        mx.reset_peak_memory()
        start = time.perf_counter_ns()
        out = layer(x, mask=mask, cache=cache)
        mx.eval(out, cache[0].keys, cache[0].values, cache[1].keys)
        sample = {
            "ms": (time.perf_counter_ns() - start) / 1e6,
            "peak_bytes": mx.get_peak_memory(),
            "history_bytes": (prefix + rows) * 1408,
            "sample": out[0, -1, :8].tolist(),
        }
        print(json.dumps(sample), flush=True)
        del out, cache


@contextlib.contextmanager
def worker(command):
    with subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as p:
        try:
            ready = p.stdout.readline().strip()
            if ready != "READY":
                raise RuntimeError(f"worker {command[0]} failed to initialize: {ready!r}")
            yield p
        finally:
            if p.poll() is None:
                p.stdin.close()
                p.wait()


def cell(commands, warmup, samples):
    with contextlib.ExitStack() as stack:
        workers = {k: stack.enter_context(worker(v)) for k, v in commands.items()}
        records = {k: [] for k in workers}
        names = list(workers)
        for i in range(warmup + samples):
            for name in names if i % 2 == 0 else reversed(names):
                p = workers[name]
                p.stdin.write("measure\n")
                p.stdin.flush()
                item = json.loads(p.stdout.readline())
                if i >= warmup:
                    records[name].append(item)
        # After paired timing, insert a synchronization at the public selection boundary.
        # This includes graph construction and all dependencies of the selected indices.
        # The second timer covers the remaining projections, attention and readout.
        stages = []
        for _ in range(3):
            p = workers["nuvec"]
            p.stdin.write("stages\n")
            p.stdin.flush()
            stages.append(json.loads(p.stdout.readline()))
        return records, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, metavar=("FILE", "PREFIX", "ROWS"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 4096, 16384])
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--release-only", action="store_true", help="Omit the dated upstream comparator"
    )
    args = parser.parse_args()
    if args.worker:
        path, prefix, rows = args.worker
        mlx_worker(path, int(prefix), int(rows))
        return
    if min(*args.contexts, *args.rows, args.samples) < 1 or args.warmups < 0:
        parser.error("contexts, rows and samples must be positive; warmups must be nonnegative")
    if len(set(args.contexts)) != len(args.contexts) or len(set(args.rows)) != len(args.rows):
        parser.error("each context and row count must be unique")
    if max(args.contexts) + max(args.rows) > CONFIG["max_position_embeddings"]:
        parser.error("workload exceeds the checkpoint position limit")
    import mlx.core as mx

    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    info = mx.device_info()
    if not str(info.get("architecture", "")).startswith("applegpu"):
        parser.error("a physical Apple GPU is required")
    # The released batched layer expands dense keys and values and computes rotary scores
    # for every query/key pair. Include the input and histories retained by each worker.
    queries = max(args.rows)
    extent = max(args.contexts) + queries
    workers = 2 if args.release_only else 3
    retained = workers * extent * (2 * CONFIG["hidden_size"] + 1408)
    heads = CONFIG["num_attention_heads"]
    rotary_scores = 2 * heads * queries * extent
    dense_kv = 2 * heads * extent * (CONFIG["qk_nope_head_dim"] + CONFIG["v_head_dim"])
    minimum = max(12 * 1024**3, retained + max(rotary_scores, dense_kv if queries > 1 else 0))
    if minimum > physical:
        parser.error(f"workload requires at least {minimum / 1024**3:.1f} GiB of physical memory")
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "nuvec-lm",
            "--example",
            "sparse_mla_bench",
        ],
        cwd=ROOT,
        check=True,
    )
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT
        )
    )
    executable = Path(metadata["target_directory"]) / "release/examples/sparse_mla_bench"
    with executable.open("rb") as binary:
        binary_digest = hashlib.file_digest(binary, "sha256").hexdigest()
    report = {
        "chip": info["device_name"],
        "architecture": info["architecture"],
        "os": platform.mac_ver()[0],
        "memory_bytes": physical,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "binary_sha256": binary_digest,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "load_average_at_start": os.getloadavg(),
        "mlx": mx.__version__,
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "upstream": None if args.release_only else UPSTREAM,
        "dtype": "bfloat16",
        "route": "absorbed MLA composition",
        "seed": 119,
        "warmups": args.warmups,
        "samples": args.samples,
        "config": CONFIG,
        "cells": [],
    }
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "layer.safetensors"
        fixture(path, max(args.contexts) + max(args.rows))
        mx.clear_cache()
        with path.open("rb") as fp:
            report["sha256"] = hashlib.file_digest(fp, "sha256").hexdigest()
        for prefix in args.contexts:
            for rows in args.rows:
                inputs = [str(path), str(prefix), str(rows)]
                commands = {
                    "nuvec": [str(executable), *inputs],
                    "release": [sys.executable, __file__, "--worker", *inputs],
                }
                if not args.release_only:
                    commands["upstream"] = [
                        "uv",
                        "run",
                        "--no-project",
                        "--with",
                        "mlx==0.32.5",
                        "--with",
                        f"mlx-lm @ git+https://github.com/ml-explore/mlx-lm@{UPSTREAM}",
                        "python",
                        __file__,
                        "--worker",
                        *inputs,
                    ]
                result = {"prefix": prefix, "rows": rows}
                try:
                    raw, stages = cell(commands, args.warmups, args.samples)
                    # Eight checked outputs are a timed-path smoke test. Full-array validation
                    # belongs to the independent offline fixture suite.
                    oracle = raw["release"][0]["sample"]
                    for name, trials in raw.items():
                        for trial in trials:
                            if trial["history_bytes"] != (prefix + rows) * 1408:
                                raise ValueError(f"{name}: unexpected logical history size")
                            if len(trial["sample"]) != 8 or any(
                                not math.isfinite(a)
                                or not math.isfinite(b)
                                or abs(a - b) > 0.03 * (1 + abs(b))
                                for a, b in zip(trial["sample"], oracle)
                            ):
                                raise ValueError(f"{name}: sampled outputs diverged from release")
                    med = {
                        k: statistics.median(x["ms"] for x in series) for k, series in raw.items()
                    }
                    result.update(
                        raw=raw,
                        nuvec_stages=stages,
                        medians_ms=med,
                        ratios={k: med["nuvec"] / v for k, v in med.items() if k != "nuvec"},
                    )
                    print(prefix, rows, result["ratios"], flush=True)
                except (RuntimeError, OSError, ValueError) as exc:
                    result["unavailable"] = str(exc)
                    print(prefix, rows, "unavailable", exc, flush=True)
                report["cells"].append(result)
                if args.output:
                    write_atomic(args.output, report)
    if not args.output:
        print(json.dumps(report, indent=2))
    if any("unavailable" in c for c in report["cells"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
