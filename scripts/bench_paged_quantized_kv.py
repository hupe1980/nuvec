# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5", "mlx-lm==0.32.2"]
# ///
"""Compare dense and eight-bit pages on the same full-attention Qwen3 and token IDs.

Run with uv from the repository root. Set --quality for two fixed, 32K-token
corpora drawn from the v0.15.0 tree. The MLX reference is serial contiguous
eight-bit inference, reported separately from multi-request throughput.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from bench_parity import ROOT, build, environment, snapshot, source_identity


def run(*cmd):
    return subprocess.check_output(cmd, text=True)


def binary_sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(**values):
    print(json.dumps(values), flush=True)


def mlx_reference(model_id, prompts, steps):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import QuantizedKVCache

    model, _ = load(model_id)
    mx.eval(model.parameters())
    seconds = 0.0
    for prompt in prompts:
        caches = [QuantizedKVCache(group_size=64, bits=8) for _ in model.layers]
        ids = mx.array(prompt)
        for i in range(0, len(prompt) - 1, 2048):
            model(ids[None, i : min(i + 2048, len(prompt) - 1)], cache=caches)
            mx.eval([c.state for c in caches])
        tok = model(ids[None, -1:], cache=caches)[:, -1].argmax(-1)
        mx.eval(tok)
        start = time.monotonic()
        for _ in range(steps - 1):
            tok = model(tok[:, None], cache=caches)[:, -1].argmax(-1)
            mx.eval(tok)
        seconds += time.monotonic() - start
    return seconds


def quality(checkpoint, temp):
    for corpus, globs in [
        ("text", ["README.md", "CHANGELOG.md", "site/content/docs"]),
        ("code", ["crates/nuvec-core/src"]),
    ]:
        files = sorted(
            run(
                "git", "ls-tree", "-r", "--name-only", "v0.15.0", "--", *globs
            ).splitlines()
        )
        files = [p for p in files if p.endswith(".md" if corpus == "text" else ".rs")]
        data = "\n\n".join(run("git", "show", f"v0.15.0:{f}") for f in files)
        path = temp / corpus
        path.write_text(data)
        records = {}
        for mode, flags in [("dense", []), ("affine8-g64", ["--kv-bits", "8"])]:
            rec = json.loads(
                run(
                    "target/release/examples/perplexity",
                    "--model",
                    str(checkpoint),
                    "--text-file",
                    str(path),
                    "--tokens",
                    "32768",
                    "--chunk",
                    "512",
                    *flags,
                )
            )
            if rec["tokens"] != 32768:
                raise ValueError(f"{corpus}: only {rec['tokens']} scored tokens")
            records[mode] = rec
        write_json(
            corpus=corpus,
            snapshot=checkpoint.name,
            binary_sha256=binary_sha256(ROOT / "target/release/examples/perplexity"),
            git_tag="v0.15.0",
            paths=files,
            sha256=hashlib.sha256(data.encode()).hexdigest(),
            chunk=512,
            observations=records,
            ppl_ratio=records["affine8-g64"]["perplexity"]
            / records["dense"]["perplexity"],
        )


def benchmark(args, checkpoint, temp, prompts):
    candidate, _ = build(None)
    baseline, _ = build(args.baseline)
    path = temp / "ids"
    path.write_text("".join(",".join(map(str, p)) + "\n" for p in prompts))
    modes = {
        "baseline-dense": (baseline, []),
        "dense": (candidate, ["--timing-only"]),
        "affine8-g64": (candidate, ["--timing-only", "--paged-kv8"]),
    }
    if args.gather_binary:
        modes["affine8-g64-gather"] = (
            args.gather_binary,
            ["--timing-only", "--paged-kv8"],
        )
    times = {m: [] for m in modes}
    write_json(
        model=args.model,
        snapshot=checkpoint.name,
        lengths=list(map(len, prompts)),
        prompt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        steps=args.steps,
        baseline=run("git", "rev-parse", args.baseline).strip(),
        source_sha256=source_identity(),
        candidate_binary_sha256=binary_sha256(candidate),
        baseline_binary_sha256=binary_sha256(baseline),
        gather_binary_sha256=binary_sha256(args.gather_binary)
        if args.gather_binary
        else None,
        environment=environment(None),
        thermal_status_at_start=run("pmset", "-g", "therm").splitlines(),
    )
    # Rotate processes across the same host and warmed model files.
    for rep in range(args.repeats + 1):
        order = list(modes)
        offset = rep % len(order)
        for mode in order[offset:] + order[:offset]:
            binary, flags = modes[mode]
            row = json.loads(
                run(
                    binary,
                    "--model",
                    str(checkpoint),
                    "--batch",
                    str(args.batch),
                    "--ids-file",
                    str(path),
                    "--max-tokens",
                    str(args.steps),
                    *flags,
                )
            )
            if (
                row["total_tokens"] != args.batch * args.steps
                or row["decode_tokens"] != args.batch * (args.steps - 1)
                or not math.isfinite(row["decode_wall_s"])
                or row["decode_wall_s"] <= 0
            ):
                raise ValueError(f"{mode}: incomplete or invalid timing: {row}")
            rate = row["decode_tokens"] / row["decode_wall_s"]
            if rep:
                times[mode].append(rate)
            write_json(
                repeat=rep,
                warmup=rep == 0,
                mode=mode,
                tokens_per_s=rate,
                load_average_after=os.getloadavg(),
                **row,
            )
    medians = {mode: statistics.median(rates) for mode, rates in times.items()}
    write_json(
        medians_tokens_per_s=medians,
        packed_over_dense=medians["affine8-g64"] / medians["dense"],
        gather_over_dense=medians["affine8-g64-gather"] / medians["dense"]
        if args.gather_binary
        else None,
        dense_over_baseline=medians["dense"] / medians["baseline-dense"],
        thermal_status_at_end=run("pmset", "-g", "therm").splitlines(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    parser.add_argument("--baseline", default="v0.15.0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--gather-binary",
        type=Path,
        metavar="BENCH",
        help="also time a release bench built with per-row gathers from the packed KV pool",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quality", action="store_true")
    mode.add_argument("--mlx-only", action="store_true")
    args = parser.parse_args()
    if (
        min(args.repeats, args.batch) < 1
        or args.length <= 16 * (args.batch - 1)
        or args.steps < 2
    ):
        parser.error(
            "require positive repeats and batch, a prompt for each row, and at least two decode tokens"
        )
    checkpoint = snapshot(args.model)
    if checkpoint is None:
        parser.error(f"no cached checkpoint for {args.model}")
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        if args.quality:
            subprocess.run(
                [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "nuvec-lm",
                    "--example",
                    "perplexity",
                ],
                cwd=ROOT,
                check=True,
            )
            quality(checkpoint, temp)
            return
        lengths = [
            args.length if i == 0 else args.length + 1 - 16 * i
            for i in range(args.batch)
        ]
        prompts = [
            [(j + i) % 514 + 20 for j in range(n)] for i, n in enumerate(lengths)
        ]
        if args.mlx_only:
            env = environment(None)
            elapsed = mlx_reference(str(checkpoint), prompts, args.steps)
            count = args.batch * (args.steps - 1)
            write_json(
                mode="mlx-lm-serial-contiguous-affine8",
                model=args.model,
                snapshot=checkpoint.name,
                lengths=lengths,
                steps=args.steps,
                decode_tokens=count,
                decode_wall_s=elapsed,
                decode_tokens_per_s=count / elapsed,
                environment=env,
            )
        else:
            benchmark(args, checkpoint, temp, prompts)


if __name__ == "__main__":
    main()
