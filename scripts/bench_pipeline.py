# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""Pipeline- or tensor-parallel generation: nuvec against mlx-lm on the same machines.

Both engines run under `mlx.launch` with the same rank count, model and prompt:
- nuvec runs `pipeline_generate` (`--tensor`: `pipeline_generate --tensor`);
- mlx-lm runs `mlx_lm.benchmark --pipeline` (`--tensor`: `mlx_lm.benchmark`, which shards).
  mlx-lm pipelines only some models (Qwen3 MoE, DeepSeek, GLM 4 MoE, Qwen 3.5, …), and
  shards Qwen3, Llama and most MoE models.

The script prints each engine's best prompt and decode rates over two runs (mlx-lm: two
trials after its warm-up) and the nuvec/mlx-lm ratios. With
`--batch`, it also prints nuvec's Batcher decode throughput with and without micro-batches.
By default the ranks run on this Mac over loopback (`-n`). With `--hostfile`, they run on
the hostfile's Macs; the repository must be at the same path on every Mac.

Usage: `uv run scripts/bench_pipeline.py --model mlx-community/Qwen3-8B-4bit [-n 2]
[--prompt-tokens 2048] [--max-tokens 128] [--batch 8] [--hostfile hosts.json]`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# `mlx.launch` (its module has no `__main__` guard, so `python -m` would do nothing).
LAUNCH = [sys.executable, "-c", "from mlx._distributed_utils.launch import main; main()"]


def launch(args: argparse.Namespace, cmd: list[str]) -> str:
    where = ["--hostfile", args.hostfile] if args.hostfile else ["-n", str(args.n)]
    out = subprocess.run(
        [*LAUNCH, "--backend", "ring", *where, "--", *cmd],
        capture_output=True, text=True, timeout=3600,
    )
    # The launcher exits 0 when a rank fails; it only warns.
    if out.returncode != 0 or re.search(r"exited with code -?[1-9]", out.stdout + out.stderr):
        sys.exit(f"{cmd[0]} failed ({out.returncode}):\n{out.stderr[-4000:]}")
    return out.stdout


def rates(text: str) -> tuple[float, float]:
    """nuvec's `Prompt:`/`Generation:` lines, or the best of mlx-lm's `Trial` lines."""
    if trials := re.findall(r"Trial \d+: +prompt_tps=([\d.]+), generation_tps=([\d.]+)", text):
        return max(float(p) for p, _ in trials), max(float(g) for _, g in trials)
    prompt = re.search(r"Prompt: \d+ tokens, ([\d.]+) tokens-per-sec", text)
    decode = re.search(r"Generation: \d+ tokens, ([\d.]+) tokens-per-sec", text)
    if not (prompt and decode):
        sys.exit(f"no rates in the output:\n{text[-4000:]}")
    return float(prompt[1]), float(decode[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("-n", type=int, default=2)
    p.add_argument("--hostfile")
    p.add_argument("--prompt-tokens", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--tensor", action="store_true", help="tensor parallelism instead")
    args = p.parse_args()
    subprocess.run(
        ["cargo", "build", "-q", "--release", "-p", "nuvec-lm", "--example", "pipeline_generate"],
        cwd=ROOT, check=True,
    )
    exe = str(ROOT / "target/release/examples/pipeline_generate")
    # About `prompt-tokens` tokens of plain text (a word is about one token).
    prompt = " ".join(["word"] * args.prompt_tokens)
    common = ["--model", args.model, "--prompt", prompt, "--max-tokens", str(args.max_tokens)]
    mode = "tensor" if args.tensor else "pipeline"
    print(f"{mode} over {args.hostfile or f'{args.n} ranks on loopback'}, {args.model}, "
          f"{args.prompt_tokens}-token prompt, {args.max_tokens} tokens")
    print("| engine | prompt tok/s | decode tok/s |")
    print("|---|---:|---:|")
    best: dict[str, tuple[float, float]] = {}
    nuvec = [exe, *common, *(["--tensor"] if args.tensor else [])]
    for _ in range(2):
        pr, dr = rates(launch(args, nuvec))
        old = best.get("nuvec", (0.0, 0.0))
        best["nuvec"] = (max(old[0], pr), max(old[1], dr))
    best["mlx-lm"] = rates(launch(args, [
        sys.executable, "-m", "mlx_lm.benchmark", "--model", args.model,
        "-p", str(args.prompt_tokens), "-g", str(args.max_tokens), "-n", "2",
        *([] if args.tensor else ["--pipeline"]),
    ]))
    for name, (pr, dr) in best.items():
        print(f"| {name} | {pr:.1f} | {dr:.2f} |")
    (np, nd), (mp, md) = best["nuvec"], best["mlx-lm"]
    print(f"\ntime to first token nuvec/mlx-lm {mp / np:.2f}×; decode nuvec/mlx-lm {nd / md:.2f}×")
    if args.batch:
        for micro in (1, None):
            extra = ["--batch", str(args.batch)] + ([] if micro is None else ["--micro-batches", "1"])
            text = launch(args, [exe, *common, *extra])
            tps = float(re.search(r"Batch: \d+ requests, ([\d.]+) decode", text)[1])
            label = "one batch" if micro else "one micro-batch per stage"
            print(f"Batcher, {args.batch} requests, {label}: {tps:.1f} decode tok/s")


if __name__ == "__main__":
    main()
