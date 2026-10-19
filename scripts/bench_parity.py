# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""Parity table: nuvec-lm vs mlx-lm over a roster of models, on the same machine and day.

For each model in the local Hugging Face cache (nothing is downloaded; missing models are
listed as skipped), both engines load the model once per pass, warm up, and generate
greedily after prompts of three lengths. Passes alternate between the engines (nuvec,
mlx-lm, nuvec, mlx-lm) and the table shows medians. Both are timed the same way, from
outside the generation loop: prefill is prompt tokens over the time to the first token
(memory wired before the clock, as mlx-lm), decode is the tokens after the first over the
time after it.

Usage: `just bench-parity [models…]` (default: the roster below).
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROSTER = [
    ("mlx-community/Qwen3-0.6B-4bit", "dense"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "dense"),
    ("mlx-community/Qwen3-8B-4bit", "dense"),
    ("mlx-community/Qwen3.5-9B-4bit", "hybrid"),
    ("mlx-community/Qwen3.8-27B-4bit", "hybrid"),
    ("mlx-community/Qwen3-30B-A3B-4bit", "MoE"),
    ("mlx-community/Qwen3.6-35B-A3B-4bit", "hybrid MoE"),
]
TEXT = ("The lighthouse keeper climbed the spiral stairs every evening, counting the steps "
        "and listening to the sea against the rocks below. ")
REPEATS = [1, 40, 160]  # about 25, 960 and 3,840 prompt tokens
RUNS = 2  # per pass
PASSES = 2
MAX_TOKENS = 128


def snapshot(model: str) -> Path | None:
    """The cached snapshot directory: the one `refs/main` names, else the newest."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    repo = hub / f"models--{model.replace('/', '--')}"
    ref = repo / "refs" / "main"
    if ref.exists():
        path = repo / "snapshots" / ref.read_text().strip()
        return path if (path / "config.json").exists() else None
    found = sorted((p.parent for p in (repo / "snapshots").glob("*/config.json")),
                   key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def mlx_worker(model: str) -> None:
    """One pass of mlx-lm in this process, printing nuvec's JSON lines."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    m, tokenizer = load(model)

    def run(prompt: list[int]) -> tuple[float, float, int]:
        start = time.perf_counter()
        first = last = None
        count = 0
        for _ in stream_generate(m, tokenizer, prompt, max_tokens=MAX_TOKENS):
            now = time.perf_counter()
            first = first or now
            last = now
            count += 1
        prefill = len(prompt) / (first - start)
        decode = (max(count, 2) - 1) / (last - first) if last > first else 0.0
        return prefill, decode, count

    run(tokenizer.encode(TEXT))
    for r in REPEATS:
        ids = tokenizer.encode(TEXT * r)
        for _ in range(RUNS):
            mx.reset_peak_memory()
            prefill, decode, tokens = run(ids)
            print(json.dumps({"repeats": r, "prompt_tokens": len(ids), "prefill": prefill,
                              "decode": decode, "tokens": tokens,
                              "peak_bytes": mx.get_peak_memory()}), flush=True)


def lines(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def nuvec(model: str) -> list[dict]:
    out = subprocess.run(
        [str(ROOT / "target/release/examples/bench"), "--model", str(snapshot(model)), "--text", TEXT,
         "--repeats", ",".join(map(str, REPEATS)), "--runs", str(RUNS),
         "--max-tokens", str(MAX_TOKENS)],
        capture_output=True, text=True, check=True)
    return lines(out.stdout)


def mlx(model: str) -> list[dict]:
    out = subprocess.run([sys.executable, __file__, "--mlx-worker", str(snapshot(model))],
                         capture_output=True, text=True, check=True,
                         env={**os.environ, "HF_HUB_OFFLINE": "1"})
    return lines(out.stdout)


def median(rows: list[dict], r: int, key: str) -> float:
    return statistics.median(row[key] for row in rows if row["repeats"] == r)


def swap_gb() -> str:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
    used = out.stdout.split("used = ")[1].split("M")[0] if "used = " in out.stdout else "?"
    return f"{float(used) / 1024:.1f}" if used != "?" else "?"


def main(models: list[str]) -> None:
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "nuvec-lm", "--example", "bench"],
                   cwd=ROOT, check=True)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip()
    print(f"{chip}, macOS {platform.mac_ver()[0]}; mlx 0.32.4, mlx-lm 0.32.2; greedy, "
          f"{MAX_TOKENS} tokens, medians of {RUNS * PASSES} runs per engine; "
          f"swap in use at start: {swap_gb()} GB")
    print()
    print("| model | kind | prompt | prefill nuvec / mlx-lm (tok/s) | ratio "
          "| decode nuvec / mlx-lm (tok/s) | ratio |")
    print("|---|---|---:|---:|---:|---:|---:|")
    kinds = dict(ROSTER)
    decode_ratios, prefill_ratios, skipped, failed = [], [], [], []
    for model in models:
        if snapshot(model) is None:
            skipped.append(model)
            continue
        ours, theirs = [], []
        try:
            for _ in range(PASSES):
                ours += nuvec(model)
                theirs += mlx(model)
        except subprocess.CalledProcessError as e:
            failed.append(f"{model} ({Path(e.cmd[0]).name}: {e.stderr.strip().splitlines()[-1:]})")
            continue
        for r in REPEATS:
            n = next(row["prompt_tokens"] for row in ours if row["repeats"] == r)
            m = next(row["prompt_tokens"] for row in theirs if row["repeats"] == r)
            tokens = f"{n}" if n == m else f"{n} / {m}"
            np_, mp = median(ours, r, "prefill"), median(theirs, r, "prefill")
            nd, md = median(ours, r, "decode"), median(theirs, r, "decode")
            prefill_ratios.append(np_ / mp)
            decode_ratios.append(nd / md)
            print(f"| {model.split('/')[1]} | {kinds.get(model, '')} | {tokens} "
                  f"| {np_:.1f} / {mp:.1f} | {np_ / mp:.2f}× | {nd:.1f} / {md:.1f} "
                  f"| {nd / md:.2f}× |", flush=True)
    if decode_ratios:
        print()
        print(f"Geometric means: prefill {statistics.geometric_mean(prefill_ratios):.2f}×, "
              f"decode {statistics.geometric_mean(decode_ratios):.2f}× "
              f"(median decode {statistics.median(decode_ratios):.2f}×)")
    if skipped:
        print(f"Skipped (not in the local cache): {', '.join(skipped)}")
    if failed:
        print(f"Failed: {'; '.join(failed)}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--mlx-worker":
        mlx_worker(sys.argv[2])
    else:
        main(sys.argv[1:] or [m for m, _ in ROSTER])
