# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""Text generation: nuvec-lm vs mlx-lm on the same machine and model.

For each prompt length, runs `mlx_lm generate` and nuvec-lm's `generate` example (greedy,
no chat template) and prints prompt (prefill) and generation throughput, as both report
them. The model must be in the local Hugging Face cache (nothing is downloaded).

Usage: `just bench-lm [model] [max_tokens]` (default mlx-community/Qwen3-8B-4bit, 256).
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-8B-4bit"
MAX_TOKENS = sys.argv[2] if len(sys.argv) > 2 else "256"
PARAGRAPH = ("The lighthouse keeper climbed the spiral stairs every evening, counting the "
             "steps and listening to the sea against the rocks below. ")
REPEATS = [1, 40, 160]  # about 20, 800 and 3200 prompt tokens


def numbers(text: str) -> tuple[int, float, float]:
    prompt = re.search(r"Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec", text)
    gen = re.search(r"Generation: \d+ tokens, ([\d.]+) tokens-per-sec", text)
    return int(prompt.group(1)), float(prompt.group(2)), float(gen.group(1))


def mlx(prompt: str) -> tuple[int, float, float]:
    out = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "generate", "--model", MODEL, "--prompt", prompt,
         "--max-tokens", MAX_TOKENS, "--ignore-chat-template"],
        capture_output=True, text=True, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    return numbers(out.stdout)


def nuvec(prompt: str) -> tuple[int, float, float]:
    out = subprocess.run(
        ["cargo", "run", "-q", "--release", "-p", "nuvec-lm", "--example", "generate", "--",
         "--model", MODEL, "--prompt", prompt, "--max-tokens", MAX_TOKENS],
        cwd=ROOT, capture_output=True, text=True, check=True)
    return numbers(out.stdout)


def main() -> None:
    print(f"{MODEL}, {MAX_TOKENS} generated tokens, greedy; {platform.machine()} "
          f"{platform.mac_ver()[0]}")
    print("| prompt tokens | nuvec prefill tok/s | mlx-lm prefill tok/s | nuvec decode tok/s "
          "| mlx-lm decode tok/s | decode nuvec/mlx-lm |")
    print("|---:|---:|---:|---:|---:|---:|")
    for repeats in REPEATS:
        prompt = PARAGRAPH * repeats
        n, np_, nd = nuvec(prompt)
        _, mp, md = mlx(prompt)
        print(f"| {n} | {np_:.1f} | {mp:.1f} | {nd:.1f} | {md:.1f} | {nd / md:.2f}× |")


if __name__ == "__main__":
    main()
