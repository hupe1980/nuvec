# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""Speculative decoding: nuvec-lm vs mlx-lm on the same machine, the committed natural
prompt, 256 greedy tokens, one model process at a time.

- mlx-lm: plain decode, and its speculative decoding with a draft model where a model of
  the same tokenizer is cached (Qwen3-0.6B for Qwen3-8B).
- nuvec-lm (`bench`): plain decode, MTP drafts (Qwen 3.5–3.8, with the `mtp.*` file of
  `scripts/extract_mtp.py` in `~/.cache/nuvec/mtp/`) and prompt lookup.

mlx-vlm's Qwen 3.5 MTP speculation needs the original checkpoint's `mtp.*` weights and is
not run here (the speculative-decoding page has one manual measurement). Usage: `just bench-spec`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = (ROOT / "crates/nuvec-lm/benches/prompts/natural.txt").read_text()
MTP = Path.home() / ".cache/nuvec/mtp"
ROSTER = [
    ("mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-0.6B-4bit", None),
    ("mlx-community/Qwen3.5-9B-4bit", None, "Qwen--Qwen3.5-9B"),
    ("mlx-community/Qwen3.8-27B-4bit", None, "Qwen--Qwen3.8-27B"),
    ("mlx-community/Qwen3.6-35B-A3B-4bit", None, "Qwen--Qwen3.6-35B-A3B"),
]


def cached(model: str) -> bool:
    return (Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--"))).is_dir()


def mlx(model: str, draft: str | None) -> float:
    cmd = [sys.executable, "-m", "mlx_lm", "generate", "--model", model, "--prompt", PROMPT,
           "--max-tokens", "256", "--ignore-chat-template"]
    if draft:
        cmd += ["--draft-model", draft, "--num-draft-tokens", "3"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(re.search(r"Generation: \d+ tokens, ([\d.]+) tokens-per-sec", out).group(1))


def nuvec(model: str, extra: list[str]) -> float:
    exe = ROOT / "target/release/examples/bench"
    cmd = [str(exe), "--model", model, "--text", PROMPT, "--repeats", "1", "--runs", "2",
           "--max-tokens", "256", "--ignore-eos", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])["decode"]


def main() -> None:
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "nuvec-lm", "--example", "bench"],
                   cwd=ROOT, check=True)
    print("| model | mlx-lm plain | mlx-lm draft model | nuvec plain | nuvec MTP | nuvec lookup |")
    print("|---|---:|---:|---:|---:|---:|")
    for model, draft, mtp in ROSTER:
        if not cached(model):
            continue
        cells = [mlx(model, None)]
        cells.append(mlx(model, draft) if draft and cached(draft) else None)
        cells.append(nuvec(model, []))
        weights = MTP / f"{mtp}.safetensors" if mtp else None
        cells.append(nuvec(model, ["--draft", "1", "--drafter", "mtp", "--mtp-weights", str(weights)])
                     if weights and weights.is_file() else None)
        cells.append(nuvec(model, ["--draft", "4", "--drafter", "lookup", "--policy", "adaptive"]))
        print(f"| {model.split('/')[-1]} | " + " | ".join("—" if c is None else f"{c:.1f}" for c in cells) + " |",
              flush=True)


if __name__ == "__main__":
    main()
