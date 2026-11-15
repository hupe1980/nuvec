# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2", "dflash-mlx==0.1.8"]
# ///
"""DFlash speculative decoding against MLX on the same machine: Qwen3-8B 4-bit with
z-lab/Qwen3-8B-DFlash-b16 (4-bit drafter, group 64), the GSM8K "Natalia" problem in Qwen3's
chat format (non-thinking), up to 256 greedy tokens. Plain decoding (mlx-lm, nuvec) and
DFlash (dflash-mlx, nuvec); decode tokens/s, the best of `--runs`, and the share of
generated tokens that came from drafts.

Usage: `just bench-dflash`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

MODEL = "mlx-community/Qwen3-8B-4bit"
DRAFT = "z-lab/Qwen3-8B-DFlash-b16"
QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
    "in May. How many clips did Natalia sell altogether in April and May?"
)
ROOT = Path(__file__).resolve().parent.parent


def prompt_text(tokenizer) -> str:
    messages = [{"role": "user", "content": QUESTION}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def mlx_plain(model, tokenizer, prompt: str, max_tokens: int) -> float:
    from mlx_lm import stream_generate

    last = None
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        last = r
    return last.generation_tps


def mlx_dflash(prompt: str, max_tokens: int) -> tuple[float, float]:
    from dflash_mlx.engine.events import SummaryEvent
    from dflash_mlx.generate import generation_tps_from_summary
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    context = build_offline_runtime_context()
    bundle = load_runtime_bundle(
        model_ref=MODEL, draft_ref=DRAFT, draft_quant="w4:gs64", verify_config=context.verify
    )
    best = (0.0, 0.0)
    for _ in range(2):
        summary = None
        for event in stream_dflash_generate(
            target_model=bundle.target_model,
            target_ops=bundle.target_ops,
            tokenizer=bundle.tokenizer,
            draft_model=bundle.draft_model,
            draft_backend=bundle.draft_backend,
            prompt=prompt,
            max_new_tokens=max_tokens,
            use_chat_template=False,
            stop_token_ids=get_stop_token_ids(bundle.tokenizer),
            runtime_context=context,
        ):
            if isinstance(event, SummaryEvent):
                summary = event
        tps = generation_tps_from_summary(summary)
        if tps > best[0]:
            best = (tps, float(summary.acceptance_ratio))
    return best


def nuvec(prompt_file: Path, max_tokens: int, runs: int, extra: list[str]) -> dict:
    cmd = [
        "cargo", "run", "--release", "-q", "-p", "nuvec-lm", "--example", "bench", "--",
        "--model", MODEL, "--repeats", "1", "--runs", str(runs),
        "--max-tokens", str(max_tokens), "--text-file", str(prompt_file), *extra,
    ]
    out = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    return max(rows, key=lambda r: r["decode"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from mlx_lm import load

    draft_dir = snapshot_download(DRAFT)
    model, tokenizer = load(MODEL)
    prompt = prompt_text(tokenizer)
    plain_mlx = max(mlx_plain(model, tokenizer, prompt, args.max_tokens) for _ in range(args.runs))
    del model
    time.sleep(1)
    dflash_mlx, accept_mlx = mlx_dflash(prompt, args.max_tokens)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = Path(f.name)
    plain_nv = nuvec(prompt_file, args.max_tokens, args.runs, [])
    dflash_nv = nuvec(
        prompt_file, args.max_tokens, args.runs,
        ["--draft", "7", "--drafter", "dflash", "--dflash-dir", draft_dir, "--dflash-bits", "4"],
    )
    # As dflash-mlx counts it: generated tokens that came from drafts.
    accept_nv = dflash_nv["accepted"] / max(dflash_nv["tokens"], 1)

    print(f"{'':<8}{'plain tok/s':>13}{'DFlash tok/s':>14}{'speedup':>9}{'drafted':>10}")
    print(f"{'MLX':<8}{plain_mlx:>13.1f}{dflash_mlx:>14.1f}{dflash_mlx / plain_mlx:>8.2f}×{accept_mlx:>10.0%}")
    print(
        f"{'nuvec':<8}{plain_nv['decode']:>13.1f}{dflash_nv['decode']:>14.1f}"
        f"{dflash_nv['decode'] / plain_nv['decode']:>8.2f}×{accept_nv:>10.0%}"
    )
    print(f"nuvec DFlash / dflash-mlx: {dflash_nv['decode'] / dflash_mlx:.2f}×")


if __name__ == "__main__":
    main()
