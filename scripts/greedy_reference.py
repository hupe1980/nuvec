# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""mlx-lm's greedy tokens for `examples/greedy_parity.rs`.

`uv run scripts/greedy_reference.py MODEL [--out FILE] [--tokens 256] [--lengths 24,500,…]`
writes a reference in mrun's format (`scripts/gen_reference.py` there): per prompt the ids
(the model's chat template around a slice of nuvec's documentation), mlx-lm's greedy tokens
as `generate_step` produces them (prefill in chunks of 2,048, the last prompt token fed
alone, end tokens not stopping) and the two best tokens with their log-probabilities at each
step. Models come from the local Hugging Face cache.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = [24, 200, 1000, 1537, 2048, 3500, 4350, 8192]
MARKER = "@@@NUVEC-PROMPT@@@"


def frame(tokenizer):
    """Template ids before and after the user's text (none without a chat template)."""
    if not getattr(tokenizer, "chat_template", None):
        return [], []
    messages = [{"role": "user", "content": "Summarize this text.\n\n" + MARKER}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    before, after = text.split(MARKER)
    return (
        tokenizer.encode(before, add_special_tokens=False),
        tokenizer.encode(after, add_special_tokens=False),
    )


def prompts(tokenizer, lengths):
    docs = sorted((ROOT / "site/content").rglob("*.md")) + [ROOT / "README.md"]
    ids = tokenizer.encode("\n\n".join(p.read_text(encoding="utf-8") for p in docs), add_special_tokens=False)
    before, after = frame(tokenizer)
    out, offset = [], 0
    for length in lengths:
        # At least one token of text when the template is longer than the length.
        n = max(length - len(before) - len(after), 1)
        if offset + n > len(ids):
            offset = 0
        out.append(before + ids[offset : offset + n] + after)
        offset += n
    return out


def run(model, prompt, tokens):
    sampler = make_sampler(temp=0.0)
    ids, top2 = [], []
    for token, logprobs in generate_step(
        mx.array(prompt), model, max_tokens=tokens, sampler=sampler, prefill_step_size=2048
    ):
        order = mx.argsort(-logprobs.astype(mx.float32))[:2]
        values = logprobs[order].astype(mx.float32)
        ids.append(int(token))
        top2.append([[int(t), float(v)] for t, v in zip(order.tolist(), values.tolist())])
    return ids, top2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)))
    args = parser.parse_args()
    model, tokenizer = load(args.model)
    cases = []
    for prompt in prompts(tokenizer, [int(n) for n in args.lengths.split(",")]):
        tokens, top2 = run(model, prompt, args.tokens)
        cases.append({"prompt": prompt, "tokens": tokens, "top2": top2})
        print(f"{len(prompt)} prompt tokens", file=sys.stderr)
        mx.clear_cache()
    info = mx.device_info()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model,
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "prefill_step_size": 2048,
        "machine": {"chip": info.get("device_name"), "gpu_arch": info.get("architecture"), "macos": platform.mac_ver()[0]},
        "cases": cases,
    }, separators=(",", ":")))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
