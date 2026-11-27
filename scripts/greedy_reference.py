# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.5", "mlx-lm==0.32.2"]
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
import hashlib
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
        # For a prompt shorter than the chat template, use raw document tokens.
        # This keeps the requested length without truncating the template's control tokens.
        prefix, suffix = (before, after) if length > len(before) + len(after) else ([], [])
        n = length - len(prefix) - len(suffix)
        if offset + n > len(ids):
            offset = 0
        out.append(prefix + ids[offset : offset + n] + suffix)
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
    parser.add_argument("--prompts-from", type=Path, help="use the exact prompt IDs in a frozen corpus")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    model, tokenizer = load(args.model)
    cases = []
    lengths = [int(n) for n in args.lengths.split(",")]
    if args.prompts_from:
        selected = [case["prompt"] for case in json.loads(args.prompts_from.read_text())["cases"]]
    else:
        selected = prompts(tokenizer, lengths)
    if [len(case) for case in selected] != lengths or min(lengths) < 1:
        parser.error("prompt IDs do not match the requested nonempty lengths")
    for prompt in selected:
        tokens, top2 = run(model, prompt, args.tokens)
        cases.append({"prompt": prompt, "tokens": tokens, "top2": top2})
        print(f"{len(prompt)} prompt tokens", file=sys.stderr)
        mx.clear_cache()
    info = mx.device_info()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model,
        "snapshot": Path(args.model).resolve().name if Path(args.model).is_dir() else None,
        "prompt_sha256": hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest(),
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "prefill_step_size": 2048,
        "machine": {"chip": info.get("device_name"), "gpu_arch": info.get("architecture"), "macos": platform.mac_ver()[0]},
        "cases": cases,
    }, separators=(",", ":")))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
