# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""The first token of a short prompt, part by part, as mlx-lm's `generate_step` makes it:
the prompt but its last token (one chunk, evaluated with the cache), the cache clear, then the
last token's step, the next step submitted, and the first token read. Pairs with nuvec-lm's
`ttft_parts` example.

    uv run scripts/ttft_parts.py MODEL_DIR [--tokens 25] [--runs 20] [--no-clear]

Prints one JSON line per run: milliseconds per part, the cache bytes before and after the
clear, and the growth of active plus cached memory after it.
"""

import argparse
import json
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import make_prompt_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=25)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--no-clear", action="store_true")
    p.add_argument("--generate", action="store_true")
    a = p.parse_args()
    model, _ = load(a.model)
    prompt = mx.array([1000 + i for i in range(a.tokens)])

    def step(y, cache):
        # mlx-lm's `_step` with the greedy sampler.
        with mx.stream(generation_stream):
            logits = model(y[None], cache=cache)[:, -1, :]
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            return mx.argmax(logprobs, axis=-1)

    ms = lambda s, e: (e - s) * 1e3
    if a.generate:
        # `stream_generate` itself, timed as the parity record times it.
        from mlx_lm import stream_generate
        model_, tok = model, load(a.model)[1]
        tok.eos_token_ids = []
        ids = prompt.tolist()
        for run in range(a.runs + 1):
            t0 = time.perf_counter()
            first = None
            for r in stream_generate(model_, tok, ids, max_tokens=128):
                first = first or time.perf_counter()
            t3 = time.perf_counter()
            if run:
                print(json.dumps({"engine": "mlx", "mode": "generate",
                                  "first_token_ms": round(ms(t0, first), 3),
                                  "decode_ms": round(ms(first, t3), 3)}), flush=True)
        return
    for run in range(a.runs + 1):
        cache = make_prompt_cache(model)
        start = mx.get_active_memory() + mx.get_cache_memory()
        t0 = time.perf_counter()
        with mx.stream(generation_stream):
            model(prompt[:-1][None], cache=cache)
        mx.async_eval([c.state for c in cache])
        encoded = time.perf_counter()
        mx.eval([c.state for c in cache])
        t1 = time.perf_counter()
        before = mx.get_cache_memory()
        chunk_new = mx.get_active_memory() + before - start
        if not a.no_clear:
            mx.clear_cache()
        after = mx.get_cache_memory()
        base = mx.get_active_memory() + after
        t2 = time.perf_counter()
        y = step(prompt[-1:], cache)
        mx.async_eval(y)
        nxt = step(y, cache)
        mx.async_eval(nxt)
        t3 = time.perf_counter()
        mx.eval(y)
        token = y.item()
        t4 = time.perf_counter()
        grown = mx.get_active_memory() + mx.get_cache_memory() - base
        y = nxt
        for _ in range(8):
            n = step(y, cache)
            mx.async_eval(n)
            y.item()
            y = n
        y.item()
        if run == 0:
            continue
        print(json.dumps({"engine": "mlx", "tokens": a.tokens, "clear": not a.no_clear,
                          "chunk_ms": round(ms(t0, t1), 3), "encode_ms": round(ms(t0, encoded), 3), "clear_ms": round(ms(t1, t2), 3),
                          "submit_ms": round(ms(t2, t3), 3), "wait_ms": round(ms(t3, t4), 3),
                          "first_token_ms": round(ms(t0, t4), 3), "cache_before": before,
                          "cache_after": after, "grown": grown, "chunk_new_bytes": chunk_new, "token": token}), flush=True)


if __name__ == "__main__":
    main()
