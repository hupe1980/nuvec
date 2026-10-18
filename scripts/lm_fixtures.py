# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""Records tiny random models of each Llama-family variant with mlx-lm, and their outputs,
as test fixtures for nuvec-lm.

Each case is a directory under `crates/nuvec-lm/tests/fixtures/` with `config.json` and
`model.safetensors` (as a Hugging Face or MLX checkpoint) and `expected.safetensors`: the
logits of a fixed prompt (run at once) and 12 greedy tokens generated after it
(`generate_step`, with the KV cache). Runs on MLX's GPU, as nuvec's tests do.

Usage: `just lm-fixtures` (or `uv run scripts/lm_fixtures.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step
from mlx_lm.models import llama, qwen2, qwen3

OUT = Path(__file__).resolve().parent.parent / "crates/nuvec-lm/tests/fixtures"
PROMPT = [3, 17, 92, 5, 64, 33, 101, 8, 77, 2, 45, 120, 9, 56]

BASE = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=128,
            max_position_embeddings=512)
CASES = {
    # Qwen 3: per-head q/k norms, separate output head, float32.
    "qwen3_f32": (qwen3, dict(BASE, model_type="qwen3", head_dim=16, rope_theta=1e6,
                              tie_word_embeddings=False), None, None),
    # Llama 3: llama3 RoPE scaling, tied embeddings, bfloat16.
    "llama_bf16": (llama, dict(BASE, model_type="llama", rope_theta=500000.0,
                               tie_word_embeddings=True,
                               rope_scaling={"rope_type": "llama3", "factor": 8.0,
                                             "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                             "original_max_position_embeddings": 64}),
                   mx.bfloat16, None),
    # Qwen 2: q/k/v biases, tied (quantized) embeddings, 4-bit weights.
    "qwen2_q4": (qwen2, dict(BASE, model_type="qwen2", rope_theta=1e6,
                             tie_word_embeddings=True), mx.float16, (32, 4)),
}


def main() -> None:
    mx.random.seed(20261018)
    for name, (family, args, dtype, quant) in CASES.items():
        model = family.Model(family.ModelArgs.from_dict(args))
        # Visible biases and norm scales (initialised to 0 and 1).
        model.update(tree_map_random(model.parameters()))
        if dtype is not None:
            model.set_dtype(dtype)
        config = dict(args)
        if quant is not None:
            group_size, bits = quant
            nn.quantize(model, group_size=group_size, bits=bits)
            config["quantization"] = {"group_size": group_size, "bits": bits}
        mx.eval(model.parameters())
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2))
        mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())))
        prompt = mx.array(PROMPT, dtype=mx.uint32)
        logits = model(prompt[None]).astype(mx.float32)
        tokens = [int(t) for t, _ in zip(
            (t for t, _ in generate_step(prompt, model, max_tokens=12)), range(12))]
        mx.save_safetensors(str(out / "expected.safetensors"), {
            "prompt": prompt, "logits": logits, "greedy": mx.array(tokens, dtype=mx.uint32)})
        print(f"{name}: {len(tree_flatten(model.parameters()))} tensors, greedy {tokens}")


def tree_map_random(params):
    """Every parameter redrawn with a visible scale (weights 0.2, norms around 1)."""
    from mlx.utils import tree_map

    def draw(p):
        noise = mx.random.normal(p.shape) * 0.2
        return (1.0 + noise) if p.ndim == 1 and mx.all(p == 1).item() else noise
    return tree_map(draw, params)


if __name__ == "__main__":
    main()
