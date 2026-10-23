# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2"]
# ///
"""Records tiny random models of each Llama-family variant with mlx-lm, and their outputs,
as test fixtures for nuvec-lm.

Each case is a directory under `crates/nuvec-lm/tests/fixtures/` with `config.json` and
`model.safetensors` (as a Hugging Face or MLX checkpoint) and `expected.safetensors`: the
logits of a fixed prompt (run at once) and 12 greedy tokens generated after it
(`generate_step`, with the KV cache). `samplers.safetensors` holds log-probabilities and
mlx-lm's top-p, min-p and top-k filters of them, `samplers_chained.safetensors` the three
chained by `make_sampler`, and `eos/` end-of-sequence ids with a `generation_config.json`.
Runs on MLX's GPU, as nuvec's tests do.

`adapters_qwen3/` and `adapters_qwen2_q4/` are LoRA adapter directories in mlx-lm's layout
(`adapters.safetensors`, `adapter_config.json`) for the `qwen3_f32` and `qwen2_q4` models (made
by mlx-lm's `linear_to_lora_layers`, with random `lora_b`), with the adapted models' logits in
evaluation mode (`expected.safetensors`).

Usage: `just lm-fixtures` (or `uv run scripts/lm_fixtures.py`; `--only chained_sampler,eos`
records just those).

`--cached` records, for each model of `CACHED` in the local Hugging Face cache, mlx-lm's 32
greedy tokens after a fixed prompt with the log-probability of each and the top 5 ids and
log-probabilities at every step (teacher-forced, one forward pass) in
`cached/<model>.safetensors`, for the ignored tests in `tests/cached.rs`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.generate import generate_step
from mlx_lm.models import llama, qwen2, qwen3, qwen3_5, qwen3_5_moe, qwen3_moe
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

OUT = Path(__file__).resolve().parent.parent / "crates/nuvec-lm/tests/fixtures"
PROMPT = [3, 17, 92, 5, 64, 33, 101, 8, 77, 2, 45, 120, 9, 56]

BASE = dict(hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=128,
            max_position_embeddings=512)
QWEN35 = dict(model_type="qwen3_5_text", hidden_size=64, num_hidden_layers=4,
              intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
              head_dim=32, rms_norm_eps=1e-6, vocab_size=128, max_position_embeddings=512,
              linear_num_key_heads=2, linear_num_value_heads=4, linear_value_head_dim=16,
              linear_conv_kernel_dim=4, full_attention_interval=4, attn_output_gate=True,
              rope_parameters=dict(rope_type="default", rope_theta=1e7,
                                   partial_rotary_factor=0.25, mrope_section=[2, 1, 1]))
QWEN35_MOE = dict(QWEN35, model_type="qwen3_5_moe_text", linear_key_head_dim=32,
                  num_experts=16, num_experts_per_tok=8, moe_intermediate_size=32,
                  shared_expert_intermediate_size=64)
# Qwen 3 MoE: layers 0 and 2 sparse (16 experts, 4 per token), layer 1 dense.
QWEN3_MOE = dict(BASE, model_type="qwen3_moe", num_hidden_layers=3, head_dim=16, rope_theta=1e6,
                 tie_word_embeddings=False, num_experts=16, num_experts_per_tok=4,
                 moe_intermediate_size=32, decoder_sparse_step=1, mlp_only_layers=[1])
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
    # Qwen 3.5 in float32 (tight tolerances).
    "qwen3_5_f32": (qwen3_5, dict(model_type="qwen3_5", tie_word_embeddings=False,
                                  text_config=dict(QWEN35, linear_key_head_dim=32)),
                    None, None),
    # Qwen 3.5: Gated DeltaNet and gated attention layers (3:1), partial RoPE, bfloat16,
    # 32-feature keys (the generic kernel).
    "qwen3_5_bf16": (qwen3_5, dict(model_type="qwen3_5", tie_word_embeddings=False,
                                   text_config=dict(QWEN35, linear_key_head_dim=32)),
                     mx.bfloat16, None),
    # Qwen 3.5 MoE: 16 experts, 8 per token (the prompt's 112 pairs take the sorted path),
    # a gated shared expert; float32.
    "qwen3_5_moe_f32": (qwen3_5_moe, dict(model_type="qwen3_5_moe", tie_word_embeddings=False,
                                          text_config=dict(QWEN35_MOE)), None, None),
    # Qwen 3.5 MoE with 4-bit experts and 8-bit routers (as mlx-community's checkpoints).
    "qwen3_5_moe_q4": (qwen3_5_moe, dict(model_type="qwen3_5_moe", tie_word_embeddings=False,
                                         text_config=dict(QWEN35_MOE)), mx.bfloat16, (32, 4)),
    # Qwen 3 MoE, renormalised top-k weights, float32.
    "qwen3_moe_f32": (qwen3_moe, dict(QWEN3_MOE, norm_topk_prob=True), None, None),
    # Qwen 3 MoE with the probabilities as weights, 4-bit experts and 8-bit routers.
    "qwen3_moe_q4": (qwen3_moe, dict(QWEN3_MOE, norm_topk_prob=False), mx.bfloat16, (32, 4)),
    # Qwen 3.5 with 128-feature keys (the packed kernel), 4-bit weights.
    "qwen3_5_q4": (qwen3_5, dict(model_type="qwen3_5", tie_word_embeddings=False,
                                 text_config=dict(QWEN35, linear_key_head_dim=128)),
                   mx.bfloat16, (32, 4)),
}


CACHED = ["mlx-community/Qwen3.8-27B-4bit", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"]
CACHED_PROMPT = "The history of the lighthouse begins with the fires that sailors lit on hills"


def cached() -> None:
    import os

    from mlx_lm import load

    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    for repo in CACHED:
        if not (hub / f"models--{repo.replace('/', '--')}").exists():
            print(f"skipped (not cached): {repo}")
            continue
        model, tokenizer = load(repo)
        prompt = tokenizer.encode(CACHED_PROMPT)
        greedy = []
        for token, _ in generate_step(mx.array(prompt), model, max_tokens=32):
            greedy.append(int(token))
        full = mx.array(prompt + greedy[:-1])[None]
        logits = model(full)[0, len(prompt) - 1:].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        top5 = mx.argsort(-logprobs, axis=-1)[:, :5]
        out = OUT / "cached"
        out.mkdir(parents=True, exist_ok=True)
        ids = mx.array(greedy)
        mx.save_safetensors(str(out / f"{repo.split('/')[1]}.safetensors"), {
            "prompt": mx.array(prompt, dtype=mx.uint32),
            "greedy": ids.astype(mx.uint32),
            "greedy_logprob": mx.take_along_axis(logprobs, ids[:, None], axis=-1)[:, 0],
            "top5_ids": top5.astype(mx.uint32),
            "top5_logprobs": mx.take_along_axis(logprobs, top5, axis=-1),
        })
        print(f"recorded {repo}")


def main() -> None:
    if "--cached" in sys.argv:
        cached()
        return
    # `--only chained_sampler,eos` records just those (the models are kept as they are).
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    if only is not None:
        for name in only:
            {"chained_sampler": chained_sampler, "eos": eos, "samplers": samplers,
             "adapters": adapters}[name]()
        return
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
            config["quantization"] = {"group_size": group_size, "bits": bits}

            def predicate(path, module):
                if not hasattr(module, "to_quantized"):
                    return False
                if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                    config["quantization"][path] = {"group_size": group_size, "bits": 8}
                    return {"group_size": group_size, "bits": 8}
                return True
            nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
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
        if name in ("qwen3_5_f32", "qwen3_5_moe_f32"):
            hugging_face_layout(out, config, dict(tree_flatten(model.parameters())))
        if name == "qwen3_moe_f32":
            hugging_face_experts(out, config, dict(tree_flatten(model.parameters())))
    samplers()
    chained_sampler()
    eos()
    adapters()


def adapters() -> None:
    """LoRA adapter directories in mlx-lm's layout for two fixture models: `qwen3_f32` with
    mlx-lm's default keys (every linear layer) on its last block, rank 4, dropout 0.1; and
    the 4-bit `qwen2_q4` with `self_attn.q_proj` and `self_attn.v_proj` on both blocks,
    rank 2. `lora_b` is drawn at random so the adapters change the logits; the logits are
    taken in evaluation mode, as `mlx_lm.load(..., adapter_path=...)` leaves the model."""
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load_model

    for base, name, num_layers, params in [
        ("qwen3_f32", "adapters_qwen3", 1, {"rank": 4, "scale": 20.0, "dropout": 0.1}),
        ("qwen2_q4", "adapters_qwen2_q4", 2,
         {"rank": 2, "scale": 10.0, "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}),
    ]:
        mx.random.seed(20261023)
        model, _ = load_model(OUT / base)
        model.freeze()
        linear_to_lora_layers(model, num_layers, params)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        model.update(tree_unflatten([
            (k, mx.random.normal(v.shape) * 0.05 if k.endswith("lora_b") else v)
            for k, v in trainable.items()]))
        model.eval()
        mx.eval(model.parameters())
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        adapters = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(out / "adapters.safetensors"), adapters)
        config = {"fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": params}
        (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
        prompt = mx.array(PROMPT, dtype=mx.uint32)
        logits = model(prompt[None]).astype(mx.float32)
        mx.save_safetensors(str(out / "expected.safetensors"), {"prompt": prompt, "logits": logits})
        print(f"{name}: {len(adapters)} adapter tensors: {sorted(adapters)[:4]}...")


def chained_sampler() -> None:
    """mlx-lm's `make_sampler` with top-p 0.9, min-p 0.05 and top-k 20 chained, on random
    log-probabilities [3, 128] (`samplers_chained.safetensors`): the filtered
    log-probabilities in `make_sampler`'s order (top-p, min-p, then top-k; read from the
    sampler itself by stubbing its final draw), and with top-k first, to show the order
    matters."""
    from mlx_lm import sample_utils

    mx.random.seed(20261022)
    logits = mx.random.normal((3, 128)) * 2.0
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    draw = sample_utils.categorical_sampling
    sample_utils.categorical_sampling = lambda lp, _temp: lp
    try:
        chained = sample_utils.make_sampler(temp=1.0, top_p=0.9, min_p=0.05, top_k=20)(logprobs)
    finally:
        sample_utils.categorical_sampling = draw
    top_k_first = apply_min_p(apply_top_p(apply_top_k(logprobs, 20), 0.9), 0.05)
    mx.save_safetensors(str(OUT / "samplers_chained.safetensors"), {
        "logprobs": logprobs, "chained": chained, "top_k_first": top_k_first})
    kept = [int(mx.sum(row > -mx.inf).item()) for row in chained]
    kept_first = [int(mx.sum(row > -mx.inf).item()) for row in top_k_first]
    print(f"chained sampler: tokens kept {kept} (top-k first: {kept_first})")


def eos() -> None:
    """End-of-sequence ids with a `generation_config.json` (`eos/`): a Llama 3 style
    config (`eos_token_id` 128001) and a generation config adding `<|eot_id|>` (128009).
    `expected.json` holds the ids mlx-lm's `load_config` reads; mlx-lm's tokenizer wrapper
    also adds the tokenizer's own `eos_token_id` (128001 for Llama 3), so its stop set is the
    union of both files, as nuvec's."""
    from mlx_lm.utils import load_config

    out = OUT / "eos"
    out.mkdir(parents=True, exist_ok=True)
    config = dict(BASE, model_type="llama", vocab_size=128256, eos_token_id=128001)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "generation_config.json").write_text(
        json.dumps({"bos_token_id": 128000, "eos_token_id": [128001, 128009]}, indent=2))
    ids = load_config(out)["eos_token_id"]
    ids = sorted(set(ids if isinstance(ids, list) else [ids]) | {config["eos_token_id"]})
    (out / "expected.json").write_text(json.dumps({"eos_token_ids": ids}) + "\n")
    print(f"eos: {ids}")


# mlx-lm 0.32.2's `qwen3_5.sanitize` drops `mtp.` weights and shifts the norms by one only
# when the convolutions are unsanitized (Hugging Face layout), whether or not `mtp.`
# weights are present (`qwen3_next`, which shifts on `mtp.` weights, is not a supported
# model type). nuvec's `sanitize` does the same: verified equal, no fixture needed.


def samplers() -> None:
    """Each filter of mlx-lm's sampler on random log-probabilities [3, 128]."""
    logits = mx.random.normal((3, 128)) * 2.0
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    filtered = {
        "top_p_0.9": apply_top_p(logprobs, 0.9),
        "top_p_0.3": apply_top_p(logprobs, 0.3),
        "min_p_0.1": apply_min_p(logprobs, 0.1),
        "min_p_0.5_keep_4": apply_min_p(logprobs, 0.5, 4),
        "top_k_1": apply_top_k(logprobs, 1),
        "top_k_7": apply_top_k(logprobs, 7),
    }
    mx.save_safetensors(str(OUT / "samplers.safetensors"), {"logprobs": logprobs, **filtered})
    kept = {k: int(mx.sum(v > -mx.inf).item()) for k, v in filtered.items()}
    print(f"samplers: tokens kept {kept}")


def hugging_face_layout(src: Path, config: dict, params: dict) -> None:
    """The same Qwen 3.5 model as a Hugging Face checkpoint (`qwen3_5_hf`): text weights
    under `model.language_model.`, convolutions as `[C, 1, K]`, zero-centred norms."""
    out = OUT / src.name.replace("_f32", "_hf")
    out.mkdir(parents=True, exist_ok=True)
    norms = (".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
             ".q_norm.weight", ".k_norm.weight")
    hf = {}
    for k, v in params.items():
        if "conv1d.weight" in k:
            v = v.moveaxis(2, 1)
        elif v.ndim == 1 and k.endswith(norms):
            v = v - 1.0
        k = k.replace("language_model.model.", "model.language_model.")
        hf[k.replace("language_model.lm_head", "lm_head")] = v
    # Experts as Hugging Face stores them: gate and up fused, under `experts.`.
    for k in [k for k in hf if k.endswith("switch_mlp.gate_proj.weight")]:
        prefix = k[: -len("switch_mlp.gate_proj.weight")]
        up = hf.pop(prefix + "switch_mlp.up_proj.weight")
        hf[prefix + "experts.gate_up_proj"] = mx.concatenate([hf.pop(k), up], axis=-2)
        hf[prefix + "experts.down_proj"] = hf.pop(prefix + "switch_mlp.down_proj.weight")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    mx.save_safetensors(str(out / "model.safetensors"), hf)
    (out / "expected.safetensors").write_bytes((src / "expected.safetensors").read_bytes())
    print(f"{out.name}: the same model in Hugging Face's layout")


def hugging_face_experts(src: Path, config: dict, params: dict) -> None:
    """The same Qwen 3 MoE model as a Hugging Face checkpoint (`qwen3_moe_hf`): each
    expert's projections stored separately (`experts.{e}.{name}.weight`)."""
    out = OUT / src.name.replace("_f32", "_hf")
    out.mkdir(parents=True, exist_ok=True)
    hf = {}
    for k, v in params.items():
        if ".switch_mlp." in k:
            prefix, name = k.split(".switch_mlp.")
            for e in range(v.shape[0]):
                hf[f"{prefix}.experts.{e}.{name}"] = v[e]
        else:
            hf[k] = v
    (out / "config.json").write_text(json.dumps(config, indent=2))
    mx.save_safetensors(str(out / "model.safetensors"), hf)
    (out / "expected.safetensors").write_bytes((src / "expected.safetensors").read_bytes())
    print(f"{out.name}: the same model with experts stored separately")


def tree_map_random(params):
    """Every parameter redrawn with a visible scale (weights 0.2, norms around 1)."""
    from mlx.utils import tree_map

    def draw(p):
        noise = mx.random.normal(p.shape) * 0.2
        return (1.0 + noise) if p.ndim == 1 and mx.all(p == 1).item() else noise
    return tree_map(draw, params)


if __name__ == "__main__":
    main()
