# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "mlx-lm==0.32.2", "tokenizers==0.22.1"]
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

`--cached [repo…]` records, for each model of `CACHED` in the local Hugging Face cache (or
the repos named; one at a time, skipping any whose weights exceed half the memory), mlx-lm's 32
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
from mlx_lm.models import (gemma4, gemma4_text, gpt_oss, llama, qwen2, qwen3, qwen3_5,
                           qwen3_5_moe, qwen3_moe)
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


CACHED = ["mlx-community/Qwen3.8-27B-4bit", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
          "mlx-community/gpt-oss-20b-MXFP4-Q8", "lmstudio-community/gpt-oss-20b-MLX-8bit",
          "mlx-community/gemma-4-e4b-it-4bit", "mlx-community/gemma-4-26b-a4b-it-4bit"]
CACHED_PROMPT = "The history of the lighthouse begins with the fires that sailors lit on hills"


def cached() -> None:
    import os

    from mlx_lm import load

    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    only = sys.argv[sys.argv.index("--cached") + 1:]
    for repo in CACHED:
        if only and repo not in only:
            continue
        cache = hub / f"models--{repo.replace('/', '--')}"
        if not cache.exists():
            print(f"skipped (not cached): {repo}")
            continue
        # One model at a time, and only one that fits: its weights at most half the
        # machine's memory (the rest for activations, MLX's cache and the system).
        weights = sum(f.stat().st_size for f in cache.glob("snapshots/*/*.safetensors"))
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if weights > ram // 2:
            print(f"skipped (needs {weights / 2**30:.1f} GiB, over half of "
                  f"{ram / 2**30:.0f} GiB): {repo}")
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
        del model, tokenizer, logits, logprobs
        mx.clear_cache()


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
             "adapters": adapters, "fp": fp_models, "packed": packed_models,
             "ops": ops_fixtures, "gpt_oss": gpt_oss_models, "gemma4": gemma4_models,
             "o200k": o200k_splits, "spm": spm_tokenizers, "mtp": mtp_heads}[name]()
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
    fp_models()
    packed_models()
    ops_fixtures()
    gpt_oss_models()
    mtp_heads()


# Floating-point quantized models, as `mlx_lm.convert --q-mode` writes them (its own
# `quantize_model`): dense, mixture-of-experts with 8-bit affine routers (a per-layer
# config), nvfp4 experts with a global scale each, a hybrid Qwen 3.5 MoE and Llama.
FP_CASES = {
    "qwen3_mxfp4": (qwen3, dict(BASE, model_type="qwen3", head_dim=16, rope_theta=1e6,
                                tie_word_embeddings=False), "mxfp4", False, False),
    "qwen3_nvfp4": (qwen3, dict(BASE, model_type="qwen3", head_dim=16, rope_theta=1e6,
                                tie_word_embeddings=False), "nvfp4", False, False),
    "qwen3_mxfp8": (qwen3, dict(BASE, model_type="qwen3", head_dim=16, rope_theta=1e6,
                                tie_word_embeddings=False), "mxfp8", False, False),
    "qwen3_moe_mxfp4": (qwen3_moe, dict(QWEN3_MOE, norm_topk_prob=True), "mxfp4", True, False),
    "qwen3_moe_nvfp4_gs": (qwen3_moe, dict(QWEN3_MOE, norm_topk_prob=True), "nvfp4", True, True),
    "qwen3_5_moe_nvfp4": (qwen3_5_moe, dict(model_type="qwen3_5_moe", tie_word_embeddings=False,
                                            text_config=dict(QWEN35_MOE)), "nvfp4", True, False),
    "llama_mxfp8": (llama, dict(BASE, model_type="llama", rope_theta=500000.0,
                                tie_word_embeddings=True), "mxfp8", False, False),
}


def fp_models() -> None:
    from mlx_lm.utils import quantize_model

    mx.random.seed(20261024)
    for name, (family, args, mode, routers, global_scale) in FP_CASES.items():
        model = family.Model(family.ModelArgs.from_dict(args))
        model.update(tree_map_random(model.parameters()))
        model.set_dtype(mx.bfloat16)

        def predicate(path, module):
            if routers and (path.endswith("mlp.gate") or path.endswith("shared_expert_gate")):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        model, config = quantize_model(model, dict(args), None, None, mode=mode, global_scale=global_scale,
                                       quant_predicate=predicate)
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


# Hugging Face integer packings (AutoAWQ GEMM, GPTQ v1 and v2) of float16 models: random
# codes, zeros and scales per group, written once in MLX's affine layout (the reference
# mlx-lm runs: `bias = -zero * scale`) and once in the packing, so the packed checkpoint
# must give MLX's logits exactly. mlx-lm loads the AWQ files itself too (its
# `_transform_awq_weights`); its GPTQ path reads GPTQ as AWQ's layout, so GPTQ's reference
# is the affine checkpoint only.
PACKED_CASES = {
    "qwen3_awq": (qwen3, dict(BASE, model_type="qwen3", head_dim=16, rope_theta=1e6,
                              tie_word_embeddings=False), "awq", 4, 64),
    "llama_gptq": (llama, dict(BASE, model_type="llama", rope_theta=500000.0,
                               tie_word_embeddings=False), "gptq", 4, 32),
    "qwen3_gptq8_v2": (qwen3, dict(BASE, model_type="qwen3", head_dim=16, rope_theta=1e6,
                                   tie_word_embeddings=False), "gptq_v2", 8, 32),
}
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def packed_models() -> None:
    import numpy as np

    rng = np.random.default_rng(20261025)

    def mlx_pack(q, bits):  # q [out, in] -> uint32 [out, in * bits / 32]
        pw = 32 // bits
        q = q.astype(np.uint32).reshape(q.shape[0], -1, pw)
        return (q << (np.arange(pw, dtype=np.uint32) * bits)).sum(-1, dtype=np.uint32)

    def awq_pack(m):  # m [rows, cols] -> int32 [rows, cols / 8], AWQ's order
        m = m.astype(np.uint32).reshape(m.shape[0], -1, 8)
        shifts = np.array(AWQ_ORDER, dtype=np.uint32) * 4
        return (m << shifts).sum(-1, dtype=np.uint32).view(np.int32)

    def seq_pack(m, bits, axis):  # sequential packing along `axis` -> int32
        pw = 32 // bits
        m = np.moveaxis(m.astype(np.uint32), axis, -1)
        m = m.reshape(*m.shape[:-1], -1, pw)
        w = (m << (np.arange(pw, dtype=np.uint32) * bits)).sum(-1, dtype=np.uint32)
        return np.moveaxis(w, -1, axis).view(np.int32)

    for name, (family, args, method, bits, gs) in PACKED_CASES.items():
        model = family.Model(family.ModelArgs.from_dict(args))
        model.update(tree_map_random(model.parameters()))
        model.set_dtype(mx.float16)
        # The decoder's linear layers, as AWQ and GPTQ quantize them.
        nn.quantize(model, group_size=gs, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and ".layers." in f".{p}")
        flat = dict(tree_flatten(model.parameters()))
        hf = {k: v for k, v in flat.items() if not k.endswith((".scales", ".biases"))}
        for prefix in sorted({k[:-len(".scales")] for k in flat if k.endswith(".scales")}):
            out_f, packed_in = flat[prefix + ".weight"].shape
            in_f, groups = packed_in * 32 // bits, packed_in * 32 // bits // gs
            q = rng.integers(0, 1 << bits, (out_f, in_f))
            zero = rng.integers(1 if method == "gptq" else 0, 1 << bits, (out_f, groups))
            scale = (rng.random((out_f, groups)) * 0.02 + 0.005).astype(np.float16)
            bias = mx.array((-zero.astype(np.float32) * scale.astype(np.float32))).astype(mx.float16)
            flat[prefix + ".weight"] = mx.array(mlx_pack(q, bits))
            flat[prefix + ".scales"] = mx.array(scale)
            flat[prefix + ".biases"] = bias
            del hf[prefix + ".weight"]
            hf[prefix + ".scales"] = mx.array(np.ascontiguousarray(scale.T))  # [groups, out]
            if method == "awq":
                hf[prefix + ".qweight"] = mx.array(awq_pack(q.T))
                hf[prefix + ".qzeros"] = mx.array(awq_pack(zero.T))
            else:
                stored = zero - 1 if method == "gptq" else zero
                hf[prefix + ".qweight"] = mx.array(seq_pack(q.T, bits, 0))
                hf[prefix + ".qzeros"] = mx.array(seq_pack(stored.T, bits, 1))
                hf[prefix + ".g_idx"] = mx.array((np.arange(in_f) // gs).astype(np.int32))
        model.update(tree_unflatten(list(flat.items())))
        mx.eval(model.parameters())
        config = dict(args)
        qc = {"bits": bits, "group_size": gs}
        if method == "awq":
            qc |= {"quant_method": "awq", "version": "gemm", "zero_point": True}
        else:
            qc |= {"quant_method": "gptq", "sym": False, "desc_act": False,
                   "checkpoint_format": "gptq_v2" if method == "gptq_v2" else "gptq"}
        config["quantization_config"] = qc
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2))
        mx.save_safetensors(str(out / "model.safetensors"), hf)
        prompt = mx.array(PROMPT, dtype=mx.uint32)
        logits = model(prompt[None]).astype(mx.float32)
        if method == "awq":
            from mlx_lm.utils import load_model
            loaded, _ = load_model(out)
            again = loaded(prompt[None]).astype(mx.float32)
            assert mx.array_equal(logits, again).item(), f"{name}: mlx-lm's AWQ path differs"
        tokens = [int(t) for t, _ in zip(
            (t for t, _ in generate_step(prompt, model, max_tokens=12)), range(12))]
        mx.save_safetensors(str(out / "expected.safetensors"), {
            "prompt": prompt, "logits": logits, "greedy": mx.array(tokens, dtype=mx.uint32)})
        print(f"{name}: {len(hf)} tensors, greedy {tokens}")


# mlx-lm's RoPE scalings, activations and gpt-oss's YaRN attention (on MLX's GPU, as
# mlx-lm runs them), for nuvec-nn's `RopeScaling`, `Rope::scaled`, the mscale² fold and
# the compiled activations: crates/nuvec-nn/tests/fixtures/mlx_lm_ops.safetensors.
def ops_fixtures() -> None:
    import numpy as np
    from mlx_lm.models import activations, gemma4_text, rope_utils

    g = np.random.default_rng(20261026)

    def rand(*shape, scale=1.0, dtype=mx.float32):
        return mx.array((g.standard_normal(shape) * scale).astype(np.float32)).astype(dtype)

    out = {}
    long_factor = [1.0 + 0.05 * i for i in range(48)]
    ropes = {
        "yarn": (64, rope_utils.YarnRoPE(64, base=150000.0, scaling_factor=32.0,
                                         original_max_position_embeddings=4096, beta_fast=32,
                                         beta_slow=1)),
        "llama3": (128, rope_utils.Llama3RoPE(128, base=500000.0, scaling_config={
            "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192})),
        "longrope": (96, rope_utils.SuScaledRoPE(96, base=10000.0, max_position_embeddings=131072,
                                                original_max_position_embeddings=4096,
                                                long_factor=long_factor)),
        "proportional": (512, rope_utils.ProportionalRoPE(512, rotated_dims=128, base=1e6)),
    }
    for name, (dims, rope) in ropes.items():
        x = rand(1, 4, 5, dims)
        mscale = getattr(rope, "mscale", getattr(rope, "_scale", 1.0))
        out[f"{name}.freqs"] = rope._freqs
        out[f"{name}.mscale"] = mx.array(float(mscale))
        out[f"{name}.x"] = x
        out[f"{name}.out"] = rope(x, offset=3)
    out["longrope.factor"] = mx.array(long_factor)
    # gpt-oss: YaRN on q and k (mlx-lm multiplies both by mscale), then attention with sinks.
    yarn = ropes["yarn"][1]
    q, k, v = rand(1, 8, 6, 64), rand(1, 2, 9, 64), rand(1, 2, 9, 64)
    sinks = rand(8)
    o = mx.fast.scaled_dot_product_attention(yarn(q, offset=3), yarn(k), v, scale=0.125,
                                             mask="causal", sinks=sinks)
    out.update({"yarn_attn.q": q, "yarn_attn.k": k, "yarn_attn.v": v, "yarn_attn.sinks": sinks,
                "yarn_attn.out": o})
    for dt, label in [(mx.float32, "f32"), (mx.bfloat16, "bf16")]:
        gate, x = rand(4, 33, scale=5.0, dtype=dt), rand(4, 33, scale=5.0, dtype=dt)
        out[f"act_{label}.gate"], out[f"act_{label}.x"] = gate, x
        out[f"act_{label}.swiglu_oai"] = activations.swiglu_oai(gate, x, alpha=1.702, limit=7.0)
        out[f"act_{label}.geglu"] = gemma4_text.geglu(gate, x)
        out[f"act_{label}.softcap"] = gemma4_text.logit_softcap(30.0, gate * 10)
    mx.eval(out)
    path = OUT.parent.parent.parent / "nuvec-nn/tests/fixtures/mlx_lm_ops.safetensors"
    mx.save_safetensors(str(path), out)
    print(f"ops: {len(out)} tensors -> {path}")


# gpt-oss (mlx-lm's `gpt_oss`): sliding (W 4, shorter than the prompt) and full layers,
# YaRN, random sinks, biased attention, router and experts (clamped SwiGLU), in float32, in
# 4-bit affine and MXFP4 (8-bit routers, mlx-lm's `quant_predicate`), and two Hugging Face
# layouts: the MXFP4 originals (`*_blocks`/`*_scales`, `quant_method: mxfp4`, which mlx-lm's
# own loader must give the same logits) and transformers' full-precision experts
# (`gate_up_proj` `[E, D, 2I]` interleaved on the last axis, `down_proj` `[E, I, D]`).
GPT_OSS = dict(model_type="gpt_oss", hidden_size=64, num_hidden_layers=2, intermediate_size=64,
               num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
               rms_norm_eps=1e-5, num_local_experts=8, num_experts_per_tok=2, sliding_window=4,
               rope_theta=150000.0, layer_types=["sliding_attention", "full_attention"],
               rope_scaling={"rope_type": "yarn", "factor": 4.0,
                             "original_max_position_embeddings": 8, "beta_fast": 32.0,
                             "beta_slow": 1.0}, tie_word_embeddings=False)


def record(model, out: Path, config: dict, weights: dict) -> mx.array:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    prompt = mx.array(PROMPT, dtype=mx.uint32)
    logits = model(prompt[None]).astype(mx.float32)
    tokens = [int(t) for t, _ in zip((t for t, _ in generate_step(prompt, model, max_tokens=12)),
                                     range(12))]
    mx.save_safetensors(str(out / "expected.safetensors"), {
        "prompt": prompt, "logits": logits, "greedy": mx.array(tokens, dtype=mx.uint32)})
    print(f"{out.name}: {len(weights)} tensors, greedy {tokens}")
    return logits


def gpt_oss_models() -> None:
    """Every variant from one seed, the first from 20261027 whose half-precision routers have
    no near tie on the prompt (top-k margins above 2^-7): a top-k choice decided by an ulp
    tests rounding, not routing."""
    margins = []
    block = gpt_oss.MLPBlock.__call__

    def watched(self, x):
        if x.dtype == mx.float32 or x.shape[1] != len(PROMPT):
            return block(self, x)
        g = mx.sort(self.router(x).astype(mx.float32), axis=-1)[..., ::-1]
        k = self.num_experts_per_tok
        gap = (g[..., k - 1] - g[..., k]) / mx.maximum(mx.abs(g[..., k - 1]), 1.0)
        margins.append(mx.min(gap).item())
        return block(self, x)

    gpt_oss.MLPBlock.__call__ = watched
    try:
        for seed in range(20261027, 20261127):
            margins.clear()
            gpt_oss_variants(seed)
            if min(margins) > 2**-7:
                return
            print(f"gpt-oss seed {seed}: router margin {min(margins):.2e}, next seed")
        raise RuntimeError("no gpt-oss seed without router near ties")
    finally:
        gpt_oss.MLPBlock.__call__ = block


def gpt_oss_variants(seed: int) -> None:
    from mlx_lm.utils import load_model, quantize_model

    def build(dtype):
        mx.random.seed(seed)
        model = gpt_oss.Model(gpt_oss.ModelArgs.from_dict(GPT_OSS))
        model.update(tree_map_random(model.parameters()))
        model.set_dtype(dtype)
        mx.eval(model.parameters())
        return model

    model = build(mx.float32)
    record(model, OUT / "gpt_oss_f32", dict(GPT_OSS), dict(tree_flatten(model.parameters())))
    # Transformers' layout of the same weights.
    hf = {}
    for k, v in tree_flatten(model.parameters()):
        if ".experts.gate_proj." in k or ".experts.up_proj." in k:
            continue
        if ".experts.down_proj.weight" in k:
            hf[k.replace("down_proj.weight", "down_proj")] = v.transpose(0, 2, 1)
        elif ".experts.down_proj.bias" in k:
            hf[k.replace("down_proj.bias", "down_proj_bias")] = v
        else:
            hf[k] = v
    for i in range(GPT_OSS["num_hidden_layers"]):
        e = model.model.layers[i].mlp.experts
        p = f"model.layers.{i}.mlp.experts."
        g, u = e.gate_proj.weight.transpose(0, 2, 1), e.up_proj.weight.transpose(0, 2, 1)
        hf[p + "gate_up_proj"] = mx.stack([g, u], axis=-1).reshape(*g.shape[:-1], -1)
        hf[p + "gate_up_proj_bias"] = mx.stack([e.gate_proj.bias, e.up_proj.bias], axis=-1).reshape(
            e.gate_proj.bias.shape[0], -1)
    record(model, OUT / "gpt_oss_hf", dict(GPT_OSS), hf)

    model = build(mx.bfloat16)
    record(model, OUT / "gpt_oss_bf16", dict(GPT_OSS), dict(tree_flatten(model.parameters())))

    model = build(mx.bfloat16)
    q4, cfg = quantize_model(model, dict(GPT_OSS), 64, 4, quant_predicate=model.quant_predicate)
    record(q4, OUT / "gpt_oss_q4", cfg, dict(tree_flatten(q4.parameters())))

    model = build(mx.bfloat16)
    fp4, cfg = quantize_model(model, dict(GPT_OSS), None, None, mode="mxfp4",
                              quant_predicate=model.quant_predicate)
    record(fp4, OUT / "gpt_oss_mxfp4", cfg, dict(tree_flatten(fp4.parameters())))

    # The originals: MXFP4 experts only (attention and router in bfloat16), as blocks.
    model = build(mx.bfloat16)
    experts, _ = quantize_model(model, dict(GPT_OSS), None, None, mode="mxfp4",
                                quant_predicate=lambda p, m: ".experts." in f".{p}.")
    hf = {}
    for k, v in tree_flatten(experts.parameters()):
        if ".experts." not in k:
            hf[k] = v
    for i in range(GPT_OSS["num_hidden_layers"]):
        e = experts.model.layers[i].mlp.experts
        p = f"model.layers.{i}.mlp.experts."

        def blocks(w):  # uint32 [E, out, in / 8] -> uint8 [E, out, in / 32, 16]
            return w.view(mx.uint8).reshape(*w.shape[:-1], -1, 16)

        def interleave(a, b):
            return mx.stack([a, b], axis=2).reshape(a.shape[0], -1, *a.shape[2:])

        hf[p + "gate_up_proj_blocks"] = blocks(interleave(e.gate_proj.weight, e.up_proj.weight))
        hf[p + "gate_up_proj_scales"] = interleave(e.gate_proj.scales, e.up_proj.scales)
        hf[p + "gate_up_proj_bias"] = interleave(e.gate_proj.bias, e.up_proj.bias)
        hf[p + "down_proj_blocks"] = blocks(e.down_proj.weight)
        hf[p + "down_proj_scales"] = e.down_proj.scales
        hf[p + "down_proj_bias"] = e.down_proj.bias
    cfg = dict(GPT_OSS, quantization_config={"quant_method": "mxfp4"})
    out = OUT / "gpt_oss_hf_mxfp4"
    logits = record(experts, out, cfg, hf)
    loaded, _ = load_model(out)
    again = loaded(mx.array(PROMPT, dtype=mx.uint32)[None]).astype(mx.float32)
    assert mx.array_equal(logits, again).item(), "mlx-lm's gpt-oss sanitize differs"


# Gemma 4 (mlx-lm's `gemma4_text`, and its `gemma4` wrapper): sliding layers (W 4, head 16,
# default RoPE) and full ones (head 32, proportional RoPE over a quarter of it), 2:1, with
# per-layer inputs (8 wide), two KV-shared last layers with double-wide MLPs and the logit
# soft cap (`gemma4_e_bf16`, and `gemma4_e_f32`); K = V full layers with one key head (`gemma4_kv_f32`); the
# mixture-of-experts branch (8 experts, top 2) beside the MLP, 4-bit with 8-bit routers
# (`gemma4_moe_q4`), and the same experts in Hugging Face's layout (fused
# `experts.gate_up_proj`, `model.language_model.` names, a vision tower to drop).
GEMMA4 = dict(model_type="gemma4_text", hidden_size=64, num_hidden_layers=6, intermediate_size=64,
              num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
              rms_norm_eps=1e-6, vocab_size=128, vocab_size_per_layer_input=128,
              hidden_size_per_layer_input=8, num_kv_shared_layers=2, sliding_window=4,
              layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
              final_logit_softcapping=30.0, use_double_wide_mlp=True, tie_word_embeddings=True)
GEMMA4_KV = dict(GEMMA4, num_hidden_layers=4, attention_k_eq_v=True, num_global_key_value_heads=1,
                 hidden_size_per_layer_input=0, num_kv_shared_layers=0,
                 layer_types=["sliding_attention", "full_attention"] * 2)
GEMMA4_MOE = dict(GEMMA4, num_hidden_layers=3, hidden_size_per_layer_input=0,
                  num_kv_shared_layers=0, enable_moe_block=True, num_experts=8, top_k_experts=2,
                  moe_intermediate_size=32, layer_types=["sliding_attention", "sliding_attention",
                                                         "full_attention"])
# Full layers at head size 512 with 8 query heads per key head, as Gemma 4 26B-A4B's
# (`gemma4_d512_bf16`): MLX decodes them with its composition below 1,024 keys.
GEMMA4_D512 = dict(GEMMA4, num_hidden_layers=3, num_attention_heads=16, global_head_dim=512,
                   hidden_size_per_layer_input=0, num_kv_shared_layers=0,
                   layer_types=["sliding_attention", "sliding_attention", "full_attention"])


def gemma4_models() -> None:
    """The mixture-of-experts models from the first seed from 20261024 whose half-precision
    routers have no near tie on the prompt (top-k margins above 2^-8, about two ulps at
    these scores), as for gpt-oss."""
    from mlx_lm.utils import load_model, quantize_model

    margins = []
    route = gemma4_text.Router.__call__

    def watched(self, x):
        idx, w = route(self, x)
        if x.dtype != mx.float32 and x.shape[1] == len(PROMPT):
            xn = mx.fast.rms_norm(x, self.scale * self._root_size, self.eps)
            g = mx.sort(self.proj(xn).astype(mx.float32), axis=-1)[..., ::-1]
            k = self.config.top_k_experts
            gap = (g[..., k - 1] - g[..., k]) / mx.maximum(mx.abs(g[..., k - 1]), 1.0)
            margins.append(mx.min(gap).item())
        return idx, w

    def build(args, dtype, seed, wrapper=False):
        mx.random.seed(seed)
        if wrapper:
            cfg = dict(model_type="gemma4", vocab_size=args["vocab_size"],
                       text_config={k: v for k, v in args.items() if k != "model_type"})
            model = gemma4.Model(gemma4.ModelArgs.from_dict(cfg))
        else:
            cfg = dict(args)
            model = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(cfg))
        model.update(tree_map_random(model.parameters()))
        # Routers spread four times wider: expert gaps well above half precision's rounding.
        for layer in model.layers:
            if getattr(layer, "enable_moe", False):
                layer.router.proj.weight = layer.router.proj.weight * 4
        model.set_dtype(dtype)
        mx.eval(model.parameters())
        return model, cfg

    model, cfg = build(GEMMA4, mx.bfloat16, 20261024, wrapper=True)
    record(model, OUT / "gemma4_e_bf16", cfg, dict(tree_flatten(model.parameters())))
    model, cfg = build(GEMMA4, mx.float32, 20261024, wrapper=True)
    record(model, OUT / "gemma4_e_f32", cfg, dict(tree_flatten(model.parameters())))
    model, cfg = build(GEMMA4_KV, mx.float32, 20261024)
    record(model, OUT / "gemma4_kv_f32", cfg, dict(tree_flatten(model.parameters())))
    model, cfg = build(GEMMA4_D512, mx.bfloat16, 20261024)
    record(model, OUT / "gemma4_d512_bf16", cfg, dict(tree_flatten(model.parameters())))

    gemma4_text.Router.__call__ = watched
    try:
        for seed in range(20261024, 20261124):
            margins.clear()
            model, cfg = build(GEMMA4_MOE, mx.bfloat16, seed)
            q4, qcfg = quantize_model(model, cfg, 32, 4, quant_predicate=model.quant_predicate)
            record(q4, OUT / "gemma4_moe_q4", qcfg, dict(tree_flatten(q4.parameters())))
            # Hugging Face's layout of the full-precision model.
            model, cfg = build(GEMMA4_MOE, mx.bfloat16, seed)
            hf = {"model.vision_tower.patch.weight": mx.zeros((4, 4), dtype=mx.bfloat16)}
            for k, v in tree_flatten(model.parameters()):
                if ".experts.switch_glu." in k:
                    continue
                hf[k.replace("model.", "model.language_model.", 1)] = v
            for i, layer in enumerate(model.model.layers):
                g = layer.experts.switch_glu
                p = f"model.language_model.layers.{i}.experts."
                hf[p + "gate_up_proj"] = mx.concatenate([g.gate_proj.weight, g.up_proj.weight], -2)
                hf[p + "down_proj"] = g.down_proj.weight
            hcfg = dict(model_type="gemma4", vocab_size=cfg["vocab_size"],
                        text_config={k: v for k, v in cfg.items() if k != "model_type"})
            out = OUT / "gemma4_moe_hf"
            logits = record(model, out, hcfg, hf)
            loaded, _ = load_model(out)
            again = loaded(mx.array(PROMPT, dtype=mx.uint32)[None]).astype(mx.float32)
            assert mx.array_equal(logits, again).item(), "mlx-lm's gemma4 sanitize differs"
            if min(margins) > 2**-8:
                return
            print(f"gemma4 seed {seed}: router margin {min(margins):.2e}, next seed")
        raise RuntimeError("no gemma4 seed without router near ties")
    finally:
        gemma4_text.Router.__call__ = route


# o200k's split pattern (GPT-4o, gpt-oss), split by Hugging Face `tokenizers`
# (`Split(..., "isolated")`) on texts that exercise its letter classes, contractions,
# marks, digit runs, slashes and newlines: `o200k_splits.json` for nuvec's scanner.
O200K = ("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
         "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
         "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}"
         "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
O200K_TEXTS = [
    "Hello  world's 123456!? don't I'LL HTTPRequest iPhone McDonald's",
    "ÜBER über Ärger naïve café ﬁne ǅemo ʰello",
    "a\u0301b \u0301x 日本語のテキスト 한국어 mixed中文Text",
    "path/to/file ,,, ...\n\n  \n\tx  y\r\n\r\nz   ",
    "x'S y'Ve 'll 1 22 333 4444 ٣٤٥ Ⅷ ½",
    "!!!/\n// hi/there\n  \n",
    "CamelCaseWord ALLCAPS's lower's MiXeD ABCdef",
]


def o200k_splits() -> None:
    from tokenizers import Regex, pre_tokenizers

    split = pre_tokenizers.Split(Regex(O200K), behavior="isolated")
    cases = [{"text": t, "pieces": [p for p, _ in split.pre_tokenize_str(t)]} for t in O200K_TEXTS]
    path = OUT / "o200k_splits.json"
    path.write_text(json.dumps({"pattern": O200K, "cases": cases}, ensure_ascii=False, indent=1))
    print(f"o200k: {len(cases)} texts -> {path}")


# SentencePiece-style tokenizers (byte-fallback BPE) from the local Hugging Face cache
# (their tokenizer files alone suffice): Hugging Face `tokenizers`' ids (without the
# post-processor's tokens) and decodes, for `tests/cached.rs`.
SPM_TOKENIZERS = ["mlx-community/gemma-4-e4b-it-4bit", "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                  "mlx-community/Mistral-7B-Instruct-v0.2-4bit"]
SPM_TEXTS = [
    "The history of the lighthouse begins with the fires that sailors lit on hills",
    " leading space,  two spaces,\tand a tab\nnew line\n\n",
    "Ünïcödé, 日本語, 🦀 crabs, 𝔘𝔫𝔦 and ǅ: byte fallback",
    "<start_of_turn>user\nHi!<end_of_turn>\n<start_of_turn>model\n",
    "[INST] code: fn main() { println!(\"{}\", 1 + 2); } [/INST]",
    "   ",
]


def spm_tokenizers() -> None:
    import os

    from tokenizers import Tokenizer

    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    out = {}
    for repo in SPM_TOKENIZERS:
        files = sorted(hub.glob(f"models--{repo.replace('/', '--')}/snapshots/*/tokenizer.json"))
        if not files:
            print(f"skipped (not cached): {repo}")
            continue
        t = Tokenizer.from_file(str(files[0]))
        cases = []
        for text in SPM_TEXTS:
            ids = t.encode(text, add_special_tokens=False).ids
            cases.append({"text": text, "ids": ids, "decoded": t.decode(ids, skip_special_tokens=False)})
        out[repo] = cases
    path = OUT / "cached" / "spm_tokenizers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"spm: {len(out)} tokenizers -> {path}")


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


def mtp_heads() -> None:
    """Qwen 3.5 multi-token-prediction heads (mlx-vlm's drafter structure) on
    `qwen3_5_f32` and `qwen3_5_moe_f32`, built from mlx-lm's own classes with random weights:
    `h = fc([pre_fc_norm_embedding(embed(t + 1)), pre_fc_norm_hidden(hidden(t))])`, one
    full-attention decoder layer (causal over the prompt), `norm`, then the model's head.
    Written in the original checkpoints' layout (`mtp.*`, norms stored minus one, MoE
    experts as fused `gate_up_proj`/`down_proj`) to `mtp.safetensors`, with the logits of
    every prompt position's next-but-one token in `mtp_expected.safetensors`."""
    mx.random.seed(20261025)
    for name, family in (("qwen3_5_f32", qwen3_5), ("qwen3_5_moe_f32", qwen3_5_moe)):
        out = OUT / name
        config = json.loads((out / "config.json").read_text())
        model = family.Model(family.ModelArgs.from_dict(config))
        model.load_weights(list(mx.load(str(out / "model.safetensors")).items()))
        text = model.language_model
        args = text.args
        d, eps = args.hidden_size, args.rms_norm_eps
        layer = qwen3_5.DecoderLayer(args, args.full_attention_interval - 1)
        assert not layer.is_linear
        head = {"fc": nn.Linear(2 * d, d, bias=False), "pre_fc_norm_embedding": nn.RMSNorm(d, eps=eps),
                "pre_fc_norm_hidden": nn.RMSNorm(d, eps=eps), "norm": nn.RMSNorm(d, eps=eps)}
        layer.update(tree_map_random(layer.parameters()))
        for m in head.values():
            m.update(tree_map_random(m.parameters()))
        prompt = mx.array(PROMPT, dtype=mx.uint32)[None]
        hidden = text.model(prompt)  # after the final norm
        emb = text.model.embed_tokens(prompt[:, 1:])
        h = head["fc"](mx.concatenate([head["pre_fc_norm_embedding"](emb),
                                       head["pre_fc_norm_hidden"](hidden[:, :-1])], axis=-1))
        h = head["norm"](layer(h, mask="causal", cache=None))
        logits = (text.model.embed_tokens.as_linear(h) if args.tie_word_embeddings
                  else text.lm_head(h)).astype(mx.float32)
        norms = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight",
                 ".k_norm.weight")
        weights = {}
        for k, v in tree_flatten(layer.parameters()):
            key = f"mtp.layers.0.{k}"
            weights[key] = v - 1 if v.ndim == 1 and key.endswith(norms) else v
        moe = [k for k in weights if ".switch_mlp." in k]
        if moe:
            prefix = moe[0].split("switch_mlp.")[0]
            gate = weights.pop(f"{prefix}switch_mlp.gate_proj.weight")
            up = weights.pop(f"{prefix}switch_mlp.up_proj.weight")
            weights[f"{prefix}experts.gate_up_proj"] = mx.concatenate([gate, up], axis=-2)
            weights[f"{prefix}experts.down_proj"] = weights.pop(f"{prefix}switch_mlp.down_proj.weight")
        weights["mtp.fc.weight"] = head["fc"].weight
        for n in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            weights[f"mtp.{n}.weight"] = head[n].weight - 1
        mx.save_safetensors(str(out / "mtp.safetensors"), weights)
        mx.save_safetensors(str(out / "mtp_expected.safetensors"), {"logits": logits})
        print(f"{name}: MTP head, {len(weights)} tensors, logits {logits.shape}")


def tree_map_random(params):
    """Every parameter redrawn with a visible scale (weights 0.2, norms around 1)."""
    from mlx.utils import tree_map

    def draw(p):
        noise = mx.random.normal(p.shape) * 0.2
        return (1.0 + noise) if p.ndim == 1 and mx.all(p == 1).item() else noise
    return tree_map(draw, params)


if __name__ == "__main__":
    main()
