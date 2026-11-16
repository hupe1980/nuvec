# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx==0.32.4",
#   "mlx-lm==0.32.2",
# ]
# ///
"""Records tiny DFlash drafters (z-lab's `DFlashDraftModel`, vendored below, MIT) for nuvec-lm's fixture
targets, with the reference logits of the first draft after the fixture prompt.

Each case is `crates/nuvec-lm/tests/fixtures/dflash_<name>/` with `config.json` and
`model.safetensors` in z-lab's layout and `expected.safetensors`: `target_layers` (the
target's selected layer outputs at every prompt position, concatenated), `first` (the
target's greedy token after the prompt, the draft's anchor) and `logits` (`[block, vocab]`,
float32: the drafter's logits for `[first, mask, …]`).

Usage: `uv run scripts/dflash_fixtures.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm.utils import load_model

# ── Vendored from z-lab/dflash (dflash/model_mlx.py, commit 07ebd93, MIT License,
# Copyright (c) 2026 Z Lab): the config, RoPE, attention, decoder layers, DFlash 2's
# convolutions and candidate selector, and the draft models, unchanged. ──
from dataclasses import dataclass, field
from typing import Any

from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: dict[str, Any] | None = None
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    sliding_window: int | None = None
    final_logit_softcapping: float | None = None
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    is_causal: bool | None = None


def _build_rope(
    head_dim: int,
    rope_theta: float,
    max_position_embeddings: int,
    rope_scaling: dict[str, Any] | None,
):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.is_causal = self.is_sliding if config.is_causal is None else config.is_causal
        self.q_proj = nn.Linear(dim, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        if self.is_sliding:
            keep_ctx = self.sliding_window - 1
            if S > keep_ctx:
                skip = S - keep_ctx
                x_ctx = x_ctx[:, skip:]
                S = x_ctx.shape[1]
                cache.offset += skip
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        mask = create_causal_mask(L, offset=ctx_len) if self.is_causal else None
        if self.is_sliding:
            query = ctx_len + mx.arange(L)[:, None]
            key = mx.arange(ctx_len + L)[None]
            context = (key < ctx_len) & (query - key < self.sliding_window)
            block = key >= ctx_len
            if self.is_causal:
                block = block & (key <= query)
            mask = context | block
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))



class DFlashDraftModel(nn.Module):
    layer_class = DFlashDecoderLayer

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = ("full_attention",) * self.config.num_hidden_layers
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [self.layer_class(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(
            config.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            config.rope_scaling,
        )
        self.embed_tokens = None
        self.lm_head = None
        self.embed_scale = 1.0

    def bind(self, target_model):
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            inner = target_model.model
        elif (hasattr(target_model, "language_model") and
              hasattr(target_model.language_model, "model") and
              hasattr(target_model.language_model.model, "embed_tokens")):
            inner = target_model.language_model.model
        else:
            raise AttributeError(f"Cannot find embed_tokens in {type(target_model).__name__}")
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = (
            getattr(self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0))
            * self.config.input_embedding_scale
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = getattr(target_model, "lm_head", None) or getattr(lm, "lm_head", None) or self.embed_tokens.as_linear
        return self

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError("Draft config must define sliding_window for sliding_attention layers.")
                caches.append(RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0))
            else:
                caches.append(KVCache())
        return caches

    def hidden_states(
        self,
        inputs,
        target_hidden,
        cache,
        logits_start: int = 0,
    ):
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        if logits_start:
            h = h[:, logits_start:]
        return self.norm(h)

    def __call__(self, inputs, target_hidden, cache, logits_start: int = 0):
        return self.compute_logits(
            self.hidden_states(inputs, target_hidden, cache, logits_start)
        )

    def compute_logits(self, hidden):
        logits = self.lm_head(hidden)
        logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None and self.config.final_logit_softcapping > 0:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits




def _sampling_probs(logits, temperature, top_p=1.0, top_k=0):
    scores = logits.astype(mx.float32) / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        indices = mx.argpartition(-scores, top_k - 1, axis=-1)[..., :top_k]
        scores = mx.take_along_axis(scores, indices, axis=-1)
    else:
        indices = None

    probs = mx.softmax(scores, axis=-1)
    if top_p < 1.0:
        order = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, order, axis=-1)
        keep = mx.cumsum(sorted_probs, axis=-1) - sorted_probs < top_p
        sorted_probs = mx.where(keep, sorted_probs, 0)
        probs = mx.put_along_axis(mx.zeros_like(probs), order, sorted_probs, axis=-1)
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)

    if indices is not None:
        probs = mx.put_along_axis(
            mx.zeros(logits.shape, dtype=probs.dtype), indices, probs, axis=-1
        )
    return probs


def _sample_probs(probs):
    return mx.random.categorical(mx.log(probs))


def _grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else mx.concatenate(
            (mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]), axis=1
        )
        kernel = base[offset].reshape(1, 1, groups, group_size).astype(hidden.dtype)
        output = output + kernel * values
        output = output + dynamic[:, :, offset] * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False
        )

    def prepare(self, hidden):
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_dynamic_convolve(
                hidden, dynamic[..., 0, :, :], self.base_kernel[0], self.group_size
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden, dynamic):
        return _grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size
        )


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )

    def __call__(self, x, x_ctx, rope, cache):
        residual = x
        x, kernel = self.attention_conv.prepare(self.input_layernorm(x))
        x = residual + self.attention_conv.finish(
            self.self_attn(x, x_ctx, rope, cache), kernel
        )
        residual = x
        x, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return residual + self.mlp_conv.finish(self.mlp(x), kernel)


class CandidateSelector(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = nn.Linear(config.hidden_size, config.selector_rank, bias=False)

    def select(self, hidden, logits, anchor_ids, temperature):
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k:]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path, q_rows = [], []
        for position in range(hidden.shape[1]):
            edges = mx.sum(
                self.predecessor_codebook(predecessor)[:, None]
                * hidden[:, position, None]
                * self.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            scores = unary[:, position] + edges
            if temperature > 0:
                q = _sampling_probs(scores[:, None], temperature)[:, 0]
                selected = _sample_probs(q)
                q_rows.append(q)
            else:
                selected = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1
            )[:, 0]
            path.append(predecessor)
        return (
            mx.stack(path, axis=1),
            candidates,
            mx.stack(q_rows, axis=1) if q_rows else None,
        )


class DFlash2DraftModel(DFlashDraftModel):
    layer_class = DFlash2DecoderLayer

    def __init__(self, config: DFlashConfig):
        super().__init__(config)
        self.candidate_selector = CandidateSelector(config)

    def propose(self, inputs, target_hidden, cache, temperature, logits_start: int = 0):
        hidden = self.hidden_states(inputs, target_hidden, cache, logits_start)
        return self.candidate_selector.select(
            hidden, self.compute_logits(hidden), inputs[:, 0], temperature
        )

# ── End of the vendored code. ──

# ── DSpark (deepseek-ai/DeepSpec, deepspec/modeling/dspark: `VanillaMarkov`,
#    `AcceptRatePredictor`, `Qwen3DSparkModel.predict_confidence_step`, eval/dspark/draft_ops.py;
#    MIT License, Copyright (c) 2026 The DeepSpec Authors), ported from PyTorch to MLX. The
#    backbone is DFlash's; every slot of `[anchor, mask, …]` predicts the next token. ──


class DSparkDraftModel(DFlashDraftModel):
    def __init__(self, config: DFlashConfig, markov_rank: int):
        super().__init__(config)
        self.markov_w1 = nn.Embedding(config.vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, config.vocab_size, bias=False)
        self.confidence = nn.Linear(config.hidden_size + markov_rank, 1)

    def propose(self, inputs, target_hidden, cache):
        """`build_dspark_proposal` at temperature 0: the base logits of every slot, the
        Markov-corrected greedy draft and each drafted position's confidence logit."""
        hidden = self.hidden_states(inputs, target_hidden, cache)
        base = self.compute_logits(hidden)
        prev = inputs[:, 0]
        tokens, prevs = [], []
        for k in range(base.shape[1]):
            prevs.append(prev)
            step = base[:, k] + self.markov_w2(self.markov_w1(prev))
            prev = mx.argmax(step, axis=-1).astype(mx.uint32)
            tokens.append(prev)
        prevs = mx.stack(prevs, axis=1)
        features = mx.concatenate([hidden, self.markov_w1(prevs).astype(hidden.dtype)], axis=-1)
        confidence = self.confidence(features)[..., 0]
        return base, mx.stack(tokens, axis=1), confidence

# ── End of the DSpark port. ──

FIXTURES = Path(__file__).resolve().parent.parent / "crates/nuvec-lm/tests/fixtures"
PROMPT = [3, 17, 92, 5, 64, 33, 101, 8, 77, 2, 45, 120, 9, 56]

FULL2 = ["full_attention"] * 2
# (name, target, drafter settings, seed, quantized)
CASES = [
    ("qwen3", "qwen3_f32", dict(layer_ids=[0, 1], layer_types=FULL2, window=None, block=8),
     20261120, False),
    ("qwen3_q4", "qwen3_f32", dict(layer_ids=[0, 1], layer_types=FULL2, window=None, block=8),
     20261120, True),
    ("qwen3_5", "qwen3_5_f32",
     dict(layer_ids=[1, 3], layer_types=["sliding_attention", "full_attention"], window=8,
          block=8), 20261121, False),
    # DFlash 2: convolutions around attention and MLP, the candidate selector.
    ("qwen3_v2", "qwen3_f32",
     dict(layer_ids=[0, 1], layer_types=["sliding_attention"] * 2, window=8, block=8,
          v2=dict(conv_kernel_size=2, conv_group_size=16, selector_rank=8, selector_top_k=4)),
     20261122, False),
    # Gemma 4: the target's scaled embedding, the drafter's logit soft cap.
    ("gemma4", "gemma4_e_f32",
     dict(layer_ids=[1, 4], layer_types=["sliding_attention", "full_attention"], window=8,
          block=8, softcap=1.5), 20261123, False),
    # DSpark (DeepSpec's `Qwen3DSparkModel` layout): Markov head and confidence head.
    ("qwen3_dspark", "qwen3_f32",
     dict(layer_ids=[0, 1], layer_types=FULL2, window=None, block=7, dspark=8),
     20261125, False),
    # gpt-oss: YaRN RoPE and attention biases.
    ("gpt_oss", "gpt_oss_f32",
     dict(layer_ids=[0, 1], layer_types=FULL2, window=None, block=8, bias=True,
          rope={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8,
                "beta_fast": 32.0, "beta_slow": 1.0}), 20261124, False),
]


def hf_config(target_cfg: dict, s: dict, layers: int) -> dict:
    text = target_cfg.get("text_config", target_cfg)
    dflash = {"mask_token_id": text["vocab_size"] - 1, "target_layer_ids": s["layer_ids"]}
    dflash.update(s.get("v2", {}))
    cfg = {
        "architectures": ["DFlash2DraftModel" if "v2" in s else "DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 96,
        "vocab_size": text["vocab_size"],
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 4096,
        "layer_types": s["layer_types"],
        "sliding_window": s["window"],
        "block_size": s["block"],
        "num_target_layers": layers,
        "attention_bias": s.get("bias", False),
        "dflash_config": dflash,
    }
    if "softcap" in s:
        cfg["final_logit_softcapping"] = s["softcap"]
    if "rope" in s:
        cfg["rope_scaling"] = s["rope"]
    if "dspark" in s:
        # DeepSpec's flat layout (no `dflash_config`).
        del cfg["dflash_config"]
        cfg.update({
            "architectures": ["Qwen3DSparkModel"], "target_layer_ids": s["layer_ids"],
            "mask_token_id": text["vocab_size"] - 1, "markov_rank": s["dspark"],
            "markov_head_type": "vanilla", "enable_confidence_head": True,
            "confidence_head_with_markov": True, "num_anchors": 512,
        })
    return cfg


def config_of(cfg: dict) -> DFlashConfig:
    d = cfg.get("dflash_config", cfg)
    return DFlashConfig(
        hidden_size=cfg["hidden_size"], num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"], rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"], block_size=cfg["block_size"],
        target_layer_ids=tuple(d["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"], mask_token_id=d["mask_token_id"],
        rope_scaling=cfg.get("rope_scaling"),
        layer_types=tuple(cfg["layer_types"]), sliding_window=cfg["sliding_window"],
        final_logit_softcapping=cfg.get("final_logit_softcapping"),
        conv_kernel_size=d.get("conv_kernel_size", 0),
        conv_group_size=d.get("conv_group_size", 0),
        selector_rank=d.get("selector_rank", 0), selector_top_k=d.get("selector_top_k", 0))


def add_attention_biases(draft) -> None:
    """nuvec's change for gpt-oss drafters (`attention_bias`, as z-lab's PyTorch reference):
    the projections gain biases (the MLX reference has none)."""
    for layer in draft.layers:
        a = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            old = getattr(a, name)
            out_dim, in_dim = old.weight.shape
            setattr(a, name, nn.Linear(in_dim, out_dim, bias=True))


class _Hooked:
    """A decoder layer whose calls go through `hook` (attributes pass through)."""

    def __init__(self, layer, hook):
        self._layer, self._hook = layer, hook

    def __call__(self, *args, **kwargs):
        return self._hook(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._layer, name)


def main() -> None:
    for name, target_name, s, seed, quantized in CASES:
        target_dir = FIXTURES / target_name
        target, _ = load_model(target_dir)
        target_cfg = json.loads((target_dir / "config.json").read_text())
        inner = getattr(target, "language_model", target).model
        layers = len(inner.layers)
        cfg = hf_config(target_cfg, s, layers)
        mx.random.seed(seed)
        v2, dspark = "v2" in s, "dspark" in s
        if dspark:
            draft = DSparkDraftModel(config_of(cfg), s["dspark"])
        else:
            draft = (DFlash2DraftModel if v2 else DFlashDraftModel)(config_of(cfg))
        if s.get("bias"):
            add_attention_biases(draft)
        draft.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.2, draft.parameters()))
        for module in [draft.hidden_norm, draft.norm] + [
                n for layer in draft.layers
                for n in (layer.input_layernorm, layer.post_attention_layernorm,
                          layer.self_attn.q_norm, layer.self_attn.k_norm)]:
            module.weight = 1.0 + module.weight * 0.1
        if quantized:
            nn.quantize(draft, group_size=32, bits=4)
            cfg["quantization"] = {"group_size": 32, "bits": 4}
        mx.eval(draft.parameters())
        draft.bind(target)
        # The target's selected layer outputs at every prompt position (as z-lab's hook:
        # each decoder layer's output).
        captured = {}
        for i in s["layer_ids"]:
            layer = inner.layers[i]

            def hook(*args, _layer=layer, _i=i, **kwargs):
                out = type(_layer).__call__(_layer, *args, **kwargs)
                captured[_i] = out[0] if isinstance(out, tuple) else out
                return out
            inner.layers[i] = _Hooked(layer, hook)
        prompt = mx.array(PROMPT, dtype=mx.uint32)[None]
        logits = target(prompt)
        hidden = mx.concatenate([captured[i] for i in s["layer_ids"]], axis=-1)
        first = mx.argmax(logits[0, -1]).astype(mx.uint32)
        mask = cfg.get("dflash_config", cfg)["mask_token_id"]
        block = mx.array([[int(first)] + [mask] * (s["block"] - 1)], dtype=mx.uint32)
        extra = {}
        if dspark:
            out, tokens, confidence = draft.propose(block, hidden, draft.make_cache())
            out = out.astype(mx.float32)[0]
            tokens = tokens[0].astype(mx.uint32)
            extra["confidence"] = confidence[0].astype(mx.float32)
        elif v2:
            # Every row's logits (the norm is per row, so rows 1… equal `logits_start=1`).
            h = draft.hidden_states(block, hidden, draft.make_cache())
            out = draft.compute_logits(h).astype(mx.float32)[0]
            tokens, _, _ = draft.candidate_selector.select(
                h[:, 1:], draft.compute_logits(h[:, 1:]), block[:, 0], 0.0)
            tokens = tokens[0].astype(mx.uint32)
        else:
            out = draft(block, hidden, draft.make_cache()).astype(mx.float32)[0]
            tokens = mx.argmax(out[1:], axis=-1).astype(mx.uint32)
        dest = FIXTURES / f"dflash_{name}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(json.dumps(cfg, indent=2))
        weights = dict(tree_flatten(draft.parameters()))
        weights = {k: v for k, v in weights.items()
                   if not k.startswith(("embed_tokens", "lm_head"))}
        if dspark:
            # DeepSpec's names, and its copies of the target's embeddings and head.
            for ours, theirs in [("markov_w1.weight", "markov_head.markov_w1.weight"),
                                 ("markov_w2.weight", "markov_head.markov_w2.weight"),
                                 ("confidence.weight", "confidence_head.proj.weight"),
                                 ("confidence.bias", "confidence_head.proj.bias")]:
                weights[theirs] = weights.pop(ours)
            weights["embed_tokens.weight"] = inner.embed_tokens.weight
            weights["lm_head.weight"] = draft.lm_head.weight if hasattr(draft.lm_head, "weight") \
                else inner.embed_tokens.weight
        if v2:
            # z-lab's checkpoint names (the reference's loader maps them to `.weight`).
            for c in ("predecessor_codebook", "successor_codebook"):
                key = f"candidate_selector.{c}"
                weights[key] = weights.pop(f"{key}.weight")
        mx.save_safetensors(str(dest / "model.safetensors"), weights)
        mx.save_safetensors(str(dest / "expected.safetensors"), {
            "target_layers": hidden.astype(mx.float32), "first": first[None],
            "logits": out, "draft": tokens, **extra})
        print(f"dflash_{name}: {len(weights)} tensors, first {int(first)}, "
              f"draft {tokens.tolist()}")


if __name__ == "__main__":
    main()
