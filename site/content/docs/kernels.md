+++
title = "Fused and custom Metal kernels"
description = "nuvec's fast module: fused RMSNorm, LayerNorm, RoPE, attention, cross entropy and Gated DeltaNet kernels, plus custom Metal kernels written from source."
weight = 60
[extra]
group = "GPU"
+++

## Fused kernels (`nuvec::fast`)

The `fast` module exposes MLX's fused operations and a few of nuvec's own. Each one is a single Metal kernel on the GPU and an equivalent composition of ordinary operations elsewhere; the composition also defines its derivatives and batching, so they work under every transform.

| Function | What it computes |
|---|---|
| `rms_norm(x, weight, eps)` | RMS normalization over the last axis |
| `add_rms_norm(x, residual, weight, eps)` | `x + residual` and its RMS normalization (a transformer block's residual connection) |
| `layer_norm(x, weight, bias, eps)` | Layer normalization |
| `rope(x, dims, traditional, base, scale, offset, freqs)` | Rotary position embeddings, optionally on the first `dims` features only |
| `norm_rope(x, norm, rotation, offset, cache)` | Per-head RMS normalization and RoPE, optionally written straight into a KV cache |
| `scaled_dot_product_attention(q, k, v, scale, mask, sinks)` | Attention with grouped-query heads, causal, boolean or additive masks and attention sinks |
| `cross_entropy(logits, targets)` | Fused log-softmax and negative log-likelihood |
| `gated_delta(q, k, v, g, beta, state)` | The Gated DeltaNet recurrence (Qwen 3.5's linear attention), returning the outputs and the new state |
| `gated_delta_steps(q, k, v, g, beta, state)` | The same, returning the state after every step (to roll the recurrence back) |
| `paged_attention(q, k_cache, v_cache, block_tables, lengths, scale)` | Attention for a batch of sequences of different lengths over a KV cache stored in fixed-size blocks |
| `paged_write(cache, new, slots)` | Writes new keys or values into their blocks, in place when the cache is owned |

Attention picks MLX's kernel for the shape: vector kernels for short queries (decode), a two-pass split-K kernel for long contexts, and the tiled flash-style kernel for prefill. `softmax` and `logsumexp` over the last axis use fused kernels too.

## Custom kernels

`MetalKernel` runs your own kernel body, written in Metal Shading Language, the way MLX's `mx.fast.metal_kernel` does. You give a name, the input and output names and the body; nuvec generates the function signature:

```rust
use nuvec::fast::{KernelConfig, MetalKernel, TemplateArg};

let scale = MetalKernel::new(
    "scale",
    &["inp"],
    &["out"],
    "uint i = thread_position_in_grid.x;\nout[i] = static_cast<T>(inp[i] * N);",
);
let config = KernelConfig {
    grid: [n, 1, 1],
    threadgroup: [256, 1, 1],
    template: vec![("T".into(), TemplateArg::DType(DType::F16)), ("N".into(), TemplateArg::Int(3))],
    ..Default::default()
};
let y = scale.call(&[x], &[(&[n], DType::F16)], &config)?.remove(0);
```

Signature rules, as in MLX:

- an input is `const device T* name`; a 0-d input is `const constant T& name`;
- `name_shape`, `name_strides` and `name_ndim` are added when the body uses them;
- an output is `device T* name`, or `device atomic<T>* name` with atomic outputs (optionally initialized);
- thread attributes the body mentions (`thread_position_in_grid`, `thread_index_in_simdgroup`, …) become parameters;
- template arguments become `constexpr` values and type aliases; MLX's `bfloat16_t`, `float16_t` and `elem_to_loc` are defined.

Kernels compile once per generated source and run with one thread per grid position. They have no derivative or batching rule of their own: wrap them in `custom_function` to give them one. mlx-lm's own Gated DeltaNet kernel source runs unchanged through `MetalKernel`.
