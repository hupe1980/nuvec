# Changelog

All notable changes to nuvec are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the API).

## [Unreleased]

### Added
- `fast::paged_attention` and `fast::paged_write`: attention over a block-table KV cache
  for a batch of sequences of different lengths, and the write of new positions into it.
- `nuvec_lm::PagedCache` and `decode_batch`: several sequences of different lengths decode
  in one forward pass over a paged KV cache, hybrid (Gated DeltaNet) models included, with
  the same tokens each gets alone; at 8 sequences Qwen3-8B reaches 149 tok/s and
  Qwen3.6-35B-A3B 223 tok/s in total.
- `fast::gated_delta_slots`: the gated delta rule for a batch whose states live at slots of a
  state pool, updated in place.

### Changed
- Element-wise operations and softmax write into an input buffer the evaluation owns, and
  large intermediates are released as soon as nothing refers to them (as MLX): attention
  without a fused kernel holds one score matrix instead of three, and Qwen3.6-35B-A3B
  prefill at 32K tokens needs 5 GB of activations instead of 12 GB.
- Element-wise and copy kernels over strided layouts use 32-bit offsets when they fit
  (Apple GPUs emulate 64-bit integer products): a broadcast `where` over attention scores
  is 1.5× faster.
- Attention with head size 256 uses the tiled kernel for causal prefill chunks over up to
  8,192 keys (measured faster), the composition beyond: Qwen3.6-35B-A3B prefill at 32K
  tokens 0.89× → 0.96–0.97× mlx-lm.

### Fixed
- Attention with head size 192 or 256 over very long caches ran out of memory (it
  materialised scores of many gigabytes); it now uses the tiled kernel above 4 GiB of
  scores. Qwen3.6-35B-A3B prefills a 252K-token prompt.
- `slice_update` on the CPU reads a strided update in place instead of copying it first.
- `slice_update` (and `norm_rope` into a cache) writes in place into a base computed
  earlier in the same evaluation instead of copying it, so a chain of updates evaluated at
  once holds one buffer.

## [0.2.0] - 2026-10-20

Faster decoding through fused kernels, speculative decoding on hybrid models, and MoE
prefill at mlx-lm's speed.

### Added
- `fast::add_rms_norm`: a residual addition and its RMS normalisation in one kernel.
- `fast::norm_rope` and `fast::Rotation`: per-head RMS norm, RoPE and an in-place KV cache
  write in one kernel.
- `fast::gated_delta_steps` and `KvCache::keep_steps`: speculative decoding on hybrid
  (Gated DeltaNet) models, which roll back rejected drafts.
- `MemoryStats::mapped` and `MemoryStats::wired_mapped`: memory-mapped checkpoints, which
  `active` and `peak` do not count.
- `random::next_key`, to pass the thread's key sequence into compiled functions.
- GPU kernels for depthwise `conv1d`, random bits and 4-bit products of a few rows.
- Run-to-run determinism tests on the GPU.

### Changed
- **Breaking:** `MemoryStats` has two new fields; `FastOp::GatedDelta` has a `steps` field.
- `nuvec-lm` fuses residual additions with the next norm and prepares queries and keys
  with `norm_rope`: Qwen3-0.6B decode 1.06–1.09× mlx-lm (was 1.00–1.04×), same tokens.
- Sorted `gather_qmm` groups rows only with enough rows per expert, as MLX:
  Qwen3.6-35B-A3B prefill 0.97–1.14× mlx-lm (was 0.92–0.96×).
- Speculative decoding on Qwen3-8B: 2.16× on repetitive text (was 1.98×), 1.02× on
  natural text (was 0.96×).
- Sampling with a temperature costs about 1% over greedy decoding (was 7–8% on small
  models): random bits run on the GPU and the categorical draw is compiled.

### Fixed
- Kernels compiled from embedded sources (no Metal toolchain at build time) gave slightly
  different results than the precompiled library; they now match, and the first launch
  takes about half as long.

## [0.1.0] - 2026-10-19

The first release: MLX's programming model in pure Rust, with an LLM runtime at mlx-lm's speed.

### Arrays and transforms
- Lazy arrays on unified memory with 14 dtypes, MLX type promotion, NumPy broadcasting,
  zero-copy views, indexing and joining, reductions, sorting, scans and matmul.
- Composable transforms: `grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`, `custom_function`,
  `checkpoint`, also through complex numbers; `compile` with element-wise kernel fusion.
- Random numbers bit-identical to MLX's for a key (uniform, normal, Gumbel, categorical,
  integers, Bernoulli).

### GPU
- Metal backend through `objc2-metal`: asynchronous evaluation (`async_eval`, `schedule`),
  hazard-tracked command encoding, MLX's allocator rules, memory reuse while the GPU runs,
  in-place updates of buffers nothing else can see, wired memory (residency sets) and a
  per-kernel GPU profiler (`NUVEC_PROFILE`).
- MLX's kernels: GEMM (split-K, fused `addmm`), GEMV, gather GEMM, affine quantization
  (1–8 bits) with the mixture-of-experts gather kernels, attention (vector, two-pass,
  tiled), RMS and layer norm, RoPE, softmax, cross entropy, sort, scan; mlx-lm's Gated
  DeltaNet kernels; custom kernels from source (`fast::MetalKernel`).

### Neural networks and I/O
- `nuvec-nn`: modules as named parameter trees, Linear, quantized layers, embeddings,
  norms, RoPE, LoRA, activations, losses, SGD, Adam, AdamW, Lion and Muon with schedules
  and gradient clipping.
- `nuvec-io`: safetensors and npy, memory-mapped without a copy; IDX.

### LLM runtime
- `nuvec-lm`: Llama, Mistral, Qwen 2, Qwen 3, Qwen 3 MoE and Qwen 3.5 (hybrid Gated
  DeltaNet, dense or mixture-of-experts) from Hugging Face or mlx-community checkpoints, a
  byte-level BPE tokenizer, mlx-lm's samplers, a pipelined decode loop and speculative
  decoding with prompt-lookup drafts.
- Decode at 0.97–1.04× and prefill at a geometric mean of 1.04× of mlx-lm 0.32.2 across
  seven models on an Apple M4 Pro (`just bench-parity`).

[Unreleased]: https://github.com/hupe1980/nuvec/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hupe1980/nuvec/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hupe1980/nuvec/releases/tag/v0.1.0
