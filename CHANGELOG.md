# Changelog

All notable changes to nuvec are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the API).

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

[0.1.0]: https://github.com/hupe1980/nuvec/releases/tag/v0.1.0
