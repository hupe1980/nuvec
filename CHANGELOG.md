# Changelog

All notable changes to nuvec are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the API).

## [Unreleased]

## [0.5.0] - 2026-10-31

### Added

- Floating-point quantization: MXFP4, MXFP8 and NVFP4 (MLX's `mode`) via `QuantMode` /
  `Quant`, equal to MLX's Metal quantizer, products and gathers. NVFP4 experts take a
  per-expert global scale (`GatherQmm::global_scale`).
- Neural Accelerator (NAX) kernels for M5-class GPUs on Metal 4 TensorOps (macOS 26.2;
  the baseline stays macOS 15): dense GEMM, affine 2/4/8-bit products, split-K and sorted
  MoE gathers. `NUVEC_NAX=auto|on|off`, `gpu::Device::nax()`; `auto` stays off until
  measured on an M5.
- Speculative decoding (nuvec-lm): `MtpDrafter` with the model's multi-token-prediction head
  (Qwen 3.5–3.8; from the checkpoint's `mtp.*` weights or `Model::attach_mtp`), `SuffixDecoding`, `DraftPolicy::Auto` and sampled
  verification (temperature, top-p, min-p, top-k). Greedy output is identical to plain
  decoding. New: `Model::{forward_hidden, logits, mtp_forward, new_mtp_cache}`, `verify_rows`,
  `nuvec::speculative::{accept_greedy, accept_sampled}`.
- `nuvec::distributed` (MLX's `mx.distributed`): `Group`, `all_sum`, `all_max`, `all_min`,
  `all_gather`, `send`, `recv`, `recv_like`, `sum_gradients`, lazy and differentiable. New
  `nuvec-distributed` crate with MLX's TCP ring (`MLX_HOSTFILE`, `MLX_RANK`). nuvec-nn:
  `sharded::{AllToShardedLinear, ShardedToAllLinear}`.
- Shapeless compile (MLX's `shapeless=True`): `compile::Options::shapeless(true)`;
  `compile::compile_flat_with` takes options.
- Sliding-window attention: `AttentionOptions::window(W)` (causal); `paged_attention` takes
  window, sinks and head size 512. Head-512 causal prefill runs MLX's blocked kernel.
- Ops: `view` (bit reinterpretation), `to_fp8` / `from_fp8`, `floor_divide`, `power`,
  `Squeeze` and `ExpandDims` primitives, `fast::norm_rope_kv`, `fast::gated_delta_steps_with`.
- Introspection: `Array::buffer_id`, `io::safetensors_footprint_filtered`,
  `DeviceInfo::physical_memory`, `gpu::Device::kernel_names`.
- nuvec-nn: quantized layers in any mode (`from_linear_with`, `from_embedding_with`,
  `quantized()`, `dequantized_weight`); `RopeScaling` (linear, Llama 3, YaRN, LongRoPE,
  proportional) with `Rope::scaled`; `swiglu_oai`, `geglu`, `logit_softcap`.
- nuvec-lm models: gpt-oss and Gemma 4 (text; vision and audio weights skipped).
- nuvec-lm checkpoints: MXFP4, MXFP8 and NVFP4; AutoAWQ and GPTQ (4-bit, 8-bit GPTQ),
  converted at load (`PackedFormat`, `quant::convert_packed`);
  `quant::mxfp4_from_hf_blocks` for gpt-oss's Hugging Face originals. Unsupported `quantization_config` formats are
  refused with a message. nuvec-io loads `F8_E4M3` and `F8_E8M0` tensors as `uint8`.
- nuvec-lm tokenizers: o200k (gpt-oss) and SentencePiece-style BPE with byte fallback
  (Gemma, Llama 2, Mistral), matching Hugging Face `tokenizers`.

### Changed

- **Breaking:** quantization takes a `Quant` and returns a `Quantized`
  (`w.quantize(Quant::affine(64, 4))`, `q.dequantize(dtype)`, `x.quantized_matmul(&q, t)`,
  `x.gather_qmm(&q, …)`) instead of `(group_size, bits)` and separate scales and biases.
  nuvec-nn's `QuantizedLinear` / `QuantizedEmbedding` hold `quant: Quant` and
  `biases: Option<Array>`; nuvec-lm's `Quantization` is `{ default, layers: Vec<LayerQuant> }`.
- **Breaking:** quantized and MoE products from 2 rows route as MLX's (`qmv_wide`,
  `qvm_split_k`), so a row's low bits can differ from its one-row product.
  `nuvec::with_row_exact` restores per-row results; `Batcher::batch_invariant(true)` serves
  each request its solo tokens. Batched Gemma 4 26B-A4B and gpt-oss-20b match mlx-lm
  (were 0.82–0.98×).
- **Breaking (nuvec-lm):** `Drafter::draft` takes the model and returns a `Draft` (`Draft::new`;
  new `observe`, `needs_hidden`, `auto`; implemented for `&mut D` and `Box<D>`);
  `speculative_generate` takes a `DraftPolicy` instead of a draft length;
  `SuffixDecoding::with_min_match`; `PromptLookup`'s host draft is `lookup`.
- **Breaking:** `paged_attention` takes `&AttentionOptions`; nuvec-lm's
  `config::RopeScaling` is nuvec-nn's; `gpu::DeviceInfo` is `#[non_exhaustive]`.
- Attention with head size 192 or 256 composes up to 8 GiB of scores (was 4 GiB), then
  uses the tiled kernel. Qwen3.6-35B-A3B's first token at 64K tokens: 115 s (was 127).
- Qwen3 and Qwen 3.5/3.6 MoE layers weight experts as mlx-lm does (precise router softmax,
  sum in the activation dtype), using less memory.
- Batched prefill splits K only for a single batch item, as MLX: Qwen3-0.6B serves 8
  requests at mlx-lm's speed (was 0.93×).
- `compile` emits constants as literals and uses Metal's default math, as `mx.compile`.
  Compiled `geglu` runs 2× faster.
- Numerics match MLX: float16/bfloat16 `softmax` computes in float32 on every path; GPU
  affine dequantize is one fused multiply-add; `gelu_approx` cubes with `power`.
- CPU updates (`slice_update`, `at(..).set`) write in place when nothing else holds the base.

### Fixed

- Tensors from unaligned (mlx-community) checkpoints are 256-byte aligned on the GPU;
  quantized products ran up to 4% slower depending on placement.
- Arrays of 2^32 elements or more: element-wise ops, row softmax, log-sum-exp and tiled
  matmuls run on the GPU with 64-bit indexing (they fell back to the CPU, e.g. 40 s instead of
  0.8 s).
- Batch-invariant `Batcher` gave Gemma 4 26B-A4B tokens that differed from solo decoding.
- Sliding-window decode depended on key placement in the cache, so gpt-oss's batched tokens
  drifted from solo decoding.
- Sorted MoE gathers gave wrong results for sliced or broadcast operands.
- A NAX load or kernel failure disabled every later NAX product; it now falls back to the
  standard kernels.
- Attention with no queries failed, and causal attention with no keys panicked.
- `compile` merged identical collectives and treated `stop_gradient` as an identity.
- `checkpoint_flat` drew new random numbers when recomputing.
- A custom batching rule returning the wrong outputs or batch axis gave wrong shapes; it is
  now an error.
- `distributed`: a failed transfer left the ring misaligned (later operations now fail with
  the first error); a mismatched `recv` drains the payload; invalid
  `NUVEC_DISTRIBUTED_TIMEOUT` values panicked.
- nuvec-lm: `attach_mtp` left a half-built head on failure; `load_adapters` refused
  `"num_layers": -1`; `linear_to_lora_layers` without keys failed on MoE models;
  SuffixDecoding eviction scanned the whole store; model-free drafters drafted twice per
  token; `resolve` ignored `HF_HUB_CACHE`; qwen2, gpt-oss and Qwen 3.5 configs without
  `rope_theta` used the wrong default.
- nuvec-nn: `MultiOptimizer` panicked on an optimizer returning too few parameters.
- GPU memory: the allocator could evict buffers still in use, fence waits could be skipped
  for reused addresses, and reuse of just-committed memory serialized command buffers
  (decode attention up to 1.2× MLX's time). Some in-place updates copied needlessly.
- Windowless attention kernels carried sliding-window code (about 8% slower).
- `softmax` on the CPU and the attention composition computed float64 in float32.
- Runtime-compiled kernels (`NUVEC_METALLIB=off`) used fast math instead of precise math.

## [0.4.1] - 2026-10-24

### Added

- Indexes for a rank known only at run time: a slice or `Vec` of ranges, one per leading axis, for `i`, `at` and `slice_update`.

## [0.4.0] - 2026-10-23

A new array API (operations are methods that return arrays; errors are sticky and located),
transforms over structured values and derived modules, and an engine surface for inference
servers: attention plans, memory accounting comparable with MLX, row-independent products.

### Added
- Indexing as MLX's `x[...]`: `x.i(..)` with integers, ranges, `..`, `Ellipsis`, `NewAxis`,
  `.step(n)`, index arrays and tuples (basic indices are views); functional updates
  `x.at(..).set(v)` / `.add(v)`.
- `array![...]`, `From` for Rust numbers, arrays, `Vec`s, slices and iterators,
  `Array::from_slice(&data, shape)`, `zeros_like`/`ones_like`/`full_like`, `var`/`std`,
  `split_n::<N>`, `m.select(a, b)`, `with_stream`, `Array::failure`; `Display` prints values
  as MLX; compiler hints for wrong axes, shapes and indices.
- Trees: `Tree`, `Param`, `Static` and `#[derive(Tree)]`, `#[derive(Static)]`,
  `#[derive(Module)]` (new crate `nuvec-macros`); `nuvec::save` and `nuvec::load_into` for
  any tree (a model and its optimizer in two lines).
- `compile_mut`: a compiled training step over `(model, optimizer)`; `compile::Options`.
- nuvec-nn: `num_params`, `summary()`, `display()`, freezing by path, `train`/`training`,
  `save_weights`/`save_trainable`, `map_modules`, `Dropout`, LoRA dropout and rsLoRA,
  `MultiOptimizer`, `AdamW`. nuvec-lm: LoRA adapters (`lora::linear_to_lora_layers`,
  `save_adapters`, `load_adapters`, mlx-lm's directory format).
- `fast::AttentionOptions` (mask, sinks, sliding `window`, `max_score_bytes` cap on score
  memory) and `fast::plan_attention` / `AttentionPlan` / `AttentionKernel` (which kernel a call
  runs and its scratch memory, from shapes).
- `MemoryStats::active_with_mapped` and `peak_with_mapped` (comparable with MLX's
  `get_peak_memory`); `io::safetensors_footprint` / `Footprint` (mapped, copied and transient
  bytes, from headers only); `io::save_safetensors_async` / `SaveHandle`; `ops::copy`.
- Row independence: each row of a product with at most `ROW_INDEPENDENT_ROWS` (8) rows has
  the bits of its one-row product; the crate docs state the determinism contract.
- `THIRD_PARTY_NOTICES` (MLX and mlx-lm, MIT) in the packages with ported code.
- docs.rs builds every crate for `aarch64-apple-darwin` without the Metal toolchain.
- Versioned benchmark records against mlx-lm (decode, time to first token up to 256K
  tokens, peak memory, load time, concurrency), published with the documentation.
- `nuvec::io::load_safetensors_filtered(path, keep)`: loads only the tensors `keep` accepts.

### Changed
- **Breaking:** every array-producing operation (methods, `ops::*`, operators, `fast::*`,
  `random::*`, nuvec-nn layers and losses, nuvec-lm `forward`) returns `Array`, not
  `Result`. Invalid input gives a failed array carrying `Error::Op { op, at, kind }` with
  the caller's location; errors surface at `check()`, `eval()`, `item()`, `to_vec()`, I/O
  and transforms. `ShapeMismatch` and similar variants are now `OpErrorKind`s.
- **Breaking:** operations are methods; the free functions are in `nuvec::ops` only.
  `nuvec::prelude` imports the everyday names.
- **Breaking:** shapes and axes are `i32` (`shape() -> &[i32]`, `shape_n::<N>()`); one
  `Axes` trait (`-1`, `[0, 2]`, `..`; an empty list means no axes, as MLX C++), `*_keepdims`
  twins and option structs instead of `bool` arguments.
- **Breaking:** graph internals (`Primitive`, `SliceArg`, `from_primitive`, …) moved to
  `nuvec::graph`.
- **Breaking:** transforms (`grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`, `checkpoint`,
  `custom_function`, `compile`) take and return trees; gradients have the model's type. The
  positional forms are `*_flat`. `eval`/`async_eval` take any tree.
- **Breaking:** nuvec-nn modules derive `Module`; parameters are `Param`; constructors are
  builders without `bool`s or `Result`; optimizers update with a gradient tree and keep
  their state as a tree; learning rates are 0-D arrays; `load_weights` is strict
  (`load_weights_partial` is MLX's `strict=False`).
- **Breaking:** `fast::scaled_dot_product_attention(q, k, v, scale, &AttentionOptions)`.
- **Breaking:** the memory limit bounds `active + mapped`; mapping or loading past it fails
  with `OutOfMemory`, before mapping. `MemoryStats` is `#[non_exhaustive]`.
- **Breaking:** products with 2–8 rows use row-grouped kernels in which every row has the
  bits of its one-row product, so they differ in the low bits from 0.3.0's tiled products.
- **Breaking:** `FinishReason` is `#[non_exhaustive]` (adds `CacheFull`, `Cancelled`,
  `Error`); `nuvec_cpu::matmul` returns `Result`; the raw `nuvec-gpu` dispatch functions are
  `unsafe`; nuvec-lm `Projection` is `#[non_exhaustive]`.
- `compile` keeps at most 64 traced signatures per function; `Sampler` settings are
  validated; Metal shaders compile in parallel, and a missing Metal toolchain falls back to
  runtime compilation.
- `gather_mm` with one row per output batch uses MLX's gathered matrix-vector kernels.
- Quantized split-K products count their partitions per batch item (`[B, L, K]`), so each
  item has the bits of its product alone (MLX counts all rows; bits differ from MLX's for
  `B > 1`).
- nuvec-lm `Batcher`: decoding is pipelined (the next step is submitted before the
  previous tokens are read; one host read per step); same-length prompts admitted together
  (quantized models, over 64 tokens) are prefilled in one batch, each with its solo tokens.
- Misaligned safetensors files (mlx-community checkpoints) are read in parallel instead of
  mapped and copied; `nuvec_lm::load_model` skips vision and MTP weights and no longer runs
  `Model::warm_up`; `safetensors_footprint`'s `transient` is `mapped + copied`.
- **Breaking:** `MappedFile::section` is replaced by `MappedFile::read_ranges`.

### Removed
- **Breaking:** `Array::scalar`, `Array::full`, `random::split_pair`, root
  `slice`/`slice_update`, `dim()`, `pub use nuvec_core as core`, nuvec-nn's hand-written
  `visit`, `Parameters`, `Frozen`, `update_trainable` and `nuvec_nn::value_and_grad`.
- **Breaking:** the empty `nuvec-distributed` crate and `nuvec::distributed`;
  `nuvec_kernels::kernel_names`.

### Fixed
- Two threads evaluating clones of one GPU graph no longer deadlock; a failed GPU command
  fails only the evaluations that depend on it; long-running threads no longer leak
  Objective-C objects.
- `compile` keeps compiling after the first optimizer step; `checkpoint` saves memory
  inside `compile`; `grad` through a compiled function that captured its input is correct
  on the CPU.
- Malformed safetensors, npy, IDX, JSON and `tokenizer.json` input returns errors instead
  of crashing; paged attention and cache writes stay inside the cache.
- CPU matrix products check sizes, and f16 products accumulate in f32.
- A panic in a `commit_then` completion handler is reported instead of aborting the process.
- Vector and paged attention kernels declare their 1,024-thread limit, so they also dispatch under Metal shader validation on M1.
- Long prefills no longer fail with a GPU out-of-memory error (Qwen3-30B-A3B at 32K tokens):
  the allocator counts mapped weights when it releases cached buffers.
- `Batcher`: one-token prompts, prompts that never fit (rejected at `submit`), requests
  that outgrow the cache, failing steps and cancellation.
- Generation stops at the end-of-sequence ids of `generation_config.json`; tokenizer
  encoding is no longer quadratic on long inputs.
- Threads that never call `random::seed` draw different numbers; optimizer state follows
  each parameter by name.
- Loading mlx-community checkpoints no longer holds the file twice: it is faster than
  mlx-lm's, and its peak memory equals MLX's (it was twice as high).

### Migrating from 0.3

| 0.3 | 0.4 |
|---|---|
| `add(&a, &b)?`, `exp(&x)?` | `a.add(&b)` (or `&a + &b`), `x.exp()`; free: `ops::add(&a, &b)` |
| `sum(&x, &[], false)?`, `sum(&x, &[-1], true)?` | `x.sum(..)`, `x.sum_keepdims(-1)` |
| `argmax(&x, Some(-1), false)?` | `x.argmax(-1)` |
| `cumsum(&x, Some(-1), true, false)?` | `x.cumsum_with(-1, ScanOptions { reverse: true, exclusive: true })` |
| `transpose(&h.reshape(&[b, l, n as isize, -1])?, &[0, 2, 1, 3])?` | `h.reshape([b, l, n, -1]).transpose([0, 2, 1, 3])` |
| `let [b, l] = ...shape()...` | `let [b, l] = x.shape_n::<2>()?;` |
| `slice(&m, &[SliceArg::full(), SliceArg::from(1..2)])?` | `m.i((.., 1..2))` |
| `slice_update(&a, &u, &[..])?` | `a.at((.., 1..2)).set(&u)` |
| `split(&qg, 2, -1)?` | `let [q, g] = qg.split_n::<2>(-1);` |
| `Array::from_slice(&[1.0f32, 2.0], &[2])?`, `Array::scalar(2.0f32)` | `array![1.0, 2.0]`, `Array::from(2.0f32)` |
| `zeros(&[2, 3], DType::F32)?`, `Array::full(&[2], 0.5f32)?` | `ops::zeros([2, 3], DType::F32)`, `ops::full([2], 0.5)` |
| `random::split_pair(&k)?` | `let [k1, k2] = random::split_n::<2>(&k);` |
| `Array::from_primitive(p, inputs)?` | `graph::from_primitive(p, inputs)?` |
| `let y = op(..)?;` before a host decision | `let y = op(..).check()?;` |
| `grad(\|x: &Array\| Ok(..))`, `value_and_grad(f, &[0, 1])(&[w, b])?` | `grad(\|x: &Array\| ..)`, `value_and_grad(\|(w, b): &(Array, Array)\| ..)(&(w, b))?` |
| `vmap(f, &[None, Some(0)], &[])` | `vmap(f).in_axes((None, 0)).call(&(m, v))?` |
| `impl Module for M { fn visit(..) }` | `#[derive(Clone, Module)] struct M { .. }` |
| `nn::value_and_grad(&model, loss)?` | `value_and_grad(\|m: &M\| loss(m))(&model)?` (grads: `M`) |
| `Linear::new(i, o, true)?`, `model.load_weights(w, false)?` | `Linear::new(i, o)`, `model.load_weights_partial(w)?` |
| `sdpa(q, k, v, s, AttentionMask::Causal, Some(&sinks))` | `sdpa(q, k, v, s, &AttentionOptions::causal().sinks(&sinks))` |

## [0.3.0] - 2026-10-21

Batched decoding and serving over a paged KV cache, seeded sampling, and long-context
prefill on hybrid MoE models.

### Added
- `nuvec_lm::Batcher`: serves many requests at once (admission, batched decoding, end of
  sequence and token budgets, preemption when the cache is full); 16 requests on Qwen3-8B
  finish 2.1× faster than one after another, with the same tokens.
- `nuvec_lm::PagedCache` and `decode_batch`: sequences of different lengths decode in one
  forward pass, hybrid models included, each with the tokens it gets alone; at 8 sequences
  Qwen3-8B reaches 149 tok/s and Qwen3.6-35B-A3B 223 tok/s in total.
- `fast::paged_attention`, `fast::paged_write` and `fast::gated_delta_slots`: attention,
  cache writes and the gated delta rule over block tables and state pools.
- `GenerateOptions::seed`: a seeded generation gives the same tokens every time, alone or
  batched (`Sampler::sample_with` and `sample::categorical_with` take an explicit key).

### Changed
- **Breaking:** `GenerateOptions` has a new field, `seed`.
- Element-wise operations, softmax and `slice_update` write into inputs the evaluation
  owns, and large intermediates are released as soon as nothing refers to them (as MLX):
  Qwen3.6-35B-A3B prefill at 32K tokens needs 5 GB of activations instead of 12 GB.
- Strided element-wise and copy kernels use 32-bit offsets when they fit: a broadcast
  `where` over attention scores is 1.5× faster.
- Attention with head size 256 uses the tiled kernel for causal prefill chunks over up to
  8,192 keys: Qwen3.6-35B-A3B prefill at 32K tokens 0.89× → 0.96–0.97× mlx-lm.

### Fixed
- Attention with head size 192 or 256 over very long caches ran out of memory; it now uses
  the tiled kernel above 4 GiB of scores. Qwen3.6-35B-A3B prefills a 252K-token prompt.

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

[Unreleased]: https://github.com/hupe1980/nuvec/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/hupe1980/nuvec/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/hupe1980/nuvec/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/hupe1980/nuvec/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/hupe1980/nuvec/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hupe1980/nuvec/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hupe1980/nuvec/releases/tag/v0.1.0
