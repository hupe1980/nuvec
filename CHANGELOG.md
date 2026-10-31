# Changelog

All notable changes to nuvec are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the API).

## [Unreleased]

## [0.5.0] - 2026-10-25

### Added

- Floating-point quantization modes MXFP4, MXFP8 and NVFP4 (MLX's `mode`), selected with
  `QuantMode` / `Quant`: codes and scales equal MLX's Metal quantizer's on both devices,
  products and gathers equal MLX's on the GPU, and up to `ROW_INDEPENDENT_ROWS` rows keep
  the bits of their one-row products. NVFP4 experts take a per-expert global scale
  (`GatherQmm::global_scale`).
- Neural Accelerator (NAX) kernels for M5-class GPUs: dense GEMM and affine 2/4/8-bit
  quantized products on Metal 4 TensorOps, in a second kernel library for macOS 26.2 (the
  baseline keeps macOS 15). `NUVEC_NAX=auto|on|off`, `nuvec::gpu::Device::nax()`; `auto`
  stays off until an M5 measurement (`just nax-report`), `on` forces them on any GPU with
  TensorOps. Thin products (16- and 32-row or column tiles), split-K products and sorted
  mixture-of-experts gathers have NAX kernels too.
- Speculative decoding v2 (nuvec-lm): `MtpDrafter` with the model's multi-token-prediction
  head (Qwen 3.5–3.8; `Model::attach_mtp` loads it from `scripts/extract_mtp.py`'s file and
  quantizes it like the body; checkpoints that keep `mtp.*` load it directly),
  `SuffixDecoding`, `DraftPolicy::Auto` (measured draft length per drafter and model, with a
  kill switch) and sampled verification (temperature, top-p, min-p, top-k). Greedy output
  stays bit-identical to plain decoding. `Model::{forward_hidden, logits, mtp_forward,
  new_mtp_cache}`, `verify_rows`.
- `nuvec::speculative::{accept_greedy, accept_sampled}`: acceptance on the device, one
  `[3]` read per verification; `fast::gated_delta_steps_with` (states rounded to a storage
  dtype per step).
- `nuvec::distributed` (MLX's `mx.distributed`): `Group`, `all_sum`, `all_max`, `all_min`,
  `all_gather`, `send`, `recv`, `recv_like` and `sum_gradients` as lazy operations with MLX's
  gradient rules, and the new `nuvec-distributed` crate with MLX's TCP ring (`init` reads
  `MLX_HOSTFILE` and `MLX_RANK`). nuvec-nn: `sharded::{AllToShardedLinear,
  ShardedToAllLinear}` for tensor parallelism. `just bench-distributed` compares with MLX's
  ring.
- `Squeeze` and `ExpandDims` primitives (MLX's): `squeeze`, `expand_dims` and reductions
  without `keepdims` record them, so their shapes follow the input's.
- Shapeless compile (MLX's `shapeless=True`): `compile::Options::shapeless(true)` traces once
  per input structure, dtypes and ranks and replays for any shapes, bit for bit the
  uncompiled function; operations that fix a shape fail the trace with an error naming them.
  `compile::compile_flat_with` takes options.
- `view` (bit reinterpretation, MLX's `mx.view`), `to_fp8` and `from_fp8` (E4M3 bytes).
- nuvec-nn: quantized layers in any mode (`from_linear_with(&l, quant)`,
  `from_embedding_with`, `quantized()`, `dequantized_weight(dtype)`).
- nuvec-lm: MXFP4, MXFP8 and NVFP4 checkpoints (mlx-lm's per-layer `mode`, modes recognised
  from the tensors, expert global scales); AutoAWQ and GPTQ checkpoints (4-bit, and 8-bit
  GPTQ), converted at load to MLX's affine layout (`PackedFormat`,
  `quant::convert_packed`); `quant::mxfp4_from_hf_blocks` for gpt-oss's blocks. Other Hugging
  Face `quantization_config` formats are refused with a message.
- nuvec-io: `F8_E4M3` and `F8_E8M0` tensors load as `uint8`.
- `gpu::Device::kernel_names`.
- `floor_divide` (MLX's, Python's `//`), `power` (MLX's, `**`), `Array::buffer_id` (which
  arrays share storage), `io::safetensors_footprint_filtered` (the footprint of a filtered
  load), and `DeviceInfo::physical_memory`.
- Sliding-window attention: `AttentionOptions::window(W)` (causal only) skips keys outside
  the window in the vector and tiled kernels and in every composition block;
  `paged_attention` takes the same options (window, sinks) and head size 512.
- Head-512 causal prefill in half precision runs MLX's blocked path
  (`AttentionKernel::BlockedD512`): 256 queries per block over only the keys they see, the
  mask applied inside the softmax; bit for bit MLX's result.
- `fast::norm_rope_kv`: for K = V attention (as Gemma 4's), normalised and rotated keys and
  normalised values from one projection in one call.
- nuvec-nn: `RopeScaling` (linear, Llama 3, YaRN, LongRoPE, proportional) with
  `Rope::scaled`, `Rope::mscale` and `Rope::rotation`; `swiglu_oai`, `geglu`,
  `logit_softcap` (compiled, as mlx-lm's).
- nuvec-lm: gpt-oss (sliding and full layers, sinks, YaRN, biased router and experts,
  MXFP4 checkpoints and Hugging Face's MXFP4 originals) and Gemma 4 (`gemma4`,
  `gemma4_text`; per-kind head sizes and RoPE, per-layer inputs, KV-shared layers, K = V,
  the experts branch, the logit soft cap; vision and audio weights skipped). Sliding-window
  layers' caches keep their window plus 256 positions (trimmable within them).
- nuvec-lm tokenizers: o200k's split pattern (gpt-oss, GPT-4o) and SentencePiece-style BPE
  with byte fallback (Gemma, Llama 2, Mistral v0.1–v0.3), equal to Hugging Face
  `tokenizers`' ids and decodes.

### Changed

- Attention with head size 192 or 256 composes (product, softmax, product) as MLX's
  fallback does up to 8 GiB of scores (was 4 GiB), and runs the tiled kernel beyond, where
  the scores no longer fit beside a large model. A 2,048-token chunk over 64K keys takes
  400 ms (was 416; MLX 399), over 131K keys 813 ms (was 841; MLX 809), with MLX's peak
  memory. Qwen3.6-35B-A3B's time to first
  token at 64K tokens: 115 s (was 127).
- Qwen3 and Qwen 3.5/3.6 mixture-of-experts layers weight their experts as mlx-lm's blocks
  do: the router's precise softmax and the weighted sum in the activations' type, the top
  experts chosen by probability. Before, the sum was a float32 product over float32 copies
  of the expert outputs (more memory, and other low bits than mlx-lm's).
- Prompts prefilled together (`[B, L, K]` quantized products) split K only for one batch
  item, as MLX; under `with_row_exact` each item is partitioned as alone. Qwen3-0.6B serves
  8 requests at mlx-lm's speed (was 0.93×).

- `compile` builds its fused kernels as `mx.compile` builds its: constant scalars are
  literals in the kernel, and Metal's default math functions apply, which only `power`
  uses. Fused `power` can differ from the uncompiled result in the low bits, as in MLX.
  Compiled `geglu` (Gemma's gate) at 1M float32 elements runs 2× faster, on par with MLX.

- **Breaking:** quantized and gathered (mixture-of-experts) products route as MLX's: from
  2 rows they take MLX's multi-vector `qmv_wide` (now ported for affine weights too) and
  MLX's matrix-vector and grouped-gather limits, which give a row other low bits than its
  one-row product. `nuvec::with_row_exact` restores per-row products up to
  `ROW_INDEPENDENT_ROWS` rows; speculative verification runs in it (still lossless), and
  `Batcher::batch_invariant(true)` serves each request its solo tokens. Batched serving of
  Gemma 4 26B-A4B and gpt-oss-20b is now on par with mlx-lm (was 0.82–0.98×).
- **Breaking (nuvec-lm):** `Drafter::draft` takes the model and returns a `Draft` (tokens as
  an `Array`, maybe on the device; `Draft::new(tokens)`, its length from the shape), with
  `observe`, `needs_hidden` and `auto`; `Drafter` is implemented for `&mut D` and `Box<D>`;
  `SuffixDecoding`'s suffix length is set with `with_min_match` (the index is keyed by it);
  `speculative_generate` takes a `DraftPolicy` instead of a draft length; `PromptLookup`'s
  host draft is `lookup`.
- **Breaking:** quantization takes a `Quant` and returns a `Quantized` (weight, scales,
  `Option` biases, quant): `w.quantize(Quant::affine(64, 4))`, `q.dequantize(dtype)`,
  `x.quantized_matmul(&q, transpose)`, `x.gather_qmm(&q, lhs, rhs, GatherQmm)` replace the
  `(group_size, bits)` arguments and the separate scales and biases.
- **Breaking:** nuvec-nn's `QuantizedLinear` and `QuantizedEmbedding` hold `biases:
  Option<Array>` and `quant: Quant` instead of `group_size` and `bits`.
- **Breaking:** nuvec-lm's `Quantization` is `{ default: Quant, layers: Vec<LayerQuant> }`,
  and the model builder's callback returns `Option<LayerQuant>`.
- **Breaking:** `gpu::DeviceInfo` is `#[non_exhaustive]`.
- Updates (`slice_update`, `at(..).set`) on the CPU write into a base nothing else holds,
  as MLX's CPU does, instead of copying it.
- `softmax` of float16 and bfloat16 computes in float32 and rounds once on every path (the
  CPU and axes other than the last too), as MLX's `precise=True`; the fused kernel already
  did.
- Untransposed quantized products with `K` from 1024 run MLX's `qvm_split_k` (affine too),
  so their results equal MLX's; sorted gathers find each expert's rows in one dispatch.
- **Breaking:** `paged_attention` takes `&AttentionOptions` (window, sinks).
- **Breaking:** nuvec-lm's `config::RopeScaling` is nuvec-nn's.
- Affine dequantize on the GPU computes `code · scale + bias` as one fused multiply-add in
  float32, as MLX's Metal kernel (equal to MLX's GPU bit for bit; the CPU keeps MLX's CPU
  rounding).
- `gelu_approx` cubes with `power`, as MLX's `x**3` (equal to MLX bit for bit in half
  precision).

### Fixed

- Tensors read from misaligned checkpoints (mlx-community's) kept their file offsets in one
  buffer; a quantized product ran up to 4% slower depending on where its weights fell. They
  now start at multiples of 256 bytes: Gemma 4 E4B decode 0.99× → 1.00× mlx-lm.
- Operations on arrays of 2^32 elements or more ran on the CPU (40 s instead of 0.8 for
  attention over 8 GiB of scores) and were never written in place. Element-wise operations,
  row softmax and log-sum-exp, and matrix products on the tiled kernel now run on the GPU
  with 64-bit indexing, as MLX's; fused kernels, reductions, gathers and updates of that
  size still fall back to the CPU.
- A batch-invariant `Batcher` now gives each request its solo tokens on Gemma 4
  26B-A4B too. Grouped prefill chose quantized and mixture-of-experts kernels by the
  group's size, and head-512 decode ran a different attention kernel than solo decoding
  below 1,024 keys.
- Sorted mixture-of-experts gathers indexed their operands as dense arrays: a weight sliced
  along an axis, a broadcast expert axis or a K-slice of a wider activation gave wrong
  results. They are made dense first.
- A NAX library or kernel that failed to load failed every NAX product from then on; any NAX
  failure now turns NAX off and the product runs on the existing kernels.
- Attention with no queries failed and with no keys (causal) panicked; both give their
  result (empty, or zeros).
- `compile` merged identical collectives (two equal `recv`s received one message) and
  treated `stop_gradient` as an identity.
- `checkpoint_flat` drew new random numbers when recomputing (the tree form did not).
- A custom batching rule returning too few outputs, or a wrong batch axis, gave wrong
  shapes; it is an error naming the primitive.
- `distributed`: after a failed transfer the ring read misaligned bytes; it now fails every
  later operation with the first error. A mismatched `recv` drains the payload.
  `NUVEC_DISTRIBUTED_TIMEOUT` rejects values that are not positive numbers (it panicked).
- nuvec-lm: `attach_mtp` left a half-built head when loading failed; `load_adapters`
  refused mlx-lm's `"num_layers": -1`; `linear_to_lora_layers` without keys failed on
  mixture-of-experts models and could leave a model partly converted; SuffixDecoding's
  eviction scanned the whole store; model-free drafters drafted twice per token;
  `resolve` ignored `HF_HUB_CACHE`; qwen2, gpt-oss and Qwen 3.5 configs without
  `rope_theta` used another default than mlx-lm.
- nuvec-nn: `MultiOptimizer` panicked on an optimizer returning too few parameters.
- The allocator evicted cached buffers that GPU work still used, and fence waits were
  deduplicated by address (a reused address could skip a wait).
- GPU work could reuse memory of the command buffer committed just before it, which made
  the new command buffer wait for that one instead of overlapping it: decode attention ran
  at 1.15–1.2× MLX's time in bursts (D-058's open gap). Buffers up to 64 MiB of the
  latest committed command buffer are no longer handed to new GPU work; idle memory is
  preferred. Head-size-512 decode in float32 (8 MiB partials) went from 1.05× to 0.97×
  MLX's time; a 28K-token prefill keeps its peak.
- Updates and element-wise operations sometimes copied an input they could have written in
  place (donation) when the GPU work that produced it completed at the same moment: the
  completion handler's brief handle, or the evaluation's stale record of the input, hid
  that nothing else held it.
- Attention kernels compiled their sliding-window code into every model's kernels (a
  function constant in the key loops), which cost about 8% where no window is used. The
  window is now a template parameter: windowless kernels compile to MLX's code
  (head-size-512 decode in float16 from 1.06× to 0.98–1.00× MLX's time).
- Sliding-window decode attention depended on where the window's keys sat in the cache
  (keys were assigned to SIMD groups by absolute position), so the `Batcher`'s paged cache
  and `generate`'s compacted cache rounded differently and gpt-oss's batched tokens drifted
  from solo decoding after tens of tokens. The vector and paged kernels now count a
  window's keys from its first key: each row equals MLX's call over exactly those keys.
- `softmax` on the CPU and the attention composition (with sinks too) computed float64 in
  float32.
- Kernels compiled at runtime (`NUVEC_METALLIB=off`) used Metal's fast math functions; they
  now use the precise ones, as the precompiled library.

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
