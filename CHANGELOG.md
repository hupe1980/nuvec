# Changelog

All notable changes to nuvec are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the API).

## [Unreleased]

## [0.8.0] - 2026-11-09

### Added

- The rest of `mlx.core`: `hadamard_transform` (MLX's kernels, its bits on the GPU),
  `segmented_mm` (MLX's segmented GEMM), `block_masked_mm`, `qqmm` and `gather_qqmm`
  (activations and weights quantized in MXFP4, MXFP8 or NVFP4; `QqmmWeight` takes a dense
  or a quantized weight; MLX's quantized backward) and `einsum_path`. Every `mlx.core` and
  `mx.linalg` function now has a nuvec counterpart.
- MLX's convolutions: `conv2d`, `conv3d`, `conv_general` (`ConvOptions`: asymmetric and
  negative padding, kernel and input dilation, groups, flip), `conv_transpose1d`, `2d`, `3d`
  and `convolve` (`ConvolveMode`); `ConvArg` takes one value or one per axis. The GPU routes
  as MLX: implicit GEMM (2-D, 3-D), the general 2-D kernel, Winograd F(6×6, 3×3), the
  depthwise 2-D kernel, otherwise unfold and GEMM; nuvec's depthwise kernel covers 1-D, 3-D
  and depth multipliers. Winograd keeps float32 tiles for half-precision inputs
  (`NUVEC_CONV_WINOGRAD=fast`: MLX's half tiles, `0`: off). Differentiable, batched under
  `vmap`.
- nuvec-nn: `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`,
  `ConvTranspose3d`, `MaxPool1d`, `MaxPool2d`, `MaxPool3d`, `AvgPool1d`, `AvgPool2d` and
  `AvgPool3d`, as MLX's `nn` (initialisation, padding values, names).
- The `conv` and `special` benchmarks against MLX's GPU (`just bench-conv`, `just bench-special`).
- `just release X.Y.Z` (version, changelog, every gate, commit and tag) and
  `just bench-parity --tier smoke` (two models, about 10 minutes): releases no longer wait
  for the long benchmarks.
- `nuvec::linalg`, MLX's `mx.linalg`: `qr`, `svd` (and `svd_values`), `eigh`, `eigvalsh`,
  `eig`, `eigvals`, `lu`, `lu_factor`, `cholesky`, `inv`, `tri_inv`, `cholesky_inv`, `pinv`,
  `solve`, `solve_triangular`, `det`, `slogdet`, `cross` and `norm`, in float32 and float64,
  batched, with MLX's conventions (LU pivots included). The decompositions run on the CPU
  (pure-Rust `faer`, or Accelerate's LAPACK with the feature `accelerate`), also on a GPU
  stream, without a copy. Unlike MLX's, they are differentiable: `inv`, `solve`,
  `solve_triangular`, `tri_inv`, `cholesky`, `cholesky_inv`, `qr`, `eigh`, `eigvalsh`,
  `svd`, `pinv`, `det` and `slogdet`.
- `matrix_norm`: the 2, −2 and nuclear orders.
- The feature `accelerate` (`nuvec`, `nuvec-core`, `nuvec-cpu`): Accelerate's LAPACK for
  `linalg` on Apple targets.
- MLX's missing element-wise functions as primitives, with CPU and Metal kernels, MLX's
  derivatives, vmap and fusion under `compile`: `erf`, `expm1`, `log1p`, `log2`, `log10`,
  `tan`, `sinh`, `cosh`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh`,
  `rsqrt`, `floor`, `ceil`, `round` (with decimals), `arctan2`, `remainder`, `bitwise_and`,
  `bitwise_or`, `bitwise_xor`, `bitwise_invert`, `left_shift`, `right_shift`, with MLX's
  NumPy-style aliases (`acos`, `atan2`, …).
- MLX's compositions: `square`, `reciprocal`, `trunc`, `degrees`, `radians`, `positive`,
  `real`, `imag`, `conj`, `divmod`, the logical operations, `isnan`, `isinf`, `isfinite`,
  `isposinf`, `isneginf`, `nan_to_num`, `isclose`, `allclose`, `array_equal`, `clip`,
  `count_nonzero`, `minmax`.
- MLX's shape, creation and linear-algebra functions: `flip`, `roll`, `tile`, `pad` (four
  modes), `moveaxis`, `swapaxes`, `matrix_transpose`, `flatten`, `unflatten`, `unstack`,
  `atleast_1d`/`2d`/`3d`, `broadcast_arrays`, `tri`, `tril`, `triu`, `eye`, `identity`,
  `diag`, `diagonal`, `trace`, `linspace`, `meshgrid`, the windows, `tensordot`, `inner`,
  `outer`, `kron`, `vecdot`, `einsum`, `diff`, `median`, `searchsorted`, `unique`,
  `as_strided`, `empty`, `empty_like`, `matrix_norm` (without the SVD-based orders), and the
  dtype predicates `issubdtype`, `isdtype`, `can_cast`, `result_type`.
- nuvec-nn: MLX's activations `gelu`, `relu2`, `relu6`, `leaky_relu`, `prelu`, `elu`,
  `selu`, `celu`, `softplus`, `softsign`, `softshrink`, `hard_shrink`, `hard_tanh`,
  `hardswish`, `mish`, `log_sigmoid`, `log_softmax`, `softmin`, `glu`, `step`.
- `scripts/op_coverage.py`: `mlx.core`'s functions against nuvec's.
- `fast::linear_cross_entropy` and `nuvec_nn::losses::linear_cross_entropy`: the cross
  entropy of an output head computed in row chunks, so the `[N, vocab]` logits never exist.
  Under a gradient transform each chunk's hidden-state gradient comes from the same logits
  (no extra product); a trainable head's gradient is computed only when used.
- `Model::token_losses` (the chunked loss over a model's head: dense, quantized, tied, or
  Gemma 4's soft-capped) and `Model::forward_hidden_uncached`.
- Fine-tuning options of `mlx_lm.lora`, in `nuvec_lm::tuner` and the `lora` example:
  `--fine-tune-type` (`lora`, `dora`, `full`), `--optimizer` (`adam`, `adamw`, `muon`,
  `sgd`, `adafactor`), `--grad-checkpoint`, `--resume-adapter-file`, `--test` and
  `--test-batches`, and `--config` YAML files with `lora_parameters`, `optimizer_config`
  and `lr_schedule`. `tuner::build_optimizer`, `tuner::build_schedule`, `tuner::read_config`
  and `lora::apply` are the library side.
- `Model::set_grad_checkpoint`: blocks recompute their activations in the backward pass.
- DoRA: `LoraLinear::dora` (mlx-lm's `DoRALinear`) and `lora::linear_to_dora_layers`;
  DoRA and full fine-tuning adapter directories load in mlx-lm and nuvec-lm.
- `nuvec_nn::optim::Adafactor`, as MLX's `optimizers.Adafactor`.
- `lora::save_weights` and `lora::load_weights` (trained weights, mlx-lm's numbered
  checkpoints and `resume_adapter_file`).
- Chat templates (`nuvec_lm::chat`, feature `chat`, on by default): `ChatTemplate` renders a
  checkpoint's templates as transformers' `apply_chat_template` (its environment, template
  selection, special tokens, tools, reasoning fields and `continue_final_message`), with
  `minijinja`. Text and ids equal transformers 5.22.1's for 33 mlx-community checkpoints.
  `chat::prompt_ids` builds `mlx_lm.generate`'s prompt.
- `TokenizerConfig`: `tokenizer_config.json`'s special tokens and chat templates.
- mlx-lm's chat (`{"messages", "tools"}`) and completion (`{"prompt", "completion"}`)
  training data and `--mask-prompt` (`tuner::DataOptions`, `Dataset::from_rows`,
  `Dataset::load_with`, `Dataset::rows`), with the `*_feature` keys of the `--config` file.
- Tekken's split pattern (Mistral NeMo, Ministral, Mistral Small 3): these tokenizers load.
- The `generate` example takes `mlx_lm.generate`'s template options: `--system-prompt`,
  `--prefill-response`, `--chat-template-config` and `--ignore-chat-template`.

### Changed

- `einsum` contracts three operands or more in MLX's greedy order (opt_einsum's), as
  `einsum_path` reports, instead of left to right.
- `conv1d` runs MLX's convolution kernels on the GPU (its depthwise kernel stays nuvec's).
  `graph::FastOp::DepthwiseConv1d` is now `graph::FastOp::Conv`, every convolution's fused
  operation.
- Adafactor, Adam and the fast operations' fallbacks use `square` and `rsqrt`.
- The `generate` example renders the prompt with the checkpoint's chat template by default,
  as `mlx_lm.generate` (`--chat` and its hard-coded ChatML are gone).
- Batches come in `mlx_lm.lora`'s order: a loaded dataset is sorted by its raw rows' number
  of keys (mlx-lm's `CacheDataset`; tokenized datasets stay sorted by length), and batches
  are visited in numpy's `np.random.permutation` order, from one `np.random.seed(seed)`
  state shared by training and validation. `TrainingArgs::seed` and `iterate_batches`'s
  seed are `u32`, as numpy's.
- The `lora` and `generate` examples need the `chat` feature.
- LoRA training: `default_loss` uses the chunked loss; a LoRA layer's backward pass keeps
  only its input (not its float32 copy) and frees its cotangent before the layer below;
  `swiglu` keeps only its two inputs. Against `mlx_lm.lora` (batch 4, 50 iterations):
  Qwen3-0.6B-4bit trains at 1.13× its speed at 0.42× its peak memory (from 0.98× and
  0.86×), Llama-3.2-3B-Instruct-4bit at 1.02× and 0.60×, and Qwen3-0.6B with rows of
  about 2,000 tokens and gradient checkpointing in both at 1.03× and 0.28×.

### Fixed

- `just bench-parity --release` runs again after a release run: its own records no longer
  count as uncommitted changes.
- The `lora` example seeds the generator with `--seed` before drawing the adapters, as
  mlx-lm; its losses no longer change from run to run.
- Text training rows end with the tokenizer's EOS (`tokenizer_config.json`), as mlx-lm's,
  not `config.json`'s first: Llama 3.x rows ended with `<|end_of_text|>` instead of
  `<|eot_id|>` (`tuner::eos_token_id`).
- A gradient checkpoint differentiates only the inputs whose gradients are used, so a
  checkpointed block over frozen quantized weights no longer fails.
- `tuner::train` takes any optimizer, compiles the accumulated steps too (as mlx-lm), saves
  the trainable weights (mlx-lm's `trainable_parameters`) and numbered checkpoints.
- The `lora` example trains only with `--train`, as `mlx_lm.lora`.
- `AdapterConfig` has a `fine_tune_type`; `adapter_config.json` records it.
- `Model::forward_hidden` and `Model::logits` work for Gemma 4 (the soft-capped head).

## [0.7.0] - 2026-11-06

### Added

- Encrypted ring traffic: with a key, every connection derives a key per direction
  (HKDF-SHA256) and seals every frame with ChaCha20-Poly1305. A changed, replayed or
  reordered frame, or a rank that does not encrypt, fails with an error naming the peer.
  Results are bit-identical to an unencrypted ring's. An encrypted 32 MB hop runs at about
  1.7 GB/s over loopback. `Options::encrypt(false)` or `NUVEC_DISTRIBUTED_ENCRYPT=0` keeps
  authentication only.
- A test of a peer that stops mid-transfer without closing its connections (a sleeping Mac):
  the waiting transfer fails within the timeout on the GPU stream, and the others re-form.
- `scripts/bench_distributed.py` measures encrypted rings and a 32 MB pipeline hop.
- LoRA training as mlx-lm's tuner (`nuvec_lm::tuner`): `text` JSONL datasets, mlx-lm's
  length-sorted padded batches and masked loss, a compiled training step, gradient
  accumulation, validation, reports and adapters in mlx-lm's layout; the `lora` example
  takes `mlx_lm.lora`'s options. Qwen3-0.6B-4bit trains at 0.96× `mlx_lm.lora`'s speed at
  0.86× its peak memory (`scripts/bench_lora.py`).
- `Model::forward_uncached`: logits without a cache, as mlx-lm's `model(inputs)`.
- The `ttft_parts` example (nuvec-lm) and `scripts/ttft_parts.py` time a short prompt's
  first token part by part, in both engines; `scripts/bench_creation.py` times creation ops.

### Changed

- The ring's handshake carries the mode (authenticated or encrypted), so keyed ranks of
  earlier versions cannot join a keyed group of this one.
- Attention gradients follow MLX's backward: under a gradient transform the tiled
  attention kernel keeps each row's log-sum-exp, and the backward is one operation over
  tiles of 1,024 × 1,024 that reuses one set of tile buffers and accumulates in float32
  (no softmax recomputed). Forward plus backward takes 1.00–1.02× MLX's time at 2,048 and
  4,096 keys (from up to 2.3×), with peak memory 0.78–0.87× MLX's (from up to 6.1×).
- `zeros`, `ones` and `full` of 64 KB and more are filled when they are evaluated, on the
  evaluating stream's device, as MLX's: creating one takes about 1 µs and allocates nothing
  until it is evaluated (256 MB `full`: 16.8 ms on the host, now 1.4 ms on the GPU, MLX
  1.6 ms). Smaller ones are filled at once on the host. An allocation that fails is
  reported when the array is evaluated.
- A copy of one element broadcast fills 16 bytes per GPU thread, as MLX's scalar copy.
- nuvec-lm's KV cache grows with zeros written on the GPU, as mlx-lm's lazy `mx.zeros`,
  instead of zeroing fresh memory on the host: the first token of a 25-token prompt on
  Qwen3-0.6B after mlx-lm's cache clears 16.5 → 14.5 ms (mlx-lm 15.2 ms).

## [0.6.1] - 2026-11-03

### Fixed

- A collective on a GPU stream that waits more than a few seconds for its peers (a peer in a
  long prefill, or starting late) no longer fails with `GPU Timeout Error`: the GPU work
  that reads its result is committed once it is done, so the GPU never waits for the
  network (MLX's communication stream waits on the GPU). Tensor-parallel decode on one Mac
  is up to 7% slower than in 0.6.0 (Qwen3-0.6B, 2 ranks), about 3% on Qwen3-8B.

## [0.6.0] - 2026-11-03

### Added

- Distributed backends as MLX's: `distributed::init_backend(strict, Backend::{Any, Ring, Jaccl})`
  and `distributed::is_available(backend)`; a second `init` of a backend returns its group.
- Experimental JACCL backend (feature `jaccl`): MLX's collectives over Thunderbolt 5 RDMA,
  mesh or ring, started by `mlx.launch --backend jaccl|jaccl-ring`; `jaccl::devices()` lists
  RDMA devices. Not yet verified on Thunderbolt 5 hardware.
- `distributed::sum_scatter`, on every backend, with gradient, forward-mode and `vmap` rules.
- `Group::close` and `Group::health`; TCP keepalive on every connection. `init` never returns
  a failed or closed group, so a group can re-form in the same process.
- Pre-shared-key authentication of distributed connections (`NUVEC_DISTRIBUTED_KEY`,
  `Options::key`, HMAC-SHA256).
- Tensor parallelism (nuvec-lm), as mlx-lm's `model.shard(group)`: `Model::shard` and
  `load_model_sharded`.
- Pipeline parallelism (nuvec-lm), as mlx-lm's `--pipeline`: `load_model_pipelined(dir,
  group, Split)` and `Model::pipeline`. Each rank reads only its stage's tensors; the `Batcher`
  keeps one micro-batch per stage in flight. Gemma 4 is not supported.
- Generation controls (nuvec-lm): `GenerateOptions::stop` (`Stop::tokens`, `Stop::strings`),
  `ignore_eos` and `min_tokens` in `generate`, `Batcher` and `speculative_generate`;
  `TextStream` streams text without stop strings; `finish_reason` on `Generation` and
  `SpeculativeGeneration`.
- `nuvec::memory` (MLX's memory functions: `active_memory`, `peak_memory`, `set_memory_limit`,
  `set_wired_limit`, `clear_cache`, `physical_memory`, …) and `nuvec::metal` (`is_available`,
  `device_info`, `nax`, `profile`, `start_capture`/`stop_capture`), with or without a GPU.
- `ops::depends`, `Error::is_oom`, `Model::kv_bytes_per_position`, `Quantized::affine` and
  `Quantized::fp`.
- `io::safetensors_tensors`, `io::align_safetensors` and `io::load_safetensors_cached`;
  nuvec-lm keeps aligned copies under `NUVEC_ALIGNED_CACHE`, so mlx-community checkpoints map
  without a copy after the first load. Filtered loads of aligned files map only kept tensors.
- Fused kernels, bit-identical to their compositions: `fast::norm_rope_at`,
  `fast::moe_route`, `fast::weighted_sum`, `fast::residual_rms_norm` and
  `fast::causal_conv1d_slots`; `PagedCache::insert_many`.
- `MLX_GDN_PACKED=0` keeps Gated DeltaNet off mlx-lm's packed kernel, as in mlx-lm.
- nuvec-gpu: an experimental CUDA runtime (feature `cuda`, Linux); arrays do not use it.
- `just bench-parity` records speculative decoding against mlx-lm's plain decode.

### Changed

- Fewer dispatches per layer, tokens unchanged: the `Batcher`'s decode step, batched prefill
  into the paged cache, Qwen 3 MoE, gpt-oss and Gemma 4 expert routing, Gemma 4 norm chains,
  and Gated DeltaNet convolution in a `Batcher` step.
- Decode attention at head size 512 (and 256 in float32) reads one key per step: 5–7× faster
  in bfloat16 from 1K to 32K keys; Gemma 4 26B decodes 1.2–1.4× mlx-lm's speed at 3.7K–7.4K
  tokens of context.
- Wired memory uses residency sets of at most 5% of the working set each, as MLX 0.32
  (`MLX_RESIDENCY_SET_MAX_PCT`, `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`);
  `MemoryStats::residency_sets` counts them.
- `fast::paged_write` takes several sequences (`[B, T, H, D]`, slots `[B, T]`) and strided
  sources.
- `distributed::init(strict)` tries every backend (the ring, then JACCL when built in).
- `send`/`recv` frame an array as two messages; ranks must run the same nuvec version.
- Collectives on a GPU stream run on the group's thread without ending the command buffer,
  as MLX's communication stream, and reduce a donated input in place.
- The TCP ring follows MLX's ring algorithm (both directions, pipelined, in place), with one
  poll(2) thread per transfer, no copy of row-contiguous inputs and 3.7× faster bfloat16 sums:
  all_sum takes 0.24–0.87× MLX's time on 2 ranks and 0.32–0.83× on 4 (4 KB–256 MB).
  Two-rank tensor-parallel Qwen3-0.6B on one Mac: decode 1.36× mlx-lm, prefill 1.03×.
- Flat element-wise kernels of 2^16 elements and more compute 8 bytes' worth per thread, as
  MLX's: a bfloat16 `[2048, 1024]` times a scalar 31.7 → 20.8 µs (MLX 19.6 µs).
- Prefill attention writes the layout the heads' transpose needs, so the transpose no longer
  copies.
- A command buffer that makes new buffers commits after 50 dispatches, as MLX's: first token
  of a 4-token prompt on Qwen3-0.6B after `clear_cache` 9.6 → 8.8 ms.
- Tensor-parallel generation samples on every rank when sampling is greedy or seeded, as
  mlx-lm (`Model::share_sampled`).
- The `pipeline_generate` example runs as `mlx_lm.benchmark` (`--random-prompt`,
  `--ignore-eos`, `--trials`).

### Fixed

- The TCP ring retries transient send failures (`ENOBUFS`, would-block, interrupted) instead
  of failing the group, giving up after 10 s without progress.
- nuvec-lm empties the allocator's cache where mlx-lm does (after each prefill chunk, after
  the first token and every 256 after it, and every 512 `Batcher` steps); two pipeline ranks
  of Qwen3-30B-A3B on one Mac no longer swap (0.1–1.4 → 104–131 tok/s).
- Casts to bfloat16 on the CPU round to nearest even, as MLX's (through float32); about one
  float32 value in 20,000 was rounded down.

### Breaking

| Before | Now |
|---|---|
| `nuvec::gpu::Device::shared()…memory()`, `.set_memory_limit(..)`, … | `nuvec::memory::{stats, set_memory_limit, …}` |
| `nuvec::gpu::Device::shared()…info()`, `.nax()` | `nuvec::metal::{device_info, nax}` |
| `nuvec::gpu::profile` | `nuvec::metal::profile` |
| `nuvec::gpu`, `nuvec::cpu` (the backend crates) | removed from the facade |
| `Error::Op { at: &Location, .. }` | `at: Option<&Location>` (none for an unlocated op error) |
| `Batcher::new(&model, blocks, block_size, max_batch)` | `Batcher::new(&model, BatcherOptions { blocks: Some(blocks), block_size, max_batch, ..Default::default() })` |
| `Batcher::batch_invariant(on)`, `Batcher::micro_batches(n)` | `BatcherOptions::batch_invariant`, `BatcherOptions::micro_batches` |
| `FinishReason::Error(String)` | `FinishReason::Error(nuvec::Error)` |
| `GenerateOptions: Copy` | `Clone` only (it holds stop lists) |

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

[Unreleased]: https://github.com/hupe1980/nuvec/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/hupe1980/nuvec/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/hupe1980/nuvec/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/hupe1980/nuvec/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/hupe1980/nuvec/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/hupe1980/nuvec/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/hupe1980/nuvec/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/hupe1980/nuvec/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/hupe1980/nuvec/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hupe1980/nuvec/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hupe1980/nuvec/releases/tag/v0.1.0
