# nuvec

A pure-Rust array framework for Apple Silicon in the spirit of
[MLX](https://github.com/ml-explore/mlx): lazy arrays on unified memory, composable function
transforms (`grad`, `vjp`, `jvp`, `vmap`, `compile`) and Metal kernels. It is written in
Rust from the ground up, with **no bindings** to MLX or any other C/C++ library.

> **Status: pre-alpha.** Lazy arrays with 14 dtypes, MLX type promotion, broadcasting,
> zero-copy views, indexing and joining (`take`, `take_along_axis`, `put_along_axis`,
> `concatenate`, `stack`, `split`, `slice_update`, written in place on the GPU when nothing
> else refers to the updated array), element-wise operations with weak Rust scalars (`&x * 2.0` keeps `x`'s
> dtype), reductions, sorting and cumulative scans (`sort`, `argsort`, `partition`,
> `argpartition`, `topk`, `cumsum`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`; MLX's
> Metal kernels), matmul and composable transforms (`grad`, `value_and_grad`, `vjp`,
> `jvp`, `vmap`, `custom_function`, `checkpoint`; also through complex numbers), `compile`
> (traced once per input signature, simplified, element-wise chains fused into generated
> Metal kernels), custom Metal kernels from source (`fast::MetalKernel`, as MLX's
> `metal_kernel`), random numbers (uniform, normal, Gumbel, categorical, integers, Bernoulli)
> bit-identical to MLX's, MLX's GEMM and GEMV kernels (ported, with split-K and a fused
> `addmm`), MLX-compatible affine quantization (`quantize`, `dequantize` and
> `quantized_matmul` with MLX's packing and kernels, 1–8 bits, and `gather_qmm` for
> mixture-of-experts layers), MLX's fused kernels
> (`fast::rms_norm`, `fast::layer_norm`, `fast::rope`, `fast::cross_entropy`,
> `fast::scaled_dot_product_attention`, last-axis `softmax` and `logsumexp`; mlx-lm's
> Gated DeltaNet kernels as `fast::gated_delta`), checkpoint I/O
> (`nuvec-io`: safetensors and npy, memory-mapped without a copy), and neural-network
> building blocks (`nuvec-nn`: modules as named parameter trees loaded by name, `Linear`,
> `QuantizedLinear`, `Embedding`, `RmsNorm`, `LayerNorm`, `Rope`, LoRA, activations,
> cross-entropy, SGD, Adam, AdamW, Lion and Muon with learning-rate schedules and gradient
> clipping; `just mnist` trains MLX's MNIST example model), and an LLM runtime (`nuvec-lm`:
> Llama, Mistral, Qwen 2, Qwen 3 and Qwen 3.5 (hybrid Gated DeltaNet and attention, dense
> or mixture-of-experts) models
> from Hugging Face or MLX checkpoints, including
> MLX's 4-bit ones, a byte-level BPE tokenizer read from `tokenizer.json`, a KV cache and
> mlx-lm's sampler (greedy, temperature, top-p, min-p, top-k); `cargo run --release -p nuvec-lm --example generate`). Arrays are
> evaluated on the GPU by default (Metal kernels on unified memory, submitted asynchronously;
> `async_eval` overlaps host work with GPU work, and `schedule` orders a graph ahead of
> starting it) or on the CPU
> (`set_default_stream(Stream::default_for(Device::cpu()))`).

## Platforms

- **macOS 15+ on Apple Silicon:** the primary target, with the full Metal backend.
- **Linux:** the core and CPU crates build for development. The GPU crates compile as stubs.

## Prerequisites

| Tool | Install |
|---|---|
| Rust (pinned in `rust-toolchain.toml`) | [rustup](https://rustup.rs). The pinned toolchain installs automatically |
| Xcode and the Metal toolchain | `xcodebuild -downloadComponent MetalToolchain` (check with `xcrun -sdk macosx -f metal`) |
| [just](https://just.systems) | `brew install just` |
| [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) | `cargo install cargo-deny --locked` |
| Miri (nightly) | `rustup +nightly component add miri` |
| jq | `brew install jq` |

Without the Metal toolchain the build still succeeds. It prints one warning and embeds the
shader sources for runtime compilation. Set `NUVEC_METALLIB=off` to choose this mode
explicitly.

## Quick start

```sh
git clone https://github.com/hupe1980/nuvec && cd nuvec
just ci
```

`just` lists every task. `just ci` runs the same gates as CI: formatting, clippy, tests,
docs, Miri, cargo-deny, the Metal kernel library check, and the crate-layering and
repository-hygiene checks.

## Testing against MLX

The test suite replays outputs recorded from real MLX (`crates/nuvec-core/tests/fixtures/mlx.bin`).
Python is not needed to build or test. With [uv](https://docs.astral.sh/uv/) installed:

- `just fixtures`: re-records the fixtures with the pinned MLX version on its CPU device
- `just bench-mlx`: prints a same-machine benchmark table, nuvec vs MLX
- `just bench-gemm`, `just bench-quant`, `just bench-fast`: GPU matrix products, quantized
  matrix products and fused operations, nuvec vs MLX
- `just bench-compile`: compiled element-wise functions, nuvec vs MLX
- `just bench-load`: loading a 2 GiB safetensors checkpoint, nuvec vs MLX
- `just bench-lm`: text generation (prefill and decode) with a model from the local Hugging
  Face cache, nuvec-lm vs mlx-lm
- `just io-fixtures`: re-records the safetensors and npy files written by MLX and NumPy
- `just lm-fixtures`: re-records tiny models of each LLM variant and their outputs with mlx-lm
- `just nn-fixtures`: re-records the `mlx.nn` and `mlx.optimizers` results `nuvec-nn` is
  tested against

## Crates

| Crate | Responsibility |
|---|---|
| `nuvec` | Public facade: re-exports the layers below |
| `nuvec-core` | Lazy array graph, primitives, dtypes, streams and evaluation |
| `nuvec-compile` | `compile`: tracing, graph simplification and element-wise kernel fusion |
| `nuvec-gpu` | Metal runtime: device, kernel pipelines, shared buffers, command encoding with minimal barriers |
| `nuvec-kernels` | Metal kernel sources and the precompiled `nuvec.metallib` |
| `nuvec-cpu` | Pure-Rust CPU backend |
| `nuvec-io` | Zero-copy safetensors and npy loading, IDX |
| `nuvec-distributed` | Collectives over TCP rings and Thunderbolt RDMA |
| `nuvec-nn` | Neural-network modules, layers and optimizers |
| `nuvec-lm` | LLM runtime: model loading, KV caches, sampling |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
