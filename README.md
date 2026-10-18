# nuvec

A pure-Rust array framework for Apple Silicon in the spirit of
[MLX](https://github.com/ml-explore/mlx): lazy arrays on unified memory, composable function
transforms (`grad`, `vjp`, `jvp`, `vmap`, `compile`) and Metal kernels. It is written in
Rust from the ground up, with **no bindings** to MLX or any other C/C++ library.

> **Status: pre-alpha.** Lazy arrays with 14 dtypes, MLX type promotion, broadcasting,
> zero-copy views, element-wise operations with weak Rust scalars (`&x * 2.0` keeps `x`'s
> dtype), reductions, matmul and composable transforms (`grad`, `value_and_grad`, `vjp`,
> `jvp`, `vmap`, `custom_function`, `checkpoint`; also through complex numbers), random numbers
> bit-identical to MLX's, MLX's GEMM and GEMV kernels (ported, with split-K and a fused
> `addmm`), MLX-compatible affine quantization (`quantize`, `dequantize` and
> `quantized_matmul` with MLX's packing and kernels, 1–8 bits), MLX's fused kernels
> (`fast::rms_norm`, `fast::layer_norm`, `fast::rope`, `fast::cross_entropy`,
> `fast::scaled_dot_product_attention`, last-axis `softmax` and `logsumexp`), checkpoint I/O
> (`nuvec-io`: safetensors and npy, memory-mapped without a copy), and first
> neural-network pieces (`nuvec-nn`: `Linear`, `QuantizedLinear`,
> cross-entropy, SGD, Adam; `just mnist` trains MLX's MNIST example model). Arrays are
> evaluated on the GPU by default (Metal kernels on unified memory, submitted asynchronously;
> `async_eval` overlaps host work with GPU work) or on the CPU
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
- `just bench-load`: loading a 2 GiB safetensors checkpoint, nuvec vs MLX
- `just io-fixtures`: re-records the safetensors and npy files written by MLX and NumPy

## Crates

| Crate | Responsibility |
|---|---|
| `nuvec` | Public facade: re-exports the layers below |
| `nuvec-core` | Lazy array graph, primitives, dtypes, streams and evaluation |
| `nuvec-compile` | Tracing, graph simplification, fusion and buffer planning |
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
