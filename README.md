<div align="center">

# ⚡ nuvec

**MLX for Rust, written in Rust.** Lazy arrays, composable transforms and Metal kernels for Apple Silicon.

[![CI](https://github.com/hupe1980/nuvec/actions/workflows/ci.yml/badge.svg)](https://github.com/hupe1980/nuvec/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-hupe1980.github.io%2Fnuvec-blue)](https://hupe1980.github.io/nuvec)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)](#-license)
![Rust 1.99+](https://img.shields.io/badge/rust-1.99%2B-orange)
![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon%20%C2%B7%20Metal-black)

[Documentation](https://hupe1980.github.io/nuvec/docs/) · [Getting started](https://hupe1980.github.io/nuvec/docs/getting-started/) · [Benchmarks](https://hupe1980.github.io/nuvec/docs/benchmarks/)

</div>

nuvec runs lazy array computations on Apple Silicon. Differentiate and batch functions with `grad` and `vmap`, fuse operations with `compile`, or load a language model with `nuvec-lm`. The Metal backend requires macOS 15 or newer. The core and CPU crates also run on Linux. Rust 1.99 or newer is required.

## Quick start

```sh
cargo add nuvec
```

```rust
use nuvec::{grad, prelude::*};

fn main() -> Result<()> {
    let x = array![1.0, 2.0, 3.0];
    let sum_squares = |x: &Array| x.multiply(x).sum(..);
    let dx = grad(sum_squares)(&x)?;
    assert_eq!(dx.to_vec::<f32>()?, [2.0, 4.0, 6.0]);
    Ok(())
}
```

Array operations build a graph. Reading a result with `item()` or `to_vec()` evaluates it and reports shape or dtype errors with the original call site. [Arrays and indexing](https://hupe1980.github.io/nuvec/docs/arrays/)

## Models

- **Training.** Add `nuvec-nn` for layers, optimizers and `#[derive(Module)]`. Use `compile_mut` to compile a training step over a model and its optimizer. [Neural networks](https://hupe1980.github.io/nuvec/docs/neural-networks/)
- **Inference.** Add `nuvec-lm` to stream tokens from local Hugging Face checkpoints. It supports batched and speculative decoding, quantized KV caches and multi-Mac inference. [Supported models and generation](https://hupe1980.github.io/nuvec/docs/llm/)

From the repository, try the generation example with a cached model.

```sh
hf download mlx-community/Qwen3-8B-4bit
cargo run --release -p nuvec-lm --example generate -- \
    --model mlx-community/Qwen3-8B-4bit --prompt "Write a haiku about Rust"
```

## Performance

Across four checkpoints on a 48 GB M4 Pro, nuvec-lm reached 1.01× the decode throughput of mlx-lm 0.32.2 and took about the same time to the first token with a 2,048-token prompt. The benchmark uses the same token IDs and reports per-model samples. [Measurements and method](https://hupe1980.github.io/nuvec/docs/benchmarks/#m4-pro-qualification-results)

## Building and testing

```sh
git clone https://github.com/hupe1980/nuvec
cd nuvec
just ci
```

Install `just`, `cargo-deny`, `jq` and nightly Miri to run the local CI checks. Rustup selects the pinned compiler. Install Apple's Metal toolchain with `xcodebuild -downloadComponent MetalToolchain` for ahead-of-time kernel compilation. The runtime can compile embedded shaders on first use.

The Rust tests replay checked-in MLX fixtures. To regenerate packed AutoAWQ and GPTQ checkpoints, run `just lm-fixtures --only packed`. To compare the numerical fixtures and two cached Qwen checkpoints with the pinned MLX reference, run `just qualify-mlx`. [Testing guide](https://hupe1980.github.io/nuvec/docs/testing/)

## API stability

Minor releases may break the pre-1.0 API. Semantic versioning covers the `nuvec` facade and `nuvec-lm`. Raising the minimum Rust version requires a minor release. [Changelog](https://github.com/hupe1980/nuvec/blob/main/CHANGELOG.md)

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Code ported from MLX and mlx-lm keeps its MIT attribution ([THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)). nuvec is not affiliated with Apple.
