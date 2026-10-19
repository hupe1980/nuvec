<div align="center">

# ⚡ nuvec

**MLX for Rust, written in Rust.** Lazy arrays, composable transforms and Metal kernels for Apple Silicon, with no bindings to MLX or any C/C++ library.

[![CI](https://github.com/hupe1980/nuvec/actions/workflows/ci.yml/badge.svg)](https://github.com/hupe1980/nuvec/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-hupe1980.github.io%2Fnuvec-blue)](https://hupe1980.github.io/nuvec)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)](#-license)
![Rust 1.99+](https://img.shields.io/badge/rust-1.99%2B-orange)
![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon%20%C2%B7%20Metal-black)

[Documentation](https://hupe1980.github.io/nuvec/docs/) · [Getting started](https://hupe1980.github.io/nuvec/docs/getting-started/) · [Benchmarks](https://hupe1980.github.io/nuvec/docs/benchmarks/)

</div>

---

nuvec brings [MLX](https://github.com/ml-explore/mlx)'s programming model to Rust: arrays that are evaluated lazily on unified memory, `grad`/`vjp`/`jvp`/`vmap`/`compile` that compose freely, and the Metal kernels that make MLX fast, ported and tested against MLX itself. On top sits an LLM runtime that loads mlx-community checkpoints and generates as fast as mlx-lm.

> 🚧 **Pre-alpha.** The API can still change between releases. Everything listed below works today and is covered by tests.

## ✨ Highlights

- 🦀 **Pure Rust.** No C++ build, no FFI to MLX. Metal is reached through Apple's system frameworks ([`objc2-metal`](https://crates.io/crates/objc2-metal)); `unsafe` stays in the Metal runtime (and one CPU matrix-product call), and every other crate forbids it.
- 💤 **Lazy arrays on unified memory.** Operations build a graph; nothing runs until you read a result. The CPU and GPU share memory, so there are no copies between them.
- 🔁 **Composable transforms.** `grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`, `custom_function`, `checkpoint` and `compile` nest in any order, through complex numbers too.
- 🔥 **MLX's Metal kernels.** GEMM and GEMV, quantized products (1–8 bit), fused attention, RMSNorm, RoPE, sort, scan, mixture-of-experts gather kernels, plus your own kernels from source (`fast::MetalKernel`).
- 🧪 **Tested against MLX.** Nearly a thousand cases replay results recorded from real MLX, with random layouts, gradients and every dtype.
- 🤖 **LLM runtime.** Llama, Mistral and Qwen 2/3/3.5, dense, hybrid (Gated DeltaNet) or mixture-of-experts, straight from mlx-community or Hugging Face checkpoints; speculative decoding included.
- 💾 **Zero-copy weights.** safetensors and npy files are memory-mapped and handed to the GPU without a copy.

## 📊 Performance

Same machine, same day, same checkpoints: `nuvec-lm` against `mlx-lm` 0.32.2 (Apple M4 Pro, greedy decoding, medians of four runs; ratios above 1 favour nuvec).

| Model | Kind | Prefill (961 tokens) | Decode |
|---|---|---:|---:|
| Qwen3-0.6B 4-bit | dense | 1.03× | 1.03–1.04× |
| Llama-3.2-3B 4-bit | dense | 1.01× | 0.98–0.99× |
| Qwen3-8B 4-bit | dense | 1.00× | 0.97–1.00× |
| Qwen3.5-9B 4-bit | hybrid | 0.99× | 0.97–1.00× |
| Qwen3-30B-A3B 4-bit | MoE | 0.99× | 1.02× |
| Qwen3.6-35B-A3B 4-bit | hybrid MoE | 0.96× | 1.00–1.01× |

Speculative decoding with prompt-lookup drafts doubles decode speed on text that repeats its input (Qwen3-8B: 53 → 106 tok/s), at about the same speed otherwise. Full tables and methodology: [benchmarks](https://hupe1980.github.io/nuvec/docs/benchmarks/).

## 🚀 Quick start

```sh
cargo add nuvec            # arrays, transforms, Metal kernels
cargo add nuvec-lm         # the LLM runtime (optional)
```

**Arrays are lazy, and gradients are arrays too:**

```rust
use nuvec::{Array, Result, grad, multiply, sum};

fn main() -> Result<()> {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
    let y = (&x * 2.0 + 1.0) * &x;              // builds a graph, computes nothing
    println!("{:?}", y.to_vec::<f32>()?);       // [3.0, 10.0, 21.0], on the GPU

    let f = |x: &Array| sum(&multiply(x, x)?, &[], false); // sum(x²)
    println!("{:?}", grad(f)(&x)?.to_vec::<f32>()?);       // [2.0, 4.0, 6.0]
    Ok(())
}
```

**Generate text with an mlx-community checkpoint:**

```rust
use nuvec_lm::{GenerateOptions, generate, load};

fn main() -> nuvec::Result<()> {
    let (model, tokenizer) = load("mlx-community/Qwen3-8B-4bit")?; // from the HF cache
    let prompt = tokenizer.encode("The capital of France is");
    let options = GenerateOptions { max_tokens: 32, ..Default::default() };
    let ids: Vec<u32> = generate(&model, &prompt, options)?.collect::<nuvec::Result<_>>()?;
    println!("{}", tokenizer.decode(&ids));
    Ok(())
}
```

Or from the command line (the model must be in the local Hugging Face cache, e.g. `hf download mlx-community/Qwen3-8B-4bit`):

```sh
cargo run --release -p nuvec-lm --example generate -- \
    --model mlx-community/Qwen3-8B-4bit --prompt "Write a haiku about Rust" --chat
```

## 📦 Crates

| Crate | What it does |
|---|---|
| [`nuvec`](crates/nuvec) | The facade: arrays, transforms, `fast` kernels, random, I/O and the GPU runtime in one import |
| [`nuvec-core`](crates/nuvec-core) | Lazy graph, primitives, dtypes, evaluation, derivative and batching rules |
| [`nuvec-compile`](crates/nuvec-compile) | `compile`: tracing, simplification, element-wise kernel fusion |
| [`nuvec-gpu`](crates/nuvec-gpu) | Metal runtime: pipelines, buffers, hazard-tracked command encoding, allocator, wired memory, profiler |
| [`nuvec-kernels`](crates/nuvec-kernels) | Metal kernel sources and the precompiled `nuvec.metallib` |
| [`nuvec-cpu`](crates/nuvec-cpu) | Pure-Rust CPU kernels |
| [`nuvec-io`](crates/nuvec-io) | safetensors, npy and IDX; memory-mapped loading |
| [`nuvec-nn`](crates/nuvec-nn) | Modules, layers (Linear, quantized, LoRA, norms, RoPE), losses, optimizers (SGD, Adam, AdamW, Lion, Muon) |
| [`nuvec-lm`](crates/nuvec-lm) | LLM runtime: models, tokenizer, KV cache, sampling, speculative decoding |
| [`nuvec-distributed`](crates/nuvec-distributed) | Reserved for distributed collectives (no functional API yet) |

## 🖥️ Platforms

| | |
|---|---|
| 🍎 **macOS 15+ on Apple Silicon** | The primary target, with the full Metal backend |
| 🐧 **Linux** | The core and CPU crates build and run; the GPU crates compile as stubs |

## 🛠️ Building from source

| Tool | Install |
|---|---|
| Rust (pinned in `rust-toolchain.toml`) | [rustup](https://rustup.rs) installs it automatically |
| Metal toolchain | `xcodebuild -downloadComponent MetalToolchain` |
| [just](https://just.systems) | `brew install just` |
| [cargo-deny](https://github.com/EmbarkStudios/cargo-deny), Miri, jq | `cargo install cargo-deny --locked`, `rustup +nightly component add miri`, `brew install jq` |

```sh
git clone https://github.com/hupe1980/nuvec && cd nuvec
just ci        # everything CI runs: fmt, clippy, tests, docs, Miri, cargo-deny, kernel checks
```

Without the Metal toolchain the build still succeeds: the shader sources are embedded and compiled at runtime (`NUVEC_METALLIB=off` selects this mode explicitly).

### 🧪 Against MLX

Python is not needed to build or test: the fixtures recorded from MLX are checked in. With [uv](https://docs.astral.sh/uv/) installed you can re-record them and compare speed on your own machine:

| Command | Compares |
|---|---|
| `just bench-parity` | nuvec-lm vs mlx-lm over a roster of cached models (the table above) |
| `just bench-gemm` / `bench-quant` / `bench-fast` | matrix products, quantized products, fused kernels |
| `just bench-compile`, `just bench-load`, `just mnist` | compiled functions, checkpoint loading, MNIST training |
| `just fixtures`, `lm-fixtures`, `nn-fixtures`, `io-fixtures` | re-record the reference results |

`NUVEC_PROFILE=1` prints the GPU time per kernel of any run (see [profiling](https://hupe1980.github.io/nuvec/docs/gpu/#profiling)).

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Ported MLX kernels keep their MIT attribution. nuvec is not affiliated with Apple.
