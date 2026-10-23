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

- 🦀 **Pure Rust.** No C++ build, no FFI to MLX. Metal is reached through Apple's system frameworks ([`objc2-metal`](https://crates.io/crates/objc2-metal)); `unsafe` stays in the Metal runtime, nuvec-core's three kernel-dispatch calls and one CPU matrix-product call, and the other crates contain none.
- 💤 **Lazy arrays on unified memory.** Operations build a graph; nothing runs until you read a result. The CPU and GPU share memory, so there are no copies between them.
- 🔁 **Composable transforms.** `grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`, `custom_function`, `checkpoint` and `compile` nest in any order, through complex numbers too, over tuples, maps and your own structs (gradients come back with the same shape).
- 🧠 **Models as plain structs.** `#[derive(Module)]` gives MLX's parameter names; optimizers, freezing, LoRA, `save_weights` and a compiled training step (`compile_mut`) work on the model's own type.
- 🔥 **MLX's Metal kernels.** GEMM and GEMV, quantized products (1, 2, 3, 4, 5, 6 and 8 bits), fused attention, RMSNorm, RoPE, sort, scan, mixture-of-experts gather kernels, plus your own kernels from source (`fast::MetalKernel`).
- 🧪 **Tested against MLX.** Nearly a thousand cases replay results recorded from real MLX, with random layouts, gradients and every dtype.
- 🤖 **LLM runtime.** Llama, Mistral and Qwen 2/3/3.5, dense, hybrid (Gated DeltaNet) or mixture-of-experts, straight from mlx-community or Hugging Face checkpoints; speculative decoding and batched serving included.
- 💾 **Memory-mapped weights.** Checkpoints whose tensors are aligned (files nuvec writes are) are mapped and handed to the GPU without a copy; mlx-community checkpoints have unpadded headers and are read once at load, in parallel, using no more memory than MLX.
- ⚙️ **An engine for servers.** Attention plans and score-memory caps, one peak-memory metric comparable with MLX's, a memory limit that refuses instead of swapping, checkpoint footprints before loading, row-independent few-row products and documented determinism (used, for example, by the [mrun](https://github.com/hupe1980/mrun) inference server).

## 📊 Performance

nuvec-lm against mlx-lm 0.32.2 with the same token ids, measured on 2026-10-23 with nuvec 0.4.0 on an Apple M4 Pro (48 GB). Decoding is greedy, and each cell is the median of four runs. Ratios above 1 favour nuvec.

| Model | Kind | Decode (25 / 961 / 3,841-token prompt) | Prefill (961 tokens) | Time to first token (2,048 tokens) |
|---|---|---:|---:|---:|
| Qwen3-0.6B 4-bit | dense | 1.10× / 1.09× / 1.06× | 1.04× | 1.03× |
| Llama-3.2-3B-Instruct 4-bit | dense | 1.00× / 1.00× / 1.00× | 1.01× | 1.00× |
| Llama-3.1-8B-Instruct 4-bit | dense | 1.01× / 1.01× / 1.00× | 1.01× | 1.00× |
| Qwen3-8B 4-bit | dense | 1.00× / 1.01× / 1.01× | 1.01× | 1.00× |
| Qwen3.5-9B 4-bit | hybrid | 0.99× / 0.99× / 0.99× | 1.00× | 1.00× |
| Qwen3.8-27B 4-bit | hybrid | 1.02× / 1.01× / 1.02× | 1.01× | 1.01× |
| Qwen3-30B-A3B 4-bit | MoE | 1.03× / 1.03× / 1.03× | 0.99× | 0.99× |
| Qwen3.6-35B-A3B 4-bit | hybrid MoE | 1.01× / 1.01× / 1.01× | 0.98× | 1.00× |

The geometric mean is 1.02× for decode and 1.00× for prefill (the 25-token rows are not counted). For Qwen3.6-35B-A3B the time to first token is 0.98× at 32K and 64K tokens. At 261,952 tokens nuvec prefills in 20.5 minutes, and mlx-lm runs out of GPU memory on the same 48 GB machine.

Beyond single requests:
- **Warm load to the first token:** 1.13–2.77× faster (geometric mean 1.63×).
- **Peak memory:** 1.02× (slightly below mlx-lm's).
- **Serving several requests at once:** 1.02× / 0.99× / 0.99× aggregate throughput at 1 / 4 / 8 requests. The lowest cells are 0.97×.

The full record (JSON) and the method are on the [benchmarks](https://hupe1980.github.io/nuvec/docs/benchmarks/) page. Speculative decoding with prompt-lookup drafts more than doubles decode speed on text that repeats its input.

## 🚀 Quick start

```sh
cargo add nuvec            # arrays, transforms, Metal kernels
cargo add nuvec-nn         # layers and optimizers (optional)
cargo add nuvec-lm         # the LLM runtime (optional)
```

**Arrays are lazy, and gradients are arrays too:**

```rust
use nuvec::prelude::*;
use nuvec::grad;

fn main() -> Result<()> {
    let x = array![1.0, 2.0, 3.0];
    let y = (&x * 2.0 + 1.0) * &x;              // builds a graph, computes nothing
    println!("{:?}", y.to_vec::<f32>()?);       // [3.0, 10.0, 21.0], on the GPU

    let f = |x: &Array| x.multiply(x).sum(..);  // sum(x²)
    println!("{:?}", grad(f)(&x)?.to_vec::<f32>()?); // [2.0, 4.0, 6.0]
    Ok(())
}
```

**Create, index and chain; errors surface where you read results:**

```rust
use nuvec::prelude::*;
use nuvec::ops as mx;

fn main() -> Result<()> {
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; // float32, shape [2, 3]
    let b = Array::from(vec![10.0f32, 20.0, 30.0]);   // from Rust values
    let z = mx::zeros([2, 3], DType::F32);            // free functions live in `ops`

    let cols = a.i((.., 1..3));                       // MLX's a[:, 1:3], a view
    let z = z.at(0).set(&b);                          // MLX's z[0] = b, a new array
    assert_eq!(cols.sum(..).item::<f32>()?, 16.0);
    assert_eq!(z.sum_keepdims(-1).shape(), [2, 1]);

    let bad = a.matmul(&a).exp();                     // [2, 3] @ [2, 3]: a failed array
    let err = bad.check().unwrap_err();               // no panic, no `Result` per op
    assert_eq!(err.op_name(), Some("matmul"));        // with your file and line
    Ok(())
}
```

Operations are methods that chain and return arrays, never `Result`. An invalid call gives a *failed array* that remembers the operation, where you called it and why; everything computed from it fails the same way, and the error surfaces at a boundary: `check()?` (no evaluation), `eval()?`, `item()?`, `to_vec()?`, I/O and transforms. Arrays come from `array![...]`, `From` (Rust numbers, arrays, `Vec`s and iterators) and `Array::from_slice(&data, shape)`; MLX's free functions (`zeros`, `arange`, `concatenate`, …) are in `nuvec::ops`.

**Train a model: a struct with `#[derive(Module)]`, a compiled step over `(model, optimizer)`:**

```rust
use nuvec::{Array, Result, array, compile_mut, random, value_and_grad};
use nuvec_nn::optim::{Optimizer, Sgd};
use nuvec_nn::{Linear, Module, losses, relu};

#[derive(Clone, Module)]
struct Mlp { hidden: Linear, out: Linear }        // parameters: hidden.weight, hidden.bias, …

fn main() -> Result<()> {
    random::seed(0);
    let mut state = (Mlp { hidden: Linear::new(2, 16), out: Linear::new(16, 2) }, Sgd::new(0.5));
    state.1.init(&state.0);
    let step = compile_mut(|(model, opt): &mut (Mlp, Sgd), (x, y): &(Array, Array)| {
        let loss = |m: &Mlp| losses::cross_entropy(&m.out.forward(&relu(&m.hidden.forward(x))), y, losses::Reduction::Mean);
        let (value, grads) = value_and_grad(loss)(model)?;  // grads: Mlp
        opt.update(model, &grads)?;
        Ok(value)
    });
    let batch = (array![[1.0, -1.0], [-1.0, 1.0]], array![0u32, 1]);
    let first = step(&mut state, &batch)?.item::<f32>()?;
    for _ in 0..50 {
        let _ = step(&mut state, &batch)?;               // traced once, then replayed
    }
    assert!(step(&mut state, &batch)?.item::<f32>()? < first);
    println!("{}", state.0.display());                   // Mlp (82 params, 82 trainable) …
    nuvec::save(std::env::temp_dir().join("ckpt.safetensors"), &state)?; // model and optimizer
    Ok(())
}
```

**Generate text with an mlx-community checkpoint:**

```rust,no_run
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
| [`nuvec-compile`](crates/nuvec-compile) | `compile` and `compile_mut`: tracing, simplification, element-wise kernel fusion |
| [`nuvec-macros`](crates/nuvec-macros) | `#[derive(Tree)]`, `#[derive(Module)]`, `#[derive(Static)]` (re-exported by `nuvec` and `nuvec-nn`) |
| [`nuvec-gpu`](crates/nuvec-gpu) | Metal runtime: pipelines, buffers, hazard-tracked command encoding, allocator, wired memory, profiler |
| [`nuvec-kernels`](crates/nuvec-kernels) | Metal kernel sources and the precompiled `nuvec.metallib` |
| [`nuvec-cpu`](crates/nuvec-cpu) | Pure-Rust CPU kernels |
| [`nuvec-io`](crates/nuvec-io) | safetensors, npy and IDX; memory-mapped loading |
| [`nuvec-nn`](crates/nuvec-nn) | Modules (`#[derive(Module)]`), layers (Linear, quantized, LoRA, norms, RoPE, dropout), losses, optimizers (SGD, Adam, AdamW, Lion, Muon, parameter groups) |
| [`nuvec-lm`](crates/nuvec-lm) | LLM runtime: models, tokenizer, KV cache, sampling, speculative decoding, LoRA adapters |

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
just ci        # CI's gates: fmt, clippy, tests, docs, Miri, cargo-deny, kernel and hygiene checks
```

Without the Metal toolchain the build still succeeds: the shader sources are embedded and compiled at runtime (`NUVEC_METALLIB=off` selects this mode explicitly).

### 🧪 Against MLX

Python is not needed to build or test: the fixtures recorded from MLX are checked in. With [uv](https://docs.astral.sh/uv/) installed you can re-record them and compare speed on your own machine:

| Command | Compares |
|---|---|
| `just bench-parity` | nuvec-lm vs mlx-lm over a roster of cached models: decode, time to first token, peak memory, load, concurrency (a JSON record; the table above) |
| `just bench-gemm` / `bench-quant` / `bench-fast` | matrix products, quantized products, fused kernels |
| `just bench-compile`, `just bench-load`, `just mnist` | compiled functions, checkpoint loading, MNIST training |
| `just fixtures`, `lm-fixtures`, `nn-fixtures`, `io-fixtures` | re-record the reference results |

`NUVEC_PROFILE=1` prints the GPU time per kernel of any run (see [profiling](https://hupe1980.github.io/nuvec/docs/gpu/#profiling)).

## 🧭 Stability

nuvec is 0.x: a minor release may break the API, and the [changelog](CHANGELOG.md) lists every break with a migration table. Semver covers every item reachable through `nuvec::` paths and `nuvec-lm`; the pinned items are listed in [`crates/nuvec/tests/api_surface.rs`](crates/nuvec/tests/api_surface.rs). The sub-crates (`nuvec-core`, `nuvec-gpu`, …) make no promise when depended on directly. The minimum supported Rust version is 1.99 (`rust-version` in `Cargo.toml`); raising it is a minor release.

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Code ported from MLX and mlx-lm keeps its MIT attribution ([THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)). nuvec is not affiliated with Apple.
