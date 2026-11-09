<div align="center">

# ⚡ nuvec

**MLX for Rust, written in Rust.** Lazy n-dimensional arrays (tensors), composable transforms and Metal kernels for Apple Silicon, with no bindings to MLX or any C/C++ library.

[![CI](https://github.com/hupe1980/nuvec/actions/workflows/ci.yml/badge.svg)](https://github.com/hupe1980/nuvec/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-hupe1980.github.io%2Fnuvec-blue)](https://hupe1980.github.io/nuvec)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)](#-license)
![Rust 1.99+](https://img.shields.io/badge/rust-1.99%2B-orange)
![Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon%20%C2%B7%20Metal-black)

[Documentation](https://hupe1980.github.io/nuvec/docs/) · [Getting started](https://hupe1980.github.io/nuvec/docs/getting-started/) · [Benchmarks](https://hupe1980.github.io/nuvec/docs/benchmarks/)

</div>

---

nuvec brings [MLX](https://github.com/ml-explore/mlx)'s programming model to Rust: arrays that are evaluated lazily on unified memory, `grad`/`vjp`/`jvp`/`vmap`/`compile` that compose freely, and the Metal kernels that make MLX fast, ported and tested against MLX itself. On top sits an LLM runtime that loads mlx-community checkpoints and generates as fast as mlx-lm.

> 🚧 **Pre-alpha.** The API can change between releases.

## ✨ Highlights

- 🦀 **Pure Rust.** No C++ build, no FFI to MLX. Metal goes through [`objc2-metal`](https://crates.io/crates/objc2-metal). `unsafe` is confined to the GPU runtime, the CPU matrix product, kernel dispatches and buffer views in nuvec-core, and system calls in nuvec-distributed (sockets, RDMA).
- 💤 **Lazy arrays on unified memory.** Operations build a graph that runs when you read a result. CPU and GPU share memory, with no copies between them.
- 🔁 **Composable transforms.** `grad`, `value_and_grad`, `vjp`, `jvp`, `vmap`, `custom_function`, `checkpoint` and `compile` (shaped or shapeless) nest in any order, over complex numbers, tuples, maps and your own structs.
- 🧠 **Models as plain structs.** `#[derive(Module)]` gives MLX's parameter names; optimizers, freezing, LoRA, `save_weights` and a compiled training step (`compile_mut`) work on the model's own type.
- 🔥 **MLX's Metal kernels.** GEMM and GEMV, quantized products (affine 1, 2, 3, 4, 5, 6 and 8 bits; MXFP4, MXFP8, NVFP4), fused attention, RMSNorm, RoPE, sort, scan, mixture-of-experts gather kernels, and your own kernels from source (`fast::MetalKernel`). Metal 4 TensorOps kernels for M5 Neural Accelerators are opt-in (`NUVEC_NAX=on`).
- 🌐 **Several Macs.** MLX's collectives (`all_sum`, `all_gather`, `sum_scatter`, `send`/`recv`, …) as lazy, differentiable array operations over MLX's TCP ring (authenticated and encrypted with a shared key) or, experimentally, JACCL over Thunderbolt 5 RDMA, with pipeline-parallel models (a model larger than one Mac, one stage per Mac) and tensor-parallel models (every layer split across Macs).
- 🧮 **MLX's operations.** Every array function of `mlx.core` and `mx.linalg`, with MLX's dtypes and gradients (also for the decompositions, which MLX cannot differentiate): element-wise math with MLX's kernels (`erf`, `expm1`, `rsqrt`, rounding, bitwise and logical operations, …), `pad`, `einsum`, `unique`, `median` and the rest (`scripts/op_coverage.py`).
- 🖼️ **Convolutions.** 1-D, 2-D and 3-D, transposed and general, on MLX's Metal kernels (implicit GEMM, Winograd with float32 tiles, depthwise) at 0.97–1.07× MLX's GPU time, differentiable; `nuvec-nn` has the `Conv*`, `ConvTranspose*`, `MaxPool*` and `AvgPool*` layers.
- 🧪 **Tested against MLX.** About 3,000 cases replay results recorded from MLX, with random layouts, gradients and every dtype.
- 🤖 **LLM runtime.** Llama, Mistral, Qwen 2/3/3.5, gpt-oss and Gemma 4 (dense, hybrid Gated DeltaNet, mixture-of-experts) from mlx-community or Hugging Face checkpoints. Speculative decoding (MTP heads, prompt lookup, SuffixDecoding) and batched serving.
- 💾 **Memory-mapped weights.** Aligned checkpoints (such as those nuvec writes) go to the GPU without a copy. Unaligned ones (mlx-community's) are read once, in parallel, with no more memory than MLX.
- ⚙️ **Built for servers.** Attention plans and score-memory caps, a memory limit that refuses instead of swapping, checkpoint footprints before loading, batch-invariant serving and documented determinism. Used by the [mrun](https://github.com/hupe1980/mrun) inference server.

## 📊 Performance

nuvec-lm against mlx-lm 0.32.2 with the same token ids on an Apple M4 Pro (48 GB), measured on 2026-10-31 with nuvec 0.5.0 (`just bench-parity --tier full`). Decoding is greedy; ratios above 1 favour nuvec.

| Model | Kind | Decode (25 / 961 / 3,841-token prompt) | Time to first token (2K / 32K tokens) | 4 requests |
|---|---|---:|---:|---:|
| Qwen3-0.6B 4-bit | dense | 1.10× / 1.10× / 1.06× | 1.01× / 1.01× | 0.98× |
| Llama 3.2 3B 4-bit | dense | 0.99× / 1.00× / 1.00× | 1.00× / 1.01× | 1.00× |
| Llama 3.1 8B 4-bit | dense | 1.01× / 1.00× / 1.00× | 1.00× / 1.00× | 1.01× |
| Qwen3-8B 4-bit | dense | 1.00× / 1.01× / 1.01× | 1.00× / 1.01× | 1.01× |
| Qwen3.5-9B 4-bit | hybrid | 1.00× / 1.00× / 1.00× | 1.00× / 0.99× | 1.00× |
| Qwen3.8-27B 4-bit | hybrid | 1.00× / 1.01× / 1.05× | 1.01× / 1.00× | 1.00× |
| Qwen3-30B-A3B 4-bit | MoE | 1.00× / 1.00× / 1.00× | 0.99× / 1.00× | 0.98× |
| Qwen3.6-35B-A3B 4-bit | hybrid MoE | 0.99× / 1.00× / 1.00× | 1.01× / 0.98× | 0.98× |
| gpt-oss-20b MXFP4 | MoE, sliding window | 0.99× / 0.99× / 0.98× | 1.04× / 1.03× | 1.00× |
| gpt-oss-20b 8-bit | MoE, sliding window | 1.01× / 1.01× / 1.00× | 1.02× / 1.02× | 1.08× |
| Gemma 4 E4B 4-bit | dense, sliding window | 0.98× / 1.00× / 1.00× | 1.07× / 1.07× | 1.02× |
| Gemma 4 26B-A4B 4-bit | MoE, sliding window | 1.01× / 1.01× / 1.01× | 1.06× / 1.08× | 0.99× |

- **Geometric means:** decode 1.01×, time to first token 1.02× at 2K and 32K tokens, peak memory 1.02× (nuvec uses slightly less), warm load to the first token 1.80×, 1 / 4 / 8 concurrent requests 1.01× / 1.00× / 1.00×.
- **Longest contexts:** Qwen3.6-35B-A3B's first token at 64K tokens is 0.98×. At 261,952 tokens nuvec prefills it in 20.4 minutes; mlx-lm runs out of GPU memory on the same machine.
- **Speculative decoding** with the model's multi-token-prediction head: Qwen3.8-27B 1.66× and Qwen3.5-9B 1.36× faster than plain decoding on natural text, with identical output.

The full record (JSON) and method are on the [benchmarks](https://hupe1980.github.io/nuvec/docs/benchmarks/) page.

## 🚀 Quick start

```sh
cargo add nuvec            # arrays, transforms, Metal kernels
cargo add nuvec-nn         # layers and optimizers (optional)
cargo add nuvec-lm         # the LLM runtime (optional)
```

Build dependencies optimised even in debug builds; otherwise CPU kernels and tests that run them are tens of times slower:

```toml
[profile.dev.package."*"]
opt-level = 3
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

Operations return arrays, never `Result`. An invalid call gives a *failed array* that records the operation, call site and reason; anything computed from it fails too. The error surfaces at `check()?`, `eval()?`, `item()?`, `to_vec()?`, I/O and transforms.

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

From the command line (after `hf download mlx-community/Qwen3-8B-4bit`):

```sh
cargo run --release -p nuvec-lm --example generate -- \
    --model mlx-community/Qwen3-8B-4bit --prompt "Write a haiku about Rust"
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
| [`nuvec-cpu`](crates/nuvec-cpu) | Pure-Rust CPU kernels and `linalg` decompositions (`faer`; Accelerate's LAPACK with the feature `accelerate`) |
| [`nuvec-io`](crates/nuvec-io) | safetensors, npy and IDX; memory-mapped loading |
| [`nuvec-distributed`](crates/nuvec-distributed) | MLX's TCP ring and JACCL (Thunderbolt 5 RDMA, experimental) for `nuvec::distributed` |
| [`nuvec-nn`](crates/nuvec-nn) | Modules (`#[derive(Module)]`), layers (Linear, quantized, LoRA and DoRA, convolutions and pooling, norms, RoPE, dropout, tensor-parallel sharded linear), MLX's activations, losses (also chunked over an output head), optimizers (SGD, Adam, AdamW, Lion, Muon, Adafactor, parameter groups) |
| [`nuvec-lm`](crates/nuvec-lm) | LLM runtime: models, tokenizer, chat templates, KV cache, sampling, speculative decoding, fine-tuning (LoRA, DoRA, full; text, chat and completion data) and adapters |

## 🖥️ Platforms

| | |
|---|---|
| 🍎 **macOS 15+ on Apple Silicon** | The primary target, with the full Metal backend |
| 🐧 **Linux** | The core and CPU crates build and run; the GPU crates compile as stubs (nuvec-gpu's `cuda` feature adds an experimental CUDA runtime that arrays do not use) |

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

Without the Metal toolchain, kernels are compiled from embedded sources at runtime (`NUVEC_METALLIB=off` forces this).

### 🧪 Against MLX

Building and testing need no Python: the MLX fixtures are checked in. With [uv](https://docs.astral.sh/uv/) you can re-record them and benchmark against MLX:

| Command | Compares |
|---|---|
| `just bench-parity` | nuvec-lm vs mlx-lm and the previous release, one model per architecture (about 45 minutes); `--tier full` for all models and long contexts (the table above) |
| `just bench-gemm` / `bench-quant` / `bench-fast` / `bench-conv` / `bench-special` | matrix products, quantized products, fused kernels, convolutions, Hadamard, einsum and the special products |
| `just bench-compile`, `just bench-load`, `just mnist` | compiled functions, checkpoint loading, MNIST training |
| `just fixtures`, `lm-fixtures`, `nn-fixtures`, `io-fixtures` | re-record the reference results |

`NUVEC_PROFILE=1` prints the GPU time per kernel of any run (see [profiling](https://hupe1980.github.io/nuvec/docs/gpu/#profiling)).

## 🧭 Stability

nuvec is 0.x: a minor release may break the API; the [changelog](CHANGELOG.md) lists every break. Semver covers items reachable through `nuvec::` paths and `nuvec-lm` (pinned in [`crates/nuvec/tests/api_surface.rs`](crates/nuvec/tests/api_surface.rs)); the sub-crates (`nuvec-core`, `nuvec-gpu`, …) make no promise when used directly. MSRV is Rust 1.99; raising it is a minor release.

## 📄 License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this work, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Code ported from MLX and mlx-lm keeps its MIT attribution ([THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)). nuvec is not affiliated with Apple.
