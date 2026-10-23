+++
title = "FAQ"
description = "Frequently asked questions about nuvec: how it relates to MLX, mlx-rs, Candle and Burn, hardware support, Python, stability and licensing."
weight = 150
[extra]
group = "Project"
+++

## Is nuvec a binding to MLX?

No. nuvec is written in Rust from the ground up and links no MLX or other C/C++ library. It reaches Metal through Apple's system frameworks and compiles its own kernels, many of them ported from MLX with their attribution kept.

## How does it compare with mlx-rs?

[mlx-rs](https://github.com/oxiglade/mlx-rs) wraps MLX's C API, so it inherits MLX's behaviour and needs MLX's C++ build. nuvec reimplements the stack in Rust: no C++ toolchain, Rust errors and ownership throughout, and the freedom to change how evaluation, memory and kernels work.

## And with Candle or Burn?

[Candle](https://github.com/huggingface/candle) is an eager tensor library; [Burn](https://github.com/tracel-ai/burn) is a portable deep-learning framework whose kernels are generated for many backends. nuvec targets one platform — Apple Silicon — and MLX's model: lazy evaluation, composable `grad`/`vmap`/`compile`, and MLX's hand-tuned Metal kernels.

## How fast is it?

On par with MLX for one request: on an M4 Pro (nuvec 0.4.0), nuvec-lm decodes at 0.99–1.10× the speed of mlx-lm across eight models (geometric mean 1.02×), and its time to first token is 1.01× at 2,048 tokens. Serving several requests at once is on par (0.99× at 4 and 8 requests), and models load faster (1.63×). See [benchmarks](@/docs/benchmarks.md).

## Which Macs are supported?

Apple Silicon (M1 or later) with macOS 15 or newer. Wired memory needs a Metal 3 GPU (all Apple Silicon Macs on macOS 15). The core and CPU crates also build and run on Linux.

## Can I use it from Python?

Not today; nuvec is a Rust library. The `nuvec` name on PyPI is reserved.

## Can I train models?

Yes: autodiff covers every operation, models are structs with `#[derive(Module)]` whose gradients have the model's type, `nuvec-nn` has layers, dropout, LoRA and optimizers (with parameter groups), quantized layers support QLoRA-style fine-tuning, nuvec-lm applies LoRA to loaded models by path and reads and writes mlx-lm's adapter directories, and a whole training step compiles with `compile_mut`. The repository trains MLX's MNIST example and matches MLX's accuracy.

## Is the API stable?

Not yet. nuvec is pre-alpha (0.x): minor versions may change the API; patch versions do not. Semantic versioning covers every item reachable through `nuvec::` paths, and `nuvec-lm`; the other crates make no promise when depended on directly, so depend on `nuvec`. Changes are listed in the [changelog](https://github.com/hupe1980/nuvec/blob/main/CHANGELOG.md).

## What is the license?

MIT or Apache-2.0, at your option. Ported MLX kernels keep their MIT notices. nuvec is not affiliated with Apple.
