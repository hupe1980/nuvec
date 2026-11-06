+++
title = "FAQ"
description = "Frequently asked questions about nuvec: how it relates to MLX, mlx-rs, Candle and Burn, why arrays and not tensors, hardware support, Python, stability and licensing."
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

## Why `Array` and not `Tensor`?

nuvec follows MLX, JAX and NumPy, where gradients come from transforming functions (`grad(f)`), not from `.backward()` on a value. The type is the same thing PyTorch calls a tensor; [Arrays and tensors](@/docs/arrays.md#arrays-and-tensors) maps PyTorch habits to nuvec.

## How fast is it?

On par with MLX. On an M4 Pro (nuvec 0.5.0 against mlx-lm 0.32.2, 12 models), nuvec-lm decodes at 0.98–1.10× mlx-lm's speed on dense, hybrid, mixture-of-experts, gpt-oss and Gemma 4 models (geometric mean 1.01×); time to first token is 1.02× at 2,048 and 32,768 tokens (0.98× at 64K), serving four or eight requests at once is 1.00×, and warm loads are 1.80× faster. See [benchmarks](@/docs/benchmarks.md).

## Which Macs are supported?

Apple Silicon (M1 or later) with macOS 15 or newer. Wired memory needs a Metal 3 GPU (all Apple Silicon Macs on macOS 15). The core and CPU crates also build and run on Linux.

## Can I use it from Python?

No; nuvec is a Rust library. The `nuvec` name on PyPI is reserved.

## Can I train models?

Yes:

- autodiff covers every operation; models are structs with `#[derive(Module)]` whose gradients have the model's type;
- `nuvec-nn` has layers, dropout, LoRA and optimizers (with parameter groups); quantized layers support QLoRA-style fine-tuning;
- nuvec-lm trains LoRA adapters on loaded models as `mlx_lm.lora` (the `lora` example) and reads and writes mlx-lm's adapter directories;
- a whole training step compiles with `compile_mut`.

The repository trains MLX's MNIST example and matches MLX's accuracy.

## Is the API stable?

No. nuvec is pre-alpha (0.x): minor versions may break the API; patch versions only add to it. Semantic versioning covers every item reachable through `nuvec::` paths, and `nuvec-lm`; the other crates make no promise when depended on directly, so depend on `nuvec`. Changes are listed in the [changelog](https://github.com/hupe1980/nuvec/blob/main/CHANGELOG.md).

## What is the license?

MIT or Apache-2.0, at your option. Ported MLX kernels keep their MIT notices. nuvec is not affiliated with Apple.
