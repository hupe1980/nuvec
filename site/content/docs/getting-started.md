+++
title = "Getting started"
description = "Install nuvec, run your first lazy array computation on the Apple GPU, take a gradient and generate text with an mlx-community model."
weight = 10
[extra]
group = "Start"
+++

## Requirements

- A Mac with Apple Silicon (M1 or later) and **macOS 15** or newer for the GPU.
- Rust **1.99** or newer ([rustup](https://rustup.rs)).
- Optional: Apple's Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`). With it, kernels are compiled ahead of time into a Metal library; without it, their sources are embedded and compiled the first time they run.

The core and CPU crates also build and run on Linux, which is handy for development and CI; GPU code compiles there as stubs.

## Install

```sh
cargo add nuvec        # arrays, transforms, Metal kernels, I/O
cargo add nuvec-nn     # optional: layers and optimizers
cargo add nuvec-lm     # optional: the LLM runtime
```

## Your first computation

```rust
use nuvec::prelude::*;
use nuvec::grad;

fn main() -> Result<()> {
    let x = array![1.0, 2.0, 3.0];          // float32, shape [3]
    let y = (&x * 2.0 + 1.0) * &x;          // records a graph; nothing runs yet
    assert_eq!(y.to_vec::<f32>()?, [3.0, 10.0, 21.0]); // evaluated here, on the GPU

    // Gradients are functions that return lazy arrays too.
    let f = |x: &Array| x.multiply(x).sum(..); // sum(x²) over all axes
    assert_eq!(grad(f)(&x)?.to_vec::<f32>()?, [2.0, 4.0, 6.0]);
    Ok(())
}
```

Operations are methods that chain (`x.multiply(x).sum(..)`); MLX's free functions are in `nuvec::ops` (`use nuvec::ops as mx;`). They return arrays, not `Result`: a mistake gives a failed array whose error, with your file and line, surfaces at `to_vec()?`, `item()?`, `eval()?` or `check()?` (see [Arrays](@/docs/arrays.md#errors)).

Three things happened:

1. **Laziness.** `(&x * 2.0 + 1.0) * &x` built three graph nodes and returned immediately. Reading the result with `to_vec` evaluated exactly those nodes.
2. **The GPU by default.** Arrays are evaluated on the Apple GPU unless you choose the CPU (see below). CPU and GPU share memory, so reading the result copies nothing.
3. **Weak scalars.** `2.0` is a Rust float, but `x` stays `float32`: scalars adopt the array's type, as in MLX.

## Choosing a device

```rust
use nuvec::prelude::*;
use nuvec::{Device, Stream, set_default_stream};

let x = array![1.0, 2.0, 3.0];
let y = with_stream(Device::cpu(), || x.exp().sum(..)); // built for the CPU
set_default_stream(Stream::default_for(Device::cpu())); // this thread now uses the CPU
# assert!((y.item::<f32>().unwrap() - 30.192_875).abs() < 1e-4);
```

`with_stream` runs a closure on another device and restores the previous default afterwards. The default stream is per thread. Operations without a GPU kernel (for example on `float64`) run on the CPU over the same memory, so they work everywhere.

## Generate text

With a model in your local Hugging Face cache (for example `hf download mlx-community/Qwen3-8B-4bit`):

```rust,no_run
use nuvec_lm::{GenerateOptions, StreamDecoder, generate, load};

fn main() -> nuvec::Result<()> {
    let (model, tokenizer) = load("mlx-community/Qwen3-8B-4bit")?;
    let prompt = tokenizer.encode("The three laws of thermodynamics are");
    let options = GenerateOptions { max_tokens: 64, ..Default::default() };
    let mut text = StreamDecoder::default();
    for id in generate(&model, &prompt, options)? {
        print!("{}", text.push(&tokenizer, id?));
    }
    println!("{}", text.finish());
    Ok(())
}
```

Or run the bundled example, which prints mlx-lm-style statistics:

```sh
cargo run --release -p nuvec-lm --example generate -- \
    --model mlx-community/Qwen3-8B-4bit --prompt "Write a haiku about Rust" --chat
```

## Next steps

- [Arrays](@/docs/arrays.md): dtypes, promotion, broadcasting, views and indexing.
- [Transforms](@/docs/transforms.md): `grad`, `vjp`, `jvp`, `vmap` and custom rules.
- [Neural networks](@/docs/neural-networks.md): modules, layers and optimizers.
- [LLM runtime](@/docs/llm.md): models, sampling and speculative decoding.
