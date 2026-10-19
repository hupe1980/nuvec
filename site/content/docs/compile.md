+++
title = "compile and kernel fusion"
description = "nuvec's compile traces a function once per input signature, simplifies the graph and fuses element-wise chains into generated Metal kernels."
weight = 40
[extra]
group = "Core"
+++

`compile` works like MLX's `mx.compile`. The first call with a given input signature (shapes and dtypes) traces the function on placeholders and simplifies the graph:

- identity views and conversions are removed;
- equal scalar constants are merged;
- equal operations on equal inputs are computed once (common-subexpression elimination).

On the GPU, chains of element-wise operations are then **fused** into one generated Metal kernel, with several outputs when needed. Later calls with the same signature replay the simplified graph without running your function again.

```rust
use nuvec::{compile, add, multiply, Array};

let f = compile(|xs: &[Array]| Ok(vec![add(&multiply(&xs[0], &xs[0])?, 1.0)?])); // one kernel
let y = f(&[x])?;   // traced on first use
let z = f(&[x2])?;  // same signature: replayed
```

## When it helps

Element-wise chains are bandwidth-bound: unfused, each step reads and writes the whole array. Fused, the chain reads its inputs once and writes its outputs once. Activations (`silu(gate) * up`, GELU), normalization epilogues and optimizer updates are typical wins. nuvec-lm compiles exactly the functions mlx-lm compiles, such as SwiGLU and the Gated DeltaNet gates.

## Limits

- The function must not evaluate arrays while it is traced.
- A new input signature means a new trace (and new kernels); keep shapes stable in hot loops.
- Products, reductions and attention are not fused; they run as their own kernels.
