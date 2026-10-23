+++
title = "compile and kernel fusion"
description = "nuvec's compile traces a function once per input signature, simplifies the graph and fuses element-wise chains into generated Metal kernels."
weight = 40
[extra]
group = "Core"
+++

`compile` works like MLX's `mx.compile`. The first call with a given input structure and signature (the tree's paths and kinds, its static fields, shapes and dtypes) traces the function on placeholders and simplifies the graph:

- identity views and conversions are removed;
- equal scalar constants are merged;
- equal operations on equal inputs are computed once (common-subexpression elimination).

On the GPU, chains of element-wise operations are then **fused** into one generated Metal kernel, with several outputs when needed. Later calls with the same signature replay the simplified graph without running your function again.

```rust
use nuvec::prelude::*;
use nuvec::compile;

# fn main() -> Result<()> {
# let (x, x2) = (array![1.0, 2.0], array![3.0, 4.0]);
let f = compile(|x: &Array| x.multiply(x) + 1.0); // one kernel
let y = f(&x)?;   // traced on first use
let z = f(&x2)?;  // same signature: replayed
# assert_eq!(y.to_vec::<f32>()?, [2.0, 5.0]);
# assert_eq!(z.to_vec::<f32>()?, [10.0, 17.0]);
# Ok(())
# }
```

The function takes and returns a tree (an array, a tuple, a struct), directly or as a `Result`, and may borrow locals. Annotate its parameter (`|x: &Array|`). `compile_flat` is the positional form over `&[Array]`.

## Compiled training steps

`compile_mut` compiles a stateful step, as MLX's `mx.compile(step, inputs=state, outputs=state)`: the state (typically `(model, optimizer)`) is a tree whose leaves are inputs and outputs of the trace.

```rust
use nuvec::{Array, Result, array, compile_mut, value_and_grad};

# fn main() -> Result<()> {
let step = compile_mut(|w: &mut Array, (x, y): &(Array, Array)| {
    let loss = |w: &Array| { let r = w.multiply(x).subtract(y); r.multiply(&r).sum(..) };
    let (l, g) = value_and_grad(loss)(w)?;
    *w = w.subtract(g.multiply(0.1));
    Ok(l)
});
let mut w = array![0.0];
for _ in 0..50 {
    let _loss = step(&mut w, &(array![1.0], array![2.0]))?;
}
# assert!((w.item::<f32>()? - 2.0).abs() < 1e-3);
# Ok(())
# }
```

With nuvec-nn, the step over `(model, optimizer)` is the whole training step (see the neural-network page). The step traces once: the optimizer's step counter and learning rate are arrays in its state, and random draws inside (dropout) advance the thread's key sequence as an uncompiled call would. Call `opt.init(&model)` before the loop, so the optimizer's state exists before the first trace.

## Static fields and retracing

A tree's structure includes its static fields: `#[tree(skip)]` fields of a `Static` type (a layer's `eps`, an optimizer's betas), the train/eval mode, frozen flags and the active enum variants. Changing any of them retraces once, instead of replaying a stale constant. A step that changes its state's structure on every call (a host counter in a `#[tree(skip)]` field) is an error; make such values arrays, or mark the field `#[tree(skip, static_unchecked)]`, whose value is then baked into the trace.

## When it helps

Element-wise chains are bandwidth-bound: unfused, each step reads and writes the whole array. Fused, the chain reads its inputs once and writes its outputs once. Activations (`silu(gate) * up`, GELU), normalization epilogues and optimizer updates are typical wins. nuvec-lm compiles exactly the functions mlx-lm compiles, such as SwiGLU and the Gated DeltaNet gates.

## Limits

- The function must not evaluate arrays while it is traced.
- A new input signature means a new trace (and new kernels); keep shapes stable in hot loops. Each compiled function keeps its 64 most recently used traces (`compile::Options::default().cache_capacity(n)` changes that).
- Arrays a pure `compile`d function captures, and random numbers drawn inside it, are constants frozen at the trace; `compile_mut` threads the random key instead.
- Products, reductions and attention are not fused; they run as their own kernels.
