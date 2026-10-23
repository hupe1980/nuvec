+++
title = "Transforms: grad, vjp, jvp and vmap"
description = "Composable function transforms in nuvec: reverse- and forward-mode autodiff, higher derivatives, vectorizing map, custom rules and gradient checkpointing."
weight = 30
[extra]
group = "Core"
+++

Transforms take a function and return a new function. Because their results are lazy arrays, they nest freely: the gradient of a gradient, the vmap of a gradient, a compiled vmap.

They work on *trees*: an array, a tuple, a `Vec`, a map, an `Option`, or your own struct with `#[derive(Tree)]` (a model is a tree too). Gradients come back with the argument's shape: a tuple for a tuple, a struct for a struct, `grads.layers[0].weight` for a model.

## Annotate closure parameters

Write the type of a transform closure's parameter: `|x: &Array|`, `|p: &(Array, Array)|`, `|m: &Mlp|`. The transforms are generic over the tree type, so Rust cannot infer it from the call: an unannotated `grad(|x| x.sum(..))` fails with E0282 ("type annotations needed for `&_`"). Closures may return the value directly or a `Result` (with `?` inside).

## Gradients

```rust
use nuvec::prelude::*;
use nuvec::grad;

# fn main() -> Result<()> {
let f = |x: &Array| x.multiply(x).multiply(x).sum(..);      // sum(x³)
let x = array![1.0, 2.0];
let d1 = grad(f)(&x)?;                                       // 3x² = [3, 12]
let d2 = grad(|x: &Array| Ok(grad(f)(x)?.sum(..)))(&x)?;     // 6x  = [6, 12]
# assert_eq!(d1.to_vec::<f32>()?, [3.0, 12.0]);
# assert_eq!(d2.to_vec::<f32>()?, [6.0, 12.0]);
# Ok(())
# }
```

`value_and_grad(f)` returns the value and the gradient of the argument's trainable leaves. Constants such as a batch are captured by the closure; return `(loss, aux)` to also get auxiliary outputs (logits, metrics) back, without gradients:

```rust
# use nuvec::prelude::*;
use nuvec::value_and_grad;

# fn main() -> Result<()> {
let (x, y) = (array![[1.0, 2.0], [3.0, 4.0]], array![1.0, 0.0]);
let loss = |(w, b): &(Array, Array)| {
    let pred = x.matmul(w).add(b);
    let err = pred.subtract(&y);
    (err.multiply(&err).mean(..), pred)                      // (loss, aux)
};
let ((value, pred), (gw, gb)) = value_and_grad(loss)(&(array![0.5, -0.5], array![0.0]))?;
# assert_eq!(pred.shape(), [2]);
# assert_eq!(gw.shape(), [2]);
# assert_eq!(gb.shape(), [1]);
# let _ = value;
# Ok(())
# }
```

The positional forms `grad_flat(f, argnums)` and `value_and_grad_flat(f, argnums)` take functions of `&[Array]`.

## vjp and jvp

The building blocks: `vjp` (reverse mode) pulls cotangents back through a function, `jvp` (forward mode) pushes tangents forward. Both return the function's outputs too, and take and return trees.

```rust
# use nuvec::prelude::*;
use nuvec::{jvp, vjp};

# fn main() -> Result<()> {
let x = Array::from(0.5f32);
let (_, tangent) = jvp(|x: &Array| x.sin(), &x, &Array::from(1.0f32))?;
let (_, grad) = vjp(|x: &Array| x.multiply(x), &x, &Array::from(1.0f32))?;
# assert!((tangent.item::<f32>()? - 0.5f32.cos()).abs() < 1e-6);
# assert_eq!(grad.item::<f32>()?, 1.0);
# Ok(())
# }
```

Every primitive has both rules; a primitive without one fails to compile. Complex functions follow MLX's convention: the cotangent times the conjugate derivative. `vjp_flat` and `jvp_flat` are the positional forms.

## vmap

`vmap(f)` maps a function over a batch axis without writing the loop: axis 0 of every top-level argument and output by default, configured with `.in_axes(..)` (`None`: not batched) and `.out_axes(..)`. Call it with `.call(&inputs)`, or turn it into a closure with `.into_fn()` to compose it with `compile` and the other transforms. Every primitive has a batching rule, so there is no slow fallback.

```rust
# use nuvec::prelude::*;
use nuvec::vmap;

# fn main() -> Result<()> {
# let matrix = array![[1.0, 0.0], [0.0, 2.0]];
# let vectors = array![[1.0, 1.0], [2.0, 3.0], [0.0, 1.0]];
// A matrix-vector product per example, sharing the matrix.
let mv = vmap(|(m, v): &(Array, Array)| m.matmul(v)).in_axes((None, 0));
let out = mv.call(&(matrix, vectors))?;
# assert_eq!(out.to_vec::<f32>()?, [1.0, 2.0, 2.0, 6.0, 0.0, 2.0]);
# Ok(())
# }
```

Per-example gradients are `vmap` of `grad`. `vmap_flat` is the positional form.

## Custom rules

`custom_function` wraps a function so that transforms use your rules instead of differentiating it:

```rust
# use nuvec::prelude::*;
use nuvec::custom_function;

# fn main() -> Result<()> {
# let x = array![1.0, 2.0];
// A straight-through estimator: forward x², gradient as if the identity.
let square = custom_function(|x: &Array| x.multiply(x))
    .with_vjp(|_x: &Array, _y: &Array, cotangent: &Array| Ok(cotangent.clone()));
let y = square.call(&x)?;
# assert_eq!(y.to_vec::<f32>()?, [1.0, 4.0]);
# Ok(())
# }
```

`with_jvp` and `with_vmap` set the other rules. A custom function without rules is re-run by transforms instead of keeping its intermediates. The function and its rules are stored in the graph, so they are `Send + Sync + 'static` (move captured data in).

## Checkpointing

`checkpoint(f)` trades compute for memory: the backward pass recomputes `f`'s intermediates instead of storing them. Wrap the blocks of a deep model to train it in less memory. Random numbers drawn inside (dropout) are drawn again with the same key, so the recomputation sees the same masks.

```rust
# use nuvec::prelude::*;
use nuvec::checkpoint;

# fn transformer_block(x: &Array) -> Array { x.tanh() }
let block = checkpoint(|x: &Array| transformer_block(x));
# assert_eq!(block(&array![0.0]).unwrap().item::<f32>().unwrap(), 0.0);
```

## Rules of thumb

- Functions passed to transforms must not evaluate arrays (no `to_vec` inside); evaluate the results instead.
- Transforms of transforms are fine; derivatives of fused or custom Metal kernels come from their rules or composed fallbacks.
- Plain integer arrays cannot be differentiated: capture integer data (labels, indices) in the closure.
