+++
title = "Transforms: grad, vjp, jvp and vmap"
description = "Composable function transforms in nuvec: reverse- and forward-mode autodiff, higher derivatives, vectorizing map, custom rules and gradient checkpointing."
weight = 30
[extra]
group = "Core"
+++

Transforms take a function of arrays and return a new function. Because their results are lazy arrays, they nest freely: the gradient of a gradient, the vmap of a gradient, a compiled vmap.

## Gradients

```rust
use nuvec::{Array, grad, multiply, sum, value_and_grad};

let f = |x: &Array| sum(&multiply(&multiply(x, x)?, x)?, &[], false); // sum(x³)
let x = Array::from_slice(&[1.0f32, 2.0], &[2])?;
grad(&f)(&x)?;                                                // 3x² = [3, 12]
grad(|x: &Array| sum(&grad(&f)(x)?, &[], false))(&x)?;        // 6x  = [6, 12]
```

`value_and_grad(f, argnums)` returns the value and the gradients with respect to the chosen arguments of a function of several arrays — the usual shape of a training step.

## vjp and jvp

The building blocks: `vjp` (reverse mode) pulls cotangents back through a function, `jvp` (forward mode) pushes tangents forward. Both return the function's outputs too.

```rust
use nuvec::{jvp, sin, vjp};

let (_, tangents) = jvp(|xs| Ok(vec![sin(&xs[0])?]), &[x.clone()], &[Array::scalar(1.0f32)])?;
let (_, grads) = vjp(|xs| Ok(vec![multiply(&xs[0], &xs[0])?]), &[x], &[ones])?;
```

Every primitive has both rules; a primitive without one fails to compile. Complex functions follow MLX's convention: the cotangent times the conjugate derivative.

## vmap

`vmap(f, in_axes, out_axes)` maps a function over a batch axis without writing the loop. Every primitive has a batching rule, so there is no slow fallback.

```rust
use nuvec::{matmul, vmap};

// A matrix-vector product per example, sharing the matrix (in_axes: None = not batched).
let mv = vmap(|xs| Ok(vec![matmul(&xs[0], &xs[1])?]), &[None, Some(0)], &[]);
let out = mv(&[matrix, vectors])?;
```

Per-example gradients are `vmap` of `grad`.

## Custom rules

`custom_function` wraps a function so that transforms use your rules instead of differentiating it:

```rust
use nuvec::custom_function;

// A straight-through estimator: forward x², gradient as if the identity.
let square = custom_function(|xs| Ok(vec![multiply(&xs[0], &xs[0])?]))
    .with_vjp(|_primals, _outputs, cotangents| Ok(cotangents.to_vec()));
let y = square.call(&[x])?;
```

`with_jvp` and `with_vmap` set the other rules. A custom function without rules is re-run by transforms instead of keeping its intermediates.

## Checkpointing

`checkpoint(f)` trades compute for memory: the backward pass recomputes `f`'s intermediates instead of storing them. Wrap the blocks of a deep model to train it in less memory.

```rust
use nuvec::checkpoint;

let block = checkpoint(|xs| Ok(vec![transformer_block(&xs[0])?]));
```

## Rules of thumb

- Functions passed to transforms must not evaluate arrays (no `to_vec` inside); evaluate the results instead.
- Transforms of transforms are fine; derivatives of fused or custom Metal kernels come from their rules or composed fallbacks.
