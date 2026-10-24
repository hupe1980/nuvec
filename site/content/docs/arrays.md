+++
title = "Arrays and lazy evaluation"
description = "How nuvec arrays (tensors) work: lazy graphs on unified memory, 14 dtypes, MLX type promotion, broadcasting, zero-copy views, indexing, reductions and asynchronous evaluation."
weight = 20
[extra]
group = "Core"
+++

An `Array` is a cheap handle to a node in a computation graph. Cloning it shares the node; operations return new handles immediately and compute nothing until a value is needed.

## Arrays and tensors

An `Array` is what PyTorch, Candle and Burn call a tensor: an n-dimensional value with a shape and a dtype. The name follows MLX, JAX and NumPy, whose model nuvec shares: gradients come from transforming functions, not from a tape on each value, and nothing runs until a result is read. Coming from PyTorch:

| PyTorch | nuvec |
|---|---|
| `torch.tensor([[1., 2.]])` | `array![[1.0, 2.0]]` |
| `w.requires_grad_()`, `loss(w).backward()`, `w.grad` | `grad(loss)(&w)?`, or `value_and_grad` for both |
| `x.detach()` | `x.stop_gradient()` |
| `with torch.no_grad():` | not needed: only the function given to `grad` is differentiated |
| each operation runs at once | operations run at `eval`, `item` or `to_vec` |
| `x.to("mps")`, `x.cpu()` | no copies (unified memory); choose where work runs with `with_stream(Device::cpu(), …)` |
| `x.view(shape)`, `x.contiguous()` | `x.reshape(shape)` (a view when the layout allows), `x.contiguous()` |
| `x.item()`, `x.tolist()` | `x.item::<f32>()?`, `x.to_vec::<f32>()?` |
| `torch.func.vmap(f)`, `torch.compile(f)` | `vmap(f)`, `compile(f)` |

```rust
use nuvec::{Array, Result, array, grad};
# fn main() -> Result<()> {
let loss = |w: &Array| w.multiply(w).sum(..); // sum(w²)
let w = array![1.0, 2.0];
let g = grad(loss)(&w)?;                      // PyTorch: loss(w).backward(); w.grad
assert_eq!(g.to_vec::<f32>()?, [2.0, 4.0]);
# Ok(())
# }
```

## Creating arrays

```rust
use nuvec::prelude::*;
use nuvec::ops as mx;

let a = array![[1.0, 2.0], [3.0, 4.0]];                  // float32, [2, 2]
let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]); // data + shape
let c = mx::full([2, 2], 0.5);
let d = mx::zeros([3, 4], DType::BF16);
let e = mx::arange(0, 10, 2);                             // int32: [0, 2, 4, 6, 8]
let f = Array::from(3u8);                                 // a 0-D uint8 scalar
let g: Array = (0..4).map(|i| i as f32).collect();        // from an iterator
```

`array!` takes nested literals (integers give `int32`, floats `float32`, a suffix such as `1u8` picks the type); `From` converts Rust numbers, arrays and `Vec`s; `x.zeros_like()`, `x.ones_like()` and `x.full_like(v)` copy another array's shape and dtype. MLX's free functions are in `nuvec::ops`, best imported as `use nuvec::ops as mx;`. Shapes are `i32`, as in MLX.

## Data types

Fourteen dtypes, as in MLX: `bool`, `u8`–`u64`, `i8`–`i64`, `f16`, `bf16`, `f32`, `f64` and `complex64`. GPU kernels cover the types MLX's do; `f64` runs on the CPU.

Binary operations **promote** mixed types with MLX's table (`int32 × float16 → float16`, `bool + int8 → int8`) and **broadcast** shapes with NumPy's rules. Rust scalars are *weak*: `&x * 2.0` keeps `x`'s dtype instead of promoting to a 64-bit float.

```rust
# use nuvec::prelude::*;
let x = array![1, 2, 3];
let y = array![0.5];
let h = y.astype(DType::F16);

assert_eq!((&x * &y).dtype(), DType::F32);  // int32 × float32, broadcast to [3]
assert_eq!((&x * 2).dtype(), DType::I32);   // an integer scalar keeps the integer type
assert_eq!((&x * 2.0).dtype(), DType::F32); // a float scalar makes integers floating point
assert_eq!((&h * 2.0).dtype(), DType::F16); // but never widens a float type
```

## Views

`reshape`, `transpose` (and `.t()`), indexing with `i`, `broadcast_to`, `expand_dims` and `squeeze` return **views** that share their input's memory whenever the layout allows. Kernels accept arbitrary strides, so views are never copied just to be read.

```rust
# use nuvec::prelude::*;
# use nuvec::ops as mx;
let m = mx::arange(0.0, 12.0, 1.0).reshape([3, 4]);
let col = m.i((.., 1..2));        // m[:, 1:2], [3, 1], a view
let rev = m.i((..).step(-1));     // m[::-1], rows reversed
let row = m.i(-1);                // m[-1], [4]
assert_eq!(col.shape(), [3, 1]);
assert_eq!(rev.i((0, 0)).item::<f32>().unwrap(), 8.0);
assert_eq!(row.sum(..).item::<f32>().unwrap(), 38.0);
```

## Indexing and joining

`x.i(...)` is MLX's `x[...]`: integers (negative from the end), ranges (`1..3`, `..=2`, `1..`), `..` for a whole axis, stepped ranges (`(0..6).step(2)`), `Ellipsis`, `NewAxis` and integer arrays (gathers), combined in a tuple. When the rank is known only at run time, a slice or `Vec` of ranges indexes the leading axes, one range each (`x.i(&ranges)`, `x.slice_update(&u, &ranges)`). `x.at(...)` is MLX's `x.at[...]`: `set` and `add` return the updated array.

```rust
# use nuvec::prelude::*;
let x = array![[1, 2, 3], [4, 5, 6]];
let picked = x.i((.., &array![2, 0]));   // x[:, [2, 0]]: [[3, 1], [6, 4]]
let x = x.at((.., 0)).set(0);            // x[:, 0] = 0
let x = x.at(1).add(10);                 // x[1] += 10
assert_eq!(picked.to_vec::<i32>().unwrap(), [3, 1, 6, 4]);
assert_eq!(x.to_vec::<i32>().unwrap(), [0, 2, 3, 10, 15, 16]);
```

`at` takes the array by value: when nothing else holds it (no other handle, view or pending consumer), the GPU writes the update **in place** — the reason a language model's KV cache costs one write per token. Keep a copy with `x.clone().at(...)`.

`take`, `take_along_axis`, `put_along_axis`, `scatter_add_along_axis`, `split`, `split_at` and the free functions `ops::concatenate` and `ops::stack` cover the rest of indexing and joining.

## Computation

- Element-wise: arithmetic, comparisons, `select` (`ops::where_`), `exp`, `log`, `sin`, `tanh`, `sigmoid`, `logaddexp`, `erfinv`, …
- Reductions over any axes: `sum`, `prod`, `max`, `min`, `mean`, `argmax`, `argmin`, `all`, `any`, `logsumexp`, `softmax`. Axes are an index (`-1`), a list (`[0, 2]`) or `..` for all of them; `sum_keepdims` and the other `*_keepdims` twins keep the reduced axes.
- Sorting and scans with MLX's kernels: `sort`, `argsort`, `partition`, `argpartition`, `topk`, `cumsum`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`.
- Linear algebra: `matmul` (batched, broadcasting), `addmm`, `gather_mm`, `conv1d`.
- Random numbers (`nuvec::random`): `uniform`, `normal`, `gumbel`, `categorical`, `randint`, `bernoulli`, bit-identical to MLX for the same key.

```rust
# use nuvec::prelude::*;
let x = array![[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]];
let total = x.sum(..);                      // every axis: a 0-D array
let rows = x.max(-1);                       // [5, 6]
let centered = &x - x.mean_keepdims(-1);    // [2, 1] broadcasts over [2, 3]
let best = x.softmax(-1).argmax(-1);        // [1, 2]
assert_eq!(total.item::<f32>().unwrap(), 21.0);
assert_eq!(rows.to_vec::<f32>().unwrap(), [5.0, 6.0]);
assert_eq!(centered.sum(..).item::<f32>().unwrap(), 0.0);
assert_eq!(best.to_vec::<u32>().unwrap(), [1, 2]);
```

## Errors

Operations never return `Result`. An invalid call — mismatched shapes, a bad axis, an out-of-range index — gives a **failed array** that records the operation's name, your source location and the reason. A failure is sticky: anything computed from a failed array fails with the same, first error. Nothing panics, and the error surfaces at a **boundary**:

- `x.check()?` reports a recorded failure without evaluating anything (O(1));
- `x.eval()?`, `x.item::<T>()?` and `x.to_vec::<T>()?` evaluate and report failures, GPU errors included;
- I/O, model loading and transforms return `Result` as well.

```rust
# use nuvec::prelude::*;
# fn main() -> Result<()> {
let x = array![[1.0, 2.0, 3.0]];
let y = x.reshape([2, 2]).exp().sum(-1);   // 3 elements cannot be [2, 2]
let err = y.check().unwrap_err();          // the first error wins
assert_eq!(err.op_name(), Some("reshape"));
assert!(err.location().is_some());         // where `reshape` was called
println!("{err}");                         // "reshape at <file>:<line>:<column>: …"

let ok = x.reshape([3, 1]).sum(..).check()?; // `?` belongs at the boundary
assert_eq!(ok.item::<f32>()?, 6.0);
# Ok(())
# }
```

A function that builds arrays and returns `Result<Array>` ends with `Ok(expr)`, or with `expr.check()` to fail early.

## Evaluation

Reading an array (`to_vec`, `item`) or calling `eval` evaluates it and everything it depends on, each node at most once; errors are reported there (see [Errors](#errors)).

`async_eval` submits the work and returns at once, so the host can prepare the next step while the GPU runs this one; reading the array later waits for exactly the work it needs. `schedule` goes one step further and orders a graph ahead of time, so that starting it later costs almost nothing:

```rust
# use nuvec::prelude::*;
use nuvec::async_eval;
# fn expensive_step(state: &Array) -> Array { state.exp().sum(..) }
# fn prepare_next_batch() -> Result<()> { Ok(()) }
# fn main() -> Result<()> {
# let state = array![1.0, 2.0];

let step = expensive_step(&state);
async_eval(&step)?;  // the GPU starts now
prepare_next_batch()?;                     // overlaps with GPU work
let value = step.item::<f32>()?;           // waits only for `step`
# let _ = value;
# Ok(())
# }
```

Evaluation order follows MLX's breadth-first scheme, so independent work (a layer's query, key and value projections) runs side by side on the GPU.
