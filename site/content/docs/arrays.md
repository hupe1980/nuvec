+++
title = "Arrays and lazy evaluation"
description = "How nuvec arrays work: lazy graphs on unified memory, 14 dtypes, MLX type promotion, broadcasting, zero-copy views, indexing, reductions and asynchronous evaluation."
weight = 20
[extra]
group = "Core"
+++

An `Array` is a cheap handle to a node in a computation graph. Cloning it shares the node; operations return new handles immediately and compute nothing until a value is needed.

## Creating arrays

```rust
use nuvec::{Array, DType, arange, ones, zeros};

let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?; // data + shape
let b = Array::full(&[2, 2], 0.5f32)?;
let c = zeros(&[3, 4], DType::BF16)?;
let d = arange(0.0, 10.0, 2.0, DType::I32)?; // [0, 2, 4, 6, 8]
let e = Array::scalar(3u8);
```

## Data types

Fourteen dtypes, as in MLX: `bool`, `u8`–`u64`, `i8`–`i64`, `f16`, `bf16`, `f32`, `f64` and `complex64`. GPU kernels cover the types MLX's do; `f64` runs on the CPU.

Binary operations **promote** mixed types with MLX's table (`int32 × float16 → float16`, `bool + int8 → int8`) and **broadcast** shapes with NumPy's rules. Rust scalars are *weak*: `&x * 2.0` keeps `x`'s dtype instead of promoting to a 64-bit float.

```rust
let x = Array::from_slice(&[1i32, 2, 3], &[3])?;
let y = Array::from_slice(&[0.5f32], &[1])?;
let h = y.astype(DType::F16)?;

(&x * &y).dtype(); // F32: int32 × float32, broadcast to [3]
(&x * 2).dtype();  // I32: an integer scalar keeps the integer type
(&x * 2.0).dtype(); // F32: a float scalar makes integers floating point
(&h * 2.0).dtype(); // F16: but never widens a float type
```

## Views

`reshape`, `transpose` (and `.t()`), `slice`, `broadcast_to`, `expand_dims` and `squeeze` return **views** that share their input's memory whenever the layout allows. Kernels accept arbitrary strides, so views are never copied just to be read.

```rust
use nuvec::{SliceArg, slice};

let m = arange(0.0, 12.0, 1.0, DType::F32)?.reshape(&[3, 4])?;
let col = slice(&m, &[SliceArg::full(), SliceArg::from(1..2)])?; // [3, 1], a view
let rev = slice(&m, &[SliceArg::new(None, None, -1), SliceArg::full()])?; // rows reversed
```

## Indexing and joining

`take`, `take_along_axis`, `put_along_axis`, `scatter_add_along_axis`, `concatenate`, `stack`, `split`, `split_at` and `slice_update`. On the GPU, `slice_update` writes **in place** when nothing else can see the updated array — the reason a language model's KV cache costs one write per token.

## Computation

- Element-wise: arithmetic, comparisons, `where_`, `exp`, `log`, `sin`, `tanh`, `sigmoid`, `logaddexp`, `erfinv`, …
- Reductions over any axes: `sum`, `prod`, `max`, `min`, `mean`, `argmax`, `argmin`, `all`, `any`, `logsumexp`, `softmax`.
- Sorting and scans with MLX's kernels: `sort`, `argsort`, `partition`, `argpartition`, `topk`, `cumsum`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`.
- Linear algebra: `matmul` (batched, broadcasting), `addmm`, `gather_mm`, `conv1d`.
- Random numbers (`nuvec::random`): `uniform`, `normal`, `gumbel`, `categorical`, `randint`, `bernoulli`, bit-identical to MLX for the same key.

## Evaluation

Reading an array (`to_vec`, `item`) or calling `eval` evaluates it and everything it depends on, each node at most once. Errors are values: every operation returns `Result`, and a failure is sticky — anything computed from a failed array reports the same error.

`async_eval` submits the work and returns at once, so the host can prepare the next step while the GPU runs this one; reading the array later waits for exactly the work it needs. `schedule` goes one step further and orders a graph ahead of time, so that starting it later costs almost nothing:

```rust
use nuvec::{async_eval, schedule};

let step = expensive_step(&state)?;
async_eval(std::slice::from_ref(&step))?; // the GPU starts now
prepare_next_batch()?;                     // overlaps with GPU work
let value = step.item::<f32>()?;           // waits only for `step`
```

Evaluation order follows MLX's breadth-first scheme, so independent work (a layer's query, key and value projections) runs side by side on the GPU.
