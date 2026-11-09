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

`array!` takes nested literals (integers give `int32`, floats `float32`, a suffix such as `1u8` picks the type); `From` converts Rust numbers, arrays and `Vec`s; `x.zeros_like()`, `x.ones_like()` and `x.full_like(v)` copy another array's shape and dtype. MLX's other constructors are there too: `linspace`, `eye`, `identity`, `tri`, `meshgrid`, `empty`, and the windows `hanning`, `hamming`, `blackman`, `bartlett`. MLX's free functions are in `nuvec::ops`, best imported as `use nuvec::ops as mx;`. Shapes are `i32`, as in MLX.

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

Rearranging, as MLX: `flip`, `roll`, `tile`, `pad` (constant, edge, reflect or symmetric), `moveaxis`, `swapaxes`, `matrix_transpose`, `flatten`, `unflatten`, `unstack`, `atleast_1d`/`2d`/`3d`, `broadcast_arrays`, `tril`, `triu`, `diag`, `diagonal` and `as_strided` (a gather, so any strides are safe).

## Computation

- Element-wise, with MLX's kernels on the CPU and the GPU, gradients and fusion under `compile`:
  - arithmetic, comparisons, `select` (`ops::where_`), `logaddexp`, `floor_divide`, `remainder` (the divisor's sign), `divmod`, `power`, `square`, `reciprocal`, `clip`;
  - `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `sqrt`, `rsqrt`, `sigmoid`, `erf`, `erfinv`;
  - `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh` and their inverses (`arcsin`, …, `arctan2`), `degrees`, `radians`;
  - `floor`, `ceil`, `round` (with decimals, ties to even), `trunc`;
  - `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_invert`, `left_shift`, `right_shift`, and `logical_and`, `logical_or`, `logical_xor`, `logical_not`;
  - `isnan`, `isinf`, `isfinite`, `isposinf`, `isneginf`, `nan_to_num`, `isclose`, `allclose`, `array_equal`.

  `erf` and `expm1` are within one unit in the last place on both devices (MLX's GPU `expm1` and its CPU `erf` are less accurate).
- Reductions over any axes: `sum`, `prod`, `max`, `min`, `mean`, `argmax`, `argmin`, `all`, `any`, `logsumexp`, `softmax` (half precision computed in float32, MLX's `precise=True`). Axes are an index (`-1`), a list (`[0, 2]`) or `..` for all of them; `sum_keepdims` and the other `*_keepdims` twins keep the reduced axes.
- Sorting and scans with MLX's kernels: `sort`, `argsort`, `partition`, `argpartition`, `topk`, `cumsum`, `cumprod`, `cummax`, `cummin`, `logcumsumexp`.
- Statistics and search: `median`, `diff`, `count_nonzero`, `minmax`, `searchsorted`, `unique` (a fixed output size, so nothing is evaluated early).
- Linear algebra: `matmul` (batched, broadcasting), `addmm`, `gather_mm`, `segmented_mm` (one product per segment of the inner axis), `block_masked_mm`, `tensordot`, `inner`, `outer`, `kron`, `vecdot`, `einsum` (MLX's greedy contraction order; `einsum_path` reports it), `trace`, `matrix_norm` (every vector and matrix norm, the SVD-based ones included).
- Transforms: `hadamard_transform` (sizes `m · 2^k`, `m` in 1, 12, 20, 28) on MLX's kernels and bits.
- Quantized products: `quantized_matmul`, `gather_qmm`, and `qqmm` and `gather_qqmm`, which quantize the activations too (MXFP4, MXFP8, NVFP4; with a dense weight `qqmm`'s gradients are quantized products, as MLX's).
- Convolutions: `conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`/`2d`/`3d`, `conv_general` and `convolve`. See [Convolutions](#convolutions).
- Decompositions (`nuvec::linalg`, MLX's `mx.linalg`): `qr`, `svd`, `eigh`, `eigvalsh`, `eig`, `eigvals`, `lu`, `lu_factor`, `cholesky`, `inv`, `tri_inv`, `cholesky_inv`, `pinv`, `solve`, `solve_triangular`, `det`, `slogdet`, `cross` and `norm`, in float32 and float64, batched. See [Linear algebra](#linear-algebra).
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

## Linear algebra

`nuvec::linalg` has MLX's `mx.linalg`, with two differences that favour nuvec:

- **Any stream:** the decompositions run on the CPU, also inside a GPU computation. On unified memory the CPU kernels read and write the GPU graph's buffers, so no copy is made; MLX asks for a CPU stream instead.
- **Gradients:** `inv`, `solve`, `solve_triangular`, `tri_inv`, `cholesky`, `cholesky_inv`, `qr` (square and tall), `eigh`, `eigvalsh`, `svd`, `pinv`, `det`, `slogdet` and the SVD-based norms are differentiable (`grad`, `vjp`, `jvp`, `vmap`); MLX's `inv`, `solve`, `qr`, `svd`, `eigh` and `cholesky` are not. `eigh` and `cholesky` take their input as symmetric. `svd`'s vectors need distinct singular values. `eig` and the LU pivots have no derivative.

```rust
# use nuvec::prelude::*;
use nuvec::linalg;

let a = array![[4.0f32, 1.0], [1.0, 3.0]];
let [w, v] = linalg::eigh(&a, false);                 // ascending eigenvalues
let g = nuvec::grad(|x: &Array| Ok(linalg::inv(x).sum(..)))(&a).unwrap();
let x = linalg::solve(&a, &array![1.0f32, 2.0]);
# let _ = (w, v, g, x);
```

The kernels are pure Rust (`faer`). With the feature `accelerate`, Apple targets call Accelerate's LAPACK, as MLX does, on the AMX units. On an M4 Pro against MLX's CPU (`scripts/bench_linalg.py`, float32, 64 to 1,024):

- **With `accelerate`:** 0.24–1.03× MLX's time. `solve` is 3–4× faster than MLX's.
- **Pure Rust:** MLX's speed or faster at 64. At 256 and above, `eigh`, `solve`, `qr` and `svd` take 1.1–1.6× MLX's time, and the BLAS-3-bound `cholesky`, `inv` and `lu` 2–7×.

## Convolutions

The convolutions take MLX's layout and arguments: the input is channels last (`[N, spatial…, C_in]`) and the weight `[C_out, kernel…, C_in / groups]`. An argument per spatial axis is one value for every axis (`1`) or one each (`[2, 1]`). `conv_general` adds asymmetric and negative padding (a crop of the input), input dilation and flipped kernels (`ConvOptions`); the transposed convolutions are flipped convolutions of the input dilated by the stride, as in MLX.

```rust
# use nuvec::prelude::*;
use nuvec::ops::{self, ConvOptions};

let x = nuvec::random::normal([1, 8, 8, 3], DType::F32, 0.0, 1.0, None);
let w = nuvec::random::normal([16, 3, 3, 3], DType::F32, 0.0, 1.0, None);
let y = ops::conv2d(&x, &w, 1, 1, 1, 1);                     // [1, 8, 8, 16]
let down = ops::conv2d(&x, &w, [2, 1], [1, 0], 1, 1);        // per-axis stride and padding
let crop = ConvOptions::new().padding_lo_hi(&[-1, 0], &[0, 1]).flip(true);
let z = ops::conv_general(&x, &w, &crop);                    // [1, 5, 7, 16]
assert_eq!(y.shape(), [1, 8, 8, 16]);
assert_eq!(down.shape(), [1, 4, 6, 16]);
assert_eq!(z.shape(), [1, 5, 7, 16]);
```

On the GPU they run MLX's Metal kernels as MLX routes them: implicit GEMM (with its small-channel and small-filter loaders, and a general kernel for strides, dilations and flips), Winograd F(6×6, 3×3) for wide 3 × 3 layers, depthwise kernels, and an unfold and a GEMM where MLX unfolds. Every convolution is differentiable. On an M4 Pro (`scripts/bench_conv.py`), nuvec takes 0.97–1.07× MLX's time on vision shapes: patch embeddings, ResNet blocks and stems, 1 × 1, depthwise 3 × 3 and 7 × 7, transposed and 3-D convolutions.

Two results differ from MLX's, in favour of accuracy:
- **Winograd keeps float32 tiles for half-precision inputs.** MLX rounds the transformed tiles to the input type and loses up to 2% (float16) and 13% (bfloat16) of the output's scale. nuvec's error is the direct convolution's (one rounding), at 1.2–1.6× MLX's time on the wide 3 × 3 layers where MLX takes Winograd. `NUVEC_CONV_WINOGRAD=fast` takes MLX's half-precision tiles instead (its bits, 1.04–1.14× its time); `NUVEC_CONV_WINOGRAD=0` turns Winograd off, as MLX's `MLX_CONV_WINOGRAD=0`.
- **Depthwise products are float32.** MLX's depthwise kernel rounds each product to the input type before the float32 sum.

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
