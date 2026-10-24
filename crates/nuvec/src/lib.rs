//! A pure-Rust array framework for Apple Silicon in the spirit of
//! [MLX](https://github.com/ml-explore/mlx): lazy n-dimensional arrays (tensors) on unified
//! memory and composable function transforms, with no bindings to any C/C++ library.
//!
//! ```
//! use nuvec::prelude::*;
//!
//! let x = array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
//! let col = array![[10], [20]];               // broadcasts over columns
//! let y = (&x * 2.0 + &col).t();              // lazy, float32, a view
//! assert_eq!(y.shape(), [3, 2]);
//! assert_eq!(y.to_vec::<f32>().unwrap(), [10.0, 26.0, 12.0, 28.0, 14.0, 30.0]);
//! ```
//!
//! Operations are methods that chain and never return `Result`: an invalid call gives a
//! *failed array* that remembers the operation, your source location and the reason, and the
//! error surfaces where you read results (`check()?`, `eval()?`, `item()?`, `to_vec()?`):
//!
//! ```
//! use nuvec::prelude::*;
//!
//! # fn main() -> Result<()> {
//! let a = array![[1.0, 2.0, 3.0]];
//! let bad = a.matmul(&a).exp().sum(-1);       // [1, 3] @ [1, 3]: fails, nothing panics
//! let err = bad.check().unwrap_err();
//! assert_eq!(err.op_name(), Some("matmul"));  // `err` also has the file and line
//! let ok = a.matmul(&a.t()).sum(..).check()?; // the boundary returns `Result`
//! assert_eq!(ok.item::<f32>()?, 14.0);
//! # Ok(())
//! # }
//! ```
//!
//! The free functions with MLX's names are in [`ops`] (`use nuvec::ops as mx;`), so MLX code
//! ports line by line. Gradients are lazy arrays too, so they compose:
//!
//! ```
//! use nuvec::{Array, Result, array, grad};
//!
//! # fn main() -> Result<()> {
//! let f = |x: &Array| x.multiply(x).multiply(x).sum(..);  // sum(x³)
//! let x = array![1.0, 2.0];
//! assert_eq!(grad(f)(&x)?.to_vec::<f32>()?, [3.0, 12.0]);  // 3x²
//! let g2 = grad(|x: &Array| Ok(grad(f)(x)?.sum(..)))(&x)?;
//! assert_eq!(g2.to_vec::<f32>()?, [6.0, 12.0]);
//! # Ok(())
//! # }
//! ```
//!
//! Transforms work on *trees*: arrays, tuples, `Vec`s, maps and your own structs with
//! `#[derive(Tree)]` (models with nuvec-nn's `#[derive(Module)]`). A gradient has the
//! argument's shape, and a training step over `(model, optimizer)` compiles with
//! [`compile_mut`]:
//!
//! ```
//! use nuvec::{Array, Result, Tree, array, value_and_grad};
//!
//! # fn main() -> Result<()> {
//! #[derive(Clone, Tree)]
//! struct Affine { w: Array, b: Array }
//!
//! let x = array![1.0, 2.0];
//! let loss = |p: &Affine| p.w.multiply(&x).add(&p.b).sum(..); // annotate the parameter
//! let (value, grads) = value_and_grad(loss)(&Affine { w: array![3.0, 4.0], b: array![1.0, 1.0] })?;
//! assert_eq!(value.item::<f32>()?, 13.0);
//! assert_eq!(grads.w.to_vec::<f32>()?, [1.0, 2.0]);              // grads: Affine
//! nuvec::eval(&(&value, &grads))?;                              // any tree, borrowed
//! # Ok(())
//! # }
//! ```
//!
//! What is inside:
//!
//! - **Arrays** with MLX/NumPy semantics: 14 dtypes, type promotion, broadcasting, zero-copy
//!   views, indexing, reductions, sorting, scans and matmul.
//! - **Transforms** that compose: [`grad`], [`value_and_grad`], [`vjp`], [`jvp`], [`vmap`],
//!   [`custom_function`], [`checkpoint`], [`compile()`] (kernel fusion) and [`compile_mut`]
//!   (stateful steps), over [trees](tree) (the positional forms are `*_flat`).
//! - **Trees** ([`Tree`], [`Param`], `#[derive(Tree)]`, `#[derive(Static)]`): paths,
//!   structure ([`TreeDef`]) and checkpoints of any tree ([`save`], [`load_into`]).
//! - **Metal kernels** ported from MLX (GEMM, GEMV, quantized products, attention,
//!   normalisation, RoPE, sort, scan), MLX-compatible affine [quantization](Array::quantize) and
//!   custom kernels from source ([`fast::MetalKernel`]).
//! - **Indexing** as MLX's `x[...]` ([`Array::i`]) and functional updates ([`Array::at`]).
//! - **Evaluation** on the GPU by default, submitted asynchronously ([`async_eval`],
//!   [`schedule`]), or on the CPU; memory limits and wired memory on [`gpu::Device`].
//! - **I/O** ([`io`]): safetensors and npy, memory-mapped without a copy.
//!
//! The default stream of each thread is the GPU's when Metal is available, else the CPU's
//! ([`set_default_stream`] changes it per thread). Several threads may evaluate arrays that
//! share work. A transform (`grad`, `vmap`, `compile`, …) traces on the calling thread:
//! evaluating its tracers on another thread inside it (rayon, say) is unsupported.
//!
//! The crate root holds the types, [`array!`], evaluation, transforms and streams; the
//! operations are [`Array`] methods with their free-function mirrors in [`ops`];
//! [`prelude`] imports the everyday names; [`graph`] is the graph-level API for compilers.
//! Neural-network layers and optimizers are in `nuvec-nn`, the LLM runtime in `nuvec-lm`.
//!
//! # Determinism
//!
//! For one nuvec version on one GPU generation, with the same shapes, dtypes and options:
//!
//! - **Run to run and across processes:** the same operations on the same inputs give
//!   bit-identical outputs, whatever memory the allocator hands out, whatever work is in
//!   flight, wherever command buffers end, on any thread and in any process.
//! - **Build modes:** precompiled kernels and runtime-compiled ones (`NUVEC_METALLIB=off`)
//!   give the same bits.
//! - **Row independence:** a product with at most [`ROW_INDEPENDENT_ROWS`] rows gives each
//!   row the bits of the same row's one-row product, by default and with no mode: dense
//!   matrix products (float32, float16, bfloat16; a multi-row matrix-vector kernel that
//!   reads the weights once), quantized products (every bit width and group size;
//!   untransposed weights run one product per row), and gathered (mixture-of-experts)
//!   products, quantized or not, with sorted or unsorted indices (up to
//!   `ROW_INDEPENDENT_ROWS` tokens of up to 8 experts each). So are the row-wise operations
//!   of a decode step: `rms_norm`, `add_rms_norm`, `rope` and `norm_rope` with per-row
//!   offsets, element-wise operations, vector attention over equal key lengths,
//!   `gated_delta` and `gated_delta_slots`.
//! - **Batch items of quantized products:** a quantized product over `[B, L, K]` (tiled,
//!   `L` above the matrix-vector limit) gives each item the bits of its product alone
//!   (`[1, L, K]`): the split-K partitions are counted per item, so prompts of the same
//!   length prefilled together get their solo bits.
//! - **Speculative verification (the serial rule):** attention with 2 to 8 causal query
//!   rows gives each row the bits of the one-query call over its causal key prefix
//!   ([`fast::scaled_dot_product_attention`]).
//! - **Routing is pure:** which kernel runs depends only on shapes, dtypes, options and the
//!   GPU generation, never on memory state ([`fast::plan_attention`] reports it).
//! - **Chunk plans:** the same chunk plan gives the same bits, and resuming at a chunk
//!   boundary from a [`ops::copy`] snapshot equals the uninterrupted run.
//!
//! Not promised: equality across nuvec versions, GPU generations or different chunk plans
//! (a prompt prefilled in other chunks may round differently), and equality with MLX.
//!
//! Not deterministic (the order in which GPU threads commit decides the rounding):
//!
//! - `scatter_add_along_axis` (and `at(..).add`) on floating-point values with repeated
//!   indices, and derivatives built on it (the reverse mode of gathers, `take`,
//!   `take_along_axis` and indexing with repeated indices): float atomics (`indexing.metal`). With unique
//!   indices the result is exact.
//! - `put_along_axis` with repeated indices *and* differing values (one of the writes wins).
//!   Unique indices, or one fill value, are deterministic (as a top-k sampler's
//!   `argpartition` indices and fill).
//! - [`fast::MetalKernel`]s whose source uses atomics.
//!
//! # Stability
//!
//! Semantic versioning covers every item reachable through `nuvec::` paths, and `nuvec-lm`;
//! the sub-crates re-exported here make no promise when depended on directly. Runtimes can
//! rely on arrays and operations, the `fast` kernels, quantized and gathered products,
//! custom Metal kernels, `compile`, random numbers, evaluation, memory limits and statistics,
//! safetensors I/O and the attention plan. The pinned items are listed in this crate's
//! `tests/api_surface.rs`.
//!
//! **Status:** pre-alpha: the API may change between releases.

#![forbid(unsafe_code)]

pub use nuvec_compile as compile;
pub use nuvec_compile::{compile, compile_flat, compile_mut};
pub use nuvec_core::tree::{
    LeafKind, Leaves, LoadMode, Param, Static, Tree, TreeDef, Visit, VisitMut, Visitor, VisitorMut,
};
pub use nuvec_core::{
    Array, ArrayIndex, At, Axes, Axis, Category, Complex64, CustomFunction, CustomFunctionFlat,
    DType, Device, DeviceKind, Element, Ellipsis, Error, GradOutput, InAxes, InAxis, IntoResult,
    IntoShape, NewAxis, OpErrorKind, OutAxes, Result, Scalar, ScalarOperand, ScalarOrArray,
    ScanOptions, Scheduled, Step, Stepped, Stream, VarOptions, Vmap, array, async_eval,
    at_least_float, bf16, checkpoint, checkpoint_flat, custom_function, custom_function_flat,
    default_device, default_stream, eval, f16, grad, grad_flat, jvp, jvp_flat, promote_types,
    schedule, set_default_stream, value_and_grad, value_and_grad_flat, vjp, vjp_flat, vmap,
    vmap_flat, with_stream,
};
pub use nuvec_core::{ROW_INDEPENDENT_ROWS, fast, graph, ops, random, tree};
pub use nuvec_io::{load_into, save};
pub use nuvec_macros::{Static, Tree};

/// What the derive macros generate code against. Not a public API.
#[doc(hidden)]
pub use nuvec_core::__derive;
pub use nuvec_cpu as cpu;
pub use nuvec_gpu as gpu;
pub use nuvec_io as io;
pub use nuvec_macros as macros;

/// The everyday names in one import: `use nuvec::prelude::*;`. The free functions are in
/// [`ops`], best imported as a module: `use nuvec::ops as mx;`.
pub mod prelude {
    pub use nuvec_core::prelude::*;
}

/// Graph internals are not at the root; they are in [`graph`]:
///
/// ```compile_fail
/// use nuvec::Primitive;
/// ```
///
/// ```compile_fail
/// use nuvec::Fused;
/// ```
///
/// ```compile_fail
/// use nuvec::SliceArg; // pinned absent from the root
/// ```
///
/// Free functions are in [`ops`], not at the root (a root `std` would shadow `std::`):
///
/// ```compile_fail
/// let _ = nuvec::zeros([2], nuvec::DType::F32);
/// ```
///
/// ```
/// let _ = nuvec::ops::zeros([2], nuvec::DType::F32);
/// use nuvec::graph::{Fused, Primitive, SliceArg};
/// ```
#[cfg(doctest)]
pub struct RootHygiene;

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The Rust snippets of the user guide (the site's pages that use only this crate), run as
/// doctests so the documentation keeps compiling.
#[cfg(doctest)]
mod site_doctests {
    #[doc = include_str!("../../../site/content/_index.md")]
    pub struct Index;
    #[doc = include_str!("../../../site/content/docs/_index.md")]
    pub struct Docs;
    #[doc = include_str!("../../../site/content/docs/architecture.md")]
    pub struct Architecture;
    #[doc = include_str!("../../../site/content/docs/arrays.md")]
    pub struct Arrays;
    #[doc = include_str!("../../../site/content/docs/benchmarks.md")]
    pub struct Benchmarks;
    #[doc = include_str!("../../../site/content/docs/compile.md")]
    pub struct Compile;
    #[doc = include_str!("../../../site/content/docs/faq.md")]
    pub struct Faq;
    #[doc = include_str!("../../../site/content/docs/gpu.md")]
    pub struct Gpu;
    #[doc = include_str!("../../../site/content/docs/io.md")]
    pub struct Io;
    #[doc = include_str!("../../../site/content/docs/kernels.md")]
    pub struct Kernels;
    #[doc = include_str!("../../../site/content/docs/quantization.md")]
    pub struct Quantization;
    #[doc = include_str!("../../../site/content/docs/testing.md")]
    pub struct Testing;
    #[doc = include_str!("../../../site/content/docs/transforms.md")]
    pub struct Transforms;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
        assert_eq!(VERSION, nuvec_core::VERSION);
    }

    #[test]
    fn root_api_works() {
        let a = array![1, 2];
        let b = a.multiply(&a) + &a;
        assert_eq!(b.to_vec::<i32>().unwrap(), [2, 6]);
        assert_eq!(ops::multiply(&a, 3).to_vec::<i32>().unwrap(), [3, 6]);
    }
}
