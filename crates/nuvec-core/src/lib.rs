//! Lazy array graph, primitives, dtypes, streams and evaluation: the core of nuvec.
//!
//! Operations on [`Array`]s record a graph and return immediately. Nothing is computed
//! until an array is evaluated, explicitly or by reading it back:
//!
//! ```
//! use nuvec_core::{Result, array, ops};
//!
//! # fn main() -> Result<()> {
//! let a = array![1.0, 2.0, 3.0];
//! let b = ops::full([3], 10.0);
//! let c = (&a + &b) * &a; // lazy: nothing computed yet
//! assert!(!c.is_evaluated());
//! assert_eq!(c.to_vec::<f32>()?, vec![11.0, 24.0, 39.0]);
//! # Ok(())
//! # }
//! ```
//!
//! Operations are methods (`x.exp().sum(-1)`), with free mirrors of MLX's names in [`ops`].
//! They never return `Result`: an invalid call gives a *failed array* carrying the op, the
//! caller's source location and the reason ([`Error::Op`]); arrays computed from it fail
//! with the same error, which surfaces at [`Array::check`], [`Array::eval`], [`Array::item`]
//! or [`Array::to_vec`]:
//!
//! ```
//! use nuvec_core::{OpErrorKind, array};
//!
//! let x = array![[1.0, 2.0, 3.0]];
//! let y = x.reshape([2, -1]).exp(); // 3 elements cannot be [2, -1]
//! let err = y.check().unwrap_err();
//! assert!(matches!(err.kind(), Some(OpErrorKind::InvalidShape(_))));
//! assert_eq!(err.location().unwrap().file(), file!());
//! ```
//!
//! Binary operations promote types (MLX's table) and broadcast shapes (NumPy rules); views
//! (`reshape`, `transpose`, basic indexing with [`Array::i`], `broadcast_to`, `expand_dims`,
//! `squeeze`) share data.
//!
//! Reductions (`sum`, `mean`, `max`, `argmax`, …) work over any axes and layouts;
//! `matmul` broadcasts batch dimensions and uses a SIMD GEMM on the CPU.
//!
//! Arrays live in unified memory. Operations on the GPU stream run as Metal kernels; anything
//! without a GPU kernel (for example float64) runs on the CPU over the same memory:
//!
//! ```
//! use nuvec_core::{Device, array, with_stream};
//!
//! # fn main() -> nuvec_core::Result<()> {
//! let x = array![1.0, 2.0, 3.0];
//! // A Metal kernel on a Mac; an `UnsupportedDevice` error elsewhere.
//! let y = with_stream(Device::gpu(0), || &x * &x);
//! # if y.eval().is_ok() { assert_eq!(y.to_vec::<f32>()?, vec![1.0, 4.0, 9.0]); }
//! # Ok(())
//! # }
//! ```
//!
//! Each thread's default stream is the GPU's when Metal is available, else the CPU's
//! ([`default_device`]); [`with_stream`] changes it for a scope, [`set_default_stream`] for
//! the calling thread.
//!
//! Threads may evaluate arrays that share work concurrently (each node runs once). A
//! transform traces on the calling thread: evaluating its tracers on other threads
//! (rayon, say) while it runs is unsupported.
//!
//! **Stability:** no promise when this crate is depended on directly; depend on `nuvec`,
//! whose re-exports are semver-tracked.
//!
//! **Status:** pre-alpha.

// The only `unsafe` code is the GPU dispatch helpers in `gpu.rs` (nuvec-gpu's dispatch is
// `unsafe`: core vouches for its own kernels staying in bounds).
#![deny(unsafe_code)]

mod array;
mod axes;
mod batching;
mod buffer;
mod cast;
mod conv;
mod creation;
mod custom;
mod derivatives;
mod device;
mod display;
mod dtype;
mod error;
mod eval;
pub mod fast;
mod file;
// Parts of it are only used by the Metal backend.
#[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
mod fused;
mod gpu;
pub mod graph;
mod index;
mod indexing;
// Parts of it are only used by the Metal backend.
#[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
mod kernel;
mod kernels;
pub mod ops;
mod primitive;
mod quantized;
pub mod random;
mod scalar;
mod shape;
mod sorting;
mod transforms;
pub mod tree;
mod view;

#[doc(hidden)]
pub use creation::__private;

pub use array::Array;
pub use axes::{Axes, Axis, IntoShape, ScanOptions, VarOptions};
pub use custom::{
    CustomFunction, CustomFunctionFlat, checkpoint, checkpoint_flat, custom_function,
    custom_function_flat,
};
pub use device::{
    Device, DeviceKind, Stream, default_device, default_stream, set_default_stream, with_stream,
};
pub use dtype::{Category, Complex64, DType, Element, at_least_float, bf16, f16, promote_types};
pub use error::{Error, OpErrorKind, Result};
pub use eval::{Scheduled, async_eval, eval, schedule};
pub use file::MappedFile;
pub use index::{ArrayIndex, At, Ellipsis, NewAxis, Step, Stepped};
pub use scalar::{Scalar, ScalarOperand, ScalarOrArray};
pub use transforms::{
    GradOutput, InAxes, InAxis, IntoResult, OutAxes, Vmap, grad, grad_flat, jvp, jvp_flat,
    value_and_grad, value_and_grad_flat, vjp, vjp_flat, vmap, vmap_flat,
};
#[doc(hidden)]
pub use tree::__derive;
pub use tree::{
    LeafKind, Leaves, LoadMode, Param, Static, Tree, TreeDef, Visit, VisitMut, Visitor, VisitorMut,
};

// Graph-level items, public in `graph`, used across the crate by their short paths.
pub(crate) use graph::SliceArg;
pub(crate) use graph::UnaryOp;

/// The everyday names in one import: `use nuvec_core::prelude::*;`. Operations are methods
/// of [`Array`]; the free functions (MLX's names) are in [`ops`], best imported as a module
/// (`use nuvec_core::ops as mx;`, then `mx::zeros([2, 3], DType::F32)`).
pub mod prelude {
    pub use crate::{Array, DType, Ellipsis, Error, NewAxis, Result, Step, array, with_stream};
}

/// Products with at most this many rows give each row the bits of the same row's
/// one-row product (the row-independence contract): dense matrix products in float32,
/// float16 and bfloat16, quantized products (every bit width and group size), and the
/// gathered (mixture-of-experts) products. See the `nuvec` crate's "Determinism" section.
pub const ROW_INDEPENDENT_ROWS: usize = 8;

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
