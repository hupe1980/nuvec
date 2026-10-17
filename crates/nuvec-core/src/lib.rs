//! Lazy array graph, primitives, dtypes, streams and evaluation: the core of nuvec.
//!
//! Operations on [`Array`]s record a graph and return immediately. Nothing is computed
//! until an array is evaluated, explicitly or by reading it back:
//!
//! ```
//! use nuvec_core::{Array, Result};
//!
//! # fn main() -> Result<()> {
//! let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
//! let b = Array::full(&[3], 10.0f32)?;
//! let c = (&a + &b) * &a; // lazy: nothing computed yet
//! assert!(!c.is_evaluated());
//! assert_eq!(c.to_vec::<f32>()?, vec![11.0, 24.0, 39.0]);
//! # Ok(())
//! # }
//! ```
//!
//! Binary operations promote types (MLX's table) and broadcast shapes (NumPy rules); views
//! (`reshape`, `transpose`, `slice`, `broadcast_to`, `expand_dims`, `squeeze`) share data.
//!
//! Reductions (`sum`, `mean`, `max`, `argmax`, …) work over any axes and layouts;
//! `matmul` broadcasts batch dimensions and uses a SIMD GEMM on the CPU.
//!
//! Arrays live in unified memory. Operations on the GPU stream run as Metal kernels; anything
//! without a GPU kernel (for example float64) runs on the CPU over the same memory:
//!
//! ```
//! use nuvec_core::{Array, Device, Stream, set_default_stream};
//!
//! # fn main() -> nuvec_core::Result<()> {
//! set_default_stream(Stream::default_for(Device::gpu(0)));
//! let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
//! let y = &x * &x; // a Metal kernel on a Mac; an `UnsupportedDevice` error elsewhere
//! # if y.eval().is_ok() { assert_eq!(y.to_vec::<f32>()?, vec![1.0, 4.0, 9.0]); }
//! # set_default_stream(Stream::default_for(Device::cpu()));
//! # Ok(())
//! # }
//! ```
//!
//! **Status:** pre-alpha. The default stream is the CPU.

#![forbid(unsafe_code)]

mod array;
mod batching;
mod buffer;
mod cast;
mod custom;
mod derivatives;
mod device;
mod dtype;
mod error;
mod eval;
mod gpu;
mod kernels;
mod ops;
mod primitive;
pub mod random;
mod scalar;
mod shape;
mod transforms;
mod view;

pub use array::Array;
pub use custom::{CustomCall, CustomFunction, checkpoint, custom_function};
pub use device::{Device, DeviceKind, Stream, default_device, default_stream, set_default_stream};
pub use dtype::{Category, Complex64, DType, Element, at_least_float, bf16, f16, promote_types};
pub use error::{Error, Result};
pub use eval::{async_eval, eval};
pub use ops::{
    abs, add, all, any, arange, argmax, argmin, astype, broadcast_to, conjugate, cos, divide,
    equal, erfinv, exp, expand_dims, greater, greater_equal, less, less_equal, log, logsumexp,
    matmul, max, maximum, mean, min, minimum, multiply, negative, not_equal, ones, prod, reshape,
    sigmoid, sin, slice, softmax, sqrt, squeeze, stop_gradient, subtract, sum, tanh, transpose,
    where_, zeros,
};
pub use primitive::{ArgReduceOp, BinaryOp, Primitive, ReduceOp, UnaryOp};
pub use scalar::{Scalar, ScalarOperand, ScalarOrArray};
pub use shape::Flags;
pub use transforms::{grad, jvp, value_and_grad, vjp, vmap};
pub use view::SliceArg;

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
