//! Lazy array graph, primitives, dtypes, streams and evaluation: the core of bessemer.
//!
//! Operations on [`Array`]s record a graph and return immediately. Nothing is computed
//! until an array is evaluated, explicitly or by reading it back:
//!
//! ```
//! use bessemer_core::{Array, Result};
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
//! **Status:** pre-alpha. Element-wise `add` and `multiply` on equal shapes and dtypes,
//! evaluated on the CPU.

#![forbid(unsafe_code)]

mod array;
mod buffer;
mod device;
mod dtype;
mod error;
mod eval;
mod ops;
mod primitive;
mod shape;

pub use array::Array;
pub use device::{Device, DeviceKind, Stream, default_stream, set_default_stream};
pub use dtype::{Category, Complex64, DType, Element, bf16, f16};
pub use error::{Error, Result};
pub use eval::eval;
pub use ops::{add, multiply};
pub use primitive::Primitive;
pub use shape::Flags;

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
