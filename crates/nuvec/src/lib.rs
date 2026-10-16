//! A pure-Rust array framework for Apple Silicon in the spirit of
//! [MLX](https://github.com/ml-explore/mlx): lazy arrays on unified memory and
//! composable function transforms, with no bindings to any C/C++ library.
//!
//! ```
//! use nuvec::{Array, DType, Result, arange};
//!
//! # fn main() -> Result<()> {
//! let x = arange(0.0, 6.0, 1.0, DType::F32)?.reshape(&[2, 3])?; // [[0,1,2],[3,4,5]]
//! let col = Array::from_slice(&[10i32, 20], &[2, 1])?;          // broadcasts over columns
//! let y = (&x + &col).t()?;                                     // lazy, float32, a view
//! assert_eq!(y.shape(), &[3, 2]);
//! assert_eq!(y.to_vec::<f32>()?, vec![10.0, 23.0, 11.0, 24.0, 12.0, 25.0]);
//! # Ok(())
//! # }
//! ```
//!
//! The crate root re-exports the everyday API; the layers below are available as modules.
//!
//! **Status:** pre-alpha. NumPy/MLX semantics on the CPU: promotion, broadcasting,
//! zero-copy views, element-wise operations, reductions and matmul. Metal and transforms
//! come next.

#![forbid(unsafe_code)]

pub use nuvec_compile as compile;
pub use nuvec_core as core;
pub use nuvec_core::{
    ArgReduceOp, Array, BinaryOp, Category, Complex64, DType, Device, DeviceKind, Element, Error,
    Flags, Primitive, ReduceOp, Result, SliceArg, Stream, UnaryOp, abs, add, all, any, arange,
    argmax, argmin, astype, at_least_float, bf16, broadcast_to, cos, default_stream, divide, equal,
    eval, exp, expand_dims, f16, greater, greater_equal, less, less_equal, log, matmul, max,
    maximum, mean, min, minimum, multiply, negative, not_equal, ones, prod, promote_types, reshape,
    set_default_stream, sigmoid, sin, slice, sqrt, squeeze, subtract, sum, tanh, transpose, where_,
    zeros,
};
pub use nuvec_cpu as cpu;
pub use nuvec_distributed as distributed;
pub use nuvec_gpu as gpu;
pub use nuvec_io as io;

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
        assert_eq!(VERSION, crate::core::VERSION);
    }

    #[test]
    fn root_api_works() {
        let a = Array::from_slice(&[1i32, 2], &[2]).unwrap();
        let b = multiply(&a, &a).unwrap() + &a;
        assert_eq!(b.to_vec::<i32>().unwrap(), vec![2, 6]);
    }
}
