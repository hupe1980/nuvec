//! A pure-Rust array framework for Apple Silicon in the spirit of
//! [MLX](https://github.com/ml-explore/mlx): lazy arrays on unified memory and
//! composable function transforms, with no bindings to any C/C++ library.
//!
//! ```
//! use bessemer::{Array, Result};
//!
//! # fn main() -> Result<()> {
//! let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
//! let b = Array::full(&[3], 10.0f32)?;
//! let c = (&a + &b) * &a; // lazy: nothing computed yet
//! assert_eq!(c.to_vec::<f32>()?, vec![11.0, 24.0, 39.0]);
//! # Ok(())
//! # }
//! ```
//!
//! The crate root re-exports the everyday API; the layers below are available as modules.
//!
//! **Status:** pre-alpha. Element-wise `add` and `multiply` on equal shapes and dtypes,
//! evaluated on the CPU.

#![forbid(unsafe_code)]

pub use bessemer_compile as compile;
pub use bessemer_core as core;
pub use bessemer_core::{
    Array, Category, Complex64, DType, Device, DeviceKind, Element, Error, Flags, Primitive,
    Result, Stream, add, bf16, default_stream, eval, f16, multiply, set_default_stream,
};
pub use bessemer_cpu as cpu;
pub use bessemer_distributed as distributed;
pub use bessemer_gpu as gpu;
pub use bessemer_io as io;

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
