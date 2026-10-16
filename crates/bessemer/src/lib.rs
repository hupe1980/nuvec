//! A pure-Rust array framework for Apple Silicon in the spirit of
//! [MLX](https://github.com/ml-explore/mlx): lazy arrays on unified memory and
//! composable function transforms, with no bindings to any C/C++ library.
//!
//! This facade re-exports the layers below it.
//!
//! **Status:** pre-alpha. This crate has no functional API yet.

#![forbid(unsafe_code)]

pub use bessemer_compile as compile;
pub use bessemer_core as core;
pub use bessemer_cpu as cpu;
pub use bessemer_distributed as distributed;
pub use bessemer_gpu as gpu;
pub use bessemer_io as io;

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::VERSION.is_empty());
        assert_eq!(super::VERSION, crate::core::VERSION);
    }
}
