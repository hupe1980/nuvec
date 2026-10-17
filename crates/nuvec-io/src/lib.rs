//! Reading and writing arrays: the IDX format (MNIST) so far.
//!
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

mod idx;

pub use idx::{parse_idx, read_idx};

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
