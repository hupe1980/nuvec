//! Reading and writing arrays: safetensors (MLX's checkpoint format) and NumPy's `.npy`,
//! loaded without a copy by memory-mapping the file, and the IDX format (MNIST).
//!
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

mod idx;
mod json;
mod npy;
mod safetensors;

pub use idx::{parse_idx, read_idx};
pub use npy::{load_npy, save_npy};
pub use safetensors::{SafeTensors, load_safetensors, save_safetensors};

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
