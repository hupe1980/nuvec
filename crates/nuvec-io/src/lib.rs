//! Reading and writing arrays: safetensors (MLX's checkpoint format) and NumPy's `.npy`,
//! any tree of arrays by path ([`save`], [`load_into`]),
//! the IDX format (MNIST) and a minimal [`json`] parser for model files. Files whose
//! tensors are aligned (files nuvec writes are) are memory-mapped without a copy; others,
//! such as mlx-community checkpoints with unpadded headers, are read once at load, in
//! parallel, without being mapped.
//!
//! **Stability:** no promise when this crate is depended on directly; depend on `nuvec`,
//! whose re-exports are semver-tracked.
//!
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

mod idx;
pub mod json;
mod npy;
mod safetensors;
mod tree;

pub use idx::{parse_idx, read_idx};
pub use npy::{load_npy, save_npy};
pub use safetensors::{
    Footprint, SafeTensors, SaveHandle, load_safetensors, load_safetensors_filtered,
    safetensors_footprint, save_safetensors, save_safetensors_async,
};
pub use tree::{load_into, save};

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
