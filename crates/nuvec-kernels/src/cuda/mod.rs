//! CUDA kernels (feature `cuda`), in Rust (constitution I): PTX that nuvec generates for
//! element-wise and reduction kernels, and Tile kernels (cutile, Linux) for matrix
//! products, JIT-compiled through the toolkit's `tileiras`. The runtime that launches them
//! is `nuvec_gpu::cuda`.

pub mod ptx;
#[cfg(target_os = "linux")]
pub mod tile;
