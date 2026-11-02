//! The CUDA runtime (Linux, feature `cuda`), on NVIDIA's cuda-rust: the device, a
//! managed-memory allocator, and launches of the kernels of `nuvec_kernels::cuda` (generated
//! PTX and Tile kernels). The counterpart of [`crate::Device`]'s Metal runtime, with the
//! same role: no CUDA C, NVRTC or vendor libraries (constitution I).

mod device;
mod error;
pub mod kernels;
mod memory;
pub mod tile;

pub use device::{Device, DeviceInfo};
pub use error::{CudaError, Result};
pub use memory::{Buffer, MemoryStats};
pub use nuvec_kernels::cuda::ptx;
