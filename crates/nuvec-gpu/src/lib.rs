//! Metal backend: device discovery, kernel library and pipelines, shared buffers, and
//! command encoding with minimal barriers.
//!
//! ```no_run
//! # #[cfg(target_os = "macos")]
//! # fn main() -> nuvec_gpu::Result<()> {
//! use nuvec_gpu::{Arg, Device};
//!
//! let Some(device) = Device::system_default() else { return Ok(()) }; // no Metal here
//! let add = device.pipeline("nuvec_add_f32")?;
//! let a = device.buffer_from(&1.0f32.to_le_bytes().repeat(4))?;
//! let b = device.buffer_from(&2.0f32.to_le_bytes().repeat(4))?;
//! let mut out = device.new_buffer(16)?;
//! let mut cmd = device.command_buffer()?;
//! let n = 4u32.to_le_bytes();
//! let args = [Arg::read(&a), Arg::read(&b), Arg::write(&mut out), Arg::bytes(&n)];
//! // SAFETY: `nuvec_add_f32` touches element `i < n` of each buffer; all hold 4 floats.
//! unsafe { cmd.dispatch(&add, &args, 4)? };
//! cmd.commit().wait()?;
//! assert_eq!(&out.contents()?[..4], &3.0f32.to_le_bytes());
//! # Ok(())
//! # }
//! # #[cfg(not(target_os = "macos"))]
//! # fn main() {} // Metal exists only on Apple platforms
//! ```
//!
//! Buffers live in shared (unified) memory: the CPU reads GPU results in place. CPU access
//! is refused while submitted work uses a buffer, and GPU writes need `&mut Buffer`, so
//! CPU and GPU never race on the same memory. Dispatching is `unsafe`: nuvec validates the
//! arguments and orders the work, but cannot check that a kernel stays within its buffers.
//!
//! Requires macOS 15 or later (Metal 3.2, as the kernel library is built).
//!
//! Arrays evaluated on the GPU (`nuvec-core`) use the process-wide [`Device::shared`]; its
//! allocator's limits and statistics are set and read there.
//!
//! **Stability:** no promise when this crate is depended on directly; depend on `nuvec`,
//! whose re-exports are semver-tracked.

mod error;
mod info;
mod memory;

pub use error::{GpuError, Result};
pub use info::DeviceInfo;
pub use memory::{MemoryStats, allocation_size, mapping_size};

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use metal::{
    Arg, Buffer, CommandBuffer, Constant, Device, Pending, Pipeline, Submitted, Timeline, profile,
};

#[cfg(not(target_os = "macos"))]
mod stub;
#[cfg(not(target_os = "macos"))]
pub use stub::{Device, profile};

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_set() {
        assert!(!super::VERSION.is_empty());
        let _ = nuvec_kernels::MODE;
    }
}
