//! Metal backend: device discovery, kernel library and pipelines, shared buffers, and
//! command encoding with minimal barriers.
//!
//! ```no_run
//! use nuvec_gpu::{Arg, Device};
//!
//! # fn main() -> nuvec_gpu::Result<()> {
//! let Some(device) = Device::system_default() else { return Ok(()) }; // no Metal here
//! let add = device.pipeline("nuvec_add_f32")?;
//! let a = device.buffer_from(&1.0f32.to_le_bytes().repeat(4))?;
//! let b = device.buffer_from(&2.0f32.to_le_bytes().repeat(4))?;
//! let mut out = device.new_buffer(16)?;
//! let mut cmd = device.command_buffer()?;
//! cmd.dispatch(&add, &[Arg::read(&a), Arg::read(&b), Arg::write(&mut out), Arg::bytes(&4u32.to_le_bytes())], 4)?;
//! cmd.commit().wait()?;
//! assert_eq!(&out.contents()?[..4], &3.0f32.to_le_bytes());
//! # Ok(())
//! # }
//! ```
//!
//! Buffers live in shared (unified) memory: the CPU reads GPU results in place. CPU access
//! is refused while submitted work uses a buffer, and GPU writes need `&mut Buffer`, so
//! CPU and GPU never race on the same memory.
//!
//! **Status:** pre-alpha. Not yet wired into array evaluation.

mod error;
mod info;

pub use error::{GpuError, Result};
pub use info::DeviceInfo;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use metal::{Arg, Buffer, CommandBuffer, Device, MemoryStats, Pipeline, Submitted};

#[cfg(not(target_os = "macos"))]
mod stub;
#[cfg(not(target_os = "macos"))]
pub use stub::Device;

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
