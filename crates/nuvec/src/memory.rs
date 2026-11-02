//! Memory statistics and limits (MLX's `mx.get_active_memory`, `mx.set_memory_limit`, …),
//! in bytes, of the shared Metal device's allocator. Without a Metal device the
//! statistics are zero and setting a limit returns 0 and changes nothing, as MLX's CPU
//! build.

use nuvec_gpu::Device;

pub use nuvec_gpu::MemoryStats;

use crate::{Error, Result};

fn device() -> Option<&'static Device> {
    Device::shared()
}

/// The machine's physical memory in bytes, with or without a GPU (0 where the OS does
/// not say): budgets work the same in CPU-only builds and tests.
pub fn physical_memory() -> u64 {
    nuvec_gpu::physical_memory()
}

/// Every statistic and limit at once.
pub fn stats() -> MemoryStats {
    device().map(Device::memory).unwrap_or_default()
}

/// Memory held by live arrays, memory-mapped checkpoints included (MLX's
/// `get_active_memory`, which counts the weights MLX copies).
pub fn active_memory() -> usize {
    stats().active_with_mapped
}

/// The highest [`active_memory`] since the start or the last [`reset_peak_memory`] (MLX's
/// `get_peak_memory`).
pub fn peak_memory() -> usize {
    stats().peak_with_mapped
}

/// Memory held in the allocator's cache for reuse (MLX's `get_cache_memory`).
pub fn cache_memory() -> usize {
    stats().cache
}

/// Resets the peak to the current use (MLX's `reset_peak_memory`).
pub fn reset_peak_memory() {
    if let Some(d) = device() {
        d.reset_peak();
    }
}

/// Sets the memory limit and returns the previous one (MLX's `set_memory_limit`).
/// Allocations and mappings that would pass it fail with an out-of-memory error
/// ([`Error::is_oom`]); MLX treats its limit as a guideline.
pub fn set_memory_limit(bytes: usize) -> usize {
    device().map_or(0, |d| d.set_memory_limit(bytes))
}

/// Sets the cache limit, trimming the cache to it, and returns the previous one (MLX's
/// `set_cache_limit`).
pub fn set_cache_limit(bytes: usize) -> usize {
    device().map_or(0, |d| d.set_cache_limit(bytes))
}

/// Sets the wired limit and returns the previous one (MLX's `set_wired_limit`): memory
/// kept resident while GPU work runs. Above the device's recommended working set it is
/// an error.
pub fn set_wired_limit(bytes: usize) -> Result<usize> {
    let Some(d) = device() else {
        return Ok(0);
    };
    d.set_wired_limit(bytes).map_err(|e| match e {
        nuvec_gpu::GpuError::InvalidArgument(msg) => Error::InvalidArgument(msg),
        e => Error::Backend(e.to_string()),
    })
}

/// Whether memory can be wired on this system (residency sets: a Metal 3 GPU).
pub fn can_wire() -> bool {
    device().is_some_and(Device::can_wire)
}

/// Releases every cached buffer (MLX's `clear_cache`).
pub fn clear_cache() {
    if let Some(d) = device() {
        d.clear_cache();
    }
}
