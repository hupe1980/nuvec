//! The Metal device (MLX's `mx.metal`): availability, device facts, the Neural
//! Accelerator kernels' status, GPU captures and the kernel profile. Every function works
//! without a Metal device.

use std::path::Path;

use nuvec_gpu::Device;

pub use nuvec_gpu::{DeviceInfo, NaxReason, NaxStatus};

use crate::{Error, Result};

/// Whether a Metal GPU is available (MLX's `mx.metal.is_available`).
pub fn is_available() -> bool {
    Device::shared().is_some()
}

/// What the GPU is and can do (MLX's `mx.metal.device_info`), or `None` without one.
pub fn device_info() -> Option<&'static DeviceInfo> {
    Device::shared().map(Device::info)
}

/// Whether products run on the Neural Accelerator (NAX) kernels, and why not.
pub fn nax() -> NaxStatus {
    Device::shared().map_or(NaxStatus::Off(NaxReason::NoDevice), Device::nax)
}

/// Starts capturing GPU work into `path`, a `.gputrace` document that does not exist yet
/// (MLX's `mx.metal.start_capture`). Metal allows it only with `MTL_CAPTURE_ENABLED=1` in
/// the environment, or under Xcode.
pub fn start_capture(path: impl AsRef<Path>) -> Result<()> {
    let d = Device::shared()
        .ok_or_else(|| Error::Backend("GPU capture: there is no Metal device".into()))?;
    d.start_capture(path)
        .map_err(|e| Error::Backend(e.to_string()))
}

/// Stops the running capture and writes its document (MLX's `mx.metal.stop_capture`);
/// nothing if none runs.
pub fn stop_capture() {
    if let Some(d) = Device::shared() {
        d.stop_capture();
    }
}

/// Per-kernel GPU times, collected when `NUVEC_PROFILE` is set (to anything but `0`): every dispatch runs alone
/// and is timed, so use the shares to find hotspots and the benchmarks for speed.
pub mod profile {
    /// Whether profiling is on.
    pub fn enabled() -> bool {
        nuvec_gpu::profile::enabled()
    }

    /// The totals so far, `(kernel, dispatches, GPU seconds)`, the most expensive first.
    pub fn report() -> Vec<(String, u64, f64)> {
        nuvec_gpu::profile::report()
    }

    /// Clears the totals.
    pub fn reset() {
        nuvec_gpu::profile::reset();
    }
}
