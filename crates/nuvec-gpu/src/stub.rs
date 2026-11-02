//! Non-macOS targets have no Metal: discovery always reports "no device".

use crate::{DeviceInfo, MemoryStats};

/// A Metal device. Never available on this target.
#[derive(Debug, Clone)]
pub struct Device(std::convert::Infallible);

impl Device {
    /// Always `None`: Metal exists only on Apple platforms.
    pub fn system_default() -> Option<Device> {
        None
    }

    /// Always `None`: Metal exists only on Apple platforms.
    pub fn shared() -> Option<&'static Device> {
        None
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn info(&self) -> &DeviceInfo {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn kernel_names(&self) -> crate::Result<Vec<String>> {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn nax(&self) -> crate::NaxStatus {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn memory(&self) -> MemoryStats {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn set_wired_limit(&self, _bytes: usize) -> crate::Result<usize> {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn set_memory_limit(&self, _bytes: usize) -> usize {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn set_cache_limit(&self, _bytes: usize) -> usize {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn reset_peak(&self) {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn can_wire(&self) -> bool {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    #[doc(hidden)]
    pub fn __set_residency_set_cap(&self, _bytes: usize) {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn clear_cache(&self) {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn start_capture(&self, _path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        match self.0 {}
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn stop_capture(&self) {
        match self.0 {}
    }
}

/// GPU profiling: never enabled without Metal.
pub mod profile {
    /// Always `false`.
    pub fn enabled() -> bool {
        false
    }

    /// Always empty.
    pub fn report() -> Vec<(String, u64, f64)> {
        Vec::new()
    }

    /// Does nothing.
    pub fn reset() {}
}
