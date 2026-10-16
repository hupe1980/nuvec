//! Non-macOS targets have no Metal: discovery always reports "no device".

use crate::DeviceInfo;

/// A Metal device. Never available on this target.
#[derive(Debug, Clone)]
pub struct Device(std::convert::Infallible);

impl Device {
    /// Always `None`: Metal exists only on Apple platforms.
    pub fn system_default() -> Option<Device> {
        None
    }

    /// Unreachable: no `Device` can be constructed on this target.
    pub fn info(&self) -> &DeviceInfo {
        match self.0 {}
    }
}
