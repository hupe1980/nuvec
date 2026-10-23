//! Devices and streams.

use std::cell::Cell;
use std::fmt;

/// The kind of processor a device is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// The CPU.
    Cpu,
    /// A GPU.
    Gpu,
}

/// A processor that executes array operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    /// The kind of processor.
    pub kind: DeviceKind,
    /// Index among devices of the same kind.
    pub index: u32,
}

impl Device {
    /// The CPU.
    pub const fn cpu() -> Self {
        Self {
            kind: DeviceKind::Cpu,
            index: 0,
        }
    }

    /// The GPU with the given index.
    pub const fn gpu(index: u32) -> Self {
        Self {
            kind: DeviceKind::Gpu,
            index,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            DeviceKind::Cpu => write!(f, "cpu:{}", self.index),
            DeviceKind::Gpu => write!(f, "gpu:{}", self.index),
        }
    }
}

/// An ordered queue of operations on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Stream {
    /// Index among the device's streams.
    pub index: u32,
    /// The device the stream executes on.
    pub device: Device,
}

impl Stream {
    /// The first stream of `device`.
    pub const fn default_for(device: Device) -> Self {
        Self { index: 0, device }
    }
}

thread_local! {
    static DEFAULT_STREAM: Cell<Option<Stream>> = const { Cell::new(None) };
}

/// The default device: the GPU when Metal is available, else the CPU (as in MLX).
pub fn default_device() -> Device {
    if crate::gpu::available() {
        Device::gpu(0)
    } else {
        Device::cpu()
    }
}

/// The stream new operations on this thread are recorded on: the default device's first
/// stream until [`set_default_stream`] is called on this thread.
pub fn default_stream() -> Stream {
    DEFAULT_STREAM.get().unwrap_or_else(|| {
        let stream = Stream::default_for(default_device());
        DEFAULT_STREAM.set(Some(stream));
        stream
    })
}

/// Sets the stream new operations on this thread are recorded on.
pub fn set_default_stream(stream: Stream) {
    DEFAULT_STREAM.set(Some(stream));
}

/// Runs `f` with `stream` as this thread's default stream, and restores the previous
/// default afterwards, also when `f` panics. A device stands for its first stream.
///
/// This replaces per-operation stream arguments: every operation `f` records goes on
/// `stream`.
///
/// ```
/// use nuvec_core::{Device, array, default_stream, with_stream};
///
/// let x = array![1.0, 2.0];
/// let y = with_stream(Device::cpu(), || x.exp().sum(..));
/// assert_eq!(y.device(), Device::cpu());
/// assert_eq!(default_stream(), nuvec_core::default_stream()); // restored
/// ```
pub fn with_stream<R>(stream: impl Into<Stream>, f: impl FnOnce() -> R) -> R {
    let _guard = StreamGuard::set(stream.into());
    f()
}

/// Restores the thread's default stream when dropped, also when unwinding.
pub(crate) struct StreamGuard(Option<Stream>);

impl StreamGuard {
    /// Makes `stream` the default until the guard is dropped.
    pub(crate) fn set(stream: Stream) -> Self {
        let guard = Self(DEFAULT_STREAM.get());
        DEFAULT_STREAM.set(Some(stream));
        guard
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        DEFAULT_STREAM.set(self.0);
    }
}

/// A device's first stream.
impl From<Device> for Stream {
    fn from(device: Device) -> Self {
        Self::default_for(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_the_default_device_and_settable_per_thread() {
        assert_eq!(default_stream().device, default_device());
        let expected = if cfg!(all(target_os = "macos", not(miri))) && crate::gpu::available() {
            Device::gpu(0)
        } else {
            Device::cpu()
        };
        assert_eq!(default_device(), expected);
        set_default_stream(Stream::default_for(Device::cpu()));
        assert_eq!(default_stream().device, Device::cpu());
        std::thread::spawn(|| assert_eq!(default_stream().device, default_device()))
            .join()
            .unwrap();
    }

    #[test]
    fn display() {
        assert_eq!(Device::gpu(1).to_string(), "gpu:1");
    }
}
