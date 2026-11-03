//! Shared events: GPU work that waits for a value the CPU signals (MLX's `metal/event.cpp`).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLSharedEvent};

use super::Device;
use crate::{GpuError, Result};

/// How long [`Event::wait`] spins before it blocks.
const SPIN: std::time::Duration = std::time::Duration::from_micros(50);

/// A counter command buffers wait on
/// ([`CommandBuffer::wait_event`](super::CommandBuffer::wait_event)) and the CPU raises
/// ([`signal`](Self::signal)). Waiters for a value run once the counter reaches it.
pub struct Event {
    raw: Retained<ProtocolObject<dyn MTLSharedEvent>>,
}

// SAFETY: MTLSharedEvent is thread-safe (its value is read and set from any thread, and
// command buffers on any queue wait on it); the handle is only retained and released.
unsafe impl Send for Event {}
// SAFETY: as above; every method takes `&self` and is safe to call concurrently.
unsafe impl Sync for Event {}

impl Event {
    /// A new event on `device`, at value 0.
    pub fn new(device: &Device) -> Result<Self> {
        let raw = device
            .raw()
            .newSharedEvent()
            .ok_or_else(|| GpuError::CommandBuffer("cannot create a shared event".into()))?;
        Ok(Self { raw })
    }

    /// Raises the counter to `value`, releasing GPU work that waits for it.
    pub fn signal(&self, value: u64) {
        self.raw.setSignaledValue(value);
    }

    /// Waits on the CPU until the counter reaches `value`, for at most `timeout`; whether it
    /// did. Spins briefly first: a collective's input is usually ready within microseconds.
    pub fn wait(&self, value: u64, timeout: std::time::Duration) -> bool {
        let spin = std::time::Instant::now();
        while spin.elapsed() < SPIN.min(timeout) {
            if self.value() >= value {
                return true;
            }
            std::hint::spin_loop();
        }
        let ms = timeout.saturating_sub(SPIN).as_millis().max(1) as u64;
        self.raw.waitUntilSignaledValue_timeoutMS(value, ms)
    }

    /// The counter.
    pub fn value(&self) -> u64 {
        self.raw.signaledValue()
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.raw
    }
}

impl std::fmt::Debug for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Event")
            .field("value", &self.value())
            .finish()
    }
}
