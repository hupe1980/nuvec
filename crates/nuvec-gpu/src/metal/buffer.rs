//! Shared-memory buffers.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::Device;
use crate::{GpuError, Result};

/// GPU memory in unified (shared) storage: the CPU reads and writes it in place.
///
/// Hazard tracking is disabled at the Metal level (as in MLX); nuvec orders accesses itself.
/// CPU access is refused while submitted GPU work uses the buffer, and GPU writes require
/// `&mut Buffer`, so the CPU and the GPU never race on it.
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    /// Number of submitted-but-unwaited command buffers that use this buffer.
    in_flight: Arc<AtomicUsize>,
}

// SAFETY: Metal buffer objects may be retained, released and bound from any thread. The only
// non-thread-safe part is the memory behind `contents()`, and access to it is synchronised by
// Rust borrows (`contents_mut` needs `&mut self`; GPU writes need `&mut Buffer`) plus the
// `in_flight` guard, which refuses CPU access while GPU work is pending.
unsafe impl Send for Buffer {}
// SAFETY: as above; `&Buffer` only allows shared reads of memory no GPU work is writing.
unsafe impl Sync for Buffer {}

impl Device {
    /// A zero-initialised buffer of `len` bytes.
    pub fn new_buffer(&self, len: usize) -> Result<Buffer> {
        let options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
        // Metal rejects zero-length buffers; allocate one byte and expose zero.
        let raw = self
            .raw()
            .newBufferWithLength_options(len.max(1), options)
            .ok_or(GpuError::OutOfMemory(len))?;
        Ok(Buffer {
            raw,
            len,
            in_flight: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// A buffer holding a copy of `bytes`.
    pub fn buffer_from(&self, bytes: &[u8]) -> Result<Buffer> {
        let mut buffer = self.new_buffer(bytes.len())?;
        buffer.contents_mut()?.copy_from_slice(bytes);
        Ok(buffer)
    }
}

impl Buffer {
    /// Size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether submitted GPU work still uses the buffer.
    pub fn in_use(&self) -> bool {
        self.in_flight.load(Ordering::Acquire) != 0
    }

    /// The bytes, if no submitted GPU work uses the buffer.
    pub fn contents(&self) -> Result<&[u8]> {
        if self.in_use() {
            return Err(GpuError::BufferInUse);
        }
        // SAFETY: `contents()` points to at least `len` bytes of CPU-visible shared storage
        // that lives as long as `self.raw`, which this borrow keeps alive. No GPU work uses
        // the buffer (checked above), and new GPU writes need `&mut self`, which this shared
        // borrow excludes.
        Ok(unsafe {
            std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<u8>(), self.len)
        })
    }

    /// The bytes, mutably, if no submitted GPU work uses the buffer.
    pub fn contents_mut(&mut self) -> Result<&mut [u8]> {
        if self.in_use() {
            return Err(GpuError::BufferInUse);
        }
        // SAFETY: as in `contents`; additionally `&mut self` is exclusive, and `Buffer` is not
        // `Clone`, so no other CPU reference to these bytes exists.
        Ok(unsafe {
            std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast::<u8>(), self.len)
        })
    }

    pub(crate) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    pub(crate) fn retained(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.raw.clone()
    }

    pub(crate) fn guard(&self) -> Arc<AtomicUsize> {
        self.in_flight.clone()
    }

    /// Identity for hazard tracking.
    pub(crate) fn id(&self) -> usize {
        Retained::as_ptr(&self.raw).cast::<()>() as usize
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("len", &self.len)
            .field("in_use", &self.in_use())
            .finish()
    }
}
