//! Shared-memory buffers.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use super::Device;
use super::allocator::{Pool, Raw};
use crate::{GpuError, Result};

/// GPU memory in unified (shared) storage: the CPU reads and writes it in place.
///
/// Buffers come from the device's caching allocator and return to it when dropped. Hazard
/// tracking is disabled at the Metal level (as in MLX); nuvec orders accesses itself. CPU
/// access is refused while submitted GPU work uses the buffer, and GPU writes require
/// `&mut Buffer`, so the CPU and the GPU never race on it.
pub struct Buffer {
    /// `None` for zero-length buffers, which own no Metal memory.
    raw: Option<Raw>,
    /// `raw.contents()`, fixed for the buffer's lifetime (cached: each lookup is an
    /// Objective-C message).
    ptr: NonNull<u8>,
    len: usize,
    /// Accounted (rounded) size in the pool.
    size: usize,
    pool: Arc<Pool>,
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
    /// A buffer of `len` bytes from the caching allocator. Its contents are unspecified
    /// (a reused buffer holds stale bytes); fails with [`GpuError::OutOfMemory`] beyond the
    /// memory limit.
    pub fn new_buffer(&self, len: usize) -> Result<Buffer> {
        let pool = self.pool().clone();
        let (raw, size) = match pool.alloc(self.raw(), len)? {
            Some((raw, size)) => (Some(raw), size),
            None => (None, 0),
        };
        let ptr = raw
            .as_ref()
            .map_or(NonNull::dangling(), |r| r.contents().cast());
        Ok(Buffer {
            raw,
            ptr,
            len,
            size,
            pool,
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
        if self.raw.is_none() {
            return Ok(&[]);
        }
        // SAFETY: `ptr` is `raw.contents()`, at least `len` bytes (the allocation is at least the
        // rounded size) of CPU-visible shared storage that lives as long as `raw`, which this
        // borrow keeps alive. No GPU work uses the buffer (checked above), and new GPU writes
        // need `&mut self`, which this shared borrow excludes.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) })
    }

    /// The bytes, mutably, if no submitted GPU work uses the buffer.
    pub fn contents_mut(&mut self) -> Result<&mut [u8]> {
        if self.in_use() {
            return Err(GpuError::BufferInUse);
        }
        if self.raw.is_none() {
            return Ok(&mut []);
        }
        // SAFETY: as in `contents`; additionally `&mut self` is exclusive, and `Buffer` is not
        // `Clone`, so no other CPU reference to these bytes exists.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) })
    }

    /// The Metal buffer to bind: the device's placeholder for empty buffers.
    pub(crate) fn raw(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.raw
            .as_deref()
            .or_else(|| self.pool.placeholder().map(|r| &**r))
    }

    pub(crate) fn retained(&self) -> Option<Raw> {
        self.raw
            .clone()
            .or_else(|| self.pool.placeholder().cloned())
    }

    pub(crate) fn guard(&self) -> Arc<AtomicUsize> {
        self.in_flight.clone()
    }

    /// Identity for hazard tracking (unique per live buffer, including empty ones).
    pub(crate) fn id(&self) -> usize {
        Arc::as_ptr(&self.in_flight) as usize
    }
}

impl Drop for Buffer {
    /// Returns the memory to the allocator's cache, unless GPU work still uses it (then it
    /// is released when the GPU finishes, never reused early).
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            let reusable = !self.in_use();
            self.pool.free(raw, self.size, reusable);
        }
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
