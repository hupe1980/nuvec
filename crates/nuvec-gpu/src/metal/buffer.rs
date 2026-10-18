//! Shared-memory buffers.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use super::Device;
use super::allocator::{Pool, Raw};
use super::command::Fence;
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
    /// Submitted-but-unfinished GPU uses of this buffer.
    in_flight: Arc<Guard>,
}

// SAFETY: Metal buffer objects may be retained, released and bound from any thread. The only
// non-thread-safe part is the memory behind `contents()`, and access to it is synchronised by
// Rust borrows (`contents_mut` needs `&mut self`; GPU writes need `&mut Buffer`) plus the
// `in_flight` guard, which refuses CPU reads while GPU writes are pending and CPU writes while
// any GPU use is pending.
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
            in_flight: Arc::default(),
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

    /// Whether submitted GPU work still uses (reads or writes) the buffer.
    pub fn in_use(&self) -> bool {
        self.in_flight.users.load(Ordering::Acquire) != 0
    }

    /// The bytes, if no submitted GPU work writes the buffer (concurrent GPU reads are fine).
    pub fn contents(&self) -> Result<&[u8]> {
        if self.in_flight.writers.load(Ordering::Acquire) != 0 {
            return Err(GpuError::BufferInUse);
        }
        if self.raw.is_none() {
            return Ok(&[]);
        }
        // SAFETY: `ptr` is `raw.contents()`, at least `len` bytes (the allocation is at least the
        // rounded size) of CPU-visible shared storage that lives as long as `raw`, which this
        // borrow keeps alive. No submitted GPU work writes the buffer (checked above), so any
        // concurrent GPU access only reads; new GPU writes need `&mut self`, which this shared
        // borrow excludes.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) })
    }

    /// The bytes, mutably, if no submitted GPU work uses (reads or writes) the buffer.
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

    pub(crate) fn guard(&self) -> Arc<Guard> {
        self.in_flight.clone()
    }

    /// Identity for hazard tracking (unique per live buffer, including empty ones).
    pub(crate) fn id(&self) -> usize {
        Arc::as_ptr(&self.in_flight) as usize
    }
}

/// Counts submitted GPU uses of a buffer: all uses, and the writing ones; and the fences of
/// the ordered command buffers using it.
#[derive(Debug, Default)]
pub(crate) struct Guard {
    users: AtomicUsize,
    writers: AtomicUsize,
    fences: Mutex<Fences>,
}

/// The fences of ordered command buffers that use a buffer, while any use is pending.
#[derive(Debug, Default)]
struct Fences {
    /// The last writer's.
    writer: Option<Fence>,
    /// Every user's since the buffer was last idle.
    users: Vec<Fence>,
}

impl Guard {
    /// The fences to wait for before using the buffer: the pending writer's for a read, and
    /// every pending user's for a write.
    pub(crate) fn pending_fences(&self, write: bool) -> Vec<Fence> {
        if self.users.load(Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let fences = self.fences.lock().unwrap_or_else(PoisonError::into_inner);
        if write {
            fences.users.clone()
        } else if self.writers.load(Ordering::Acquire) != 0 {
            fences.writer.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Records the fence of a command buffer that uses the buffer (before it is committed).
    pub(crate) fn record(&self, fence: &Fence, write: bool) {
        let mut fences = self.fences.lock().unwrap_or_else(PoisonError::into_inner);
        if write {
            fences.writer = Some(fence.clone());
        }
        fences.users.push(fence.clone());
    }

    pub(crate) fn acquire(&self, write: bool) {
        self.users.fetch_add(1, Ordering::AcqRel);
        if write {
            self.writers.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub(crate) fn release(&self, write: bool) {
        if write {
            self.writers.fetch_sub(1, Ordering::AcqRel);
        }
        if self.users.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Idle: forget the fences, unless a new use began meanwhile (checked under the
            // lock that `record` takes, so a new user's fence is never dropped).
            let mut fences = self.fences.lock().unwrap_or_else(PoisonError::into_inner);
            if self.users.load(Ordering::Acquire) == 0 {
                *fences = Fences::default();
            }
        }
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
