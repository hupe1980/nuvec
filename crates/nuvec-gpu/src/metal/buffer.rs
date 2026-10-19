//! Shared-memory buffers.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

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
    /// For memory reused while the previous owner's committed GPU work still used it: that
    /// work's uses. GPU work on this buffer waits for their fences; CPU access waits until
    /// they complete.
    previous: Option<Arc<Guard>>,
    /// Memory of a mapped file (not the allocator's): released with the Metal buffer.
    mapped: bool,
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
        self.alloc_buffer(len, false)
    }

    /// A buffer of `len` bytes for GPU work to write: as [`new_buffer`](Self::new_buffer),
    /// but it may be memory that committed GPU work still uses (as MLX reuses memory). GPU
    /// work encoded with it is ordered after those uses; the CPU cannot access it until
    /// they complete ([`Buffer::in_use`]).
    pub fn new_gpu_buffer(&self, len: usize) -> Result<Buffer> {
        self.alloc_buffer(len, true)
    }

    fn alloc_buffer(&self, len: usize, gpu: bool) -> Result<Buffer> {
        let pool = self.pool().clone();
        let (raw, size, busy) = match pool.alloc(self.raw(), len, gpu)? {
            Some((raw, size, busy)) => (Some(raw), size, busy),
            None => (None, 0, None),
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
            previous: busy,
            mapped: false,
        })
    }

    /// The file at `path`, memory-mapped read-only and wrapped without a copy
    /// (`newBufferWithBytesNoCopy`): its pages are the page cache's, read from disk when
    /// first touched by the CPU or the GPU. The buffer cannot be written (CPU writes and GPU
    /// writes are refused). The mapping is released when the buffer and every GPU use of
    /// it are gone.
    ///
    /// A shared read-only mapping is used because a private (copy-on-write) one costs
    /// about 4× more to fault in on first use.
    pub fn map_file(&self, path: &std::path::Path) -> Result<Buffer> {
        use std::os::fd::AsRawFd;
        let io = |e: std::io::Error| GpuError::Io(format!("{}: {e}", path.display()));
        let file = std::fs::File::open(path).map_err(io)?;
        let len = usize::try_from(file.metadata().map_err(io)?.len())
            .map_err(|_| GpuError::Io(format!("{}: too large", path.display())))?;
        if len == 0 {
            return self.new_buffer(0);
        }
        // SAFETY: `sysconf` has no preconditions.
        let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(16384);
        let map_len = len.div_ceil(page) * page;
        // SAFETY: a fresh read-only mapping of an open file descriptor; the result is
        // checked below. The descriptor may be closed afterwards (the mapping keeps the file).
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io(std::io::Error::last_os_error()));
        }
        let start = NonNull::new(ptr.cast::<u8>()).expect("mmap returned a non-null address");
        let unmap = block2::RcBlock::new(move |p: NonNull<std::ffi::c_void>, n: usize| {
            // SAFETY: Metal calls this once, with the mapping created above, after the
            // buffer is released (so neither the CPU nor the GPU uses it any more).
            unsafe { libc::munmap(p.as_ptr(), n) };
        });
        let options = objc2_metal::MTLResourceOptions::StorageModeShared
            | objc2_metal::MTLResourceOptions::HazardTrackingModeUntracked;
        // SAFETY: the pointer is page-aligned (from `mmap`), valid for `map_len` bytes, and
        // owned by the buffer from now on: the deallocator unmaps it exactly once.
        let raw = unsafe {
            self.raw()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    start.cast(),
                    map_len,
                    options,
                    Some(&unmap),
                )
        };
        let Some(raw) = raw else {
            // SAFETY: Metal did not take the mapping, so it is still ours to release.
            unsafe { libc::munmap(ptr, map_len) };
            return Err(GpuError::OutOfMemory(len));
        };
        self.pool().track(&raw);
        Ok(Buffer {
            raw: Some(raw),
            ptr: start,
            len,
            size: 0,
            pool: self.pool().clone(),
            in_flight: Arc::default(),
            previous: None,
            mapped: true,
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

    /// Waits until the previous owner's GPU work on reused memory has completed (it is
    /// committed, so it does).
    fn wait_for_previous(&self) {
        if let Some(previous) = &self.previous {
            while !previous.idle() {
                std::thread::yield_now();
            }
        }
    }

    /// The bytes, if no submitted GPU work writes the buffer (concurrent GPU reads are fine).
    pub fn contents(&self) -> Result<&[u8]> {
        self.wait_for_previous();
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

    /// The bytes, mutably, if no submitted GPU work uses (reads or writes) the buffer and
    /// it is not a read-only mapped file.
    pub fn contents_mut(&mut self) -> Result<&mut [u8]> {
        if self.mapped {
            return Err(GpuError::ReadOnly);
        }
        if self.in_use() {
            return Err(GpuError::BufferInUse);
        }
        self.wait_for_previous();
        self.previous = None;
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

    /// Whether the buffer is a read-only mapped file.
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    pub(crate) fn guard(&self) -> Arc<Guard> {
        self.in_flight.clone()
    }

    /// The fences GPU work using the buffer waits for: its own pending uses' and, for reused
    /// memory, the previous owner's.
    pub(crate) fn pending_fences(&self, write: bool) -> Vec<Fence> {
        let mut fences = self.in_flight.pending_fences(write);
        if let Some(previous) = &self.previous {
            fences.extend(previous.pending_fences(true));
        }
        fences
    }

    /// Identity for hazard tracking (unique per live buffer, including empty ones).
    pub(crate) fn id(&self) -> usize {
        Arc::as_ptr(&self.in_flight) as usize
    }
}

/// Counts submitted GPU uses of a buffer: all uses, the writing ones and those not yet
/// committed; and the fences of the ordered command buffers using it. While uses are
/// pending it stays with the memory, also when the memory is cached and reused.
#[derive(Default)]
pub(crate) struct Guard {
    users: AtomicUsize,
    writers: AtomicUsize,
    /// Uses encoded in command buffers not committed yet: their fences cannot order other
    /// work yet.
    uncommitted: AtomicUsize,
    fences: Mutex<Fences>,
}

impl std::fmt::Debug for Guard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Guard")
            .field("users", &self.users)
            .field("writers", &self.writers)
            .finish_non_exhaustive()
    }
}

/// The fences of ordered command buffers whose uses of a buffer are pending.
#[derive(Debug, Default)]
struct Fences {
    /// The last writer's.
    writer: Option<Fence>,
    /// Every pending use's (a command buffer's fence leaves when it completes, so a buffer
    /// that is never idle, such as weights read by every pipelined step, keeps only the
    /// fences of work still in flight).
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

    /// Whether no submitted GPU work uses the buffer.
    pub(crate) fn idle(&self) -> bool {
        self.users.load(Ordering::Acquire) == 0
    }

    /// Whether every pending use is committed (so its fence orders later work after it).
    pub(crate) fn committed_uses(&self) -> bool {
        self.uncommitted.load(Ordering::Acquire) == 0
    }

    /// A use's command buffer was committed (or abandoned).
    pub(crate) fn committed(&self) {
        self.uncommitted.fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn acquire(&self, write: bool) {
        self.uncommitted.fetch_add(1, Ordering::AcqRel);
        self.users.fetch_add(1, Ordering::AcqRel);
        if write {
            self.writers.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// A use's command buffer completed (or was abandoned); `fence` is its fence, if it
    /// recorded one: nothing has to wait for that work any more.
    pub(crate) fn release(&self, write: bool, fence: Option<&Fence>) {
        if let Some(fence) = fence {
            let mut fences = self.fences.lock().unwrap_or_else(PoisonError::into_inner);
            let same = |f: &Fence| Retained::as_ptr(f) == Retained::as_ptr(fence);
            if let Some(i) = fences.users.iter().position(same) {
                fences.users.swap_remove(i);
            }
            if fences.writer.as_ref().is_some_and(same) {
                fences.writer = None;
            }
        }
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
            drop(fences);
        }
    }
}

impl Drop for Buffer {
    /// Returns the memory to the allocator's cache at once, with the hazard state of GPU
    /// work that still uses it (only GPU work reuses it before that work completes).
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            if self.mapped {
                // Metal unmaps the file once the last reference (ours or a command
                // buffer's) is released.
                self.pool.untrack(&raw);
                drop(raw);
                return;
            }
            // Its own GPU work runs after the previous owner's: while that is pending, it
            // orders reuse; otherwise the previous owner's (if pending) does.
            let busy = if self.in_use() {
                Some(self.in_flight.clone())
            } else {
                self.previous.take().filter(|p| !p.idle())
            };
            self.pool.free(raw, self.size, busy);
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

#[cfg(test)]
mod tests {
    use objc2_metal::MTLDevice;

    use super::*;

    /// A buffer that is never idle (as weights read by every pipelined step) keeps only
    /// the fences of uses still pending, not one per command buffer since it was idle.
    #[test]
    fn completed_fences_leave_buffers_that_stay_in_use() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let new_fence = || device.raw().newFence().expect("a fence");
        let guard = Guard::default();
        // A long-running use keeps the buffer busy throughout.
        let long = new_fence();
        guard.acquire(false);
        guard.record(&long, false);
        guard.committed();
        for _ in 0..1000 {
            let fence = new_fence();
            guard.acquire(true);
            guard.record(&fence, true);
            guard.committed();
            guard.release(true, Some(&fence));
        }
        let fences = guard.fences.lock().unwrap();
        assert_eq!(fences.users.len(), 1);
        assert!(fences.writer.is_none());
        drop(fences);
        guard.release(false, Some(&long));
        assert!(guard.idle());
    }
}
