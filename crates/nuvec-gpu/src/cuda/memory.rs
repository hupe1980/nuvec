//! Managed memory (`cuMemAllocManaged`) through a size-class cache, as MLX's CUDA
//! allocator sizes its blocks (powers of two below 16 KiB, then 16 KiB pages). The host
//! reads and writes the same bytes (after the stream is synchronised), so CPU kernels need no
//! copies; the driver migrates pages, and a prefetch moves them to the GPU before kernels.

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use cuda_core::sys;

use super::device::Device;
use super::error::{CudaError, Result, check};

const PAGE: usize = 16 << 10;

/// Use and limits of the allocator, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct MemoryStats {
    /// Bytes in live buffers (rounded sizes).
    pub active: usize,
    /// The highest `active` since creation or the last reset.
    pub peak: usize,
    /// Bytes held in the cache for reuse.
    pub cache: usize,
    /// Allocations that would push `active` past this fail with out-of-memory.
    pub memory_limit: usize,
    /// The cache is trimmed to at most this many bytes.
    pub cache_limit: usize,
    /// Blocks obtained from the driver (cache misses) so far.
    pub system_allocations: u64,
}

#[derive(Default)]
struct State {
    free: HashMap<usize, Vec<sys::CUdeviceptr>>,
    stats: MemoryStats,
}

/// The device's allocator.
pub(crate) struct Pool {
    state: Mutex<State>,
    /// The context every allocation belongs to, made current on the calling thread before
    /// each driver call (evaluations run on any thread).
    ctx: sys::CUcontext,
}

// SAFETY: a driver context handle; the driver allows any thread to make it current.
unsafe impl Send for Pool {}
// SAFETY: as above; the state is behind a mutex.
unsafe impl Sync for Pool {}

/// The size class of `n` bytes.
fn rounded(n: usize) -> usize {
    match n {
        0..=8 => 8,
        n if n < PAGE => n.next_power_of_two(),
        n => n.div_ceil(PAGE) * PAGE,
    }
}

impl Pool {
    pub(crate) fn new(total: usize, ctx: sys::CUcontext) -> Self {
        let limit = total / 20 * 19;
        Self {
            ctx,
            state: Mutex::new(State {
                stats: MemoryStats {
                    memory_limit: limit,
                    cache_limit: limit,
                    ..Default::default()
                },
                ..Default::default()
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // Driver calls follow under the lock: the context must be current on this thread.
        // SAFETY: a live context of this device.
        unsafe { sys::cuCtxSetCurrent(self.ctx) };
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn alloc(&self, len: usize) -> Result<(sys::CUdeviceptr, usize)> {
        let size = rounded(len);
        let mut s = self.lock();
        if s.stats.active + size > s.stats.memory_limit {
            return Err(CudaError::OutOfMemory(len));
        }
        let ptr = match s.free.get_mut(&size).and_then(Vec::pop) {
            Some(p) => {
                s.stats.cache -= size;
                p
            }
            None => {
                let mut p: sys::CUdeviceptr = 0;
                // SAFETY: an out-parameter and a positive size; global attachment makes the
                // memory accessible from every stream and the host.
                let r = unsafe {
                    sys::cuMemAllocManaged(
                        &mut p,
                        size,
                        sys::CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL,
                    )
                };
                if r != sys::cudaError_enum_CUDA_SUCCESS {
                    // Give the cache back to the driver and try once more.
                    Self::release(&mut s, usize::MAX);
                    // SAFETY: as above.
                    let r = unsafe {
                        sys::cuMemAllocManaged(
                            &mut p,
                            size,
                            sys::CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL,
                        )
                    };
                    if r == sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY {
                        return Err(CudaError::OutOfMemory(len));
                    }
                    check("cuMemAllocManaged", r)?;
                }
                s.stats.system_allocations += 1;
                p
            }
        };
        s.stats.active += size;
        s.stats.peak = s.stats.peak.max(s.stats.active);
        Ok((ptr, size))
    }

    fn free(&self, ptr: sys::CUdeviceptr, size: usize) {
        let mut s = self.lock();
        s.stats.active -= size;
        s.stats.cache += size;
        s.free.entry(size).or_default().push(ptr);
        let excess = s.stats.cache.saturating_sub(s.stats.cache_limit);
        if excess > 0 {
            Self::release(&mut s, excess);
        }
    }

    /// Frees cached blocks, the largest first, until `bytes` are released.
    fn release(s: &mut State, bytes: usize) {
        let mut freed = 0;
        let mut sizes: Vec<usize> = s.free.keys().copied().collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        for size in sizes {
            while freed < bytes {
                let Some(p) = s.free.get_mut(&size).and_then(Vec::pop) else {
                    break;
                };
                // SAFETY: a block this pool allocated, not in use (it is in the cache).
                unsafe { sys::cuMemFree_v2(p) };
                s.stats.cache -= size;
                freed += size;
            }
        }
    }

    pub(crate) fn stats(&self) -> MemoryStats {
        self.lock().stats
    }

    pub(crate) fn set_memory_limit(&self, bytes: usize) -> usize {
        std::mem::replace(&mut self.lock().stats.memory_limit, bytes)
    }

    pub(crate) fn set_cache_limit(&self, bytes: usize) -> usize {
        let mut s = self.lock();
        let old = std::mem::replace(&mut s.stats.cache_limit, bytes);
        if s.stats.cache > bytes {
            let excess = s.stats.cache - bytes;
            Self::release(&mut s, excess);
        }
        old
    }

    pub(crate) fn clear_cache(&self) {
        Self::release(&mut self.lock(), usize::MAX);
    }

    pub(crate) fn reset_peak(&self) {
        let mut s = self.lock();
        s.stats.peak = s.stats.active;
    }
}

/// A managed-memory buffer of the shared device; freed into the cache on drop.
pub struct Buffer {
    ptr: sys::CUdeviceptr,
    len: usize,
    size: usize,
    device: &'static Device,
}

// SAFETY: a plain allocation handle; access is ordered by the caller (stream order on the
// GPU, synchronisation before host access).
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer({:#x}, {} bytes)", self.ptr, self.len)
    }
}

impl Buffer {
    /// `len` bytes of uninitialised managed memory.
    pub fn new(device: &'static Device, len: usize) -> Result<Self> {
        let (ptr, size) = device.pool.alloc(len)?;
        Ok(Self {
            ptr,
            len,
            size,
            device,
        })
    }

    /// The device address.
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }

    /// Its length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes, for the host. The caller must have synchronised the stream after any GPU
    /// work that writes them.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: managed memory is host-accessible at the same address; `len` bytes were
        // allocated; GPU writes are complete when the caller synchronised (documented).
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// The bytes, writable from the host (same precondition as [`Self::as_slice`], and no
    /// GPU work may read them until the host is done).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_slice`, and `&mut self` makes the host access exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    /// Asks the driver to move the pages to the GPU ahead of kernels (a hint; ordered on the
    /// stream).
    pub fn prefetch(&self) -> Result<()> {
        // SAFETY: a plain C struct; every bit pattern (zero) is valid.
        let mut location: sys::CUmemLocation = unsafe { std::mem::zeroed() };
        location.type_ = sys::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE;
        location.__bindgen_anon_1.id = self.device.device.cu_device();
        self.device
            .device
            .bind_to_thread()
            .map_err(super::error::driver("cuCtxSetCurrent"))?;
        // SAFETY: a live managed allocation of `len` bytes and the device's stream.
        check("cuMemPrefetchAsync", unsafe {
            sys::cuMemPrefetchAsync_v2(
                self.ptr,
                self.len,
                location,
                0,
                self.device.stream.cu_stream(),
            )
        })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device.pool.free(self.ptr, self.size);
    }
}

impl Device {
    /// The allocator's use and limits.
    pub fn memory(&self) -> MemoryStats {
        self.pool.stats()
    }

    /// Sets the memory limit and returns the previous one (MLX's `set_memory_limit`).
    pub fn set_memory_limit(&self, bytes: usize) -> usize {
        self.pool.set_memory_limit(bytes)
    }

    /// Sets the cache limit, trimming the cache, and returns the previous one.
    pub fn set_cache_limit(&self, bytes: usize) -> usize {
        self.pool.set_cache_limit(bytes)
    }

    /// Frees every cached block.
    pub fn clear_cache(&self) {
        self.pool.clear_cache();
    }

    /// Resets the peak to the current use.
    pub fn reset_peak(&self) {
        self.pool.reset_peak();
    }
}
