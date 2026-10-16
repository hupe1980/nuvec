//! The buffer cache: MLX's Metal allocator rules (`mlx/backend/metal/allocator.cpp`).
//!
//! Freed buffers are cached and handed back for similar sizes, so steady-state loops make
//! almost no system allocations. The cache is bounded (least-recently-used eviction), and
//! allocations beyond the memory limit fail instead of exhausting shared system RAM.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, MutexGuard, PoisonError};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::{GpuError, Result};

/// The virtual-memory page size on Apple silicon.
pub(crate) const PAGE: usize = 16 * 1024;
/// Granularity of allocations up to one page.
const SMALL: usize = 256;
/// Number of small size classes: 256 B steps up to one page.
const SMALL_CLASSES: usize = PAGE / SMALL + 1;

pub(crate) type Raw = Retained<ProtocolObject<dyn MTLBuffer>>;

/// A cached buffer. Cached buffers are never accessed, only retained and released.
struct Cached(Raw);

// SAFETY: retaining and releasing Metal buffer objects is thread-safe; a cached buffer's
// memory is not touched until it is handed out again, under the pool's lock.
unsafe impl Send for Cached {}
// SAFETY: shared access only retains or binds the buffer (the placeholder); its memory is
// never read or written through a shared reference.
unsafe impl Sync for Cached {}

/// Memory use and limits of a device's allocator, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Bytes in live buffers (rounded allocation sizes).
    pub active: usize,
    /// Highest `active` since creation or the last [`Device::reset_peak`](super::Device::reset_peak).
    pub peak: usize,
    /// Bytes held in the cache for reuse.
    pub cache: usize,
    /// Allocations that would push `active` past this fail.
    pub memory_limit: usize,
    /// The cache is trimmed to at most this many bytes.
    pub cache_limit: usize,
    /// Buffers obtained from Metal (cache misses) so far.
    pub system_allocations: u64,
}

/// The size an allocation of `len` bytes is rounded up to.
pub(crate) fn rounded(len: usize) -> usize {
    match len {
        0 => 0,
        n if n > PAGE => n.div_ceil(PAGE) * PAGE,
        n => n.div_ceil(SMALL) * SMALL,
    }
}

pub(crate) struct Pool {
    state: Mutex<State>,
    /// A tiny buffer bound in place of empty buffers (Metal requires a binding for every
    /// kernel argument; kernels never touch it, as empty buffers have no elements).
    placeholder: Option<Cached>,
}

struct State {
    /// Cached buffers up to one page, by size class (`size / 256`), most recent last: O(1)
    /// reuse for the many small outputs of a computation.
    small: Vec<Vec<(u64, Cached)>>,
    /// Cached larger buffers by `(size, stamp)`: best fit is the first key at or above a size.
    large: BTreeMap<(usize, u64), Cached>,
    /// Stamps order cached buffers by age, for least-recently-used eviction.
    next_stamp: u64,
    cached: usize,
    active: usize,
    peak: usize,
    memory_limit: usize,
    cache_limit: usize,
    working_set: usize,
    system_allocations: u64,
}

impl State {
    /// A cached buffer of at least `size` bytes within MLX's slack (smaller than
    /// `min(2·size, size + 2 pages)`), and its size.
    fn take(&mut self, size: usize) -> Option<(Raw, usize)> {
        let limit = (2 * size).min(size + 2 * PAGE);
        let found = if size <= PAGE {
            let last = ((limit - 1) / SMALL).min(SMALL_CLASSES - 1);
            (size / SMALL..=last)
                .find_map(|c| self.small[c].pop().map(|(_, Cached(raw))| (raw, c * SMALL)))
        } else {
            let key = self
                .large
                .range((size, 0)..(limit, 0))
                .next()
                .map(|(k, _)| *k);
            key.and_then(|k| self.large.remove(&k).map(|Cached(raw)| (raw, k.0)))
        };
        if let Some((_, bucket)) = &found {
            self.cached -= bucket;
        }
        found
    }

    fn put(&mut self, raw: Raw, size: usize) {
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        if size <= PAGE {
            self.small[size / SMALL].push((stamp, Cached(raw)));
        } else {
            self.large.insert((size, stamp), Cached(raw));
        }
        self.cached += size;
    }

    /// Trims the cache to at most `target` bytes, oldest buffers first. Eviction is rare,
    /// so it sorts by age on demand instead of maintaining an ordered index on every reuse.
    fn evict_to(&mut self, target: usize) {
        if self.cached <= target {
            return;
        }
        if target == 0 {
            self.small.iter_mut().for_each(Vec::clear);
            self.large.clear();
            self.cached = 0;
            return;
        }
        let mut ages: Vec<(u64, usize)> = self
            .small
            .iter()
            .enumerate()
            .flat_map(|(class, v)| v.iter().map(move |(stamp, _)| (*stamp, class * SMALL)))
            .chain(self.large.keys().map(|&(size, stamp)| (stamp, size)))
            .collect();
        ages.sort_unstable();
        let mut evict = HashSet::new();
        for (stamp, size) in ages {
            if self.cached <= target {
                break;
            }
            evict.insert(stamp);
            self.cached -= size;
        }
        self.small
            .iter_mut()
            .for_each(|v| v.retain(|(stamp, _)| !evict.contains(stamp)));
        self.large.retain(|&(_, stamp), _| !evict.contains(&stamp));
    }

    /// GC threshold: cached memory is released before active + cached reaches it.
    fn gc_limit(&self) -> usize {
        ((self.working_set as f64 * 0.95) as usize).min(self.memory_limit)
    }
}

impl Pool {
    /// A pool with MLX's default limits for a GPU with the given working set and RAM.
    pub(crate) fn new(working_set: u64, ram: u64, placeholder: Option<Raw>) -> Self {
        let memory_limit = (1.5 * working_set as f64).min(0.95 * ram as f64) as usize;
        Pool {
            state: Mutex::new(State {
                small: (0..SMALL_CLASSES).map(|_| Vec::new()).collect(),
                large: BTreeMap::new(),
                next_stamp: 0,
                cached: 0,
                active: 0,
                peak: 0,
                memory_limit,
                cache_limit: memory_limit,
                working_set: working_set as usize,
                system_allocations: 0,
            }),
            placeholder: placeholder.map(Cached),
        }
    }

    /// The buffer bound for empty buffers.
    pub(crate) fn placeholder(&self) -> Option<&Raw> {
        self.placeholder.as_ref().map(|Cached(raw)| raw)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A buffer of at least `len` bytes and its accounted size; `None` for zero bytes.
    /// Reused buffers hold stale contents.
    pub(crate) fn alloc(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
    ) -> Result<Option<(Raw, usize)>> {
        let size = rounded(len);
        if size == 0 {
            return Ok(None);
        }
        let mut s = self.lock();
        if s.active + size > s.memory_limit {
            s.evict_to(0);
            return Err(GpuError::OutOfMemory(len));
        }
        if let Some((raw, bucket)) = s.take(size) {
            s.active += bucket;
            s.peak = s.peak.max(s.active);
            return Ok(Some((raw, bucket)));
        }
        let gc = s.gc_limit();
        if s.active + s.cached + size >= gc {
            let target = gc.saturating_sub(s.active + size);
            s.evict_to(target);
        }
        // Reserve before the system call so concurrent allocations respect the limit.
        s.active += size;
        s.peak = s.peak.max(s.active);
        drop(s);
        let options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
        match device.newBufferWithLength_options(size, options) {
            Some(raw) => {
                self.lock().system_allocations += 1;
                Ok(Some((raw, size)))
            }
            None => {
                self.lock().active -= size;
                Err(GpuError::OutOfMemory(len))
            }
        }
    }

    /// Returns a buffer of accounted `size`. It is cached for reuse if `reusable` (no GPU
    /// work still uses it), otherwise released.
    pub(crate) fn free(&self, raw: Raw, size: usize, reusable: bool) {
        let mut s = self.lock();
        s.active -= size;
        if reusable {
            s.put(raw, size);
            let limit = s.cache_limit;
            s.evict_to(limit);
        }
    }

    pub(crate) fn stats(&self) -> MemoryStats {
        let s = self.lock();
        MemoryStats {
            active: s.active,
            peak: s.peak,
            cache: s.cached,
            memory_limit: s.memory_limit,
            cache_limit: s.cache_limit,
            system_allocations: s.system_allocations,
        }
    }

    pub(crate) fn set_memory_limit(&self, limit: usize) -> usize {
        std::mem::replace(&mut self.lock().memory_limit, limit)
    }

    pub(crate) fn set_cache_limit(&self, limit: usize) -> usize {
        let mut s = self.lock();
        let previous = std::mem::replace(&mut s.cache_limit, limit);
        s.evict_to(limit);
        previous
    }

    pub(crate) fn clear_cache(&self) {
        self.lock().evict_to(0);
    }

    pub(crate) fn reset_peak(&self) {
        let mut s = self.lock();
        s.peak = s.active;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding() {
        assert_eq!(rounded(0), 0);
        assert_eq!(rounded(1), 256);
        assert_eq!(rounded(256), 256);
        assert_eq!(rounded(257), 512);
        assert_eq!(rounded(PAGE), PAGE);
        assert_eq!(rounded(PAGE + 1), 2 * PAGE);
    }

    #[test]
    fn default_limits_follow_mlx() {
        let pool = Pool::new(40_000_000_000, 48_000_000_000, None);
        let stats = pool.stats();
        assert_eq!(stats.memory_limit, 45_600_000_000); // min(1.5 * 40e9, 0.95 * 48e9)
        assert_eq!(stats.cache_limit, stats.memory_limit);
    }
}
