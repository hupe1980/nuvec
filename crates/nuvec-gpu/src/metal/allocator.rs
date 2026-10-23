//! The buffer cache: MLX's Metal allocator rules (`mlx/backend/metal/allocator.cpp`).
//!
//! Freed buffers are cached and handed back for similar sizes, so steady-state loops make
//! almost no system allocations. The cache is bounded (least-recently-used eviction), and
//! allocations or mappings that would push `active + mapped` beyond the memory limit fail
//! instead of exhausting shared system RAM (MLX treats its limit as a guideline).
//!
//! As in MLX, a buffer freed while submitted GPU work still uses it is cached at once: GPU
//! work encoded later may reuse it, ordered after those uses by the buffer's fences (its
//! [`Guard`] goes with it). CPU access waits until the buffer is idle.
//!
//! New buffers are wired up to the wired limit ([`Residency`]); evicted ones are unwired
//! before they are released.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::buffer::Guard;
use super::residency::Residency;
use crate::{GpuError, MemoryStats, Result};

/// The virtual-memory page size on Apple silicon.
pub(crate) const PAGE: usize = 16 * 1024;
/// Granularity of allocations up to one page.
const SMALL: usize = 256;
/// Number of small size classes: 256 B steps up to one page.
const SMALL_CLASSES: usize = PAGE / SMALL + 1;

pub(crate) type Raw = Retained<ProtocolObject<dyn MTLBuffer>>;

/// An allocation: the buffer, its accounted size and the hazard state of GPU work that
/// still uses reused memory.
pub(crate) type Allocation = (Raw, usize, Option<Arc<Guard>>);

/// A cached buffer. Cached buffers are never accessed, only retained and released.
struct Cached(Raw);

/// A cached buffer and, while submitted GPU work still uses it, that work's hazard state.
struct Entry {
    raw: Cached,
    busy: Option<Arc<Guard>>,
}

impl Entry {
    /// Whether an allocation may take it: GPU work may take a buffer whose pending uses are
    /// all committed (their fences order the new use after them); the CPU only an idle one.
    fn takeable(&self, gpu: bool) -> bool {
        match &self.busy {
            None => true,
            Some(g) if g.idle() => true,
            Some(g) => gpu && g.committed_uses(),
        }
    }

    /// The buffer and its pending hazard state (`None` once idle).
    fn into_parts(self) -> (Raw, Option<Arc<Guard>>) {
        let busy = self.busy.filter(|g| !g.idle());
        (self.raw.0, busy)
    }
}

// SAFETY: retaining and releasing Metal buffer objects is thread-safe; a cached buffer's
// memory is not touched until it is handed out again, under the pool's lock.
unsafe impl Send for Cached {}
// SAFETY: shared access only retains or binds the buffer (the placeholder); its memory is
// never read or written through a shared reference.
unsafe impl Sync for Cached {}

/// The size an allocation of `len` bytes is rounded up to; `None` if that overflows.
pub(crate) fn rounded(len: usize) -> Option<usize> {
    crate::memory::allocation_size(len)
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
    small: Vec<Vec<(u64, Entry)>>,
    /// Cached larger buffers by `(size, stamp)`: best fit is the first key at or above a size.
    large: BTreeMap<(usize, u64), Entry>,
    /// Stamps order cached buffers by age, for least-recently-used eviction.
    next_stamp: u64,
    cached: usize,
    active: usize,
    peak: usize,
    /// The highest `active + mapped`.
    peak_with_mapped: usize,
    memory_limit: usize,
    cache_limit: usize,
    working_set: usize,
    system_allocations: u64,
    /// Bytes of live buffers made outside the pool (mapped files), not in `active`.
    mapped: usize,
    residency: Residency,
}

impl State {
    /// Records `active` and `mapped` in both peaks (after every change of either).
    fn note_peaks(&mut self) {
        self.peak = self.peak.max(self.active);
        self.peak_with_mapped = self.peak_with_mapped.max(self.active + self.mapped);
    }

    /// A cached buffer of at least `size` bytes within MLX's slack (smaller than
    /// `min(2·size, size + 2 pages)`) that an allocation for the GPU (or the CPU) may take,
    /// its size and its pending hazard state.
    fn take(&mut self, size: usize, gpu: bool) -> Option<Allocation> {
        let limit = size.saturating_mul(2).min(size.saturating_add(2 * PAGE));
        let found = if size <= PAGE {
            let last = ((limit - 1) / SMALL).min(SMALL_CLASSES - 1);
            (size / SMALL..=last).find_map(|c| {
                let class = &mut self.small[c];
                // The most recently freed first.
                let i = class.iter().rposition(|(_, e)| e.takeable(gpu))?;
                let (raw, busy) = class.remove(i).1.into_parts();
                Some((raw, c * SMALL, busy))
            })
        } else {
            let key = self
                .large
                .range((size, 0)..(limit, 0))
                .find(|(_, e)| e.takeable(gpu))
                .map(|(k, _)| *k);
            key.and_then(|k| {
                let (raw, busy) = self.large.remove(&k)?.into_parts();
                Some((raw, k.0, busy))
            })
        };
        if let Some((_, bucket, _)) = &found {
            self.cached -= bucket;
        }
        found
    }

    fn put(&mut self, raw: Raw, size: usize, busy: Option<Arc<Guard>>) {
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        let entry = Entry {
            raw: Cached(raw),
            busy,
        };
        if size <= PAGE {
            self.small[size / SMALL].push((stamp, entry));
        } else {
            self.large.insert((size, stamp), entry);
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
            for (_, e) in self.small.iter().flatten() {
                self.residency.erase(&e.raw.0);
            }
            for e in self.large.values() {
                self.residency.erase(&e.raw.0);
            }
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
        let residency = &mut self.residency;
        let mut keep = |stamp: &u64, e: &Entry| {
            let evicted = evict.contains(stamp);
            if evicted {
                residency.erase(&e.raw.0);
            }
            !evicted
        };
        self.small
            .iter_mut()
            .for_each(|v| v.retain(|(stamp, e)| keep(stamp, e)));
        self.large.retain(|&(_, stamp), e| keep(&stamp, e));
    }

    /// GC threshold: cached memory is released before active + mapped + cached reaches it.
    fn gc_limit(&self) -> usize {
        ((self.working_set as f64 * 0.95) as usize).min(self.memory_limit)
    }
}

impl Pool {
    /// A pool with MLX's default limits for a GPU with the given working set and RAM.
    pub(crate) fn new(
        working_set: u64,
        ram: u64,
        placeholder: Option<Raw>,
        residency: Residency,
    ) -> Self {
        let memory_limit = (1.5 * working_set as f64).min(0.95 * ram as f64) as usize;
        Pool {
            state: Mutex::new(State {
                small: (0..SMALL_CLASSES).map(|_| Vec::new()).collect(),
                large: BTreeMap::new(),
                next_stamp: 0,
                cached: 0,
                active: 0,
                peak: 0,
                peak_with_mapped: 0,
                memory_limit,
                cache_limit: memory_limit,
                working_set: working_set as usize,
                system_allocations: 0,
                mapped: 0,
                residency,
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

    /// A buffer of at least `len` bytes, its accounted size and, for a buffer that
    /// submitted GPU work still uses (only handed out with `gpu`), that work's hazard state;
    /// `None` for zero bytes. Reused buffers hold stale contents.
    pub(crate) fn alloc(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
        gpu: bool,
    ) -> Result<Option<Allocation>> {
        let size = rounded(len).ok_or(GpuError::OutOfMemory(len))?;
        if size == 0 {
            return Ok(None);
        }
        // Metal refuses (and its validation layer aborts on) lengths above the device's
        // maximum, whatever the memory limit says.
        if size > device.maxBufferLength() {
            return Err(GpuError::OutOfMemory(len));
        }
        let mut s = self.lock();
        // Checked: `active + mapped + size` must not pass the limit (nor wrap); the
        // result is the new `active`.
        let fits = |s: &State, size: usize| {
            s.active
                .checked_add(size)
                .filter(|&total| total.saturating_add(s.mapped) <= s.memory_limit)
        };
        let Some(after) = fits(&s, size) else {
            s.evict_to(0);
            return Err(GpuError::OutOfMemory(len));
        };
        if let Some((raw, bucket, busy)) = s.take(size, gpu) {
            // The limit applies to what is charged: the cached bucket, possibly larger
            // than `size`. One that does not fit goes back; a new buffer of `size` may.
            if let Some(total) = fits(&s, bucket) {
                s.active = total;
                s.note_peaks();
                return Ok(Some((raw, bucket, busy)));
            }
            s.put(raw, bucket, busy);
        }
        // Mapped files count: they are resident too (MLX loads weights as active memory).
        let gc = s.gc_limit();
        let used = after.saturating_add(s.mapped);
        if used.saturating_add(s.cached) >= gc {
            let target = gc.saturating_sub(used);
            s.evict_to(target);
        }
        // Reserve before the system call so concurrent allocations respect the limit.
        s.active = after;
        s.note_peaks();
        drop(s);
        let options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked;
        match device.newBufferWithLength_options(size, options) {
            Some(raw) => {
                let mut s = self.lock();
                s.system_allocations += 1;
                s.residency.insert(&raw, false);
                Ok(Some((raw, size, None)))
            }
            None => {
                self.lock().active -= size;
                Err(GpuError::OutOfMemory(len))
            }
        }
    }

    /// Returns a buffer of accounted `size` to the cache, with the hazard state of the
    /// submitted GPU work that still uses it (if any).
    pub(crate) fn free(&self, raw: Raw, size: usize, busy: Option<Arc<Guard>>) {
        let mut s = self.lock();
        s.active -= size;
        s.put(raw, size, busy);
        let limit = s.cache_limit;
        s.evict_to(limit);
    }

    pub(crate) fn stats(&self) -> MemoryStats {
        let s = self.lock();
        MemoryStats {
            active: s.active,
            peak: s.peak,
            active_with_mapped: s.active + s.mapped,
            peak_with_mapped: s.peak_with_mapped,
            cache: s.cached,
            memory_limit: s.memory_limit,
            cache_limit: s.cache_limit,
            system_allocations: s.system_allocations,
            wired_limit: s.residency.capacity(),
            wired: s.residency.wired(),
            mapped: s.mapped,
            wired_mapped: s.residency.wired_mapped(),
        }
    }

    /// Counts a buffer made outside the pool (a mapped file) in `mapped` and wires it
    /// while it fits under the wired limit, or refuses it (`OutOfMemory`, nothing wired)
    /// when `active + mapped` would pass the memory limit.
    pub(crate) fn track(&self, raw: &Raw) -> Result<()> {
        let size = objc2_metal::MTLResource::allocatedSize(&**raw);
        let mut s = self.lock();
        let total = s.active.saturating_add(s.mapped).saturating_add(size);
        if total > s.memory_limit {
            return Err(GpuError::OutOfMemory(size));
        }
        s.mapped += size;
        s.note_peaks();
        s.residency.insert(raw, true);
        Ok(())
    }

    /// Unwires a buffer made outside the pool before it is released.
    pub(crate) fn untrack(&self, raw: &Raw) {
        let mut s = self.lock();
        s.mapped -= objc2_metal::MTLResource::allocatedSize(&**raw);
        s.residency.erase(raw);
    }

    /// Whether buffers can be wired on this system (residency sets: a Metal 3 GPU).
    pub(crate) fn can_wire(&self) -> bool {
        self.lock().residency.supported()
    }

    pub(crate) fn set_wired_limit(&self, limit: usize) -> usize {
        let mut s = self.lock();
        let previous = s.residency.capacity();
        s.residency.resize(limit);
        previous
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
        s.peak_with_mapped = s.active + s.mapped;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding() {
        assert_eq!(rounded(0), Some(0));
        assert_eq!(rounded(1), Some(256));
        assert_eq!(rounded(256), Some(256));
        assert_eq!(rounded(257), Some(512));
        assert_eq!(rounded(PAGE), Some(PAGE));
        assert_eq!(rounded(PAGE + 1), Some(2 * PAGE));
        assert_eq!(rounded(usize::MAX - PAGE + 1), Some(usize::MAX - PAGE + 1));
        assert_eq!(rounded(usize::MAX - PAGE + 2), None);
    }

    #[test]
    fn default_limits_follow_mlx() {
        let pool = Pool::new(40_000_000_000, 48_000_000_000, None, Residency::new(None));
        let stats = pool.stats();
        assert_eq!(stats.memory_limit, 45_600_000_000); // min(1.5 * 40e9, 0.95 * 48e9)
        assert_eq!(stats.cache_limit, stats.memory_limit);
    }
}
