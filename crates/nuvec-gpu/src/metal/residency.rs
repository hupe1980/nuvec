//! Wired memory: MLX's residency sets (`mlx/backend/metal/resident.cpp`).
//!
//! Buffers in a residency set attached to the command queue stay resident while GPU work
//! runs, instead of being made resident (and possibly paged back in) per command buffer.
//! New buffers are wired while the wired total stays under the wired limit; the rest wait
//! outside every set and join when the limit grows. A limit of 0 (MLX's default) wires
//! nothing. As MLX's, the wired buffers are spread over sets of at most
//! `max(working set · pct / 100, 64 MiB)` bytes (`MLX_RESIDENCY_SET_MAX_PCT`, default 5; 0
//! or 100 and above: one set), up to 32: macOS makes residency decisions per set, so a set
//! that loses residency under memory pressure brings back only its own buffers. Each set
//! holds one standing residency request, made when it is created (and renewed by
//! [`Residency::refresh`]). Residency sets exist on macOS 15 (nuvec's floor) for Metal 3
//! GPUs; on other GPUs nothing is wired.

use std::collections::{HashMap, HashSet};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLCommandQueue, MTLDevice, MTLGPUFamily, MTLResidencySet,
    MTLResidencySetDescriptor,
};

use super::allocator::Raw;

type Set = Retained<ProtocolObject<dyn MTLResidencySet>>;

/// A command queue accepts a limited number of residency sets (Metal's feature tables);
/// past this many, buffers go to the emptiest set, which then grows past the cap.
const MAX_SETS: usize = 32;

/// The smallest per-set cap.
const MIN_SET_BYTES: usize = 64 << 20;

/// A buffer by address (its identity while it is retained here).
fn key(raw: &Raw) -> usize {
    Retained::as_ptr(raw).addr()
}

fn allocation(raw: &Raw) -> &ProtocolObject<dyn MTLAllocation> {
    ProtocolObject::from_ref(&**raw)
}

/// The device and queue residency sets are made for and attached to.
pub(crate) struct Metal {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

/// The device and queue, if the system supports residency sets.
pub(crate) fn metal(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
) -> Option<Metal> {
    // macOS 15 (nuvec's floor) has residency sets, for Metal 3 GPUs.
    device.supportsFamily(MTLGPUFamily::Metal3).then(|| Metal {
        device: device.clone(),
        queue: queue.clone(),
    })
}

/// MLX's per-set cap for a GPU working set of `working_set` bytes
/// (`MLX_RESIDENCY_SET_MAX_PCT`); 0 is one set.
fn set_cap(working_set: u64) -> usize {
    let pct = std::env::var("MLX_RESIDENCY_SET_MAX_PCT")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .unwrap_or(5);
    if !(1..100).contains(&pct) {
        return 0;
    }
    ((working_set as usize / 100) * pct as usize).max(MIN_SET_BYTES)
}

/// A tracked buffer: retained here, with the size it was inserted with and its set
/// (`None` while it waits outside the wired limit).
struct Placement {
    raw: Raw,
    size: usize,
    set: Option<usize>,
}

/// The wired sets and every tracked buffer. Buffers are retained here until
/// [`erase`](Self::erase)d, which the allocator does before releasing them.
pub(crate) struct Residency {
    metal: Option<Metal>,
    /// Sets by id (append-only) and their wired bytes.
    sets: Vec<(Set, usize)>,
    /// Bytes per set (0: one set).
    cap: usize,
    capacity: usize,
    placements: HashMap<usize, Placement>,
    wired_bytes: usize,
    /// The buffers made outside the pool (mapped files), and their wired bytes.
    mapped: HashSet<usize>,
    wired_mapped_bytes: usize,
}

// SAFETY: the residency sets, the device, the queue and the buffers are only retained,
// released and passed to Metal's residency-set methods, always under the allocator's lock.
unsafe impl Send for Residency {}

impl Residency {
    /// Residency for a GPU of `working_set` bytes (`metal`: none without set support).
    pub(crate) fn new(metal: Option<Metal>, working_set: u64) -> Self {
        let mut residency = Residency {
            metal,
            sets: Vec::new(),
            cap: set_cap(working_set),
            capacity: 0,
            placements: HashMap::new(),
            wired_bytes: 0,
            mapped: HashSet::new(),
            wired_mapped_bytes: 0,
        };
        if !residency.add_set() {
            residency.metal = None;
        }
        residency
    }

    /// Whether buffers can be wired on this system.
    pub(crate) fn supported(&self) -> bool {
        self.metal.is_some()
    }

    /// Bytes currently wired.
    pub(crate) fn wired(&self) -> usize {
        self.wired_bytes
    }

    /// Bytes currently wired that belong to mapped files.
    pub(crate) fn wired_mapped(&self) -> usize {
        self.wired_mapped_bytes
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Residency sets in use.
    pub(crate) fn sets(&self) -> usize {
        self.sets.len()
    }

    /// Sets the per-set cap for later placements, without the 64 MiB floor (0: one set).
    /// A test hook, as MLX's `set_max_bytes_per_set`.
    pub(crate) fn set_cap(&mut self, bytes: usize) {
        self.cap = bytes;
    }

    /// A new set, committed, with its standing residency request, attached to the queue.
    fn add_set(&mut self) -> bool {
        let Some(metal) = &self.metal else {
            return false;
        };
        let Ok(set) = metal
            .device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
        else {
            return false;
        };
        set.commit();
        set.requestResidency();
        metal.queue.addResidencySet(&set);
        self.sets.push((set, 0));
        true
    }

    /// MLX's choice of set for `bytes` more: the first with room, else the first emptied
    /// one (after set 0), else a new one while fewer than [`MAX_SETS`] exist, else the
    /// emptiest.
    fn choose(&mut self, bytes: usize) -> usize {
        if self.cap == 0 {
            return 0;
        }
        let (mut empty, mut emptiest) = (None, 0);
        for (i, &(_, size)) in self.sets.iter().enumerate() {
            if size + bytes <= self.cap {
                return i;
            }
            if empty.is_none() && i != 0 && size == 0 {
                empty = Some(i);
            }
            if size < self.sets[emptiest].1 {
                emptiest = i;
            }
        }
        if let Some(i) = empty {
            return i;
        }
        if self.sets.len() < MAX_SETS && self.add_set() {
            return self.sets.len() - 1;
        }
        emptiest
    }

    /// Wires buffer `k` into a chosen set (not committed); returns the set.
    fn wire(&mut self, k: usize) -> usize {
        let size = self.placements[&k].size;
        let i = self.choose(size);
        let p = self.placements.get_mut(&k).expect("tracked");
        self.sets[i].0.addAllocation(allocation(&p.raw));
        self.sets[i].1 += size;
        p.set = Some(i);
        self.wired_bytes += size;
        if self.mapped.contains(&k) {
            self.wired_mapped_bytes += size;
        }
        i
    }

    /// Unwires buffer `k` (not committed); returns the set it left.
    fn unwire(&mut self, k: usize) -> Option<usize> {
        let p = self.placements.get_mut(&k)?;
        let i = p.set.take()?;
        self.sets[i].0.removeAllocation(allocation(&p.raw));
        self.sets[i].1 -= p.size;
        self.wired_bytes -= p.size;
        if self.mapped.contains(&k) {
            self.wired_mapped_bytes -= p.size;
        }
        Some(i)
    }

    /// Tracks a new buffer (`mapped`: a mapped file's): wired if it fits under the limit.
    pub(crate) fn insert(&mut self, raw: &Raw, mapped: bool) {
        if self.metal.is_none() {
            return;
        }
        let size = raw.allocatedSize();
        let k = key(raw);
        if mapped {
            self.mapped.insert(k);
        }
        let raw = raw.clone();
        self.placements.insert(
            k,
            Placement {
                raw,
                size,
                set: None,
            },
        );
        if self.wired_bytes + size <= self.capacity {
            let i = self.wire(k);
            self.sets[i].0.commit();
        }
    }

    /// Stops tracking a buffer that is about to be released.
    pub(crate) fn erase(&mut self, raw: &Raw) {
        let k = key(raw);
        if let Some(i) = self.unwire(k) {
            self.sets[i].0.commit();
        }
        self.placements.remove(&k);
        self.mapped.remove(&k);
    }

    /// Sets the wired limit: waiting buffers join while they fit, or wired ones leave until
    /// the wired total is under the new limit. Each changed set commits once.
    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity;
        if self.metal.is_none() {
            return;
        }
        let mut changed = vec![false; MAX_SETS.max(self.sets.len())];
        if self.wired_bytes < capacity {
            let mut room = capacity - self.wired_bytes;
            let waiting: Vec<usize> = self
                .placements
                .iter()
                .filter(|(_, p)| p.set.is_none())
                .map(|(&k, _)| k)
                .collect();
            for k in waiting {
                let size = self.placements[&k].size;
                if size <= room {
                    room -= size;
                    let i = self.wire(k);
                    changed.resize(changed.len().max(i + 1), false);
                    changed[i] = true;
                }
            }
        } else if self.wired_bytes > capacity {
            let wired: Vec<usize> = self
                .placements
                .iter()
                .filter(|(_, p)| p.set.is_some())
                .map(|(&k, _)| k)
                .collect();
            for k in wired {
                if self.wired_bytes <= capacity {
                    break;
                }
                if let Some(i) = self.unwire(k) {
                    changed[i] = true;
                }
            }
        }
        for (i, (set, _)) in self.sets.iter().enumerate() {
            if changed.get(i).copied().unwrap_or(false) {
                set.commit();
            }
        }
    }

    /// Renews the residency request of every non-empty set: macOS can drop residency after
    /// the GPU idles despite a standing request.
    pub(crate) fn refresh(&self) {
        for (set, size) in &self.sets {
            if *size != 0 {
                set.requestResidency();
            }
        }
    }
}
