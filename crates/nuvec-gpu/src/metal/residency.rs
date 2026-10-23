//! Wired memory: MLX's residency set (`mlx/backend/metal/resident.cpp`).
//!
//! Buffers in a residency set attached to the command queue stay resident while GPU work
//! runs, instead of being made resident (and possibly paged back in) per command buffer.
//! New buffers join the set while it holds less than the wired limit; the rest wait in an
//! unwired list and join when the limit grows. A limit of 0 (MLX's default) wires nothing.
//! Residency sets exist on macOS 15 (nuvec's floor) for Metal 3 GPUs; on other GPUs nothing
//! is wired.

use std::collections::{HashMap, HashSet};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLCommandQueue, MTLDevice, MTLGPUFamily, MTLResidencySet,
    MTLResidencySetDescriptor,
};

use super::allocator::Raw;

type Set = Retained<ProtocolObject<dyn MTLResidencySet>>;

/// A buffer by address (its identity while it is retained here).
fn key(raw: &Raw) -> usize {
    Retained::as_ptr(raw).addr()
}

fn allocation(raw: &Raw) -> &ProtocolObject<dyn MTLAllocation> {
    ProtocolObject::from_ref(&**raw)
}

/// A residency set attached to `queue`, if the system supports them.
pub(crate) fn attach(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Option<Set> {
    // macOS 15 (nuvec's floor) has residency sets, for Metal 3 GPUs.
    if !device.supportsFamily(MTLGPUFamily::Metal3) {
        return None;
    }
    let set = device
        .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
        .ok()?;
    queue.addResidencySet(&set);
    Some(set)
}

/// Counts `size` bytes of buffer `k` as wired (`added`) or unwired, in the total and, for a
/// mapped file's buffer, in the mapped part.
fn count_wired(
    mapped: &HashSet<usize>,
    (total, of_mapped): (&mut usize, &mut usize),
    k: usize,
    size: usize,
    added: bool,
) {
    let part = if mapped.contains(&k) { size } else { 0 };
    if added {
        *total += size;
        *of_mapped += part;
    } else {
        *total -= size;
        *of_mapped -= part;
    }
}

/// The wired set and the buffers waiting outside it. Buffers are retained here until
/// [`erase`](Self::erase)d, which the allocator does before releasing them.
pub(crate) struct Residency {
    set: Option<Set>,
    capacity: usize,
    wired: HashMap<usize, (Raw, usize)>,
    wired_bytes: usize,
    unwired: HashMap<usize, (Raw, usize)>,
    /// The buffers made outside the pool (mapped files), and their wired bytes.
    mapped: HashSet<usize>,
    wired_mapped_bytes: usize,
}

// SAFETY: the residency set and the buffers are only retained, released and passed to
// Metal's residency-set methods, always under the allocator's lock.
unsafe impl Send for Residency {}

impl Residency {
    pub(crate) fn new(set: Option<Set>) -> Self {
        Residency {
            set,
            capacity: 0,
            wired: HashMap::new(),
            wired_bytes: 0,
            unwired: HashMap::new(),
            mapped: HashSet::new(),
            wired_mapped_bytes: 0,
        }
    }

    /// Whether buffers can be wired on this system.
    pub(crate) fn supported(&self) -> bool {
        self.set.is_some()
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

    /// Tracks a new buffer (`mapped`: a mapped file's): wired if it fits under the limit.
    pub(crate) fn insert(&mut self, raw: &Raw, mapped: bool) {
        let Some(set) = &self.set else { return };
        let size = raw.allocatedSize();
        if mapped {
            self.mapped.insert(key(raw));
        }
        if self.wired_bytes + size <= self.capacity {
            set.addAllocation(allocation(raw));
            set.commit();
            set.requestResidency();
            self.wired.insert(key(raw), (raw.clone(), size));
            count_wired(
                &self.mapped,
                (&mut self.wired_bytes, &mut self.wired_mapped_bytes),
                key(raw),
                size,
                true,
            );
        } else {
            self.unwired.insert(key(raw), (raw.clone(), size));
        }
    }

    /// Stops tracking a buffer that is about to be released.
    pub(crate) fn erase(&mut self, raw: &Raw) {
        if self.set.is_none() {
            return;
        }
        let k = key(raw);
        if self.unwired.remove(&k).is_none()
            && let Some((raw, size)) = self.wired.remove(&k)
        {
            let set = self.set.as_ref().expect("checked");
            set.removeAllocation(allocation(&raw));
            set.commit();
            count_wired(
                &self.mapped,
                (&mut self.wired_bytes, &mut self.wired_mapped_bytes),
                k,
                size,
                false,
            );
        }
        self.mapped.remove(&k);
    }

    /// Sets the wired limit: waiting buffers join while they fit, or wired ones leave until
    /// the set is under the new limit.
    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity;
        let Some(set) = &self.set else { return };
        if self.wired_bytes < capacity {
            let mut fitting = Vec::new();
            let mut room = capacity - self.wired_bytes;
            for (&k, &(_, size)) in &self.unwired {
                if size <= room {
                    room -= size;
                    fitting.push(k);
                }
            }
            if fitting.is_empty() {
                return;
            }
            for k in fitting {
                let (raw, size) = self.unwired.remove(&k).expect("listed above");
                set.addAllocation(allocation(&raw));
                self.wired.insert(k, (raw, size));
                count_wired(
                    &self.mapped,
                    (&mut self.wired_bytes, &mut self.wired_mapped_bytes),
                    k,
                    size,
                    true,
                );
            }
            set.commit();
            set.requestResidency();
        } else if self.wired_bytes > capacity {
            let mut leaving = Vec::new();
            let mut bytes = self.wired_bytes;
            for (&k, &(_, size)) in &self.wired {
                if bytes <= capacity {
                    break;
                }
                bytes -= size;
                leaving.push(k);
            }
            for k in leaving {
                let (raw, size) = self.wired.remove(&k).expect("listed above");
                set.removeAllocation(allocation(&raw));
                self.unwired.insert(k, (raw, size));
                count_wired(
                    &self.mapped,
                    (&mut self.wired_bytes, &mut self.wired_mapped_bytes),
                    k,
                    size,
                    false,
                );
            }
            set.commit();
        }
    }
}
