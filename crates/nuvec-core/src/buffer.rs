//! Storage for array data.
//!
//! On a Mac with a GPU, array bytes live in shared Metal buffers from the device's caching
//! allocator (unified memory): CPU kernels and GPU kernels work on the same bytes, with no
//! copies. Elsewhere (other platforms, machines without Metal, Miri) host memory is used,
//! with identical behaviour.

use bytemuck::{Pod, Zeroable};

use crate::{Error, Result};

/// A 64-byte block. Backing host buffers with these makes them 64-byte aligned.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

enum Storage {
    Host(Vec<Align64>),
    #[cfg(all(target_os = "macos", not(miri)))]
    Metal(nuvec_gpu::Buffer),
}

/// A byte buffer, at least 64-byte aligned (Metal buffers are page aligned).
pub(crate) struct Buffer {
    storage: Storage,
    len: usize,
}

/// The GPU whose allocator array storage uses, if any.
#[cfg(all(target_os = "macos", not(miri)))]
fn device() -> Option<&'static nuvec_gpu::Device> {
    nuvec_gpu::Device::shared()
}

impl Buffer {
    /// A buffer of `len` bytes with unspecified contents (a reused allocation holds stale
    /// bytes). For outputs that are overwritten entirely.
    pub(crate) fn try_alloc(len: usize) -> Result<Self> {
        #[cfg(all(target_os = "macos", not(miri)))]
        if let Some(device) = device() {
            let buffer = device
                .new_buffer(len)
                .map_err(|_| Error::OutOfMemory(len))?;
            return Ok(Self::new(Storage::Metal(buffer), len));
        }
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(len.div_ceil(64))
            .map_err(|_| Error::OutOfMemory(len))?;
        blocks.resize(len.div_ceil(64), Align64([0; 64]));
        Ok(Self::new(Storage::Host(blocks), len))
    }

    /// A buffer of `len` zero bytes.
    pub(crate) fn try_zeroed(len: usize) -> Result<Self> {
        let mut buffer = Self::try_alloc(len)?;
        if !matches!(buffer.storage, Storage::Host(_)) {
            buffer.bytes_mut().fill(0); // host blocks are already zeroed
        }
        Ok(buffer)
    }

    fn new(storage: Storage, len: usize) -> Self {
        #[cfg(test)]
        live::inc();
        Self { storage, len }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Host(blocks) => &bytemuck::cast_slice(blocks)[..self.len],
            #[cfg(all(target_os = "macos", not(miri)))]
            Storage::Metal(buffer) => buffer
                .contents()
                .expect("no GPU work uses array storage during CPU access"),
        }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.storage {
            Storage::Host(blocks) => &mut bytemuck::cast_slice_mut(blocks)[..self.len],
            #[cfg(all(target_os = "macos", not(miri)))]
            Storage::Metal(buffer) => buffer
                .contents_mut()
                .expect("no GPU work uses array storage during CPU access"),
        }
    }

    /// The contents as `T`s. `len` must be a multiple of `size_of::<T>()`.
    pub(crate) fn typed<T: Pod>(&self) -> &[T] {
        // An empty slice may carry an unaligned dangling pointer; never cast it.
        if self.len == 0 {
            &[]
        } else {
            bytemuck::cast_slice(self.bytes())
        }
    }

    /// The contents as mutable `T`s. `len` must be a multiple of `size_of::<T>()`.
    pub(crate) fn typed_mut<T: Pod>(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            bytemuck::cast_slice_mut(self.bytes_mut())
        }
    }

    /// Whether the bytes live in GPU-visible unified memory.
    #[cfg_attr(not(all(test, target_os = "macos", not(miri))), allow(dead_code))]
    pub(crate) fn is_metal(&self) -> bool {
        !matches!(self.storage, Storage::Host(_))
    }
}

#[cfg(test)]
impl Drop for Buffer {
    fn drop(&mut self) {
        live::dec();
    }
}

/// Per-thread count of live buffers, for memory-retention tests.
#[cfg(test)]
pub(crate) mod live {
    use std::cell::Cell;

    thread_local!(static LIVE: Cell<isize> = const { Cell::new(0) });

    pub(crate) fn inc() {
        LIVE.set(LIVE.get() + 1);
    }
    pub(crate) fn dec() {
        LIVE.set(LIVE.get() - 1);
    }
    pub(crate) fn count() -> isize {
        LIVE.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_and_sized() {
        let mut b = Buffer::try_zeroed(12).unwrap();
        assert_eq!(b.bytes().len(), 12);
        assert_eq!(b.bytes().as_ptr() as usize % 64, 0);
        b.typed_mut::<f32>()[2] = 1.5;
        assert_eq!(b.typed::<f32>(), &[0.0, 0.0, 1.5]);
        assert!(Buffer::try_zeroed(0).unwrap().bytes().is_empty());
    }

    #[test]
    fn zeroed_even_when_reused() {
        let mut dirty = Buffer::try_alloc(4096).unwrap();
        dirty.bytes_mut().fill(0xAB);
        drop(dirty);
        assert!(
            Buffer::try_zeroed(4096)
                .unwrap()
                .bytes()
                .iter()
                .all(|&b| b == 0)
        );
    }

    #[test]
    #[cfg(all(target_os = "macos", not(miri)))]
    fn arrays_live_in_unified_memory() {
        let Some(device) = device() else { return };
        let before = device.memory();
        let a = crate::zeros(&[1 << 18], crate::DType::F32).unwrap(); // 1 MiB
        let storage = a.data().expect("leaf data");
        assert!(storage.buffer.is_metal());
        assert!(
            device.memory().active >= before.active + (1 << 20) || device.memory().peak >= 1 << 20
        );
        drop(a);
    }
}
