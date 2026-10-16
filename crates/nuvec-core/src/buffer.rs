//! Host storage for array data.
//!
//! The seam where unified-memory Metal buffers replace host allocations (F-006).

use bytemuck::{Pod, Zeroable};

use crate::{Error, Result};

/// A 64-byte block. Backing buffers with these makes every buffer 64-byte aligned.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Align64([u8; 64]);

/// A 64-byte-aligned, zero-initialised byte buffer.
pub(crate) struct Buffer {
    blocks: Vec<Align64>,
    len: usize,
}

impl Buffer {
    /// A buffer of `len` zero bytes, or an error if the allocation fails.
    pub(crate) fn try_zeroed(len: usize) -> Result<Self> {
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(len.div_ceil(64))
            .map_err(|e| Error::InvalidArgument(format!("cannot allocate {len} bytes: {e}")))?;
        blocks.resize(len.div_ceil(64), Align64([0; 64]));
        #[cfg(test)]
        live::inc();
        Ok(Self { blocks, len })
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.blocks)[..self.len]
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut(&mut self.blocks)[..self.len]
    }

    /// The contents as `T`s. `len` must be a multiple of `size_of::<T>()`.
    pub(crate) fn typed<T: Pod>(&self) -> &[T] {
        bytemuck::cast_slice(self.bytes())
    }

    /// The contents as mutable `T`s. `len` must be a multiple of `size_of::<T>()`.
    pub(crate) fn typed_mut<T: Pod>(&mut self) -> &mut [T] {
        bytemuck::cast_slice_mut(self.bytes_mut())
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
}
