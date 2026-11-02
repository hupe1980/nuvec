//! Frames: the page-aligned buffers that RDMA sends and receives (MLX's `SharedBuffer`),
//! and their sizes. Every raw access to frame memory is in this module.
//!
//! A frame is written by the algorithm, then posted; while posted, only the fabric touches
//! it (the NIC, or the emulated fabric under its lock); after its completion the algorithm
//! reads it. The algorithms' accesses and the emulated fabric's copies go through a lock
//! per frame; the NIC writes posted frames directly (the device-memory boundary).

#![allow(unsafe_code)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, PoisonError};

use nuvec_core::{Error, Result};

/// MLX's smallest frame.
pub(crate) const FRAME: usize = 4096;

/// MLX's number of frame sizes: 4 KB · 2^k for k < 8 (4 KB to 512 KB).
pub(crate) const CLASSES: usize = 8;

/// The largest frame.
#[cfg(test)]
pub(crate) const MAX_FRAME: usize = FRAME << (CLASSES - 1);

/// Frames are aligned to Apple Silicon's 16 KB pages.
const ALIGN: usize = 16384;

/// The size class and frame size for a message of `bytes` (MLX's
/// `buffer_size_from_message`): the largest frame not larger than the message, or 4 KB when
/// large frames are off.
pub(crate) fn class(bytes: usize, large: bool) -> (usize, usize) {
    if large {
        for k in (1..CLASSES).rev() {
            if bytes >= FRAME << k {
                return (k, FRAME << k);
            }
        }
    }
    (0, FRAME)
}

struct Memory {
    ptr: NonNull<u8>,
    len: usize,
    lock: Mutex<()>,
}

// SAFETY: the memory is a plain heap allocation owned by `Memory`; every access from safe
// code takes `lock`, and the NIC's writes happen only while the frame is posted.
unsafe impl Send for Memory {}
// SAFETY: as above.
unsafe impl Sync for Memory {}

impl Drop for Memory {
    fn drop(&mut self) {
        // SAFETY: `ptr` was allocated in `Frame::new` with this layout and is freed once.
        unsafe { dealloc(self.ptr.as_ptr(), Self::layout(self.len)) }
    }
}

impl Memory {
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len, ALIGN).expect("frame sizes are small powers of two")
    }
}

/// A page-aligned frame; clones share the memory.
#[derive(Clone)]
pub(crate) struct Frame(Arc<Memory>);

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Frame({} bytes)", self.0.len)
    }
}

impl Frame {
    /// A zeroed frame of `len` bytes (a positive multiple of 4 KB).
    pub(crate) fn new(len: usize) -> Result<Self> {
        debug_assert!(len > 0 && len.is_multiple_of(FRAME));
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { alloc_zeroed(Memory::layout(len)) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            Error::Backend(format!("jaccl: allocating a {len}-byte frame failed"))
        })?;
        Ok(Self(Arc::new(Memory {
            ptr,
            len,
            lock: Mutex::new(()),
        })))
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len
    }

    /// The frame's address, for registering it with the RDMA library.
    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.0.ptr.as_ptr()
    }

    /// Runs `f` on the frame's first `len` bytes.
    pub(crate) fn read<T>(&self, len: usize, f: impl FnOnce(&[u8]) -> T) -> T {
        let _guard = self.0.lock.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(len <= self.0.len);
        // SAFETY: `len` bytes from `ptr` are allocated; the lock excludes the other safe
        // accesses, and a posted frame is not read (the protocol above).
        f(unsafe { std::slice::from_raw_parts(self.0.ptr.as_ptr(), len) })
    }

    /// Copies `data` to the start of the frame.
    pub(crate) fn write(&self, data: &[u8]) {
        let _guard = self.0.lock.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(data.len() <= self.0.len);
        // SAFETY: the frame has room for `data`, which is another allocation (the caller
        // holds a shared borrow of it and none of the frame); the lock excludes other safe
        // accesses.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.0.ptr.as_ptr(), data.len()) }
    }

    #[cfg(test)]
    /// Copies the first `len` bytes of `src` to the start of `self` (the emulated fabric's
    /// delivery). `src` and `self` are different frames.
    pub(crate) fn copy_from(&self, src: &Frame, len: usize) {
        assert!(!Arc::ptr_eq(&self.0, &src.0) && len <= self.0.len && len <= src.0.len);
        let _dst = self.0.lock.lock().unwrap_or_else(PoisonError::into_inner);
        let _src = src.0.lock.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: two distinct allocations of at least `len` bytes, both locked.
        unsafe { std::ptr::copy_nonoverlapping(src.0.ptr.as_ptr(), self.0.ptr.as_ptr(), len) }
    }
}

/// Whether this system takes frames above 4 KB (macOS 26.3+, as MLX's
/// `__builtin_available(macOS 26.3)`).
pub(crate) fn large_frames() -> bool {
    #[cfg(target_os = "macos")]
    {
        os_version().is_some_and(|v| v >= (26, 3))
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// The macOS version (major, minor) from `kern.osproductversion`.
#[cfg(target_os = "macos")]
fn os_version() -> Option<(u32, u32)> {
    let mut buf = [0u8; 64];
    let mut len = buf.len();
    // SAFETY: the name is NUL-terminated; `buf` has `len` writable bytes and `len` is
    // updated to the bytes written.
    let rc = unsafe {
        libc::sysctlbyname(
            c"kern.osproductversion".as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    let text = std::str::from_utf8(&buf[..len])
        .ok()?
        .trim_end_matches('\0');
    let mut parts = text.split('.').map(|p| p.parse::<u32>().ok());
    Some((parts.next()??, parts.next().flatten().unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_follow_mlx() {
        assert_eq!(class(0, true), (0, 4096));
        assert_eq!(class(8191, true), (0, 4096));
        assert_eq!(class(8192, true), (1, 8192));
        assert_eq!(class(MAX_FRAME, true), (7, MAX_FRAME));
        assert_eq!(class(64 << 20, true), (7, MAX_FRAME));
        assert_eq!(class(64 << 20, false), (0, 4096));
    }

    #[test]
    fn frames_are_aligned_and_copy() {
        let (a, b) = (Frame::new(8192).unwrap(), Frame::new(4096).unwrap());
        assert_eq!(a.as_ptr() as usize % ALIGN, 0);
        a.write(&[1, 2, 3]);
        b.copy_from(&a, 4096);
        assert_eq!(b.read(4, <[u8]>::to_vec), [1, 2, 3, 0]);
        #[cfg(target_os = "macos")]
        assert!(os_version().is_some_and(|v| v.0 >= 15));
    }
}
