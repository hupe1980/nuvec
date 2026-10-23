//! Allocator statistics (shared by every target).

/// Memory use and limits of a device's allocator, in bytes.
///
/// `peak_with_mapped` is nuvec's one peak-memory metric: compare it with MLX's
/// `get_peak_memory()` (which counts MLX's copied weights), each reset after loading, in a
/// fresh process per phase, cross-checked by the operating system's peak resident size.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Bytes in live allocator buffers (rounded allocation sizes), weights copied from
    /// checkpoints included; mapped files not included. MLX: `get_active_memory`.
    pub active: usize,
    /// The highest `active` since creation or the last `Device::reset_peak`; mapped files
    /// not included.
    pub peak: usize,
    /// `active + mapped`: everything the allocator holds for live arrays, mapped
    /// checkpoints included. MLX: `get_active_memory` (MLX copies every weight, so its
    /// active memory includes them).
    pub active_with_mapped: usize,
    /// The highest `active + mapped` since creation or the last `Device::reset_peak`,
    /// updated on every change of either: **the one peak metric** (see above). MLX:
    /// `get_peak_memory`.
    pub peak_with_mapped: usize,
    /// Bytes held in the cache for reuse (not in `active`). MLX: `get_cache_memory`.
    pub cache: usize,
    /// Allocations and mappings that would push `active + mapped` past this fail with
    /// `OutOfMemory` (a hard refusal; MLX's `set_memory_limit` is a guideline).
    pub memory_limit: usize,
    /// The cache is trimmed to at most this many bytes. MLX: `set_cache_limit`.
    pub cache_limit: usize,
    /// Buffers obtained from Metal (cache misses) so far.
    pub system_allocations: u64,
    /// Buffers and mapped files are kept resident up to this many bytes (see
    /// `Device::set_wired_limit`; MLX: `set_wired_limit`).
    pub wired_limit: usize,
    /// Bytes currently wired, mapped files included (`wired ≤ wired_limit`).
    pub wired: usize,
    /// Bytes of live memory-mapped files (checkpoints loaded without a copy): each file's
    /// whole length, header included, rounded up to pages. Not in `active` or `peak`. MLX
    /// does not map files.
    pub mapped: usize,
    /// The part of `wired` that belongs to memory-mapped files.
    pub wired_mapped: usize,
}

/// The virtual-memory page size of Apple silicon (the allocator's large-buffer granule).
const PAGE: usize = 16 * 1024;
/// The allocator's granule for buffers up to one page.
const SMALL: usize = 256;

/// The bytes the allocator charges (in `active`) for a buffer of `len` bytes: its size
/// class (256-byte steps up to one page, whole pages beyond); `None` if that overflows.
pub fn allocation_size(len: usize) -> Option<usize> {
    match len {
        n if n > PAGE => n.checked_next_multiple_of(PAGE),
        n => Some(n.div_ceil(SMALL) * SMALL),
    }
}

/// The bytes mapping a file of `len` bytes counts in `mapped`: the whole file, rounded up
/// to the system's pages.
pub fn mapping_size(len: usize) -> usize {
    len.div_ceil(page()) * page()
}

/// The system's page size.
pub(crate) fn page() -> usize {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: `sysconf` has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(page).unwrap_or(PAGE)
    }
    #[cfg(not(target_os = "macos"))]
    PAGE
}
