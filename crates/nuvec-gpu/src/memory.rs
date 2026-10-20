//! Allocator statistics (shared by every target).

/// Memory use and limits of a device's allocator, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Bytes in live buffers (rounded allocation sizes).
    pub active: usize,
    /// Highest `active` since creation or the last `Device::reset_peak`.
    pub peak: usize,
    /// Bytes held in the cache for reuse.
    pub cache: usize,
    /// Allocations that would push `active` past this fail.
    pub memory_limit: usize,
    /// The cache is trimmed to at most this many bytes.
    pub cache_limit: usize,
    /// Buffers obtained from Metal (cache misses) so far.
    pub system_allocations: u64,
    /// Buffers are kept resident up to this many bytes (see
    /// `Device::set_wired_limit`).
    pub wired_limit: usize,
    /// Bytes currently wired, including `wired_mapped`.
    pub wired: usize,
    /// Bytes of live memory-mapped files (checkpoints loaded without a copy). They are not
    /// in `active` or `peak`, which count the allocator's own buffers only; MLX copies
    /// weights into its allocator, so its `active` includes them.
    pub mapped: usize,
    /// The part of `wired` that belongs to memory-mapped files.
    pub wired_mapped: usize,
}
