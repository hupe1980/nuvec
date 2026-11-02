//! `nuvec::memory` and `nuvec::metal`: MLX's quantities on the GPU, zeros without one.

use std::sync::{Mutex, MutexGuard, PoisonError};

use nuvec::{DType, Device, Stream, memory, metal, ops, with_stream};

/// The limits are process-wide: the tests take turns.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

#[test]
fn memory_functions_track_the_allocator() {
    let _serial = serial();
    // Physical memory is known with or without a GPU, and agrees with the device's.
    assert!(memory::physical_memory() > 0);
    if let Some(info) = metal::device_info() {
        assert_eq!(memory::physical_memory(), info.physical_memory);
    }
    if !metal::is_available() {
        // No Metal: every figure is zero and limits change nothing (MLX's CPU build).
        assert_eq!(memory::stats(), memory::MemoryStats::default());
        assert_eq!(memory::set_memory_limit(1 << 20), 0);
        assert_eq!(memory::set_wired_limit(1 << 20), Ok(0));
        assert!(metal::device_info().is_none());
        assert!(!metal::nax().is_active());
        return;
    }
    let info = metal::device_info().unwrap();
    assert!(!info.name.is_empty() && info.recommended_working_set > 0);
    let before = memory::active_memory();
    with_stream(Stream::default_for(Device::gpu(0)), || {
        let x = ops::zeros([64 << 20], DType::U8);
        x.eval().unwrap();
        assert!(memory::active_memory() >= before + (64 << 20));
        assert!(memory::peak_memory() >= memory::active_memory());
        assert_eq!(memory::active_memory(), memory::stats().active_with_mapped);
        drop(x);
    });
    // The buffer went to the cache; clearing it empties the cache.
    assert!(memory::cache_memory() >= 64 << 20);
    memory::clear_cache();
    assert_eq!(memory::cache_memory(), 0);
    memory::reset_peak_memory();
    assert_eq!(memory::peak_memory(), memory::active_memory());
    // Each setter returns the previous limit.
    let limit = memory::stats().memory_limit;
    assert_eq!(memory::set_memory_limit(limit / 2), limit);
    assert_eq!(memory::set_memory_limit(limit), limit / 2);
    let cache = memory::set_cache_limit(1 << 20);
    assert_eq!(memory::set_cache_limit(cache), 1 << 20);
    // Above the recommended working set, the wired limit is refused.
    let e = memory::set_wired_limit(info.recommended_working_set as usize + 1).unwrap_err();
    assert!(matches!(e, nuvec::Error::InvalidArgument(_)), "{e}");
}

#[test]
fn an_allocation_over_the_limit_is_out_of_memory() {
    let _serial = serial();
    if !metal::is_available() {
        return;
    }
    let previous = memory::set_memory_limit(memory::active_memory() + (1 << 20));
    let e = with_stream(Stream::default_for(Device::gpu(0)), || {
        ops::zeros([256 << 20], DType::U8).eval()
    })
    .unwrap_err();
    memory::set_memory_limit(previous);
    assert!(e.is_oom(), "{e}");
}
