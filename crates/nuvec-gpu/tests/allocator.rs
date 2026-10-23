//! The caching allocator. Every test uses its own device (and so its own pool) and skips
//! when no Metal device is available.

#![cfg(target_os = "macos")]

use nuvec_gpu::{Arg, Device, GpuError};

const PAGE: usize = 16 * 1024;

/// The Metal device, or `None` (the test skips) without one. With `NUVEC_REQUIRE_GPU=1` a
/// missing device fails the test instead of skipping it.
fn device() -> Option<Device> {
    let device = Device::system_default();
    if device.is_none() {
        assert!(
            std::env::var("NUVEC_REQUIRE_GPU").as_deref() != Ok("1"),
            "NUVEC_REQUIRE_GPU=1 but no Metal device is available"
        );
        eprintln!("skipped: no Metal device");
    }
    device
}

#[test]
fn steady_state_reuses_one_buffer() {
    let Some(device) = device() else { return };
    for _ in 0..10_000 {
        let buffer = device.new_buffer(4096).unwrap();
        drop(buffer);
    }
    let stats = device.memory();
    assert_eq!(stats.system_allocations, 1);
    assert_eq!((stats.active, stats.cache), (0, 4096));
}

#[test]
fn stats_are_exact() {
    let Some(device) = device() else { return };
    let a = device.new_buffer(1000).unwrap(); // rounds to 1024
    let b = device.new_buffer(PAGE + 1).unwrap(); // rounds to 2 pages
    let stats = device.memory();
    assert_eq!(
        (stats.active, stats.peak, stats.cache),
        (1024 + 2 * PAGE, 1024 + 2 * PAGE, 0)
    );
    drop(a);
    let stats = device.memory();
    assert_eq!(
        (stats.active, stats.peak, stats.cache),
        (2 * PAGE, 1024 + 2 * PAGE, 1024)
    );
    device.reset_peak();
    assert_eq!(device.memory().peak, 2 * PAGE);
    drop(b);
    device.clear_cache();
    let stats = device.memory();
    assert_eq!((stats.active, stats.cache), (0, 0));
}

#[test]
fn large_cached_buffers_are_not_wasted_on_small_requests() {
    let Some(device) = device() else { return };
    drop(device.new_buffer(1 << 20).unwrap());
    let small = device.new_buffer(256).unwrap();
    let stats = device.memory();
    assert_eq!(
        stats.system_allocations, 2,
        "1 MiB is more than twice 256 B: not reused"
    );
    assert_eq!(stats.cache, 1 << 20);
    drop(small);
    // A request within the slack reuses: 1 MiB - 1 page still fits min(2x, +2 pages).
    let near = device.new_buffer((1 << 20) - PAGE).unwrap();
    assert_eq!(device.memory().system_allocations, 2);
    assert_eq!(near.len(), (1 << 20) - PAGE);
}

#[test]
fn cache_limit_evicts_oldest_first() {
    let Some(device) = device() else { return };
    device.set_cache_limit(3 * PAGE);
    let buffers: Vec<_> = (1..=4)
        .map(|pages| device.new_buffer(pages * PAGE).unwrap())
        .collect();
    drop(buffers); // freed in order 1, 2, 3, 4 pages
    // Caching 1+2 pages fits; caching 3 more evicts the oldest (1, then 2) to get back to
    // 3 pages; caching 4 more evicts the 3 and then the 4 itself, which alone exceeds 3.
    let stats = device.memory();
    assert!(stats.cache <= 3 * PAGE, "{stats:?}");
    assert_eq!(
        stats.cache, 0,
        "the 4-page buffer alone exceeds the limit, older ones were evicted before it"
    );
    device.set_cache_limit(10 * PAGE);
    let kept: Vec<_> = [1, 2, 3]
        .iter()
        .map(|&p| device.new_buffer(p * PAGE).unwrap())
        .collect();
    drop(kept);
    assert_eq!(device.memory().cache, 6 * PAGE);
    assert_eq!(
        device.set_cache_limit(2 * PAGE),
        10 * PAGE,
        "setter returns the previous limit"
    );
    assert_eq!(
        device.memory().cache,
        0,
        "3 pages (newest) alone exceed 2 pages; everything older went first"
    );
}

#[test]
fn gc_threshold_releases_cache_before_allocating() {
    let Some(device) = device() else { return };
    device.set_memory_limit(4 * PAGE);
    drop(device.new_buffer(2 * PAGE).unwrap());
    assert_eq!(device.memory().cache, 2 * PAGE);
    let big = device.new_buffer(3 * PAGE).unwrap(); // 2 cached + 3 >= 4: cache released first
    let stats = device.memory();
    assert_eq!((stats.active, stats.cache), (3 * PAGE, 0));
    drop(big);
}

#[test]
fn memory_limit_is_enforced() {
    let Some(device) = device() else { return };
    device.set_memory_limit(2 * PAGE);
    let a = device.new_buffer(PAGE).unwrap();
    assert_eq!(
        device.new_buffer(2 * PAGE).unwrap_err(),
        GpuError::OutOfMemory(2 * PAGE)
    );
    assert_eq!(
        device.memory().active,
        PAGE,
        "a failed allocation reserves nothing"
    );
    drop(a);
    assert!(device.new_buffer(2 * PAGE).is_ok());
}

#[test]
fn memory_in_use_is_reused_only_by_gpu_work_ordered_after_it() {
    let Some(device) = device() else { return };
    let n = 1 << 20;
    let a = device.buffer_from(&vec![0u8; n * 4]).unwrap();
    let add = device.pipeline("nuvec_add_f32").unwrap();
    let encode = |out: &mut nuvec_gpu::Buffer| {
        let mut cmd = device.ordered_command_buffer().unwrap();
        let count = (n as u32).to_le_bytes();
        let args = [
            Arg::read(&a),
            Arg::read(&a),
            Arg::write(out),
            Arg::bytes(&count),
        ];
        // SAFETY: `nuvec_add_f32` touches the first `n` floats of each buffer, which
        // all hold `n` floats.
        unsafe { cmd.dispatch(&add, &args, n) }.unwrap();
        cmd
    };
    device.clear_cache();
    let mut out = device.new_buffer(n * 4).unwrap();
    let cmd = encode(&mut out);
    drop(out); // dropped while encoded GPU work uses it: cached at once
    let stats = device.memory();
    assert_eq!((stats.active, stats.cache), (a.len(), a.len()));
    let allocations = stats.system_allocations;
    // Not committed yet: no fence orders other work after it, so nobody gets it.
    drop((
        device.new_buffer(n * 4).unwrap(),
        device.new_gpu_buffer(n * 4).unwrap(),
    ));
    assert_eq!(device.memory().system_allocations, allocations + 2);
    // Committed (complete or not): GPU work gets it, ordered after it by its fence.
    device.clear_cache();
    let mut out = device.new_buffer(n * 4).unwrap();
    drop(cmd); // abandoned: never submitted
    let pending = encode(&mut out).commit_then(|r| r.unwrap());
    drop(out);
    let allocations = device.memory().system_allocations;
    let reused = device.new_gpu_buffer(n * 4).unwrap();
    assert_eq!(
        device.memory().system_allocations,
        allocations,
        "reused by GPU work"
    );
    // The CPU waits for the previous owner's work before touching the memory.
    assert_eq!(reused.contents().unwrap().len(), n * 4);
    drop(reused);
    pending.wait();
}

#[test]
fn zero_bytes_cost_nothing() {
    let Some(device) = device() else { return };
    let empty = device.new_buffer(0).unwrap();
    assert!(empty.contents().unwrap().is_empty());
    let stats = device.memory();
    assert_eq!((stats.active, stats.system_allocations), (0, 0));
}

#[test]
fn concurrent_alloc_and_free() {
    let Some(device) = device() else { return };
    std::thread::scope(|s| {
        for t in 0..8 {
            let device = device.clone();
            s.spawn(move || {
                for i in 0..500 {
                    let mut b = device.new_buffer(256 * (1 + (i + t) % 64)).unwrap();
                    b.contents_mut().unwrap()[0] = t as u8;
                }
            });
        }
    });
    assert_eq!(device.memory().active, 0);
}

#[test]
fn shared_device_is_a_singleton() {
    let (Some(a), Some(b)) = (Device::shared(), Device::shared()) else {
        return;
    };
    assert!(std::ptr::eq(a, b));
}

#[test]
fn wired_limit_wires_buffers_up_to_it() {
    let Some(device) = device() else { return };
    let wired = |d: &Device| d.memory().wired;
    assert_eq!((device.memory().wired_limit, wired(&device)), (0, 0)); // MLX's default
    let too_much = device.info().recommended_working_set as usize + 1;
    assert!(matches!(
        device.set_wired_limit(too_much),
        Err(GpuError::InvalidArgument(_))
    ));
    assert_eq!(device.set_wired_limit(4 * PAGE).unwrap(), 0);
    let mut buffers: Vec<_> = (0..3)
        .map(|_| device.new_buffer(2 * PAGE).unwrap())
        .collect();
    if !device.can_wire() {
        assert_eq!(wired(&device), 0);
        return;
    }
    // Two fit; the third waits outside the set.
    assert_eq!(wired(&device), 4 * PAGE);
    // Lowering the limit unwires, raising it wires the waiting buffers.
    device.set_wired_limit(2 * PAGE).unwrap();
    assert_eq!(wired(&device), 2 * PAGE);
    assert_eq!(device.set_wired_limit(8 * PAGE).unwrap(), 2 * PAGE);
    assert_eq!(wired(&device), 6 * PAGE);
    // Wired buffers work as usual.
    let add = device.pipeline("nuvec_binary_add_float32_c").unwrap();
    let n = 2 * PAGE / 4;
    let (a, b) = (buffers.pop().unwrap(), buffers.pop().unwrap());
    let (offset, scalars) = (0i64.to_le_bytes(), 0u32.to_le_bytes());
    let mut cmd = device.command_buffer().unwrap();
    let args = [
        Arg::read(&a),
        Arg::read(&b),
        Arg::write(&mut buffers[0]),
        Arg::bytes(&offset),
        Arg::bytes(&offset),
        Arg::bytes(&scalars),
    ];
    // SAFETY: the contiguous binary kernel touches element `i < n` of each buffer (offset
    // 0, no scalar operands); every buffer holds `n` floats.
    unsafe { cmd.dispatch(&add, &args, n) }.unwrap();
    cmd.commit().wait().unwrap();
    // Cached buffers stay wired; released ones leave the set.
    drop((a, b, buffers));
    assert_eq!(wired(&device), 6 * PAGE);
    device.clear_cache();
    assert_eq!(wired(&device), 0);
    // Mapped files are wired too, and unwired when released.
    let path = std::env::temp_dir().join(format!("nuvec-wired-{}", std::process::id()));
    std::fs::write(&path, vec![1u8; 3 * PAGE]).unwrap();
    let (mapped_before, active_before) = (device.memory().mapped, device.memory().active);
    let mapped = device.map_file(&path).unwrap();
    assert_eq!(wired(&device), 3 * PAGE);
    // Reported apart: in `mapped` and `wired_mapped`, not in `active`.
    let m = device.memory();
    assert_eq!(m.mapped - mapped_before, 3 * PAGE);
    assert_eq!((m.wired_mapped, m.active), (3 * PAGE, active_before));
    // Unwired under a lower limit, still mapped.
    device.set_wired_limit(2 * PAGE).unwrap();
    let m = device.memory();
    assert_eq!(
        (m.wired, m.wired_mapped, m.mapped - mapped_before),
        (0, 0, 3 * PAGE)
    );
    device.set_wired_limit(8 * PAGE).unwrap();
    assert_eq!(device.memory().wired_mapped, 3 * PAGE);
    drop(mapped);
    let m = device.memory();
    assert_eq!((m.wired, m.wired_mapped, m.mapped), (0, 0, mapped_before));
    std::fs::remove_file(&path).unwrap();
    device.set_wired_limit(0).unwrap();
}

#[test]
fn sizes_near_usize_max_are_out_of_memory() {
    let Some(device) = device() else { return };
    // Rounding up to a page used to wrap to 0: a bogus empty buffer.
    for len in [usize::MAX - 10, usize::MAX - PAGE + 2, usize::MAX] {
        assert!(
            matches!(device.new_buffer(len), Err(GpuError::OutOfMemory(_))),
            "{len}"
        );
    }
    // With no limit, `active + size` must not wrap either.
    device.set_memory_limit(usize::MAX);
    let held = device.new_buffer(1000).unwrap();
    let active = device.memory().active;
    assert!(matches!(
        device.new_buffer(usize::MAX - 2 * PAGE),
        Err(GpuError::OutOfMemory(_))
    ));
    assert_eq!(device.memory().active, active);
    drop(held);
}

#[test]
fn the_limit_applies_to_the_charged_bucket() {
    let Some(device) = device() else { return };
    // A cached 3-page buffer is within the slack of a 2-page request, but charging it
    // would exceed a limit just above 2 pages.
    drop(device.new_buffer(3 * PAGE).unwrap());
    assert_eq!(device.memory().cache, 3 * PAGE);
    let limit = 2 * PAGE + 1;
    device.set_memory_limit(limit);
    match device.new_buffer(2 * PAGE) {
        Ok(buffer) => {
            assert!(device.memory().active <= limit, "{:?}", device.memory());
            drop(buffer);
        }
        Err(e) => assert!(matches!(e, GpuError::OutOfMemory(_)), "{e}"),
    }
    assert_eq!(device.memory().active, 0);
}

/// A temporary file of `len` bytes, removed on drop.
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(tag: &str, len: usize) -> Self {
        let path = std::env::temp_dir().join(format!("nuvec-{tag}-{}", std::process::id()));
        std::fs::write(&path, vec![7u8; len]).unwrap();
        TempFile(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `active_with_mapped` is `active + mapped`; `peak_with_mapped` is the highest
/// `active + mapped` (not the highest `active` plus `mapped`); `reset_peak` resets both.
#[test]
fn peak_with_mapped_tracks_active_plus_mapped() {
    let Some(device) = device() else { return };
    let file = TempFile::new("peak", 64 * PAGE - 5); // mapped page-rounded: 64 pages
    let a = device.new_buffer(32 * PAGE).unwrap();
    drop(a); // peak 32 pages, nothing mapped yet
    let mapped = device.map_file(&file.0).unwrap();
    let m = device.memory();
    assert_eq!(m.mapped, 64 * PAGE);
    assert_eq!(m.active_with_mapped, m.active + m.mapped);
    // max(active) + mapped would be 96 pages; the high-water mark of the sum is 64.
    assert_eq!((m.peak, m.peak_with_mapped), (32 * PAGE, 64 * PAGE));
    let b = device.new_buffer(8 * PAGE).unwrap();
    assert_eq!(device.memory().peak_with_mapped, 72 * PAGE);
    drop(b);
    device.reset_peak();
    let m = device.memory();
    assert_eq!(
        (m.peak, m.peak_with_mapped),
        (m.active, m.active + m.mapped)
    );
    assert_eq!(m.peak_with_mapped, 64 * PAGE);
    drop(mapped);
    let m = device.memory();
    assert_eq!((m.mapped, m.active_with_mapped), (0, m.active));
}

/// The memory limit bounds `active + mapped`: an allocation or a mapping past it is
/// `OutOfMemory`, and a refused mapping is neither counted nor wired.
#[test]
fn memory_limit_counts_mapped_files() {
    let Some(device) = device() else { return };
    let file = TempFile::new("limit", 16 * PAGE);
    device.set_wired_limit(64 * PAGE).unwrap();
    device.set_memory_limit(20 * PAGE);
    let mapped = device.map_file(&file.0).unwrap();
    // 16 pages mapped: 4 pages fit, 5 do not.
    assert_eq!(
        device.new_buffer(5 * PAGE).unwrap_err(),
        GpuError::OutOfMemory(5 * PAGE)
    );
    let small = device.new_buffer(4 * PAGE).unwrap();
    // A second mapping would pass the limit.
    let before = device.memory();
    assert!(matches!(
        device.map_file(&file.0),
        Err(GpuError::OutOfMemory(_))
    ));
    let after = device.memory();
    assert_eq!((after.mapped, after.wired), (before.mapped, before.wired));
    // A limit below what is in use is accepted; the next allocation fails.
    assert_eq!(device.set_memory_limit(PAGE), 20 * PAGE);
    assert!(device.new_buffer(256).is_err());
    drop((small, mapped));
    device.set_memory_limit(1 << 40);
    device.set_wired_limit(0).unwrap();
}

/// Wired bytes never pass the wired limit (mapped files count against it), mapping never
/// changes the limit, and a limit of 0 wires nothing; without residency sets nothing is
/// wired at all.
#[test]
fn wired_limit_invariants_hold_with_mapped_files() {
    let Some(device) = device() else { return };
    let file = TempFile::new("wired", 6 * PAGE);
    let check = |d: &Device| {
        let m = d.memory();
        assert!(m.wired <= m.wired_limit, "{m:?}");
        assert!(m.wired_mapped <= m.wired.min(m.mapped), "{m:?}");
        if !d.can_wire() {
            assert_eq!(m.wired, 0);
        }
    };
    // Limit 0: nothing is wired.
    let a = device.map_file(&file.0).unwrap();
    let b = device.new_buffer(4 * PAGE).unwrap();
    assert_eq!(device.memory().wired, 0);
    assert_eq!(
        device.memory().wired_limit,
        0,
        "mapping never changes the limit"
    );
    check(&device);
    // A limit set after mapping: the mapped file and buffers fill it, never past it.
    device.set_wired_limit(8 * PAGE).unwrap();
    check(&device);
    let c = device.map_file(&file.0).unwrap();
    assert_eq!(device.memory().wired_limit, 8 * PAGE);
    check(&device);
    if device.can_wire() {
        // The file (6 pages) or the buffer (4) fits, not both; the second file neither.
        assert!(device.memory().wired >= 4 * PAGE);
    }
    device.set_wired_limit(3 * PAGE).unwrap();
    check(&device);
    drop((a, b, c));
    device.clear_cache();
    check(&device);
    device.set_wired_limit(0).unwrap();
    assert_eq!(device.memory().wired, 0);
}

/// Mapped files count toward the GC threshold: a model's mapped weights plus active and
/// cached buffers must not grow past the working set (a 32K-token prefill of
/// Qwen3-30B-A3B failed with a command-buffer out-of-memory error before).
#[test]
fn gc_threshold_counts_mapped_files() {
    let Some(device) = device() else { return };
    let file = TempFile::new("gc", 16 * PAGE);
    device.set_memory_limit(20 * PAGE);
    let mapped = device.map_file(&file.0).unwrap();
    drop(device.new_buffer(2 * PAGE).unwrap());
    assert_eq!(device.memory().cache, 2 * PAGE);
    // 3 active + 16 mapped + 2 cached >= 20: the cache is released first.
    let big = device.new_buffer(3 * PAGE).unwrap();
    let m = device.memory();
    assert_eq!((m.active, m.mapped, m.cache), (3 * PAGE, 16 * PAGE, 0));
    drop((big, mapped));
}
