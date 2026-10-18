//! The caching allocator. Every test uses its own device (and so its own pool) and skips
//! when no Metal device is available.

#![cfg(target_os = "macos")]

use nuvec_gpu::{Arg, Device, GpuError};

const PAGE: usize = 16 * 1024;

fn device() -> Option<Device> {
    let device = Device::system_default();
    if device.is_none() {
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
        cmd.dispatch(
            &add,
            &[
                Arg::read(&a),
                Arg::read(&a),
                Arg::write(out),
                Arg::bytes(&(n as u32).to_le_bytes()),
            ],
            n,
        )
        .unwrap();
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
