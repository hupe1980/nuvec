//! The Metal runtime end to end. Every test skips when no Metal device is available.

#![cfg(target_os = "macos")]

use nuvec_gpu::{Arg, Buffer, Device, GpuError};

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

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(buffer: &Buffer) -> Vec<f32> {
    buffer
        .contents()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// `out = a + b` over `n` floats.
fn add(
    cmd: &mut nuvec_gpu::CommandBuffer,
    device: &Device,
    a: &Buffer,
    b: &Buffer,
    out: &mut Buffer,
    n: u32,
) {
    let pipeline = device.pipeline("nuvec_add_f32").unwrap();
    try_add(cmd, &pipeline, a, b, out, n).unwrap();
}

/// `out = a + b` over `n` floats, reporting a rejected dispatch.
fn try_add(
    cmd: &mut nuvec_gpu::CommandBuffer,
    pipeline: &nuvec_gpu::Pipeline,
    a: &Buffer,
    b: &Buffer,
    out: &mut Buffer,
    n: u32,
) -> nuvec_gpu::Result<()> {
    let threads = (n as usize)
        .min(a.len() / 4)
        .min(b.len() / 4)
        .min(out.len() / 4);
    let n = n.to_le_bytes();
    let args = [Arg::read(a), Arg::read(b), Arg::write(out), Arg::bytes(&n)];
    // SAFETY: `nuvec_add_f32` touches element `i` of each buffer only for `i < n` and
    // `i < threads`, and `threads` is at most every buffer's length in floats.
    unsafe { cmd.dispatch(pipeline, &args, threads) }
}

/// Runs `f` on another thread and fails the test if it takes longer than `limit` (a hang
/// fails by timeout instead of blocking the suite).
fn with_timeout<T: Send + 'static>(
    limit: std::time::Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)));
    });
    match rx.recv_timeout(limit) {
        Ok(Ok(value)) => value,
        Ok(Err(panic)) => std::panic::resume_unwind(panic),
        Err(_) => panic!("timed out after {limit:?}"),
    }
}

#[test]
fn device_info() {
    let Some(device) = device() else { return };
    let info = device.info();
    assert!(!info.name.is_empty());
    assert!(
        info.architecture.starts_with("applegpu_g") || info.generation.is_none(),
        "{info:?}"
    );
    assert!(info.recommended_working_set > 0);
    assert!(info.max_threads_per_threadgroup >= 256);
}

#[test]
fn add_one_million() {
    let Some(device) = device() else { return };
    let n = 1_000_000u32;
    let a = device
        .buffer_from(&f32s(&(0..n).map(|i| i as f32).collect::<Vec<_>>()))
        .unwrap();
    let b = device.buffer_from(&f32s(&vec![0.5; n as usize])).unwrap();
    let mut out = device.new_buffer(n as usize * 4).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    add(&mut cmd, &device, &a, &b, &mut out, n);
    cmd.commit().wait().unwrap();
    let got = read_f32(&out);
    assert!(got.iter().enumerate().all(|(i, &v)| v == i as f32 + 0.5));
}

#[test]
fn pipelines_are_cached() {
    let Some(device) = device() else { return };
    let first = device.pipeline("nuvec_add_f32").unwrap();
    let second = device.pipeline("nuvec_add_f32").unwrap();
    assert!(first.same_as(&second));
    assert_eq!(first.name(), "nuvec_add_f32");
    assert!(first.execution_width() >= 1);
}

/// The loaded kernel library (precompiled or compiled from source) exports the kernels by
/// name, as `newFunctionWithName` finds them.
#[test]
fn precompiled_library_contains_the_kernels() {
    let Some(device) = device() else { return };
    for name in [
        "nuvec_binary_add_float32_c",
        "nuvec_add_f32",
        "nuvec_unary_exp_float32_c",
    ] {
        let pipeline = device
            .pipeline(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(pipeline.name(), name);
    }
    assert!(matches!(
        device.pipeline("nuvec_no_such_kernel"),
        Err(GpuError::MissingKernel(_))
    ));
}

#[test]
fn barriers_only_for_dependencies() {
    let Some(device) = device() else { return };
    let n = 4096u32;
    let one = device.buffer_from(&f32s(&vec![1.0; n as usize])).unwrap();

    // A chain x[i+1] = x[i] + 1: every dispatch after the first depends on its predecessor.
    let mut xs: Vec<Buffer> = (0..101)
        .map(|_| device.new_buffer(n as usize * 4).unwrap())
        .collect();
    let mut cmd = device.command_buffer().unwrap();
    for i in 0..100 {
        let (done, rest) = xs.split_at_mut(i + 1);
        add(&mut cmd, &device, &done[i], &one, &mut rest[0], n);
    }
    assert_eq!(cmd.barriers(), 99);
    cmd.commit().wait().unwrap();
    assert!(
        read_f32(&xs[100]).iter().all(|&v| v == 100.0),
        "a missing barrier would read stale data"
    );

    // Independent dispatches: no barriers.
    let mut outs: Vec<Buffer> = (0..100)
        .map(|_| device.new_buffer(n as usize * 4).unwrap())
        .collect();
    let mut cmd = device.command_buffer().unwrap();
    for out in &mut outs {
        add(&mut cmd, &device, &one, &one, out, n);
    }
    assert_eq!(cmd.barriers(), 0);
    cmd.commit().wait().unwrap();
    assert!(outs.iter().all(|o| read_f32(o).iter().all(|&v| v == 2.0)));
}

#[test]
fn write_after_read_needs_a_barrier() {
    let Some(device) = device() else { return };
    let n = 1024u32;
    let mut a = device.buffer_from(&f32s(&vec![1.0; n as usize])).unwrap();
    let c = device.buffer_from(&f32s(&vec![5.0; n as usize])).unwrap();
    let mut b = device.new_buffer(n as usize * 4).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    add(&mut cmd, &device, &a, &a, &mut b, n); // reads a
    add(&mut cmd, &device, &c, &c, &mut a, n); // then overwrites a
    assert_eq!(cmd.barriers(), 1);
    cmd.commit().wait().unwrap();
    assert!(
        read_f32(&b).iter().all(|&v| v == 2.0),
        "b must see the old a"
    );
    assert!(read_f32(&a).iter().all(|&v| v == 10.0));
}

#[test]
fn misuse_is_reported() {
    let Some(device) = device() else { return };
    assert_eq!(
        device.pipeline("no_such_kernel").unwrap_err(),
        GpuError::MissingKernel("no_such_kernel".into())
    );
    let pipeline = device.pipeline("nuvec_add_f32").unwrap();
    let small = device.new_buffer(16).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    // SAFETY: rejected before anything is encoded (as the dispatches below).
    let past_end = unsafe { cmd.dispatch(&pipeline, &[Arg::Read(&small, 17)], 4) };
    assert!(matches!(past_end, Err(GpuError::InvalidArgument(_))));
    let bytes = [0u8; 4];
    let many: Vec<Arg<'_>> = (0..32).map(|_| Arg::bytes(&bytes)).collect();
    // SAFETY: rejected.
    let too_many = unsafe { cmd.dispatch(&pipeline, &many, 4) };
    assert!(matches!(too_many, Err(GpuError::InvalidArgument(_))));
    let big = vec![0u8; 5000];
    // SAFETY: rejected.
    let too_big = unsafe { cmd.dispatch(&pipeline, &[Arg::bytes(&big)], 4) };
    assert!(matches!(too_big, Err(GpuError::InvalidArgument(_))));
    assert_eq!(cmd.barriers(), 0, "rejected dispatches encode nothing");
}

#[test]
fn cpu_access_waits_for_the_gpu() {
    let Some(device) = device() else { return };
    let n = 1 << 20;
    let mut a = device.buffer_from(&f32s(&vec![1.0; n as usize])).unwrap();
    let mut out = device.new_buffer(n as usize * 4).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    add(&mut cmd, &device, &a, &a, &mut out, n);
    let submitted = cmd.commit();
    assert!(out.in_use());
    assert_eq!(out.contents().unwrap_err(), GpuError::BufferInUse);
    assert!(a.in_use());
    assert_eq!(
        a.contents_mut().unwrap_err(),
        GpuError::BufferInUse,
        "inputs refuse CPU writes while the GPU reads them"
    );
    drop(submitted); // waits
    assert!(!out.in_use());
    assert_eq!(read_f32(&out)[0], 2.0);
}

#[test]
fn abandoned_command_buffer_releases_buffers() {
    let Some(device) = device() else { return };
    let a = device.buffer_from(&f32s(&[1.0; 4])).unwrap();
    let mut out = device.new_buffer(16).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    add(&mut cmd, &device, &a, &a, &mut out, 4);
    drop(cmd); // never submitted
    assert!(!out.in_use() && !a.in_use());
    assert!(out.contents().is_ok());
}

#[test]
fn empty_buffers_and_zero_threads() {
    let Some(device) = device() else { return };
    let empty = device.new_buffer(0).unwrap();
    assert!(empty.is_empty() && empty.contents().unwrap().is_empty());
    let pipeline = device.pipeline("nuvec_add_f32").unwrap();
    let mut cmd = device.command_buffer().unwrap();
    // SAFETY: no threads run.
    unsafe { cmd.dispatch(&pipeline, &[], 0) }.unwrap();
    cmd.commit().wait().unwrap();
}

#[test]
fn timeline_orders_command_buffers_committed_without_waiting() {
    let Some(device) = device() else { return };
    let n = 1 << 16;
    let ones = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let mut x = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut y = device.new_buffer(n * 4).unwrap();
    let mut timeline = device.timeline().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    // 200 dependent additions, alternating between two buffers, one per command buffer.
    for i in 0..200 {
        let mut cmd = device.command_buffer_on(&mut timeline).unwrap();
        let (src, dst) = if i % 2 == 0 {
            (&x, &mut y)
        } else {
            (&y, &mut x)
        };
        add(&mut cmd, &device, src, &ones, dst, n as u32);
        let tx = tx.clone();
        drop(cmd.commit_then(move |r| tx.send(r).unwrap()));
    }
    for _ in 0..200 {
        rx.recv().unwrap().unwrap();
    }
    assert_eq!(device.in_flight(), 0);
    assert!(read_f32(&x).iter().all(|&v| v == 200.0));
}

#[test]
fn abandoned_command_buffers_keep_the_timeline_moving() {
    let Some(device) = device() else { return };
    let ones = device.buffer_from(&f32s(&[1.0; 4])).unwrap();
    let mut out = device.buffer_from(&f32s(&[5.0; 4])).unwrap();
    let mut timeline = device.timeline().unwrap();
    drop(device.command_buffer_on(&mut timeline).unwrap());
    let mut cmd = device.command_buffer_on(&mut timeline).unwrap();
    add(&mut cmd, &device, &ones, &ones, &mut out, 4);
    cmd.commit().wait().unwrap();
    assert_eq!(read_f32(&out), [2.0; 4]);
}

#[test]
fn completion_handlers_report_and_back_pressure_bounds_in_flight_work() {
    let Some(device) = device() else { return };
    let n = 1 << 20;
    let a = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let max_seen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for _ in 0..40 {
        device.wait_for_capacity(4);
        let seen = device.in_flight();
        max_seen.fetch_max(seen, std::sync::atomic::Ordering::SeqCst);
        let mut out = device.new_buffer(n * 4).unwrap();
        let mut cmd = device.command_buffer().unwrap();
        add(&mut cmd, &device, &a, &a, &mut out, n as u32);
        // CPU access is refused while the GPU uses the buffer.
        assert!(matches!(out.contents(), Err(GpuError::BufferInUse)));
        let tx = tx.clone();
        drop(cmd.commit_then(move |r| tx.send((r, read_f32(&out)[0])).unwrap()));
    }
    for _ in 0..40 {
        let (result, first) = rx.recv().unwrap();
        result.unwrap();
        assert_eq!(first, 2.0);
    }
    assert!(max_seen.load(std::sync::atomic::Ordering::SeqCst) < 4);
}

#[test]
fn ordered_command_buffers_read_in_flight_results_across_threads() {
    let Some(device) = device() else { return };
    let n = 1 << 18;
    let ones = std::sync::Arc::new(device.buffer_from(&f32s(&vec![1.0; n])).unwrap());
    // Thread A commits a long chain without waiting; thread B then reads its in-flight
    // result through an ordered command buffer.
    let mut x = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut y = device.new_buffer(n * 4).unwrap();
    for i in 0..50 {
        let mut cmd = device.ordered_command_buffer().unwrap();
        let (src, dst) = if i % 2 == 0 {
            (&x, &mut y)
        } else {
            (&y, &mut x)
        };
        add(&mut cmd, &device, src, &ones, dst, n as u32);
        drop(cmd.commit_then(|r| r.unwrap()));
    }
    let (d, o) = (device.clone(), ones.clone());
    let reader = std::thread::spawn(move || {
        let mut out = d.new_buffer(n * 4).unwrap();
        let mut cmd = d.ordered_command_buffer().unwrap();
        add(&mut cmd, &d, &x, &o, &mut out, n as u32);
        cmd.commit().wait().unwrap();
        read_f32(&out)
    });
    let v = reader.join().unwrap();
    assert!(v.iter().all(|&e| e == 51.0), "{}", v[0]);
}

#[test]
fn cpu_reads_may_overlap_gpu_reads_but_not_gpu_writes() {
    let Some(device) = device() else { return };
    let n = 1 << 20;
    let mut a = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let mut out = device.new_buffer(n * 4).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    add(&mut cmd, &device, &a, &a, &mut out, n as u32);
    let submitted = cmd.commit();
    assert!(a.contents().is_ok(), "a is only read by the GPU");
    assert!(
        matches!(a.contents_mut(), Err(GpuError::BufferInUse)),
        "no CPU writes while read"
    );
    assert!(
        matches!(out.contents(), Err(GpuError::BufferInUse)),
        "out is being written"
    );
    submitted.wait().unwrap();
    assert_eq!(read_f32(&out)[0], 2.0);
    assert!(a.contents_mut().is_ok());
}

#[test]
fn function_constants_specialise_and_cache_pipelines() {
    let Some(device) = device() else { return };
    for (flag, expected) in [(false, 1.0), (true, 2.0), (false, 1.0)] {
        let pipeline = device
            .pipeline_with("nuvec_function_constant_test", &[(300, flag)])
            .unwrap();
        let mut out = device.new_buffer(16).unwrap();
        let mut cmd = device.command_buffer().unwrap();
        // SAFETY: the kernel writes `out[i]` for the 4 threads; `out` holds 4 floats.
        unsafe { cmd.dispatch(&pipeline, &[Arg::write(&mut out)], 4) }.unwrap();
        cmd.commit().wait().unwrap();
        assert_eq!(read_f32(&out), [expected; 4]);
    }
}

#[test]
fn ordered_command_buffers_wait_only_for_their_writers_and_overlap_otherwise() {
    let Some(device) = device() else { return };
    // Two interleaved chains (`x += 1`, `y += 2`), one command buffer per step, all committed
    // without waiting; a final command buffer reads both chains (a wait on two writers).
    let n = 1 << 22;
    let ones = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let twos = device.buffer_from(&f32s(&vec![2.0; n])).unwrap();
    let mut x = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut y = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut pending = Vec::new();
    let steps = 48;
    for _ in 0..steps {
        for (acc, step) in [(&mut x, &ones), (&mut y, &twos)] {
            let mut next = device.new_buffer(n * 4).unwrap();
            let mut cmd = device.ordered_command_buffer().unwrap();
            add(&mut cmd, &device, acc, step, &mut next, n as u32);
            pending.push(cmd.commit_then(|r| r.unwrap()));
            *acc = next;
        }
    }
    let mut sum = device.new_buffer(n * 4).unwrap();
    let mut cmd = device.ordered_command_buffer().unwrap();
    add(&mut cmd, &device, &x, &y, &mut sum, n as u32);
    cmd.commit_then(|r| r.unwrap()).wait();
    for p in pending {
        p.wait();
    }
    let want = 3.0 * steps as f32;
    assert!(read_f32(&sum).iter().all(|&v| v == want));
    assert!(read_f32(&x).iter().all(|&v| v == steps as f32));
}

#[test]
fn ordered_writes_wait_for_in_flight_readers() {
    let Some(device) = device() else { return };
    // A slow chain reads `a`; a later command buffer overwrites `a`. The chain must see the
    // old values (write after read across command buffers).
    // Small kernels, so that independent command buffers overlap on the GPU.
    let n = 1 << 14;
    let mut a = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let zeros = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut acc = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut pending = Vec::new();
    for _ in 0..200 {
        let mut next = device.new_buffer(n * 4).unwrap();
        let mut cmd = device.ordered_command_buffer().unwrap();
        add(&mut cmd, &device, &acc, &a, &mut next, n as u32);
        pending.push(cmd.commit_then(|r| r.unwrap()));
        acc = next;
    }
    // Overwrite `a` with zeros while the chain may still be reading it.
    let mut cmd = device.ordered_command_buffer().unwrap();
    add(&mut cmd, &device, &zeros, &zeros, &mut a, n as u32);
    cmd.commit_then(|r| r.unwrap()).wait();
    for p in pending {
        p.wait();
    }
    assert!(read_f32(&a).iter().all(|&v| v == 0.0));
    assert!(read_f32(&acc).iter().all(|&v| v == 200.0));
}

#[test]
fn reused_memory_is_written_after_in_flight_readers() {
    let Some(device) = device() else { return };
    // A slow chain reads `a`; `a` is dropped while the chain is in flight, and its memory
    // reused for a buffer later work overwrites. The chain must still see `a`'s values.
    let n = 1 << 14;
    let a = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let zeros = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut acc = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut pending = Vec::new();
    for _ in 0..200 {
        let mut next = device.new_buffer(n * 4).unwrap();
        let mut cmd = device.ordered_command_buffer().unwrap();
        add(&mut cmd, &device, &acc, &a, &mut next, n as u32);
        pending.push(cmd.commit_then(|r| r.unwrap()));
        acc = next;
    }
    let before = device.memory().system_allocations;
    drop(a);
    let mut reused = device.new_gpu_buffer(n * 4).unwrap();
    let reused_a = device.memory().system_allocations == before;
    let mut cmd = device.ordered_command_buffer().unwrap();
    add(&mut cmd, &device, &zeros, &zeros, &mut reused, n as u32);
    cmd.commit_then(|r| r.unwrap()).wait();
    for p in pending {
        p.wait();
    }
    assert!(reused_a, "`a`'s memory is reused");
    assert!(read_f32(&reused).iter().all(|&v| v == 0.0));
    assert!(read_f32(&acc).iter().all(|&v| v == 200.0));
}

#[test]
fn mapped_files_are_read_without_a_copy_and_never_written() {
    let Some(device) = device() else { return };
    let path = std::env::temp_dir().join(format!("nuvec-gpu-map-{}", std::process::id()));
    let n = 5000; // not a whole number of pages
    std::fs::write(&path, f32s(&(0..n).map(|i| i as f32).collect::<Vec<_>>())).unwrap();
    let mut mapped = device.map_file(&path).unwrap();
    assert_eq!(mapped.len(), n * 4);
    assert_eq!(read_f32(&mapped)[4999], 4999.0);
    assert_eq!(mapped.contents_mut().unwrap_err(), GpuError::ReadOnly);
    // The GPU reads it in place.
    let ones = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let mut out = device.new_buffer(n * 4).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    add(&mut cmd, &device, &mapped, &ones, &mut out, n as u32);
    cmd.commit().wait().unwrap();
    assert_eq!(read_f32(&out)[10], 11.0);
    // GPU writes are refused.
    let mut cmd = device.command_buffer().unwrap();
    let pipeline = device.pipeline("nuvec_add_f32").unwrap();
    // SAFETY: rejected before anything is encoded.
    let err = unsafe {
        cmd.dispatch(
            &pipeline,
            &[Arg::read(&ones), Arg::read(&ones), Arg::write(&mut mapped)],
            n,
        )
    };
    assert_eq!(err.unwrap_err(), GpuError::ReadOnly);
    drop(mapped);
    std::fs::remove_file(&path).unwrap();
    assert!(device.map_file(&path).is_err());
}

#[test]
fn same_must_refer_to_a_written_argument() {
    let Some(device) = device() else { return };
    let pipeline = device.pipeline("nuvec_add_f32").unwrap();
    let a = device.buffer_from(&f32s(&[1.0; 4])).unwrap();
    let mut out = device.new_buffer(16).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    let n = 4u32.to_le_bytes();
    // `Same` of a read-only argument would let the kernel write a buffer nobody counts as
    // written.
    let args = [
        Arg::read(&a),
        Arg::Same(0, 0),
        Arg::write(&mut out),
        Arg::bytes(&n),
    ];
    // SAFETY: rejected before anything is encoded.
    let result = unsafe { cmd.dispatch(&pipeline, &args, 4) };
    assert!(
        matches!(result, Err(GpuError::InvalidArgument(_))),
        "{result:?}"
    );
    // `Same` of a written argument is the in-place pattern.
    let args = [
        Arg::read(&a),
        Arg::Same(2, 0),
        Arg::write(&mut out),
        Arg::bytes(&n),
    ];
    // SAFETY: four floats in each bound buffer, four threads.
    unsafe { cmd.dispatch(&pipeline, &args, 4) }.unwrap();
    cmd.commit().wait().unwrap();
}

#[test]
fn wait_returns_after_the_completion_has_run() {
    let Some(device) = device() else { return };
    let a = device.buffer_from(&f32s(&[1.0; 64])).unwrap();
    let mut out = device.new_buffer(64 * 4).unwrap();
    for _ in 0..10_000 {
        let mut cmd = device.ordered_command_buffer().unwrap();
        add(&mut cmd, &device, &a, &a, &mut out, 64);
        cmd.commit_then(|r| r.unwrap()).wait();
        assert_eq!(device.in_flight(), 0);
        assert!(!out.in_use());
        assert_eq!(out.contents().unwrap().len(), 64 * 4);
    }
}

#[test]
fn reading_reused_memory_waits_for_the_previous_owner() {
    with_timeout(std::time::Duration::from_secs(30), || {
        let Some(device) = device() else { return };
        let n = 1 << 20;
        let a = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
        device.clear_cache();
        let mut out = device.new_buffer(n * 4).unwrap();
        let mut cmd = device.ordered_command_buffer().unwrap();
        add(&mut cmd, &device, &a, &a, &mut out, n as u32);
        let submitted = cmd.commit(); // not waited for
        drop(out);
        let before = device.memory().system_allocations;
        let reused = device.new_gpu_buffer(n * 4).unwrap();
        let was_reused = device.memory().system_allocations == before;
        // Returns once the GPU has finished with the memory.
        assert_eq!(reused.contents().unwrap().len(), n * 4);
        assert!(was_reused, "the in-flight memory is reused");
        submitted.wait().unwrap();
    });
}

#[test]
fn unordered_work_on_busy_memory_and_reads_of_unsubmitted_writes_are_refused() {
    let Some(device) = device() else { return };
    let pipeline = device.pipeline("nuvec_add_f32").unwrap();
    // A writer encoded but not committed: its fence orders nothing yet.
    let a = device.buffer_from(&f32s(&[1.0; 4])).unwrap();
    let mut x = device.new_buffer(16).unwrap();
    let mut y = device.new_buffer(16).unwrap();
    let mut writer = device.ordered_command_buffer().unwrap();
    add(&mut writer, &device, &a, &a, &mut x, 4);
    let mut reader = device.ordered_command_buffer().unwrap();
    assert_eq!(
        try_add(&mut reader, &pipeline, &x, &a, &mut y, 4),
        Err(GpuError::BufferInUse)
    );
    // The writer's own later dispatches may read it.
    let mut z = device.new_buffer(16).unwrap();
    add(&mut writer, &device, &x, &a, &mut z, 4);
    writer.commit().wait().unwrap();
    drop(reader);
    assert_eq!(read_f32(&z), [3.0; 4]);

    // Memory reused while its previous owner's ordered work runs: an unordered command
    // buffer does not wait for that work's fence, so it is refused.
    let n = 1 << 22;
    let ones = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let mut acc = device.buffer_from(&f32s(&vec![0.0; n])).unwrap();
    let mut chain = device.new_buffer(n * 4).unwrap();
    let mut cmd = device.ordered_command_buffer().unwrap();
    for _ in 0..50 {
        add(&mut cmd, &device, &acc, &ones, &mut chain, n as u32);
        std::mem::swap(&mut acc, &mut chain);
    }
    let pending = cmd.commit_then(|r| r.unwrap());
    device.clear_cache();
    drop(chain);
    let before = device.memory().system_allocations;
    let mut reused = device.new_gpu_buffer(n * 4).unwrap();
    let was_reused = device.memory().system_allocations == before;
    let mut unordered = device.command_buffer().unwrap();
    let result = try_add(
        &mut unordered,
        &pipeline,
        &ones,
        &ones,
        &mut reused,
        n as u32,
    );
    let still_running = device.in_flight() > 0;
    pending.wait();
    if was_reused && still_running {
        assert_eq!(result, Err(GpuError::BufferInUse));
    } else {
        eprintln!("inconclusive: the chain finished before the dispatch");
    }
    drop(unordered);
    assert!(read_f32(&acc).iter().all(|&v| v == 50.0));
}

/// Metal source for kernel `name` (`out[i] = 7`) with `count` distinct helper functions,
/// slow to compile for large counts. `salt` keeps Metal's compiler cache from returning an
/// earlier compilation.
fn slow_source(name: &str, count: usize, salt: u128) -> String {
    use std::fmt::Write;
    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    for k in 0..count {
        writeln!(
            source,
            "float f{k}_{salt}(float x) {{ return sin(x * {k}.5f) + cos(x - {k}.25f) \
             * tanh(x * 0.{k}f); }}"
        )
        .unwrap();
    }
    writeln!(
        source,
        "[[kernel]] void {name}(device float* out [[buffer(0)]], uint i \
         [[thread_position_in_grid]]) {{ float x = float(i); float s = 0;"
    )
    .unwrap();
    for k in 0..count {
        writeln!(source, "s += f{k}_{salt}(x);").unwrap();
    }
    source.push_str("out[i] = 7.0f + 0.0f * s; }\n");
    source
}

/// While one thread compiles a slow source kernel, another gets a cached pipeline at once
/// (the cache's lock is not held while compiling).
#[test]
fn compiling_does_not_block_cached_pipelines() {
    use std::time::{Duration, Instant};
    let Some(device) = device() else { return };
    // Loaded and cached before the slow compilation starts.
    device.pipeline("nuvec_add_f32").unwrap();
    // Compile times vary (Metal caches parts of compilations): grow the source until the
    // compilation outlasts the lookup by a clear margin.
    for count in [2000, 4000, 8000] {
        let salt = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let (started, compiling) = std::sync::mpsc::channel();
        let d = device.clone();
        let name = format!("nuvec_slow_test_{count}");
        let slow = std::thread::spawn(move || {
            let p = d.pipeline_from_source(&name, || {
                started.send(()).unwrap();
                slow_source(&name, count, salt)
            });
            (p, Instant::now())
        });
        compiling.recv().unwrap();
        std::thread::sleep(Duration::from_millis(20));
        let t = Instant::now();
        device.pipeline("nuvec_add_f32").unwrap();
        let lookup = t.elapsed();
        let (pipeline, done) = slow.join().unwrap();
        assert!(
            lookup < Duration::from_millis(100),
            "a cached lookup took {lookup:?} while another kernel compiled"
        );
        // Some GPUs (virtualised CI runners) refuse the deliberately huge kernel instead of
        // compiling it slowly; the property above still held while they tried.
        if let Err(e) = pipeline {
            eprintln!("inconclusive: the slow kernel did not compile here ({e})");
            return;
        }
        if done.duration_since(t) >= Duration::from_millis(500) {
            return; // the lookup ran early in a long compilation
        }
    }
    eprintln!("inconclusive: no source compiled slowly enough");
}

#[test]
fn source_pipelines_have_their_own_namespace_and_language_version() {
    let Some(device) = device() else { return };
    let library = device.pipeline("nuvec_add_f32").unwrap();
    // A source kernel named like a library kernel is its own pipeline.
    let source = device
        .pipeline_from_source("nuvec_add_f32", || {
            "#include <metal_stdlib>\nusing namespace metal;\n[[kernel]] void nuvec_add_f32(\
             device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) \
             { out[i] = 7.0f; }\n"
                .to_owned()
        })
        .unwrap();
    assert!(!source.same_as(&library));
    assert!(device.pipeline("nuvec_add_f32").unwrap().same_as(&library));
    let mut out = device.new_buffer(16).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    // SAFETY: the kernel writes `out[i]` for the 4 threads; `out` holds 4 floats.
    unsafe { cmd.dispatch(&source, &[Arg::write(&mut out)], 4) }.unwrap();
    cmd.commit().wait().unwrap();
    assert_eq!(read_f32(&out), [7.0; 4]);
    // Recent language features compile (`bfloat`, Metal 3.1+, as nuvec's kernels use).
    device
        .pipeline_from_source("nuvec_bfloat_test", || {
            "#include <metal_stdlib>\nusing namespace metal;\n[[kernel]] void \
             nuvec_bfloat_test(device bfloat* out [[buffer(0)]], uint i \
             [[thread_position_in_grid]]) { out[i] = bfloat(1.0f); }\n"
                .to_owned()
        })
        .unwrap();
}

/// Run under `MTL_DEBUG_LAYER=1` (as `just metal-check` does): two threads take turns
/// extending one chain through ordered command buffers committed without waiting, so each
/// reads the other's in-flight result through its fence.
#[test]
fn ordered_command_buffers_from_two_threads_stay_ordered() {
    let Some(device) = device() else { return };
    let n = 1024;
    let steps = 10_000;
    let ones = std::sync::Arc::new(device.buffer_from(&f32s(&vec![1.0; n])).unwrap());
    let acc = std::sync::Arc::new(std::sync::Mutex::new(
        device.buffer_from(&f32s(&vec![0.0; n])).unwrap(),
    ));
    let threads: Vec<_> = (0..2)
        .map(|_| {
            let (device, ones, acc) = (device.clone(), ones.clone(), acc.clone());
            std::thread::spawn(move || {
                for _ in 0..steps / 2 {
                    device.wait_for_capacity(64);
                    let mut acc = acc.lock().unwrap();
                    let mut next = device.new_gpu_buffer(n * 4).unwrap();
                    let mut cmd = device.ordered_command_buffer().unwrap();
                    add(&mut cmd, &device, &acc, &ones, &mut next, n as u32);
                    drop(cmd.commit_then(|r| r.unwrap()));
                    *acc = next;
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    with_timeout(std::time::Duration::from_secs(30), {
        let device = device.clone();
        move || device.wait_for_capacity(1)
    });
    let acc = acc.lock().unwrap();
    assert!(read_f32(&acc).iter().all(|&v| v == steps as f32));
}
