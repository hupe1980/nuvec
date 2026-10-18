//! The Metal runtime end to end. Every test skips when no Metal device is available.

#![cfg(target_os = "macos")]

use nuvec_gpu::{Arg, Buffer, Device, GpuError};

fn device() -> Option<Device> {
    let device = Device::system_default();
    if device.is_none() {
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
    cmd.dispatch(
        &pipeline,
        &[
            Arg::read(a),
            Arg::read(b),
            Arg::write(out),
            Arg::bytes(&n.to_le_bytes()),
        ],
        n as usize,
    )
    .unwrap();
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
fn placeholder_kernel() {
    let Some(device) = device() else { return };
    let pipeline = device.pipeline("nuvec_placeholder").unwrap();
    let mut out = device.new_buffer(1000 * 4).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    cmd.dispatch(&pipeline, &[Arg::write(&mut out)], 1000)
        .unwrap();
    cmd.commit().wait().unwrap();
    assert_eq!(
        read_f32(&out),
        (0..1000).map(|i| 2.0 * i as f32).collect::<Vec<_>>()
    );
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
    let past_end = cmd.dispatch(&pipeline, &[Arg::Read(&small, 17)], 4);
    assert!(matches!(past_end, Err(GpuError::InvalidArgument(_))));
    let bytes = [0u8; 4];
    let many: Vec<Arg<'_>> = (0..32).map(|_| Arg::bytes(&bytes)).collect();
    assert!(matches!(
        cmd.dispatch(&pipeline, &many, 4),
        Err(GpuError::InvalidArgument(_))
    ));
    let big = vec![0u8; 5000];
    assert!(matches!(
        cmd.dispatch(&pipeline, &[Arg::bytes(&big)], 4),
        Err(GpuError::InvalidArgument(_))
    ));
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
    cmd.dispatch(&pipeline, &[], 0).unwrap();
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
        cmd.commit_then(move |r| tx.send(r).unwrap());
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
        cmd.commit_then(move |r| tx.send((r, read_f32(&out)[0])).unwrap());
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
        cmd.commit_then(|r| r.unwrap());
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
        cmd.dispatch(&pipeline, &[Arg::write(&mut out)], 4).unwrap();
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
