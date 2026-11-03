//! Shared events: GPU work that waits for a value runs only once the CPU signals it.

#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use nuvec_gpu::{Arg, Device, Event};

#[test]
fn a_waiting_command_buffer_runs_after_the_signal() {
    let Some(device) = Device::shared() else {
        return;
    };
    let event = Event::new(device).unwrap();
    let mut cmd = device.command_buffer().unwrap();
    cmd.wait_event(&event, 1);
    let pending = cmd.commit_then(|_| {});
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(event.value(), 0);
    let start = Instant::now();
    let signaler = std::thread::spawn({
        let event = std::sync::Arc::new(event);
        move || {
            std::thread::sleep(Duration::from_millis(20));
            event.signal(1);
            event
        }
    });
    pending.wait_result().unwrap();
    assert!(start.elapsed() >= Duration::from_millis(15));
    assert_eq!(signaler.join().unwrap().value(), 1);
}

#[test]
fn a_value_already_reached_does_not_wait() {
    let Some(device) = Device::shared() else {
        return;
    };
    let event = Event::new(device).unwrap();
    event.signal(5);
    let mut cmd = device.command_buffer().unwrap();
    cmd.wait_event(&event, 3);
    let start = Instant::now();
    cmd.commit_then(|_| {}).wait_result().unwrap();
    assert!(start.elapsed() < Duration::from_secs(1));
}

#[test]
fn a_cpu_wait_times_out_and_then_sees_the_signal() {
    let Some(device) = Device::shared() else {
        return;
    };
    let event = Event::new(device).unwrap();
    assert!(!event.wait(1, Duration::from_millis(5)));
    event.signal(2);
    assert!(event.wait(1, Duration::from_millis(5)));
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The pattern of a collective inside one command buffer: the GPU signals after the
/// producer, the CPU waits for that, rewrites the produced buffer and signals back, and the
/// consumer encoded after the wait reads the CPU's bytes.
#[test]
fn a_cpu_step_inside_one_command_buffer_is_ordered() {
    let Some(device) = Device::shared() else {
        return;
    };
    let n = 4096usize;
    let a = device.buffer_from(&f32s(&vec![1.0; n])).unwrap();
    let mut b = device.new_buffer(n * 4).unwrap();
    let mut c = device.new_buffer(n * 4).unwrap();
    let add = device.pipeline("nuvec_add_f32").unwrap();
    let len = (n as u32).to_le_bytes();
    let (ready, done) = (
        std::sync::Arc::new(Event::new(device).unwrap()),
        std::sync::Arc::new(Event::new(device).unwrap()),
    );
    let mut cmd = device.ordered_command_buffer().unwrap();
    let args = [
        Arg::read(&a),
        Arg::read(&a),
        Arg::write(&mut b),
        Arg::bytes(&len),
    ];
    // SAFETY: `nuvec_add_f32` touches element `i < n` of each buffer, all `n` floats long.
    unsafe { cmd.dispatch(&add, &args, n) }.unwrap();
    cmd.signal_event(&ready, 1);
    cmd.wait_event(&done, 1);
    let args = [
        Arg::read(&b),
        Arg::read(&b),
        Arg::write(&mut c),
        Arg::bytes(&len),
    ];
    // SAFETY: as above.
    unsafe { cmd.dispatch(&add, &args, n) }.unwrap();
    let (ptr, bytes) = b.unsynchronized_ptr().unwrap();
    let ptr = ptr.as_ptr() as usize;
    let cpu = std::thread::spawn({
        let (ready, done) = (ready.clone(), done.clone());
        move || {
            assert!(ready.wait(1, Duration::from_secs(5)));
            // SAFETY: the GPU wrote `b` before signalling `ready` and reads it only after
            // `done`; nothing else touches it meanwhile.
            let b = unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, bytes / 4) };
            assert!(b.iter().all(|&v| v == 2.0));
            b.fill(10.0);
            done.signal(1);
        }
    });
    cmd.commit_then(|_| {}).wait_result().unwrap();
    cpu.join().unwrap();
    let out: Vec<f32> = c
        .contents()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    assert!(out.iter().all(|&v| v == 20.0));
}

#[test]
fn a_deferred_command_buffer_and_those_after_it_wait_for_its_commit() {
    let Some(device) = Device::shared() else {
        return;
    };
    let (first, second) = (Event::new(device).unwrap(), Event::new(device).unwrap());
    let mut cmd = device.command_buffer().unwrap();
    cmd.signal_event(&first, 1);
    let (deferred_pending, deferred) = cmd.enqueue_then(|_| {});
    let mut cmd = device.command_buffer().unwrap();
    cmd.signal_event(&second, 1);
    let pending = cmd.commit_then(|_| {});
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!((first.value(), second.value()), (0, 0));
    // Waiting blocks until another thread commits it.
    let committer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        deferred.commit();
    });
    deferred_pending.wait_result().unwrap();
    pending.wait_result().unwrap();
    committer.join().unwrap();
    assert_eq!((first.value(), second.value()), (1, 1));
}
