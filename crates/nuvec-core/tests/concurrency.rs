//! Concurrent evaluation of shared graphs: threads evaluating clones of the same lazy
//! arrays never deadlock, each node runs once, and every thread sees the same result (or
//! the same error).

mod common;

use std::sync::{Arc, Barrier};
use std::time::Duration;

use common::{require_gpu, with_timeout};
use nuvec_core::ops;
use nuvec_core::*;

/// A four-operation lazy chain over `x`, recorded on `device`.
fn chain(x: &Array, device: Device) -> Array {
    set_default_stream(Stream::default_for(device));
    let a = (x * 0.5f32).exp();
    let b = &a + x;
    let c = b.sin();
    &c * &a
}

/// `threads` threads each evaluate their clone of the same chain, `iterations` times; the
/// results equal a single-threaded evaluation.
fn stress(device: Device, threads: usize, iterations: usize) {
    // A few iterations under Miri, which checks the interleavings it explores for races.
    let iterations = if cfg!(miri) {
        iterations.min(3)
    } else {
        iterations
    };
    let x = (0..256).map(|i| i as f32 / 64.0).collect::<Array>();
    let expected = chain(&x, device).to_vec::<f32>().unwrap();
    with_timeout(Duration::from_secs(30), move || {
        for _ in 0..iterations {
            let y = chain(&x, device);
            let barrier = Arc::new(Barrier::new(threads));
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let (y, barrier) = (y.clone(), barrier.clone());
                    std::thread::spawn(move || {
                        set_default_stream(Stream::default_for(device));
                        barrier.wait();
                        y.eval().unwrap();
                        y.to_vec::<f32>().unwrap()
                    })
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), expected);
            }
        }
    });
}

#[test]
fn shared_chains_evaluate_concurrently_on_the_cpu() {
    stress(Device::cpu(), 4, 200);
}

#[test]
fn shared_chains_evaluate_concurrently_on_the_gpu() {
    if require_gpu().is_none() {
        return;
    }
    stress(Device::gpu(0), 2, 1000);
    stress(Device::gpu(0), 4, 1000);
}

/// Inner nodes shared between two different consumers on two threads: each thread
/// evaluates its own consumer, both depending on the same unevaluated chain.
#[test]
fn shared_intermediates_with_different_consumers_on_the_gpu() {
    if require_gpu().is_none() {
        return;
    }
    let device = Device::gpu(0);
    with_timeout(Duration::from_secs(30), move || {
        for i in 0..500 {
            set_default_stream(Stream::default_for(device));
            let x = Array::from([i as f32 * 0.01, 1.0, 2.0, 3.0]);
            let shared = chain(&x, device);
            let left = &shared + 1.0f32;
            let right = &shared * 2.0f32;
            let expected_shared = shared.to_vec::<f32>().unwrap();
            // A second copy of the graph, evaluated concurrently.
            let shared = chain(&x, device);
            let (l, r) = (&shared + 1.0f32, &shared * 2.0f32);
            let barrier = Arc::new(Barrier::new(2));
            let spawn = |a: Array, barrier: Arc<Barrier>| {
                std::thread::spawn(move || {
                    set_default_stream(Stream::default_for(device));
                    barrier.wait();
                    a.to_vec::<f32>().unwrap()
                })
            };
            let (hl, hr) = (spawn(l, barrier.clone()), spawn(r, barrier));
            assert_eq!(hl.join().unwrap(), left.to_vec::<f32>().unwrap());
            assert_eq!(hr.join().unwrap(), right.to_vec::<f32>().unwrap());
            assert_eq!(shared.to_vec::<f32>().unwrap(), expected_shared);
        }
    });
}

/// A shared node that fails: every thread evaluating it gets the same error, including
/// threads that wait for the one running it, and nobody hangs.
fn failure_reaches_every_waiter(device: Device) {
    with_timeout(Duration::from_secs(30), move || {
        for i in 0..100 {
            set_default_stream(Stream::default_for(device));
            let a = ops::full([96, 96], 0.01f32 * i as f32);
            // Slow enough that the other thread usually arrives while it runs.
            let mut slow = a.clone();
            for _ in 0..4 {
                slow = slow.matmul(&a);
            }
            let failing = &slow + &graph::placeholder(DType::F32, &[96, 96]).unwrap();
            let top = failing.sum(..);
            let barrier = Arc::new(Barrier::new(2));
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    let (top, barrier) = (top.clone(), barrier.clone());
                    std::thread::spawn(move || {
                        set_default_stream(Stream::default_for(device));
                        barrier.wait();
                        top.eval().unwrap_err().to_string()
                    })
                })
                .collect();
            let errors: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            assert_eq!(errors[0], errors[1]);
            assert!(errors[0].contains("traced"), "{}", errors[0]);
        }
    });
}

#[test]
fn a_failed_evaluation_reaches_every_waiter_on_the_cpu() {
    failure_reaches_every_waiter(Device::cpu());
}

#[test]
fn a_failed_evaluation_reaches_every_waiter_on_the_gpu() {
    if require_gpu().is_none() {
        return;
    }
    failure_reaches_every_waiter(Device::gpu(0));
}
