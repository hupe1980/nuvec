//! Two ranks of one group in one process: on the CPU stream they exchange data; on the GPU
//! (one command stream per GPU) their collectives fail at once, naming the ranks, instead
//! of waiting for each other until the timeout.

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use nuvec_core::distributed::{self, Group};
use nuvec_core::{Array, DType, Device, Stream, set_default_stream};
use nuvec_distributed::{Options, init_with};

mod common;

/// Runs `f(rank, group)` for ranks 0 and 1 of one loopback ring, each on its own thread,
/// once both have joined.
fn ranks<T: Send + 'static>(f: impl Fn(usize, Group) -> T + Send + Sync + 'static) -> Vec<T> {
    let nodes: Vec<Vec<String>> = common::free_addresses(2)
        .into_iter()
        .map(|a| vec![a])
        .collect();
    let (f, joined) = (Arc::new(f), Arc::new(Barrier::new(2)));
    let threads: Vec<_> = (0..2)
        .map(|rank| {
            let (nodes, f, joined) = (nodes.clone(), f.clone(), joined.clone());
            std::thread::spawn(move || {
                let options = Options::default().timeout(Duration::from_secs(5));
                let group = init_with(&nodes, rank, &options).unwrap();
                joined.wait();
                let out = f(rank, group.clone());
                joined.wait();
                group.close();
                out
            })
        })
        .collect();
    threads.into_iter().map(|t| t.join().unwrap()).collect()
}

fn exchange(rank: usize, group: &Group) -> nuvec_core::Result<Vec<f32>> {
    if rank == 0 {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0], [3]);
        distributed::send(&x, 1, Some(group)).eval()?;
        Ok(Vec::new())
    } else {
        let y = distributed::recv([3], DType::F32, 0, Some(group));
        y.eval()?;
        y.to_vec::<f32>()
    }
}

#[test]
fn ranks_in_one_process_work_on_the_cpu() {
    let got = ranks(|rank, g| {
        set_default_stream(Stream::default_for(Device::cpu()));
        exchange(rank, &g).unwrap()
    });
    assert_eq!(got[1], [1.0, 2.0, 3.0]);
}

#[test]
fn ranks_in_one_process_fail_at_once_on_the_gpu() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let got = ranks(|rank, g| {
        set_default_stream(Stream::default_for(Device::gpu(0)));
        let start = Instant::now();
        let e = exchange(rank, &g).unwrap_err().to_string();
        (start.elapsed(), e)
    });
    for (took, e) in got {
        assert!(took < Duration::from_millis(100), "took {took:?}");
        assert!(e.contains("ranks") && e.contains("one process"), "{e}");
    }
    // Closed groups no longer count: a group of this process alone works.
    let alone = init_with(&[vec!["127.0.0.1:1".into()]], 0, &Options::default()).unwrap();
    assert_eq!(alone.size(), 1);
}
