//! Workloads compared against MLX by `scripts/bench_vs_mlx.py` (`just bench-mlx`).
//!
//! Prints one `name<TAB>median_us` line per workload. Each workload is warmed up, then
//! timed over repeated runs; the median is reported.

use std::time::Instant;

use nuvec_core::{Array, DType, Device, Result, Stream, ops, set_default_stream};

fn median_us(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2] * 1e6
}

fn bench(name: &str, reps: usize, mut run: impl FnMut() -> Result<()>) -> Result<()> {
    for _ in 0..3 {
        run()?;
    }
    let samples = (0..reps)
        .map(|_| {
            let start = Instant::now();
            run().map(|()| start.elapsed().as_secs_f64())
        })
        .collect::<Result<Vec<_>>>()?;
    println!("{name}\t{:.1}", median_us(samples));
    Ok(())
}

fn evaluated(a: Array) -> Result<Array> {
    a.eval()?;
    Ok(a)
}

fn main() -> Result<()> {
    // `cargo test --all-targets` runs bench binaries too: measure only under `cargo bench`.
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    // Rows without a prefix measure the CPU (the default device is the GPU when present).
    set_default_stream(Stream::default_for(Device::cpu()));
    let one = Array::from(1.0f32);
    bench("dispatch_10k", 20, || {
        let mut x = Array::from(0.0f32);
        for _ in 0..10_000 {
            x = &x + &one;
        }
        x.eval()
    })?;
    // The same chain through the named function instead of the operator.
    bench("dispatch_10k_named", 20, || {
        let mut x = Array::from(0.0f32);
        for _ in 0..10_000 {
            x = x.add(&one);
        }
        x.eval()
    })?;

    let a = evaluated(ops::arange(0.0, 1e6, 1.0))?;
    let b = evaluated(ops::ones([1_000_000], DType::F32))?;
    bench("add_1m", 50, || (&a + &b).eval())?;
    bench("sum_1m", 50, || a.sum(..).eval())?;

    let m = evaluated(a.reshape([1000, 1000]))?;
    let n = evaluated(b.reshape([1000, 1000]))?;
    bench("add_strided_1m", 50, || (&m.t() + &n).eval())?;
    bench("sum_axis_1k", 50, || m.sum(1).eval())?;

    for size in [256usize, 1024] {
        let x = evaluated(ops::ones([size, size], DType::F32))?;
        let y = evaluated(ops::ones([size, size], DType::F32))?;
        let reps = if size > 512 { 10 } else { 50 };
        bench(&format!("matmul_{size}"), reps, || x.matmul(&y).eval())?;
    }

    // The same workloads on the GPU stream (skipped without Metal).
    set_default_stream(Stream::default_for(Device::gpu(0)));
    if (&one + &one).eval().is_err() {
        return Ok(());
    }
    bench("gpu_dispatch_10k", 10, || {
        let mut x = Array::from(0.0f32);
        for _ in 0..10_000 {
            x = &x + &one;
        }
        x.eval()
    })?;
    bench("gpu_add_1m", 50, || (&a + &b).eval())?;
    bench("gpu_sum_1m", 50, || a.sum(..).eval())?;
    bench("gpu_add_strided_1m", 50, || (&m.t() + &n).eval())?;
    bench("gpu_sum_axis_1k", 50, || m.sum(1).eval())?;
    for size in [256usize, 1024] {
        let x = evaluated(ops::ones([size, size], DType::F32))?;
        let y = evaluated(ops::ones([size, size], DType::F32))?;
        bench(&format!("gpu_matmul_{size}"), 20, || x.matmul(&y).eval())?;
    }
    Ok(())
}
