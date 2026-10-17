//! Workloads compared against MLX by `scripts/bench_vs_mlx.py` (`just bench-mlx`).
//!
//! Prints one `name<TAB>median_us` line per workload. Each workload is warmed up, then
//! timed over repeated runs; the median is reported.

use std::time::Instant;

use nuvec_core::{
    Array, DType, Device, Result, Stream, arange, matmul, ones, set_default_stream, sum,
};

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
    // Rows without a prefix measure the CPU (the default device is the GPU when present).
    set_default_stream(Stream::default_for(Device::cpu()));
    let one = Array::scalar(1.0f32);
    bench("dispatch_10k", 20, || {
        let mut x = Array::scalar(0.0f32);
        for _ in 0..10_000 {
            x = &x + &one;
        }
        x.eval()
    })?;

    let a = evaluated(arange(0.0, 1e6, 1.0, DType::F32)?)?;
    let b = evaluated(ones(&[1_000_000], DType::F32)?)?;
    bench("add_1m", 50, || (&a + &b).eval())?;
    bench("sum_1m", 50, || sum(&a, &[], false)?.eval())?;

    let m = evaluated(a.reshape(&[1000, 1000])?)?;
    let n = evaluated(b.reshape(&[1000, 1000])?)?;
    bench("add_strided_1m", 50, || (&m.t()? + &n).eval())?;
    bench("sum_axis_1k", 50, || sum(&m, &[1], false)?.eval())?;

    for size in [256usize, 1024] {
        let x = evaluated(ones(&[size, size], DType::F32)?)?;
        let y = evaluated(ones(&[size, size], DType::F32)?)?;
        let reps = if size > 512 { 10 } else { 50 };
        bench(&format!("matmul_{size}"), reps, || matmul(&x, &y)?.eval())?;
    }

    // The same workloads on the GPU stream (skipped without Metal).
    set_default_stream(Stream::default_for(Device::gpu(0)));
    if (&one + &one).eval().is_err() {
        return Ok(());
    }
    bench("gpu_dispatch_10k", 10, || {
        let mut x = Array::scalar(0.0f32);
        for _ in 0..10_000 {
            x = &x + &one;
        }
        x.eval()
    })?;
    bench("gpu_add_1m", 50, || (&a + &b).eval())?;
    bench("gpu_sum_1m", 50, || sum(&a, &[], false)?.eval())?;
    bench("gpu_add_strided_1m", 50, || (&m.t()? + &n).eval())?;
    bench("gpu_sum_axis_1k", 50, || sum(&m, &[1], false)?.eval())?;
    for size in [256usize, 1024] {
        let x = evaluated(ones(&[size, size], DType::F32)?)?;
        let y = evaluated(ones(&[size, size], DType::F32)?)?;
        bench(&format!("gpu_matmul_{size}"), 20, || matmul(&x, &y)?.eval())?;
    }
    Ok(())
}
