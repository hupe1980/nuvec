//! `all_sum` over the backend `init` selects, one rank per process
//! (`scripts/bench_distributed.py` starts the ranks over loopback, or with `mlx.launch` on
//! several Macs, and runs MLX the same way). Rank 0 prints `bytes<TAB>us` per size: the
//! best of three bursts, each a barrier then `reps` sums of a float32 array evaluated one
//! at a time, on the CPU stream. With `--decode`, decode's shape instead: 56 GPU ops each
//! followed by an all_sum of `[1, 1024]` bfloat16 per evaluation, on the GPU stream; rank 0
//! prints `decode<TAB>us` per all_sum (the best of five bursts of 20 evaluations).
//! `NUVEC_BENCH_ROWS=2048` gives the arrays 2,048 rows (prefill's 4 MB per all_sum). With
//! `--hop`, a pipeline hop: 32 MB float32 sent from rank 0 to rank 1 (on the CPU stream);
//! rank 0 prints `hop<TAB>us` per send, the best of five bursts of five ending in a barrier.
//! With `NUVEC_DISTRIBUTED_KEY` set, every connection is encrypted.

use std::time::Instant;

use nuvec_core::distributed::{all_sum, recv, send};
use nuvec_core::{DType, Device, Result, Stream, ops, set_default_stream};

const SIZES: &[usize] = &[4 << 10, 1 << 20, 4 << 20, 16 << 20, 256 << 20];

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") || std::env::var("MLX_RANK").is_err() {
        return Ok(());
    }
    if std::env::args().any(|a| a == "--decode") {
        return decode();
    }
    if std::env::args().any(|a| a == "--hop") {
        return hop();
    }
    set_default_stream(Stream::default_for(Device::cpu()));
    let group = nuvec_distributed::init(true)?;
    for &bytes in SIZES {
        let x = ops::ones([(bytes / 4) as i32], DType::F32);
        x.eval()?;
        let reps = (64 << 20) / bytes;
        let reps = reps.clamp(3, 200);
        let mut best = f64::INFINITY;
        for burst in 0..4 {
            all_sum(&ops::ones([1], DType::F32), None).eval()?; // barrier
            let t = Instant::now();
            for _ in 0..reps {
                all_sum(&x, None).eval()?;
            }
            let us = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
            if burst > 0 {
                best = best.min(us); // the first burst warms up
            }
        }
        if group.rank() == 0 {
            println!("{bytes}\t{best:.1}");
        }
    }
    Ok(())
}

fn decode() -> Result<()> {
    let group = nuvec_distributed::init(true)?;
    let rows: i32 = std::env::var("NUVEC_BENCH_ROWS").map_or(1, |r| r.parse().expect("a number"));
    let x = ops::ones([rows, 1024], DType::BF16);
    x.eval()?;
    let mut best = f64::INFINITY;
    for burst in 0..6 {
        all_sum(&ops::ones([1], DType::F32), None).eval()?; // barrier
        let t = Instant::now();
        for _ in 0..20 {
            let mut y = x.clone();
            for _ in 0..56 {
                y = all_sum(&y.multiply(0.5), None);
            }
            y.eval()?;
        }
        let us = t.elapsed().as_secs_f64() / (20.0 * 56.0) * 1e6;
        if burst > 0 {
            best = best.min(us);
        }
    }
    if group.rank() == 0 {
        println!("decode\t{best:.1}");
    }
    Ok(())
}

fn hop() -> Result<()> {
    set_default_stream(Stream::default_for(Device::cpu()));
    let group = nuvec_distributed::init(true)?;
    let n = (32 << 20) / 4;
    let x = ops::ones([n], DType::F32);
    x.eval()?;
    let mut best = f64::INFINITY;
    for burst in 0..6 {
        all_sum(&ops::ones([1], DType::F32), None).eval()?; // barrier
        let t = Instant::now();
        for _ in 0..5 {
            match group.rank() {
                0 => send(&x, 1, None).eval()?,
                1 => recv([n], DType::F32, 0, None).eval()?,
                _ => {}
            }
        }
        all_sum(&ops::ones([1], DType::F32), None).eval()?; // the last receive is done
        let us = t.elapsed().as_secs_f64() / 5.0 * 1e6;
        if burst > 0 {
            best = best.min(us);
        }
    }
    if group.rank() == 0 {
        println!("hop\t{best:.1}");
    }
    Ok(())
}
