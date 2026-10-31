//! `all_sum` over the ring, one rank per process (`scripts/bench_distributed.py` starts the
//! ranks with `MLX_HOSTFILE` and `MLX_RANK`, and runs MLX's ring the same way). Rank 0
//! prints `bytes<TAB>us` per size: the best of three bursts, each a barrier then `reps`
//! sums of a float32 array evaluated one at a time, on the CPU stream.

use std::time::Instant;

use nuvec_core::distributed::all_sum;
use nuvec_core::{DType, Device, Result, Stream, ops, set_default_stream};

const SIZES: &[usize] = &[4 << 10, 1 << 20, 16 << 20, 256 << 20];

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") || std::env::var("MLX_RANK").is_err() {
        return Ok(());
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
