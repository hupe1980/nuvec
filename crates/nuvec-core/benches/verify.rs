//! Verify attention (speculative decoding): 1, 2, 4 and 8 causal query rows of one sequence
//! over a key cache, for `scripts/bench_verify.py` (`just bench-verify`). Prints
//! `d<TAB>gqa<TAB>lq<TAB>lk<TAB>us` per cell: microseconds per call, the median over
//! repetitions of 32 independent calls evaluated together (so launch overhead is amortised
//! as in a decode step).

use std::time::Instant;

use nuvec_core::fast::AttentionOptions;
use nuvec_core::{Array, DType, Device, Result, Stream, ops, set_default_stream};

/// Calls per evaluation.
const CALLS: usize = 32;

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    // (head size, query heads, key heads): Qwen3-8B and Qwen3.6-35B-A3B.
    for &(d, hq, hkv) in &[(128, 32, 8), (256, 16, 2)] {
        for &lk in &[4096, 16384, 32768] {
            for &lq in &[1, 2, 4, 8] {
                let name = format!("{d}-{lq}-{lk}");
                if !name.contains(filter.as_str()) {
                    continue;
                }
                let shape = |h: i32, l: i32| [1, h, l, d];
                let k = ops::zeros(shape(hkv, lk), DType::BF16) + 0.02;
                let v = ops::zeros(shape(hkv, lk), DType::BF16) + 0.03;
                let qs: Vec<Array> = (0..CALLS)
                    .map(|i| ops::zeros(shape(hq, lq), DType::BF16) + (0.01 + i as f64 * 1e-3))
                    .collect();
                let mut all = vec![k.clone(), v.clone()];
                all.extend(qs.iter().cloned());
                nuvec_core::eval(&all.iter().collect::<Vec<_>>())?;
                let opts = AttentionOptions::causal();
                let run = || -> Result<f64> {
                    let t = Instant::now();
                    let outs: Vec<Array> = qs
                        .iter()
                        .map(|q| {
                            nuvec_core::fast::scaled_dot_product_attention(q, &k, &v, 0.088, &opts)
                        })
                        .collect();
                    nuvec_core::eval(&outs.iter().collect::<Vec<_>>())?;
                    Ok(t.elapsed().as_secs_f64() * 1e6 / CALLS as f64)
                };
                run()?;
                let mut times = (0..9).map(|_| run()).collect::<Result<Vec<_>>>()?;
                times.sort_by(f64::total_cmp);
                println!(
                    "{d}\t{}\t{lq}\t{lk}\t{:.1}",
                    hq / hkv,
                    times[times.len() / 2]
                );
            }
        }
    }
    Ok(())
}
