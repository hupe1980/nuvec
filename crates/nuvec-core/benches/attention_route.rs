//! Attention under a score-memory cap at mrun's chunk shapes, for the capped routing table
//! (`scripts/bench_attention_route.py`, `just bench-attention-route`). Prints
//! `variant<TAB>d<TAB>lq<TAB>lk<TAB>dtype<TAB>ms` per cell: the median of a few causal
//! attention calls over a key cache, after a warm-up.
//!
//! Variants: `uncapped` (today's routing), and under the cap `tiled` or `blocks` (the
//! composition in query blocks), forced with `NUVEC_ATTENTION_CAPPED_KERNEL`, the
//! benchmarking override of the capped choice.

use std::time::Instant;

use nuvec_core::fast::{AttentionOptions, plan_attention};
use nuvec_core::{DType, Device, Result, Stream, ops, set_default_stream};

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let variant = std::env::var("BENCH_VARIANT").unwrap_or_else(|_| "uncapped".into());
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    // The score cap (`BENCH_CAP` bytes; 1 forces the tiled kernel wherever it applies).
    let cap = std::env::var("BENCH_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512usize << 20);
    let dtypes = [
        (DType::BF16, "bfloat16"),
        (DType::F16, "float16"),
        (DType::F32, "float32"),
    ];
    // (head size, query heads, key heads): Qwen3.6 and Qwen3-8B.
    for &(d, hq, hkv) in &[(256, 16, 2), (128, 32, 8)] {
        for &lq in &[9, 64, 342, 1024, 2048] {
            for &lk in &[2048, 4096, 8192, 32768, 131072, 262144] {
                for &(dtype, label) in &dtypes {
                    let name = format!("{d}-{lq}-{lk}-{label}");
                    if !name.contains(filter.as_str()) || lq > lk {
                        continue;
                    }
                    // float32 only at the shorter caches (the table's f32 column).
                    if dtype == DType::F32 && lk > 32768 {
                        continue;
                    }
                    let opts = match variant.as_str() {
                        "uncapped" => AttentionOptions::causal(),
                        _ => AttentionOptions::causal().max_score_bytes(cap),
                    };
                    let shape = |h: i32, l: i32| [1, h, l, d];
                    let plan = plan_attention(
                        &shape(hq, lq),
                        &shape(hkv, lk),
                        &shape(hkv, lk),
                        dtype,
                        &opts,
                    )?;
                    // The uncapped composition of the largest cells needs tens of GiB.
                    let scores = hq as usize * lq as usize * lk as usize * dtype.size();
                    if variant == "uncapped" && scores > 8 << 30 {
                        continue;
                    }
                    let q = ops::zeros(shape(hq, lq), dtype) + 0.01;
                    let k = ops::zeros(shape(hkv, lk), dtype) + 0.02;
                    nuvec_core::eval(&[&q, &k])?;
                    let run = || -> Result<f64> {
                        let t = Instant::now();
                        nuvec_core::fast::scaled_dot_product_attention(&q, &k, &k, 0.0625, &opts)
                            .eval()?;
                        Ok(t.elapsed().as_secs_f64() * 1e3)
                    };
                    run()?;
                    let reps = if lq as usize * lk as usize > 1 << 26 {
                        3
                    } else {
                        7
                    };
                    let mut times = (0..reps).map(|_| run()).collect::<Result<Vec<_>>>()?;
                    times.sort_by(f64::total_cmp);
                    println!(
                        "{variant}\t{d}\t{lq}\t{lk}\t{label}\t{:.3}\t{:?}",
                        times[times.len() / 2],
                        plan.kernel
                    );
                }
            }
        }
    }
    Ok(())
}
