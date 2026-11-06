//! Attention forward and backward on the GPU (`cargo bench -p nuvec-core --bench
//! attention_vjp`): `value_and_grad` of `sum(attention(q, k, v) · w)` for q, k and v at
//! training shapes. Pairs with `scripts/bench_attention_vjp.py` (MLX). Prints
//! `name<TAB>dtype<TAB>us<TAB>peak_bytes` (with `NUVEC_PROFILE=1`, each kernel's GPU time
//! to stderr).

use std::time::Instant;

use nuvec_core::fast::{self, AttentionOptions};
use nuvec_core::{
    Array, DType, Device, Result, Stream, eval, random, set_default_stream, value_and_grad,
};

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    for (dtype, label) in [(DType::F16, "float16"), (DType::BF16, "bfloat16")] {
        for l in [512, 2048, 4096] {
            for d in [64, 128] {
                for causal in [true, false] {
                    let name = format!(
                        "b4-h16-kv8-l{l}-d{d}-{}",
                        if causal { "causal" } else { "full" }
                    );
                    if !name.contains(filter.as_str()) {
                        continue;
                    }
                    let u =
                        |s: &[i32]| random::uniform(-1.0, 1.0, s, DType::F32, None).astype(dtype);
                    let (q, k, v) = (u(&[4, 16, l, d]), u(&[4, 8, l, d]), u(&[4, 8, l, d]));
                    let w = u(&[4, 16, l, d]);
                    eval(&[&q, &k, &v, &w])?;
                    let opts = if causal {
                        AttentionOptions::causal()
                    } else {
                        AttentionOptions::default()
                    };
                    let scale = (d as f32).powf(-0.5);
                    let f = |(q, k, v): &(Array, Array, Array)| {
                        (fast::scaled_dot_product_attention(q, k, v, scale, &opts)
                            .astype(DType::F32)
                            * &w)
                            .sum(..)
                    };
                    let step = || -> Result<f64> {
                        let t = Instant::now();
                        let (loss, (gq, gk, gv)) =
                            value_and_grad(f)(&(q.clone(), k.clone(), v.clone()))?;
                        eval(&[&loss, &gq, &gk, &gv])?;
                        Ok(t.elapsed().as_secs_f64() * 1e6)
                    };
                    step()?;
                    if let Some(d) = nuvec_gpu::Device::shared() {
                        d.reset_peak();
                    }
                    nuvec_gpu::profile::reset();
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        best = best.min(step()?);
                    }
                    // `NUVEC_PROFILE=1`: GPU time per kernel over the five steps, to stderr.
                    for (kernel, count, secs) in nuvec_gpu::profile::report() {
                        eprintln!("{name}\t{label}\t{kernel}\t{count}\t{:.0}", secs * 1e6);
                    }
                    println!(
                        "{name}\t{label}\t{best:.1}\t{}",
                        nuvec_gpu::Device::shared().map_or(0, |d| d.memory().peak_with_mapped)
                    );
                }
            }
        }
    }
    Ok(())
}
