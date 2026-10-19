//! Quantized matrix products on the GPU, compared against MLX by `scripts/bench_quant.py`
//! (`just bench-quant`). Prints `name<TAB>dtype<TAB>us_single<TAB>us_grouped` per case: the
//! time per product in sustained `async_eval` bursts after a warm-up, submitting one product
//! per `async_eval` (one command buffer each) or eight. The shape list and method match the
//! script's.

use std::time::Instant;

use nuvec_core::{
    Array, DType, Device, Result, Stream, async_eval, gather_qmm, quantize, quantized_matmul,
    random, set_default_stream,
};

/// (name, M, K, N, transpose, bits, group size): `x [M, K]` times weights of `N` outputs.
const SHAPES: &[(&str, usize, usize, usize, bool, u32, usize)] = &[
    ("qmv4096", 1, 4096, 4096, true, 4, 64),
    ("qmvup", 1, 4096, 11008, true, 4, 64),
    ("qmvdown", 1, 11008, 4096, true, 4, 64),
    ("qmv4096-b8", 1, 4096, 4096, true, 8, 64),
    ("qmv4096-b3", 1, 4096, 4096, true, 3, 64),
    ("qmv4096-g32", 1, 4096, 4096, true, 4, 32),
    ("qmv-m4", 4, 4096, 4096, true, 4, 64),
    ("qmm-m32", 32, 4096, 4096, true, 4, 64),
    ("qmm-m512", 512, 4096, 4096, true, 4, 64),
    ("qvm4096", 1, 4096, 4096, false, 4, 64),
    ("qmm-n128", 128, 4096, 4096, false, 4, 64),
    // MoE prefill (Qwen3.6-35B-A3B, 961 tokens): dense projections, then the experts'
    // products over rows sorted by expert (`gqmm`: M rows, 256 experts, 8 per token).
    ("qmm-961x8192", 961, 2048, 8192, true, 4, 64),
    ("qmm-961x2048", 961, 2048, 2048, true, 4, 64),
    ("gqmm-up", 7688, 2048, 512, true, 4, 64),
    ("gqmm-down", 7688, 512, 2048, true, 4, 64),
];

/// Experts of the `gqmm` cases.
const EXPERTS: usize = 256;

fn main() -> Result<()> {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    // Cases whose name contains this (all when unset).
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    for (dtype, label) in [(DType::F16, "float16"), (DType::BF16, "bfloat16")] {
        for &(name, m, k, n, transpose, bits, gs) in SHAPES {
            if !name.contains(filter.as_str()) {
                continue;
            }
            let gathered = name.starts_with("gqmm");
            let w_shape = match (gathered, transpose) {
                (true, _) => vec![EXPERTS, n, k],
                (false, true) => vec![n, k],
                (false, false) => vec![k, n],
            };
            let w = random::uniform(-1.0, 1.0, &w_shape, DType::F32, None)?.astype(dtype)?;
            let (wq, s, b) = quantize(&w, gs, bits)?;
            let x_shape = if gathered { vec![m, 1, k] } else { vec![m, k] };
            let x = random::uniform(-1.0, 1.0, &x_shape, DType::F32, None)?.astype(dtype)?;
            let ids: Vec<u32> = (0..m).map(|i| (i * EXPERTS / m) as u32).collect();
            let experts = Array::from_slice(&ids, &[m])?;
            nuvec_core::eval(&[wq.clone(), s.clone(), b.clone(), x.clone()])?;
            let product = || -> Result<Array> {
                match gathered {
                    true => gather_qmm(&x, &wq, &s, &b, None, Some(&experts), true, gs, bits, true),
                    false => quantized_matmul(&x, &wq, &s, &b, transpose, gs, bits),
                }
            };
            // About 0.1 s of work per measurement (at least 20 products).
            let bytes = k * n * bits as usize / 8;
            let reps = (40_000_000_000 / 10 / (bytes + 2 * m * k * n).max(1)).clamp(20, 4000);
            let burst = |count: usize, per_eval: usize| -> Result<f64> {
                let t = Instant::now();
                let mut pending: Vec<Array> = Vec::new();
                for _ in 0..count {
                    pending.push(product()?);
                    if pending.len() == per_eval {
                        async_eval(&pending)?;
                        pending.clear();
                    }
                }
                nuvec_core::eval(&pending)?;
                nuvec_core::eval(&[product()?])?;
                Ok(t.elapsed().as_secs_f64() / count as f64)
            };
            let best = |per_eval: usize| -> Result<f64> {
                burst(reps, per_eval)?; // warm-up: GPU clocks ramp up
                let times = (0..3)
                    .map(|_| burst(reps, per_eval))
                    .collect::<Result<Vec<_>>>()?;
                Ok(times.into_iter().fold(f64::INFINITY, f64::min) * 1e6)
            };
            println!("{name}\t{label}\t{:.2}\t{:.2}", best(1)?, best(8)?);
        }
    }
    Ok(())
}
