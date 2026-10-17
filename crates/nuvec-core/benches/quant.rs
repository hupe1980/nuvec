//! Quantized matrix products on the GPU, compared against MLX by `scripts/bench_quant.py`
//! (`just bench-quant`). Prints `name<TAB>dtype<TAB>us_single<TAB>us_grouped` per case: the
//! time per product in sustained `async_eval` bursts after a warm-up, submitting one product
//! per `async_eval` (one command buffer each) or eight. The shape list and method match the
//! script's.

use std::time::Instant;

use nuvec_core::{
    Array, DType, Device, Result, Stream, async_eval, quantize, quantized_matmul, random,
    set_default_stream,
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
];

fn main() -> Result<()> {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    for (dtype, label) in [(DType::F16, "float16"), (DType::BF16, "bfloat16")] {
        for &(name, m, k, n, transpose, bits, gs) in SHAPES {
            let w_shape = if transpose { [n, k] } else { [k, n] };
            let w = random::uniform(-1.0, 1.0, &w_shape, DType::F32, None)?.astype(dtype)?;
            let (wq, s, b) = quantize(&w, gs, bits)?;
            let x = random::uniform(-1.0, 1.0, &[m, k], DType::F32, None)?.astype(dtype)?;
            nuvec_core::eval(&[wq.clone(), s.clone(), b.clone(), x.clone()])?;
            // About 0.1 s of work per measurement (at least 20 products).
            let bytes = k * n * bits as usize / 8;
            let reps = (40_000_000_000 / 10 / (bytes + 2 * m * k * n).max(1)).clamp(20, 4000);
            let burst = |count: usize, per_eval: usize| -> Result<f64> {
                let t = Instant::now();
                let mut pending: Vec<Array> = Vec::new();
                for _ in 0..count {
                    pending.push(quantized_matmul(&x, &wq, &s, &b, transpose, gs, bits)?);
                    if pending.len() == per_eval {
                        async_eval(&pending)?;
                        pending.clear();
                    }
                }
                nuvec_core::eval(&pending)?;
                nuvec_core::eval(&[quantized_matmul(&x, &wq, &s, &b, transpose, gs, bits)?])?;
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
