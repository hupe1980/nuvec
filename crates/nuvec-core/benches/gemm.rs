//! GEMM shape matrix on the GPU, compared against MLX by `scripts/bench_gemm.py`
//! (`just bench-gemm`). Prints `name<TAB>dtype<TAB>us_per_product` per case; the shape list
//! and method are the same in both programs: sustained throughput (products submitted
//! back to back with `async_eval` after a warm-up), which keeps the GPU at full clocks.

use std::time::Instant;

use nuvec_core::{
    Array, DType, Device, Result, Stream, async_eval, matmul, random, set_default_stream,
};

/// (name, batch, M, K, N, B transposed)
const SHAPES: &[(&str, usize, usize, usize, usize, bool)] = &[
    ("sq256", 1, 256, 256, 256, false),
    ("sq512", 1, 512, 512, 512, false),
    ("sq1024", 1, 1024, 1024, 1024, false),
    ("sq2048", 1, 2048, 2048, 2048, false),
    ("sq4096", 1, 4096, 4096, 4096, false),
    ("nt2048", 1, 2048, 2048, 2048, true),
    ("skinny16", 1, 16, 4096, 4096, true),
    ("tallk", 1, 256, 8192, 256, false),
    ("wide", 1, 4096, 512, 4096, false),
    ("batch8", 8, 512, 512, 512, false),
    ("mnist", 1, 256, 784, 32, true),
    ("vec4096", 1, 1, 4096, 4096, true),
    ("vecup", 1, 1, 4096, 11008, true),
    ("vecdown", 1, 1, 11008, 4096, true),
    ("matvec", 1, 4096, 4096, 1, false),
];

fn main() -> Result<()> {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    for (dtype, label) in [
        (DType::F32, "float32"),
        (DType::F16, "float16"),
        (DType::BF16, "bfloat16"),
    ] {
        for &(name, batch, m, k, n, bt) in SHAPES {
            let a_shape: Vec<usize> = if batch > 1 {
                vec![batch, m, k]
            } else {
                vec![m, k]
            };
            let b_shape: Vec<usize> = match (batch > 1, bt) {
                (true, _) => vec![batch, k, n],
                (false, true) => vec![n, k],
                (false, false) => vec![k, n],
            };
            let a = random::uniform(-1.0, 1.0, &a_shape, DType::F32, None)?.astype(dtype)?;
            let b = random::uniform(-1.0, 1.0, &b_shape, DType::F32, None)?.astype(dtype)?;
            nuvec_core::eval(&[a.clone(), b.clone()])?;
            let b = if bt && batch == 1 { b.t()? } else { b };
            let flops = 2 * batch * m * n * k;
            // About 0.1 s of work per measurement (at least 20 products).
            let reps = (2_000_000_000_000 / 10 / flops.max(1)).clamp(20, 2000);
            let burst = |count: usize| -> Result<f64> {
                let t = Instant::now();
                let mut last: Option<Array> = None;
                for _ in 0..count {
                    let y = matmul(&a, &b)?;
                    async_eval(std::slice::from_ref(&y))?;
                    last = Some(y);
                }
                if let Some(y) = last {
                    y.eval()?;
                }
                Ok(t.elapsed().as_secs_f64() / count as f64)
            };
            burst(reps)?; // warm-up: GPU clocks ramp up
            let best = (0..3).map(|_| burst(reps)).collect::<Result<Vec<_>>>()?;
            let best = best.into_iter().fold(f64::INFINITY, f64::min);
            println!("{name}\t{label}\t{:.2}", best * 1e6);
        }
    }
    Ok(())
}
