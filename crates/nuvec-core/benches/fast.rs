//! Fused operations on the GPU, compared against MLX by `scripts/bench_fast.py`
//! (`just bench-fast`). Prints `name<TAB>dtype<TAB>us_single<TAB>us_grouped` per case: time
//! per operation in sustained `async_eval` bursts after a warm-up, with one or eight
//! operations per `async_eval`. The case list and method match the script's.

use std::time::Instant;

use nuvec_core::{
    Array, DType, Device, Result, Stream, async_eval, fast, logsumexp, random, set_default_stream,
    softmax,
};

/// (name, shape): the operation is chosen by the name's prefix.
const CASES: &[(&str, &[usize])] = &[
    ("rms_norm-decode", &[1, 4096]),
    ("rms_norm-prefill", &[2048, 4096]),
    ("layer_norm-prefill", &[2048, 4096]),
    ("rope-decode", &[1, 32, 1, 128]),
    ("rope-prefill", &[1, 32, 2048, 128]),
    ("softmax-attn", &[32, 512, 512]),
    ("logsumexp-vocab", &[64, 32000]),
    ("cross_entropy-vocab", &[64, 32000]),
    ("cross_entropy-mnist", &[256, 10]),
];

fn op(name: &str, x: &Array, w: &Array, targets: &Array) -> Result<Array> {
    match name.split('-').next().expect("prefix") {
        "rms_norm" => fast::rms_norm(x, Some(w), 1e-5),
        "layer_norm" => fast::layer_norm(x, Some(w), Some(w), 1e-5),
        "rope" => fast::rope(
            x,
            128,
            false,
            Some(10000.0),
            1.0,
            &Array::scalar(7i32),
            None,
        ),
        "softmax" => softmax(x, &[-1]),
        "logsumexp" => logsumexp(x, &[-1], false),
        _ => fast::cross_entropy(x, targets),
    }
}

fn main() -> Result<()> {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    for (dtype, label) in [(DType::F32, "float32"), (DType::F16, "float16")] {
        for &(name, shape) in CASES {
            let x = random::uniform(-1.0, 1.0, shape, DType::F32, None)?.astype(dtype)?;
            let w = random::uniform(-1.0, 1.0, &shape[shape.len() - 1..], DType::F32, None)?
                .astype(dtype)?;
            let rows = &shape[..shape.len() - 1];
            let classes = shape[shape.len() - 1];
            let n: usize = rows.iter().product();
            let t: Vec<i32> = (0..n).map(|i| (i * 7919 % classes) as i32).collect();
            let targets = Array::from_slice(&t, rows)?;
            nuvec_core::eval(&[x.clone(), w.clone(), targets.clone()])?;
            let elements: usize = shape.iter().product();
            let reps = (200_000_000 / elements.max(1)).clamp(20, 4000);
            let burst = |count: usize, per_eval: usize| -> Result<f64> {
                let t = Instant::now();
                let mut pending: Vec<Array> = Vec::new();
                for _ in 0..count {
                    pending.push(op(name, &x, &w, &targets)?);
                    if pending.len() == per_eval {
                        async_eval(&pending)?;
                        pending.clear();
                    }
                }
                nuvec_core::eval(&pending)?;
                nuvec_core::eval(&[op(name, &x, &w, &targets)?])?;
                Ok(t.elapsed().as_secs_f64() / count as f64)
            };
            let best = |per_eval: usize| -> Result<f64> {
                burst(reps, per_eval)?;
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
