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
    // Attention: (B, Hq, Hkv, Lq, Lk, D); decode with GQA, causal prefill.
    ("sdpa-decode512", &[1, 32, 8, 1, 512, 128]),
    ("sdpa-decode4096", &[1, 32, 8, 1, 4096, 128]),
    // Keys and values too large for the system cache (as in a model's decode, whose
    // layers' caches never all fit).
    ("sdpa-decode32k", &[1, 32, 8, 1, 32768, 128]),
    ("sdpa-prefill", &[1, 32, 8, 1024, 1024, 128]),
    ("sdpa-prefill64", &[1, 16, 16, 2048, 2048, 64]),
    // Language-model prefill (Qwen3-0.6B heads): a 2,048-token chunk, then the second chunk
    // of a 3,841-token prompt over the whole cache.
    ("sdpa-chunk2048", &[1, 16, 8, 2048, 2048, 128]),
    ("sdpa-chunk1793", &[1, 16, 8, 1793, 3841, 128]),
];

fn op(name: &str, x: &Array, w: &Array, targets: &Array) -> Result<Array> {
    match name.split('-').next().expect("prefix") {
        // `x` is q, `w` the keys and values (one array), as set up in `main`.
        "sdpa" => {
            let mask = if x.shape()[2] > 1 {
                fast::AttentionMask::Causal
            } else {
                fast::AttentionMask::None
            };
            fast::scaled_dot_product_attention(x, w, w, 0.088, mask, None)
        }
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
    // `cargo test --all-targets` runs bench binaries too: measure only under `cargo bench`.
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    // Cases whose name contains this (all when unset).
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    let dtypes = [
        (DType::F32, "float32"),
        (DType::F16, "float16"),
        (DType::BF16, "bfloat16"),
    ];
    for (dtype, label) in dtypes {
        for &(name, shape) in CASES.iter().filter(|(n, _)| n.contains(filter.as_str())) {
            let uniform =
                |s: &[usize]| random::uniform(-1.0, 1.0, s, DType::F32, None)?.astype(dtype);
            let (x, w, shape) = if name.starts_with("sdpa") {
                let (b, hq, hkv, lq, lk, d) =
                    (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
                (
                    uniform(&[b, hq, lq, d])?,
                    uniform(&[b, hkv, lk, d])?,
                    &[b * hq * lq, d][..],
                )
            } else {
                (uniform(shape)?, uniform(&shape[shape.len() - 1..])?, shape)
            };
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
