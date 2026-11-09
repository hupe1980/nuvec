//! Fused operations and element-wise functions on the GPU, compared against MLX by `scripts/bench_fast.py`
//! (`just bench-fast`). Prints `name<TAB>dtype<TAB>us_single<TAB>us_grouped` per case: time
//! per operation in sustained `async_eval` bursts after a warm-up, with one or eight
//! operations per `async_eval`. The case list and method match the script's.

use std::time::Instant;

use nuvec_core::{
    Array, DType, Device, Result, Stream, async_eval, fast, random, set_default_stream,
};

/// (name, shape): the operation is chosen by the name's prefix.
const CASES: &[(&str, &[usize])] = &[
    ("rms_norm-decode", &[1, 4096]),
    ("rms_norm-e4b", &[1, 2560]),
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
    // Qwen3.6's attention: the last 2,048-token chunk of a 34K-token prompt.
    ("sdpa-chunk-q36-32k", &[1, 16, 2, 2048, 32768, 256]),
    // Speculative verification: 2–8 causal query rows over the cache (Qwen3-8B and
    // Qwen3.6 heads), including rows on both sides of a key-block bucket edge.
    ("sdpa-verify2-1k", &[1, 32, 8, 2, 1025, 128]),
    ("sdpa-verify4-4k", &[1, 32, 8, 4, 4096, 128]),
    ("sdpa-verify8-4k", &[1, 32, 8, 8, 4096, 128]),
    ("sdpa-verify4-q36", &[1, 16, 2, 4, 8194, 256]),
    ("sdpa-verify8-q36", &[1, 16, 2, 8, 32768, 256]),
    // Sliding windows (`-w<keys>`): gpt-oss (D 64, W 128) and Gemma 4 sliding layers
    // (D 256, W 1024).
    ("sdpa-wdecode-w128", &[1, 64, 8, 1, 4096, 64]),
    ("sdpa-wprefill64-w128", &[1, 64, 8, 2048, 2175, 64]),
    ("sdpa-wdecode256-w1024", &[1, 16, 8, 1, 8192, 256]),
    ("sdpa-wprefill256-w1024", &[1, 16, 8, 2048, 3071, 256]),
    // Decode over exactly the window's keys (what mlx-lm's rotating cache passes).
    ("sdpa-wsdecode-w128", &[1, 64, 8, 1, 128, 64]),
    ("sdpa-wsdecode256-w1024", &[1, 16, 8, 1, 1024, 256]),
    // The same without a window (plain causal decode at head size 256).
    ("sdpa-decode256-1k", &[1, 16, 8, 1, 1024, 256]),
    // Gemma 4 global layers (D 512): decode, and MLX's blocked causal prefill.
    ("sdpa-d512-decode4k", &[1, 32, 4, 1, 4096, 512]),
    ("sdpa-d512-decode32k", &[1, 32, 4, 1, 32768, 512]),
    ("sdpa-d512-prefill", &[1, 16, 2, 2048, 2048, 512]),
    // Gemma 4 E4B's decode: sliding layers (D 256, 512 keys) and global layers (D 512 with
    // 4 query heads per key head: the composition in both engines).
    ("sdpa-e4b-d256-512", &[1, 8, 2, 1, 512, 256]),
    ("sdpa-e4b-d512-1k", &[1, 8, 2, 1, 1024, 512]),
    ("sdpa-e4b-d512-4k", &[1, 8, 2, 1, 4096, 512]),
    // MLX's element-wise functions over 2^24 contiguous elements (`ew-<name>`).
    ("ew-erf", &[1 << 24]),
    ("ew-expm1", &[1 << 24]),
    ("ew-log1p", &[1 << 24]),
    ("ew-log2", &[1 << 24]),
    ("ew-tan", &[1 << 24]),
    ("ew-sinh", &[1 << 24]),
    ("ew-arccos", &[1 << 24]),
    ("ew-arcsinh", &[1 << 24]),
    ("ew-arctanh", &[1 << 24]),
    ("ew-rsqrt", &[1 << 24]),
    ("ew-floor", &[1 << 24]),
    ("ew-round", &[1 << 24]),
    ("ew-arctan2", &[1 << 24]),
    ("ew-remainder", &[1 << 24]),
];

fn op(name: &str, x: &Array, w: &Array, targets: &Array) -> Array {
    if let Some(f) = name.strip_prefix("ew-") {
        return match f {
            "erf" => x.erf(),
            "expm1" => x.expm1(),
            "log1p" => x.log1p(),
            "log2" => x.log2(),
            "tan" => x.tan(),
            "sinh" => x.sinh(),
            "arccos" => x.arccos(),
            "arcsinh" => x.arcsinh(),
            "arctanh" => x.arctanh(),
            "rsqrt" => x.rsqrt(),
            "floor" => x.floor(),
            "round" => x.round(0),
            "arctan2" => x.arctan2(w),
            _ => x.remainder(w),
        };
    }
    match name.split('-').next().expect("prefix") {
        // `x` is q, `w` the keys and values (one array), as set up in `main`.
        "sdpa" => {
            let mask = if x.shape()[2] > 1 {
                fast::AttentionMask::Causal
            } else {
                fast::AttentionMask::None
            };
            let mut options = fast::AttentionOptions::from(mask);
            if let Some((_, w)) = name.rsplit_once("-w") {
                options = fast::AttentionOptions::causal().window(w.parse().expect("window"));
            }
            fast::scaled_dot_product_attention(x, w, w, 0.088, &options)
        }
        "rms_norm" => fast::rms_norm(x, w, 1e-5),
        "layer_norm" => fast::layer_norm(x, w, w, 1e-5),
        "rope" => fast::rope(x, 128, false, Some(10000.0), 1.0, &Array::from(7i32), None),
        "softmax" => x.softmax(-1),
        "logsumexp" => x.logsumexp(-1),
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
        // `BENCH_FILTER`: names containing it, or any of its `|`-separated alternatives.
        let chosen = |n: &str| filter.split('|').any(|f| n.contains(f));
        for &(name, shape) in CASES.iter().filter(|(n, _)| chosen(n)) {
            let uniform =
                |s: &[usize]| random::uniform(-1.0, 1.0, s, DType::F32, None).astype(dtype);
            let (x, w, shape) = if name.starts_with("sdpa") {
                let (b, hq, hkv, lq, lk, d) =
                    (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
                (
                    uniform(&[b, hq, lq, d]),
                    uniform(&[b, hkv, lk, d]),
                    &[b * hq * lq, d][..],
                )
            } else {
                (uniform(shape), uniform(&shape[shape.len() - 1..]), shape)
            };
            let rows = &shape[..shape.len() - 1];
            let classes = shape[shape.len() - 1];
            let n: usize = rows.iter().product();
            let t: Vec<i32> = (0..n).map(|i| (i * 7919 % classes) as i32).collect();
            let targets = Array::from_slice(&t, rows);
            nuvec_core::eval(&[x.clone(), w.clone(), targets.clone()])?;
            let elements: usize = shape.iter().product();
            // `BENCH_REPS` caps the operations per burst (slow cases).
            let cap = std::env::var("BENCH_REPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4000);
            let reps = (200_000_000 / elements.max(1)).clamp(20, 4000).min(cap);
            let burst = |count: usize, per_eval: usize| -> Result<f64> {
                let t = Instant::now();
                let mut pending: Vec<Array> = Vec::new();
                for _ in 0..count {
                    pending.push(op(name, &x, &w, &targets));
                    if pending.len() == per_eval {
                        async_eval(&pending)?;
                        pending.clear();
                    }
                }
                nuvec_core::eval(&pending)?;
                nuvec_core::eval(&[op(name, &x, &w, &targets)])?;
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
