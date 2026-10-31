//! Quantized matrix products on the GPU, compared against MLX by `scripts/bench_quant.py`
//! (`just bench-quant`). Prints `name<TAB>dtype<TAB>us_single<TAB>us_grouped` per case: the
//! time per product in sustained `async_eval` bursts after a warm-up, submitting one product
//! per `async_eval` (one command buffer each) or eight. The shape list and method match the
//! script's.

use std::time::Instant;

use nuvec_core::{
    Array, DType, Device, GatherQmm, Quant, QuantMode, Result, Stream, async_eval, random,
    set_default_stream,
};

/// (name, M, K, N, transpose, bits, group size): `x [M, K]` times weights of `N` outputs.
const SHAPES: &[(&str, usize, usize, usize, bool, i32, i32)] = &[
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
    // Gemma 4 E4B's decode (one row): MLP, attention, the per-layer input projection
    // (K = 256: no `qmv_fast`) and the 262,144-row output head.
    ("e4b-up", 1, 2560, 10240, true, 4, 64),
    ("e4b-down", 1, 10240, 2560, true, 4, 64),
    ("e4b-q", 1, 2560, 2048, true, 4, 64),
    ("e4b-kv", 1, 2560, 512, true, 4, 64),
    ("e4b-o", 1, 2048, 2560, true, 4, 64),
    ("e4b-pli", 1, 256, 2560, true, 4, 64),
    ("e4b-head", 1, 2560, 262144, true, 4, 64),
    // Gemma 4 26B-A4B's decode at 8 requests (hidden 2,816: no `qmv_fast`): the dense
    // projections, then the experts' (`g26g`: 64 rows over 128 experts).
    ("g26-q", 8, 2816, 4096, true, 4, 64),
    ("g26-kv", 8, 2816, 2048, true, 4, 64),
    ("g26-o", 8, 4096, 2816, true, 4, 64),
    ("g26-mlp", 8, 2816, 2112, true, 4, 64),
    ("g26-down", 8, 2112, 2816, true, 4, 64),
    ("g26g-up", 64, 2816, 704, true, 4, 64),
    ("g26g-down", 64, 704, 2816, true, 4, 64),
    // Dense prefill (Qwen3-8B, 2,048 tokens).
    ("qmm-2048x4096", 2048, 4096, 4096, true, 4, 64),
    ("qmm-2048x12288", 2048, 4096, 12288, true, 4, 64),
    ("qmm-2048-down", 2048, 12288, 4096, true, 4, 64),
    // MoE prefill at 2,048-token chunks (Qwen3.6-35B-A3B): attention and linear-attention
    // projections, the shared expert, then the routed experts (16,384 rows, 8 per token).
    ("q36-8192", 2048, 2048, 8192, true, 4, 64),
    ("q36-4096", 2048, 2048, 4096, true, 4, 64),
    ("q36-out", 2048, 4096, 2048, true, 4, 64),
    ("q36-512", 2048, 2048, 512, true, 4, 64),
    ("q36-shared-down", 2048, 512, 2048, true, 4, 64),
    ("gqmm-2048-up", 16384, 2048, 512, true, 4, 64),
    ("gqmm-2048-down", 16384, 512, 2048, true, 4, 64),
];

/// Floating-point modes (`mode-name` cases): (name, M, K, N, transpose), each run in mxfp4,
/// mxfp8 and nvfp4, including gpt-oss's dense shapes (`oss-*`, K = N = 2880).
const FP_SHAPES: &[(&str, usize, usize, usize, bool)] = &[
    ("qmv4096", 1, 4096, 4096, true),
    ("qmvup", 1, 4096, 11008, true),
    ("qmvdown", 1, 11008, 4096, true),
    ("qmv-m4", 4, 4096, 4096, true),
    ("qmv-m8", 8, 4096, 4096, true),
    ("qmm-m32", 32, 4096, 4096, true),
    ("qmm-m512", 512, 4096, 4096, true),
    ("qvm4096", 1, 4096, 4096, false),
    ("qmv-k128", 1, 128, 4096, true),
    ("oss-qmv2880", 1, 2880, 2880, true),
    ("oss-qmv2880-m4", 4, 2880, 2880, true),
    ("oss-qmm2880", 512, 2880, 2880, true),
];

/// Floating-point gathers (`mode-name` cases, each mode; nvfp4 also with per-expert global
/// scales, `nvfp4-name-gs`): (name, rows, K, N, experts), one row per (token, expert), rows
/// sorted by expert. Qwen3-30B-A3B's experts (128, K = 2048, N = 768) and gpt-oss-20b's
/// (`oss-*`, 32, K = N = 2880), at decode (one token's top-k) and 512 rows (grouped).
const FP_GATHERS: &[(&str, usize, usize, usize, usize)] = &[
    ("gather-decode", 8, 2048, 768, 128),
    ("gather-sorted", 512, 2048, 768, 128),
    ("oss-gather-decode", 4, 2880, 2880, 32),
    ("oss-gather-sorted", 512, 2880, 2880, 32),
];

/// The floating-point modes of [`FP_SHAPES`].
const FP_MODES: [(&str, QuantMode); 3] = [
    ("mxfp4", QuantMode::Mxfp4),
    ("mxfp8", QuantMode::Mxfp8),
    ("nvfp4", QuantMode::Nvfp4),
];

/// Experts of the `gqmm` cases.
const EXPERTS: usize = 256;

/// A case: name, M, K, N, transpose, quantization, experts (0: a dense product) and
/// whether nvfp4 experts carry a global scale.
type Case = (String, usize, usize, usize, bool, Quant, usize, bool);

fn main() -> Result<()> {
    // `cargo test --all-targets` runs bench binaries too: measure only under `cargo bench`.
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    // Cases whose name contains this (all when unset).
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    // Affine cases keep their names; floating-point cases are `mode-name`.
    let mut cases: Vec<Case> = SHAPES
        .iter()
        .map(|&(name, m, k, n, t, bits, gs)| {
            let experts = if name.starts_with("gqmm") {
                EXPERTS
            } else if name.starts_with("g26g") {
                128
            } else {
                0
            };
            (
                name.to_owned(),
                m,
                k,
                n,
                t,
                Quant::affine(gs, bits),
                experts,
                false,
            )
        })
        .collect();
    for (label, mode) in FP_MODES {
        for &(name, m, k, n, t) in FP_SHAPES {
            cases.push((
                format!("{label}-{name}"),
                m,
                k,
                n,
                t,
                Quant::new(mode),
                0,
                false,
            ));
        }
        for &(name, m, k, n, e) in FP_GATHERS {
            let q = Quant::new(mode);
            cases.push((format!("{label}-{name}"), m, k, n, true, q, e, false));
            if mode == QuantMode::Nvfp4 {
                cases.push((format!("{label}-{name}-gs"), m, k, n, true, q, e, true));
            }
        }
    }
    for (dtype, label) in [(DType::F16, "float16"), (DType::BF16, "bfloat16")] {
        for (name, m, k, n, transpose, quant, experts, global) in &cases {
            let (name, m, k, n, transpose, quant) = (name.as_str(), *m, *k, *n, *transpose, *quant);
            let (experts, global) = (*experts, *global);
            let bits = quant.bits;
            if !name.contains(filter.as_str()) {
                continue;
            }
            let gathered = experts > 0;
            let w_shape = match (gathered, transpose) {
                (true, _) => vec![experts, n, k],
                (false, true) => vec![n, k],
                (false, false) => vec![k, n],
            };
            let w = random::uniform(-1.0, 1.0, &w_shape[..], DType::F32, None).astype(dtype);
            let wq = w.quantize(quant);
            let x_shape = if gathered { vec![m, 1, k] } else { vec![m, k] };
            let x = random::uniform(-1.0, 1.0, &x_shape[..], DType::F32, None).astype(dtype);
            let ids: Vec<u32> = (0..m).map(|i| (i * experts.max(1) / m) as u32).collect();
            let indices = Array::from_slice(&ids, [m]);
            let scale = random::uniform(0.5, 2.0, [experts.max(1)], DType::F32, None);
            nuvec_core::eval(&(&wq, &x, &scale))?;
            let product = || -> Array {
                let options = GatherQmm::default().sorted_indices(true);
                let options = if global {
                    options.global_scale(&scale)
                } else {
                    options
                };
                match gathered {
                    true => x.gather_qmm(&wq, None, Some(&indices), options),
                    false => x.quantized_matmul(&wq, transpose),
                }
            };
            // About 0.1 s of work per measurement (at least 20 products).
            let bytes = k * n * bits as usize / 8;
            let reps = (40_000_000_000 / 10 / (bytes + 2 * m * k * n).max(1)).clamp(20, 4000);
            let burst = |count: usize, per_eval: usize| -> Result<f64> {
                let t = Instant::now();
                let mut pending: Vec<Array> = Vec::new();
                for _ in 0..count {
                    pending.push(product());
                    if pending.len() == per_eval {
                        async_eval(&pending)?;
                        pending.clear();
                    }
                }
                nuvec_core::eval(&pending)?;
                nuvec_core::eval(&[product()])?;
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
