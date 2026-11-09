//! MLX's special products and transforms on the default device, compared against MLX by
//! `scripts/bench_special.py`. Prints `name<TAB>ms` per case: milliseconds per call, ten
//! independent calls per evaluation, the best of ten timed evaluations after a warm-up.

use std::time::Instant;

use nuvec_core::{Array, DType, QuantMode, Result, ops, random};

/// Calls per evaluation.
const REPS: usize = 10;

fn normal(shape: &[usize], dtype: DType) -> Array {
    random::normal(shape, DType::F32, 0.0, 1.0, None).astype(dtype)
}

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    let bf = DType::BF16;
    let h1 = normal(&[256, 4096], DType::F16);
    let h2 = normal(&[64, 16384], DType::F16);
    let h3 = normal(&[256, 28 * 128], DType::F16);
    let (e1, e2, e3) = (
        normal(&[64, 2048], bf),
        normal(&[2048, 2048], bf),
        normal(&[2048, 16], bf),
    );
    let (sa, sb) = (normal(&[256, 4096], bf), normal(&[4096, 256], bf));
    let segments = Array::from_slice(
        &(0..8u32)
            .flat_map(|i| [i * 512, (i + 1) * 512])
            .collect::<Vec<u32>>(),
        [8, 2],
    );
    let (ma, mb) = (normal(&[1024, 1024], bf), normal(&[1024, 1024], bf));
    let mask = random::bernoulli(0.5, [32, 32], None);
    let (qx, qx1, qw) = (
        normal(&[256, 4096], bf),
        normal(&[1, 4096], bf),
        normal(&[4096, 4096], bf),
    );
    let qq = ops::quantize(&qw, QuantMode::Nvfp4);
    nuvec_core::eval(&[
        h1.clone(),
        h2.clone(),
        h3.clone(),
        e1.clone(),
        e2.clone(),
        e3.clone(),
        sa.clone(),
        sb.clone(),
        segments.clone(),
        ma.clone(),
        mb.clone(),
        mask.clone(),
        qx.clone(),
        qx1.clone(),
        qq.weight.clone(),
        qq.scales.clone(),
    ])?;
    type Case<'a> = (&'static str, Box<dyn Fn() -> Array + 'a>);
    let cases: Vec<Case> = vec![
        (
            "hadamard-4096",
            Box::new(|| ops::hadamard_transform(&h1, None)),
        ),
        (
            "hadamard-16384",
            Box::new(|| ops::hadamard_transform(&h2, None)),
        ),
        (
            "hadamard-28x128",
            Box::new(|| ops::hadamard_transform(&h3, None)),
        ),
        (
            "einsum-chain",
            Box::new(|| ops::einsum("ij,jk,kl->il", &[e1.clone(), e2.clone(), e3.clone()])),
        ),
        (
            "segmented_mm",
            Box::new(|| ops::segmented_mm(&sa, &sb, &segments)),
        ),
        (
            "block_masked_mm",
            Box::new(|| ops::block_masked_mm(&ma, &mb, 32, Some(&mask), None, None)),
        ),
        ("qqmm-256", Box::new(|| ops::qqmm(&qx, &qq))),
        ("qqmm-1", Box::new(|| ops::qqmm(&qx1, &qq))),
    ];
    let filter = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_default();
    for (name, f) in &cases {
        if !name.contains(filter.as_str()) {
            continue;
        }
        let batch = || (0..REPS).map(|_| f()).collect::<Vec<_>>();
        nuvec_core::eval(&batch())?;
        let mut best = f64::INFINITY;
        for _ in 0..10 {
            let t = Instant::now();
            nuvec_core::eval(&batch())?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("{name}\t{:.4}", best * 1e3 / REPS as f64);
    }
    Ok(())
}
