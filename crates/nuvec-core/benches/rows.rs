//! Few-row products (B = 1..16 rows: dense, quantized, gathered), compared against 0.3.0's
//! routing and MLX by `scripts/bench_rows.py` (`just bench-rows`). Prints
//! `name<TAB>dtype<TAB>B<TAB>us` per case: time per product in sustained `async_eval`
//! bursts after a warm-up (best of three).

use std::time::Instant;

use nuvec_core::{Array, DType, Device, Result, Stream, async_eval, eval, ops, set_default_stream};

fn uniform(shape: &[usize], dtype: DType) -> Result<Array> {
    let n: usize = shape.iter().product();
    let t = ops::arange(0, n as i32, 1).astype(DType::F32);
    let x = (&t * 0.7381).sin() * 0.5;
    let x = x
        .reshape(shape.iter().map(|&d| d as i32).collect::<Vec<_>>())
        .astype(dtype);
    x.eval()?;
    Ok(x)
}

type Op = Box<dyn Fn() -> Result<Array>>;

/// The product of case `name` at `b` rows.
fn make(name: &str, dtype: DType, b: usize) -> Result<Op> {
    let dims = |s: &str| -> Vec<usize> {
        s.rsplit('-')
            .next()
            .unwrap()
            .split('x')
            .map(|v| v.parse().unwrap())
            .collect()
    };
    if name.starts_with("dense") {
        let (k, n) = (dims(name)[0], dims(name)[1]);
        let (x, w) = (uniform(&[b, k], dtype)?, uniform(&[n, k], dtype)?);
        return Ok(Box::new(move || Ok(x.matmul(&w.t()))));
    }
    if name.starts_with("affine") {
        let bits = if name.starts_with("affine8") { 8 } else { 4 };
        let (n, k) = (dims(name)[0], dims(name)[1]);
        let (wq, s, bi) = uniform(&[n, k], dtype)?.quantize(64, bits);
        eval(&[&wq, &s, &bi])?;
        let x = uniform(&[b, k], dtype)?;
        return Ok(Box::new(move || {
            Ok(x.quantized_matmul(&wq, &s, &bi, true, 64, bits))
        }));
    }
    if name.starts_with("gather_qmm") {
        let (n, k) = (dims(name)[0], dims(name)[1]);
        let (wq, s, bi) = uniform(&[128, n, k], dtype)?.quantize(64, 4);
        eval(&[&wq, &s, &bi])?;
        let sorted = name.contains("sorted");
        let mut ids: Vec<u32> = (0..b * 8)
            .map(|i| ((i * 37 + i / 8 * 11) % 128) as u32)
            .collect();
        if sorted {
            ids.sort_unstable();
            let x = uniform(&[b * 8, 1, k], dtype)?;
            let ids = Array::from(ids);
            return Ok(Box::new(move || {
                Ok(ops::gather_qmm(
                    &x,
                    &wq,
                    &s,
                    &bi,
                    None,
                    Some(&ids),
                    true,
                    64,
                    4,
                    true,
                ))
            }));
        }
        let x = uniform(&[b, 1, 1, k], dtype)?;
        let ids = Array::from_slice(&ids, [b as i32, 8]);
        return Ok(Box::new(move || {
            Ok(ops::gather_qmm(
                &x,
                &wq,
                &s,
                &bi,
                None,
                Some(&ids),
                true,
                64,
                4,
                false,
            ))
        }));
    }
    let (n, k) = (dims(name)[0], dims(name)[1]);
    let w = uniform(&[64, k, n], dtype)?;
    let x = uniform(&[b * 8, 1, k], dtype)?;
    let mut ids: Vec<u32> = (0..b * 8).map(|i| ((i * 13) % 64) as u32).collect();
    ids.sort_unstable();
    let ids = Array::from(ids);
    Ok(Box::new(move || {
        Ok(ops::gather_mm(&x, &w, None, Some(&ids), true))
    }))
}

/// The cases: `(name, dtype)`; each runs at B = 1..=16 rows.
const CASES: &[(&str, DType)] = &[
    ("dense-4096x4096", DType::F16),
    ("dense-4096x4096", DType::BF16),
    ("dense-4096x4096", DType::F32),
    ("dense-head-1024x151936", DType::BF16),
    ("affine4-4096x4096", DType::BF16),
    ("affine4-12288x4096", DType::BF16),
    ("affine8-4096x4096", DType::BF16),
    ("gather_qmm4-moe128-768x2048", DType::BF16),
    ("gather_qmm4-sorted-moe128-768x2048", DType::BF16),
    ("gather_mm-moe64-512x1024", DType::BF16),
];

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let filter = std::env::var("BENCH_FILTER").unwrap_or_default();
    for &(name, dtype) in CASES.iter().filter(|(n, _)| n.contains(filter.as_str())) {
        let label = match dtype {
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            _ => "float32",
        };
        for b in 1..=16usize {
            let op = make(name, dtype, b)?;
            let reps = 200;
            let burst = |count: usize| -> Result<f64> {
                let t = Instant::now();
                let mut pending = Vec::new();
                for _ in 0..count {
                    pending.push(op()?);
                    if pending.len() == 8 {
                        async_eval(&pending)?;
                        pending.clear();
                    }
                }
                eval(&pending)?;
                eval(&[op()?])?;
                Ok(t.elapsed().as_secs_f64() / count as f64)
            };
            burst(reps)?;
            let best = (0..3)
                .map(|_| burst(reps))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .fold(f64::INFINITY, f64::min);
            println!("{name}\t{label}\t{b}\t{:.2}", best * 1e6);
        }
    }
    Ok(())
}
