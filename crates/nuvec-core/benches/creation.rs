//! Creation ops on the GPU (`cargo bench -p nuvec-core --bench creation`): for zeros, ones
//! and full, the creating call alone and creation plus evaluation, from 4 KB to 256 MB;
//! and 1,000 small creations each added to a GPU array, evaluated once. Pairs with
//! `scripts/bench_creation.py` (MLX). Prints `op<TAB>dtype<TAB>bytes<TAB>create_us<TAB>
//! create_eval_us`, then `small<TAB>shape<TAB>us`.

use std::time::Instant;

use nuvec_core::{Array, DType, Device, Result, Stream, eval, ops, set_default_stream};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let make = |op: &str, n: i32, dtype: DType| -> Array {
        match op {
            "zeros" => ops::zeros([n], dtype),
            "ones" => ops::ones([n], dtype),
            _ => ops::full([n], Array::from(1.5f32).astype(dtype)),
        }
    };
    // Warm-up: the GPU's clocks rise under load.
    for _ in 0..300 {
        eval(&[&make("ones", 1 << 22, DType::F32)])?;
    }
    for dtype in [DType::F32, DType::BF16] {
        for op in ["zeros", "ones", "full"] {
            for bytes in [4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20] {
                let n = (bytes / dtype.size()) as i32;
                let reps = if bytes >= 16 << 20 { 10 } else { 100 };
                eval(&[&make(op, n, dtype)])?;
                let (mut create, mut both) = (Vec::new(), Vec::new());
                for _ in 0..reps {
                    let t = Instant::now();
                    let a = make(op, n, dtype);
                    create.push(t.elapsed().as_secs_f64() * 1e6);
                    eval(&[&a])?;
                    both.push(t.elapsed().as_secs_f64() * 1e6);
                }
                println!(
                    "{op}\t{dtype}\t{bytes}\t{:.1}\t{:.1}",
                    median(create),
                    median(both)
                );
            }
        }
    }
    for (label, shape) in [("0-d", vec![]), ("16", vec![16])] {
        let x = ops::ones(shape.clone(), DType::F32);
        eval(&[&x])?;
        let mut runs = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            let mut y = x.clone();
            for _ in 0..1000 {
                y = &y + ops::zeros(shape.clone(), DType::F32);
            }
            eval(&[&y])?;
            runs.push(t.elapsed().as_secs_f64() * 1e6);
        }
        println!("small\t{label}\t{:.1}", median(runs));
    }
    Ok(())
}
