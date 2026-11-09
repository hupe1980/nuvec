//! Decompositions on the CPU, compared against MLX's CPU by `scripts/bench_linalg.py`.
//! Prints `name<TAB>size<TAB>ms` per case: the best of five timed runs after a warm-up.

use std::time::Instant;

use nuvec_core::{Array, DType, Result, linalg, ops, random};

/// A benchmarked case: its name and the outputs it evaluates.
type Case<'a> = (&'static str, Box<dyn Fn() -> Vec<Array> + 'a>);

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    for n in [64usize, 256, 1024] {
        let a = random::normal([n, n], DType::F32, 0.0, 1.0, Some(&random::key(n as u64)));
        let spd = a
            .matmul(&ops::swapaxes(&a, -1, -2))
            .add(ops::eye(n as i32, None, 0, DType::F32).multiply(n as f64));
        let b = random::normal([n, 16], DType::F32, 0.0, 1.0, Some(&random::key(7)));
        nuvec_core::eval(&[a.clone(), spd.clone(), b.clone()])?;
        let cases: [Case; 8] = [
            (
                "cholesky_upper",
                Box::new(|| vec![linalg::cholesky(&spd, true)]),
            ),
            ("qr", Box::new(|| linalg::qr(&a).to_vec())),
            ("svd", Box::new(|| linalg::svd(&a).to_vec())),
            ("eigh", Box::new(|| linalg::eigh(&spd, false).to_vec())),
            ("inv", Box::new(|| vec![linalg::inv(&a)])),
            ("solve", Box::new(|| vec![linalg::solve(&a, &b)])),
            ("cholesky", Box::new(|| vec![linalg::cholesky(&spd, false)])),
            ("lu", Box::new(|| linalg::lu_factor(&a).to_vec())),
        ];
        for (name, f) in &cases {
            nuvec_core::eval(&f())?;
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                nuvec_core::eval(&f())?;
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!("{name}\t{n}\t{:.3}", best * 1e3);
        }
    }
    Ok(())
}
