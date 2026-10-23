//! Compiled element-wise functions on the GPU, compared against MLX by
//! `scripts/bench_compile.py` (`just bench-compile`). Prints
//! `name<TAB>size<TAB>dtype<TAB>us_uncompiled<TAB>us_compiled` per case: time per call in
//! sustained `async_eval` bursts after a warm-up. The cases and method match the script's.

use std::time::Instant;

use nuvec_compile::compile_flat;
use nuvec_core::{Array, DType, Device, Result, Stream, async_eval, random, set_default_stream};

const SIZES: &[usize] = &[1 << 12, 1 << 20, 1 << 24];

/// GELU, tanh approximation: `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`.
fn gelu(xs: &[Array]) -> Result<Vec<Array>> {
    let x = &xs[0];
    let cube = x.multiply(x).multiply(x);
    let inner = x.add(cube.multiply(0.044715)).multiply(0.797_884_6);
    Ok(vec![x.multiply(0.5).multiply(inner.tanh().add(1.0))])
}

/// SwiGLU gating: `silu(g) · u`.
fn swiglu(xs: &[Array]) -> Result<Vec<Array>> {
    let (g, u) = (&xs[0], &xs[1]);
    Ok(vec![g.multiply(g.sigmoid()).multiply(u)])
}

/// One Adam step over `(p, m, v, g)`: the new parameters and moments.
fn adam(xs: &[Array]) -> Result<Vec<Array>> {
    let (p, m, v, g) = (&xs[0], &xs[1], &xs[2], &xs[3]);
    let m = m.multiply(0.9).add(g.multiply(0.1));
    let v = v.multiply(0.999).add(g.multiply(g).multiply(0.001));
    let step = m.divide(v.sqrt().add(1e-8));
    Ok(vec![p.subtract(step.multiply(1e-3)), m, v])
}

type Case = (&'static str, usize, fn(&[Array]) -> Result<Vec<Array>>);
const CASES: &[Case] = &[("gelu", 1, gelu), ("swiglu", 2, swiglu), ("adam", 4, adam)];

fn main() -> Result<()> {
    // `cargo test --all-targets` runs bench binaries too: measure only under `cargo bench`.
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    for (dtype, label) in [(DType::F32, "float32"), (DType::F16, "float16")] {
        for &(name, arity, f) in CASES {
            let compiled = compile_flat(f);
            for &n in SIZES {
                let xs = (0..arity)
                    .map(|_| random::uniform(0.0, 1.0, [n], DType::F32, None).astype(dtype))
                    .collect::<Vec<_>>();
                nuvec_core::eval(&xs)?;
                let reps = (100_000_000 / n).clamp(20, 4000);
                let burst = |run: &dyn Fn() -> Result<Vec<Array>>| -> Result<f64> {
                    let t = Instant::now();
                    for _ in 0..reps {
                        async_eval(&run()?)?;
                    }
                    nuvec_core::eval(&run()?)?;
                    Ok(t.elapsed().as_secs_f64() / reps as f64)
                };
                let best = |run: &dyn Fn() -> Result<Vec<Array>>| -> Result<f64> {
                    burst(run)?; // warm-up: GPU clocks ramp up, kernels compile
                    let times = (0..3).map(|_| burst(run)).collect::<Result<Vec<_>>>()?;
                    Ok(times.into_iter().fold(f64::INFINITY, f64::min) * 1e6)
                };
                let plain = best(&|| f(&xs))?;
                let fused = best(&|| compiled(&xs))?;
                println!("{name}\t{n}\t{label}\t{plain:.2}\t{fused:.2}");
            }
        }
    }
    Ok(())
}
