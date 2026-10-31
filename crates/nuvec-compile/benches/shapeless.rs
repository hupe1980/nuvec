//! Shapeless replay against shaped replay (spec 067 SC-003), on the GPU: mrun's compiled
//! activations at decode and prefill chunk sizes. The shaped function is called at one
//! size only, so it replays a cached trace; the shapeless one has traced once, at another
//! size. Prints `name<TAB>rows<TAB>dtype<TAB>us_shaped<TAB>us_shapeless<TAB>ratio`.

use std::time::Instant;

use nuvec_compile::{Options, compile_flat_with};
use nuvec_core::{Array, DType, Device, Result, Stream, async_eval, random, set_default_stream};

fn silu(xs: &[Array]) -> Result<Vec<Array>> {
    Ok(vec![xs[0].multiply(xs[0].sigmoid())])
}

fn swiglu(xs: &[Array]) -> Result<Vec<Array>> {
    let (g, u) = (&xs[0], &xs[1]);
    Ok(vec![g.multiply(g.sigmoid()).multiply(u)])
}

type Flat<'a> = dyn Fn(&[Array]) -> Result<Vec<Array>> + 'a;

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let uniform =
        |s: &[usize], dtype| random::uniform(-1.0, 1.0, s, DType::F32, None).astype(dtype);
    for (dtype, label) in [(DType::F16, "float16"), (DType::BF16, "bfloat16")] {
        // Gated DeltaNet's decay (mrun's `compute_g`) over 32 heads, with captured constants.
        let a_log = uniform(&[32], DType::F32);
        let dt_bias = uniform(&[32], DType::F32);
        nuvec_core::eval(&[&a_log, &dt_bias])?;
        let gate = |xs: &[Array]| -> Result<Vec<Array>> {
            let a = xs[0].astype(DType::F32).add(&dt_bias);
            Ok(vec![
                a_log
                    .exp()
                    .negative()
                    .multiply(a.logaddexp(0.0))
                    .astype(xs[0].dtype()),
            ])
        };
        let flat: [(&str, usize, usize, &Flat); 3] = [
            ("silu", 1, 4096, &silu),
            ("swiglu", 2, 12288, &swiglu),
            ("compute_g", 1, 32, &gate),
        ];
        for (name, arity, width, f) in flat {
            let shapeless = compile_flat_with(Options::default().shapeless(true), f);
            // The one trace, at a size the measurements do not use.
            let first = (0..arity)
                .map(|_| uniform(&[3, width], dtype))
                .collect::<Vec<_>>();
            nuvec_core::eval(&shapeless(&first)?)?;
            for rows in [1, 512, 2048, 4096] {
                let shaped = compile_flat_with(Options::default(), f);
                let xs = (0..arity)
                    .map(|_| uniform(&[rows, width], dtype))
                    .collect::<Vec<_>>();
                nuvec_core::eval(&xs)?;
                let reps = (50_000_000 / (rows * width)).clamp(50, 4000);
                let burst = |run: &dyn Fn() -> Result<Vec<Array>>| -> Result<f64> {
                    let t = Instant::now();
                    for _ in 0..reps {
                        async_eval(&run()?)?;
                    }
                    nuvec_core::eval(&run()?)?;
                    Ok(t.elapsed().as_secs_f64() / reps as f64)
                };
                let best = |run: &dyn Fn() -> Result<Vec<Array>>| -> Result<f64> {
                    burst(run)?;
                    let times = (0..5).map(|_| burst(run)).collect::<Result<Vec<_>>>()?;
                    Ok(times.into_iter().fold(f64::INFINITY, f64::min) * 1e6)
                };
                // Alternating, so drift affects both.
                let (mut a, mut b) = (f64::INFINITY, f64::INFINITY);
                for _ in 0..2 {
                    a = a.min(best(&|| shaped(&xs))?);
                    b = b.min(best(&|| shapeless(&xs))?);
                }
                println!("{name}\t{rows}\t{label}\t{a:.2}\t{b:.2}\t{:.3}", b / a);
            }
        }
    }
    Ok(())
}
