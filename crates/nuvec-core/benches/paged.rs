//! Paged decode attention against the same keys stored contiguously (`just bench-paged`):
//! gpt-oss's shape (64 query heads, 8 key heads, head size 64, a window of 128 keys,
//! sinks), 8 sequences, 4K and 32K positions each; and Qwen3-0.6B's served decode (16
//! query heads, 8 key heads, head size 128, 4 sequences of 640 positions). MLX has no paged
//! attention, so the reference is nuvec's contiguous vector kernel (MLX's). Prints
//! `name<TAB>dtype<TAB>us_paged<TAB>us_contiguous<TAB>paged/contiguous`.

use std::time::Instant;

use nuvec_core::fast::{self, AttentionOptions};
use nuvec_core::{
    Array, DType, Device, Result, Stream, async_eval, eval, random, set_default_stream,
};

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    // (name, sequences, query heads, key heads, head size, window, sinks, positions)
    type Case = (&'static str, i32, i32, i32, i32, Option<i32>, bool, i32);
    let cases: [Case; 3] = [
        ("paged-window-sinks-4k", 8, 64, 8, 64, Some(128), true, 4096),
        (
            "paged-window-sinks-32k",
            8,
            64,
            8,
            64,
            Some(128),
            true,
            32768,
        ),
        ("paged-qwen3-0.6b-b4", 4, 16, 8, 128, None, false, 640),
    ];
    let bs = 16i32;
    for (dtype, label) in [(DType::F16, "float16"), (DType::BF16, "bfloat16")] {
        for (name, b, hq, hkv, d, w, with_sinks, len) in cases {
            let uniform = |s: &[i32]| random::uniform(-1.0, 1.0, s, DType::F32, None).astype(dtype);
            let q = uniform(&[b, hq, 1, d]);
            let blocks = len / bs;
            // Sequence `i` owns blocks `i·blocks..(i + 1)·blocks`, in order.
            let pool = uniform(&[b * blocks, hkv, bs, d]);
            let tables = nuvec_core::ops::arange(0, b * blocks, 1).reshape([b, blocks]);
            let lengths = Array::from(vec![len; b as usize]);
            let keys = pool
                .reshape([b, blocks, hkv, bs, d])
                .transpose([0, 2, 1, 3, 4])
                .reshape([b, hkv, len, d])
                .contiguous();
            let sinks = uniform(&[hq]);
            eval(&[&q, &pool, &tables, &lengths, &keys, &sinks])?;
            let mut opts = AttentionOptions::causal();
            if let Some(w) = w {
                opts = opts.window(w);
            }
            if with_sinks {
                opts = opts.sinks(Some(&sinks));
            }
            let paged = || fast::paged_attention(&q, &pool, &pool, &tables, &lengths, 0.125, &opts);
            let contiguous = || fast::scaled_dot_product_attention(&q, &keys, &keys, 0.125, &opts);
            let time = |f: &dyn Fn() -> Array| -> Result<f64> {
                let reps = 400;
                eval(&[&f()])?; // warm-up: pipelines and clocks
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let t = Instant::now();
                    for _ in 0..reps {
                        async_eval(&[f()])?;
                    }
                    eval(&[&f()])?;
                    best = best.min(t.elapsed().as_secs_f64() / reps as f64);
                }
                Ok(best * 1e6)
            };
            let (p, c) = (time(&paged)?, time(&contiguous)?);
            println!("{name}\t{label}\t{p:.1}\t{c:.1}\t{:.2}", p / c);
        }
    }
    Ok(())
}
