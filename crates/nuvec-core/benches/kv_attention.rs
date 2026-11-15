//! Decode attention over a quantized KV cache (`scripts/bench_kv_attention.py`,
//! `just bench-kv-attention`): one query row over 4K–64K keys, bf16 vector attention against
//! mlx-lm's quantized-cache composition (scores and values as quantized products, a precise
//! softmax) at 8 and 4 bits, groups of 64. Prints `variant<TAB>lk<TAB>us` per cell:
//! microseconds per call, the median over repetitions of 32 independent calls evaluated
//! together.

use std::time::Instant;

use nuvec_core::fast::AttentionOptions;
use nuvec_core::{
    Array, DType, Device, Quant, Quantized, Result, Stream, random, set_default_stream,
};

const CALLS: usize = 32;
const D: i32 = 128;
const HQ: i32 = 32;
const HKV: i32 = 8;

fn quantized_cache(a: &Array, bits: i32) -> Quantized {
    let q = a.quantize(Quant::affine(64, bits));
    let e = |x: &Array| x.expand_dims(-3);
    Quantized::affine(e(&q.weight), e(&q.scales), e(&q.biases.unwrap()), 64, bits)
}

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let scale = (D as f32).powf(-0.5);
    for &lk in &[4096, 16384, 65536] {
        let k = random::normal([1, HKV, lk, D], DType::BF16, 0.0, 1.0, None);
        let v = random::normal([1, HKV, lk, D], DType::BF16, 0.0, 1.0, None);
        let qs: Vec<Array> = (0..CALLS)
            .map(|_| random::normal([1, HQ, 1, D], DType::BF16, 0.0, 1.0, None))
            .collect();
        let (k8, v8) = (quantized_cache(&k, 8), quantized_cache(&v, 8));
        let (k4, v4) = (quantized_cache(&k, 4), quantized_cache(&v, 4));
        nuvec_core::eval(&(&k, &v, &qs, &k8, &v8, &k4, &v4))?;
        let opts = AttentionOptions::default();
        let composed = |q: &Array, kq: &Quantized, vq: &Quantized| {
            let q = q.multiply(scale).reshape([1, HKV, HQ / HKV, 1, D]);
            let p = q.quantized_matmul(kq, true).softmax(-1);
            p.quantized_matmul(vq, false).reshape([1, HQ, 1, D])
        };
        type Variant<'a> = (&'a str, Box<dyn Fn(&Array) -> Array + 'a>);
        let variants: [Variant<'_>; 3] = [
            (
                "bf16",
                Box::new(|q| {
                    nuvec_core::fast::scaled_dot_product_attention(q, &k, &v, scale, &opts)
                }),
            ),
            ("q8", Box::new(|q| composed(q, &k8, &v8))),
            ("q4", Box::new(|q| composed(q, &k4, &v4))),
        ];
        for (name, f) in &variants {
            let run = || -> Result<f64> {
                let t = Instant::now();
                let outs: Vec<Array> = qs.iter().map(f).collect();
                nuvec_core::eval(&outs)?;
                Ok(t.elapsed().as_secs_f64() * 1e6 / CALLS as f64)
            };
            run()?;
            let mut times = (0..9).map(|_| run()).collect::<Result<Vec<_>>>()?;
            times.sort_by(f64::total_cmp);
            println!("{name}\t{lk}\t{:.1}", times[times.len() / 2]);
        }
    }
    Ok(())
}
