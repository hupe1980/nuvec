//! The first token of a short prompt, part by part, in mlx-lm's `generate_step` order: the
//! prompt but its last token (one chunk, evaluated with the cache), the cache clear, then the
//! last token's step, the next step submitted, and the first token read. Pairs with
//! `scripts/ttft_parts.py`, which times mlx-lm the same way.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example ttft_parts -- --model DIR [--tokens 25] [--runs 20] [--no-clear]
//! ```
//!
//! Prints one JSON line per run: milliseconds per part, the cache bytes before and after the
//! clear, and the buffers the system made after it.

use std::path::PathBuf;
use std::time::Instant;

use nuvec::{Array, Device, Stream, async_eval, eval, set_default_stream};
use nuvec_lm::load_model;

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let dir = PathBuf::from(value("--model").expect("--model DIR"));
    let tokens: usize = value("--tokens").map_or(25, |v| v.parse().expect("a number"));
    let runs: usize = value("--runs").map_or(20, |v| v.parse().expect("a number"));
    let clear = !args.iter().any(|a| a == "--no-clear");
    let whole = args.iter().any(|a| a == "--generate");
    let wire = args.iter().any(|a| a == "--wire");
    let decode: usize = value("--decode").map_or(8, |v| v.parse().expect("a number"));
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let model = load_model(&dir)?;
    // Token ids 1000.. as the prompt: both engines see the same ids.
    let prompt: Vec<u32> = (0..tokens as u32).map(|i| 1000 + i).collect();
    let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
    if whole {
        // `generate` itself, timed as the parity record times it.
        let options = nuvec_lm::GenerateOptions {
            max_tokens: 128,
            ignore_eos: true,
            ..Default::default()
        };
        for run in 0..=runs {
            let t0 = Instant::now();
            let mut g = nuvec_lm::generate(&model, &prompt, options.clone())?;
            let t1 = Instant::now();
            let token = g.next().expect("a token")?;
            let t2 = Instant::now();
            for t in g.by_ref() {
                t?;
            }
            let t3 = Instant::now();
            drop(g);
            let t4 = Instant::now();
            if run > 0 {
                println!(
                    "{{\"engine\":\"nuvec\",\"mode\":\"generate\",\"start_ms\":{:.3},\"next_ms\":{:.3},\"first_token_ms\":{:.3},\"decode_ms\":{:.3},\"drop_ms\":{:.3},\"token\":{token}}}",
                    ms(t0, t1),
                    ms(t1, t2),
                    ms(t0, t2),
                    ms(t2, t3),
                    ms(t3, t4),
                );
            }
        }
        return Ok(());
    }
    for run in 0..=runs {
        let mut cache = model.new_cache();
        let t0 = Instant::now();
        let wired = wire.then(nuvec_lm::WiredLimit::new);
        let start = nuvec::memory::stats();
        let head = Array::from_slice(&prompt[..tokens - 1], [1, tokens as i32 - 1]);
        let _ = model.forward(&head, &mut cache);
        let state: Vec<_> = cache.iter().flat_map(|c| c.state()).collect();
        async_eval(&state)?;
        let encoded = Instant::now();
        eval(&state)?;
        let t1 = Instant::now();
        let before = nuvec::memory::stats();
        if clear {
            nuvec::memory::clear_cache();
        }
        let after = nuvec::memory::stats();
        let t2 = Instant::now();
        let last = Array::from_slice(&prompt[tokens - 1..], [1, 1]);
        let y = step(&model, &last, &mut cache);
        async_eval(std::slice::from_ref(&y))?;
        let next = step(&model, &y.reshape([1, 1]), &mut cache);
        async_eval(std::slice::from_ref(&next))?;
        let t3 = Instant::now();
        let token = y.item::<u32>()?;
        let t4 = Instant::now();
        let made = nuvec::memory::stats().system_allocations - after.system_allocations;
        // A few decode steps, as the parity record's generation continues after its first
        // token, so each run starts from a cache in the same state.
        let mut y = next;
        for _ in 0..decode {
            let n = step(&model, &y.reshape([1, 1]), &mut cache);
            async_eval(std::slice::from_ref(&n))?;
            y.item::<u32>()?;
            y = n;
        }
        y.item::<u32>()?;
        drop(wired);
        if run == 0 {
            continue;
        }
        println!(
            "{{\"engine\":\"nuvec\",\"tokens\":{tokens},\"clear\":{clear},\"chunk_ms\":{:.3},\"encode_ms\":{:.3},\"clear_ms\":{:.3},\"submit_ms\":{:.3},\"wait_ms\":{:.3},\"first_token_ms\":{:.3},\"cache_before\":{},\"cache_after\":{},\"made\":{made},\"chunk_made\":{},\"chunk_new_bytes\":{},\"token\":{token}}}",
            ms(t0, t1),
            ms(t0, encoded),
            ms(t1, t2),
            ms(t2, t3),
            ms(t3, t4),
            ms(t0, t4),
            before.cache,
            after.cache,
            before.system_allocations - start.system_allocations,
            (before.active + before.cache) as i64 - (start.active + start.cache) as i64,
        );
    }
    Ok(())
}

/// nuvec-lm's greedy step: the last position's logits, then the token.
fn step(model: &nuvec_lm::Model, x: &Array, cache: &mut [nuvec_lm::KvCache]) -> Array {
    let logits = model.forward(x, cache);
    logits.i((.., -1..)).reshape([1, -1]).argmax(-1)
}
