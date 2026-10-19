//! Generation throughput for several prompt lengths with one model load, as mlx-lm's
//! `stream_generate` measures it (prompt time includes the first token; memory is wired
//! before the clock starts). Prints one JSON object per run.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example bench -- --model mlx-community/Qwen3-8B-4bit \
//!     --text "Some paragraph. " --repeats 1,40,160 [--runs 2] [--max-tokens 128]
//! ```
//!
//! Each prompt is `text` repeated; generation is greedy and stops at an end-of-sequence
//! token or after `--max-tokens`. With `NUVEC_PROFILE=1`, the GPU time per kernel of each
//! prompt length's last run goes to stderr (every dispatch then runs alone, so throughput
//! drops; `--max-tokens 1` profiles prefill). `--draft N` decodes speculatively with
//! prompt-lookup drafts of up to `N` tokens and adds the draft counts.

use std::time::Instant;

use nuvec_lm::{GenerateOptions, PromptLookup, WiredLimit, generate, load, speculative_generate};

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model_id = value("--model").unwrap_or_else(|| "mlx-community/Qwen3-8B-4bit".into());
    let text = value("--text").unwrap_or_else(|| "Hello. ".into());
    let repeats: Vec<usize> = value("--repeats")
        .unwrap_or_else(|| "1".into())
        .split(',')
        .map(|r| r.parse().expect("a number"))
        .collect();
    let runs: usize = value("--runs").map_or(2, |v| v.parse().expect("a number"));
    let draft: usize = value("--draft").map_or(0, |v| v.parse().expect("a number"));
    let options = GenerateOptions {
        max_tokens: value("--max-tokens").map_or(128, |v| v.parse().expect("a number")),
        ..Default::default()
    };

    let (model, tokenizer) = load(&model_id)?;
    let eos = model.config.eos_token_ids.clone();
    let _wired = WiredLimit::new();
    let drafts = std::cell::Cell::new((0, 0));
    let run = |ids: &[u32]| -> nuvec::Result<(f64, f64, usize)> {
        let start = Instant::now();
        let mut speculative = None;
        let mut plain = None;
        let generation: &mut dyn Iterator<Item = nuvec::Result<u32>> = if draft > 0 {
            let g = speculative_generate(&model, ids, options, PromptLookup::default(), draft)?;
            speculative.insert(g)
        } else {
            plain.insert(generate(&model, ids, options)?)
        };
        let first = generation.next().transpose()?;
        let prompt_seconds = start.elapsed().as_secs_f64();
        let decode_start = Instant::now();
        let mut count = 0;
        for id in first.into_iter().map(Ok).chain(generation) {
            count += 1;
            if eos.contains(&id?) {
                break;
            }
        }
        let decode_seconds = decode_start.elapsed().as_secs_f64();
        if let Some(g) = &speculative {
            let stats = g.stats();
            drafts.set((stats.drafted, stats.accepted));
        }
        let decode = (count.max(2) - 1) as f64 / decode_seconds;
        Ok((ids.len() as f64 / prompt_seconds, decode, count))
    };
    // Warm-up (kernel compilation, first faults of the weights).
    run(&tokenizer.encode(&text))?;
    for &r in &repeats {
        let ids = tokenizer.encode(&text.repeat(r));
        for _ in 0..runs {
            nuvec::gpu::profile::reset();
            let (prefill, decode, tokens) = run(&ids)?;
            let peak = nuvec::gpu::Device::shared().map_or(0, |d| d.memory().peak);
            println!(
                "{{\"repeats\": {r}, \"prompt_tokens\": {}, \"prefill\": {prefill:.2}, \
                 \"decode\": {decode:.2}, \"tokens\": {tokens}, \"peak_bytes\": {peak}, \
                 \"drafted\": {}, \"accepted\": {}}}",
                ids.len(),
                drafts.get().0,
                drafts.get().1
            );
        }
        if nuvec::gpu::profile::enabled() {
            let rows = nuvec::gpu::profile::report();
            let total: f64 = rows.iter().map(|r| r.2).sum();
            eprintln!(
                "GPU time, {} prompt tokens: {:.1} ms",
                ids.len(),
                total * 1e3
            );
            for (kernel, n, seconds) in rows.iter().take(25) {
                eprintln!(
                    "{:>8.2} ms {:>5.1}% {n:>6}  {kernel}",
                    seconds * 1e3,
                    100.0 * seconds / total
                );
            }
        }
    }
    Ok(())
}
