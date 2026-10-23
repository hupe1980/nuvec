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
//! token or after `--max-tokens` (`--ignore-eos` always runs to `--max-tokens`). With
//! `NUVEC_PROFILE=1`, the GPU time per kernel of each prompt length's last run goes to
//! stderr (every dispatch then runs alone, so throughput drops; `--max-tokens 1` profiles
//! prefill). `--draft N` decodes speculatively with prompt-lookup drafts of up to `N`
//! tokens and adds the draft counts.
//!
//! Token-id prompts (files with one prompt per line, comma-separated ids):
//!
//! - `--ids-file A[,B…]`: time to first token of each file's first line, per run.
//! - `--phase memory-load|memory-decode|memory-prefill:N --ids-file F`: one phase in this
//!   process, then the peak memory (allocator plus mapped files) and its parts.
//! - `--phase load --first-token-ids F`: seconds to a loaded model and to the first token.
//! - `--batch N --ids-file F`: the first `N` lines served together by a `Batcher`
//!   (128 tokens each, end-of-sequence ignored), compared with each prompt alone.

use std::time::Instant;

use nuvec_lm::{
    Batcher, Event, GenerateOptions, PromptLookup, WiredLimit, generate, load, speculative_generate,
};

/// The prompts of an ids file: one per line, comma-separated token ids.
fn read_ids(path: &str) -> Vec<Vec<u32>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|t| {
                    t.trim()
                        .parse()
                        .unwrap_or_else(|_| fail(&format!("{path}: bad id {t:?}")))
                })
                .collect()
        })
        .collect()
}

fn fail(msg: &str) -> ! {
    eprintln!("bench: {msg}");
    std::process::exit(1)
}

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let number = |flag: &str, default: usize| {
        value(flag).map_or(default, |v| {
            v.parse()
                .unwrap_or_else(|_| fail(&format!("{flag}: not a number")))
        })
    };
    let Some(device) = nuvec::gpu::Device::shared() else {
        fail("no Metal device");
    };
    let model_id = value("--model").unwrap_or_else(|| "mlx-community/Qwen3-8B-4bit".into());
    let text = value("--text").unwrap_or_else(|| "Hello. ".into());
    let repeats: Vec<usize> = value("--repeats").map_or_else(Vec::new, |v| {
        v.split(',')
            .map(|r| {
                r.parse()
                    .unwrap_or_else(|_| fail("--repeats: not a number"))
            })
            .collect()
    });
    let runs = number("--runs", 2);
    let draft = number("--draft", 0);
    let ignore_eos = args.iter().any(|a| a == "--ignore-eos");
    let phase = value("--phase");
    let options = GenerateOptions {
        max_tokens: number("--max-tokens", 128),
        ..Default::default()
    };

    let load_start = Instant::now();
    let (mut model, tokenizer) = load(&model_id)?;
    let model_ready = load_start.elapsed().as_secs_f64();
    if ignore_eos || value("--batch").is_some() || phase.is_some() {
        model.config.eos_token_ids.clear();
    }
    let _wired = WiredLimit::new();

    // The time to the first token, then the tokens after it, the time they took and the
    // count (the first included).
    let drafts = std::cell::Cell::new((0, 0));
    let run = |ids: &[u32], options: GenerateOptions| -> nuvec::Result<(f64, f64, usize)> {
        let start = Instant::now();
        let mut speculative = None;
        let mut plain = None;
        let generation: &mut dyn Iterator<Item = nuvec::Result<u32>> = if draft > 0 {
            let g = speculative_generate(&model, ids, options, PromptLookup::default(), draft)?;
            speculative.insert(g)
        } else {
            plain.insert(generate(&model, ids, options)?)
        };
        let first = generation.next().map_or(Ok(None), |t| t.map(Some))?;
        let first_seconds = start.elapsed().as_secs_f64();
        let decode_start = Instant::now();
        let mut count = 0;
        // Both end at an end-of-sequence token (not yielded).
        for id in first.into_iter().map(Ok).chain(generation) {
            id?;
            count += 1;
        }
        let decode_seconds = decode_start.elapsed().as_secs_f64();
        if let Some(g) = &speculative {
            let stats = g.stats();
            drafts.set((stats.drafted, stats.accepted));
        }
        Ok((first_seconds, decode_seconds, count))
    };
    let warm_up = || run(&tokenizer.encode(&text), options);

    match phase.as_deref() {
        Some("load") => {
            let path = value("--first-token-ids").unwrap_or_else(|| fail("--first-token-ids"));
            let ids = &read_ids(&path)[0];
            let one = GenerateOptions {
                max_tokens: 1,
                ..options
            };
            let (first, _, _) = run(ids, one)?;
            println!(
                "{{\"phase\": \"load\", \"model_ready_s\": {model_ready:.4}, \
                 \"first_token_s\": {:.4}}}",
                model_ready + first
            );
            return Ok(());
        }
        Some(p) if p.starts_with("memory-") => {
            if p != "memory-load" {
                let path = value("--ids-file").unwrap_or_else(|| fail("--ids-file"));
                let ids = &read_ids(&path)[0];
                warm_up()?;
                device.reset_peak();
                let max_tokens = if p == "memory-decode" {
                    options.max_tokens
                } else {
                    1
                };
                run(
                    ids,
                    GenerateOptions {
                        max_tokens,
                        ..options
                    },
                )?;
            }
            let m = device.memory();
            println!(
                "{{\"phase\": \"{p}\", \"peak_with_mapped_bytes\": {}, \"peak_bytes\": {}, \
                 \"active_bytes\": {}, \"mapped_bytes\": {}}}",
                m.peak_with_mapped,
                m.peak,
                m.active,
                m.active_with_mapped.saturating_sub(m.active)
            );
            return Ok(());
        }
        Some(p) => fail(&format!("unknown phase {p:?}")),
        None => {}
    }

    if let Some(n) = value("--batch") {
        let n: usize = n.parse().unwrap_or_else(|_| fail("--batch: not a number"));
        let path = value("--ids-file").unwrap_or_else(|| fail("--ids-file"));
        let prompts: Vec<Vec<u32>> = read_ids(&path).into_iter().take(n).collect();
        if prompts.len() < n {
            fail(&format!("{path}: fewer than {n} prompts"));
        }
        warm_up()?;
        let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let block_size = 16;
        let blocks = n * (longest + options.max_tokens).div_ceil(block_size) + n;
        let mut batcher = Batcher::new(&model, blocks, block_size, n)?;
        let start = Instant::now();
        let mut outputs = vec![Vec::new(); n];
        let mut ids = Vec::new();
        for p in &prompts {
            ids.push(batcher.submit(p.clone(), options)?);
        }
        let mut finished = 0;
        let mut first_tokens = 0;
        let mut decode_start = start;
        while finished < n {
            for event in batcher.step()? {
                match event {
                    Event::Token { id, token } => {
                        let i = ids
                            .iter()
                            .position(|&r| r == id)
                            .expect("a submitted request");
                        if outputs[i].is_empty() {
                            first_tokens += 1;
                            if first_tokens == n {
                                decode_start = Instant::now();
                            }
                        }
                        outputs[i].push(token);
                    }
                    Event::Finished { .. } => finished += 1,
                }
            }
        }
        let wall = start.elapsed().as_secs_f64();
        let decode_wall = decode_start.elapsed().as_secs_f64();
        drop(batcher);
        let total: usize = outputs.iter().map(Vec::len).sum();
        let mut matches = true;
        for (prompt, out) in prompts.iter().zip(&outputs) {
            let solo: Vec<u32> =
                generate(&model, prompt, options)?.collect::<nuvec::Result<_>>()?;
            matches &= &solo == out;
        }
        println!(
            "{{\"level\": {n}, \"total_tokens\": {total}, \"wall_s\": {wall:.4}, \
             \"decode_wall_s\": {decode_wall:.4}, \"decode_tokens\": {}, \
             \"matches_solo\": {matches}}}",
            total - n
        );
        return Ok(());
    }

    // Warm-up (kernel compilation, first faults of the weights).
    warm_up()?;
    device.reset_peak();
    if let Some(files) = value("--ids-file") {
        let one = GenerateOptions {
            max_tokens: 1,
            ..options
        };
        for path in files.split(',') {
            let ids = &read_ids(path)[0];
            for _ in 0..runs {
                let (first, _, _) = run(ids, one)?;
                println!(
                    "{{\"ids_file\": \"{}\", \"prompt_tokens\": {}, \"ttft_s\": {first:.4}, \
                     \"prefill\": {:.2}}}",
                    std::path::Path::new(path)
                        .file_name()
                        .map_or_else(String::new, |f| f.to_string_lossy().into_owned()),
                    ids.len(),
                    ids.len() as f64 / first
                );
            }
        }
    }
    for &r in &repeats {
        let ids = tokenizer.encode(&text.repeat(r));
        for _ in 0..runs {
            nuvec::gpu::profile::reset();
            let (first, decode_seconds, tokens) = run(&ids, options)?;
            let prefill = ids.len() as f64 / first;
            let decode = (tokens.max(2) - 1) as f64 / decode_seconds;
            // The one peak metric: allocator memory plus mapped weights (MLX's
            // `get_peak_memory` counts its copied weights).
            let peak = device.memory().peak_with_mapped;
            println!(
                "{{\"repeats\": {r}, \"prompt_tokens\": {}, \"prefill\": {prefill:.2}, \
                 \"ttft_s\": {first:.4}, \"decode\": {decode:.2}, \"decode_s\": \
                 {decode_seconds:.4}, \"tokens\": {tokens}, \"peak_bytes\": {peak}, \
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
