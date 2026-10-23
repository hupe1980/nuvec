//! Generates text with a Llama-family model, as `mlx_lm.generate`, and prints mlx-lm's
//! statistics.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example generate -- --model mlx-community/Qwen3-8B-4bit \
//!     --prompt "Write a story" [--max-tokens 256] [--temp 0] [--top-p 0] [--min-p 0] \
//!     [--top-k 0] [--chat]
//! ```
//!
//! `--model` is a directory or a Hugging Face repository in the local cache. `--chat` wraps
//! the prompt in the ChatML template (Qwen) as a user message.

use std::io::Write;
use std::time::Instant;

use nuvec_lm::{GenerateOptions, Sampler, StreamDecoder, WiredLimit, generate, load};

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model_id = value("--model").unwrap_or_else(|| "mlx-community/Qwen3-8B-4bit".into());
    let mut prompt = value("--prompt").unwrap_or_else(|| "Hello".into());
    if args.iter().any(|a| a == "--chat") {
        prompt = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    }
    let number =
        |flag: &str, default: f32| value(flag).map_or(default, |v| v.parse().expect("a number"));
    let options = GenerateOptions {
        max_tokens: value("--max-tokens").map_or(256, |v| v.parse().expect("a number")),
        sampler: Sampler {
            temperature: number("--temp", 0.0),
            top_p: number("--top-p", 0.0),
            min_p: number("--min-p", 0.0),
            top_k: value("--top-k").map_or(0, |v| v.parse().expect("a number")),
            ..Default::default()
        },
        ..Default::default()
    };

    let (model, tokenizer) = load(&model_id)?;
    let ids = tokenizer.encode(&prompt);

    println!("==========");
    // Wired before the clock starts, as mlx-lm's `stream_generate` does.
    let _wired = WiredLimit::new();
    let start = Instant::now();
    let mut generation = generate(&model, &ids, options)?;
    let first = generation.next().map_or(Ok(None), |t| t.map(Some))?;
    let prompt_seconds = start.elapsed().as_secs_f64();
    let decode_start = Instant::now();
    let mut decoder = StreamDecoder::default();
    let mut count = 0usize;
    let mut out = std::io::stdout();
    // Generation ends at an end-of-sequence token (not yielded) or `--max-tokens`.
    for id in first.into_iter().map(Ok).chain(generation) {
        let id = id?;
        count += 1;
        write!(out, "{}", decoder.push(&tokenizer, id)).ok();
        out.flush().ok();
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    println!("{}", decoder.finish());
    println!("==========");
    println!(
        "Prompt: {} tokens, {:.3} tokens-per-sec",
        ids.len(),
        ids.len() as f64 / prompt_seconds
    );
    // The first token's time counts as the prompt's (as mlx-lm reports it).
    println!(
        "Generation: {count} tokens, {:.3} tokens-per-sec",
        count.saturating_sub(1) as f64 / decode_seconds
    );
    if let Some(d) = nuvec::gpu::Device::shared() {
        println!(
            "Peak memory: {:.3} GB allocated (memory-mapped weights not included)",
            d.memory().peak as f64 / 1e9
        );
    }
    Ok(())
}
