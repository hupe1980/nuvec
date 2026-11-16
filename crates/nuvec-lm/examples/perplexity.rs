//! Perplexity of a text under a model, read in chunks through one KV cache (bfloat16 or
//! quantized from the first position), so later chunks attend to the cached ones: what a
//! KV cache format costs in accuracy.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example perplexity -- --model mlx-community/Qwen3-8B-4bit \
//!     --text-file text.txt [--tokens 8192] [--chunk 512] [--kv-bits 4] [--kv-group-size 64] \
//!     [--kv-rotate]
//! ```
//!
//! Prints one JSON line: tokens scored, mean negative log-likelihood, perplexity and the
//! peak of allocated memory.

use nuvec::{Array, ops};
use nuvec_lm::{KvQuant, load};

fn main() -> nuvec::Result<()> {
    nuvec::set_default_stream(nuvec::Stream::default_for(nuvec::Device::gpu(0)));
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let number =
        |flag: &str, default: usize| value(flag).map_or(default, |v| v.parse().expect("a number"));
    let model_id = value("--model").unwrap_or_else(|| "mlx-community/Qwen3-8B-4bit".into());
    let path = value("--text-file").expect("--text-file");
    let text =
        std::fs::read_to_string(&path).map_err(|e| nuvec::Error::Io(format!("{path}: {e}")))?;
    let (model, tokenizer) = load(&model_id)?;
    let mut ids = tokenizer.encode_text(&text);
    ids.truncate(number("--tokens", 8192) + 1);
    let chunk = number("--chunk", 512);
    let quant = value("--kv-bits").map(|bits| KvQuant {
        bits: bits.parse().expect("a number"),
        group_size: number("--kv-group-size", 64) as i32,
        start: 0,
        rotate: args.iter().any(|a| a == "--kv-rotate"),
    });
    let mut cache = model.new_cache();
    if let Some(q) = quant {
        for c in &mut cache {
            c.to_quantized(q)?;
        }
    }
    let (mut nll, mut scored) = (0.0f64, 0usize);
    let inputs = &ids[..ids.len() - 1];
    for (i, part) in inputs.chunks(chunk).enumerate() {
        let at = i * chunk;
        let tokens = Array::from_slice(part, [1, part.len() as i32]);
        let targets = Array::from_slice(&ids[at + 1..at + 1 + part.len()], [part.len() as i32]);
        let logits = model.forward(&tokens, &mut cache);
        let logits = logits.reshape([part.len() as i32, -1]);
        let loss = ops::sum(&nuvec::fast::cross_entropy(&logits, &targets), ..);
        nll += f64::from(loss.item::<f32>()?);
        scored += part.len();
    }
    let mean = nll / scored as f64;
    println!(
        "{{\"tokens\": {scored}, \"nll\": {mean:.5}, \"perplexity\": {:.4}, \
         \"peak_bytes\": {}}}",
        mean.exp(),
        nuvec::memory::stats().peak
    );
    Ok(())
}
