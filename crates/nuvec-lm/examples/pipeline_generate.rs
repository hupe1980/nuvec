//! Generates text with a model split across processes or Macs, one pipeline stage per rank
//! (mlx-lm's `mlx_lm.generate --pipeline`). Start it on every rank with MLX's launcher:
//!
//! ```text
//! cargo build --release -p nuvec-lm --example pipeline_generate
//! mlx.launch --backend ring --hostfile hosts.json -- \
//!     ./target/release/examples/pipeline_generate --model <dir> --prompt "Write a story" \
//!     [--split even|memory|8,12,12] [--max-tokens 256] [--temp 0] \
//!     [--batch N [--micro-batches M]] [--tensor] [--warmup] \
//!     [--random-prompt N] [--ignore-eos] [--trials T]
//! ```
//!
//! Every rank loads only its stage's weights and yields the same tokens; rank 0 prints the
//! text and mlx-lm's statistics. `--split` is mlx-lm's even split (default), the split by
//! each Mac's memory, or layers per rank (rank 0 first). `--batch N` serves `N` copies of
//! the prompt with a `Batcher` and prints the decode throughput;
//! `--micro-batches M` sets the micro-batches in flight (default one per stage; 1 is one
//! batch). `--tensor` splits every layer across the ranks instead (tensor parallelism,
//! mlx-lm's `model.shard`; decode at RDMA latency over JACCL). `--warmup` generates once
//! untimed first, as `mlx_lm.benchmark` does. As `mlx_lm.benchmark`: `--random-prompt N`
//! uses `N` random token ids (the same on every rank), `--ignore-eos` runs to
//! `--max-tokens`, and `--trials T` times `T` generations and prints their means. Decode
//! is printed both as `(tokens − 1) / time after the first token` and as mlx-lm computes
//! it (`tokens / time after the first token`).

use std::time::Instant;

use nuvec_lm::{
    Batcher, BatcherOptions, Event, GenerateOptions, Sampler, Split, StreamDecoder, Tokenizer,
    WiredLimit, generate,
};

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let dir = nuvec_lm::resolve(&value("--model").expect("--model"))?;
    let prompt = value("--prompt").unwrap_or_else(|| "Hello".into());
    let split = match value("--split").as_deref() {
        None | Some("even") => Split::Even,
        Some("memory") => Split::memory(),
        Some(counts) => Split::Layers(
            counts
                .split(',')
                .map(|c| c.trim().parse().expect("layers per rank"))
                .collect(),
        ),
    };
    let options = GenerateOptions {
        max_tokens: value("--max-tokens").map_or(256, |v| v.parse().expect("a number")),
        sampler: Sampler {
            temperature: value("--temp").map_or(0.0, |v| v.parse().expect("a number")),
            ..Default::default()
        },
        // Every rank must sample with the same keys.
        seed: Some(0),
        ignore_eos: args.iter().any(|a| a == "--ignore-eos"),
        ..Default::default()
    };

    let group = nuvec::distributed::init(false)?;
    let start = Instant::now();
    let model = if args.iter().any(|a| a == "--tensor") {
        nuvec_lm::load_model_sharded(&dir, &group)?
    } else {
        nuvec_lm::load_model_pipelined(&dir, &group, split)?
    };
    let load_seconds = start.elapsed().as_secs_f64();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let ids = match value("--random-prompt").map(|n| n.parse::<usize>().expect("a number")) {
        // A fixed-seed generator: every rank builds the same prompt.
        Some(n) => {
            let vocab = model.config.vocab_size as u64;
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % vocab) as u32
                })
                .collect()
        }
        None => tokenizer.encode(&prompt),
    };
    let rank0 = group.rank() == 0;
    if let Some(stage) = model.stage() {
        eprintln!(
            "rank {} of {}: layers {:?}, loaded in {load_seconds:.2} s",
            group.rank(),
            group.size(),
            stage.layers()
        );
    }

    let _wired = WiredLimit::new();
    if let Some(n) = value("--batch").map(|v| v.parse::<usize>().expect("a number")) {
        let micro_batches = value("--micro-batches").map(|m| m.parse().expect("a number"));
        let mut batcher = Batcher::new(
            &model,
            BatcherOptions {
                blocks: Some(64 * n),
                block_size: 64,
                max_batch: n,
                micro_batches,
                ..Default::default()
            },
        )?;
        for _ in 0..n {
            batcher.submit(ids.clone(), options.clone())?;
        }
        // The first token of every request (prefill), then the decode clock.
        let (mut tokens, mut started) = (0usize, None);
        let mut first = std::collections::HashSet::new();
        while batcher.is_busy() {
            for event in batcher.step()? {
                if let Event::Token { id, .. } = event {
                    tokens += 1;
                    if first.insert(id) && first.len() == n {
                        started = Some((Instant::now(), tokens));
                    }
                }
            }
        }
        if rank0 && let Some((t, at)) = started {
            println!(
                "Batch: {n} requests, {:.3} decode tokens-per-sec",
                (tokens - at) as f64 / t.elapsed().as_secs_f64()
            );
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--warmup") {
        for token in generate(&model, &ids, options.clone())? {
            token?;
        }
    }
    let trials: usize = value("--trials").map_or(1, |t| t.parse().expect("a number"));
    let mut sums = [0.0f64; 3];
    for trial in 0..trials {
        let [prompt_tps, decode_tps, mlx_tps] =
            timed(&model, &ids, &options, &tokenizer, rank0, &group)?;
        if rank0 {
            println!(
                "Trial {}: prompt {prompt_tps:.3} tokens-per-sec, generation {decode_tps:.3} tokens-per-sec (mlx-lm: {mlx_tps:.3})",
                trial + 1
            );
        }
        for (s, v) in sums.iter_mut().zip([prompt_tps, decode_tps, mlx_tps]) {
            *s += v;
        }
    }
    if rank0 {
        let [p, d, m] = sums.map(|s| s / trials as f64);
        println!("==========");
        println!("Prompt: {} tokens, {p:.3} tokens-per-sec", ids.len());
        println!(
            "Generation: {} tokens, {d:.3} tokens-per-sec",
            options.max_tokens
        );
        println!("Generation (mlx-lm): {m:.3} tokens-per-sec");
    }
    Ok(())
}

/// One timed generation: prompt, decode and mlx-lm's decode rates.
fn timed(
    model: &nuvec_lm::Model,
    ids: &[u32],
    options: &GenerateOptions,
    tokenizer: &Tokenizer,
    rank0: bool,
    group: &nuvec::distributed::Group,
) -> nuvec::Result<[f64; 3]> {
    nuvec::metal::profile::reset();
    let start = Instant::now();
    let mut generation = generate(model, ids, options.clone())?;
    let first = generation.next().map_or(Ok(None), |t| t.map(Some))?;
    let prompt_seconds = start.elapsed().as_secs_f64();
    if nuvec::metal::profile::enabled() {
        // GPU time per kernel of this rank's prefill (`NUVEC_PROFILE=1`), to stderr.
        let rows = nuvec::metal::profile::report();
        let total: f64 = rows.iter().map(|r| r.2).sum();
        eprintln!(
            "rank {}: GPU time, prefill: {:.1} ms",
            group.rank(),
            total * 1e3
        );
        for (kernel, n, seconds) in rows.iter().take(20) {
            eprintln!(
                "rank {} {:>8.2} ms {n:>6}  {kernel}",
                group.rank(),
                seconds * 1e3
            );
        }
    }
    let decode_start = Instant::now();
    let mut decoder = StreamDecoder::default();
    let mut count = 0usize;
    let mut text = String::new();
    for id in first.into_iter().map(Ok).chain(generation) {
        let id = id?;
        count += 1;
        if rank0 {
            text.push_str(&decoder.push(tokenizer, id));
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    if rank0 {
        text.push_str(&decoder.finish());
        eprintln!("{text}");
    }
    Ok([
        ids.len() as f64 / prompt_seconds,
        count.saturating_sub(1) as f64 / decode_seconds,
        count as f64 / decode_seconds,
    ])
}
