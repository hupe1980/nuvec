//! Pipeline parallelism across processes: each test re-runs this binary as 2–4 ranks over
//! loopback. On every model family's tiny fixture the pipelined logits
//! equal the unpipelined model's bit for bit, generation yields the same tokens on every
//! rank, and each rank loads only its stage's weights.

use std::path::PathBuf;

use nuvec::distributed::Group;
use nuvec::{Array, Device, Stream, set_default_stream};
use nuvec_lm::{
    Batcher, BatcherOptions, Event, GenerateOptions, Model, Split, generate, load_model,
    load_model_pipelined,
};
use nuvec_nn::Module;

#[path = "common/ranks.rs"]
mod ranks;

use ranks::{agree, ranks};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// The families on the shared code path, with the stage counts their layers allow.
const FAMILIES: &[(&str, &[usize])] = &[
    ("llama_bf16", &[2]),
    ("qwen2_q4", &[2]),
    ("qwen3_f32", &[2]),
    ("qwen3_mxfp4", &[2]),
    ("qwen3_moe_q4", &[2, 3]),
    ("qwen3_5_f32", &[2, 3, 4]),
    ("qwen3_5_moe_q4", &[2, 4]),
    ("gpt_oss_f32", &[2]),
];

fn prompt() -> Array {
    Array::from_slice(&[1u32, 5, 9, 13, 17, 21, 25, 29, 2, 4], [1, 10])
}

fn bits(a: &Array) -> Vec<u32> {
    a.astype(nuvec::DType::F32)
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// On every rank: the pipelined logits (prefill, then one decode step) equal the whole
/// model's on rank 0; only the stage's weights are loaded; greedy generation yields the
/// whole model's tokens on every rank.
fn stages(g: Group, name: &str) {
    set_default_stream(Stream::default_for(Device::cpu()));
    let dir = fixture(name);
    let whole = load_model(&dir).unwrap();
    let staged = load_model_pipelined(&dir, &g, Split::Even).unwrap();
    let stage = staged.stage().expect("a stage").clone();

    // Only the stage's weights are loaded.
    for (path, a) in staged.parameters() {
        let layer = path
            .split("layers.")
            .nth(1)
            .and_then(|r| r.split('.').next())
            .and_then(|i| i.parse::<usize>().ok());
        let inside = match layer {
            Some(i) if !path.starts_with("mtp.") => stage.layers().contains(&i),
            _ if path.contains("embed_tokens") => {
                stage.is_first() || (stage.is_last() && whole.config.tie_word_embeddings)
            }
            _ => stage.is_last(),
        };
        assert_eq!(
            !a.is_failed(),
            inside,
            "{name}: {path} loaded = {}",
            !a.is_failed()
        );
    }

    // Prefill and one decode step: rank 0's logits equal the whole model's, bit for bit.
    let (mut wc, mut sc) = (whole.new_cache(), staged.new_cache());
    let want = whole.forward(&prompt(), &mut wc);
    let got = staged.forward(&prompt(), &mut sc);
    let next = Array::from_slice(&[7u32], [1, 1]);
    let want_step = whole.forward(&next, &mut wc);
    let got_step = staged.forward(&next, &mut sc);
    if stage.is_last() {
        assert_eq!(bits(&got), bits(&want), "{name}: prefill logits");
        assert_eq!(bits(&got_step), bits(&want_step), "{name}: decode logits");
    } else {
        got.eval().unwrap();
        got_step.eval().unwrap();
    }

    // Greedy generation: every rank yields the whole model's tokens.
    let options = GenerateOptions {
        max_tokens: 8,
        ..GenerateOptions::default()
    };
    let tokens: Vec<u32> = (1..=10).collect();
    let want: Vec<u32> = generate(&whole, &tokens, options.clone())
        .unwrap()
        .map(|t| t.unwrap())
        .collect();
    let got: Vec<u32> = generate(&staged, &tokens, options)
        .unwrap()
        .map(|t| t.unwrap())
        .collect();
    assert_eq!(got, want, "{name}: generated tokens on rank {}", g.rank());

    // The Batcher: several requests (ending early, one cancelled) give, on every rank, each
    // request's batch-invariant events of the whole model (one micro-batch per stage in
    // flight interleaves the requests differently).
    let want = per_request(&serve(&whole));
    let got = per_request(&serve(&staged));
    assert_eq!(got, want, "{name}: Batcher events on rank {}", g.rank());
    println!("\nBITS {name} {got:?}");
}

/// Each request's events, in order.
fn per_request(events: &[Event]) -> Vec<Vec<Event>> {
    let mut out: Vec<Vec<Event>> = Vec::new();
    for e in events {
        let id = match e {
            Event::Token { id, .. } | Event::Finished { id, .. } => *id,
        };
        if out.len() <= id {
            out.resize(id + 1, Vec::new());
        }
        out[id].push(e.clone());
    }
    out
}

/// A Batcher's events for a fixed workload: four requests of different lengths and
/// budgets, a fifth submitted later, one cancelled.
fn serve(model: &Model) -> Vec<Event> {
    let mut b = Batcher::new(
        model,
        BatcherOptions {
            blocks: Some(32),
            block_size: 4,
            max_batch: 4,
            batch_invariant: true,
            ..Default::default()
        },
    )
    .unwrap();
    let greedy = |max_tokens| GenerateOptions {
        max_tokens,
        ..GenerateOptions::default()
    };
    let prompts: [&[u32]; 4] = [&[1, 2, 3], &[4, 5, 6, 7, 8], &[9, 10], &[11, 12, 13, 14]];
    let mut ids = Vec::new();
    for (p, max) in prompts.iter().zip([6, 3, 8, 5]) {
        ids.push(b.submit(p.to_vec(), greedy(max)).unwrap());
    }
    // Driven by tokens, not steps (one micro-batch per stage reads fewer tokens per step):
    // a fifth request after request 0's second token; request 3 cancelled after its third.
    let mut events: Vec<Event> = Vec::new();
    let count = |events: &[Event], who: usize| {
        events
            .iter()
            .filter(|e| matches!(e, Event::Token { id, .. } if *id == who))
            .count()
    };
    let (mut submitted, mut cancelled) = (false, false);
    for _ in 0..256 {
        if !submitted && count(&events, ids[0]) >= 2 {
            ids.push(b.submit(vec![15, 16, 17], greedy(4)).unwrap());
            submitted = true;
        }
        if !cancelled && count(&events, ids[3]) >= 3 {
            b.cancel(ids[3]);
            cancelled = true;
        }
        if !b.is_busy() {
            break;
        }
        events.extend(b.step().unwrap());
    }
    events
}

#[test]
fn pipelined_families_equal_the_whole_model() {
    for &(name, counts) in FAMILIES {
        for &n in counts {
            let test = "pipelined_families_equal_the_whole_model";
            if std::env::var("NUVEC_TEST_STAGE_CASE").is_ok_and(|c| c != format!("{name}/{n}")) {
                continue;
            }
            // Each case is its own set of child processes.
            let Some(out) = ranks_case(test, name, n) else {
                return;
            };
            agree(&out);
        }
    }
}

/// Runs case `name` on `n` ranks (the children run only that case).
fn ranks_case(test: &str, name: &str, n: usize) -> Option<Vec<String>> {
    if std::env::var(ranks::CHILD).is_ok() {
        if std::env::var("NUVEC_TEST_STAGE_CASE").ok().as_deref() == Some(&format!("{name}/{n}")) {
            ranks(test, n, |g| stages(g, name));
        }
        return None;
    }
    // The case reaches the children through their environment.
    ranks::ranks_with(
        test,
        n,
        &[("NUVEC_TEST_STAGE_CASE", &format!("{name}/{n}"))],
        |g| stages(g, name),
    )
}

#[test]
fn invalid_pipelines_are_errors() {
    // A group of one leaves the model whole.
    let dir = fixture("qwen3_f32");
    let g = Group::single();
    let m: Model = load_model_pipelined(&dir, &g, Split::Even).unwrap();
    assert!(m.stage().is_none());
    // Gemma 4 is not supported (mlx-lm does not pipeline it either): checked on a
    // multi-rank group only, by the stage planner's unit test.
}

/// Sampling, the memory split and failures on a 2-stage pipeline of the hybrid fixture:
/// a seeded sampled generation yields the whole model's tokens on every rank (only rank 0
/// samples); explicit budgets give a contiguous split; budgets too small fail on every rank
/// before loading, naming the bytes; a rank that leaves mid-generation fails the other.
fn sampling_split_and_failure(g: Group) {
    set_default_stream(Stream::default_for(Device::cpu()));
    let dir = fixture("qwen3_5_f32");
    let whole = load_model(&dir).unwrap();
    let sampled = GenerateOptions {
        max_tokens: 8,
        seed: Some(7),
        sampler: nuvec_lm::Sampler {
            temperature: 0.8,
            ..nuvec_lm::Sampler::default()
        },
        ..GenerateOptions::default()
    };
    let tokens: Vec<u32> = (1..=6).collect();
    let want: Vec<u32> = generate(&whole, &tokens, sampled.clone())
        .unwrap()
        .map(|t| t.unwrap())
        .collect();

    // Budgets: rank 1 (the first stage) offers three times rank 0's room.
    let total: u64 = whole
        .parameters()
        .iter()
        .map(|(_, a)| a.size() as u64 * a.dtype().size() as u64)
        .sum();
    let budgets = vec![total * 6 / 10, total * 18 / 10];
    let staged = load_model_pipelined(&dir, &g, Split::Budgets(budgets, 0.0)).unwrap();
    let stage = staged.stage().unwrap().clone();
    assert!(!stage.layers().is_empty());
    let got: Vec<u32> = generate(&staged, &tokens, sampled.clone())
        .unwrap()
        .map(|t| t.unwrap())
        .collect();
    assert_eq!(got, want, "sampled tokens on rank {}", g.rank());

    // Speculative decoding is not supported on a stage.
    let e = nuvec_lm::speculative_generate(
        &staged,
        &tokens,
        GenerateOptions::default(),
        nuvec_lm::PromptLookup::default(),
        nuvec_lm::DraftPolicy::default(),
    )
    .err()
    .unwrap()
    .to_string();
    assert!(e.contains("pipelined"), "{e}");

    // The memory split from the devices (one Mac: equal budgets) loads too.
    let by_memory = load_model_pipelined(&dir, &g, Split::memory()).unwrap();
    assert!(!by_memory.stage().unwrap().layers().is_empty());

    // Too small: every rank fails before loading, naming the bytes.
    let e = load_model_pipelined(&dir, &g, Split::Budgets(vec![total / 10, total / 10], 0.0))
        .unwrap_err()
        .to_string();
    assert!(e.contains("budgets hold") && e.contains("bytes"), "{e}");

    // Rank 1 leaves after the prompt: rank 0's generation fails, naming it.
    if g.rank() == 1 {
        let mut tokens_out = generate(&staged, &tokens, sampled).unwrap();
        let _ = tokens_out.next();
        println!("\nBITS done");
        std::process::exit(0);
    }
    let start = std::time::Instant::now();
    let result: nuvec::Result<Vec<u32>> =
        generate(&staged, &tokens, sampled).and_then(|out| out.collect());
    let e = result.unwrap_err().to_string();
    assert!(e.contains("rank 1"), "{e}");
    assert!(
        start.elapsed() < std::time::Duration::from_secs(25),
        "{:?}",
        start.elapsed()
    );
    println!("\nBITS done");
}

#[test]
fn pipelines_sample_split_and_fail() {
    if let Some(out) = ranks(
        "pipelines_sample_split_and_fail",
        2,
        sampling_split_and_failure,
    ) {
        agree(&out);
    }
}
