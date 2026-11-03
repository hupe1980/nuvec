//! Tensor parallelism across processes: every family's tiny fixture sharded
//! over 2 and 4 ranks gives the whole model's logits within the reduction order, the whole
//! model's greedy tokens on the float32 fixtures, and the same tokens on every rank. Sizes
//! that do not split are errors that leave the model whole.

use std::path::PathBuf;

use nuvec::distributed::Group;
use nuvec::{Array, Device, Stream, set_default_stream};
use nuvec_lm::{Batcher, BatcherOptions, Event, GenerateOptions, Model, generate, load_model};
use nuvec_nn::Module;

#[path = "common/ranks.rs"]
mod ranks;

use ranks::{agree, ranks, ranks_with};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// `(fixture, ranks, splits)`: whether its sizes split across that many ranks (quantized
/// columns split in whole groups of 32, which the quantized MoE fixtures' 32 expert units
/// are; the hybrids have 2 key heads).
const CASES: &[(&str, usize, bool)] = &[
    ("llama_bf16", 2, true),
    ("llama_bf16", 4, true),
    ("qwen2_q4", 2, true),
    ("qwen2_q4", 4, false),
    ("qwen3_f32", 2, true),
    ("qwen3_f32", 4, true),
    ("qwen3_mxfp4", 2, true),
    ("qwen3_5_f32", 2, true),
    ("qwen3_5_f32", 4, false),
    ("qwen3_5_moe_q4", 2, false),
    ("qwen3_5_moe_f32", 2, true),
    ("qwen3_moe_q4", 2, false),
    ("qwen3_moe_f32", 2, true),
    ("gpt_oss_f32", 2, true),
    ("gpt_oss_f32", 4, true),
];

fn prompt() -> Array {
    Array::from_slice(&[1u32, 5, 9, 13, 17, 21, 25, 29, 2, 4], [1, 10])
}

fn values(a: &Array) -> Vec<f32> {
    a.astype(nuvec::DType::F32).to_vec::<f32>().unwrap()
}

/// The largest difference relative to the largest magnitude.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
        / scale
}

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

fn serve(model: &Model) -> Vec<Vec<Event>> {
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
    for (p, max) in [(&[1u32, 2, 3][..], 6), (&[4, 5, 6, 7, 8], 3), (&[9, 10], 5)] {
        let o = GenerateOptions {
            max_tokens: max,
            ..GenerateOptions::default()
        };
        b.submit(p.to_vec(), o).unwrap();
    }
    let mut events = Vec::new();
    while b.is_busy() {
        events.extend(b.step().unwrap());
    }
    per_request(&events)
}

fn case(g: Group, name: &str, splits: bool) {
    set_default_stream(Stream::default_for(Device::cpu()));
    let dir = fixture(name);
    let whole = load_model(&dir).unwrap();
    let mut sharded = load_model(&dir).unwrap();
    let result = sharded.shard(&g);
    if !splits {
        let e = result.unwrap_err().to_string();
        assert!(
            e.contains("do not split") || e.contains("do not repeat"),
            "{name}: {e}"
        );
        // Atomic: the model is still whole and unsharded.
        assert!(sharded.sharding().is_none());
        let (a, b) = (
            values(&whole.forward(&prompt(), &mut whole.new_cache())),
            values(&sharded.forward(&prompt(), &mut sharded.new_cache())),
        );
        assert_eq!(a, b, "{name}: unchanged after a failed shard");
        println!("\nBITS {name} refused");
        return;
    }
    result.unwrap();
    // Each sharded projection holds 1/N of its whole bytes.
    let size = |m: &Model, path: &str| {
        m.parameters()
            .into_iter()
            .find(|(p, _)| p == path)
            .map(|(_, a)| a.size())
    };
    if let (Some(w), Some(s)) = (
        size(&whole, "model.layers.0.mlp.down_proj.weight"),
        size(&sharded, "model.layers.0.mlp.down_proj.weight"),
    ) {
        assert_eq!(s * g.size(), w, "{name}: down_proj slice");
    }
    let f32_fixture = name.ends_with("_f32");
    // Half precision: every rank rounds its partial sums before the group adds them.
    let tol = if f32_fixture { 1e-5 } else { 5e-2 };
    let (mut wc, mut sc) = (whole.new_cache(), sharded.new_cache());
    let want = values(&whole.forward(&prompt(), &mut wc));
    let got = values(&sharded.forward(&prompt(), &mut sc));
    let d = rel(&got, &want);
    assert!(
        d <= tol,
        "{name}: prefill logits differ by {d} (rank {})",
        g.rank()
    );
    let next = Array::from_slice(&[7u32], [1, 1]);
    let d = rel(
        &values(&sharded.forward(&next, &mut sc)),
        &values(&whole.forward(&next, &mut wc)),
    );
    assert!(d <= tol, "{name}: decode logits differ by {d}");

    let options = GenerateOptions {
        max_tokens: 8,
        ..GenerateOptions::default()
    };
    let tokens: Vec<u32> = (1..=10).collect();
    let got: Vec<u32> = generate(&sharded, &tokens, options.clone())
        .unwrap()
        .map(|t| t.unwrap())
        .collect();
    let served = serve(&sharded);
    if f32_fixture {
        let want: Vec<u32> = generate(&whole, &tokens, options)
            .unwrap()
            .map(|t| t.unwrap())
            .collect();
        assert_eq!(got, want, "{name}: greedy tokens");
        assert_eq!(served, serve(&whole), "{name}: Batcher events");
    }
    println!("\nBITS {name} {got:?} {served:?}");
}

#[test]
fn sharded_families_match_the_whole_model() {
    let test = "sharded_families_match_the_whole_model";
    for &(name, n, splits) in CASES {
        let key = format!("{name}/{n}");
        if std::env::var(ranks::CHILD).is_ok() {
            if std::env::var("NUVEC_TEST_TP_CASE").ok().as_deref() == Some(key.as_str()) {
                ranks(test, n, |g| case(g, name, splits));
            }
            continue;
        }
        if let Some(out) = ranks_with(test, n, &[("NUVEC_TEST_TP_CASE", &key)], |g| {
            case(g, name, splits)
        }) {
            agree(&out);
        }
    }
}

/// Seeded sampling and speculative decoding on a sharded model: every rank yields the
/// same tokens (rank 0 samples); a rank that leaves fails the other, naming it.
fn sampling_and_failure(g: Group) {
    set_default_stream(Stream::default_for(Device::cpu()));
    let dir = fixture("qwen3_f32");
    let whole = load_model(&dir).unwrap();
    let sharded = nuvec_lm::load_model_sharded(&dir, &g).unwrap();
    let options = GenerateOptions {
        max_tokens: 8,
        seed: Some(3),
        sampler: nuvec_lm::Sampler {
            temperature: 0.8,
            ..nuvec_lm::Sampler::default()
        },
        ..GenerateOptions::default()
    };
    let tokens: Vec<u32> = (1..=6).collect();
    let got: Vec<u32> = generate(&sharded, &tokens, options.clone())
        .unwrap()
        .map(|t| t.unwrap())
        .collect();
    let want: Vec<u32> = generate(&whole, &tokens, options.clone())
        .unwrap()
        .map(|t| t.unwrap())
        .collect();
    assert_eq!(got, want, "sampled tokens");
    let spec: Vec<u32> = nuvec_lm::speculative_generate(
        &sharded,
        &[1, 2, 3, 1, 2, 3, 1, 2],
        GenerateOptions {
            max_tokens: 8,
            ..GenerateOptions::default()
        },
        nuvec_lm::PromptLookup::default(),
        nuvec_lm::DraftPolicy::default(),
    )
    .unwrap()
    .map(|t| t.unwrap())
    .collect();
    println!("\nBITS {got:?} {spec:?}");
    if g.rank() == 1 {
        std::process::exit(0);
    }
    let e = generate(&sharded, &tokens, options)
        .and_then(|t| t.collect::<nuvec::Result<Vec<u32>>>())
        .unwrap_err()
        .to_string();
    assert!(e.contains("rank 1"), "{e}");
}

#[test]
fn sharded_models_sample_once_and_fail_cleanly() {
    if let Some(out) = ranks(
        "sharded_models_sample_once_and_fail_cleanly",
        2,
        sampling_and_failure,
    ) {
        agree(&out);
    }
}
