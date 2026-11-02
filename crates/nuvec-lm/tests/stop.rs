//! Stops, `ignore_eos` and `min_tokens` end generation alike in `generate`, the `Batcher`
//! and `speculative_generate` (greedy, tiny float32 fixtures).

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use nuvec::{Array, Device, Stream, set_default_stream};
use nuvec_lm::{
    Batcher, BatcherOptions, DraftPolicy, Event, FinishReason, GenerateOptions, Model,
    PromptLookup, Stop, Stopped, Tokenizer, generate, load_model, speculative_generate,
};

const FIXTURES: [&str; 5] = [
    "qwen3_f32",
    "qwen3_5_f32",
    "qwen3_moe_f32",
    "gemma4_e_f32",
    "gpt_oss_f32",
];

fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = (&Array::from(1.0f32) + 1.0).eval().is_ok();
    if !ok {
        eprintln!("skipped: no Metal device");
    }
    ok
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture(name: &str) -> (Model, Vec<u32>) {
    let dir = fixtures().join(name);
    let expected = nuvec::io::load_safetensors(dir.join("expected.safetensors"))
        .unwrap()
        .tensors;
    (
        load_model(&dir).unwrap(),
        expected["prompt"].to_vec::<u32>().unwrap(),
    )
}

/// A byte-level tokenizer over the fixtures' small vocabularies: id `i` is the `i`-th of
/// GPT-2's 256 byte characters, so every token is one byte.
fn tokenizer() -> Tokenizer {
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    let mut extra = 0;
    let chars: Vec<char> = (0..256u32)
        .map(|b| {
            if printable(b) {
                char::from_u32(b).unwrap()
            } else {
                extra += 1;
                char::from_u32(255 + extra).unwrap()
            }
        })
        .collect();
    let vocab: Vec<String> = chars
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let c = match c {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                c => c.to_string(),
            };
            format!("\"{c}\":{i}")
        })
        .collect();
    let json = format!(
        r#"{{"pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}},
          "model":{{"type":"BPE","vocab":{{{}}},"merges":[]}}}}"#,
        vocab.join(",")
    );
    // One file per call: the tests run in parallel.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = format!("nuvec-stop-{}-{n}.json", std::process::id());
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, json).unwrap();
    let t = Tokenizer::from_file(&path).unwrap();
    let _ = std::fs::remove_file(path);
    t
}

#[derive(Debug, Clone, PartialEq)]
struct Run {
    tokens: Vec<u32>,
    reason: FinishReason,
}

fn solo(model: &Model, prompt: &[u32], options: &GenerateOptions) -> Run {
    let mut g = generate(model, prompt, options.clone()).unwrap();
    let tokens = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
    Run {
        tokens,
        reason: g.finish_reason().cloned().expect("finished"),
    }
}

fn speculative(model: &Model, prompt: &[u32], options: &GenerateOptions) -> Run {
    let mut g = speculative_generate(
        model,
        prompt,
        options.clone(),
        PromptLookup::default(),
        DraftPolicy::Fixed(4),
    )
    .unwrap();
    let tokens = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
    Run {
        tokens,
        reason: g.finish_reason().cloned().expect("finished"),
    }
}

fn batched(
    model: &Model,
    requests: &[(Vec<u32>, GenerateOptions)],
    blocks: usize,
) -> (Vec<Run>, usize) {
    let mut b = Batcher::new(
        model,
        BatcherOptions {
            blocks: Some(blocks),
            block_size: 4,
            max_batch: requests.len(),
            batch_invariant: true,
            ..Default::default()
        },
    )
    .unwrap();
    for (prompt, options) in requests {
        b.submit(prompt.clone(), options.clone()).unwrap();
    }
    let mut tokens = vec![Vec::new(); requests.len()];
    let mut reasons = vec![None; requests.len()];
    while b.is_busy() {
        for e in b.step().unwrap() {
            match e {
                Event::Token { id, token } => tokens[id].push(token),
                Event::Finished { id, reason } => reasons[id] = Some(reason),
            }
        }
    }
    let runs = tokens
        .into_iter()
        .zip(reasons)
        .map(|(tokens, r)| Run {
            tokens,
            reason: r.expect("finished"),
        })
        .collect();
    (runs, b.preemptions())
}

/// The run of every path, checked equal.
fn all_paths(name: &str, model: &Model, prompt: &[u32], options: &GenerateOptions) -> Run {
    let alone = solo(model, prompt, options);
    let spec = speculative(model, prompt, options);
    assert_eq!(spec, alone, "{name}: speculative");
    let (batch, _) = batched(model, &[(prompt.to_vec(), options.clone())], 256);
    assert_eq!(batch[0], alone, "{name}: batcher");
    alone
}

/// The first stop string in `tokens`' text: the token completing it and the cut.
fn reference(t: &Tokenizer, tokens: &[u32], stop: &str) -> Option<(usize, usize)> {
    (0..tokens.len()).find_map(|k| {
        let text = t.decode_bytes(&tokens[..=k]);
        let at = text
            .windows(stop.len())
            .position(|w| w == stop.as_bytes())?;
        Some((k, text.len() - at))
    })
}

/// A stop string made of the text of tokens `i..i + 2` of `tokens`, for the first `i ≥ 2`
/// where it is valid UTF-8 and not blank.
fn stop_from(t: &Tokenizer, tokens: &[u32]) -> String {
    let len = |k: usize| t.decode_bytes(&tokens[..k]).len();
    let full = t.decode_bytes(tokens);
    (2..tokens.len() - 2)
        .find_map(|i| {
            let s = std::str::from_utf8(&full[len(i)..len(i + 2)]).ok()?;
            (!s.trim().is_empty()).then(|| s.to_string())
        })
        .expect("a printable stretch of output")
}

fn plain(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_tokens,
        ..Default::default()
    }
}

#[test]
fn stop_strings_end_every_path_at_the_same_token() {
    if !gpu() {
        return;
    }
    let t = tokenizer();
    for name in FIXTURES {
        let (model, prompt) = fixture(name);
        let free = all_paths(name, &model, &prompt, &plain(24));
        let stop = stop_from(&t, &free.tokens);
        let (k, cut) = reference(&t, &free.tokens, &stop).unwrap();
        // The real stop at index 1; index 0 never matches.
        let options = GenerateOptions {
            stop: Stop::strings(&t, ["\u{1F600} never", &stop]).unwrap(),
            ..plain(24)
        };
        let run = all_paths(name, &model, &prompt, &options);
        assert_eq!(run.tokens, free.tokens[..=k], "{name}: {stop:?}");
        assert_eq!(
            run.reason,
            FinishReason::Stop(Stopped::String { index: 1, cut }),
            "{name}"
        );
        let text = t.decode_bytes(&run.tokens);
        assert!(
            !text[..text.len() - cut]
                .windows(stop.len())
                .any(|w| w == stop.as_bytes())
        );
        assert!(
            text[text.len() - cut..].starts_with(stop.as_bytes()),
            "{name}"
        );
    }
}

#[test]
fn stop_tokens_eos_ignore_eos_and_min_tokens() {
    if !gpu() {
        return;
    }
    for name in FIXTURES {
        let (mut model, prompt) = fixture(name);
        let free = all_paths(name, &model, &prompt, &plain(16));
        let token = free.tokens[3];
        let j = free.tokens.iter().position(|&x| x == token).unwrap();

        // A stop token: not emitted, the run ends before it.
        let options = GenerateOptions {
            stop: Stop::tokens([token]),
            ..plain(16)
        };
        let run = all_paths(name, &model, &prompt, &options);
        assert_eq!(run.tokens, free.tokens[..j], "{name}");
        assert_eq!(run.reason, FinishReason::Stop(Stopped::Token(token)));

        // The same token as end-of-sequence, then ignored.
        model.config.eos_token_ids = vec![token];
        let run = all_paths(name, &model, &prompt, &plain(16));
        assert_eq!(run.tokens, free.tokens[..j], "{name}");
        assert_eq!(run.reason, FinishReason::EndOfSequence);
        let options = GenerateOptions {
            ignore_eos: true,
            ..plain(16)
        };
        let run = all_paths(name, &model, &prompt, &options);
        assert_eq!(
            run,
            Run {
                tokens: free.tokens.clone(),
                reason: FinishReason::Length
            },
            "{name}"
        );

        // `min_tokens`: the end-of-sequence token cannot come among the first ones.
        let min = j + 2;
        let options = GenerateOptions {
            min_tokens: min,
            ..plain(16)
        };
        let run = all_paths(name, &model, &prompt, &options);
        assert!(run.tokens.len() >= min, "{name}: {run:?}");
        assert!(!run.tokens[..min].contains(&token), "{name}: {run:?}");
        assert_eq!(run.tokens[..j], free.tokens[..j], "{name}");
        // Out of range.
        let options = GenerateOptions {
            min_tokens: 17,
            ..plain(16)
        };
        assert!(generate(&model, &prompt, options).is_err());
    }
}

/// Eight requests with different endings, batched: each as alone (SC-002); and with a
/// cache small enough to preempt, every request still ends at its own first stop.
#[test]
fn mixed_requests_end_as_alone_and_across_preemption() {
    if !gpu() {
        return;
    }
    let t = tokenizer();
    let (mut model, prompt) = fixture("qwen3_f32");
    let free = solo(&model, &prompt, &plain(24));
    let stop = stop_from(&t, &free.tokens);
    model.config.eos_token_ids = vec![free.tokens[5]];
    let options = [
        plain(24),
        GenerateOptions {
            stop: Stop::strings(&t, [&stop]).unwrap(),
            ..plain(24)
        },
        GenerateOptions {
            stop: Stop::tokens([free.tokens[2]]),
            ..plain(24)
        },
        GenerateOptions {
            ignore_eos: true,
            ..plain(20)
        },
        GenerateOptions {
            min_tokens: 8,
            ..plain(24)
        },
        GenerateOptions {
            ignore_eos: true,
            stop: Stop::strings(&t, [&stop])
                .unwrap()
                .with_tokens([free.tokens[9]]),
            ..plain(24)
        },
        plain(3),
        GenerateOptions {
            min_tokens: 4,
            stop: Stop::strings(&t, [&stop]).unwrap(),
            ..plain(24)
        },
    ];
    let prompts: Vec<Vec<u32>> = (0..8)
        .map(|i| prompt[..prompt.len() - i % 3].to_vec())
        .collect();
    let requests: Vec<(Vec<u32>, GenerateOptions)> = prompts
        .iter()
        .cloned()
        .zip(options.iter().cloned())
        .collect();
    let (runs, _) = batched(&model, &requests, 256);
    for (i, (run, (p, o))) in runs.iter().zip(&requests).enumerate() {
        assert_eq!(*run, solo(&model, p, o), "request {i}");
    }
    // Preemption: a stop string spanning it is still found at the first occurrence in
    // each request's own output.
    let stopping: Vec<(Vec<u32>, GenerateOptions)> = (0..4)
        .map(|i| (prompts[i].clone(), options[1].clone()))
        .collect();
    let (runs, preemptions) = batched(&model, &stopping, 12);
    assert!(preemptions > 0, "no preemption");
    for (i, run) in runs.iter().enumerate() {
        match reference(&t, &run.tokens, &stop) {
            Some((k, cut)) => {
                assert_eq!(k + 1, run.tokens.len(), "request {i}");
                assert_eq!(
                    run.reason,
                    FinishReason::Stop(Stopped::String { index: 0, cut })
                );
            }
            None => assert!(
                matches!(
                    run.reason,
                    FinishReason::EndOfSequence | FinishReason::Length
                ),
                "request {i}: {run:?}"
            ),
        }
    }
}
