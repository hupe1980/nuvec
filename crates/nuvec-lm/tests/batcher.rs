//! The serving batcher never loses, starves or corrupts a request: one-token
//! prompts, prompts that can never fit, requests that outgrow the cache, cancellation, and
//! the paged cache's edge cases. On the tiny fixture models of `tests/lm.rs`; skipped
//! without a Metal device.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use nuvec::{Array, Device, Stream, set_default_stream};
use nuvec_lm::{
    Batcher, Event, FinishReason, GenerateOptions, Model, PagedCache, decode_batch, generate,
    load_model,
};

fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = (&Array::from(1.0f32) + 1.0).eval().is_ok();
    if !ok {
        eprintln!("skipped: no Metal device");
    }
    ok
}

/// A fixture's model and its prompt.
fn fixture(name: &str) -> (Model, Vec<u32>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let expected = nuvec::io::load_safetensors(dir.join("expected.safetensors"))
        .unwrap()
        .tensors;
    (
        load_model(&dir).unwrap(),
        expected["prompt"].to_vec::<u32>().unwrap(),
    )
}

fn greedy(max_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_tokens,
        ..Default::default()
    }
}

/// `generate`'s tokens alone.
fn solo(model: &Model, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
    generate(model, prompt, greedy(max_tokens))
        .unwrap()
        .collect::<nuvec::Result<_>>()
        .unwrap()
}

/// Steps until the batcher is idle (at most `limit` steps, a watchdog): each request's
/// tokens and finish reason.
fn run(
    batcher: &mut Batcher<'_>,
    requests: usize,
    limit: usize,
) -> (Vec<Vec<u32>>, Vec<Option<FinishReason>>) {
    let mut tokens = vec![Vec::new(); requests];
    let mut finished = vec![None; requests];
    for _ in 0..limit {
        if !batcher.is_busy() {
            return (tokens, finished);
        }
        for event in batcher.step().unwrap() {
            match event {
                Event::Token { id, token } => {
                    assert!(
                        finished[id].is_none(),
                        "request {id}: a token after finishing"
                    );
                    tokens[id].push(token);
                }
                Event::Finished { id, reason } => {
                    assert!(finished[id].is_none(), "request {id} finished twice");
                    finished[id] = Some(reason);
                }
            }
        }
    }
    panic!("the batcher is still busy after {limit} steps");
}

/// A one-token prompt yields `generate`'s tokens (an empty sequence in the paged cache,
/// also as the first sequence of a hybrid's state pools); an empty prompt is rejected;
/// `max_tokens = 0` finishes with `Length` and no token.
#[test]
fn one_token_prompts_are_served() {
    if !gpu() {
        return;
    }
    for name in ["qwen3_f32", "qwen3_5_f32"] {
        let (model, prompt) = fixture(name);
        let mut batcher = Batcher::new(&model, 32, 4, 3).unwrap();
        assert!(batcher.submit(Vec::new(), greedy(4)).is_err(), "{name}");
        let requests = [
            (vec![prompt[0]], 8),
            (prompt.clone(), 6),
            (vec![prompt[3]], 0),
        ];
        for (p, max) in &requests {
            batcher.submit(p.clone(), greedy(*max)).unwrap();
        }
        let (tokens, finished) = run(&mut batcher, requests.len(), 100);
        for (i, (p, max)) in requests.iter().enumerate() {
            assert_eq!(tokens[i], solo(&model, p, *max), "{name}: request {i}");
            assert_eq!(
                finished[i],
                Some(FinishReason::Length),
                "{name}: request {i}"
            );
        }
    }
}

/// A prompt that cannot fit in an empty cache is rejected by `submit` (the requests behind
/// it are served), and a lone request that outgrows the cache finishes with `CacheFull`

#[test]
fn unfittable_prompts_are_rejected_and_outgrown_requests_finish() {
    if !gpu() {
        return;
    }
    for name in ["qwen3_f32", "qwen3_5_f32"] {
        let (model, prompt) = fixture(name);
        // 4 blocks of 4 positions: 16 positions in all.
        let mut batcher = Batcher::new(&model, 4, 4, 2).unwrap();
        let long: Vec<u32> = prompt.iter().cycle().take(16).copied().collect();
        assert!(
            batcher.submit(long, greedy(1)).is_err(),
            "{name}: never fits"
        );
        batcher.submit(prompt[..3].to_vec(), greedy(4)).unwrap();
        let (tokens, finished) = run(&mut batcher, 1, 100);
        assert_eq!(tokens[0], solo(&model, &prompt[..3], 4), "{name}");
        assert_eq!(finished[0], Some(FinishReason::Length), "{name}");
        // Alone, 3 + 40 tokens do not fit in 16 positions.
        let mut batcher = Batcher::new(&model, 4, 4, 2).unwrap();
        batcher.submit(prompt[..3].to_vec(), greedy(40)).unwrap();
        let (tokens, finished) = run(&mut batcher, 1, 100);
        assert_eq!(finished[0], Some(FinishReason::CacheFull), "{name}");
        let want = solo(&model, &prompt[..3], 40);
        assert!(
            !tokens[0].is_empty() && want.starts_with(&tokens[0]),
            "{name}: {:?} not a prefix of {want:?}",
            tokens[0]
        );
        assert!(!batcher.is_busy());
    }
}

/// `cancel` finishes a waiting and a running request with `Cancelled` at the next step,
/// returns their blocks and leaves the others alone; an unknown or finished request is
/// not cancelled.
#[test]
fn cancelled_requests_finish_and_free_their_blocks() {
    if !gpu() {
        return;
    }
    let (model, prompt) = fixture("qwen3_f32");
    // 8 blocks of 4 positions, two requests at a time.
    let mut batcher = Batcher::new(&model, 8, 4, 2).unwrap();
    let running = batcher.submit(prompt[..5].to_vec(), greedy(10)).unwrap();
    let kept = batcher.submit(prompt[2..6].to_vec(), greedy(6)).unwrap();
    let waiting = batcher.submit(prompt[1..4].to_vec(), greedy(4)).unwrap();
    let mut tokens = vec![Vec::new(); 4];
    for event in batcher.step().unwrap() {
        if let Event::Token { id, token } = event {
            tokens[id].push(token);
        }
    }
    assert_eq!(batcher.running(), 2);
    assert!(batcher.cancel(running));
    assert!(batcher.cancel(waiting));
    assert!(!batcher.cancel(running), "already cancelled");
    assert!(!batcher.cancel(99), "unknown");
    let events = batcher.step().unwrap();
    for id in [running, waiting] {
        assert!(
            events.contains(&Event::Finished {
                id,
                reason: FinishReason::Cancelled
            }),
            "{events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::Token { id: t, .. } if *t == id)),
            "no token for a cancelled request"
        );
    }
    for event in events {
        if let Event::Token { id, token } = event {
            tokens[id].push(token);
        }
    }
    // Its blocks are back: a request needing almost the whole cache is admitted next to
    // the remaining one.
    let big = batcher
        .submit(prompt.iter().cycle().take(20).copied().collect(), greedy(2))
        .unwrap();
    let (rest, finished) = run(&mut batcher, 4, 100);
    tokens[kept].extend(&rest[kept]);
    assert_eq!(
        tokens[kept],
        solo(&model, &prompt[2..6], 6),
        "the other request"
    );
    assert_eq!(finished[kept], Some(FinishReason::Length));
    assert_eq!(finished[big], Some(FinishReason::Length));
    assert!(!batcher.cancel(kept), "finished");
}

/// The paged cache: a length-0 sequence (decoded like a one-token prompt), a double
/// `remove`, and the blocks returned when an `insert` fails while writing.
#[test]
fn paged_cache_edge_cases() {
    if !gpu() {
        return;
    }
    let (model, prompt) = fixture("qwen3_f32");
    let mut cache = PagedCache::new(&model, 8, 4, 2).unwrap();
    let empty = cache.insert(&model.new_cache()).unwrap();
    assert_eq!((cache.len(empty), cache.free_blocks()), (Some(0), 8));
    let logits = decode_batch(&model, &prompt[..1], &[empty], &mut cache).unwrap();
    let first = logits.argmax(-1).to_vec::<u32>().unwrap();
    assert_eq!(first, solo(&model, &prompt[..1], 1));
    assert_eq!((cache.len(empty), cache.free_blocks()), (Some(1), 7));
    cache.remove(empty).unwrap();
    assert!(cache.remove(empty).is_err(), "removed twice");
    assert_eq!(cache.free_blocks(), 8);
    // A hybrid's sequence, then one of another hybrid with wider Gated DeltaNet states
    // (as many layers): writing its states fails after its blocks were taken. They are
    // returned, and the first sequence decodes as before.
    let (model, prompt) = fixture("qwen3_5_f32");
    let (other, _) = fixture("qwen3_5_q4");
    let prefilled = |m: &Model, n: usize| {
        let mut kv = m.new_cache();
        let tokens = Array::from_slice(&prompt[..n], [1, n]);
        let _ = m.forward(&tokens, &mut kv).check().unwrap();
        kv
    };
    let mut cache = PagedCache::new(&model, 8, 4, 2).unwrap();
    let first = cache.insert(&prefilled(&model, 5)).unwrap();
    let free = cache.free_blocks();
    assert!(cache.insert(&prefilled(&other, 9)).is_err());
    assert_eq!(cache.free_blocks(), free, "blocks returned");
    let logits = decode_batch(&model, &prompt[5..6], &[first], &mut cache).unwrap();
    let next = logits.argmax(-1).to_vec::<u32>().unwrap();
    assert_eq!(next, solo(&model, &prompt[..6], 1), "pools unchanged");
}

/// A request reaching an end-of-sequence token while its next step is already in flight
/// emits nothing after it, and the others go on with their solo tokens.
#[test]
fn a_request_ending_with_a_step_in_flight_emits_nothing_more() {
    if !gpu() {
        return;
    }
    for name in ["qwen3_f32", "qwen3_5_f32"] {
        let (mut model, prompt) = fixture(name);
        model.config.eos_token_ids.clear();
        let first = solo(&model, &prompt[..5], 8);
        // Its third token ends it (where it first appears).
        let eos = first[2];
        let end = first.iter().position(|&t| t == eos).expect("present");
        model.config.eos_token_ids = vec![eos];
        let mut batcher = Batcher::new(&model, 64, 4, 2).unwrap();
        batcher.submit(prompt[..5].to_vec(), greedy(8)).unwrap();
        batcher.submit(prompt[1..7].to_vec(), greedy(8)).unwrap();
        let (tokens, finished) = run(&mut batcher, 2, 100);
        assert_eq!(tokens[0], first[..end], "{name}");
        assert_eq!(finished[0], Some(FinishReason::EndOfSequence), "{name}");
        let other = solo(&model, &prompt[1..7], 8);
        assert_eq!(tokens[1], other, "{name}");
        // The blocks are back: a request needing the whole cache but one block runs.
        let long: Vec<u32> = prompt.iter().cycle().take(4 * 62).copied().collect();
        batcher.submit(long, greedy(1)).unwrap();
        let (_, finished) = run(&mut batcher, 3, 100);
        assert!(finished[2].is_some(), "{name}");
    }
}
