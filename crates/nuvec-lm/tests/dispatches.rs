//! GPU dispatches of paged serving: moving prefilled sequences into the paged cache, and
//! one decode step. One test, so no other evaluation encodes while it counts; skipped
//! without a Metal device.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use nuvec::{Array, Device, Stream, set_default_stream};
use nuvec_lm::{PagedCache, decode_batch, load_model};

/// The dispatches `f` encodes.
fn dispatches<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = nuvec::__gpu_dispatches();
    let out = f();
    (out, nuvec::__gpu_dispatches() - before)
}

#[test]
fn paged_serving_dispatches() {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    if (&Array::from(1.0f32) + 1.0).eval().is_err() {
        eprintln!("skipped: no Metal device");
        return;
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3_f32");
    let model = load_model(&dir).unwrap();
    let layers = model.config.num_hidden_layers as u64;
    let (rows, n) = (8usize, 6usize);
    let tokens: Vec<u32> = (0..(rows * n) as u32).map(|t| t % 100 + 1).collect();
    let prefill = || {
        let mut kv = model.new_cache();
        let ids = Array::from_slice(&tokens, [rows as i32, n as i32]);
        model.forward(&ids, &mut kv).eval().unwrap();
        kv
    };
    let mut cache = PagedCache::new(&model, 64, 4, rows).unwrap();
    let kv = prefill();
    // The pools are allocated (zero-filled) by the first insert.
    for id in cache.insert_many(&kv).unwrap() {
        cache.remove(id).unwrap();
    }
    nuvec::eval(&cache.state()).unwrap();
    let (ids, inserted) = dispatches(|| {
        let ids = cache.insert_many(&kv).unwrap();
        nuvec::eval(&cache.state()).unwrap();
        ids
    });
    // One write of every row's keys and one of their values per layer.
    assert_eq!(inserted, 2 * layers, "insert_many");
    let next: Vec<u32> = (0..rows as u32).map(|r| r + 3).collect();
    let ((), step) = dispatches(|| {
        let logits = decode_batch(&model, &next, &ids, &mut cache).unwrap();
        nuvec::eval(&[&[logits][..], &cache.state()].concat()).unwrap();
    });
    eprintln!("decode step over {rows} sequences: {step} dispatches ({layers} layers)");
    // A hybrid (Gated DeltaNet and attention layers): one decode step.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3_5_f32");
    let model = load_model(&dir).unwrap();
    let mut kv = model.new_cache();
    let ids = Array::from_slice(&tokens[..rows * n], [rows as i32, n as i32]);
    model.forward(&ids, &mut kv).eval().unwrap();
    let mut cache = PagedCache::new(&model, 64, 4, rows).unwrap();
    let ids = cache.insert_many(&kv).unwrap();
    nuvec::eval(&cache.state()).unwrap();
    // The first step allocates nothing new; the second is counted.
    decode_batch(&model, &next, &ids, &mut cache)
        .unwrap()
        .eval()
        .unwrap();
    nuvec::eval(&cache.state()).unwrap();
    let ((), step) = dispatches(|| {
        let logits = decode_batch(&model, &next, &ids, &mut cache).unwrap();
        nuvec::eval(&[&[logits][..], &cache.state()].concat()).unwrap();
    });
    let layers = model.config.num_hidden_layers;
    eprintln!("hybrid decode step over {rows} sequences: {step} dispatches ({layers} layers)");
    // Gemma 4 (per-layer inputs and KV sharing; experts; head 512): one decode token, its
    // residual chains one kernel each.
    for (name, at_most) in [
        ("gemma4_e_bf16", 143),
        ("gemma4_moe_q4", 112),
        ("gemma4_d512_bf16", 64),
    ] {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        let model = load_model(&dir).unwrap();
        let mut kv = model.new_cache();
        let prompt = Array::from_slice(&tokens[..8], [1, 8]);
        model.forward(&prompt, &mut kv).eval().unwrap();
        let token = Array::from_slice(&[7u32], [1, 1]);
        model.forward(&token, &mut kv).eval().unwrap();
        let ((), step) = dispatches(|| model.forward(&token, &mut kv).eval().unwrap());
        let layers = model.config.num_hidden_layers;
        eprintln!("{name} decode token: {step} dispatches ({layers} layers)");
        assert!(
            step <= at_most,
            "{name}: {step} dispatches, at most {at_most}"
        );
    }
}
