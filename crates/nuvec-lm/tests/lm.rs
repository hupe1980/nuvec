//! nuvec-lm against mlx-lm 0.32.2: tiny random models of each supported variant, recorded
//! with their logits and greedy continuations by `scripts/lm_fixtures.py`
//! (`just lm-fixtures`). Skipped without a Metal device.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use nuvec::{Array, DType, Device, Stream, set_default_stream};
use nuvec_lm::{GenerateOptions, generate, load_model};

fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = (&Array::scalar(1.0f32) + 1.0).eval().is_ok();
    if !ok {
        eprintln!("skipped: no Metal device");
    }
    ok
}

fn fixture(name: &str) -> (PathBuf, Vec<u32>, Vec<f32>, Vec<usize>, Vec<u32>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let expected = nuvec::io::load_safetensors(dir.join("expected.safetensors"))
        .unwrap()
        .tensors;
    let logits = &expected["logits"];
    (
        dir,
        expected["prompt"].to_vec::<u32>().unwrap(),
        logits.to_vec::<f32>().unwrap(),
        logits.shape().to_vec(),
        expected["greedy"].to_vec::<u32>().unwrap(),
    )
}

/// The model's logits equal mlx-lm's (within `tol` of the largest logit), and greedy
/// generation picks mlx-lm's tokens, with prompts prefilled at once or in chunks.
fn check(name: &str, tol: f32) {
    if !gpu() {
        return;
    }
    let (dir, prompt, logits, shape, greedy) = fixture(name);
    let model = load_model(&dir).unwrap();
    let tokens = Array::from_slice(&prompt, &[1, prompt.len()]).unwrap();
    let got = model
        .forward(&tokens, &mut model.new_cache())
        .unwrap()
        .astype(DType::F32)
        .unwrap();
    assert_eq!(got.shape(), shape, "{name}: logits shape");
    let scale = logits.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    for (i, (g, e)) in got.to_vec::<f32>().unwrap().iter().zip(&logits).enumerate() {
        assert!(
            (g - e).abs() <= tol * scale,
            "{name}: logit {i}: {g} vs mlx-lm {e}"
        );
    }
    for prefill_step in [2048, 5] {
        let options = GenerateOptions {
            max_tokens: greedy.len(),
            prefill_step,
            ..Default::default()
        };
        let ids: Vec<u32> = generate(&model, &prompt, options)
            .unwrap()
            .collect::<nuvec::Result<_>>()
            .unwrap();
        assert_eq!(
            ids, greedy,
            "{name}: greedy tokens (prefill step {prefill_step})"
        );
    }
}

#[test]
fn qwen3_matches_mlx_lm() {
    check("qwen3_f32", 1e-4);
}

#[test]
fn llama3_with_rope_scaling_and_tied_embeddings_matches_mlx_lm() {
    check("llama_bf16", 2e-2);
}

#[test]
fn quantized_qwen2_with_biases_matches_mlx_lm() {
    check("qwen2_q4", 1e-2);
}

#[test]
fn the_cache_grows_past_its_first_capacity() {
    if !gpu() {
        return;
    }
    // 300 positions cross the 256-position step; a fresh prefill of the same tokens
    // agrees with the step-by-step cache.
    let (dir, prompt, ..) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let long: Vec<u32> = prompt.iter().cycle().take(300).copied().collect();
    let options = GenerateOptions {
        max_tokens: 4,
        prefill_step: 100,
        ..Default::default()
    };
    let chunked: Vec<u32> = generate(&model, &long, options)
        .unwrap()
        .collect::<nuvec::Result<_>>()
        .unwrap();
    let at_once: Vec<u32> = generate(
        &model,
        &long,
        GenerateOptions {
            prefill_step: 4096,
            ..options
        },
    )
    .unwrap()
    .collect::<nuvec::Result<_>>()
    .unwrap();
    assert_eq!(chunked, at_once);
}

#[test]
fn qwen3_tokenizer_matches_hugging_face() {
    // Ids recorded with `transformers` for a text with NFC-sensitive letters, combining
    // marks, emoji, CJK, added and special tokens, tabs and newline runs. Needs the
    // model's tokenizer.json in the local Hugging Face cache.
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3_tokenizer.json");
    let fixture = nuvec::io::json::parse(&std::fs::read(path).unwrap()).unwrap();
    let model = fixture.get("model").and_then(|m| m.as_str()).unwrap();
    let Ok(dir) = nuvec_lm::resolve(model) else {
        eprintln!("skipped: {model} is not in the Hugging Face cache");
        return;
    };
    let tokenizer = nuvec_lm::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let text = fixture.get("text").and_then(|t| t.as_str()).unwrap();
    let expected: Vec<u32> = fixture
        .get("ids")
        .and_then(|i| i.as_array())
        .unwrap()
        .iter()
        .map(|i| i.as_i64().unwrap() as u32)
        .collect();
    let ids = tokenizer.encode(text);
    assert_eq!(ids, expected);
    assert_eq!(tokenizer.decode(&ids), text);
}
