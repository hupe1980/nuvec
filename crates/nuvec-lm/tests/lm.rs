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

/// The prompt's logits from a fixture's model.
fn logits(name: &str) -> (Vec<f32>, Vec<f32>) {
    let (dir, prompt, expected, _, _) = fixture(name);
    let model = load_model(&dir).unwrap();
    let tokens = Array::from_slice(&prompt, &[1, prompt.len()]).unwrap();
    let got = model.forward(&tokens, &mut model.new_cache()).unwrap();
    (
        got.astype(DType::F32).unwrap().to_vec::<f32>().unwrap(),
        expected,
    )
}

#[test]
fn hybrid_qwen3_5_matches_mlx_lm() {
    check("qwen3_5_f32", 1e-4);
}

/// In half precision the hybrid's recurrence amplifies rounding differences: nuvec's
/// element-wise kernels round once in float32 where MLX's bfloat16 ones round after each
/// step, and MLX's convolution kernel sums in its own order. The logits stay close
/// (measured against the float32 computation, nuvec's error is MLX's), but greedy tokens
/// may part.
#[test]
fn half_precision_qwen3_5_stays_close_to_mlx_lm() {
    if !gpu() {
        return;
    }
    for name in [
        "qwen3_5_bf16",
        "qwen3_5_q4",
        "qwen3_5_moe_q4",
        "qwen3_moe_q4",
    ] {
        let (got, want) = logits(name);
        let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
        let diffs: Vec<f32> = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).collect();
        let max = diffs.iter().fold(0.0f32, |m, &d| m.max(d));
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let close = diffs.iter().filter(|&&d| d <= 0.2 * scale).count() as f32 / diffs.len() as f32;
        let stats = format!(
            "max {max}, mean {mean}, {:.1}% close (scale {scale})",
            close * 100.0
        );
        eprintln!("{name}: {stats}");
        // Routing near a tie may pick another expert after a one-ulp difference (which
        // differs between GPU generations); the positions that follow then move far more
        // than in a dense layer. Mixtures of experts are held to the mean and to the share
        // of close logits, dense models to the largest difference.
        let moe = name.contains("moe");
        if moe {
            assert!(close >= 0.9, "{name}: {stats}");
        } else {
            assert!(max <= 0.2 * scale, "{name}: {stats}");
        }
        let mean_bound = if moe { 0.05 } else { 0.03 };
        assert!(mean <= mean_bound * scale, "{name}: {stats}");
    }
}

/// Mixture-of-experts layers: the prompt's token-expert pairs take the expert-sorted path
/// (grouped kernel), decoding and 5-token chunks the unsorted one.
#[test]
fn mixture_of_experts_qwen3_5_matches_mlx_lm() {
    check("qwen3_5_moe_f32", 1e-4);
}

/// The MoE model in Hugging Face's layout (fused expert gate and up projections).
/// Qwen 3 MoE: sparse and dense layers, no shared expert.
#[test]
fn mixture_of_experts_qwen3_matches_mlx_lm() {
    check("qwen3_moe_f32", 1e-4);
}

/// Hugging Face Qwen 3 MoE checkpoints store each expert separately.
#[test]
fn hugging_face_qwen3_moe_checkpoints_are_stacked() {
    check("qwen3_moe_hf", 1e-4);
}

#[test]
fn hugging_face_moe_checkpoints_are_sanitized() {
    check("qwen3_5_moe_hf", 1e-4);
}

/// The same model as `qwen3_5_f32` in Hugging Face's layout (prefixes, `[C, 1, K]`
/// convolutions, zero-centred norms).
#[test]
fn hugging_face_qwen3_5_checkpoints_are_sanitized() {
    check("qwen3_5_hf", 1e-4);
}

#[test]
fn logits_are_bit_identical_across_runs() {
    if !gpu() {
        return;
    }
    // Prefill and several decode steps (fused norms, cache writes in place, Gated
    // DeltaNet, quantized experts), repeated: every run gives the same bits.
    for name in ["qwen3_5_bf16", "qwen3_moe_q4", "llama_bf16"] {
        let (dir, prompt, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        let run = || -> Vec<u32> {
            let mut cache = model.new_cache();
            let mut bits = Vec::new();
            let mut tokens = Array::from_slice(&prompt, &[1, prompt.len()]).unwrap();
            for _ in 0..6 {
                let logits = model.forward(&tokens, &mut cache).unwrap();
                let last = logits.astype(DType::F32).unwrap().to_vec::<f32>().unwrap();
                bits.extend(last.iter().map(|v| v.to_bits()));
                // Greedy: the last position's largest logit.
                let row = &last[last.len() - logits.shape()[2]..];
                let next = (0..row.len())
                    .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                    .unwrap() as u32;
                tokens = Array::from_slice(&[next], &[1, 1]).unwrap();
            }
            bits
        };
        let first = run();
        for i in 0..3 {
            assert!(run() == first, "{name}: run {i} differs");
        }
    }
}

/// Batched greedy decoding over a paged cache gives each sequence the tokens it gets alone:
/// sequences of different lengths, one leaving early, and its blocks (and state slot)
/// reused by a new one.
#[test]
fn paged_batched_decoding_equals_decoding_each_sequence_alone() {
    use nuvec_lm::{PagedCache, decode_batch};
    if !gpu() {
        return;
    }
    // Hybrids keep their Gated DeltaNet states at state slots.
    for name in [
        "qwen3_f32",
        "llama_bf16",
        "qwen3_moe_f32",
        "qwen3_5_f32",
        "qwen3_5_moe_f32",
    ] {
        let (dir, prompt, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        let prompts: Vec<Vec<u32>> = [prompt.len(), prompt.len() / 2, 3, prompt.len() * 2]
            .iter()
            .map(|&n| prompt.iter().cycle().take(n.max(2)).copied().collect())
            .collect();
        let steps = 12;
        let alone: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| {
                let options = GenerateOptions {
                    max_tokens: steps,
                    ..Default::default()
                };
                generate(&model, p, options)
                    .unwrap()
                    .collect::<nuvec::Result<_>>()
                    .unwrap()
            })
            .collect();
        // Prefill all but the last prompt token into a contiguous cache, then page it.
        let mut cache = PagedCache::new(&model, 64, 8, 4).unwrap();
        let add = |cache: &mut PagedCache, p: &[u32]| {
            let mut kv = model.new_cache();
            let head = Array::from_slice(&p[..p.len() - 1], &[1, p.len() - 1]).unwrap();
            model.forward(&head, &mut kv).unwrap();
            cache.insert(&kv).unwrap()
        };
        // The first three decode together; the fourth joins when the second leaves.
        let mut live: Vec<(usize, usize, u32)> = (0..3)
            .map(|i| (i, add(&mut cache, &prompts[i]), *prompts[i].last().unwrap()))
            .collect();
        let mut got: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
        while !live.is_empty() {
            let tokens: Vec<u32> = live.iter().map(|&(_, _, t)| t).collect();
            let ids: Vec<usize> = live.iter().map(|&(_, id, _)| id).collect();
            let logits = decode_batch(&model, &tokens, &ids, &mut cache).unwrap();
            let next = nuvec::argmax(&logits, Some(-1), false)
                .unwrap()
                .to_vec::<u32>()
                .unwrap();
            for (entry, &t) in live.iter_mut().zip(&next) {
                got[entry.0].push(t);
                entry.2 = t;
            }
            let mut finished = Vec::new();
            for &(i, id, _) in &live {
                let limit = if i == 1 { steps / 2 } else { steps };
                if got[i].len() == limit {
                    finished.push((i, id));
                }
            }
            for (i, id) in finished {
                cache.remove(id).unwrap();
                live.retain(|&(j, _, _)| j != i);
                if i == 1 {
                    live.push((3, add(&mut cache, &prompts[3]), *prompts[3].last().unwrap()));
                }
            }
        }
        for (i, want) in alone.iter().enumerate() {
            let want = if i == 1 {
                &want[..steps / 2]
            } else {
                &want[..]
            };
            assert_eq!(got[i], want, "{name}: sequence {i}");
        }
        assert!(
            cache.is_empty() && cache.free_blocks() == 64,
            "{name}: blocks returned"
        );
    }
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

/// Each sampler filter keeps exactly the tokens mlx-lm's keeps, with unchanged values.
#[test]
fn sampler_filters_match_mlx_lm() {
    use nuvec_lm::sample::{apply_min_p, apply_top_k, apply_top_p};
    if !gpu() {
        return;
    }
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/samplers.safetensors");
    let fixtures = nuvec::io::load_safetensors(path).unwrap().tensors;
    let logprobs = &fixtures["logprobs"];
    let cases: [(&str, Array); 6] = [
        ("top_p_0.9", apply_top_p(logprobs, 0.9).unwrap()),
        ("top_p_0.3", apply_top_p(logprobs, 0.3).unwrap()),
        ("min_p_0.1", apply_min_p(logprobs, 0.1, 1).unwrap()),
        ("min_p_0.5_keep_4", apply_min_p(logprobs, 0.5, 4).unwrap()),
        ("top_k_1", apply_top_k(logprobs, 1).unwrap()),
        ("top_k_7", apply_top_k(logprobs, 7).unwrap()),
    ];
    for (name, got) in cases {
        let want = fixtures[name].to_vec::<f32>().unwrap();
        let got = got.to_vec::<f32>().unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g == w || (g - w).abs() <= 1e-6, "{name}[{i}]: {g} != {w}");
        }
    }
}

/// The compiled draw picks the token the uncompiled Gumbel-max formula picks with the same
/// key, and each draw takes a new key from the thread's sequence.
#[test]
fn compiled_categorical_draws_equal_the_formula_with_fresh_keys() {
    use nuvec::random;
    use nuvec_lm::sample::categorical;
    if !gpu() {
        return;
    }
    let logits = (nuvec::sin(&nuvec::arange(0.0, 4096.0, 1.0, DType::F32).unwrap()).unwrap() * 3.0)
        .reshape(&[1, 4096])
        .unwrap()
        .astype(DType::BF16)
        .unwrap();
    let formula = |key: &Array, t: f32| -> u32 {
        let u = random::uniform(1e-30, 1.0, logits.shape(), DType::F32, Some(key)).unwrap();
        let g = nuvec::negative(
            &nuvec::log(&nuvec::negative(&nuvec::log(&u).unwrap()).unwrap()).unwrap(),
        )
        .unwrap();
        let scaled =
            nuvec::multiply(logits.astype(DType::F32).unwrap(), 1.0 / f64::from(t)).unwrap();
        nuvec::argmax(&nuvec::add(&scaled, &g).unwrap(), Some(-1), false)
            .unwrap()
            .to_vec::<u32>()
            .unwrap()[0]
    };
    for (i, t) in [0.7f32, 1.0, 1.3, 0.7, 2.0, 0.7].into_iter().enumerate() {
        // The key the compiled draw will take next.
        random::seed(100 + i as u64);
        let key = random::next_key().unwrap();
        random::seed(100 + i as u64);
        let got = categorical(&logits, t).unwrap().to_vec::<u32>().unwrap()[0];
        assert_eq!(got, formula(&key, t), "temperature {t}");
    }
    // Consecutive draws from one sequence use different keys: at temperature 2 over 4,096
    // tokens, eight draws are not all equal.
    random::seed(7);
    let draws: Vec<u32> = (0..8)
        .map(|_| categorical(&logits, 2.0).unwrap().to_vec::<u32>().unwrap()[0])
        .collect();
    assert!(draws.iter().any(|&d| d != draws[0]), "{draws:?}");
}

/// Sampling draws only kept tokens, and top-k 1 at any temperature is greedy.
#[test]
fn sampling_draws_only_kept_tokens() {
    use nuvec_lm::Sampler;
    if !gpu() {
        return;
    }
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/samplers.safetensors");
    let fixtures = nuvec::io::load_safetensors(path).unwrap().tensors;
    let logprobs = &fixtures["logprobs"];
    let greedy = Sampler::default()
        .sample(logprobs)
        .unwrap()
        .to_vec::<u32>()
        .unwrap();
    let top1 = Sampler {
        temperature: 2.0,
        top_k: 1,
        ..Default::default()
    };
    assert_eq!(
        top1.sample(logprobs).unwrap().to_vec::<u32>().unwrap(),
        greedy
    );
    let nucleus = fixtures["top_p_0.3"].to_vec::<f32>().unwrap();
    let sampler = Sampler {
        temperature: 1.0,
        top_p: 0.3,
        ..Default::default()
    };
    for _ in 0..50 {
        let tokens = sampler.sample(logprobs).unwrap().to_vec::<u32>().unwrap();
        for (row, &t) in tokens.iter().enumerate() {
            assert!(
                nucleus[row * 128 + t as usize].is_finite(),
                "row {row} drew {t}"
            );
        }
    }
}

/// Speculative decoding gives mlx-lm's greedy tokens whatever the drafter proposes: the
/// true continuation (everything accepted, fewer steps), wrong tokens (everything
/// rejected), or prompt lookup; hybrid models are refused.
#[test]
fn speculative_decoding_keeps_the_greedy_output() {
    use nuvec_lm::{Drafter, PromptLookup, speculative_generate};
    if !gpu() {
        return;
    }
    /// Proposes `tokens` after the prompt (right or wrong).
    struct Fixed {
        prompt: usize,
        tokens: Vec<u32>,
    }
    impl Drafter for Fixed {
        fn draft(&mut self, context: &[u32], max: usize) -> Vec<u32> {
            let done = context.len() - self.prompt;
            self.tokens.iter().skip(done).take(max).copied().collect()
        }
    }
    // Hybrids roll their Gated DeltaNet state back to the last accepted position.
    for name in ["qwen3_f32", "llama_bf16", "qwen3_5_f32", "qwen3_5_moe_f32"] {
        let (dir, prompt, _, _, greedy) = fixture(name);
        let model = load_model(&dir).unwrap();
        let options = GenerateOptions {
            max_tokens: greedy.len(),
            ..Default::default()
        };
        let run = |drafter: Box<dyn Drafter>| {
            let mut g = speculative_generate(&model, &prompt, options, drafter, 4).unwrap();
            let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
            (ids, g.stats())
        };
        let oracle = Fixed {
            prompt: prompt.len(),
            tokens: greedy.clone(),
        };
        let (ids, stats) = run(Box::new(oracle));
        assert_eq!(ids, greedy, "{name}: oracle drafts");
        assert_eq!(
            stats.accepted, stats.drafted,
            "{name}: oracle drafts accepted"
        );
        assert!(stats.steps <= greedy.len().div_ceil(5), "{name}: {stats:?}");
        let wrong = Fixed {
            prompt: prompt.len(),
            tokens: greedy.iter().map(|t| (t + 1) % 128).collect(),
        };
        let (ids, stats) = run(Box::new(wrong));
        assert_eq!(ids, greedy, "{name}: wrong drafts");
        assert_eq!((stats.accepted, stats.steps), (0, greedy.len()), "{name}");
        // Right for two tokens, then wrong: partial acceptance, a rollback mid-draft.
        let half = Fixed {
            prompt: prompt.len(),
            tokens: greedy
                .iter()
                .enumerate()
                .map(|(i, t)| if i % 4 < 2 { *t } else { (t + 1) % 128 })
                .collect(),
        };
        let (ids, stats) = run(Box::new(half));
        assert_eq!(ids, greedy, "{name}: partly right drafts");
        assert!(
            stats.accepted > 0 && stats.accepted < stats.drafted,
            "{name}: {stats:?}"
        );
        let (ids, _) = run(Box::<PromptLookup>::default());
        assert_eq!(ids, greedy, "{name}: prompt lookup");
    }
}
