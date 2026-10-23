//! nuvec-lm against mlx-lm 0.32.2: tiny random models of each supported variant, recorded
//! with their logits and greedy continuations by `scripts/lm_fixtures.py`
//! (`just lm-fixtures`). Skipped without a Metal device.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use nuvec::{Array, DType, Device, Stream, set_default_stream};
use nuvec_lm::{GenerateOptions, generate, load_model};

fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = !cfg!(miri) && (&Array::from(1.0f32) + 1.0).eval().is_ok();
    if !ok {
        // CI on a GPU machine must not pass by skipping (as `require_gpu` in nuvec-core).
        assert!(
            std::env::var("NUVEC_REQUIRE_GPU").as_deref() != Ok("1"),
            "NUVEC_REQUIRE_GPU=1 but no Metal device is available"
        );
        eprintln!("skipped: no Metal device");
    }
    ok
}

fn fixture(name: &str) -> (PathBuf, Vec<u32>, Vec<f32>, Vec<i32>, Vec<u32>) {
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
    let tokens = Array::from_slice(&prompt, [1, prompt.len()]);
    let got = model
        .forward(&tokens, &mut model.new_cache())
        .astype(DType::F32);
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
    let tokens = Array::from_slice(&prompt, [1, prompt.len()]);
    let got = model.forward(&tokens, &mut model.new_cache());
    (got.astype(DType::F32).to_vec::<f32>().unwrap(), expected)
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
            let mut tokens = Array::from_slice(&prompt, [1, prompt.len()]);
            for _ in 0..6 {
                let logits = model.forward(&tokens, &mut cache);
                let last = logits.astype(DType::F32).to_vec::<f32>().unwrap();
                bits.extend(last.iter().map(|v| v.to_bits()));
                // Greedy: the last position's largest logit.
                let row = &last[last.len() - logits.shape()[2] as usize..];
                let next = (0..row.len())
                    .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                    .unwrap() as u32;
                tokens = Array::from_slice(&[next], [1, 1]);
            }
            bits
        };
        let first = run();
        for i in 0..3 {
            assert!(run() == first, "{name}: run {i} differs");
        }
    }
}

/// FNV-1a over the logits' bits of prefill and 8 greedy decode steps of the dense, the
/// quantized mixture-of-experts and the hybrid (Gated DeltaNet) fixtures.
fn logits_hash() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for name in ["llama_bf16", "qwen3_moe_q4", "qwen3_5_bf16"] {
        let (dir, prompt, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        let mut cache = model.new_cache();
        let mut tokens = Array::from_slice(&prompt, [1, prompt.len()]);
        for _ in 0..9 {
            let logits = model.forward(&tokens, &mut cache);
            let last = logits.astype(DType::F32).to_vec::<f32>().unwrap();
            for v in &last {
                for byte in v.to_bits().to_le_bytes() {
                    h ^= u64::from(byte);
                    h = h.wrapping_mul(0x0100_0000_01b3);
                }
            }
            let row = &last[last.len() - logits.shape()[2] as usize..];
            let next = (0..row.len())
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap() as u32;
            tokens = Array::from_slice(&[next], [1, 1]);
        }
    }
    h
}

/// Prints the logits' hash for `logits_are_bit_identical_across_processes`.
#[test]
#[ignore = "run by logits_are_bit_identical_across_processes in a child process"]
fn child_prints_the_logits_hash() {
    if std::env::var_os("NUVEC_DETERMINISM_CHILD").is_none() || !gpu() {
        return;
    }
    println!("\nhash={:016x}", logits_hash());
}

/// Two fresh processes give the same logits bit for bit (and equal this process's).
#[test]
fn logits_are_bit_identical_across_processes() {
    if !gpu() {
        return;
    }
    let child = || {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "child_prints_the_logits_hash",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("NUVEC_DETERMINISM_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find_map(|l| l.strip_prefix("hash=").map(str::to_string))
            .expect("a hash line")
    };
    let (a, b) = (child(), child());
    assert_eq!(a, b, "two processes differ");
    assert_eq!(
        a,
        format!("{:016x}", logits_hash()),
        "a child differs from this process"
    );
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
            let head = Array::from_slice(&p[..p.len() - 1], [1, p.len() - 1]);
            let _ = model.forward(&head, &mut kv).check().unwrap();
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
            let next = logits.argmax(-1).to_vec::<u32>().unwrap();
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

/// A batcher serves more requests than its batch, each getting its solo greedy tokens (up
/// to an end-of-sequence token), also when a small cache forces preemptions.
#[test]
fn batcher_serves_each_request_its_solo_tokens() {
    use nuvec_lm::{Batcher, Event, FinishReason};
    if !gpu() {
        return;
    }
    for (name, blocks) in [
        ("qwen3_f32", 64),
        ("qwen3_f32", 9),
        ("qwen3_moe_f32", 64),
        ("qwen3_5_f32", 64),
        ("qwen3_5_f32", 9),
    ] {
        let (dir, prompt, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        let requests: Vec<(Vec<u32>, usize)> = (0..6)
            .map(|i| {
                let n = 3 + (i * 5) % (prompt.len() + 4);
                (
                    prompt.iter().cycle().skip(i).take(n).copied().collect(),
                    4 + i * 3,
                )
            })
            .collect();
        let mut batcher = Batcher::new(&model, blocks, 4, 3).unwrap();
        for (p, max) in &requests {
            let options = GenerateOptions {
                max_tokens: *max,
                ..Default::default()
            };
            batcher.submit(p.clone(), options).unwrap();
        }
        let mut got: Vec<Vec<u32>> = vec![Vec::new(); requests.len()];
        let mut finished = vec![None; requests.len()];
        let mut most = 0;
        while batcher.is_busy() {
            for event in batcher.step().unwrap() {
                match event {
                    Event::Token { id, token } => got[id].push(token),
                    Event::Finished { id, reason } => finished[id] = Some(reason),
                }
            }
            most = most.max(batcher.running());
        }
        assert!(most <= 3, "{name}: batch limit");
        assert_eq!(
            batcher.preemptions() > 0,
            blocks < 64,
            "{name} ({blocks} blocks): preemptions"
        );
        for (i, (p, max)) in requests.iter().enumerate() {
            let options = GenerateOptions {
                max_tokens: *max,
                ..Default::default()
            };
            let alone: Vec<u32> = generate(&model, p, options)
                .unwrap()
                .collect::<nuvec::Result<_>>()
                .unwrap();
            // Both end at an end-of-sequence token without emitting it.
            assert_eq!(got[i], alone, "{name} ({blocks} blocks): request {i}");
            let reason = if alone.len() < *max {
                FinishReason::EndOfSequence
            } else {
                FinishReason::Length
            };
            assert_eq!(finished[i], Some(reason), "{name}: request {i} finish");
        }
    }
}

/// A seed reproduces sampled tokens, alone and in a batch with other requests.
#[test]
fn seeded_sampling_is_reproducible_alone_and_batched() {
    use nuvec_lm::{Batcher, Event, Sampler};
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let options = |seed: Option<u64>, temperature: f32| GenerateOptions {
        max_tokens: 16,
        sampler: Sampler {
            temperature,
            ..Default::default()
        },
        seed,
        ..Default::default()
    };
    let alone = |p: &[u32], o: GenerateOptions| -> Vec<u32> {
        generate(&model, p, o)
            .unwrap()
            .collect::<nuvec::Result<_>>()
            .unwrap()
    };
    let a = alone(&prompt, options(Some(7), 1.5));
    assert_eq!(a, alone(&prompt, options(Some(7), 1.5)), "same seed");
    assert_ne!(a, alone(&prompt, options(Some(8), 1.5)), "another seed");
    // Batched with others (greedy, another seed): each gets its solo tokens.
    let requests = [
        (prompt.clone(), options(Some(7), 1.5)),
        (prompt[1..].to_vec(), options(None, 0.0)),
        (prompt.clone(), options(Some(11), 1.0)),
    ];
    let mut batcher = Batcher::new(&model, 64, 8, 3).unwrap();
    for (p, o) in &requests {
        batcher.submit(p.clone(), *o).unwrap();
    }
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); requests.len()];
    while batcher.is_busy() {
        for event in batcher.step().unwrap() {
            if let Event::Token { id, token } = event {
                got[id].push(token);
            }
        }
    }
    for (i, (p, o)) in requests.iter().enumerate() {
        assert_eq!(got[i], alone(p, *o), "request {i}");
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
    let logits = (nuvec::ops::arange(0.0, 4096.0, 1.0).sin() * 3.0)
        .reshape([1, 4096])
        .astype(DType::BF16);
    let formula = |key: &Array, t: f32| -> u32 {
        let u = random::uniform(1e-30, 1.0, logits.shape(), DType::F32, key);
        let g = u.log().negative().log().negative();
        let scaled = logits.astype(DType::F32).multiply(1.0 / f64::from(t));
        scaled.add(&g).argmax(-1).to_vec::<u32>().unwrap()[0]
    };
    for (i, t) in [0.7f32, 1.0, 1.3, 0.7, 2.0, 0.7].into_iter().enumerate() {
        // The key the compiled draw will take next.
        random::seed(100 + i as u64);
        let key = random::next_key();
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
/// rejected), or prompt lookup; hybrid models too (their Gated DeltaNet states roll back).
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

/// A copy of a fixture's model whose `config.json` is `edit`ed, with a
/// `generation_config.json` when given.
fn model_copy(name: &str, edit: impl Fn(&str) -> String, generation: Option<&str>) -> PathBuf {
    static COPIES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let (src, ..) = fixture(name);
    let n = COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nuvec-lm-{}-{name}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(src.join("model.safetensors"), dir.join("model.safetensors")).unwrap();
    let config = std::fs::read_to_string(src.join("config.json")).unwrap();
    std::fs::write(dir.join("config.json"), edit(&config)).unwrap();
    if let Some(g) = generation {
        std::fs::write(dir.join("generation_config.json"), g).unwrap();
    }
    dir
}

/// `config.json` with `eos_token_id` set (after `model_type`).
fn with_eos(config: &str, eos: &str) -> String {
    config.replacen(
        "\"model_type\":",
        &format!("\"eos_token_id\": {eos},\n  \"model_type\":"),
        1,
    )
}

fn tokens(model: &nuvec_lm::Model, prompt: &[u32], options: GenerateOptions) -> Vec<u32> {
    generate(model, prompt, options)
        .unwrap()
        .collect::<nuvec::Result<_>>()
        .unwrap()
}

/// A model's forward pass rejects a cache of another number of layers.
#[test]
fn forward_rejects_a_cache_of_another_length() {
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let tokens = Array::from_slice(&prompt, [1, prompt.len()]);
    let mut cache = model.new_cache();
    assert!(model.forward(&tokens, &mut cache[..1]).check().is_err());
    cache.push(Default::default());
    assert!(model.forward(&tokens, &mut cache).check().is_err());
}

/// The end-of-sequence ids are the union of `config.json`'s and
/// `generation_config.json`'s (as mlx-lm's stop set, recorded by `lm_fixtures.py`);
/// `generate` and the batcher stop at either without yielding it.
#[test]
fn generation_stops_at_either_files_end_of_sequence_ids() {
    use nuvec_lm::{Batcher, Config, Event, FinishReason};
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/eos");
    let config = Config::from_file(dir.join("config.json")).unwrap();
    let expected =
        nuvec::io::json::parse(&std::fs::read(dir.join("expected.json")).unwrap()).unwrap();
    let want: Vec<u32> = expected
        .get("eos_token_ids")
        .and_then(|i| i.as_array())
        .unwrap()
        .iter()
        .map(|i| i.as_i64().unwrap() as u32)
        .collect();
    let mut got = config.eos_token_ids.clone();
    got.sort_unstable();
    assert_eq!(got, want);
    if !gpu() {
        return;
    }
    let (_, prompt, _, _, greedy) = fixture("qwen3_f32");
    // The first token not seen before, after two.
    let k = (2..greedy.len())
        .find(|&i| !greedy[..i].contains(&greedy[i]))
        .unwrap();
    let eos = greedy[k];
    let options = GenerateOptions {
        max_tokens: greedy.len(),
        ..Default::default()
    };
    for (edit, generation) in [
        (Some(eos.to_string()), None),
        (None, Some(format!("{{\"eos_token_id\": [{eos}]}}"))),
        (
            Some("[999]".to_owned()),
            Some(format!("{{\"eos_token_id\": {eos}}}")),
        ),
    ] {
        let dir = model_copy(
            "qwen3_f32",
            |c| match &edit {
                Some(e) => with_eos(c, e),
                None => c.to_owned(),
            },
            generation.as_deref(),
        );
        let model = load_model(&dir).unwrap();
        assert!(model.config.eos_token_ids.contains(&eos));
        assert_eq!(tokens(&model, &prompt, options), greedy[..k], "{edit:?}");
        // The batcher too, finishing with `EndOfSequence`.
        let mut batcher = Batcher::new(&model, 16, 8, 2).unwrap();
        batcher.submit(prompt.clone(), options).unwrap();
        let (mut got, mut reason) = (Vec::new(), None);
        while batcher.is_busy() {
            for event in batcher.step().unwrap() {
                match event {
                    Event::Token { token, .. } => got.push(token),
                    Event::Finished { reason: r, .. } => reason = Some(r),
                }
            }
        }
        assert_eq!(got, greedy[..k]);
        assert_eq!(reason, Some(FinishReason::EndOfSequence));
        std::fs::remove_dir_all(dir).ok();
    }
    // End of sequence first: no token; `max_tokens = 0`: no token.
    let first = greedy[0].to_string();
    let dir = model_copy("qwen3_f32", |c| with_eos(c, &first), None);
    let model = load_model(&dir).unwrap();
    assert!(tokens(&model, &prompt, options).is_empty());
    let none = GenerateOptions {
        max_tokens: 0,
        ..Default::default()
    };
    assert!(tokens(&model, &prompt, none).is_empty());
    std::fs::remove_dir_all(dir).ok();
}

/// Invalid sampler settings are errors when generation starts; a `top_k` of the whole
/// vocabulary filters nothing; the filters chain in mlx-lm's order.
#[test]
fn sampler_settings_are_checked_and_filters_chain_as_mlx_lm() {
    use nuvec_lm::sample::{apply_min_p, apply_top_k, apply_top_p};
    use nuvec_lm::{Batcher, PromptLookup, Sampler, speculative_generate};
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let invalid = [
        Sampler {
            temperature: -1.0,
            ..Default::default()
        },
        Sampler {
            temperature: f32::NAN,
            ..Default::default()
        },
        Sampler {
            temperature: f32::INFINITY,
            ..Default::default()
        },
        Sampler {
            top_p: 1.5,
            ..Default::default()
        },
        Sampler {
            top_p: -0.1,
            ..Default::default()
        },
        Sampler {
            top_p: f32::NAN,
            ..Default::default()
        },
        Sampler {
            min_p: 1.5,
            ..Default::default()
        },
        Sampler {
            min_p: -0.1,
            ..Default::default()
        },
        Sampler {
            min_p: f32::NAN,
            ..Default::default()
        },
        Sampler {
            min_tokens_to_keep: 0,
            ..Default::default()
        },
    ];
    let mut batcher = Batcher::new(&model, 16, 8, 2).unwrap();
    for sampler in invalid {
        let options = GenerateOptions {
            max_tokens: 4,
            sampler,
            ..Default::default()
        };
        assert!(sampler.validate().is_err(), "{sampler:?}");
        assert!(generate(&model, &prompt, options).is_err(), "{sampler:?}");
        assert!(
            speculative_generate(&model, &prompt, options, PromptLookup::default(), 2).is_err(),
            "{sampler:?}"
        );
        assert!(
            batcher.submit(prompt.clone(), options).is_err(),
            "{sampler:?}"
        );
    }
    // `top_k` of the vocabulary or more: the same draws as without.
    let sampled = |top_k: usize| {
        let options = GenerateOptions {
            max_tokens: 8,
            sampler: Sampler {
                temperature: 1.0,
                top_k,
                ..Default::default()
            },
            seed: Some(3),
            ..Default::default()
        };
        tokens(&model, &prompt, options)
    };
    assert_eq!(sampled(128), sampled(0));
    assert_eq!(sampled(1000), sampled(0));
    // Chained: top-p 0.9, min-p 0.05, then top-k 20 keep exactly mlx-lm's tokens.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/samplers_chained.safetensors");
    let fixtures = nuvec::io::load_safetensors(path).unwrap().tensors;
    let logprobs = &fixtures["logprobs"];
    let chained = apply_top_k(
        &apply_min_p(&apply_top_p(logprobs, 0.9).unwrap(), 0.05, 1).unwrap(),
        20,
    )
    .unwrap()
    .to_vec::<f32>()
    .unwrap();
    let want = fixtures["chained"].to_vec::<f32>().unwrap();
    for (i, (g, w)) in chained.iter().zip(&want).enumerate() {
        assert!(g == w || (g - w).abs() <= 1e-6, "chained[{i}]: {g} != {w}");
    }
    // The sampler draws only those (top-k first would keep fewer, and other, tokens):
    // at a high temperature, the draws reach tokens only mlx-lm's order keeps.
    let first = fixtures["top_k_first"].to_vec::<f32>().unwrap();
    let sampler = Sampler {
        temperature: 10.0,
        top_p: 0.9,
        min_p: 0.05,
        top_k: 20,
        ..Default::default()
    };
    let mut only_in_order = 0;
    for seed in 0..200 {
        let key = nuvec::random::key(seed);
        let drawn = sampler
            .sample_with(logprobs, Some(&key))
            .unwrap()
            .to_vec::<u32>()
            .unwrap();
        for (row, &t) in drawn.iter().enumerate() {
            let at = row * 128 + t as usize;
            assert!(
                want[at].is_finite(),
                "row {row} drew {t}, filtered by mlx-lm"
            );
            only_in_order += usize::from(!first[at].is_finite());
        }
    }
    assert!(only_in_order > 0, "the order is visible");
}

/// Malformed configs are errors, not panics; without `tie_word_embeddings`, a checkpoint
/// without `lm_head.weight` is tied and one with it is not.
#[test]
fn malformed_configs_are_errors_and_tying_is_inferred() {
    use nuvec_lm::Config;
    let parse = |text: &str| Config::from_json(&nuvec::io::json::parse(text.as_bytes()).unwrap());
    let base = r#""model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2,
        "intermediate_size": 128, "num_key_value_heads": 2, "vocab_size": 128"#;
    assert!(parse(&format!("{{{base}, \"num_attention_heads\": 4}}")).is_ok());
    for bad in [
        r#""num_attention_heads": 0"#,
        r#""num_attention_heads": -4"#,
        r#""num_attention_heads": 4, "head_dim": -16"#,
        r#""num_attention_heads": 4, "eos_token_id": -1"#,
        r#""num_attention_heads": 4, "quantization": {"group_size": 0, "bits": 4}"#,
        r#""num_attention_heads": 4, "quantization": {"group_size": 64, "bits": 4,
            "model.layers.0.mlp.gate_proj": {"group_size": 0, "bits": 8}}"#,
        r#""num_attention_heads": 4, "quantization": {"group_size": -64, "bits": 4}"#,
    ] {
        assert!(parse(&format!("{{{base}, {bad}}}")).is_err(), "{bad}");
    }
    let hybrid = r#""model_type": "qwen3_5", "hidden_size": 64, "num_hidden_layers": 4,
        "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "vocab_size": 128, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 16"#;
    assert!(parse(&format!("{{{hybrid}}}")).is_ok());
    for bad in [
        r#""full_attention_interval": 0"#,
        r#""linear_conv_kernel_dim": 0"#,
        r#""num_experts": 4, "num_experts_per_tok": 0, "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8"#,
    ] {
        assert!(parse(&format!("{{{hybrid}, {bad}}}")).is_err(), "{bad}");
    }
    if !gpu() {
        return;
    }
    // Tied (no `lm_head` in the checkpoint) and untied, without the setting.
    for (name, line) in [
        ("llama_bf16", "\n  \"tie_word_embeddings\": true,"),
        ("qwen3_f32", ",\n  \"tie_word_embeddings\": false"),
    ] {
        let dir = model_copy(name, |c| c.replacen(line, "", 1), None);
        let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
        assert!(!config.contains("tie_word_embeddings"), "{name}");
        let model = load_model(&dir).unwrap();
        assert_eq!(model.config.tie_word_embeddings, name == "llama_bf16");
        let (_, prompt, _, _, greedy) = fixture(name);
        let options = GenerateOptions {
            max_tokens: greedy.len(),
            ..Default::default()
        };
        assert_eq!(tokens(&model, &prompt, options), greedy, "{name}");
        std::fs::remove_dir_all(dir).ok();
    }
}

/// A byte-level tokenizer from its vocabulary (the 256 byte characters first, then
/// `extra`), merges and added tokens (`(content, id)`).
fn byte_level_tokenizer(
    extra: &[String],
    merges: &[(String, String)],
    added: &[(String, u32)],
) -> nuvec_lm::Tokenizer {
    // GPT-2's byte-to-character table.
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    let mut shifted = 0;
    let chars: Vec<char> = (0..256u32)
        .map(|b| {
            if printable(b) {
                char::from_u32(b).unwrap()
            } else {
                shifted += 1;
                char::from_u32(255 + shifted).unwrap()
            }
        })
        .collect();
    let quote = |s: &str| {
        let mut out = String::from("\"");
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out + "\""
    };
    let vocab: Vec<String> = chars
        .iter()
        .map(char::to_string)
        .chain(extra.iter().cloned())
        .enumerate()
        .map(|(i, t)| format!("{}:{i}", quote(&t)))
        .collect();
    let merges: Vec<String> = merges
        .iter()
        .map(|(a, b)| quote(&format!("{a} {b}")))
        .collect();
    let added: Vec<String> = added
        .iter()
        .map(|(c, id)| format!(r#"{{"id":{id},"content":{},"special":true}}"#, quote(c)))
        .collect();
    let text = format!(
        r#"{{"pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}},
          "added_tokens":[{}],
          "model":{{"type":"BPE","vocab":{{{}}},"merges":[{}]}}}}"#,
        added.join(","),
        vocab.join(","),
        merges.join(",")
    );
    nuvec_lm::Tokenizer::from_json(&nuvec::io::json::parse(text.as_bytes()).unwrap()).unwrap()
}

/// Encoding a 10× longer text takes at most 15× as long (best of three): text with many
/// of 300 added tokens, a long letter run and a base64-like blob.
#[test]
fn encoding_time_grows_linearly() {
    // Merges of letter runs (`a a`, `aa aa`, …) and of letter pairs.
    let mut extra = Vec::new();
    let mut merges = Vec::new();
    let mut run = "a".to_owned();
    for _ in 0..8 {
        merges.push((run.clone(), run.clone()));
        run = run.repeat(2);
        extra.push(run.clone());
    }
    let letters: Vec<char> = ('A'..='Z').chain('a'..='z').collect();
    for &x in &letters {
        for &y in &letters {
            if x == 'a' && y == 'a' {
                continue;
            }
            merges.push((x.to_string(), y.to_string()));
            extra.push(format!("{x}{y}"));
        }
    }
    let added: Vec<(String, u32)> = (0..300)
        .map(|i| (format!("<|extra_{i}|>"), 10_000 + i))
        .collect();
    let tokenizer = byte_level_tokenizer(&extra, &merges, &added);
    let mut state = 12345u64;
    let mut random = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let chat: String = (0..4000)
        .map(|_| format!("Hello there {} ", added[random() % 300].0))
        .collect();
    const BASE64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let blob: String = (0..200_000)
        .map(|_| BASE64[random() % 64] as char)
        .collect();
    let letters_run = "a".repeat(200_000);
    for (name, text) in [
        ("added tokens", &chat),
        ("letters", &letters_run),
        ("base64", &blob),
    ] {
        let large = &text[..text.len().min(200_000)];
        // A prefix of a tenth (at a character boundary: the texts are ASCII).
        let small = &large[..large.len() / 10];
        // Each run with an empty word cache (a clone's), as a first encoding.
        let best = |t: &str| {
            (0..3)
                .map(|_| {
                    let fresh = tokenizer.clone();
                    let start = std::time::Instant::now();
                    std::hint::black_box(fresh.encode_text(t));
                    start.elapsed().as_secs_f64()
                })
                .fold(f64::INFINITY, f64::min)
        };
        let ratio = best(large) / best(small);
        eprintln!("{name}: 10× the text took {ratio:.1}× as long");
        assert!(
            ratio <= 15.0,
            "{name}: 10× the text took {ratio:.1}× as long"
        );
    }
    // The added tokens are found (the longest at the earliest position).
    let ids = tokenizer.encode_text("x<|extra_12|><|extra_1|>");
    assert_eq!(ids[1..], [10_012, 10_001]);
}

/// After an invalid byte, `StreamDecoder` keeps an unfinished valid character: `[0xFF,
/// 0xC3]` then `[0xA9]` is U+FFFD then `é`.
#[test]
fn stream_decoder_keeps_a_character_after_an_invalid_byte() {
    use nuvec_lm::StreamDecoder;
    // Byte-level characters of 0xFF, 0xC3 and 0xA9 are themselves.
    let tokenizer =
        byte_level_tokenizer(&["ÿÃ".to_owned()], &[("ÿ".to_owned(), "Ã".to_owned())], &[]);
    let both = tokenizer.token_id("ÿÃ").unwrap();
    let last = tokenizer.token_id("©").unwrap();
    assert_eq!(tokenizer.decode_bytes(&[both, last]), [0xFF, 0xC3, 0xA9]);
    let mut decoder = StreamDecoder::default();
    let mut text = decoder.push(&tokenizer, both);
    assert_eq!(text, "\u{FFFD}");
    text += &decoder.push(&tokenizer, last);
    text += &decoder.finish();
    assert_eq!(text, "\u{FFFD}é");
}

/// A drafter proposing more than asked is cut to the requested count (the same output and
/// counts as one that is not); one-token prompts and `max_tokens = 1` work; dropping a
/// generation mid-way waits for its step.
#[test]
fn speculative_decoding_handles_its_edge_cases() {
    use nuvec_lm::{Drafter, speculative_generate};
    if !gpu() {
        return;
    }
    /// The true continuation, `extra` tokens longer than asked.
    struct Oracle {
        prompt: usize,
        tokens: Vec<u32>,
        extra: usize,
    }
    impl Drafter for Oracle {
        fn draft(&mut self, context: &[u32], max: usize) -> Vec<u32> {
            let done = context.len() - self.prompt;
            let mut d: Vec<u32> = self.tokens.iter().skip(done).take(max).copied().collect();
            d.extend(std::iter::repeat_n(7, self.extra));
            d
        }
    }
    let (dir, prompt, _, _, greedy) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let options = |max_tokens| GenerateOptions {
        max_tokens,
        ..Default::default()
    };
    let run = |extra: usize, max: usize| {
        let drafter = Oracle {
            prompt: prompt.len(),
            tokens: greedy.clone(),
            extra,
        };
        let mut g = speculative_generate(&model, &prompt, options(max), drafter, 4).unwrap();
        let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
        (ids, g.stats())
    };
    let (ids, stats) = run(0, greedy.len());
    assert_eq!(ids, greedy);
    assert_eq!(run(5, greedy.len()), (ids, stats), "over-returning drafter");
    assert_eq!(run(5, 1).0, greedy[..1], "max_tokens = 1");
    let one = speculative_generate(
        &model,
        &prompt[..1],
        options(4),
        Oracle {
            prompt: 1,
            tokens: Vec::new(),
            extra: 3,
        },
        4,
    )
    .unwrap()
    .collect::<nuvec::Result<Vec<u32>>>()
    .unwrap();
    assert_eq!(
        one,
        tokens(&model, &prompt[..1], options(4)),
        "one-token prompt"
    );
    // Dropped after one token, with the next step in flight.
    let mut g = speculative_generate(
        &model,
        &prompt,
        options(8),
        Oracle {
            prompt: prompt.len(),
            tokens: Vec::new(),
            extra: 0,
        },
        4,
    )
    .unwrap();
    assert_eq!(g.next().unwrap().unwrap(), greedy[0]);
    drop(g);
    assert_eq!(
        tokens(&model, &prompt, options(2)),
        greedy[..2],
        "usable after"
    );
}

/// The derived models' parameter names, shapes and dtypes equal the checkpoints' (MLX
/// layout fixtures, loaded strictly), and quantized tensors are buffers.
#[test]
fn parameter_lists_equal_the_checkpoints() {
    use nuvec_nn::Module;
    if !gpu() {
        return;
    }
    for name in [
        "qwen3_f32",
        "llama_bf16",
        "qwen2_q4",
        "qwen3_5_f32",
        "qwen3_5_moe_q4",
        "qwen3_moe_q4",
    ] {
        let (dir, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        let mut file: std::collections::BTreeMap<String, Array> = std::collections::BTreeMap::new();
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let file_name = path.file_name().unwrap().to_str().unwrap().to_owned();
            if file_name.starts_with("model") && file_name.ends_with(".safetensors") {
                file.extend(nuvec::io::load_safetensors(&path).unwrap().tensors);
            }
        }
        // mlx-lm's sanitize drops the multimodal wrapper's prefix and rotary frequencies.
        let mut file: std::collections::BTreeMap<String, Array> = file
            .into_iter()
            .map(|(n, a)| {
                (
                    n.strip_prefix("language_model.").unwrap_or(&n).to_owned(),
                    a,
                )
            })
            .collect();
        file.retain(|n, _| !n.contains("rotary_emb.inv_freq"));
        if model.config.tie_word_embeddings {
            file.retain(|n, _| !n.starts_with("lm_head."));
        }
        let ours: Vec<(String, Vec<i32>, DType)> = model
            .parameters()
            .into_iter()
            .map(|(n, a)| (n, a.shape().to_vec(), a.dtype()))
            .collect();
        let mut sorted = ours.clone();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let theirs: Vec<(String, Vec<i32>, DType)> = file
            .iter()
            .map(|(n, a)| (n.clone(), a.shape().to_vec(), a.dtype()))
            .collect();
        assert_eq!(sorted, theirs, "{name}");
        for (n, a) in model.trainable_parameters() {
            assert!(
                !n.ends_with(".scales") && !n.ends_with(".biases") && a.dtype() != DType::U32,
                "{name}: {n} is a quantized tensor and must not be trainable"
            );
        }
        assert!(model.num_params() > 0);
    }
}

mod lora {
    use nuvec::{Array, DType};
    use nuvec_lm::load_model;
    use nuvec_lm::lora::{self, AdapterConfig, LoraParams};
    use nuvec_nn::Module;

    use super::{fixture, gpu};

    fn logits(model: &nuvec_lm::Model, prompt: &[u32]) -> Vec<f32> {
        let tokens = Array::from_slice(prompt, [1, prompt.len()]);
        model
            .forward(&tokens, &mut model.new_cache())
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap()
    }

    fn names(model: &nuvec_lm::Model) -> Vec<String> {
        model.parameters().into_iter().map(|(n, _)| n).collect()
    }

    /// LoRA by path gives mlx-lm's names on the selected blocks only, keeps the base
    /// model's logits (zero `lora_b`), and freezing everything but the adapters leaves only
    /// them trainable.
    #[test]
    fn lora_by_path_on_float_and_quantized_bases() {
        if !gpu() {
            return;
        }
        for name in ["qwen3_f32", "qwen2_q4"] {
            let (dir, prompt, ..) = fixture(name);
            let base = load_model(&dir).unwrap();
            let mut model = base.clone();
            let keys = ["self_attn.q_proj", "self_attn.v_proj"];
            let n = lora::linear_to_lora_layers(&mut model, 1, Some(&keys), LoraParams::default())
                .unwrap();
            assert_eq!(n, 2, "{name}");
            let all = names(&model);
            for proj in ["q_proj", "v_proj"] {
                for leaf in ["linear.weight", "lora_a", "lora_b"] {
                    let p = format!("model.layers.1.self_attn.{proj}.{leaf}");
                    assert!(all.contains(&p), "{name}: {p}");
                }
                assert!(
                    all.contains(&format!("model.layers.0.self_attn.{proj}.weight")),
                    "{name}: block 0 unchanged"
                );
            }
            assert_eq!(
                logits(&model, &prompt),
                logits(&base, &prompt),
                "{name}: zero lora_b"
            );
            model.freeze(|_| true).unwrap();
            model.unfreeze(|p| p.contains("lora_")).unwrap();
            let trainable: Vec<String> = model
                .trainable_parameters()
                .into_iter()
                .map(|(n, _)| n)
                .collect();
            assert_eq!(trainable.len(), 4, "{name}: {trainable:?}");
            assert!(
                trainable
                    .iter()
                    .all(|p| p.ends_with("lora_a") || p.ends_with("lora_b"))
            );
            let path = std::env::temp_dir().join(format!(
                "nuvec-lm-{}-{name}-trainable.safetensors",
                std::process::id()
            ));
            model.save_trainable(&path).unwrap();
            let file = nuvec::io::load_safetensors(&path).unwrap();
            let mut saved: Vec<&String> = file.tensors.keys().collect();
            saved.sort();
            let mut want: Vec<&String> = trainable.iter().collect();
            want.sort();
            assert_eq!(saved, want, "{name}");
        }
    }

    /// Without keys, every linear layer of the selected blocks (mlx-lm's default); a
    /// mixture-of-experts layer is an error naming its type, and nothing changes.
    #[test]
    fn default_keys_and_unsupported_layers() {
        if !gpu() {
            return;
        }
        let (dir, ..) = fixture("qwen3_f32");
        let mut model = load_model(&dir).unwrap();
        let n = lora::linear_to_lora_layers(&mut model, 1, None, LoraParams::default()).unwrap();
        assert_eq!(n, 7, "q, k, v, o, gate, up and down");
        assert!(
            lora::linear_to_lora_layers(&mut model, 1, None, LoraParams::default()).is_err(),
            "already LoRA"
        );
        let (dir, ..) = fixture("qwen3_moe_f32");
        let mut moe = load_model(&dir).unwrap();
        let before = names(&moe);
        let err = lora::linear_to_lora_layers(
            &mut moe,
            1,
            Some(&["mlp.switch_mlp.gate_proj"]),
            LoraParams::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("SwitchProjection"), "{err}");
        assert_eq!(names(&moe), before, "nothing changed");
        // The router (a plain projection) is fine.
        assert_eq!(
            lora::linear_to_lora_layers(&mut moe, 1, Some(&["mlp.gate"]), LoraParams::default())
                .unwrap(),
            1
        );
    }

    /// Adapter directories written by mlx-lm load with mlx-lm's names, values and logits, and
    /// directories nuvec writes have mlx-lm's layout and load back.
    #[test]
    fn mlx_lm_adapter_directories() {
        if !gpu() {
            return;
        }
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        for (base, adapters, n) in [
            ("qwen3_f32", "adapters_qwen3", 14),
            ("qwen2_q4", "adapters_qwen2_q4", 8),
        ] {
            let (dir, prompt, ..) = fixture(base);
            let mut model = load_model(&dir).unwrap();
            let adapter_dir = root.join(adapters);
            let config = lora::load_adapters(&adapter_dir, &mut model).unwrap();
            assert!(
                !model.training(),
                "load_model leaves the model in eval mode"
            );
            let file =
                nuvec::io::load_safetensors(adapter_dir.join("adapters.safetensors")).unwrap();
            assert_eq!(file.tensors.len(), n);
            let params: std::collections::BTreeMap<String, Array> =
                model.parameters().into_iter().collect();
            for (name, w) in &file.tensors {
                let ours = &params[name];
                assert_eq!(
                    ours.to_vec::<f32>().unwrap(),
                    w.astype(DType::F32).to_vec::<f32>().unwrap(),
                    "{adapters}: {name}"
                );
            }
            let expected = nuvec::io::load_safetensors(adapter_dir.join("expected.safetensors"))
                .unwrap()
                .tensors;
            let want = expected["logits"].to_vec::<f32>().unwrap();
            let got = logits(&model, &prompt);
            let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
            let tol = if base == "qwen2_q4" { 2e-2 } else { 1e-4 };
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - e).abs() <= tol * scale,
                    "{adapters}: logit {i}: {g} vs mlx-lm {e}"
                );
            }
            // Written back: mlx-lm's layout, and it loads again.
            let out =
                std::env::temp_dir().join(format!("nuvec-lm-{}-{adapters}", std::process::id()));
            lora::save_adapters(&out, &model, &config).unwrap();
            let ours = nuvec::io::load_safetensors(out.join("adapters.safetensors")).unwrap();
            assert_eq!(
                ours.tensors.keys().collect::<Vec<_>>(),
                file.tensors.keys().collect::<Vec<_>>()
            );
            let written =
                nuvec::io::json::parse(&std::fs::read(out.join("adapter_config.json")).unwrap())
                    .unwrap();
            let original = nuvec::io::json::parse(
                &std::fs::read(adapter_dir.join("adapter_config.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(written, original, "{adapters}: adapter_config.json");
            let mut again = load_model(&dir).unwrap();
            assert_eq!(lora::load_adapters(&out, &mut again).unwrap(), config);
            assert_eq!(logits(&again, &prompt), got);
        }
        // Another fine-tuning type is an error.
        let bad = std::env::temp_dir().join(format!("nuvec-lm-{}-dora", std::process::id()));
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("adapter_config.json"), r#"{"fine_tune_type": "dora", "num_layers": 1, "lora_parameters": {"rank": 2, "scale": 1.0}}"#).unwrap();
        let (dir, ..) = fixture("qwen3_f32");
        let mut model = load_model(&dir).unwrap();
        let err = lora::load_adapters(&bad, &mut model)
            .unwrap_err()
            .to_string();
        assert!(err.contains("dora"), "{err}");
        let config = AdapterConfig::new(2, LoraParams::default()).keys(["self_attn.q_proj"]);
        assert_eq!(
            config.keys.as_deref(),
            Some(&["self_attn.q_proj".to_owned()][..])
        );
    }
}
