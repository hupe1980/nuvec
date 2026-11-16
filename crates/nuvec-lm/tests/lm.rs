//! nuvec-lm against mlx-lm 0.32.2: tiny random models of each supported variant, recorded
//! with their logits and greedy continuations by `scripts/lm_fixtures.py`
//! (`just lm-fixtures`). Skipped without a Metal device.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use nuvec::{Array, DType, Device, Stream, set_default_stream};
use nuvec_lm::{GenerateOptions, generate, load_model};

/// `options` with tokens equal to `generate`'s, bit for bit ([`GenerateOptions::row_exact`]).
fn exact(options: GenerateOptions) -> GenerateOptions {
    GenerateOptions {
        row_exact: true,
        ..options
    }
}

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
    check_prefills(name, tol, &[2048, 5]);
}

/// [`check`] with the given prefill steps.
fn check_prefills(name: &str, tol: f32, prefill_steps: &[usize]) {
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
    for &prefill_step in prefill_steps {
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

/// Floating-point quantized checkpoints as `mlx_lm.convert --q-mode` writes them: dense
/// mxfp4, nvfp4 and mxfp8, mixture-of-experts with 8-bit affine routers (a per-layer
/// config) and nvfp4 experts with a global scale each; each layer's mode is recognised from
/// its tensors, as mlx-lm's loader does.
#[test]
fn fp_quantized_models_match_mlx_lm() {
    for name in [
        "qwen3_mxfp4",
        "qwen3_nvfp4",
        "qwen3_mxfp8",
        "qwen3_moe_mxfp4",
        "qwen3_moe_nvfp4_gs",
        "llama_mxfp8",
    ] {
        check(name, 2e-2);
    }
}

/// AutoAWQ and GPTQ (v1 4-bit, v2 8-bit with an identity `g_idx`) checkpoints, converted at
/// load, give the logits and greedy tokens of the same codes, zeros and scales in MLX's
/// affine layout under mlx-lm (which loads the AWQ files to the same logits itself).
#[test]
fn awq_and_gptq_checkpoints_match_mlx_lm() {
    // Bit for bit on the GPU the logits were recorded on (an M4 Pro); other generations
    // round their kernels' sums differently (up to 0.25% of the largest logit on an M1,
    // 0.6% on the CPU).
    let tol = if recorded_gpu() { 1e-6 } else { 1e-2 };
    for name in ["qwen3_awq", "llama_gptq", "qwen3_gptq8_v2"] {
        check(name, tol);
    }
}

/// Whether this is the GPU architecture the fixtures' mlx-lm outputs were recorded on.
fn recorded_gpu() -> bool {
    nuvec::metal::device_info().is_some_and(|d| d.architecture == "applegpu_g16s")
}

/// Gemma 4's model types (`gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text`)
/// read as one family, a wrapper's vocabulary taking precedence (mlx-lm's `gemma4`).
#[test]
fn gemma4_model_types_read_alike() {
    use nuvec::io::json;
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gemma4_kv_f32/config.json"),
    )
    .unwrap();
    let base = nuvec_lm::Config::from_json(&json::parse(text.as_bytes()).unwrap()).unwrap();
    for kind in ["gemma4_text", "gemma4_unified_text"] {
        let c = text.replace("\"gemma4_text\"", &format!("\"{kind}\""));
        let c = nuvec_lm::Config::from_json(&json::parse(c.as_bytes()).unwrap()).unwrap();
        assert_eq!(c, base, "{kind}");
    }
    for kind in ["gemma4", "gemma4_unified"] {
        let wrapper =
            format!("{{\"model_type\": \"{kind}\", \"vocab_size\": 128, \"text_config\": {text}}}");
        let c = nuvec_lm::Config::from_json(&json::parse(wrapper.as_bytes()).unwrap()).unwrap();
        assert_eq!(c.model_type, "gemma4", "{kind}");
        assert_eq!(c.gemma4, base.gemma4, "{kind}");
    }
}

/// Every fixture model's parameter paths are distinct, transparent fields included (the
/// derive checks named siblings only).
#[test]
fn parameter_paths_are_distinct() {
    use nuvec::Tree;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if !path.join("config.json").is_file() || !path.join("model.safetensors").is_file() {
            continue;
        }
        if path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("dflash_")
        {
            let e = load_model(&path).unwrap_err().to_string();
            assert!(e.contains("DFlashDrafter::load"), "{e}");
            continue;
        }
        let model = load_model(&path).unwrap();
        let def = model.def();
        let mut seen = std::collections::HashSet::new();
        for (leaf, ..) in def.leaves() {
            assert!(
                seen.insert(leaf.to_owned()),
                "{}: {leaf} twice",
                path.display()
            );
        }
    }
}

/// A one-token prompt (an empty cache) reusing a removed sequence's slot starts from zero
/// Gated DeltaNet states: its tokens equal those of the same prompt in a fresh cache.
#[test]
fn empty_sequences_reuse_slots_with_zero_states() {
    use nuvec_lm::{PagedCache, decode_batch};
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("qwen3_5_f32");
    let model = load_model(&dir).unwrap();
    let run = |cache: &mut PagedCache| {
        let id = cache.insert(&model.new_cache()).unwrap();
        let mut token = prompt[0];
        let mut out = Vec::new();
        for _ in 0..4 {
            let logits = decode_batch(&model, &[token], &[id], cache).unwrap();
            token = logits.argmax(-1).to_vec::<u32>().unwrap()[0];
            out.push(token);
        }
        cache.remove(id).unwrap();
        out
    };
    let want = run(&mut PagedCache::new(&model, 32, 4, 1).unwrap());
    let mut cache = PagedCache::new(&model, 32, 4, 1).unwrap();
    // A longer sequence first, at the same slot.
    let mut kv = model.new_cache();
    model
        .forward(&Array::from_slice(&prompt, [1, prompt.len()]), &mut kv)
        .eval()
        .unwrap();
    let id = cache.insert(&kv).unwrap();
    decode_batch(&model, &[prompt[1]], &[id], &mut cache)
        .unwrap()
        .eval()
        .unwrap();
    cache.remove(id).unwrap();
    assert_eq!(run(&mut cache), want);
}

/// A paged cache's sliding-window layers hold only their window: grown to 8·W positions,
/// a sequence keeps at most `⌈(W − 1) / bs⌉ + 1` blocks of the window group, freed
/// blocks serve another sequence, and every block is free once both are removed; the
/// tokens equal contiguous decoding's.
#[test]
fn window_groups_hold_only_their_window() {
    use nuvec_lm::{PagedCache, decode_batch};
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("gpt_oss_f32");
    let model = load_model(&dir).unwrap();
    let (w, bs) = (4usize, 2usize);
    let mut cache = PagedCache::new(&model, 64, bs, 2).unwrap();
    // Layer 0 slides (group 0), layer 1 is full (group 1).
    let (window_total, window) = cache.free_blocks_in(0).unwrap();
    assert_eq!(window, Some(w));
    assert_eq!(cache.free_blocks_in(1).unwrap(), (64, None));
    let held = |cache: &PagedCache| window_total - cache.free_blocks_in(0).unwrap().0;
    let most = (w - 1).div_ceil(bs) + 1;
    // Contiguous greedy decoding, the reference.
    let greedy = |prompt: &[u32], steps: usize| -> Vec<u32> {
        let mut kv = model.new_cache();
        let mut tokens = prompt.to_vec();
        let mut logits = model.forward(&Array::from_slice(prompt, [1, prompt.len()]), &mut kv);
        for _ in 0..steps {
            let last = logits.i((0, -1)).argmax(-1).item::<u32>().unwrap();
            tokens.push(last);
            logits = model.forward(&Array::from_slice(&[last], [1, 1]), &mut kv);
        }
        tokens[prompt.len()..].to_vec()
    };
    let insert = |cache: &mut PagedCache, prompt: &[u32]| {
        let mut kv = model.new_cache();
        let (rest, last) = prompt.split_at(prompt.len() - 1);
        model
            .forward(&Array::from_slice(rest, [1, rest.len()]), &mut kv)
            .eval()
            .unwrap();
        (cache.insert(&kv).unwrap(), last[0])
    };
    let a_prompt = &prompt[..5];
    let (a, mut a_next) = insert(&mut cache, a_prompt);
    let (mut b, mut b_next, mut b_out) = (None, 0, Vec::new());
    let mut a_out = Vec::new();
    for step in 0..8 * w {
        let mut ids = vec![a];
        let mut tokens = vec![a_next];
        if let Some(b) = b {
            ids.push(b);
            tokens.push(b_next);
        }
        let logits = decode_batch(&model, &tokens, &ids, &mut cache).unwrap();
        let next = logits.argmax(-1).to_vec::<u32>().unwrap();
        a_next = next[0];
        a_out.push(a_next);
        if b.is_some() {
            b_next = next[1];
            b_out.push(b_next);
        }
        assert!(
            held(&cache) <= most * ids.len(),
            "step {step}: {} window blocks",
            held(&cache)
        );
        if step == 2 * w {
            let (id, next) = insert(&mut cache, &prompt[2..9]);
            (b, b_next) = (Some(id), next);
        }
    }
    assert_eq!(a_out, greedy(a_prompt, a_out.len()));
    assert_eq!(b_out, greedy(&prompt[2..9], b_out.len()));
    cache.remove(a).unwrap();
    cache.remove(b.unwrap()).unwrap();
    assert_eq!(cache.free_blocks_in(0).unwrap().0, window_total);
    assert_eq!(cache.free_blocks(), 64);
}

/// Gemma 4 (sliding and full layers with their own head sizes and RoPE, q/k/v norms,
/// sandwich norms, GeGLU, the logit soft cap): with per-layer inputs and KV-shared layers
/// (the `gemma4` wrapper, bfloat16), with K = V full layers (float32, chunked prefill
/// too), and with the mixture-of-experts branch (4-bit, and Hugging Face's layout). The
/// half-precision model without experts is bit for bit mlx-lm's; with experts the logits
/// equal mlx-lm's with a precise router softmax (mlx-lm's default differs by up to
/// 0.6% of the largest logit here).
#[test]
fn gemma4_matches_mlx_lm() {
    check("gemma4_kv_f32", 1e-4);
    check("gemma4_e_f32", 1e-4);
    for name in ["gemma4_e_bf16", "gemma4_d512_bf16"] {
        check_prefills(name, 2e-2, &[2048]);
    }
    for name in ["gemma4_moe_q4", "gemma4_moe_hf"] {
        check_experts(name);
    }
}

/// A sliding-window layer's cache keeps its window and slack only: prefilled in chunks
/// over a sequence that compacts it several times, the logits equal one forward of the
/// whole sequence, its buffers stay bounded, and it trims within its slack (only).
#[test]
fn windowed_caches_hold_their_window() {
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("gpt_oss_f32");
    let model = load_model(&dir).unwrap();
    let (n, chunk) = (1300, 37);
    let tokens: Vec<u32> = prompt.iter().cycle().take(n).copied().collect();
    let last_rows = |logits: &Array, rows: usize| {
        let v = logits.astype(DType::F32).to_vec::<f32>().unwrap();
        v[v.len() - rows * 128..].to_vec()
    };
    let whole = |len: usize, rows: usize| {
        let t = Array::from_slice(&tokens[..len], [1, len]);
        last_rows(&model.forward(&t, &mut model.new_cache()), rows)
    };
    let close = |got: &[f32], want: &[f32], what: &str| {
        let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * scale,
                "{what}: logit {i}: {g} vs {w}"
            );
        }
    };
    let mut cache = model.new_cache();
    let mut last = None;
    for c in tokens.chunks(chunk) {
        last = Some(model.forward(&Array::from_slice(c, [1, c.len()]), &mut cache));
    }
    close(&last_rows(&last.unwrap(), 1), &whole(n, 1), "chunked");
    // Layer 0 slides (W 4): at most W − 1, the slack and two steps beyond a chunk; layer 1
    // is full.
    let capacity = |c: &nuvec_lm::KvCache| c.state()[0].shape()[2] as usize;
    assert!(
        capacity(&cache[0]) <= 3 + 256 + 512 + chunk,
        "{}",
        capacity(&cache[0])
    );
    assert!(capacity(&cache[1]) >= n);
    // Within the slack: the cache continues as the shorter sequence would.
    for c in &mut cache {
        c.trim(100).unwrap();
    }
    let t = Array::from_slice(&tokens[n - 100..n - 90], [1, 10]);
    close(
        &last_rows(&model.forward(&t, &mut cache), 10),
        &whole(n - 90, 10),
        "trimmed",
    );
    // Beyond it: an error, the cache unchanged.
    let offset = cache[0].offset();
    assert!(cache[0].trim(offset - 1).is_err());
    assert_eq!(cache[0].offset(), offset);
}

/// gpt-oss (sliding and full layers, YaRN, sinks, biased router and experts with the
/// clamped SwiGLU) in float32, 4-bit affine and MXFP4, and from both Hugging Face layouts
/// (the MXFP4 originals' `_blocks`/`_scales`, transformers' full-precision experts).
/// Chunked prefill (the sliding layers' rotating cache across chunks) is checked in
/// float32: in half precision these random models' greedy tokens change with mlx-lm's own
/// prefill step (flat distributions, near ties), so they follow mlx-lm's default path.
#[test]
fn gpt_oss_matches_mlx_lm() {
    check("gpt_oss_f32", 1e-4);
    check("gpt_oss_hf", 1e-4);
    for name in [
        "gpt_oss_bf16",
        "gpt_oss_hf_mxfp4",
        "gpt_oss_q4",
        "gpt_oss_mxfp4",
    ] {
        check_experts(name);
    }
}

/// A half-precision mixture of experts: as [`check`] on the GPU the fixture was recorded
/// on (one prefill); on other generations, kernels round differently by an ulp, which can
/// tip a near-tied router to another expert, so the logits are held to
/// [`assert_experts_close`] and greedy tokens may part (as MLX's own across GPUs).
fn check_experts(name: &str) {
    if !gpu() {
        return;
    }
    if recorded_gpu() {
        check_prefills(name, 2e-2, &[2048]);
    } else {
        let (got, want) = logits(name);
        assert_experts_close(name, &got, &want);
    }
}

/// Logits of a mixture of experts after a routing difference: at least 90% of them within
/// 20% of the largest, and the mean difference within 5% of it.
fn assert_experts_close(name: &str, got: &[f32], want: &[f32]) {
    let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    let diffs: Vec<f32> = got.iter().zip(want).map(|(g, w)| (g - w).abs()).collect();
    let max = diffs.iter().fold(0.0f32, |m, &d| m.max(d));
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let close = diffs.iter().filter(|&&d| d <= 0.2 * scale).count() as f32 / diffs.len() as f32;
    let stats = format!(
        "max {max}, mean {mean}, {:.1}% close (scale {scale})",
        close * 100.0
    );
    eprintln!("{name}: {stats}");
    assert!(
        close >= 0.9 && mean <= 0.05 * scale && !mean.is_nan(),
        "{name}: {stats}"
    );
}

/// The prompt's logits from a fixture's model.
fn logits(name: &str) -> (Vec<f32>, Vec<f32>) {
    let (dir, prompt, expected, _, _) = fixture(name);
    let model = load_model(&dir).unwrap();
    let tokens = Array::from_slice(&prompt, [1, prompt.len()]);
    let got = model.forward(&tokens, &mut model.new_cache());
    (got.astype(DType::F32).to_vec::<f32>().unwrap(), expected)
}

/// Granite 4 (`granitemoehybrid`): Mamba-2 and attention layers, experts with a shared MLP,
/// and its embedding, residual, attention and logit multipliers.
#[test]
fn granite4_matches_mlx_lm() {
    check("granite4_moe_f32", 1e-4);
}

/// Granite 4 dense, without rotary embeddings (`position_embedding_type: nope`).
#[test]
fn granite4_dense_matches_mlx_lm() {
    check("granite4_dense_f32", 1e-4);
}

/// Granite 4 in Hugging Face's layout (`[C, 1, K]` convolutions, `input_linear` and
/// `output_linear` experts).
#[test]
fn hugging_face_granite4_checkpoints_are_sanitized() {
    check("granite4_moe_hf", 1e-4);
}

/// LFM2: gated short convolutions and attention with q/k layer norms (its config gives
/// only `full_attn_idxs`).
#[test]
fn lfm2_matches_mlx_lm() {
    check("lfm2_f32", 1e-4);
}

/// LFM2 in Hugging Face's layout (`[C, 1, K]` convolutions).
#[test]
fn hugging_face_lfm2_checkpoints_are_sanitized() {
    check("lfm2_hf", 1e-4);
}

/// LFM2-MoE: a dense first layer, then sigmoid-routed experts whose bias chooses but does
/// not weight them (renormalised and scaled), with LFM2's convolutions and attention.
#[test]
fn lfm2_moe_matches_mlx_lm() {
    check("lfm2_moe_f32", 1e-4);
}

/// LFM2-MoE in Hugging Face's layout (`w1`/`w3`/`w2`, one tensor per expert).
#[test]
fn hugging_face_lfm2_moe_checkpoints_are_sanitized() {
    check("lfm2_moe_hf", 1e-4);
}

/// Granite 4 and LFM2 keep states that cannot be paged: the paged `Batcher` refuses them,
/// naming the family; a quantized KV cache quantizes their attention layers and leaves the
/// states as they are.
#[test]
fn granite4_and_lfm2_refuse_paging() {
    use nuvec_lm::{Batcher, BatcherOptions, KvQuant};
    if !gpu() {
        return;
    }
    for (name, family) in [
        ("granite4_moe_f32", "granitemoehybrid"),
        ("lfm2_f32", "lfm2"),
        ("lfm2_moe_f32", "lfm2_moe"),
    ] {
        let (dir, prompt, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        let options = GenerateOptions {
            max_tokens: 4,
            ..Default::default()
        };
        let Err(err) = Batcher::new(&model, BatcherOptions::default()) else {
            panic!("{name}: the paged Batcher not refused")
        };
        assert!(err.to_string().contains(family), "{name}: {err}");
        let quantized = GenerateOptions {
            kv_quant: Some(KvQuant {
                group_size: 32,
                start: 0,
                ..Default::default()
            }),
            ..options.clone()
        };
        let plain: Vec<u32> = generate(&model, &prompt, options)
            .unwrap()
            .collect::<nuvec::Result<_>>()
            .unwrap();
        let ids: Vec<u32> = generate(&model, &prompt, quantized)
            .unwrap()
            .collect::<nuvec::Result<_>>()
            .unwrap();
        assert_eq!(
            ids.len(),
            plain.len(),
            "{name}: a quantized cache generates"
        );
    }
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
        "qwen3_5_moe_nvfp4",
        "granite4_moe_q4",
        "lfm2_q4",
        "lfm2_moe_q4",
    ] {
        let (got, want) = logits(name);
        // Routing near a tie may pick another expert after a one-ulp difference (which
        // differs between GPU generations); the positions that follow then move far more
        // than in a dense layer. Mixtures of experts are held to the mean and to the share
        // of close logits, dense models to the largest difference.
        if name.contains("moe") {
            assert_experts_close(name, &got, &want);
            continue;
        }
        let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
        let diffs: Vec<f32> = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).collect();
        let max = diffs.iter().fold(0.0f32, |m, &d| m.max(d));
        let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let stats = format!("max {max}, mean {mean} (scale {scale})");
        eprintln!("{name}: {stats}");
        assert!(
            max <= 0.2 * scale && mean <= 0.03 * scale,
            "{name}: {stats}"
        );
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

/// At head size 512 (Gemma 4 26B-A4B's full layers), a paged step's logits equal one-token
/// decoding's over the contiguous cache bit for bit, for each sequence of a batch: below
/// 1,024 keys MLX decodes those layers with its composition, not the vector kernel.
#[test]
fn paged_decode_at_head_512_equals_contiguous_decode() {
    nuvec::with_row_exact(|| {
        use nuvec_lm::{PagedCache, decode_batch};
        if !gpu() {
            return;
        }
        let (dir, prompt, ..) = fixture("gemma4_d512_bf16");
        let model = load_model(&dir).unwrap();
        let bits = |a: &Array| {
            a.astype(DType::F32)
                .view(DType::U32)
                .to_vec::<u32>()
                .unwrap()
        };
        let prompts: Vec<Vec<u32>> = [prompt.len(), 3, prompt.len() * 2]
            .iter()
            .map(|&n| prompt.iter().cycle().take(n).copied().collect())
            .collect();
        let mut cache = PagedCache::new(&model, 64, 8, 3).unwrap();
        let mut want = Vec::new();
        let mut ids = Vec::new();
        for p in &prompts {
            let mut kv = model.new_cache();
            let head = Array::from_slice(&p[..p.len() - 1], [1, p.len() - 1]);
            let _ = model.forward(&head, &mut kv).check().unwrap();
            ids.push(cache.insert(&kv).unwrap());
            let last = Array::from_slice(&p[p.len() - 1..], [1, 1]);
            want.push(bits(&model.forward(&last, &mut kv).i((.., -1))));
        }
        let tokens: Vec<u32> = prompts.iter().map(|p| *p.last().unwrap()).collect();
        let logits = decode_batch(&model, &tokens, &ids, &mut cache).unwrap();
        for (r, want) in want.iter().enumerate() {
            let r = r as i32;
            assert!(
                bits(&logits.i(r..r + 1)) == *want,
                "sequence {r}: logits differ"
            );
        }
    });
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
        "gpt_oss_f32",
        "gemma4_e_f32",
        "gemma4_kv_f32",
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
    use nuvec_lm::{Batcher, BatcherOptions, Event, FinishReason};
    if !gpu() {
        return;
    }
    for (name, blocks) in [
        ("qwen3_f32", 64),
        ("qwen3_f32", 9),
        ("qwen3_moe_f32", 64),
        ("qwen3_5_f32", 64),
        ("qwen3_5_f32", 9),
        ("gpt_oss_f32", 64),
        ("gpt_oss_f32", 9),
        ("gpt_oss_bf16", 64),
        ("gpt_oss_mxfp4", 64),
        ("gemma4_e_f32", 64),
        ("gemma4_e_bf16", 64),
        ("gemma4_moe_q4", 64),
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
        let mut batcher = Batcher::new(
            &model,
            BatcherOptions {
                blocks: Some(blocks),
                block_size: 4,
                max_batch: 3,
                ..Default::default()
            },
        )
        .unwrap();
        for (p, max) in &requests {
            let options = GenerateOptions {
                max_tokens: *max,
                row_exact: true,
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

/// Prompts of the same length over 64 tokens are prefilled together (quantized models), and
/// each request still gets its solo tokens: dense, mixture-of-experts, and sliding-window
/// models with sinks (gpt-oss, Gemma 4), with prompts longer than the window.
#[test]
fn batched_prefill_gives_each_request_its_solo_tokens() {
    use nuvec_lm::{Batcher, BatcherOptions, Event};
    if !gpu() {
        return;
    }
    for name in [
        "qwen3_5_q4",
        "qwen3_5_moe_q4",
        "gpt_oss_mxfp4",
        "gpt_oss_q4",
        "gemma4_moe_q4",
        "gemma4_d512_bf16",
    ] {
        let (dir, prompt, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        let options = GenerateOptions {
            max_tokens: 12,
            ..Default::default()
        };
        let prompts: Vec<Vec<u32>> = (0..4)
            .map(|i| prompt.iter().cycle().skip(i).take(96).copied().collect())
            .collect();
        let mut batcher = Batcher::new(
            &model,
            BatcherOptions {
                blocks: Some(4 * 108 / 8 + 4),
                block_size: 8,
                max_batch: 4,
                ..Default::default()
            },
        )
        .unwrap();
        for p in &prompts {
            let options = GenerateOptions {
                row_exact: true,
                ..options.clone()
            };
            batcher.submit(p.clone(), options).unwrap();
        }
        let mut got: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
        while batcher.is_busy() {
            for event in batcher.step().unwrap() {
                if let Event::Token { id, token } = event {
                    got[id].push(token);
                }
            }
        }
        for (i, p) in prompts.iter().enumerate() {
            let alone: Vec<u32> = generate(&model, p, options.clone())
                .unwrap()
                .collect::<nuvec::Result<_>>()
                .unwrap();
            assert_eq!(got[i], alone, "{name}: request {i}");
        }
    }
}

/// A seed reproduces sampled tokens, alone and in a batch with other requests.
#[test]
fn seeded_sampling_is_reproducible_alone_and_batched() {
    use nuvec_lm::{Batcher, BatcherOptions, Event, Sampler};
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
    let mut batcher = Batcher::new(
        &model,
        BatcherOptions {
            blocks: Some(64),
            block_size: 8,
            max_batch: 3,
            ..Default::default()
        },
    )
    .unwrap();
    for (p, o) in &requests {
        let o = GenerateOptions {
            row_exact: true,
            ..o.clone()
        };
        batcher.submit(p.clone(), o).unwrap();
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
        assert_eq!(got[i], alone(p, o.clone()), "request {i}");
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
    let chunked: Vec<u32> = generate(&model, &long, options.clone())
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
/// rejected), or prompt lookup; hybrid models too (their Gated DeltaNet, Mamba-2 and
/// short-convolution states roll back).
#[test]
fn speculative_decoding_keeps_the_greedy_output() {
    use nuvec_lm::{Draft, DraftPolicy, Drafter, Model, PromptLookup, speculative_generate};
    if !gpu() {
        return;
    }
    /// Proposes `tokens` after the prompt (right or wrong).
    struct Fixed {
        prompt: usize,
        tokens: Vec<u32>,
    }
    impl Drafter for Fixed {
        fn draft(&mut self, _: &Model, context: &[u32], max: usize) -> nuvec::Result<Draft> {
            let done = context.len() - self.prompt;
            let d: Vec<u32> = self.tokens.iter().skip(done).take(max).copied().collect();
            Ok(Draft::from_tokens(&d))
        }
    }
    // Hybrids roll their recurrent and convolution states back to the last accepted
    // position.
    for name in [
        "qwen3_f32",
        "llama_bf16",
        "qwen3_5_f32",
        "qwen3_5_moe_f32",
        "gpt_oss_f32",
        "gemma4_e_f32",
        "granite4_moe_f32",
        "granite4_dense_f32",
        "lfm2_f32",
        "lfm2_moe_f32",
    ] {
        let (dir, prompt, _, _, greedy) = fixture(name);
        let model = load_model(&dir).unwrap();
        let options = GenerateOptions {
            max_tokens: greedy.len(),
            ..Default::default()
        };
        let run = |drafter: Box<dyn Drafter>| {
            let mut g = speculative_generate(
                &model,
                &prompt,
                exact(options),
                drafter,
                DraftPolicy::Fixed(4),
            )
            .unwrap();
            let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
            (ids, g.stats())
        };
        let oracle = Fixed {
            prompt: prompt.len(),
            tokens: greedy.clone(),
        };
        let (ids, stats) = run.clone()(Box::new(oracle));
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
        let (ids, stats) = run.clone()(Box::new(wrong));
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
        let (ids, stats) = run.clone()(Box::new(half));
        assert_eq!(ids, greedy, "{name}: partly right drafts");
        assert!(
            stats.accepted > 0 && stats.accepted < stats.drafted,
            "{name}: {stats:?}"
        );
        let (ids, _) = run(Box::<PromptLookup>::default());
        assert_eq!(ids, greedy, "{name}: prompt lookup");
    }
}

/// An MTP file for a Qwen 3.5 fixture, in the original checkpoints' layout: the fixture's
/// full-attention layer 3 as `mtp.layers.0`, its final norm as the three MTP norms, and a
/// random `fc` (the drafts are poor; the output must not depend on them).
fn mtp_file(source: &str) -> PathBuf {
    let (dir, ..) = fixture(source);
    let weights = nuvec::io::load_safetensors(dir.join("model.safetensors"))
        .unwrap()
        .tensors;
    let mut out: Vec<(String, nuvec::Array)> = Vec::new();
    for (k, v) in &weights {
        let k = k.strip_prefix("language_model.").unwrap_or(k);
        if let Some(rest) = k.strip_prefix("model.layers.3.") {
            out.push((format!("mtp.layers.0.{rest}"), v.clone()));
        } else if k == "model.norm.weight" {
            for n in [
                "mtp.norm",
                "mtp.pre_fc_norm_embedding",
                "mtp.pre_fc_norm_hidden",
            ] {
                out.push((format!("{n}.weight"), v.subtract(1.0)));
            }
        }
    }
    let key = nuvec::random::key(3);
    let fc = nuvec::random::normal([64, 128], nuvec::DType::F32, 0.0, 0.1, &key);
    out.push(("mtp.fc.weight".into(), fc));
    let path = std::env::temp_dir().join(format!(
        "nuvec-mtp-{}-{source}.safetensors",
        std::process::id()
    ));
    let refs: Vec<(&str, &nuvec::Array)> = out.iter().map(|(k, v)| (k.as_str(), v)).collect();
    nuvec::io::save_safetensors(&path, &refs, &[]).unwrap();
    path
}

/// The MTP head equals an independent oracle: `scripts/lm_fixtures.py` builds a random head
/// from mlx-lm's own Qwen 3.5 classes and records the logits of every prompt
/// position's next-but-one token, from the target's final hidden states and the next
/// token's embedding (dense and mixture-of-experts layers, the original checkpoints'
/// layout). Attaching it adds exactly its weights to memory; a model without a
/// head holds none.
#[test]
fn the_mtp_head_equals_mlx_lm_s_layers() {
    if !gpu() {
        return;
    }
    use nuvec_nn::Module;
    for name in ["qwen3_5_f32", "qwen3_5_moe_f32"] {
        let (dir, prompt, ..) = fixture(name);
        let want = nuvec::io::load_safetensors(dir.join("mtp_expected.safetensors"))
            .unwrap()
            .tensors["logits"]
            .to_vec::<f32>()
            .unwrap();
        let mut model = load_model(&dir).unwrap();
        assert!(!model.has_mtp());
        model.config.mtp_layers = 1;
        let file = dir.join("mtp.safetensors");
        let bytes: usize = nuvec::io::load_safetensors(&file)
            .unwrap()
            .tensors
            .values()
            .map(|t| t.size() * t.dtype().size())
            .sum();
        // The model's parameters grow by exactly the head's tensors (counted from the
        // model, not the process's allocator, which tests running alongside share).
        let held = |m: &nuvec_lm::Model| -> usize {
            m.parameters()
                .iter()
                .map(|(_, t)| t.size() * t.dtype().size())
                .sum()
        };
        let before = held(&model);
        model.attach_mtp(&file).unwrap();
        assert_eq!(
            held(&model) - before,
            bytes,
            "{name}: attaching added other bytes"
        );
        let ids = nuvec::Array::from_slice(&prompt, [1, prompt.len() as i32]);
        let mut cache = model.new_cache();
        let (_, hidden) = model.forward_hidden(&ids, &mut cache);
        let n = prompt.len() as i32;
        let next = ids.i((.., 1..n));
        let mut mtp_cache = model.new_mtp_cache();
        let h = model.mtp_forward(&next, &hidden.i((.., 0..n - 1, ..)), &mut mtp_cache);
        let got = model
            .logits(&h)
            .astype(nuvec::DType::F32)
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len(), "{name}");
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "{name}: MTP logits differ from the oracle by {worst}"
        );
    }
}

/// Speculative decoding with the model's MTP head gives the greedy tokens (mlx-lm's) for
/// every draft length, short and beyond 1,024 cached keys; the head attaches with its
/// layers quantized as the body's (an 8-bit `mlp.gate` stays 8-bit); errors name what is
/// wrong.
#[test]
fn mtp_drafts_keep_the_greedy_output() {
    use nuvec_lm::{DraftPolicy, MtpDrafter, speculative_generate};
    use nuvec_nn::Module;
    if !gpu() {
        return;
    }
    for name in ["qwen3_5_f32", "qwen3_5_moe_f32"] {
        let (dir, prompt, _, _, greedy) = fixture(name);
        let mut model = load_model(&dir).unwrap();
        assert!(MtpDrafter::new(&model).is_err(), "{name}: no head yet");
        let file = mtp_file(name);
        assert!(
            model.attach_mtp(&file).is_err(),
            "{name}: no MTP layers configured"
        );
        model.config.mtp_layers = 1;
        model.attach_mtp(&file).unwrap();
        assert!(model.has_mtp());
        let options = GenerateOptions {
            max_tokens: greedy.len(),
            ..Default::default()
        };
        for k in 1..=4 {
            let g = speculative_generate(
                &model,
                &prompt,
                exact(options.clone()),
                MtpDrafter::new(&model).unwrap(),
                DraftPolicy::Fixed(k),
            )
            .unwrap();
            let ids: Vec<u32> = g.collect::<nuvec::Result<_>>().unwrap();
            assert_eq!(ids, greedy, "{name}: MTP drafts of {k}");
        }
        // Beyond the vector kernel's one-pass limit (1,024 keys): equal to plain decoding.
        let long: Vec<u32> = prompt.iter().copied().cycle().take(1100).collect();
        let options = GenerateOptions {
            max_tokens: 24,
            ..Default::default()
        };
        let plain = tokens(&model, &long, options.clone());
        let mut g = speculative_generate(
            &model,
            &long,
            exact(options),
            MtpDrafter::new(&model).unwrap(),
            DraftPolicy::Fixed(3),
        )
        .unwrap();
        let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
        assert_eq!(ids, plain, "{name}: long context");
        assert!(g.stats().verifies > 0);
        let _ = std::fs::remove_file(&file);
    }
    // A quantized model quantizes its MTP layer as the body: 8-bit routers, 4-bit others.
    let (dir, ..) = fixture("qwen3_5_moe_q4");
    let mut model = load_model(&dir).unwrap();
    model.config.mtp_layers = 1;
    let file = mtp_file("qwen3_5_moe_f32");
    model.attach_mtp(&file).unwrap();
    let shape = |path: &str| {
        model
            .parameters()
            .into_iter()
            .find(|(k, _)| k == path)
            .map(|(_, v)| v.shape().to_vec())
            .unwrap_or_else(|| panic!("{path}"))
    };
    // Packed uint32 words per row: the router's 64 inputs at 8 bits (16), the output
    // projection's and `fc`'s 128 inputs at 4 bits (16; 32 at 8 bits).
    assert_eq!(shape("mtp.layers.0.mlp.gate.weight"), [16, 16]);
    assert_eq!(shape("mtp.layers.0.self_attn.o_proj.weight"), [64, 16]);
    assert_eq!(shape("mtp.fc.weight"), [64, 16]);
    // A tensor of another shape: an error naming it.
    let (dir, ..) = fixture("qwen3_5_f32");
    let mut model = load_model(&dir).unwrap();
    model.config.mtp_layers = 1;
    let bad =
        std::env::temp_dir().join(format!("nuvec-mtp-bad-{}.safetensors", std::process::id()));
    let fc = nuvec::ops::zeros([64, 100], nuvec::DType::F32);
    nuvec::io::save_safetensors(&bad, &[("mtp.fc.weight", &fc)], &[]).unwrap();
    let err = model.attach_mtp(&bad).unwrap_err().to_string();
    assert!(err.contains("mtp.") || err.contains("fc"), "{err}");
    let _ = std::fs::remove_file(&bad);
    let _ = std::fs::remove_file(&file);
}

/// Sampled speculative decoding keeps plain sampling's distribution: over 600 seeds, each
/// of the first three tokens' frequencies agree with plain sampling's (drafts always of the
/// greedy continuation, so most steps verify); a seed repeats a run.
#[test]
fn sampled_speculation_keeps_the_sampling_distribution() {
    use nuvec_lm::{Draft, DraftPolicy, Drafter, Model, Sampler, speculative_generate};
    if !gpu() {
        return;
    }
    struct Greedy(Vec<u32>, usize);
    impl Drafter for Greedy {
        fn draft(&mut self, _: &Model, context: &[u32], max: usize) -> nuvec::Result<Draft> {
            let done = context.len() - self.1;
            let d: Vec<u32> = self.0.iter().skip(done).take(max).copied().collect();
            Ok(Draft::from_tokens(&d))
        }
    }
    let (dir, prompt, _, _, greedy) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let sampler = Sampler {
        temperature: 0.7,
        top_k: 6,
        ..Default::default()
    };
    let options = |seed| GenerateOptions {
        max_tokens: 3,
        sampler,
        seed: Some(seed),
        ..Default::default()
    };
    let runs = 600u64;
    let mut plain = vec![std::collections::HashMap::<u32, usize>::new(); 3];
    let mut spec = plain.clone();
    let mut verifies = 0;
    for seed in 0..runs {
        for (i, t) in tokens(&model, &prompt, options(seed))
            .into_iter()
            .enumerate()
        {
            *plain[i].entry(t).or_default() += 1;
        }
        let mut g = speculative_generate(
            &model,
            &prompt,
            exact(options(1000 + seed)),
            Greedy(greedy.clone(), prompt.len()),
            DraftPolicy::Fixed(2),
        )
        .unwrap();
        let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
        verifies += g.stats().verifies;
        for (i, t) in ids.into_iter().enumerate() {
            *spec[i].entry(t).or_default() += 1;
        }
    }
    assert!(verifies > runs as usize / 2, "{verifies} verifications");
    for pos in 0..3 {
        let keys: std::collections::BTreeSet<u32> =
            plain[pos].keys().chain(spec[pos].keys()).copied().collect();
        for t in keys {
            let (a, b) = (
                plain[pos].get(&t).copied().unwrap_or(0) as f64 / runs as f64,
                spec[pos].get(&t).copied().unwrap_or(0) as f64 / runs as f64,
            );
            let p = (a + b) / 2.0;
            let se = (2.0 * p * (1.0 - p) / runs as f64).sqrt();
            assert!(
                (a - b).abs() < 5.0 * se + 0.01,
                "position {pos}, token {t}: {a} vs {b}"
            );
        }
    }
    // A seed repeats a run.
    let run = || {
        speculative_generate(
            &model,
            &prompt,
            exact(options(7)),
            Greedy(greedy.clone(), prompt.len()),
            DraftPolicy::Fixed(2),
        )
        .unwrap()
        .collect::<nuvec::Result<Vec<u32>>>()
        .unwrap()
    };
    assert_eq!(run(), run());
}

/// In full: over 20,000 seeded runs (temperature 0.8, top-p 0.9, top-k 20), each of
/// the first three tokens' frequencies under sampled speculation pass a chi-square test of
/// homogeneity against plain sampling at significance 0.001, with drafts of the greedy
/// continuation (most steps verify) and with the MTP head. A few minutes: run with `just
/// spec-chi-square`.
#[test]
#[ignore = "20,000 runs per drafter: `just spec-chi-square`"]
fn sampled_speculation_passes_a_chi_square_test() {
    use nuvec_lm::{Draft, DraftPolicy, Drafter, Model, MtpDrafter, Sampler, speculative_generate};
    if !gpu() {
        return;
    }
    struct Greedy(Vec<u32>, usize);
    impl Drafter for Greedy {
        fn draft(&mut self, _: &Model, context: &[u32], max: usize) -> nuvec::Result<Draft> {
            let done = context.len() - self.1;
            let d: Vec<u32> = self.0.iter().skip(done).take(max).copied().collect();
            Ok(Draft::from_tokens(&d))
        }
    }
    /// The chi-square statistic of a 2 × k table (categories with fewer than 5 expected
    /// counts pooled) and its degrees of freedom.
    fn chi_square(a: &[usize], b: &[usize]) -> (f64, usize) {
        let (na, nb) = (
            a.iter().sum::<usize>() as f64,
            b.iter().sum::<usize>() as f64,
        );
        let mut cells: Vec<(f64, f64)> = Vec::new();
        let mut pooled = (0.0, 0.0);
        for (&x, &y) in a.iter().zip(b) {
            let total = (x + y) as f64;
            if total * na.min(nb) / (na + nb) < 5.0 {
                pooled = (pooled.0 + x as f64, pooled.1 + y as f64);
            } else {
                cells.push((x as f64, y as f64));
            }
        }
        if pooled.0 + pooled.1 > 0.0 {
            cells.push(pooled);
        }
        let stat = cells
            .iter()
            .map(|&(x, y)| {
                let t = x + y;
                let (ea, eb) = (t * na / (na + nb), t * nb / (na + nb));
                (x - ea).powi(2) / ea + (y - eb).powi(2) / eb
            })
            .sum();
        (stat, cells.len().saturating_sub(1))
    }
    /// The chi-square distribution's upper 0.001 quantile (Wilson–Hilferty).
    fn critical(df: usize) -> f64 {
        let d = df.max(1) as f64;
        let z = 3.0902;
        d * (1.0 - 2.0 / (9.0 * d) + z * (2.0 / (9.0 * d)).sqrt()).powi(3)
    }
    let runs = 20_000u64;
    let sampler = Sampler {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 20,
        ..Default::default()
    };
    let options = |seed| GenerateOptions {
        max_tokens: 3,
        sampler,
        seed: Some(seed),
        ..Default::default()
    };
    for name in ["qwen3_f32", "qwen3_5_f32"] {
        let (dir, prompt, _, _, greedy) = fixture(name);
        let mut model = load_model(&dir).unwrap();
        let mtp = name == "qwen3_5_f32";
        if mtp {
            model.config.mtp_layers = 1;
            model.attach_mtp(dir.join("mtp.safetensors")).unwrap();
        }
        let vocab = model.config.vocab_size;
        let mut plain = vec![vec![0usize; vocab]; 3];
        let mut spec = plain.clone();
        let mut verifies = 0;
        for seed in 0..runs {
            for (i, t) in tokens(&model, &prompt, options(seed))
                .into_iter()
                .enumerate()
            {
                plain[i][t as usize] += 1;
            }
            let drafter: Box<dyn Drafter> = match mtp {
                true => Box::new(MtpDrafter::new(&model).unwrap()),
                false => Box::new(Greedy(greedy.clone(), prompt.len())),
            };
            let mut g = speculative_generate(
                &model,
                &prompt,
                exact(options(runs + seed)),
                drafter,
                DraftPolicy::Fixed(2),
            )
            .unwrap();
            let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
            verifies += g.stats().verifies;
            for (i, t) in ids.into_iter().enumerate() {
                spec[i][t as usize] += 1;
            }
        }
        assert!(
            verifies > runs as usize / 2,
            "{name}: {verifies} verifications"
        );
        for pos in 0..3 {
            let (stat, df) = chi_square(&plain[pos], &spec[pos]);
            eprintln!(
                "{name} position {pos}: chi-square {stat:.1}, {df} df, critical {:.1}",
                critical(df)
            );
            assert!(
                stat < critical(df),
                "{name} position {pos}: {stat} with {df} df"
            );
        }
    }
}

/// SuffixDecoding keeps the greedy output and drafts from an earlier sequence it was told
/// about.
#[test]
fn suffix_decoding_keeps_the_greedy_output() {
    use nuvec_lm::{DraftPolicy, SuffixDecoding, speculative_generate};
    if !gpu() {
        return;
    }
    let (dir, prompt, _, _, greedy) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let options = GenerateOptions {
        max_tokens: greedy.len(),
        ..Default::default()
    };
    let mut suffix = SuffixDecoding::default();
    let run = |suffix: &mut SuffixDecoding| {
        let mut g = speculative_generate(
            &model,
            &prompt,
            exact(options.clone()),
            &mut *suffix,
            DraftPolicy::Fixed(4),
        )
        .unwrap();
        let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
        (ids, g.stats())
    };
    let (ids, _) = run(&mut suffix);
    assert_eq!(ids, greedy);
    // The same request again: the stored output drafts it, mostly accepted.
    let mut seq = prompt.clone();
    seq.extend(&ids);
    suffix.remember(&seq);
    let (ids, stats) = run(&mut suffix);
    assert_eq!(ids, greedy);
    assert!(stats.accepted * 2 > greedy.len(), "{stats:?}");
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
    use nuvec_lm::{Batcher, BatcherOptions, Config, Event, FinishReason};
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
        assert_eq!(
            tokens(&model, &prompt, options.clone()),
            greedy[..k],
            "{edit:?}"
        );
        // The batcher too, finishing with `EndOfSequence`.
        let mut batcher = Batcher::new(
            &model,
            BatcherOptions {
                blocks: Some(16),
                block_size: 8,
                max_batch: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let exact = GenerateOptions {
            row_exact: true,
            ..options.clone()
        };
        batcher.submit(prompt.clone(), exact).unwrap();
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
    use nuvec_lm::{Batcher, BatcherOptions, PromptLookup, Sampler, speculative_generate};
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
    let mut batcher = Batcher::new(
        &model,
        BatcherOptions {
            blocks: Some(16),
            block_size: 8,
            max_batch: 2,
            ..Default::default()
        },
    )
    .unwrap();
    for sampler in invalid {
        let options = GenerateOptions {
            max_tokens: 4,
            sampler,
            ..Default::default()
        };
        assert!(sampler.validate().is_err(), "{sampler:?}");
        assert!(
            generate(&model, &prompt, options.clone()).is_err(),
            "{sampler:?}"
        );
        assert!(
            speculative_generate(
                &model,
                &prompt,
                exact(options.clone()),
                PromptLookup::default(),
                Default::default()
            )
            .is_err(),
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
    // Modes, Hugging Face `quantization_config` mappings (mlx-lm 0.32.2) and per-layer entries.
    use nuvec::{Quant, QuantMode};
    let quant = |extra: &str| {
        parse(&format!("{{{base}, \"num_attention_heads\": 4, {extra}}}"))
            .map(|c| c.quantization.unwrap())
    };
    let q = quant(r#""quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4",
        "model.layers.0.mlp.gate": {"group_size": 64, "bits": 8, "mode": "affine"},
        "model.layers.1.mlp.switch_mlp.up_proj": {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": true},
        "lm_head": false}"#)
    .unwrap();
    assert_eq!(q.default, Quant::new(QuantMode::Mxfp4));
    let gate = q.for_layer("model.layers.0.mlp.gate").unwrap();
    assert_eq!(gate.quant, Some(Quant::affine(64, 8)));
    let up = q
        .for_layer("model.layers.1.mlp.switch_mlp.up_proj")
        .unwrap();
    assert_eq!(
        (up.quant, up.global_scale),
        (Some(Quant::new(QuantMode::Nvfp4)), true)
    );
    assert_eq!(q.for_layer("lm_head").unwrap().quant, None);
    // Hugging Face packings: AWQ and GPTQ convert to affine (bits and group size from the
    // config); formats needing a per-architecture tensor converter are refused.
    use nuvec_lm::PackedFormat;
    for (config, packed, want) in [
        (
            r#""quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemm"}"#,
            PackedFormat::Awq,
            Quant::affine(128, 4),
        ),
        (
            r#""quantization_config": {"quant_method": "gptq", "bits": 8, "group_size": 32}"#,
            PackedFormat::Gptq { v1: true },
            Quant::affine(32, 8),
        ),
        (
            r#""quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 64, "checkpoint_format": "gptq_v2"}"#,
            PackedFormat::Gptq { v1: false },
            Quant::affine(64, 4),
        ),
    ] {
        let q = quant(config).unwrap();
        assert_eq!((q.default, q.packed), (want, Some(packed)), "{config}");
    }
    // A multimodal config keeps the text model's settings, quantization included, in
    // `text_config`.
    let nested = format!(
        "{{\"model_type\": \"qwen3\", \"text_config\": {{{base}, \"num_attention_heads\": 4, \
         \"quantization_config\": {{\"quant_method\": \"awq\", \"group_size\": 64}}}}}}"
    );
    let nested = parse(&nested).unwrap().quantization.unwrap();
    assert_eq!(nested.packed, Some(PackedFormat::Awq));
    for bad in [
        r#""quantization": {"group_size": 32, "bits": 4, "mode": "fp4"}"#,
        r#""quantization_config": {"quant_method": "mxfp4"}"#,
        r#""quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}"#,
        r#""quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}"#,
        r#""quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}"#,
        r#""quantization_config": {"quant_method": "awq", "version": "gemv"}"#,
        r#""quantization_config": {"quant_method": "gptq", "bits": 3}"#,
        r#""quantization_config": {"quant_method": "gptq", "desc_act": true}"#,
        r#""quantization_config": {"quant_method": "gptq", "group_size": -1}"#,
        r#""quantization_config": {"quant_method": "bitsandbytes"}"#,
    ] {
        assert!(quant(bad).is_err(), "{bad}");
    }
    let err = quant(r#""quantization_config": {"quant_method": "mxfp4"}"#).unwrap_err();
    assert!(err.to_string().contains("mlx_lm.convert"), "{err}");
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

/// The calling thread's CPU time, in seconds.
fn thread_cpu_seconds() -> f64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `t` is a valid `timespec` for the call to write.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) };
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
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
        // Each run with an empty word cache (a clone's), as a first encoding. Timed as this
        // thread's CPU time: tests running beside it do not stretch it as they stretch the
        // wall clock.
        let best = |t: &str| {
            (0..3)
                .map(|_| {
                    let fresh = tokenizer.clone();
                    let start = thread_cpu_seconds();
                    std::hint::black_box(fresh.encode_text(t));
                    thread_cpu_seconds() - start
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
    use nuvec_lm::{Draft, DraftPolicy, Drafter, Model, speculative_generate};
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
        fn draft(&mut self, _: &Model, context: &[u32], max: usize) -> nuvec::Result<Draft> {
            let done = context.len() - self.prompt;
            let mut d: Vec<u32> = self.tokens.iter().skip(done).take(max).copied().collect();
            d.extend(std::iter::repeat_n(7, self.extra));
            Ok(Draft::from_tokens(&d))
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
        let mut g = speculative_generate(
            &model,
            &prompt,
            exact(options(max)),
            drafter,
            DraftPolicy::Fixed(4),
        )
        .unwrap();
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
        exact(options(4)),
        Oracle {
            prompt: 1,
            tokens: Vec::new(),
            extra: 3,
        },
        DraftPolicy::Fixed(4),
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
        exact(options(8)),
        Oracle {
            prompt: prompt.len(),
            tokens: Vec::new(),
            extra: 0,
        },
        DraftPolicy::Fixed(4),
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

    /// Without keys, every linear layer of the selected blocks (mlx-lm's default), expert
    /// layers left as they are; naming an expert layer is an error naming its type, and so
    /// is a layer that is LoRA already; on an error nothing changes.
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
        // One layer LoRA already, the others not: an error, and none converted.
        let mut partly = load_model(&dir).unwrap();
        lora::linear_to_lora_layers(
            &mut partly,
            1,
            Some(&["self_attn.q_proj"]),
            LoraParams::default(),
        )
        .unwrap();
        let before = names(&partly);
        assert!(lora::linear_to_lora_layers(&mut partly, 1, None, LoraParams::default()).is_err());
        assert_eq!(names(&partly), before, "nothing changed");
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
        // Without keys, the attention, router and shared projections; experts untouched.
        let mut all = load_model(&dir).unwrap();
        assert!(lora::linear_to_lora_layers(&mut all, 1, None, LoraParams::default()).unwrap() > 0);
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
        // An unknown fine-tuning type is an error.
        let bad = std::env::temp_dir().join(format!("nuvec-lm-{}-prefix", std::process::id()));
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("adapter_config.json"), r#"{"fine_tune_type": "prefix", "num_layers": 1, "lora_parameters": {"rank": 2, "scale": 1.0}}"#).unwrap();
        let (dir, ..) = fixture("qwen3_f32");
        let mut model = load_model(&dir).unwrap();
        let err = lora::load_adapters(&bad, &mut model)
            .unwrap_err()
            .to_string();
        assert!(err.contains("prefix"), "{err}");
        let config = AdapterConfig::new(2, LoraParams::default()).keys(["self_attn.q_proj"]);
        assert_eq!(
            config.keys.as_deref(),
            Some(&["self_attn.q_proj".to_owned()][..])
        );
    }
}

/// mrun's B5: on the CPU, a verify of 2–4 draft tokens under `with_row_exact` gives each
/// position the logits of serial decoding, and the step after it too, so speculative
/// decoding on the CPU equals plain decoding (dense, mixture-of-experts and hybrid models).
#[test]
fn cpu_verify_rows_equal_serial_decoding() {
    let bits = |a: &Array| a.view(DType::U32).to_vec::<u32>().unwrap();
    nuvec::with_stream(Device::cpu(), || {
        nuvec::with_row_exact(|| {
            for name in [
                "qwen3_f32",
                "qwen3_moe_f32",
                "qwen3_5_f32",
                "qwen3_5_moe_f32",
            ] {
                let (dir, prompt, ..) = fixture(name);
                let model = load_model(&dir).unwrap();
                let drafts: Vec<u32> = prompt.iter().rev().take(5).copied().collect();
                let head = Array::from_slice(&prompt, [1, prompt.len()]);
                for rows in [2, 4] {
                    // Serial: one token per step.
                    let mut kv = model.new_cache();
                    model.forward(&head, &mut kv).eval().unwrap();
                    let serial: Vec<Vec<u32>> = drafts[..=rows]
                        .iter()
                        .map(|&t| bits(&model.forward(&Array::from_slice(&[t], [1, 1]), &mut kv)))
                        .collect();
                    // Verify `rows` tokens in one pass, then one step.
                    let mut kv = model.new_cache();
                    model.forward(&head, &mut kv).eval().unwrap();
                    let block = Array::from_slice(&drafts[..rows], [1, rows]);
                    let verified = model.forward(&block, &mut kv);
                    for (r, want) in serial[..rows].iter().enumerate() {
                        let row = verified.i((.., r as i32..r as i32 + 1));
                        assert!(bits(&row) == *want, "{name}: verify row {r} of {rows}");
                    }
                    let next = Array::from_slice(&drafts[rows..=rows], [1, 1]);
                    assert!(
                        bits(&model.forward(&next, &mut kv)) == serial[rows],
                        "{name}: the step after a {rows}-row verify"
                    );
                }
            }
        });
    });
}

/// `fast::ssm_update` and `fast::ssm_scan` against mlx-lm's Mamba-2 kernel and `ssm_attn`:
/// bit for bit on the GPU, within float32 rounding on the CPU.
#[test]
fn ssm_update_equals_mlx_lm() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ssm.safetensors");
    let f = nuvec::io::load_safetensors(path).unwrap().tensors;
    let bits = |a: &Array| {
        a.astype(DType::F32)
            .view(DType::U32)
            .to_vec::<u32>()
            .unwrap()
    };
    let close = |a: &Array, b: &Array, tol: f32| {
        let (a, b) = (
            a.astype(DType::F32).to_vec::<f32>().unwrap(),
            b.astype(DType::F32).to_vec::<f32>().unwrap(),
        );
        a.iter()
            .zip(&b)
            .all(|(x, y)| (x - y).abs() <= tol * y.abs().max(1.0))
    };
    for case in ["f32", "bf16", "f16"] {
        let t = |k: &str| f[&format!("{case}.{k}")].clone();
        let run = || {
            nuvec::fast::ssm_update(
                &t("x"),
                &t("a_log"),
                &t("b"),
                &t("c"),
                &t("d"),
                &t("dt_after"),
                &t("state"),
            )
        };
        // Bit for bit on the GPU the fixture was recorded on; other generations round
        // `fast::exp` and the SIMD sum differently.
        if recorded_gpu() {
            let (y, s) = run();
            assert!(bits(&y) == bits(&t("y")), "{case}: GPU output");
            assert!(bits(&s) == bits(&t("next")), "{case}: GPU state");
        } else if gpu() {
            let (y, s) = run();
            let tol = if case == "f32" { 1e-5 } else { 1e-2 };
            assert!(close(&y, &t("y"), tol), "{case}: GPU output");
            assert!(close(&s, &t("next"), tol), "{case}: GPU state");
        }
        let (y, s) = nuvec::with_stream(Device::cpu(), run);
        let tol = if case == "f32" { 1e-5 } else { 1e-2 };
        assert!(close(&y, &t("y"), tol), "{case}: CPU output");
        assert!(close(&s, &t("next"), tol), "{case}: CPU state");
    }
    // Strided inputs (offset slices of wider arrays) give the same bits.
    {
        let t = |k: &str| f[&format!("f32.{k}")].clone();
        let wide = |a: &Array| nuvec::ops::concatenate(&[a.clone(), a.clone()], -1);
        let half = |a: &Array, n: i32| wide(a).i((.., .., .., n..2 * n));
        let (x, b, c) = (t("x"), t("b"), t("c"));
        let (dh, ds) = (x.shape()[3], b.shape()[3]);
        let devices = [Some(Device::cpu()), gpu().then(|| Device::gpu(0))];
        for device in devices.into_iter().flatten() {
            nuvec::with_stream(device, || {
                let (y, s) = nuvec::fast::ssm_update(
                    &x,
                    &t("a_log"),
                    &b,
                    &c,
                    &t("d"),
                    &t("dt_after"),
                    &t("state"),
                );
                let (y2, s2) = nuvec::fast::ssm_update(
                    &half(&x, dh),
                    &t("a_log"),
                    &half(&b, ds),
                    &half(&c, ds),
                    &t("d"),
                    &t("dt_after"),
                    &t("state").swapaxes(2, 3).swapaxes(2, 3),
                );
                assert!(
                    bits(&y) == bits(&y2) && bits(&s) == bits(&s2),
                    "strided on {device:?}"
                );
            });
        }
    }
    // Prompts (mlx-lm's `ssm_attn`, 300 tokens in two chunks, and 37 in bfloat16).
    for case in ["prompt", "prompt_bf16"] {
        let t = |k: &str| f[&format!("{case}.{k}")].clone();
        let state = t("state");
        let run = || {
            nuvec::fast::ssm_scan(
                &t("x"),
                &t("a_log"),
                &t("b"),
                &t("c"),
                &t("d"),
                &t("dt_after"),
                Some(&state),
            )
        };
        // Bit for bit on the GPU the fixture was recorded on; other generations round
        // `fast::exp` and the SIMD sum differently.
        if recorded_gpu() {
            let (y, s) = run();
            assert!(bits(&y) == bits(&t("y")), "{case}: GPU output");
            assert!(bits(&s) == bits(&t("next")), "{case}: GPU state");
        } else if gpu() {
            let (y, s) = run();
            assert!(close(&y, &t("y"), 1e-3), "{case}: GPU output");
            assert!(close(&s, &t("next"), 1e-3), "{case}: GPU state");
        }
        let (y, s) = nuvec::with_stream(Device::cpu(), run);
        assert!(close(&y, &t("y"), 1e-3), "{case}: CPU output");
        assert!(close(&s, &t("next"), 1e-3), "{case}: CPU state");
    }
}

/// mlx-lm's `save_prompt_cache` files load and continue to mlx-lm's next logits, and a
/// cache nuvec saves loads back (plain, hybrid and sliding-window YaRN models).
#[test]
fn prompt_caches_load_mlx_lm_files_and_round_trip() {
    use nuvec_lm::{load_prompt_cache, save_prompt_cache};
    let bits = |a: &Array| a.view(DType::U32).to_vec::<u32>().unwrap();
    let worst = |a: &Array, b: &Array| {
        let (a, b) = (
            a.astype(DType::F32).to_vec::<f32>().unwrap(),
            b.to_vec::<f32>().unwrap(),
        );
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let dir = std::env::temp_dir().join(format!("nuvec-prompt-cache-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for name in [
        "qwen3_f32",
        "qwen3_5_f32",
        "gpt_oss_f32",
        "granite4_moe_f32",
        "lfm2_f32",
        "lfm2_moe_f32",
    ] {
        let (path, prompt, ..) = fixture(name);
        let model = load_model(&path).unwrap();
        let next = nuvec::io::load_safetensors(path.join("prompt_cache_next.safetensors"))
            .unwrap()
            .tensors;
        let token = &next["token"];
        // mlx-lm's file.
        let (mut caches, meta) =
            load_prompt_cache(path.join("prompt_cache.safetensors"), &model).unwrap();
        assert_eq!(meta.get("model").map(String::as_str), Some(name));
        let d = worst(&model.forward(token, &mut caches), &next["logits"]);
        assert!(d < 2e-4, "{name}: mlx-lm's cache continues {d} away");
        // nuvec's own file.
        let mut kv = model.new_cache();
        let head = Array::from_slice(&prompt, [1, prompt.len()]);
        model.forward(&head, &mut kv).eval().unwrap();
        let file = dir.join(format!("{name}.safetensors"));
        let meta = std::collections::BTreeMap::from([("model".to_string(), name.to_string())]);
        save_prompt_cache(&file, &model, &kv, &meta).unwrap();
        let (mut loaded, _) = load_prompt_cache(&file, &model).unwrap();
        let (want, got) = (
            model.forward(token, &mut kv),
            model.forward(token, &mut loaded),
        );
        if name == "gpt_oss_f32" {
            // Keys scaled by YaRN's mscale and back: one rounding each way.
            let d = worst(&got, &want);
            assert!(d < 1e-5, "{name}: round trip {d} away");
        } else {
            assert!(bits(&got) == bits(&want), "{name}: round trip");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Lanes of independent sequences, a decode step built inside `with_row_exact` and evaluated
/// outside it (as a server samples), give each lane its solo decode's logits bit for bit for
/// 10 steps: dense, and mixtures of experts with and without a shared expert. (These tiny
/// shapes do not separate the kernels; `row_independence.rs` `built_inside` does.)
#[test]
fn lanes_built_in_row_exact_equal_solo_decoding() {
    use nuvec_lm::{PagedCache, decode_batch};
    if !gpu() {
        return;
    }
    let bits = |a: &Array| {
        a.astype(DType::F32)
            .view(DType::U32)
            .to_vec::<u32>()
            .unwrap()
    };
    for name in [
        "qwen3_5_q4",
        "qwen3_5_moe_q4",
        "qwen3_moe_q4",
        "gemma4_moe_q4",
    ] {
        let (dir, prompt, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        for lanes in [2, 3, 4] {
            let mut cache = PagedCache::new(&model, 64, 8, lanes).unwrap();
            let (mut solo, mut ids, mut tokens) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..lanes {
                let p: Vec<u32> = prompt
                    .iter()
                    .cycle()
                    .skip(i)
                    .take(prompt.len() + i)
                    .copied()
                    .collect();
                let mut kv = model.new_cache();
                let head = Array::from_slice(&p[..p.len() - 1], [1, p.len() - 1]);
                let _ = model.forward(&head, &mut kv).check().unwrap();
                ids.push(cache.insert(&kv).unwrap());
                solo.push(kv);
                tokens.push(*p.last().unwrap());
            }
            for step in 0..10 {
                let logits =
                    nuvec::with_row_exact(|| decode_batch(&model, &tokens, &ids, &mut cache))
                        .unwrap();
                let logits = bits(&logits);
                let vocab = logits.len() / lanes;
                for (r, kv) in solo.iter_mut().enumerate() {
                    let x = Array::from_slice(&[tokens[r]], [1, 1]);
                    let one = nuvec::with_row_exact(|| model.forward(&x, kv).i((.., -1)));
                    assert!(
                        logits[r * vocab..(r + 1) * vocab] == bits(&one)[..],
                        "{name}, {lanes} lanes, step {step}: lane {r} differs from its solo decode"
                    );
                    // A varied next token per lane.
                    tokens[r] = (one.argmax(-1).item::<u32>().unwrap() + step) % 100;
                }
            }
        }
    }
}

/// mlx-lm's quantized KV caches (`kv_bits`, `kv_group_size`, `quantized_kv_start`), quantized
/// from the first step or mid-way, on dense, mixture-of-experts and hybrid models: greedy
/// generation gives mlx-lm's tokens (as far as it does with a full-precision cache), and each step's log-probabilities are within 1e-3 of
/// mlx-lm's in float32, and in half precision no farther from them on average than with a
/// full-precision cache (nuvec's half-precision forward pass is held close, not equal). The
/// attention itself is bit-exact (`quantized_attention_equals_mlx_lm`).
#[test]
fn kv_quant_equals_mlx_lm() {
    use nuvec_lm::KvQuant;
    if !gpu() {
        return;
    }
    for name in ["qwen3_5_f32", "qwen3_5_q4", "qwen3_5_moe_q4"] {
        let (dir, prompt, _, _, greedy) = fixture(name);
        let model = load_model(&dir).unwrap();
        // Where nuvec's own greedy decoding (full-precision cache) leaves mlx-lm's: a mixture
        // of experts in half precision can route another expert after an ulp near a tie.
        let plain = GenerateOptions {
            max_tokens: greedy.len(),
            ..Default::default()
        };
        let same: Vec<u32> = generate(&model, &prompt, plain)
            .unwrap()
            .collect::<nuvec::Result<_>>()
            .unwrap();
        let agree = same.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
        let want = nuvec::io::load_safetensors(dir.join("kv_quant.safetensors"))
            .unwrap()
            .tensors;
        // mlx-lm's `generate_step`: the prompt but its last token, then one token per step.
        let run = |quant: Option<KvQuant>, tokens: &[u32]| {
            let mut cache = model.new_cache();
            if let Some(q) = quant {
                for c in &mut cache {
                    c.quantize_at(q).unwrap();
                }
            }
            let head = Array::from_slice(&prompt[..prompt.len() - 1], [1, prompt.len() - 1]);
            let _ = model.forward(&head, &mut cache).check().unwrap();
            let mut token = *prompt.last().unwrap();
            let mut steps = Vec::new();
            for &t in tokens {
                let logits = model.forward(&Array::from_slice(&[token], [1, 1]), &mut cache);
                let logits = logits.i((0, -1));
                let lp = (&logits - logits.logsumexp_keepdims(-1)).astype(DType::F32);
                steps.push(lp.to_vec::<f32>().unwrap());
                token = t;
            }
            (steps, cache.iter().any(|c| c.is_quantized()))
        };
        let distance = |got: &[f32], want: &Array| -> (f32, f32) {
            let want = want.to_vec::<f32>().unwrap();
            let d: Vec<f32> = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).collect();
            let max = d.iter().fold(0.0f32, |m, &x| m.max(x));
            (max, d.iter().sum::<f32>() / d.len() as f32)
        };
        for (bits, group_size, start) in [(8, 32, 0), (4, 32, 0), (8, 32, prompt.len() + 3)] {
            let key = format!("b{bits}_g{group_size}_s{start}");
            let tokens = want[&format!("{key}_tokens")].to_vec::<u32>().unwrap();
            let logprobs = &want[&format!("{key}_logprobs")];
            let quant = KvQuant {
                bits,
                group_size,
                start,
                rotate: false,
            };
            let (quantized, converted) = run(Some(quant), &tokens);
            assert!(converted, "{name} {key}: not quantized");
            let (dense, _) = run(None, &tokens);
            for step in 0..tokens.len() {
                let (max, mean) = distance(&quantized[step], &logprobs.i(step as i32));
                if name.ends_with("_f32") {
                    assert!(max <= 1e-3, "{name} {key}: step {step} {max} from mlx-lm");
                } else {
                    let (_, base) = distance(&dense[step], &want["dense_logprobs"].i(step as i32));
                    assert!(
                        mean <= 2.0 * base + 1e-3,
                        "{name} {key}: step {step} {mean} from mlx-lm on average (dense {base})"
                    );
                }
            }
            let options = GenerateOptions {
                max_tokens: tokens.len(),
                kv_quant: Some(quant),
                ..Default::default()
            };
            let ids: Vec<u32> = generate(&model, &prompt, options)
                .unwrap()
                .collect::<nuvec::Result<_>>()
                .unwrap();
            let n = agree.min(ids.len());
            assert_eq!(ids[..n], tokens[..n], "{name} {key}: greedy tokens");
        }
    }
}

/// An 8-bit cache in groups of 32 takes 1 byte per value plus a float32 scale and bias per
/// 32 values: the attention layers' bytes are 0.3125× the float32 cache's.
#[test]
fn kv_quant_cache_bytes() {
    use nuvec_lm::KvQuant;
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("qwen3_5_f32");
    let model = load_model(&dir).unwrap();
    let ids = Array::from_slice(&prompt, [1, prompt.len()]);
    let mut dense = model.new_cache();
    model.forward(&ids, &mut dense).eval().unwrap();
    let mut quantized = dense.clone();
    let quant = KvQuant {
        group_size: 32,
        ..Default::default()
    };
    for c in &mut quantized {
        c.to_quantized(quant).unwrap();
    }
    let (mut q, mut d) = (0, 0);
    for (qc, dc) in quantized.iter().zip(&dense) {
        if qc.is_quantized() {
            q += qc.nbytes();
            d += dc.nbytes();
        }
    }
    assert!(
        q > 0 && q * 10_000 == d * 3125,
        "quantized {q} bytes against {d}"
    );
}

/// Invalid KV quantizations are refused: bits and groups MLX does not quantize with, head
/// dimensions that are not multiples of the group, and the paged `Batcher`.
#[test]
fn kv_quant_refusals() {
    use nuvec_lm::{Batcher, BatcherOptions, KvQuant};
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let options = |bits, group_size| GenerateOptions {
        max_tokens: 2,
        kv_quant: Some(KvQuant {
            bits,
            group_size,
            start: 0,
            rotate: false,
        }),
        ..Default::default()
    };
    assert!(generate(&model, &prompt, options(7, 32)).is_err(), "7 bits");
    assert!(
        generate(&model, &prompt, options(8, 16)).is_err(),
        "groups of 16"
    );
    // Head dimension 16: not a multiple of 32.
    let ids: nuvec::Result<Vec<u32>> =
        generate(&model, &prompt, options(8, 32)).and_then(|g| g.collect());
    assert!(ids.is_err(), "head dimension 16");
    let mut batcher = Batcher::new(&model, BatcherOptions::default()).unwrap();
    assert!(
        batcher.submit(prompt.clone(), options(8, 32)).is_err(),
        "batcher"
    );
}

/// Speculative decoding with a quantized cache gives plain greedy generation's tokens with
/// the same cache: rejected drafts trim the quantized positions.
#[test]
fn kv_quant_speculative_equals_greedy() {
    use nuvec_lm::{DraftPolicy, KvQuant, PromptLookup, speculative_generate};
    if !gpu() {
        return;
    }
    for name in ["qwen3_5_f32", "qwen3_5_moe_q4"] {
        let (dir, prompt, ..) = fixture(name);
        let model = load_model(&dir).unwrap();
        // A repetitive prompt, so prompt lookup drafts.
        let prompt: Vec<u32> = prompt
            .iter()
            .cycle()
            .take(prompt.len() * 3)
            .copied()
            .collect();
        let options = GenerateOptions {
            max_tokens: 24,
            kv_quant: Some(KvQuant {
                group_size: 32,
                start: 0,
                ..Default::default()
            }),
            ..Default::default()
        };
        let plain: Vec<u32> = generate(&model, &prompt, options.clone())
            .unwrap()
            .collect::<nuvec::Result<_>>()
            .unwrap();
        let mut g = speculative_generate(
            &model,
            &prompt,
            exact(options),
            Box::new(PromptLookup::default()),
            DraftPolicy::Fixed(4),
        )
        .unwrap();
        let spec: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
        assert_eq!(spec, plain, "{name}");
        assert!(g.stats().drafted > 0, "{name}: no drafts");
    }
}

/// mlx-lm's prompt cache with quantized attention layers (`QuantizedKVCache`, a hybrid model)
/// loads and continues to mlx-lm's next logits; nuvec's saved quantized cache loads back.
#[test]
fn kv_quant_prompt_cache_round_trips() {
    use nuvec_lm::{KvQuant, load_prompt_cache, save_prompt_cache};
    if !gpu() {
        return;
    }
    let (path, prompt, ..) = fixture("qwen3_5_f32");
    let model = load_model(&path).unwrap();
    let next = nuvec::io::load_safetensors(path.join("prompt_cache_q8_next.safetensors"))
        .unwrap()
        .tensors;
    let token = &next["token"];
    let (mut caches, _) =
        load_prompt_cache(path.join("prompt_cache_q8.safetensors"), &model).unwrap();
    assert!(caches.iter().any(|c| c.is_quantized()));
    let got = model.forward(token, &mut caches).astype(DType::F32);
    let (g, w) = (
        got.to_vec::<f32>().unwrap(),
        next["logits"].to_vec::<f32>().unwrap(),
    );
    let d = g
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(d < 2e-4, "mlx-lm's quantized cache continues {d} away");
    let mut kv = model.new_cache();
    let head = Array::from_slice(&prompt, [1, prompt.len()]);
    model.forward(&head, &mut kv).eval().unwrap();
    for c in &mut kv {
        c.to_quantized(KvQuant {
            group_size: 32,
            ..Default::default()
        })
        .unwrap();
    }
    let file =
        std::env::temp_dir().join(format!("nuvec-kv-quant-{}.safetensors", std::process::id()));
    save_prompt_cache(&file, &model, &kv, &Default::default()).unwrap();
    let (mut loaded, _) = load_prompt_cache(&file, &model).unwrap();
    std::fs::remove_file(&file).ok();
    let bits = |a: &Array| {
        a.astype(DType::F32)
            .view(DType::U32)
            .to_vec::<u32>()
            .unwrap()
    };
    assert!(bits(&model.forward(token, &mut loaded)) == bits(&model.forward(token, &mut kv)));
}

/// The largest difference between two float32 arrays, relative to the largest magnitude.
fn rel_diff(got: &nuvec::Array, want: &nuvec::Array) -> f32 {
    let (g, w) = (
        got.astype(DType::F32).to_vec::<f32>().unwrap(),
        want.astype(DType::F32).to_vec::<f32>().unwrap(),
    );
    assert_eq!(g.len(), w.len());
    let scale = w.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    g.iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
        / scale
}

/// DFlash drafters (z-lab's reference, vendored in `scripts/dflash_fixtures.py`): the
/// target's selected layer outputs over the prompt and the drafter's logits for the first
/// block `[anchor, mask, …]` and its draft equal the reference's, for full-attention and
/// sliding-window drafters, a 4-bit one, DFlash 2, a Gemma 4 target (soft cap) and a YaRN
/// drafter with attention biases (gpt-oss).
#[test]
fn dflash_matches_the_reference() {
    use nuvec_lm::{DFlashDrafter, Drafter};
    if !gpu() {
        return;
    }
    for (drafter_name, target_name, tol) in [
        ("dflash_qwen3", "qwen3_f32", 1e-4),
        ("dflash_qwen3_q4", "qwen3_f32", 1e-4),
        ("dflash_qwen3_5", "qwen3_5_f32", 1e-4),
        ("dflash_qwen3_v2", "qwen3_f32", 1e-4),
        ("dflash_gemma4", "gemma4_e_f32", 1e-4),
        ("dflash_gpt_oss", "gpt_oss_f32", 1e-4),
        ("dflash_qwen3_dspark", "qwen3_f32", 1e-4),
    ] {
        let (target_dir, prompt, ..) = fixture(target_name);
        let model = load_model(&target_dir).unwrap();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(drafter_name);
        let expected = nuvec::io::load_safetensors(dir.join("expected.safetensors"))
            .unwrap()
            .tensors;
        let mut drafter = DFlashDrafter::load(&dir, &model).unwrap();
        let ids = drafter.hidden_layers().unwrap();
        let tokens = nuvec::Array::from_slice(&prompt, [1, prompt.len() as i32]);
        let (_, features) = model.forward_layers(&tokens, &mut model.new_cache(), &ids);
        let d = rel_diff(&features, &expected["target_layers"]);
        assert!(d < tol, "{drafter_name}: target layers off by {d}");
        drafter.observe(&model, &tokens, Some(&features)).unwrap();
        let first = expected["first"].to_vec::<u32>().unwrap()[0];
        let mut block = vec![drafter.config().mask_token_id; drafter.config().block_size];
        block[0] = first;
        let block = nuvec::Array::from_slice(&block, [1, block.len() as i32]);
        let logits = drafter
            .logits(&model, &block)
            .reshape([block.size() as i32, -1]);
        let d = rel_diff(&logits, &expected["logits"]);
        assert!(d < tol, "{drafter_name}: logits off by {d}");
        // The draft: the argmax of each masked position, DFlash 2's selector path, or
        // DSpark's Markov-corrected path from every slot (with its confidences).
        let dspark = drafter.config().dspark.is_some();
        let n = drafter.config().block_size - usize::from(!dspark);
        let draft = drafter.draft(&model, &[first], n).unwrap();
        let want = expected["draft"].to_vec::<u32>().unwrap();
        assert_eq!(
            draft.tokens.to_vec::<u32>().unwrap(),
            want,
            "{drafter_name}: draft"
        );
        if dspark {
            let got = drafter.last_confidence().expect("a confidence head");
            let d = rel_diff(got, &expected["confidence"]);
            assert!(d < tol, "{drafter_name}: confidences off by {d}");
        }
    }
}

/// Speculative decoding with a DFlash drafter gives plain greedy decoding's tokens (dense and
/// hybrid targets), runs with a temperature, and a drafter for another target is refused.
#[test]
fn dflash_speculation_keeps_the_greedy_output() {
    use nuvec_lm::{DFlashDrafter, DraftPolicy, Sampler, speculative_generate};
    if !gpu() {
        return;
    }
    let drafter_dir = |name: &str| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    };
    for (drafter_name, target_name) in [
        ("dflash_qwen3", "qwen3_f32"),
        ("dflash_qwen3_5", "qwen3_5_f32"),
        ("dflash_qwen3_v2", "qwen3_f32"),
        ("dflash_gemma4", "gemma4_e_f32"),
        ("dflash_gpt_oss", "gpt_oss_f32"),
        ("dflash_qwen3_dspark", "qwen3_f32"),
    ] {
        let (dir, prompt, _, _, greedy) = fixture(target_name);
        let model = load_model(&dir).unwrap();
        let drafter = DFlashDrafter::load(drafter_dir(drafter_name), &model).unwrap();
        let options = GenerateOptions {
            max_tokens: greedy.len(),
            ..Default::default()
        };
        let mut g = speculative_generate(
            &model,
            &prompt,
            exact(options.clone()),
            drafter,
            DraftPolicy::Fixed(7),
        )
        .unwrap();
        let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
        assert_eq!(ids, greedy, "{drafter_name}: greedy");
        assert!(g.stats().verifies > 0, "{drafter_name}: {:?}", g.stats());
        // Draft trees (attention-only targets with DFlash v1; chains otherwise), routed as
        // MLX's: float32 fixtures keep the greedy tokens.
        for policy in [
            DraftPolicy::Tree(7),
            DraftPolicy::Tree(15),
            DraftPolicy::Auto,
        ] {
            let drafter = DFlashDrafter::load(drafter_dir(drafter_name), &model).unwrap();
            let mut g =
                speculative_generate(&model, &prompt, options.clone(), drafter, policy).unwrap();
            let ids: Vec<u32> = g.by_ref().collect::<nuvec::Result<_>>().unwrap();
            assert_eq!(ids, greedy, "{drafter_name}: {policy:?}");
            let stats = g.stats();
            assert!(stats.verifies > 0, "{drafter_name}: {policy:?} {stats:?}");
            // Trees of 15 nodes, 7 deep, where trees apply: more nodes than a chain holds.
            let tree = !matches!(
                drafter_name,
                "dflash_qwen3_5" | "dflash_qwen3_v2" | "dflash_qwen3_dspark"
            );
            if policy == DraftPolicy::Tree(15) && tree {
                assert!(
                    stats.drafted > 7 * stats.verifies,
                    "{drafter_name}: {stats:?}"
                );
            }
        }
        let drafter = DFlashDrafter::load(drafter_dir(drafter_name), &model).unwrap();
        let sampled_tree = GenerateOptions {
            sampler: Sampler {
                temperature: 0.8,
                ..Default::default()
            },
            seed: Some(7),
            ..options.clone()
        };
        let n = speculative_generate(
            &model,
            &prompt,
            sampled_tree,
            drafter,
            DraftPolicy::Tree(15),
        )
        .unwrap()
        .collect::<nuvec::Result<Vec<u32>>>()
        .unwrap()
        .len();
        assert_eq!(n, greedy.len(), "{drafter_name}: sampled tree");
        let drafter = DFlashDrafter::load(drafter_dir(drafter_name), &model).unwrap();
        let sampled = GenerateOptions {
            sampler: Sampler {
                temperature: 0.8,
                ..Default::default()
            },
            seed: Some(7),
            ..options
        };
        let n = speculative_generate(
            &model,
            &prompt,
            exact(sampled),
            drafter,
            DraftPolicy::Fixed(7),
        )
        .unwrap()
        .collect::<nuvec::Result<Vec<u32>>>()
        .unwrap()
        .len();
        assert_eq!(n, greedy.len(), "{drafter_name}: sampled");
    }
    // A drafter reading layer 3 of a two-layer target.
    let (dir, ..) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let err = DFlashDrafter::load(drafter_dir("dflash_qwen3_5"), &model)
        .err()
        .unwrap();
    assert!(err.to_string().contains("target layer 3"), "{err}");
}

/// Sampled draft trees are lossless: the first two tokens of tree speculation are
/// distributed as plain sampling's, as close as two plain samplings with other seeds are.
#[test]
fn sampled_draft_trees_keep_the_distribution() {
    use nuvec_lm::{DFlashDrafter, DraftPolicy, Sampler, speculative_generate};
    use std::collections::{HashMap, HashSet};
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("qwen3_f32");
    let model = load_model(&dir).unwrap();
    let drafter_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dflash_qwen3");
    let options = |seed| GenerateOptions {
        max_tokens: 2,
        sampler: Sampler {
            temperature: 0.3,
            ..Default::default()
        },
        seed: Some(seed),
        ..Default::default()
    };
    let n = 400u64;
    type Counts = HashMap<(u32, u32), f64>;
    let mut counts: [Counts; 3] = Default::default();
    for seed in 0..n {
        let plain = |seed| -> Vec<u32> {
            generate(&model, &prompt, options(seed))
                .unwrap()
                .collect::<nuvec::Result<_>>()
                .unwrap()
        };
        let drafter = DFlashDrafter::load(&drafter_dir, &model).unwrap();
        let tree: Vec<u32> = speculative_generate(
            &model,
            &prompt,
            options(seed),
            drafter,
            DraftPolicy::Tree(7),
        )
        .unwrap()
        .collect::<nuvec::Result<_>>()
        .unwrap();
        for (c, ids) in counts
            .iter_mut()
            .zip([plain(seed), plain(seed + 100_000), tree])
        {
            *c.entry((ids[0], ids[1])).or_default() += 1.0 / n as f64;
        }
    }
    let tv = |a: &Counts, b: &Counts| {
        let keys: HashSet<_> = a.keys().chain(b.keys()).collect();
        keys.iter()
            .map(|k| (a.get(k).unwrap_or(&0.0) - b.get(k).unwrap_or(&0.0)).abs())
            .sum::<f64>()
            / 2.0
    };
    let (noise, tree) = (tv(&counts[0], &counts[1]), tv(&counts[0], &counts[2]));
    eprintln!(
        "pairs: plain/plain {noise:.3}, plain/tree {tree:.3}, {} outcomes",
        counts[0].len()
    );
    assert!(
        tree < 1.5 * noise + 0.03,
        "plain/tree {tree} against plain/plain {noise}"
    );
}

/// Rotated key caches (`KvQuant::rotate`): the logits over a rotated quantized cache equal
/// the unquantized cache's within quantization error (the rotation itself is exact), for
/// the prompt and the next token, and such a cache is not saved in mlx-lm's format.
#[test]
fn rotated_kv_caches_keep_the_attention() {
    use nuvec_lm::{KvQuant, save_prompt_cache};
    if !gpu() {
        return;
    }
    let (dir, prompt, ..) = fixture("qwen3_5_f32");
    let model = load_model(&dir).unwrap();
    let tokens = Array::from_slice(&prompt, [1, prompt.len() as i32]);
    let next = Array::from_slice(&prompt[..1], [1, 1]);
    let run = |quant: Option<KvQuant>| {
        let mut kv = model.new_cache();
        if let Some(q) = quant {
            for c in &mut kv {
                c.to_quantized(q).unwrap();
            }
        }
        let first = model.forward(&tokens, &mut kv);
        let second = model.forward(&next, &mut kv);
        (first, second, kv)
    };
    let quant = |bits, rotate| KvQuant {
        bits,
        group_size: 32,
        start: 0,
        rotate,
    };
    let (dense, dense_next, _) = run(None);
    for bits in [8, 4] {
        let (plain, _, _) = run(Some(quant(bits, false)));
        let (rotated, rotated_next, kv) = run(Some(quant(bits, true)));
        let (d, d_next) = (
            rel_diff(&rotated, &dense),
            rel_diff(&rotated_next, &dense_next),
        );
        let p = rel_diff(&plain, &dense);
        eprintln!("{bits} bits: rotated {d} / {d_next}, plain {p}");
        let bound = if bits == 8 { 1e-2 } else { 1e-1 };
        assert!(d < bound && d_next < bound, "{bits} bits: {d}, {d_next}");
        if bits == 8 {
            let file = std::env::temp_dir().join(format!(
                "nuvec-kv-rotated-{}.safetensors",
                std::process::id()
            ));
            let err = save_prompt_cache(&file, &model, &kv, &Default::default()).unwrap_err();
            assert!(err.to_string().contains("rotated"), "{err}");
        }
    }
}
