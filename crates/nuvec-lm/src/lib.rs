//! LLM runtime with mlx-lm's behaviour and speed:
//!
//! - **Models:** Llama, Mistral, Qwen 2, Qwen 3, Qwen 3 MoE, Qwen 3.5 (hybrid Gated
//!   DeltaNet and attention, dense or mixture-of-experts), gpt-oss and Gemma 4 (text), from
//!   Hugging Face or MLX checkpoints, including MLX's quantized ones. Aligned files are
//!   memory-mapped without a copy; mlx-community checkpoints (unpadded headers) are read
//!   once at load, in parallel.
//! - **Tokenizer:** BPE read from `tokenizer.json`: byte-level (GPT-2, Llama 3, Qwen 2/3/3.5,
//!   gpt-oss's o200k and others with their split patterns) and SentencePiece-style with
//!   byte fallback (Gemma, Llama 2, Mistral v0.1–v0.3). Other split patterns (Mistral's
//!   Tekken) are not supported, so [`load`] fails for those models; [`load_model`] loads
//!   their weights.
//! - **Generation:** chunked prefill, a pipelined decode loop with the KV cache updated in
//!   place, stopping at the model's end-of-sequence tokens (`config.json` and
//!   `generation_config.json`), mlx-lm's samplers (temperature, top-p, min-p, top-k), and
//!   [speculative decoding](speculative) with pluggable drafters; memory is wired while
//!   generating ([`WiredLimit`]).
//! - **Serving:** batched decoding over a [paged KV cache](paged) and a [`Batcher`] that
//!   serves many requests at once.
//! - **Fine-tuning:** a [`Model`] is a `nuvec_nn::Module` (Hugging Face's parameter names);
//!   [`lora`] replaces projections by path with LoRA layers and reads and writes mlx-lm's
//!   adapter directories.
//!
//! ```no_run
//! use nuvec_lm::{GenerateOptions, generate, load};
//!
//! # fn main() -> nuvec::Result<()> {
//! let (model, tokenizer) = load("mlx-community/Qwen3-8B-4bit")?;
//! let prompt = tokenizer.encode("The capital of France is");
//! let options = GenerateOptions { max_tokens: 8, ..Default::default() };
//! let ids = generate(&model, &prompt, options)?.collect::<nuvec::Result<Vec<u32>>>()?;
//! println!("{}", tokenizer.decode(&ids));
//! # Ok(())
//! # }
//! ```
//!
//! A forward pass returns an array: nothing runs until a boundary (`eval`, `item`,
//! `to_vec`) asks for it; errors surface at `check` (without evaluating) or there.
//!
//! ```no_run
//! use nuvec::Array;
//! use nuvec_lm::load_model;
//!
//! # fn main() -> nuvec::Result<()> {
//! let model = load_model("models/Qwen3-0.6B")?;
//! let mut cache = model.new_cache();
//! let tokens = Array::from_slice(&[9707u32, 11], [1, 2]);
//! let logits = model.forward(&tokens, &mut cache); // [1, 2, vocab]
//! let next = logits.i((.., -1)).argmax(-1).item::<u32>()?;
//! println!("next token: {next}");
//! # Ok(())
//! # }
//! ```
//!
//! **Stability:** semver covers this crate and every item reachable through `nuvec::` paths;
//! the pinned items are listed in `nuvec`'s `tests/api_surface.rs`.
//!
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nuvec::{Array, Error, Quant, QuantMode, Result};
use nuvec_nn::Module;

pub mod batch;
pub mod config;
pub mod generate;
pub mod lora;
pub mod model;
pub mod paged;
pub mod pipeline;
pub mod quant;
pub mod sample;
pub mod speculative;
pub mod stop;
mod suffix;
pub mod tokenizer;
pub mod tuner;

pub use batch::{Batcher, BatcherOptions, Event, RequestId};
pub use config::{Activation, Config, Gemma4, LayerQuant, PackedFormat, Quantization};
pub use generate::{FinishReason, GenerateOptions, Generation, WiredLimit, generate};
pub use model::{KvCache, Model};
pub use paged::{PagedCache, SeqId, decode_batch};
pub use pipeline::{Split, Stage, load_model_pipelined, load_model_sharded};
pub use sample::Sampler;
pub use speculative::{
    Auto, Draft, DraftPolicy, Drafter, MtpDrafter, PromptLookup, SpeculativeGeneration,
    SpeculativeStats, SuffixDecoding, speculative_generate, verify_rows,
};
pub use stop::{Stop, Stopped, TextStream};
pub use tokenizer::{StreamDecoder, Tokenizer};

/// The Rust snippets of the README and of the user guide's pages that use the LLM runtime,
/// run as doctests.
#[cfg(doctest)]
mod site_doctests {
    #[doc = include_str!("../../../README.md")]
    pub struct Readme;
    #[doc = include_str!("../../../site/content/docs/getting-started.md")]
    pub struct GettingStarted;
    #[doc = include_str!("../../../site/content/docs/llm.md")]
    pub struct Llm;
    #[doc = include_str!("../../../site/content/docs/speculative-decoding.md")]
    pub struct SpeculativeDecoding;
}

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The directory of `model`: a path, or a Hugging Face repository id (`org/name`) found
/// in the local cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`, as `huggingface_hub`): the snapshot `refs/main` names if it
/// has a `config.json`, else the most recently modified complete one. Nothing is
/// downloaded.
pub fn resolve(model: &str) -> Result<PathBuf> {
    let path = Path::new(model);
    if path.is_dir() {
        return Ok(path.to_owned());
    }
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
        })
        .ok_or_else(|| Error::InvalidArgument("no Hugging Face cache directory".into()))?;
    let repo = hub.join(format!("models--{}", model.replace('/', "--")));
    let missing = || {
        Error::InvalidArgument(format!(
            "{model}: not a directory and not in the Hugging Face cache at {}",
            hub.display()
        ))
    };
    if let Ok(revision) = std::fs::read_to_string(repo.join("refs/main")) {
        let snapshot = repo.join("snapshots").join(revision.trim());
        if snapshot.join("config.json").exists() {
            return Ok(snapshot);
        }
    }
    // Caches filled without a ref (copied, or downloaded by revision).
    std::fs::read_dir(repo.join("snapshots"))
        .map_err(|_| missing())?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("config.json").exists())
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
        .ok_or_else(missing)
}

/// Loads a model and its tokenizer from a directory with `config.json`,
/// `tokenizer.json` and `model*.safetensors` files (see [`resolve`] for `model`).
///
/// Aligned weight files are memory-mapped without a copy; mlx-community checkpoints
/// (unpadded headers) are copied once. A linear or embedding layer is quantized when
/// the checkpoint has its `scales` (as mlx-lm).
pub fn load(model: &str) -> Result<(Model, Tokenizer)> {
    let dir = resolve(model)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    Ok((load_model(dir)?, tokenizer))
}

/// Whether mlx-lm's `sanitize` drops a checkpoint weight: vision and audio weights, and
/// Gemma 4's clipping ranges and the keys and values
/// of its KV-shared layers. [`load_model`] does not read them.
fn dropped(name: &str, config: &Config) -> bool {
    const TOWERS: [&str; 7] = [
        "vision_tower",
        "multi_modal_projector",
        "audio_tower",
        "embed_audio",
        "embed_vision",
        "vision_embedder",
        "visual",
    ];
    let bare = name.strip_prefix("model.").unwrap_or(name);
    if TOWERS.iter().any(|t| bare.starts_with(t)) {
        return true;
    }
    let Some(g) = &config.gemma4 else {
        return false;
    };
    const CLIPS: [&str; 5] = [
        "self_attn.rotary_emb",
        "input_max",
        "input_min",
        "output_max",
        "output_min",
    ];
    if CLIPS.iter().any(|c| name.contains(c)) {
        return true;
    }
    let first_shared = g.first_shared(config.num_hidden_layers);
    let layer = name
        .split_once("layers.")
        .and_then(|(_, rest)| rest.split('.').next())
        .and_then(|i| i.parse::<usize>().ok());
    layer.is_some_and(|i| i >= first_shared)
        && [
            ".self_attn.k_proj",
            ".self_attn.v_proj",
            ".self_attn.k_norm",
        ]
        .iter()
        .any(|s| name.contains(s))
}

/// mlx-lm's `sanitize` of hybrid and multimodal checkpoints: the language model's weights
/// under their text-model names (vision and multi-token-prediction weights dropped) and, in
/// Hugging Face checkpoints (convolutions stored `[C, 1, K]`), the convolutions as
/// `[C, K, 1]` and the zero-centred norm weights plus one.
fn sanitize(weights: BTreeMap<String, Array>, config: &Config) -> Result<BTreeMap<String, Array>> {
    let weights = match config.model_type.as_str() {
        "gpt_oss" => sanitize_gpt_oss(weights)?,
        _ => weights,
    };
    let mut weights: BTreeMap<String, Array> = weights
        .into_iter()
        .filter(|(k, _)| !dropped(k, config))
        .map(|(k, v)| {
            let k = match (
                k.strip_prefix("model.language_model."),
                k.strip_prefix("language_model."),
            ) {
                (Some(rest), _) => format!("model.{rest}"),
                (_, Some(rest)) => rest.to_owned(),
                _ => k,
            };
            (k, v)
        })
        .collect();
    // Gemma 4's Hugging Face experts: `experts.gate_up_proj` (`[E, 2·I, D]`, gate then up)
    // and `experts.down_proj`, as mlx-lm's `switch_glu` weights.
    if config.gemma4.is_some() {
        let fused: Vec<String> = weights
            .keys()
            .filter(|k| k.ends_with(".experts.gate_up_proj") || k.ends_with(".experts.down_proj"))
            .cloned()
            .collect();
        for key in fused {
            let v = weights.remove(&key).expect("listed");
            if let Some(base) = key.strip_suffix(".gate_up_proj") {
                let [gate, up] = v.split_n::<2>(-2);
                weights.insert(format!("{base}.switch_glu.gate_proj.weight"), gate.check()?);
                weights.insert(format!("{base}.switch_glu.up_proj.weight"), up.check()?);
            } else if let Some(base) = key.strip_suffix(".down_proj") {
                weights.insert(format!("{base}.switch_glu.down_proj.weight"), v);
            }
        }
    }
    // Hugging Face MoE checkpoints fuse each layer's expert gate and up projections
    // (`experts.gate_up_proj`, `[E, 2·I, D]`), as mlx-lm's `qwen3_5_moe.sanitize` splits.
    let fused: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".mlp.experts.gate_up_proj"))
        .cloned()
        .collect();
    for key in fused {
        let prefix = key.trim_end_matches("experts.gate_up_proj").to_owned();
        let gate_up = weights.remove(&key).expect("listed");
        let [gate, up] = gate_up.split_n::<2>(-2);
        weights.insert(format!("{prefix}switch_mlp.up_proj.weight"), up.check()?);
        weights.insert(
            format!("{prefix}switch_mlp.gate_proj.weight"),
            gate.check()?,
        );
        if let Some(down) = weights.remove(&format!("{prefix}experts.down_proj")) {
            weights.insert(format!("{prefix}switch_mlp.down_proj.weight"), down);
        }
    }
    // Hugging Face Qwen 3 MoE checkpoints store each expert's projections separately
    // (`experts.{e}.{name}.weight`), stacked as mlx-lm's `qwen3_moe.sanitize` does.
    let first: Vec<String> = weights
        .keys()
        .filter(|k| k.ends_with(".mlp.experts.0.up_proj.weight"))
        .cloned()
        .collect();
    for key in first {
        let prefix = key.trim_end_matches("experts.0.up_proj.weight").to_owned();
        for name in ["up_proj", "down_proj", "gate_proj"] {
            let mut experts = Vec::new();
            while let Some(w) =
                weights.remove(&format!("{prefix}experts.{}.{name}.weight", experts.len()))
            {
                experts.push(w);
            }
            weights.insert(
                format!("{prefix}switch_mlp.{name}.weight"),
                nuvec::ops::stack(&experts, 0).check()?,
            );
        }
    }
    let unsanitized = weights
        .iter()
        .any(|(k, v)| k.contains("conv1d.weight") && v.shape().last() != Some(&1));
    if unsanitized {
        const NORMS: [&str; 8] = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ];
        for (k, v) in &mut weights {
            if k.contains("conv1d.weight") && v.shape().last() != Some(&1) {
                *v = v.transpose([0, 2, 1]).check()?;
            } else if v.ndim() == 1 && NORMS.iter().any(|n| k.ends_with(n)) {
                *v = v.add(1.0).check()?;
            }
        }
    }
    Ok(weights)
}

/// gpt-oss's Hugging Face experts as mlx-lm's `SwitchGLU` (`experts.{gate,up,down}_proj`):
/// the fused `gate_up_proj` interleaves gate and up rows. MXFP4 originals
/// (`*_blocks`/`*_scales`, `[E, 2·I, D / 32, 16]`) keep their packing, as mlx-lm's
/// `gpt_oss.sanitize` (blocks viewed as uint32 codes, E8M0 scales unchanged); full-precision
/// ones (transformers' layout, `gate_up_proj` `[E, D, 2·I]`, `down_proj` `[E, I, D]`) are
/// split along the last axis and transposed to MLX's `[E, out, in]`. Biases are split too.
fn sanitize_gpt_oss(weights: BTreeMap<String, Array>) -> Result<BTreeMap<String, Array>> {
    if weights.keys().any(|k| k.contains("experts.gate_proj.")) {
        return Ok(weights); // converted already (mlx-community)
    }
    // Every other entry along `axis`, from `first` (`v[..., first::2, ...]`); errors are
    // sticky and surface where the loader checks the weights.
    let rows = |v: &Array, first: i32, axis: i32| -> Array {
        let n = v.shape()[(axis + v.ndim() as i32) as usize];
        v.take(&nuvec::ops::arange(first, n, 2), axis)
    };
    let mut out = BTreeMap::new();
    for (k, v) in weights {
        if let Some(prefix) = k.strip_suffix("gate_up_proj_bias") {
            out.insert(format!("{prefix}gate_proj.bias"), rows(&v, 0, -1));
            out.insert(format!("{prefix}up_proj.bias"), rows(&v, 1, -1));
        } else if let Some(prefix) = k.strip_suffix("down_proj_bias") {
            out.insert(format!("{prefix}down_proj.bias"), v);
        } else if let Some(prefix) = k.strip_suffix("gate_up_proj_blocks") {
            let codes = quant::mxfp4_from_hf_blocks(&v);
            out.insert(format!("{prefix}gate_proj.weight"), rows(&codes, 0, -2));
            out.insert(format!("{prefix}up_proj.weight"), rows(&codes, 1, -2));
        } else if let Some(prefix) = k.strip_suffix("gate_up_proj_scales") {
            out.insert(format!("{prefix}gate_proj.scales"), rows(&v, 0, -2));
            out.insert(format!("{prefix}up_proj.scales"), rows(&v, 1, -2));
        } else if let Some(prefix) = k.strip_suffix("down_proj_blocks") {
            out.insert(
                format!("{prefix}down_proj.weight"),
                quant::mxfp4_from_hf_blocks(&v),
            );
        } else if let Some(prefix) = k.strip_suffix("down_proj_scales") {
            out.insert(format!("{prefix}down_proj.scales"), v);
        } else if let Some(prefix) = k.strip_suffix("experts.gate_up_proj") {
            let t = |a: Array| a.transpose([0, 2, 1]);
            out.insert(
                format!("{prefix}experts.gate_proj.weight"),
                t(rows(&v, 0, -1)),
            );
            out.insert(
                format!("{prefix}experts.up_proj.weight"),
                t(rows(&v, 1, -1)),
            );
        } else if let Some(prefix) = k.strip_suffix("experts.down_proj") {
            out.insert(
                format!("{prefix}experts.down_proj.weight"),
                v.transpose([0, 2, 1]),
            );
        } else {
            out.insert(k, v);
        }
    }
    Ok(out)
}

/// The quantization a layer's tensors were packed with (mlx-lm's `infer_quant_config`):
/// uint8 scales mean a floating-point mode, told apart by the packed codes per group (2 for
/// nvfp4, 4 for mxfp4, 8 for mxfp8); other scales are affine with the config's settings.
fn infer_quant(weight: &Array, scales: &Array, default: Quant) -> Quant {
    if scales.dtype() != nuvec::DType::U8 {
        return match default.mode {
            QuantMode::Affine => default,
            _ => Quant::default(),
        };
    }
    let last = |a: &Array| a.shape().last().copied().unwrap_or(0);
    let (packed, groups) = (last(weight), last(scales).max(1));
    match packed / groups {
        2 => Quant::new(QuantMode::Nvfp4),
        8 => Quant::new(QuantMode::Mxfp8),
        _ => Quant::new(QuantMode::Mxfp4),
    }
}

/// Loads a model (without its tokenizer) from a directory with `config.json` and
/// `model*.safetensors` files, as [`load`]. Aligned files are memory-mapped; misaligned
/// ones (as MLX writes them) are read in parallel, so their weights are in memory when it
/// returns, as mlx-lm evaluates its weights when it loads. The vision and
/// multi-token-prediction weights mlx-lm drops are not read. Nothing runs through the
/// model: [`Model::warm_up`] builds its kernels before a latency-sensitive first prompt.
/// A `generation_config.json` adds its end-of-sequence ids ([`Config::from_file`]); a
/// config without `tie_word_embeddings` is tied when the checkpoint has no `lm_head`.
pub fn load_model(dir: impl AsRef<Path>) -> Result<Model> {
    load_model_with(dir.as_ref(), None)
}

/// Decides a pipeline stage from the configuration and the checkpoint's tensors (names and
/// sizes from the headers) before any weight is read.
pub(crate) type StagePlan<'a> =
    &'a dyn Fn(&Config, &[nuvec::io::TensorInfo]) -> Result<pipeline::Stage>;

/// [`load_model`], or with `stage`, only that stage's tensors (the others stay unloaded).
pub(crate) fn load_model_with(dir: &Path, stage: Option<StagePlan<'_>>) -> Result<Model> {
    let (mut config, tie_set) = Config::read(&dir.join("config.json"))?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error::InvalidArgument(format!("{}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        // `model*.safetensors`, as mlx-lm (shards included).
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("model") && n.ends_with(".safetensors"))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "{}: no model*.safetensors files",
            dir.display()
        )));
    }
    // A pipeline stage: decided from the headers, then only its tensors are read.
    let mut has_head = None;
    let stage = match stage {
        Some(plan) => {
            let mut index = Vec::new();
            for file in &files {
                index.extend(nuvec::io::safetensors_tensors(file)?);
            }
            index.retain(|t| !dropped(&t.name, &config));
            has_head = Some(index.iter().any(|t| t.name == "lm_head.weight"));
            if !tie_set {
                config.tie_word_embeddings = has_head == Some(false);
            }
            Some(plan(&config, &index)?)
        }
        None => None,
    };
    let keep = |k: &str| {
        !dropped(k, &config)
            && stage
                .as_ref()
                .is_none_or(|s| pipeline::in_stage(k, s, config.tie_word_embeddings))
    };
    let mut weights: BTreeMap<String, Array> = BTreeMap::new();
    // `NUVEC_ALIGNED_CACHE`: misaligned checkpoints (mlx-community's) load from aligned
    // copies there, mapped without a copy after the first load.
    let cache = std::env::var_os("NUVEC_ALIGNED_CACHE").filter(|d| !d.is_empty());
    for file in &files {
        // Weights `sanitize` drops are not read (mlx-lm never evaluates them either).
        let loaded = match &cache {
            Some(dir) => nuvec::io::load_safetensors_cached(file, dir, keep)?,
            None => nuvec::io::load_safetensors_filtered(file, keep)?,
        };
        weights.extend(loaded.tensors);
    }
    let mut weights = sanitize(weights, &config)?;
    // AWQ and GPTQ packings in MLX's affine layout.
    if let Some(q) = &config.quantization
        && let Some(format) = q.packed
    {
        weights = quant::convert_packed(weights, format, q.default)?;
    }
    // Without `tie_word_embeddings` (Hugging Face's default is tied), a checkpoint
    // without an output head is tied.
    if !tie_set {
        config.tie_word_embeddings =
            !has_head.unwrap_or_else(|| weights.contains_key("lm_head.weight"));
    }
    // mlx-lm's `sanitize`: drop precomputed rotary frequencies and tied heads.
    weights.retain(|name, _| !name.contains("rotary_emb.inv_freq"));
    if config.tie_word_embeddings {
        weights.retain(|name, _| !name.starts_with("lm_head."));
    }
    let quantization = config.quantization.clone();
    // mlx-lm's rule: a layer the config names takes its entry; any other layer with scales is
    // recognised from its tensors (`infer_quant_config`), with a global scale when the
    // checkpoint has one.
    let quantized = |path: &str| -> Option<LayerQuant> {
        let q = quantization.as_ref()?;
        if let Some(entry) = q.for_layer(path) {
            return Some(entry.clone());
        }
        let scales = weights.get(&format!("{path}.scales"))?;
        let weight = weights.get(&format!("{path}.weight"))?;
        Some(LayerQuant {
            path: path.to_owned(),
            quant: Some(infer_quant(weight, scales, q.default)),
            global_scale: weights.contains_key(&format!("{path}.global_scale")),
        })
    };
    let mut model = Model::new(&config, &quantized)?;
    // Multi-token-prediction weights (original checkpoints keep them): the head is built
    // when they are present.
    if config.mtp_layers > 0 && weights.contains_key("mtp.fc.weight") {
        model.build_mtp(&quantized)?;
    } else {
        weights.retain(|k, _| !k.starts_with("mtp."));
    }
    match stage {
        // The other stages' parameters stay unloaded placeholders (no memory).
        Some(mut stage) => {
            model.load_weights_partial(weights)?;
            pipeline::share_activations(&model, &mut stage)?;
            model.set_stage(stage);
        }
        None => model.load_weights(weights)?,
    }
    // Evaluation mode, as `mlx_lm.load` leaves models (dropout in LoRA layers is off;
    // `model.train(true)` switches it on for fine-tuning).
    model.train(false);
    Ok(model)
}
