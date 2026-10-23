//! LLM runtime with mlx-lm's behaviour and speed:
//!
//! - **Models:** Llama, Mistral, Qwen 2, Qwen 3, Qwen 3 MoE and Qwen 3.5 (hybrid Gated
//!   DeltaNet and attention, dense or mixture-of-experts), from Hugging Face or MLX
//!   checkpoints, including MLX's quantized ones. Aligned files are memory-mapped without a
//!   copy; mlx-community checkpoints (unpadded headers) are read once at load, in parallel.
//! - **Tokenizer:** byte-level BPE read from `tokenizer.json` (GPT-2, Llama 3, Qwen 2/3/3.5
//!   and others with their split patterns). SentencePiece-style tokenizers (byte-fallback
//!   BPE: Llama 2, Mistral v0.1–v0.3) are not supported, so [`load`] fails for those
//!   models; [`load_model`] loads their weights.
//! - **Generation:** chunked prefill, a pipelined decode loop with the KV cache updated in
//!   place, stopping at the model's end-of-sequence tokens (`config.json` and
//!   `generation_config.json`), mlx-lm's samplers (temperature, top-p, min-p, top-k), and [speculative
//!   decoding](speculative) with pluggable drafters; memory is wired while generating
//!   ([`WiredLimit`]).
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
//! A forward pass returns an array: nothing runs until a boundary (`check`, `eval`, `item`,
//! `to_vec`) asks for it, and an error surfaces there.
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

use nuvec::{Array, Error, Result};
use nuvec_nn::Module;

pub mod batch;
pub mod config;
pub mod generate;
pub mod lora;
pub mod model;
pub mod paged;
pub mod sample;
pub mod speculative;
pub mod tokenizer;

pub use batch::{Batcher, Event, FinishReason, RequestId};
pub use config::Config;
pub use generate::{GenerateOptions, Generation, WiredLimit, generate};
pub use model::{KvCache, Model};
pub use paged::{PagedCache, SeqId, decode_batch};
pub use sample::Sampler;
pub use speculative::{
    Drafter, PromptLookup, SpeculativeGeneration, SpeculativeStats, speculative_generate,
};
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
/// in the local cache (`$HF_HOME/hub` or `~/.cache/huggingface/hub`): the snapshot
/// `refs/main` names, or else the most recently modified one. Nothing is downloaded.
pub fn resolve(model: &str) -> Result<PathBuf> {
    let path = Path::new(model);
    if path.is_dir() {
        return Ok(path.to_owned());
    }
    let hub = std::env::var_os("HF_HOME")
        .map(|h| PathBuf::from(h).join("hub"))
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
        return Ok(repo.join("snapshots").join(revision.trim()));
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

/// Whether mlx-lm's `sanitize` drops a checkpoint weight: vision-tower and
/// multi-token-prediction weights. [`load_model`] does not read them.
fn dropped(name: &str) -> bool {
    name.starts_with("vision_tower") || name.starts_with("model.visual") || name.contains("mtp.")
}

/// mlx-lm's `sanitize` of hybrid and multimodal checkpoints: the language model's weights
/// under their text-model names (vision and multi-token-prediction weights dropped) and, in
/// Hugging Face checkpoints (convolutions stored `[C, 1, K]`), the convolutions as
/// `[C, K, 1]` and the zero-centred norm weights plus one.
fn sanitize(weights: BTreeMap<String, Array>) -> Result<BTreeMap<String, Array>> {
    let mut weights: BTreeMap<String, Array> = weights
        .into_iter()
        .filter(|(k, _)| !dropped(k))
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
        const NORMS: [&str; 5] = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
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

/// Loads a model (without its tokenizer) from a directory with `config.json` and
/// `model*.safetensors` files, as [`load`]. Aligned files are memory-mapped; misaligned
/// ones (as MLX writes them) are read in parallel, so their weights are in memory when it
/// returns, as mlx-lm evaluates its weights when it loads. The vision and
/// multi-token-prediction weights mlx-lm drops are not read. Nothing runs through the
/// model: [`Model::warm_up`] builds its kernels before a latency-sensitive first prompt.
/// A `generation_config.json` adds its end-of-sequence ids ([`Config::from_file`]); a
/// config without `tie_word_embeddings` is tied when the checkpoint has no `lm_head`.
pub fn load_model(dir: impl AsRef<Path>) -> Result<Model> {
    let dir = dir.as_ref();
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
    let mut weights: BTreeMap<String, Array> = BTreeMap::new();
    for file in &files {
        // Weights `sanitize` drops are not read (mlx-lm never evaluates them either).
        weights.extend(nuvec::io::load_safetensors_filtered(file, |k| !dropped(k))?.tensors);
    }
    let mut weights = sanitize(weights)?;
    // Without `tie_word_embeddings` (Hugging Face's default is tied), a checkpoint
    // without an output head is tied.
    if !tie_set {
        config.tie_word_embeddings = !weights.contains_key("lm_head.weight");
    }
    // mlx-lm's `sanitize`: drop precomputed rotary frequencies and tied heads.
    weights.retain(|name, _| !name.contains("rotary_emb.inv_freq"));
    if config.tie_word_embeddings {
        weights.retain(|name, _| !name.starts_with("lm_head."));
    }
    let quantization = config.quantization.clone();
    let quantized = |path: &str| {
        let q = quantization.as_ref()?;
        weights
            .contains_key(&format!("{path}.scales"))
            .then(|| q.for_layer(path))
    };
    let mut model = Model::new(&config, &quantized)?;
    model.load_weights(weights)?;
    // Evaluation mode, as `mlx_lm.load` leaves models (dropout in LoRA layers is off;
    // `model.train(true)` switches it on for fine-tuning).
    model.train(false);
    Ok(model)
}
