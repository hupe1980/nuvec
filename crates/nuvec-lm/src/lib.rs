//! LLM runtime: Llama-family models (Llama, Mistral, Qwen 2, Qwen 3) loaded from Hugging
//! Face or MLX checkpoints (including MLX's quantized ones), a byte-level BPE tokenizer,
//! a key-value cache, token generation and sampling (temperature, top-p, min-p, top-k), as
//! mlx-lm.
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
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nuvec::{Array, Error, Result};
use nuvec_nn::Module;

pub mod config;
pub mod generate;
pub mod model;
pub mod sample;
pub mod tokenizer;

pub use config::Config;
pub use generate::{GenerateOptions, Generation, generate};
pub use model::{KvCache, Model};
pub use sample::Sampler;
pub use tokenizer::{StreamDecoder, Tokenizer};

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The directory of `model`: a path, or a Hugging Face repository id (`org/name`) found
/// in the local cache (`$HF_HOME/hub` or `~/.cache/huggingface/hub`). Nothing is
/// downloaded.
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
    let revision = std::fs::read_to_string(repo.join("refs/main")).map_err(|_| {
        Error::InvalidArgument(format!(
            "{model}: not a directory and not in the Hugging Face cache at {}",
            hub.display()
        ))
    })?;
    Ok(repo.join("snapshots").join(revision.trim()))
}

/// Loads a model and its tokenizer from a directory with `config.json`,
/// `tokenizer.json` and `model*.safetensors` files (see [`resolve`] for `model`).
///
/// Weights are memory-mapped, not copied. A linear or embedding layer is quantized when
/// the checkpoint has its `scales` (as mlx-lm).
pub fn load(model: &str) -> Result<(Model, Tokenizer)> {
    let dir = resolve(model)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
    Ok((load_model(dir)?, tokenizer))
}

/// mlx-lm's `sanitize` of hybrid and multimodal checkpoints: the language model's weights
/// under their text-model names (vision and multi-token-prediction weights dropped) and, in
/// Hugging Face checkpoints (convolutions stored `[C, 1, K]`), the convolutions as
/// `[C, K, 1]` and the zero-centred norm weights plus one.
fn sanitize(weights: BTreeMap<String, Array>) -> Result<BTreeMap<String, Array>> {
    let mut weights: BTreeMap<String, Array> = weights
        .into_iter()
        .filter(|(k, _)| {
            !(k.starts_with("vision_tower") || k.starts_with("model.visual") || k.contains("mtp."))
        })
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
        let mid = gate_up.shape()[gate_up.ndim() - 2] / 2;
        let mut halves = nuvec::split_at(&gate_up, &[mid], -2)?;
        let up = halves.pop().expect("two halves");
        weights.insert(format!("{prefix}switch_mlp.up_proj.weight"), up);
        weights.insert(
            format!("{prefix}switch_mlp.gate_proj.weight"),
            halves.pop().expect("two halves"),
        );
        if let Some(down) = weights.remove(&format!("{prefix}experts.down_proj")) {
            weights.insert(format!("{prefix}switch_mlp.down_proj.weight"), down);
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
                *v = nuvec::transpose(v, &[0, 2, 1])?;
            } else if v.ndim() == 1 && NORMS.iter().any(|n| k.ends_with(n)) {
                *v = nuvec::add(&*v, 1.0)?;
            }
        }
    }
    Ok(weights)
}

/// Loads a model (without its tokenizer) from a directory with `config.json` and
/// `model*.safetensors` files, as [`load`]. The weights are memory-mapped; the model is
/// then warmed up ([`Model::warm_up`]), as mlx-lm evaluates its weights when it loads.
pub fn load_model(dir: impl AsRef<Path>) -> Result<Model> {
    let dir = dir.as_ref();
    let config = Config::from_file(dir.join("config.json"))?;
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
        weights.extend(nuvec::io::load_safetensors(file)?.tensors);
    }
    let mut weights = sanitize(weights)?;
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
    model.load_weights(weights, true)?;
    model.warm_up()?;
    Ok(model)
}
