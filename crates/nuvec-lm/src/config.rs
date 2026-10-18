//! Model configuration from Hugging Face `config.json` files.

use std::path::Path;

use nuvec::io::json::{self, Json};
use nuvec::{Error, Result};

fn invalid(msg: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("config.json: {msg}"))
}

/// How positions are scaled before the rotary embedding.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// None (`default`).
    None,
    /// Positions divided by `factor` (`linear`).
    Linear {
        /// The division factor.
        factor: f32,
    },
    /// Llama 3's frequency-dependent scaling (`llama3`).
    Llama3 {
        /// Scale of low frequencies.
        factor: f32,
        /// Wavelength bound (in original contexts) of scaled low frequencies.
        low_freq_factor: f32,
        /// Wavelength bound of unscaled high frequencies.
        high_freq_factor: f32,
        /// The context length the model was trained with.
        original_max_position_embeddings: f32,
    },
}

/// MLX's affine quantization of a checkpoint's linear and embedding layers.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantization {
    /// Values per scale and bias.
    pub group_size: usize,
    /// Bits per value.
    pub bits: u32,
    /// Per-layer settings (path, group size, bits) that differ from the defaults.
    pub layers: Vec<(String, usize, u32)>,
}

impl Quantization {
    /// The group size and bits of the layer at `path`.
    pub fn for_layer(&self, path: &str) -> (usize, u32) {
        self.layers
            .iter()
            .find(|(p, _, _)| p == path)
            .map_or((self.group_size, self.bits), |&(_, g, b)| (g, b))
    }
}

/// A decoder-only transformer of the Llama family (Llama, Mistral, Qwen 2, Qwen 3).
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `llama`, `mistral`, `qwen2` or `qwen3`.
    pub model_type: String,
    /// Width of the residual stream.
    pub hidden_size: usize,
    /// Transformer blocks.
    pub num_hidden_layers: usize,
    /// Width of the MLP's hidden layer.
    pub intermediate_size: usize,
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key and value heads (grouped-query attention).
    pub num_key_value_heads: usize,
    /// Features per head.
    pub head_dim: usize,
    /// Token ids.
    pub vocab_size: usize,
    /// Added to the mean square in RMS normalisation.
    pub rms_norm_eps: f32,
    /// The rotary embedding's base.
    pub rope_theta: f32,
    /// Rotary position scaling.
    pub rope_scaling: RopeScaling,
    /// Rotate consecutive feature pairs instead of the two halves (`rope_traditional`).
    pub rope_traditional: bool,
    /// The output projection is the embedding table.
    pub tie_word_embeddings: bool,
    /// Biases on the query, key and value projections.
    pub qkv_bias: bool,
    /// A bias on the attention's output projection.
    pub o_bias: bool,
    /// Biases on the MLP's projections.
    pub mlp_bias: bool,
    /// RMS normalisation of each query and key head (Qwen 3).
    pub qk_norm: bool,
    /// The quantization, for MLX-quantized checkpoints.
    pub quantization: Option<Quantization>,
    /// Ids that end generation.
    pub eos_token_ids: Vec<u32>,
}

impl Config {
    /// Reads a `config.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error::InvalidArgument(format!("{}: {e}", path.display())))?;
        Self::from_json(&json::parse(&bytes)?)
    }

    /// Builds a configuration from a parsed `config.json`.
    pub fn from_json(c: &Json) -> Result<Self> {
        let int = |k: &str| {
            c.get(k)
                .and_then(Json::as_i64)
                .map(|v| v as usize)
                .ok_or_else(|| invalid(format!("`{k}` missing")))
        };
        let float = |k: &str| c.get(k).and_then(Json::as_f64).map(|v| v as f32);
        let flag = |k: &str| c.get(k).and_then(Json::as_bool).unwrap_or(false);
        let model_type = c
            .get("model_type")
            .and_then(Json::as_str)
            .ok_or_else(|| invalid("`model_type` missing"))?
            .to_owned();
        let (qkv_bias, o_bias, qk_norm) = match model_type.as_str() {
            "llama" | "mistral" => (flag("attention_bias"), flag("attention_bias"), false),
            "qwen2" => (true, false, false),
            "qwen3" => (flag("attention_bias"), flag("attention_bias"), true),
            other => return Err(invalid(format!("unsupported model type {other:?}"))),
        };
        let hidden_size = int("hidden_size")?;
        let heads = int("num_attention_heads")?;
        let head_dim = match c.get("head_dim") {
            Some(Json::Int(d)) => *d as usize,
            _ => hidden_size / heads,
        };
        Ok(Self {
            hidden_size,
            num_hidden_layers: int("num_hidden_layers")?,
            intermediate_size: int("intermediate_size")?,
            num_attention_heads: heads,
            num_key_value_heads: int("num_key_value_heads").unwrap_or(heads),
            head_dim,
            vocab_size: int("vocab_size")?,
            rms_norm_eps: float("rms_norm_eps").unwrap_or(1e-6),
            rope_theta: float("rope_theta").unwrap_or(10000.0),
            rope_scaling: rope_scaling(c.get("rope_scaling"))?,
            rope_traditional: model_type != "qwen3" && flag("rope_traditional"),
            tie_word_embeddings: flag("tie_word_embeddings"),
            qkv_bias,
            o_bias,
            mlp_bias: flag("mlp_bias"),
            qk_norm,
            quantization: quantization(c.get("quantization"))?,
            eos_token_ids: match c.get("eos_token_id") {
                Some(Json::Int(id)) => vec![*id as u32],
                Some(Json::Array(ids)) => ids
                    .iter()
                    .filter_map(Json::as_i64)
                    .map(|id| id as u32)
                    .collect(),
                _ => Vec::new(),
            },
            model_type,
        })
    }
}

fn rope_scaling(s: Option<&Json>) -> Result<RopeScaling> {
    let Some(s) = s.filter(|s| **s != Json::Null) else {
        return Ok(RopeScaling::None);
    };
    let kind = s
        .get("type")
        .or_else(|| s.get("rope_type"))
        .and_then(Json::as_str)
        .unwrap_or("default");
    let float = |k: &str, default: f32| {
        s.get(k)
            .and_then(Json::as_f64)
            .map_or(default, |v| v as f32)
    };
    match kind {
        "default" => Ok(RopeScaling::None),
        "linear" => Ok(RopeScaling::Linear {
            factor: float("factor", 1.0),
        }),
        "llama3" => Ok(RopeScaling::Llama3 {
            factor: float("factor", 1.0),
            low_freq_factor: float("low_freq_factor", 1.0),
            high_freq_factor: float("high_freq_factor", 4.0),
            original_max_position_embeddings: float("original_max_position_embeddings", 8192.0),
        }),
        other => Err(invalid(format!("unsupported rope scaling {other:?}"))),
    }
}

fn quantization(q: Option<&Json>) -> Result<Option<Quantization>> {
    let Some(q) = q.filter(|q| **q != Json::Null) else {
        return Ok(None);
    };
    let get = |j: &Json, k: &str| j.get(k).and_then(Json::as_i64);
    let (Some(group_size), Some(bits)) = (get(q, "group_size"), get(q, "bits")) else {
        return Err(invalid("quantization needs `group_size` and `bits`"));
    };
    if let Some(mode) = q.get("mode").and_then(Json::as_str)
        && mode != "affine"
    {
        return Err(invalid(format!("unsupported quantization mode {mode:?}")));
    }
    let layers = q
        .as_object()
        .unwrap_or_default()
        .iter()
        .filter_map(|(path, v)| {
            let (g, b) = (get(v, "group_size")?, get(v, "bits")?);
            Some((path.clone(), g as usize, b as u32))
        })
        .collect();
    Ok(Some(Quantization {
        group_size: group_size as usize,
        bits: bits as u32,
        layers,
    }))
}
