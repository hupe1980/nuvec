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

/// The token mixer of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Softmax attention over the cached keys and values.
    Attention,
    /// Gated DeltaNet linear attention (Qwen 3.5), with a recurrent state.
    Linear,
}

/// The sizes of Gated DeltaNet layers (`linear_*` settings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearAttention {
    /// Key (and query) heads.
    pub key_heads: usize,
    /// Value heads (a multiple of the key heads).
    pub value_heads: usize,
    /// Features per key head.
    pub key_head_dim: usize,
    /// Features per value head.
    pub value_head_dim: usize,
    /// Width of the causal depthwise convolution.
    pub conv_kernel: usize,
}

/// Mixture-of-experts MLPs (Qwen 3 MoE, Qwen 3.5 MoE): each token's `experts_per_token`
/// best experts (by a softmax router), weighted, plus (Qwen 3.5) a gated shared expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixtureOfExperts {
    /// Experts per layer.
    pub experts: usize,
    /// Experts each token uses.
    pub experts_per_token: usize,
    /// Width of each expert's hidden layer.
    pub intermediate_size: usize,
    /// Width of the shared expert's hidden layer, if there is one.
    pub shared_intermediate_size: Option<usize>,
    /// The chosen experts' weights are renormalised to sum to one.
    pub norm_topk_prob: bool,
    /// Every `sparse_step`-th layer is a mixture of experts (`decoder_sparse_step`).
    pub sparse_step: usize,
    /// Layers with a dense MLP regardless (`mlp_only_layers`).
    pub dense_layers: Vec<usize>,
}

impl MixtureOfExperts {
    /// Whether layer `i` is a mixture of experts (mlx-lm's rule).
    pub fn is_sparse(&self, i: usize) -> bool {
        !self.dense_layers.contains(&i) && (i + 1).is_multiple_of(self.sparse_step.max(1))
    }
}

/// A decoder-only transformer of the Llama family (Llama, Mistral, Qwen 2, Qwen 3, Qwen 3 MoE) or a
/// hybrid of attention and Gated DeltaNet layers (Qwen 3.5).
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `llama`, `mistral`, `qwen2`, `qwen3` or `qwen3_5`.
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
    /// Each layer's token mixer (all attention except in hybrid models).
    pub layer_types: Vec<LayerType>,
    /// The Gated DeltaNet sizes, for hybrid models.
    pub linear_attention: Option<LinearAttention>,
    /// The query projection also yields a per-feature gate of the attention's output,
    /// `o · sigmoid(gate)` (Qwen 3.5's `attn_output_gate`).
    pub attn_output_gate: bool,
    /// Features of each head the rotary embedding turns (`partial_rotary_factor` of
    /// `head_dim`).
    pub rope_dims: usize,
    /// Mixture-of-experts MLPs in every layer, for MoE models.
    pub moe: Option<MixtureOfExperts>,
    /// The quantization, for MLX-quantized checkpoints.
    pub quantization: Option<Quantization>,
    /// Ids that end generation.
    pub eos_token_ids: Vec<u32>,
}

/// A parsed JSON file.
fn read_json(path: &Path) -> Result<Json> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::InvalidArgument(format!("{}: {e}", path.display())))?;
    json::parse(&bytes)
}

/// A non-negative integer.
fn non_negative(k: &str, v: &Json) -> Result<usize> {
    let v = v
        .as_i64()
        .ok_or_else(|| invalid(format!("`{k}` is not an integer")))?;
    usize::try_from(v).map_err(|_| invalid(format!("`{k}` is negative ({v})")))
}

/// The ids of an `eos_token_id` entry (an id or a list of ids).
fn eos_ids(ids: Option<&Json>) -> Result<Vec<u32>> {
    let id = |v: &Json| {
        let id = non_negative("eos_token_id", v)?;
        u32::try_from(id).map_err(|_| invalid(format!("`eos_token_id` {id} is too large")))
    };
    match ids {
        None | Some(Json::Null) => Ok(Vec::new()),
        Some(Json::Array(ids)) => ids.iter().map(id).collect(),
        Some(single) => Ok(vec![id(single)?]),
    }
}

impl Config {
    /// Reads a `config.json` file and, when there is one next to it,
    /// `generation_config.json`: its `eos_token_id` (an id or a list) is added to the
    /// end-of-sequence ids, as mlx-lm does.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::read(path.as_ref())?.0)
    }

    /// [`Config::from_file`], and whether the file sets `tie_word_embeddings`.
    pub(crate) fn read(path: &Path) -> Result<(Self, bool)> {
        let top = read_json(path)?;
        let mut config = Self::from_json(&top)?;
        let generation = path.with_file_name("generation_config.json");
        if generation.is_file() {
            let ids = eos_ids(read_json(&generation)?.get("eos_token_id"))
                .map_err(|e| Error::InvalidArgument(format!("{}: {e}", generation.display())))?;
            for id in ids {
                if !config.eos_token_ids.contains(&id) {
                    config.eos_token_ids.push(id);
                }
            }
        }
        let c = top.get("text_config").unwrap_or(&top);
        let tied = top
            .get("tie_word_embeddings")
            .or_else(|| c.get("tie_word_embeddings"))
            .is_some_and(|t| *t != Json::Null);
        Ok((config, tied))
    }

    /// Builds a configuration from a parsed `config.json`. Multimodal checkpoints' text
    /// settings (`text_config`) are used; the quantization is read from the top level.
    /// Zero heads, intervals or group sizes and negative integers are errors.
    pub fn from_json(top: &Json) -> Result<Self> {
        let c = top.get("text_config").unwrap_or(top);
        let opt_int = |k: &str| match c.get(k) {
            None | Some(Json::Null) => Ok(None),
            Some(v) => non_negative(k, v).map(Some),
        };
        let int = |k: &str| opt_int(k)?.ok_or_else(|| invalid(format!("`{k}` missing")));
        let positive = |k: &str, v: usize| match v {
            0 => Err(invalid(format!("`{k}` must not be 0"))),
            v => Ok(v),
        };
        let float = |k: &str| c.get(k).and_then(Json::as_f64).map(|v| v as f32);
        let flag = |k: &str| c.get(k).and_then(Json::as_bool).unwrap_or(false);
        let model_type = top
            .get("model_type")
            .and_then(Json::as_str)
            .ok_or_else(|| invalid("`model_type` missing"))?;
        let model_type = match model_type {
            "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => "qwen3_5",
            other => other,
        }
        .to_owned();
        let (qkv_bias, o_bias, qk_norm) = match model_type.as_str() {
            "llama" | "mistral" => (flag("attention_bias"), flag("attention_bias"), false),
            "qwen2" => (true, false, false),
            "qwen3" | "qwen3_5" => (flag("attention_bias"), flag("attention_bias"), true),
            "qwen3_moe" => (false, false, true),
            other => return Err(invalid(format!("unsupported model type {other:?}"))),
        };
        let hybrid = model_type == "qwen3_5";
        let qwen3_moe = model_type == "qwen3_moe";
        let moe = match opt_int("num_experts")?.unwrap_or(0) {
            experts if (hybrid || qwen3_moe) && experts > 0 => {
                let experts_per_token =
                    positive("num_experts_per_tok", int("num_experts_per_tok")?)?;
                if experts_per_token > experts {
                    return Err(invalid(format!(
                        "{experts_per_token} experts per token of {experts}"
                    )));
                }
                Some(MixtureOfExperts {
                    experts,
                    experts_per_token,
                    intermediate_size: int("moe_intermediate_size")?,
                    shared_intermediate_size: match hybrid {
                        true => Some(int("shared_expert_intermediate_size")?),
                        false => None,
                    },
                    // mlx-lm's defaults: on for Qwen 3.5, off for Qwen 3.
                    norm_topk_prob: c
                        .get("norm_topk_prob")
                        .and_then(Json::as_bool)
                        .unwrap_or(hybrid),
                    sparse_step: opt_int("decoder_sparse_step")?.unwrap_or(1),
                    dense_layers: c
                        .get("mlp_only_layers")
                        .and_then(Json::as_array)
                        .unwrap_or_default()
                        .iter()
                        .map(|i| non_negative("mlp_only_layers", i))
                        .collect::<Result<_>>()?,
                })
            }
            _ => None,
        };
        // Qwen 3.5 keeps its rotary settings in `rope_parameters`.
        let rope = c.get("rope_parameters").filter(|r| **r != Json::Null);
        let rope_float = |k: &str| {
            rope.and_then(|r| r.get(k))
                .and_then(Json::as_f64)
                .map(|v| v as f32)
                .or_else(|| float(k))
        };
        let hidden_size = positive("hidden_size", int("hidden_size")?)?;
        let heads = positive("num_attention_heads", int("num_attention_heads")?)?;
        let kv_heads = positive(
            "num_key_value_heads",
            opt_int("num_key_value_heads")?.unwrap_or(heads),
        )?;
        let head_dim = positive(
            "head_dim",
            opt_int("head_dim")?.unwrap_or(hidden_size / heads),
        )?;
        let layers = int("num_hidden_layers")?;
        let layer_types = if hybrid {
            match c.get("layer_types").and_then(Json::as_array) {
                Some(types) => types
                    .iter()
                    .map(|t| match t.as_str() {
                        Some("full_attention") => Ok(LayerType::Attention),
                        Some("linear_attention") => Ok(LayerType::Linear),
                        other => Err(invalid(format!("unsupported layer type {other:?}"))),
                    })
                    .collect::<Result<Vec<_>>>()?,
                None => {
                    let interval = positive(
                        "full_attention_interval",
                        opt_int("full_attention_interval")?.unwrap_or(4),
                    )?;
                    (0..layers)
                        .map(|i| match (i + 1) % interval {
                            0 => LayerType::Attention,
                            _ => LayerType::Linear,
                        })
                        .collect()
                }
            }
        } else {
            vec![LayerType::Attention; layers]
        };
        if layer_types.len() != layers {
            return Err(invalid(format!(
                "{} layer types for {layers} layers",
                layer_types.len()
            )));
        }
        let linear_attention = if hybrid {
            let la = LinearAttention {
                key_heads: positive("linear_num_key_heads", int("linear_num_key_heads")?)?,
                value_heads: positive("linear_num_value_heads", int("linear_num_value_heads")?)?,
                key_head_dim: positive("linear_key_head_dim", int("linear_key_head_dim")?)?,
                value_head_dim: positive("linear_value_head_dim", int("linear_value_head_dim")?)?,
                conv_kernel: positive(
                    "linear_conv_kernel_dim",
                    opt_int("linear_conv_kernel_dim")?.unwrap_or(4),
                )?,
            };
            if !la.value_heads.is_multiple_of(la.key_heads) {
                return Err(invalid(
                    "linear value heads must be a multiple of key heads",
                ));
            }
            Some(la)
        } else {
            None
        };
        let partial = rope_float("partial_rotary_factor").unwrap_or(1.0);
        let rope_dims = (head_dim as f32 * partial) as usize;
        Ok(Self {
            hidden_size,
            num_hidden_layers: layers,
            intermediate_size: match &moe {
                // MoE configs may have no dense MLP width.
                Some(m) => opt_int("intermediate_size")?.unwrap_or(m.intermediate_size),
                None => int("intermediate_size")?,
            },
            moe,
            layer_types,
            linear_attention,
            attn_output_gate: hybrid
                && c.get("attn_output_gate").and_then(Json::as_bool) != Some(false),
            rope_dims,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            head_dim,
            vocab_size: positive("vocab_size", int("vocab_size")?)?,
            rms_norm_eps: float("rms_norm_eps").unwrap_or(1e-6),
            rope_theta: rope_float("rope_theta").unwrap_or(10000.0),
            rope_scaling: match (hybrid, qwen3_moe) {
                // `rope_parameters` also carries multimodal sections (text uses plain RoPE).
                (true, _) => rope_scaling(rope)?,
                // mlx-lm's Qwen 3 MoE ignores `rope_scaling`.
                (_, true) => RopeScaling::None,
                _ => rope_scaling(c.get("rope_scaling"))?,
            },
            rope_traditional: !matches!(model_type.as_str(), "qwen3" | "qwen3_moe")
                && flag("rope_traditional"),
            // Without the setting, [`crate::load_model`] infers it from the checkpoint.
            tie_word_embeddings: top
                .get("tie_word_embeddings")
                .or_else(|| c.get("tie_word_embeddings"))
                .and_then(Json::as_bool)
                .unwrap_or(false),
            qkv_bias,
            o_bias,
            mlp_bias: flag("mlp_bias"),
            qk_norm,
            quantization: quantization(top.get("quantization"))?,
            eos_token_ids: eos_ids(c.get("eos_token_id").or_else(|| top.get("eos_token_id")))?,
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
    // A group size of at least 1 and 1 to 8 bits.
    let settings = |j: &Json| -> Result<Option<(usize, u32)>> {
        let (Some(g), Some(b)) = (j.get("group_size"), j.get("bits")) else {
            return Ok(None);
        };
        let (g, b) = (
            non_negative("quantization.group_size", g)?,
            non_negative("quantization.bits", b)?,
        );
        if g == 0 || !(1..=8).contains(&b) {
            return Err(invalid(format!(
                "quantization: group size {g} and {b} bits (a group size of at least 1, \
                 1 to 8 bits)"
            )));
        }
        Ok(Some((g, b as u32)))
    };
    let Some((group_size, bits)) = settings(q)? else {
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
            let (g, b) = match settings(v) {
                Ok(s) => s?,
                Err(e) => return Some(Err(e)),
            };
            // Under the text model's names, as the loader renames weights.
            let path = match (
                path.strip_prefix("model.language_model."),
                path.strip_prefix("language_model."),
            ) {
                (Some(rest), _) => format!("model.{rest}"),
                (_, Some(rest)) => rest.to_owned(),
                _ => path.clone(),
            };
            Some(Ok((path, g, b)))
        })
        .collect::<Result<_>>()?;
    Ok(Some(Quantization {
        group_size,
        bits,
        layers,
    }))
}
