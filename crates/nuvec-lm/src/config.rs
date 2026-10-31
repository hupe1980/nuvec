//! Model configuration from Hugging Face `config.json` files.

use std::path::Path;

use nuvec::io::json::{self, Json};
use nuvec::{Error, Quant, QuantMode, Result};

fn invalid(msg: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("config.json: {msg}"))
}

/// How positions are scaled before the rotary embedding (nuvec-nn's, re-exported).
pub use nuvec_nn::RopeScaling;
use nuvec_nn::{Llama3, LongRope, Yarn};

/// The quantization of a checkpoint's linear and embedding layers (mlx-lm's `quantization`
/// config, or a mapped Hugging Face `quantization_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Quantization {
    /// The default (for affine layers the checkpoint does not describe; floating-point
    /// layers are recognised from their uint8 scales).
    pub default: Quant,
    /// Per-layer entries, under the text model's names.
    pub layers: Vec<LayerQuant>,
    /// A Hugging Face integer packing the loader converts to MLX's affine layout (AWQ,
    /// GPTQ); `None` for MLX checkpoints.
    pub packed: Option<PackedFormat>,
}

/// A Hugging Face checkpoint's integer packing (`quantization_config.quant_method`), which
/// the loader converts to MLX's affine layout (codes, scales, biases `-zero · scale`). The
/// bits and group size are [`Quantization::default`]'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackedFormat {
    /// AutoAWQ's GEMM layout (`"awq"`, version `gemm`): 4-bit `qweight` `[in, out / 8]`
    /// packed along the outputs in AWQ's order, `qzeros` likewise, `scales` `[groups, out]`.
    /// Converted as mlx-lm 0.32.2's `_transform_awq_weights`.
    Awq,
    /// GPTQ (`"gptq"`): 4- or 8-bit `qweight` `[in · bits / 32, out]` packed along the
    /// inputs, `qzeros` `[groups, out · bits / 32]` along the outputs, `scales`
    /// `[groups, out]`; checkpoint format v1 stores each zero minus one. A `g_idx` must be
    /// the identity grouping (no activation reordering).
    Gptq {
        /// Zeros stored minus one (`checkpoint_format` `gptq`, not `gptq_v2`).
        v1: bool,
    },
}

/// A layer's quantization: its settings (`None`: not quantized, mlx-lm's `false`) and
/// whether it has an nvfp4 global scale per expert (`global_scale: true`).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerQuant {
    /// The layer's path, as `model.layers.0.mlp.gate_proj`.
    pub path: String,
    /// The settings; `None` for a layer kept in full precision.
    pub quant: Option<Quant>,
    /// An nvfp4 switch layer with one global scale per expert.
    pub global_scale: bool,
}

impl Quantization {
    /// The entry of the layer at `path`, if the config names it.
    pub fn for_layer(&self, path: &str) -> Option<&LayerQuant> {
        self.layers.iter().find(|l| l.path == path)
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

/// The experts' gated activation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Activation {
    /// `silu(gate) · up` (mlx-lm's `swiglu`).
    SwiGlu,
    /// gpt-oss's clamped SwiGLU (mlx-lm's `swiglu_oai`).
    SwiGluOai {
        /// The sigmoid's slope (1.702).
        alpha: f32,
        /// The clamp (`swiglu_limit`, 7).
        limit: f32,
    },
    /// `gelu_approx(gate) · up` (Gemma's `geglu`).
    GeGlu,
}

impl nuvec::Static for Activation {
    fn hash_static(&self, h: &mut dyn std::hash::Hasher) {
        match self {
            Self::SwiGlu => h.write_u8(0),
            Self::SwiGluOai { alpha, limit } => {
                h.write_u8(1);
                h.write_u32(alpha.to_bits());
                h.write_u32(limit.to_bits());
            }
            Self::GeGlu => h.write_u8(2),
        }
    }
}

/// Mixture-of-experts MLPs (Qwen 3 MoE, Qwen 3.5 MoE, gpt-oss): each token's
/// `experts_per_token` best experts (by a softmax router), weighted, plus (Qwen 3.5) a
/// gated shared expert.
#[derive(Debug, Clone, PartialEq)]
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
    /// A bias on the router (gpt-oss).
    pub router_bias: bool,
    /// Biases on the experts' projections (gpt-oss).
    pub expert_bias: bool,
    /// The experts' activation.
    pub activation: Activation,
}

impl MixtureOfExperts {
    /// Whether layer `i` is a mixture of experts (mlx-lm's rule).
    pub fn is_sparse(&self, i: usize) -> bool {
        !self.dense_layers.contains(&i) && (i + 1).is_multiple_of(self.sparse_step.max(1))
    }
}

/// Gemma 4's text-model settings (mlx-lm's `gemma4_text`, its defaults where a setting is
/// absent). Sliding layers use [`Config::head_dim`] and [`Config::num_key_value_heads`].
#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4 {
    /// Features per head of the full-attention layers (`global_head_dim`).
    pub global_head_dim: usize,
    /// Key heads of the full-attention layers (`num_global_key_value_heads` with K = V,
    /// else the sliding layers').
    pub global_kv_heads: usize,
    /// Full-attention layers project keys only and use them as values too
    /// (`attention_k_eq_v`).
    pub k_eq_v: bool,
    /// The last layers project no keys and values: each reads those of the last earlier
    /// layer of its kind (`num_kv_shared_layers`).
    pub kv_shared_layers: usize,
    /// KV-shared layers' MLPs are twice as wide (`use_double_wide_mlp`).
    pub double_wide_mlp: bool,
    /// Width of each layer's per-layer input (`hidden_size_per_layer_input`; 0: none).
    pub per_layer_input: usize,
    /// Token ids of the per-layer input embedding (`vocab_size_per_layer_input`).
    pub per_layer_vocab: usize,
    /// The logits' tanh soft cap (`final_logit_softcapping`).
    pub final_logit_softcap: Option<f32>,
    /// The sliding layers' RoPE base and scaling (`rope_parameters.sliding_attention`).
    pub sliding_rope: (f32, RopeScaling),
    /// The full layers' RoPE base and scaling (`rope_parameters.full_attention`).
    pub full_rope: (f32, RopeScaling),
    /// The mixture-of-experts branch beside each layer's MLP (`enable_moe_block`): a
    /// normalised router, the top-k softmax scaled per expert, GeGLU experts.
    pub moe: Option<MixtureOfExperts>,
}

impl Gemma4 {
    /// The first KV-shared layer of `layers`.
    pub fn first_shared(&self, layers: usize) -> usize {
        layers.saturating_sub(self.kv_shared_layers)
    }
}

/// A decoder-only transformer of the Llama family (Llama, Mistral, Qwen 2, Qwen 3, Qwen 3 MoE,
/// gpt-oss, Gemma 4) or a hybrid of attention and Gated DeltaNet layers (Qwen 3.5).
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `llama`, `mistral`, `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_5`, `gpt_oss` or `gemma4`.
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
    /// Each layer's sliding window (`None`: full attention), from `layer_types`'
    /// `sliding_attention` and `sliding_window`.
    pub layer_windows: Vec<Option<usize>>,
    /// Learned attention sinks (one logit per query head, `self_attn.sinks`: gpt-oss).
    pub attention_sinks: bool,
    /// Gemma 4's settings.
    pub gemma4: Option<Gemma4>,
    /// Multi-token-prediction decoder layers (`mtp_num_hidden_layers`, Qwen 3.5–3.8; 0
    /// without). The head exists once its weights are loaded ([`crate::Model::has_mtp`]).
    pub mtp_layers: usize,
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
            "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => "gemma4",
            other => other,
        }
        .to_owned();
        let (qkv_bias, o_bias, qk_norm) = match model_type.as_str() {
            "llama" | "mistral" => (flag("attention_bias"), flag("attention_bias"), false),
            "qwen2" => (true, false, false),
            "qwen3" | "qwen3_5" => (flag("attention_bias"), flag("attention_bias"), true),
            "qwen3_moe" => (false, false, true),
            // mlx-lm's gpt-oss: biases on all four projections, no q/k norms.
            "gpt_oss" => (true, true, false),
            "gemma4" => (false, false, true),
            other => return Err(invalid(format!("unsupported model type {other:?}"))),
        };
        let gpt_oss = model_type == "gpt_oss";
        let gemma = model_type == "gemma4";
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
                    router_bias: false,
                    expert_bias: false,
                    activation: Activation::SwiGlu,
                })
            }
            _ if gpt_oss => {
                // mlx-lm's gpt-oss: experts in every layer, softmax over the top k, a biased
                // router, biased experts, the clamped SwiGLU (`swiglu_limit`, 7).
                let experts = positive("num_local_experts", int("num_local_experts")?)?;
                let experts_per_token = positive(
                    "num_experts_per_tok",
                    match opt_int("num_experts_per_tok")? {
                        Some(k) => k,
                        None => int("experts_per_token")?,
                    },
                )?;
                if experts_per_token > experts {
                    return Err(invalid(format!(
                        "{experts_per_token} experts per token of {experts}"
                    )));
                }
                Some(MixtureOfExperts {
                    experts,
                    experts_per_token,
                    intermediate_size: int("intermediate_size")?,
                    shared_intermediate_size: None,
                    norm_topk_prob: true,
                    sparse_step: 1,
                    dense_layers: Vec::new(),
                    router_bias: true,
                    expert_bias: true,
                    activation: Activation::SwiGluOai {
                        alpha: 1.702,
                        limit: float("swiglu_limit").unwrap_or(7.0),
                    },
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
        // mlx-lm's Gemma 4 defaults: 8 query heads, 1 key head, head size 256.
        let heads = positive(
            "num_attention_heads",
            match gemma {
                true => opt_int("num_attention_heads")?.unwrap_or(8),
                false => int("num_attention_heads")?,
            },
        )?;
        let kv_heads = positive(
            "num_key_value_heads",
            opt_int("num_key_value_heads")?.unwrap_or(if gemma { 1 } else { heads }),
        )?;
        let head_dim = positive(
            "head_dim",
            opt_int("head_dim")?.unwrap_or(if gemma { 256 } else { hidden_size / heads }),
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
        // Sliding windows (gpt-oss: `layer_types`, or alternating sliding and full; Gemma 4:
        // `layer_types`, or `sliding_window_pattern − 1` sliding layers per full one).
        let layer_windows = if gpt_oss || gemma {
            let window = positive(
                "sliding_window",
                match gemma {
                    true => opt_int("sliding_window")?.unwrap_or(512),
                    false => int("sliding_window")?,
                },
            )?;
            let pattern = positive(
                "sliding_window_pattern",
                opt_int("sliding_window_pattern")?.unwrap_or(5),
            )?;
            let kinds: Vec<bool> = match c.get("layer_types").and_then(Json::as_array) {
                Some(types) => types
                    .iter()
                    .map(|t| match t.as_str() {
                        Some("sliding_attention") => Ok(true),
                        Some("full_attention") => Ok(false),
                        other => Err(invalid(format!("unsupported layer type {other:?}"))),
                    })
                    .collect::<Result<_>>()?,
                None if gemma => (0..layers).map(|i| (i + 1) % pattern != 0).collect(),
                None => (0..layers).map(|i| i % 2 == 0).collect(),
            };
            if kinds.len() != layers {
                return Err(invalid(format!(
                    "{} layer types for {layers} layers",
                    kinds.len()
                )));
            }
            kinds.into_iter().map(|s| s.then_some(window)).collect()
        } else {
            vec![None; layers]
        };
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
        let gemma4 = match gemma {
            true => Some(gemma4(c, &opt_int, head_dim, kv_heads, layers)?),
            false => None,
        };
        // A Gemma 4 wrapper sets its text model's vocabulary (mlx-lm's `gemma4.ModelArgs`).
        let vocab_size = match (gemma, top.get("text_config")) {
            (true, Some(_)) => top
                .get("vocab_size")
                .map(|v| non_negative("vocab_size", v))
                .transpose()?
                .unwrap_or(262_144),
            (true, None) => opt_int("vocab_size")?.unwrap_or(262_144),
            (false, _) => int("vocab_size")?,
        };
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
            layer_windows,
            attention_sinks: gpt_oss,
            linear_attention,
            attn_output_gate: hybrid
                && c.get("attn_output_gate").and_then(Json::as_bool) != Some(false),
            rope_dims,
            num_attention_heads: heads,
            num_key_value_heads: kv_heads,
            head_dim,
            vocab_size: positive("vocab_size", vocab_size)?,
            rms_norm_eps: float("rms_norm_eps").unwrap_or(1e-6),
            // mlx-lm's defaults when the key is absent.
            rope_theta: rope_float("rope_theta").unwrap_or(match model_type.as_str() {
                "qwen2" => 1e6,
                "gpt_oss" => 150_000.0,
                "qwen3_5" | "qwen3_5_moe" => 100_000.0,
                _ => 10_000.0,
            }),
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
                .unwrap_or(gemma),
            qkv_bias,
            o_bias,
            mlp_bias: flag("mlp_bias"),
            qk_norm,
            gemma4,
            mtp_layers: opt_int("mtp_num_hidden_layers")?.unwrap_or(0),
            quantization: quantization(
                top.get("quantization"),
                top.get("quantization_config").or_else(|| {
                    top.get("text_config")
                        .and_then(|t| t.get("quantization_config"))
                }),
                &model_type,
            )?,
            eos_token_ids: eos_ids(c.get("eos_token_id").or_else(|| top.get("eos_token_id")))?,
            model_type,
        })
    }
}

/// Gemma 4's settings from its text config `c` (mlx-lm's defaults).
fn gemma4(
    c: &Json,
    opt_int: &dyn Fn(&str) -> Result<Option<usize>>,
    head_dim: usize,
    kv_heads: usize,
    layers: usize,
) -> Result<Gemma4> {
    let flag = |k: &str, default: bool| c.get(k).and_then(Json::as_bool).unwrap_or(default);
    // 512 when absent; the sliding head size when null or 0 (mlx-lm's truthiness test).
    let global_head_dim = match c.get("global_head_dim") {
        None => 512,
        Some(Json::Null) => head_dim,
        Some(v) => match non_negative("global_head_dim", v)? {
            0 => head_dim,
            d => d,
        },
    };
    let k_eq_v = flag("attention_k_eq_v", false);
    let global_kv_heads = match (k_eq_v, opt_int("num_global_key_value_heads")?) {
        (true, Some(h)) => h,
        _ => kv_heads,
    };
    let kv_shared_layers = opt_int("num_kv_shared_layers")?.unwrap_or(20);
    if kv_shared_layers >= layers && kv_shared_layers > 0 {
        return Err(invalid(format!(
            "{kv_shared_layers} KV-shared layers of {layers}"
        )));
    }
    // Per-kind RoPE (`rope_parameters`, mlx-lm's defaults): the default type rotates every
    // feature; `proportional` the `partial_rotary_factor` of the kind's head.
    let params = c.get("rope_parameters").filter(|r| **r != Json::Null);
    // mlx-lm's `ModelArgs` defaults apply only without `rope_parameters`; a kind it lacks is
    // the default RoPE at base 10,000 (mlx-lm's `{}`).
    let rope = |kind: &str, dims: usize, theta: f32, default: &str, partial: f64| {
        let p = params.map(|p| p.get(kind).filter(|e| **e != Json::Null));
        let get = |k: &str| p.flatten().and_then(|p| p.get(k));
        let (theta, default, partial) = match p {
            Some(_) => (10_000.0, "default", 1.0),
            None => (theta, default, partial),
        };
        let base = get("rope_theta")
            .and_then(Json::as_f64)
            .map_or(theta, |v| v as f32);
        let kind_name = get("type")
            .or_else(|| get("rope_type"))
            .and_then(Json::as_str)
            .unwrap_or(default);
        let scaling = match kind_name {
            "proportional" => {
                let partial = get("partial_rotary_factor")
                    .and_then(Json::as_f64)
                    .unwrap_or(partial);
                let rotated = (dims as f64 * partial) as usize;
                if rotated > dims {
                    return Err(invalid(format!(
                        "{kind}: {rotated} rotated dims of a {dims}-dim head"
                    )));
                }
                RopeScaling::Proportional {
                    rotated_dims: i32::try_from(rotated)
                        .map_err(|_| invalid("rotated dims too large"))?,
                    factor: get("factor").and_then(Json::as_f64).unwrap_or(1.0) as f32,
                }
            }
            _ => rope_scaling(p.flatten())?,
        };
        Ok::<_, Error>((base, scaling))
    };
    let moe = match flag("enable_moe_block", false) {
        true => {
            let experts = opt_int("num_experts")?
                .filter(|&e| e > 0)
                .ok_or_else(|| invalid("`enable_moe_block` without `num_experts`"))?;
            let top_k = opt_int("top_k_experts")?
                .filter(|&k| (1..=experts).contains(&k))
                .ok_or_else(|| invalid(format!("`top_k_experts` must be 1 to {experts}")))?;
            Some(MixtureOfExperts {
                experts,
                experts_per_token: top_k,
                intermediate_size: opt_int("moe_intermediate_size")?
                    .ok_or_else(|| invalid("`moe_intermediate_size` missing"))?,
                shared_intermediate_size: None,
                norm_topk_prob: true,
                sparse_step: 1,
                dense_layers: Vec::new(),
                router_bias: false,
                expert_bias: false,
                activation: Activation::GeGlu,
            })
        }
        false => None,
    };
    let per_layer_input = match c.get("hidden_size_per_layer_input") {
        Some(Json::Null) => 0,
        _ => opt_int("hidden_size_per_layer_input")?.unwrap_or(256),
    };
    Ok(Gemma4 {
        global_head_dim,
        global_kv_heads,
        k_eq_v,
        kv_shared_layers,
        double_wide_mlp: flag("use_double_wide_mlp", true),
        per_layer_input,
        per_layer_vocab: opt_int("vocab_size_per_layer_input")?.unwrap_or(262_144),
        final_logit_softcap: match c.get("final_logit_softcapping") {
            Some(Json::Null) => None,
            Some(v) => v.as_f64().map(|v| v as f32),
            None => Some(30.0),
        },
        sliding_rope: rope("sliding_attention", head_dim, 10_000.0, "default", 1.0)?,
        full_rope: rope("full_attention", global_head_dim, 1e6, "proportional", 0.25)?,
        moe,
    })
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
        "llama3" => Ok(RopeScaling::Llama3(Llama3 {
            factor: float("factor", 1.0),
            low_freq_factor: float("low_freq_factor", 1.0),
            high_freq_factor: float("high_freq_factor", 4.0),
            original_max_position_embeddings: float("original_max_position_embeddings", 8192.0),
        })),
        // mlx-lm's `YarnRoPE` (Hugging Face's `truncate` is ignored, as by mlx-lm).
        "yarn" | "deepseek_yarn" => Ok(RopeScaling::Yarn(Yarn {
            factor: float("factor", 1.0),
            original_max_position_embeddings: float("original_max_position_embeddings", 4096.0),
            beta_fast: float("beta_fast", 32.0),
            beta_slow: float("beta_slow", 1.0),
            mscale: float("mscale", 1.0),
            mscale_all_dim: float("mscale_all_dim", 0.0),
        })),
        // mlx-lm's `SuScaledRoPE` (the long factors).
        "longrope" | "su" => {
            let factors = match s.get("long_factor") {
                Some(Json::Array(v)) => v
                    .iter()
                    .filter_map(Json::as_f64)
                    .map(|v| v as f32)
                    .collect(),
                Some(v) => vec![v.as_f64().unwrap_or(1.0) as f32],
                None => vec![1.0],
            };
            Ok(RopeScaling::LongRope(LongRope {
                long_factor: factors,
                max_position_embeddings: float("max_position_embeddings", 131_072.0),
                original_max_position_embeddings: float("original_max_position_embeddings", 4096.0),
                long_mscale: s
                    .get("long_mscale")
                    .and_then(Json::as_f64)
                    .map(|v| v as f32),
            }))
        }
        other => Err(invalid(format!("unsupported rope scaling {other:?}"))),
    }
}

fn quantization(
    q: Option<&Json>,
    hf: Option<&Json>,
    model_type: &str,
) -> Result<Option<Quantization>> {
    let Some(q) = q.filter(|q| **q != Json::Null) else {
        return hf_quantization(hf, model_type);
    };
    // A mode (default affine), then its group size and bits (MLX's defaults for the mode
    // when absent); MLX's checks of the combination run in the operations.
    let settings = |j: &Json| -> Result<Quant> {
        let mode = match j.get("mode").and_then(Json::as_str) {
            None => QuantMode::Affine,
            Some(m) => m
                .parse::<QuantMode>()
                .map_err(|_| invalid(format!("unsupported quantization mode {m:?}")))?,
        };
        let mut quant = Quant::new(mode);
        if let Some(g) = j.get("group_size") {
            quant = quant.group_size(int("quantization.group_size", g)?);
        }
        if let Some(b) = j.get("bits") {
            quant = quant.bits(int("quantization.bits", b)?);
        }
        if quant.group_size < 1 || !(1..=8).contains(&quant.bits) {
            return Err(invalid(format!(
                "quantization: group size {} and {} bits (a group size of at least 1, 1 to 8 \
                 bits)",
                quant.group_size, quant.bits
            )));
        }
        Ok(quant)
    };
    if q.get("group_size").is_none() && q.get("bits").is_none() && q.get("mode").is_none() {
        return Err(invalid(
            "quantization needs `group_size` and `bits` (or a `mode`)",
        ));
    }
    let default = settings(q)?;
    let layers = q
        .as_object()
        .unwrap_or_default()
        .iter()
        .filter(|(_, v)| v.as_object().is_some() || v.as_bool() == Some(false))
        .map(|(path, v)| {
            // Under the text model's names, as the loader renames weights.
            let path = match (
                path.strip_prefix("model.language_model."),
                path.strip_prefix("language_model."),
            ) {
                (Some(rest), _) => format!("model.{rest}"),
                (_, Some(rest)) => rest.to_owned(),
                _ => path.clone(),
            };
            let quant = match v.as_bool() {
                Some(false) => None,
                _ => Some(settings(v)?),
            };
            let global_scale = v.get("global_scale").and_then(Json::as_bool) == Some(true);
            Ok(LayerQuant {
                path,
                quant,
                global_scale,
            })
        })
        .collect::<Result<_>>()?;
    Ok(Some(Quantization {
        default,
        layers,
        packed: None,
    }))
}

/// A Hugging Face `quantization_config` (checkpoints not converted by MLX): AWQ and GPTQ
/// integer packings, converted to MLX's affine layout at load (mlx-lm 0.32.2 converts AWQ
/// the same way; its GPTQ path reads GPTQ as AWQ's layout, which nuvec does not copy).
/// Formats whose tensors need a per-architecture converter (MXFP4 `_blocks`, FP8,
/// compressed-tensors, ModelOpt) are refused with a message instead of loading packed
/// codes as weights.
fn hf_quantization(q: Option<&Json>, model_type: &str) -> Result<Option<Quantization>> {
    let Some(q) = q.filter(|q| **q != Json::Null) else {
        return Ok(None);
    };
    let method = q.get("quant_method").and_then(Json::as_str).unwrap_or("");
    let int_or = |key: &str, default: i64| -> Result<i64> {
        match q.get(key) {
            None | Some(Json::Null) => Ok(default),
            Some(v) => v
                .as_i64()
                .ok_or_else(|| invalid(format!("quantization_config.{key} must be an integer"))),
        }
    };
    let (bits, group) = (int_or("bits", 4)?, int_or("group_size", 128)?);
    let affine = |bits: i64| -> Result<Quant> {
        if ![32, 64, 128].contains(&group) {
            return Err(invalid(format!(
                "{method}: group size {group} (MLX's affine kernels take 32, 64 or 128)"
            )));
        }
        Ok(Quant::affine(group as i32, bits as i32))
    };
    let (default, packed) = match method {
        "awq" => {
            let version = q.get("version").and_then(Json::as_str).unwrap_or("gemm");
            if !version.eq_ignore_ascii_case("gemm") || bits != 4 {
                return Err(invalid(format!(
                    "awq: version {version:?} with {bits} bits (4-bit GEMM checkpoints load)"
                )));
            }
            (affine(4)?, PackedFormat::Awq)
        }
        "gptq" => {
            if bits != 4 && bits != 8 {
                return Err(invalid(format!("gptq: {bits} bits (4 and 8 load)")));
            }
            if q.get("desc_act").and_then(Json::as_bool) == Some(true) {
                return Err(invalid(
                    "gptq: activation reordering (`desc_act`) is not supported; use a \
                     checkpoint quantized without it",
                ));
            }
            let format = q
                .get("checkpoint_format")
                .and_then(Json::as_str)
                .unwrap_or("gptq");
            (
                affine(bits)?,
                PackedFormat::Gptq {
                    v1: format != "gptq_v2",
                },
            )
        }
        // gpt-oss's originals: MXFP4 experts as `_blocks`/`_scales`, which the loader
        // converts (mlx-lm's `gpt_oss.sanitize`); the other layers are full precision.
        "mxfp4" if model_type == "gpt_oss" => {
            return Ok(Some(Quantization {
                default: Quant::new(QuantMode::Mxfp4),
                layers: Vec::new(),
                packed: None,
            }));
        }
        "mxfp4" | "mxfp8" | "fp8" | "compressed-tensors" | "modelopt" => {
            return Err(invalid(format!(
                "Hugging Face {method} checkpoints are not converted yet; use an MLX \
                 conversion (mlx_lm.convert, or an mlx-community checkpoint)"
            )));
        }
        other => {
            return Err(invalid(format!(
                "unsupported quantization method {other:?}"
            )));
        }
    };
    Ok(Some(Quantization {
        default,
        layers: Vec::new(),
        packed: Some(packed),
    }))
}

/// An integer setting (negative and fractional values are errors).
fn int(name: &str, v: &Json) -> Result<i32> {
    i32::try_from(non_negative(name, v)?).map_err(|_| invalid(format!("{name}: out of range")))
}
