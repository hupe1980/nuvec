//! DFlash block drafters (z-lab's DFlash v1 and DFlash 2, arXiv 2602.06036; ported from
//! z-lab's MLX code, MIT): a small Qwen3-style stack that drafts a whole block in one forward
//! pass, reading the target's intermediate layers. The target's selected layer outputs are
//! projected (`hidden_norm(fc(…))`) into context keys and values of every drafter layer ("KV
//! injection"); the block `[anchor, mask, …, mask]` goes through the target's embedding, the
//! drafter's layers and the target's output head. DFlash v1 drafts the argmax of each masked
//! position; DFlash 2 wraps attention and MLP in grouped dynamic causal convolutions and
//! drafts its candidate selector's greedy path. Only the context's keys and values are
//! cached. YaRN RoPE, attention biases, embedding scales and logit soft caps follow the
//! drafter's config (gpt-oss and Gemma 4 drafters).
//!
//! DSpark (DeepSeek's DeepSpec `Qwen3DSparkModel`, arXiv 2607.05147, MIT) shares the
//! backbone: every slot of `[anchor, mask, …]` predicts the next token, drafted one after
//! another with a Markov head's bias from the previous token (`w2 · w1[prev]`), and a
//! confidence head scores each drafted position.

use std::collections::BTreeMap;
use std::path::Path;

use nuvec::fast::AttentionOptions;
use nuvec::io::json::{self, Json};
use nuvec::{Array, DType, Error, Param, Quant, Result, ops};
use nuvec_nn::{Linear, QuantizedLinear, RmsNorm, Rope, RopeScaling};

use crate::model::{Model, Projection};
use crate::speculative::{Auto, Candidates, Draft, Drafter};

/// A DFlash drafter's settings (`config.json`: Qwen3 fields and `dflash_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashConfig {
    /// The drafter's (and the target's) hidden size.
    pub hidden_size: usize,
    /// Drafter layers.
    pub layers: usize,
    /// Query heads.
    pub heads: usize,
    /// Key and value heads.
    pub kv_heads: usize,
    /// Features per head.
    pub head_dim: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// RoPE base.
    pub rope_theta: f32,
    /// RoPE scaling (`rope_parameters` or `rope_scaling`: YaRN for gpt-oss drafters).
    pub rope_scaling: RopeScaling,
    /// Tokens per block: the anchor and `block_size − 1` draft positions.
    pub block_size: usize,
    /// The target's decoder layers read (0-indexed).
    pub target_layer_ids: Vec<usize>,
    /// The token whose (target) embedding fills the draft positions.
    pub mask_token_id: u32,
    /// Per layer: a sliding window of this many positions (`sliding_attention`), or none.
    pub windows: Vec<Option<usize>>,
    /// Whether the block attends causally within itself (sliding layers by default).
    pub causal: Vec<bool>,
    /// Multiplies the logits (`output_multiplier`).
    pub output_multiplier: f32,
    /// Multiplies the embedded block (`input_embedding_scale`, on top of the target
    /// embedding's own scale).
    pub input_embedding_scale: f32,
    /// Soft-caps the logits (`final_logit_softcapping`: `tanh(l / cap) · cap`).
    pub final_logit_softcap: Option<f32>,
    /// DFlash 2 (`DFlash2DraftModel`): the grouped dynamic causal convolutions' kernel size
    /// and group size, and the candidate selector's rank and top-k.
    pub v2: Option<DFlash2>,
    /// DSpark (`Qwen3DSparkModel`): every block slot drafts, with the Markov head's rank (0:
    /// none) and whether a confidence head scores the drafts.
    pub dspark: Option<DSpark>,
    /// The vocabulary.
    pub vocab_size: usize,
}

/// A checkpoint's tensors by name, each mapped (or read) when taken.
struct Lazy {
    index: BTreeMap<String, std::path::PathBuf>,
}

impl Lazy {
    /// The tensor `name`, if the checkpoint has it (taken once).
    fn remove(&mut self, name: &str) -> Result<Option<Array>> {
        let Some(path) = self.index.remove(name) else {
            return Ok(None);
        };
        Ok(nuvec::io::load_safetensors_filtered(&path, |n| n == name)?
            .tensors
            .remove(name))
    }
}

/// DSpark's settings ([`DFlashConfig::dspark`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DSpark {
    /// The Markov head's rank (0: no Markov head).
    pub markov_rank: usize,
    /// Whether a confidence head scores the drafts (with the Markov head's embedding of the
    /// previous token, `confidence_head_with_markov`).
    pub confidence: bool,
}

/// DFlash 2's settings ([`DFlashConfig::v2`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlash2 {
    /// Positions each convolution reads (the current one and the ones before).
    pub conv_kernel_size: usize,
    /// Features per convolution group.
    pub conv_group_size: usize,
    /// The selector's codebook rank.
    pub selector_rank: usize,
    /// Candidates per position the selector scores.
    pub selector_top_k: usize,
}

impl DFlashConfig {
    /// Parses a DFlash `config.json` (`architectures: ["DFlashDraftModel"]`,
    /// `["DFlash2DraftModel"]` or DeepSpec's `["Qwen3DSparkModel"]`). DFlash settings are read
    /// from `dflash_config`, then the top level (as z-lab's reference).
    pub fn from_json(c: &Json) -> Result<Self> {
        let invalid = |m: String| Error::InvalidArgument(format!("DFlash config: {m}"));
        let arch = c
            .get("architectures")
            .and_then(Json::as_array)
            .and_then(|a| a.first())
            .and_then(Json::as_str)
            .unwrap_or("");
        let (v2, dspark) = match arch {
            "DFlashDraftModel" => (false, false),
            "DFlash2DraftModel" => (true, false),
            "Qwen3DSparkModel" => (false, true),
            _ => {
                return Err(invalid(format!(
                    "not a DFlash drafter (architectures {arch:?}; DFlashDraftModel, \
                     DFlash2DraftModel and Qwen3DSparkModel are supported)"
                )));
            }
        };
        let int = |k: &str| {
            c.get(k)
                .and_then(Json::as_i64)
                .map(|v| v as usize)
                .ok_or_else(|| invalid(format!("`{k}` missing")))
        };
        let null = Json::Null;
        let d = c.get("dflash_config").unwrap_or(&null);
        // A DFlash setting: in `dflash_config`, else at the top level.
        let setting = |k: &str| d.get(k).or_else(|| c.get(k)).filter(|v| **v != Json::Null);
        let float = |k: &str| setting(k).and_then(Json::as_f64).map(|v| v as f32);
        let count = |k: &str| {
            setting(k)
                .and_then(Json::as_i64)
                .filter(|&v| v > 0)
                .map(|v| v as usize)
                .ok_or_else(|| invalid(format!("`{k}` missing or not positive")))
        };
        let layers = int("num_hidden_layers")?;
        let types: Vec<String> = match c.get("layer_types").and_then(Json::as_array) {
            Some(t) => t
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_owned())
                .collect(),
            None => vec!["full_attention".into(); layers],
        };
        if types.len() != layers {
            return Err(invalid("`layer_types` does not match the layers".into()));
        }
        let window = c
            .get("sliding_window")
            .and_then(Json::as_i64)
            .map(|w| w as usize);
        let explicit_causal = c.get("is_causal").and_then(Json::as_bool);
        let mut windows = Vec::new();
        let mut causal = Vec::new();
        for t in &types {
            let sliding = match t.as_str() {
                "full_attention" => false,
                "sliding_attention" => true,
                other => return Err(invalid(format!("unsupported layer type {other:?}"))),
            };
            if sliding && window.is_none() {
                return Err(invalid("sliding layers need `sliding_window`".into()));
            }
            windows.push(window.filter(|_| sliding));
            causal.push(explicit_causal.unwrap_or(sliding));
        }
        let rope = c
            .get("rope_theta")
            .and_then(Json::as_f64)
            .or_else(|| {
                c.get("rope_parameters")
                    .and_then(|r| r.get("rope_theta"))
                    .and_then(Json::as_f64)
            })
            .unwrap_or(10_000.0);
        // As z-lab's loader: `rope_parameters`, else `rope_scaling`.
        let scaling = c
            .get("rope_parameters")
            .filter(|r| **r != Json::Null)
            .or_else(|| c.get("rope_scaling"));
        let rope_scaling =
            crate::config::rope_scaling(scaling).map_err(|e| invalid(format!("RoPE: {e}")))?;
        let ids = setting("target_layer_ids")
            .and_then(Json::as_array)
            .ok_or_else(|| invalid("`target_layer_ids` missing".into()))?
            .iter()
            .map(|v| v.as_i64().map(|v| v as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("`target_layer_ids` must be integers".into()))?;
        let block = setting("block_size").and_then(Json::as_i64).unwrap_or(16) as usize;
        let v2 = match v2 {
            true => Some(DFlash2 {
                conv_kernel_size: count("conv_kernel_size")?,
                conv_group_size: count("conv_group_size")?,
                selector_rank: count("selector_rank")?,
                selector_top_k: count("selector_top_k")?,
            }),
            false => None,
        };
        let dspark = match dspark {
            false => None,
            true => {
                let rank = setting("markov_rank").and_then(Json::as_i64).unwrap_or(0) as usize;
                let head = setting("markov_head_type").and_then(Json::as_str);
                if rank > 0 && head.unwrap_or("vanilla") != "vanilla" {
                    return Err(invalid(format!(
                        "Markov head {head:?}: only \"vanilla\" is supported"
                    )));
                }
                let flag = |k: &str| setting(k).and_then(Json::as_bool).unwrap_or(false);
                let confidence = flag("enable_confidence_head");
                if confidence && !(rank > 0 && flag("confidence_head_with_markov")) {
                    return Err(invalid(
                        "a confidence head without the Markov head's embedding".into(),
                    ));
                }
                Some(DSpark {
                    markov_rank: rank,
                    confidence,
                })
            }
        };
        let hidden = int("hidden_size")?;
        if let Some(v) = v2.filter(|v| !hidden.is_multiple_of(v.conv_group_size)) {
            return Err(invalid(format!(
                "conv_group_size {} does not divide the hidden size {hidden}",
                v.conv_group_size
            )));
        }
        Ok(Self {
            hidden_size: hidden,
            layers,
            heads: int("num_attention_heads")?,
            kv_heads: int("num_key_value_heads")?,
            head_dim: int("head_dim")?,
            eps: c.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            rope_theta: rope as f32,
            rope_scaling,
            block_size: block.max(2),
            target_layer_ids: ids,
            mask_token_id: setting("mask_token_id")
                .and_then(Json::as_i64)
                .ok_or_else(|| invalid("`mask_token_id` missing".into()))?
                as u32,
            windows,
            causal,
            output_multiplier: float("output_multiplier").unwrap_or(1.0),
            input_embedding_scale: float("input_embedding_scale").unwrap_or(1.0),
            final_logit_softcap: float("final_logit_softcapping").filter(|&c| c > 0.0),
            v2,
            dspark,
            vocab_size: int("vocab_size")?,
        })
    }
}

/// One drafter layer: Qwen3's pre-norm attention and MLP, the attention's keys and values
/// also taken from the projected context.
#[derive(Debug, Clone)]
struct Layer {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_proj: Projection,
    up_proj: Projection,
    down_proj: Projection,
    /// DFlash 2: the convolutions around the attention and the MLP.
    convs: Option<(Conv, Conv)>,
}

/// DFlash 2's grouped dynamic causal convolution (z-lab's `GroupedDynamicCausalConv`): over
/// the block's positions, each group of features mixes the current and earlier positions
/// with a learned kernel plus one projected from the input.
#[derive(Debug, Clone)]
struct Conv {
    /// `[2, kernel, hidden]`: the kernels of `prepare` and `finish`.
    base: Array,
    /// `hidden → 2 · kernel · groups`.
    projection: Projection,
    kernel: usize,
    group: usize,
}

impl Conv {
    /// `Σ_o (base[o] + dynamic[o]) · x` shifted by `o` positions (zeros before the block), per
    /// group of features (the reference's `_grouped_dynamic_convolve`, in its order).
    fn convolve(&self, x: &Array, dynamic: &Array, base: &Array) -> Array {
        let [b, l, h] = x.shape_n::<3>().unwrap_or_default();
        let groups = h / self.group as i32;
        let blocks = x.reshape([b, l, groups, self.group as i32]);
        let dynamic = dynamic.reshape([b, l, self.kernel as i32, groups, 1]);
        let mut out = blocks.zeros_like();
        for o in 0..self.kernel as i32 {
            let values = match o {
                0 => blocks.clone(),
                _ => ops::concatenate(
                    &[
                        blocks.i((.., 0..o.min(l))).zeros_like(),
                        blocks.i((.., 0..(l - o).max(0))),
                    ],
                    1,
                ),
            };
            let kernel = base
                .i(o)
                .reshape([1, 1, groups, self.group as i32])
                .astype(x.dtype());
            out = out.add(kernel.multiply(&values));
            out = out.add(dynamic.i((.., .., o)).multiply(&values));
        }
        out.reshape([b, l, h])
    }

    /// The convolution before the sublayer, and the dynamic kernel of the one after.
    fn prepare(&self, x: &Array) -> (Array, Array) {
        let [b, l, h] = x.shape_n::<3>().unwrap_or_default();
        let groups = h / self.group as i32;
        let dynamic = self
            .projection
            .forward(x)
            .reshape([b, l, 2, self.kernel as i32, groups]);
        (
            self.convolve(x, &dynamic.i((.., .., 0)), &self.base.i(0)),
            dynamic.i((.., .., 1)),
        )
    }

    /// Checks the base kernel's shape (`[2, kernel, hidden]`).
    fn checked(self, name: &str, hidden: usize) -> Result<Self> {
        let want = [2, self.kernel as i32, hidden as i32];
        match self.base.shape() == want {
            true => Ok(self),
            false => Err(Error::InvalidArgument(format!(
                "DFlash drafter: `{name}.base_kernel` has shape {:?}, not {want:?}",
                self.base.shape()
            ))),
        }
    }

    /// The convolution after the sublayer.
    fn finish(&self, x: &Array, dynamic: &Array) -> Array {
        self.convolve(x, dynamic, &self.base.i(1))
    }
}

/// DFlash 2's candidate selector (z-lab's `CandidateSelector`): per draft position the top-k
/// tokens of the logits, rescored with a low-rank bigram term between the previous chosen
/// token and each candidate, conditioned on the position's hidden state; the greedy path
/// from the anchor.
#[derive(Debug, Clone)]
struct Selector {
    /// `[vocab, rank]`.
    predecessor: Array,
    /// `[vocab, rank]`.
    successor: Array,
    /// `hidden → rank`.
    projection: Projection,
    top_k: usize,
}

impl Selector {
    /// The greedy path (`[n]` uint32) after `anchor` over `hidden` (`[1, n, H]`) and its
    /// `logits` (`[1, n, vocab]`).
    fn select(&self, hidden: &Array, logits: &Array, anchor: u32) -> Array {
        let k = self.top_k as i32;
        let n = hidden.shape()[1];
        let candidates = ops::argpartition(logits, -k, -1).i((.., .., -k..));
        let unary = ops::take_along_axis(logits, &candidates, -1);
        let h = self.projection.forward(hidden);
        let mut previous = Array::from_slice(&[anchor], [1]);
        let mut path = Vec::with_capacity(n as usize);
        for p in 0..n {
            let row = candidates.i((.., p));
            let edges = ops::take(&self.predecessor, &previous, 0)
                .expand_dims(1)
                .multiply(h.i((.., p)).expand_dims(1))
                .multiply(ops::take(&self.successor, &row, 0))
                .sum(-1);
            let scores = unary.i((.., p)).add(&edges);
            let chosen = scores.argmax(-1);
            previous = ops::take_along_axis(&row, &chosen.expand_dims(-1), -1).reshape([1]);
            path.push(previous.clone());
        }
        ops::concatenate(&path, 0).astype(DType::U32)
    }
}

/// A DFlash drafter for one target (see the module docs); [`Drafter`] for
/// [`crate::speculative_generate`].
pub struct DFlashDrafter {
    config: DFlashConfig,
    fc: Projection,
    hidden_norm: RmsNorm,
    layers: Vec<Layer>,
    norm: RmsNorm,
    /// The rotation (with YaRN's scaling where the drafter has it).
    rope: Rope,
    /// DFlash 2's candidate selector.
    selector: Option<Selector>,
    /// DSpark's Markov head: the previous token's embedding (`w1`, `[V, r]`) and its
    /// projection to logits (`w2`).
    markov: Option<(Array, Projection)>,
    /// DSpark's confidence head (`[h ; w1[prev]] → 1`).
    confidence: Option<Projection>,
    /// The last draft's confidence logits (`[n]`), with a confidence head.
    last_confidence: Option<Array>,
    /// Per layer: the context's keys and values (`[1, kv_heads, S, head_dim]`), rotated at
    /// their positions; sliding layers keep the last `window − 1`.
    cache: Vec<Option<(Array, Array)>>,
    /// Context positions observed (the anchor's position).
    positions: i32,
}

impl DFlashDrafter {
    /// Loads a drafter directory (`config.json`, `*.safetensors`: z-lab's layout, bf16 or
    /// MLX-quantized) for `model`; an error if it does not fit the target (hidden size,
    /// vocabulary, layer ids).
    pub fn load(dir: impl AsRef<Path>, model: &Model) -> Result<Self> {
        Self::load_with(dir, model, None)
    }

    /// [`DFlashDrafter::load`], quantizing full-precision projections with `quant` (affine
    /// 4-bit groups of 64 read a bf16 drafter's weights at 0.28× the bytes per draft, as
    /// dflash-mlx does by default).
    pub fn load_quantized(dir: impl AsRef<Path>, model: &Model, quant: Quant) -> Result<Self> {
        Self::load_with(dir, model, Some(quant))
    }

    fn load_with(dir: impl AsRef<Path>, model: &Model, requant: Option<Quant>) -> Result<Self> {
        let dir = dir.as_ref();
        let text = std::fs::read(dir.join("config.json"))
            .map_err(|e| Error::Io(format!("{}: {e}", dir.join("config.json").display())))?;
        let raw = json::parse(&text)?;
        let config = DFlashConfig::from_json(&raw)?;
        let target = &model.config;
        let invalid = |m: String| Error::InvalidArgument(format!("DFlash drafter: {m}"));
        if config.hidden_size != target.hidden_size {
            return Err(invalid(format!(
                "hidden size {} does not match the target's {}",
                config.hidden_size, target.hidden_size
            )));
        }
        if config.vocab_size != target.vocab_size {
            return Err(invalid(format!(
                "vocabulary {} does not match the target's {}",
                config.vocab_size, target.vocab_size
            )));
        }
        if let Some(&i) = config
            .target_layer_ids
            .iter()
            .find(|&&i| i >= target.num_hidden_layers)
        {
            return Err(invalid(format!(
                "target layer {i} of a {}-layer target",
                target.num_hidden_layers
            )));
        }
        let quant = raw.get("quantization").map(|q| {
            let int = |k: &str, d: i64| q.get(k).and_then(Json::as_i64).unwrap_or(d) as i32;
            Quant::affine(int("group_size", 64), int("bits", 4))
        });
        // Each tensor's file; tensors are mapped one at a time as they are taken, so loading
        // never holds the whole full-precision checkpoint (quantized on load, it is dropped
        // tensor by tensor).
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        let mut index = BTreeMap::new();
        for f in &files {
            for t in nuvec::io::safetensors_tensors(f)? {
                index.insert(t.name, f.clone());
            }
        }
        let mut weights = Lazy { index };
        // DFlash 2's plain tensors: the convolutions' base kernels and the codebooks.
        let mut plain = |k: &str| {
            weights
                .remove(k)?
                .ok_or_else(|| invalid(format!("missing tensor `{k}`")))
        };
        let v2_tensors = match config.v2 {
            None => None,
            Some(_) => {
                let mut bases = Vec::new();
                for i in 0..config.layers {
                    bases.push((
                        plain(&format!("layers.{i}.attention_conv.base_kernel"))?,
                        plain(&format!("layers.{i}.mlp_conv.base_kernel"))?,
                    ));
                }
                let codebook = |a: Array| match a.dtype() {
                    DType::U32 => Err(invalid("quantized selector codebooks".into())),
                    _ => Ok(a),
                };
                let predecessor = codebook(plain("candidate_selector.predecessor_codebook")?)?;
                let successor = codebook(plain("candidate_selector.successor_codebook")?)?;
                Some((bases, predecessor, successor))
            }
        };
        // DSpark's Markov embedding (a plain table).
        let markov_w1 = match config.dspark {
            Some(d) if d.markov_rank > 0 => {
                let w1 = plain("markov_head.markov_w1.weight")?;
                if w1.dtype() == DType::U32 {
                    return Err(invalid("a quantized Markov embedding".into()));
                }
                Some(w1)
            }
            _ => None,
        };
        let mut take = |k: &str| {
            weights
                .remove(k)?
                .ok_or_else(|| invalid(format!("missing tensor `{k}`")))
        };
        let mut projection = |name: &str| -> Result<Projection> {
            let weight = take(&format!("{name}.weight"))?;
            let bias = take(&format!("{name}.bias")).ok().map(Param::new);
            Ok(match quant {
                Some(q) if weight.dtype() == DType::U32 => Projection::Quantized(QuantizedLinear {
                    weight,
                    scales: take(&format!("{name}.scales"))?,
                    biases: Some(take(&format!("{name}.biases"))?),
                    bias,
                    quant: q,
                }),
                _ => {
                    let linear = Linear {
                        weight: Param::new(weight),
                        bias,
                    };
                    match requant {
                        Some(q) => {
                            // Evaluated one at a time, so the transient float copies of
                            // quantization never coexist (peak memory stays near the
                            // quantized drafter's).
                            let q = QuantizedLinear::from_linear_with(&linear, q);
                            let mut parts = vec![q.weight.clone(), q.scales.clone()];
                            parts.extend(q.biases.clone());
                            nuvec::eval(&parts)?;
                            Projection::Quantized(q)
                        }
                        None => Projection::Dense(linear),
                    }
                }
            })
        };
        let fc = projection("fc")?;
        let mut layers = Vec::new();
        for i in 0..config.layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            let convs = match (&config.v2, &v2_tensors) {
                (Some(v), Some((bases, ..))) => {
                    let conv = |name: &str, base: &Array, projection: Projection| {
                        Conv {
                            base: base.clone(),
                            projection,
                            kernel: v.conv_kernel_size,
                            group: v.conv_group_size,
                        }
                        .checked(name, config.hidden_size)
                    };
                    let (attention, mlp) = &bases[i];
                    Some((
                        conv(
                            &p("attention_conv"),
                            attention,
                            projection(&p("attention_conv.kernel_projection"))?,
                        )?,
                        conv(
                            &p("mlp_conv"),
                            mlp,
                            projection(&p("mlp_conv.kernel_projection"))?,
                        )?,
                    ))
                }
                _ => None,
            };
            layers.push(Layer {
                convs,
                q_proj: projection(&p("self_attn.q_proj"))?,
                k_proj: projection(&p("self_attn.k_proj"))?,
                v_proj: projection(&p("self_attn.v_proj"))?,
                o_proj: projection(&p("self_attn.o_proj"))?,
                gate_proj: projection(&p("mlp.gate_proj"))?,
                up_proj: projection(&p("mlp.up_proj"))?,
                down_proj: projection(&p("mlp.down_proj"))?,
                q_norm: RmsNorm::new(1),
                k_norm: RmsNorm::new(1),
                input_layernorm: RmsNorm::new(1),
                post_attention_layernorm: RmsNorm::new(1),
            });
        }
        let selector = match (&config.v2, v2_tensors) {
            (Some(v), Some((_, predecessor, successor))) => Some(Selector {
                predecessor,
                successor,
                projection: projection("candidate_selector.hidden_projection")?,
                top_k: v.selector_top_k,
            }),
            _ => None,
        };
        // DSpark's heads; its checkpoints' copies of the target's embeddings and head are
        // never mapped (the target's are used).
        let (markov, confidence) = match (config.dspark, markov_w1) {
            (Some(d), Some(w1)) => {
                let w2 = projection("markov_head.markov_w2")?;
                let confidence = match d.confidence {
                    true => Some(projection("confidence_head.proj")?),
                    false => None,
                };
                (Some((w1, w2)), confidence)
            }
            _ => (None, None),
        };
        let mut norm = |name: &str| -> Result<RmsNorm> {
            Ok(RmsNorm {
                weight: Param::new(take(&format!("{name}.weight"))?),
                eps: config.eps,
            })
        };
        let hidden_norm = norm("hidden_norm")?;
        let final_norm = norm("norm")?;
        for (i, l) in layers.iter_mut().enumerate() {
            let p = |s: &str| format!("layers.{i}.{s}");
            l.q_norm = norm(&p("self_attn.q_norm"))?;
            l.k_norm = norm(&p("self_attn.k_norm"))?;
            l.input_layernorm = norm(&p("input_layernorm"))?;
            l.post_attention_layernorm = norm(&p("post_attention_layernorm"))?;
        }
        // Weights quantized on load are computed once here, not in every draft.
        let mut arrays: Vec<Array> = Vec::new();
        let mut add = |p: &Projection| match p {
            Projection::Quantized(q) => {
                arrays.extend([q.weight.clone(), q.scales.clone()]);
                arrays.extend(q.biases.clone());
            }
            Projection::Dense(l) => arrays.push(l.weight.value().clone()),
            Projection::Lora(_) => {}
        };
        add(&fc);
        if let Some(sel) = &selector {
            add(&sel.projection);
        }
        if let Some((_, w2)) = &markov {
            add(w2);
        }
        if let Some(c) = &confidence {
            add(c);
        }
        for l in &layers {
            if let Some((a, m)) = &l.convs {
                add(&a.projection);
                add(&m.projection);
            }
            for p in [
                &l.q_proj,
                &l.k_proj,
                &l.v_proj,
                &l.o_proj,
                &l.gate_proj,
                &l.up_proj,
                &l.down_proj,
            ] {
                add(p);
            }
        }
        nuvec::eval(&arrays)?;
        let cache = vec![None; config.layers];
        let rope = Rope::scaled(
            config.head_dim as i32,
            config.rope_theta,
            &config.rope_scaling,
        );
        Ok(Self {
            config,
            fc,
            hidden_norm,
            layers,
            norm: final_norm,
            rope,
            selector,
            markov,
            confidence,
            last_confidence: None,
            cache,
            positions: 0,
        })
    }

    /// The confidence logits of the last draft's positions (`[n]`; `sigmoid` gives the
    /// probability DSpark's confidence head gives each drafted token, given the ones before,
    /// of being accepted), with a confidence head.
    pub fn last_confidence(&self) -> Option<&Array> {
        self.last_confidence.as_ref()
    }

    /// DSpark's draft of `n` tokens after `anchor`: every slot of `[anchor, mask…]` predicts
    /// the next token, greedily, with the Markov head's bias from the previous token
    /// (DeepSpec's `sample_block_tokens` at temperature 0); the confidences of the drafted
    /// positions are kept.
    fn draft_dspark(&mut self, model: &Model, anchor: u32, n: usize) -> Result<Array> {
        let mut ids = vec![self.config.mask_token_id; n];
        ids[0] = anchor;
        let block = Array::from_slice(&ids, [1, n as i32]);
        let hidden = self.hidden(model, &block);
        let base = self.compute_logits(model, &hidden).reshape([n as i32, -1]);
        let Some((w1, w2)) = &self.markov else {
            return Ok(base.argmax(-1).astype(DType::U32));
        };
        let mut prev = Array::from_slice(&[anchor], [1]);
        let mut tokens = Vec::with_capacity(n);
        let mut prevs = Vec::with_capacity(n);
        for k in 0..n as i32 {
            let bias = w2.forward(&ops::take(w1, &prev, 0));
            let step = base.i(k..k + 1).add(bias.astype(base.dtype()));
            prevs.push(prev.clone());
            prev = step.argmax(-1).astype(DType::U32);
            tokens.push(prev.clone());
        }
        let tokens = ops::concatenate(&tokens, 0);
        self.last_confidence = match &self.confidence {
            Some(c) => {
                let prevs = ops::take(w1, &ops::concatenate(&prevs, 0), 0);
                let h = hidden.reshape([n as i32, -1]);
                let features = ops::concatenate(&[h.clone(), prevs.astype(h.dtype())], -1);
                Some(c.forward(&features).reshape([n as i32]).astype(DType::F32))
            }
            None => None,
        };
        Ok(tokens)
    }

    /// The drafter's settings.
    pub fn config(&self) -> &DFlashConfig {
        &self.config
    }

    /// Heads of `x` (`[1, L, n·head_dim]`) as `[1, n, L, head_dim]`, normalised by `norm`.
    fn heads(&self, x: &Array, n: usize, norm: Option<&RmsNorm>) -> Array {
        let l = x.shape()[1];
        let x = x.reshape([1, l, n as i32, self.config.head_dim as i32]);
        let x = match norm {
            Some(norm) => norm.forward(&x),
            None => x,
        };
        x.transpose([0, 2, 1, 3])
    }

    fn rope(&self, x: &Array, offset: i32) -> Array {
        self.rope.forward(x, offset)
    }

    /// The drafter's logits for `block` (`[1, B]` token ids) after the observed context.
    pub fn logits(&self, model: &Model, block: &Array) -> Array {
        self.compute_logits(model, &self.hidden(model, block))
    }

    /// The target's output head on `hidden` times `output_multiplier`, soft-capped (z-lab's
    /// `compute_logits`).
    fn compute_logits(&self, model: &Model, hidden: &Array) -> Array {
        let logits = model.raw_head(hidden);
        let m = self.config.output_multiplier;
        let logits = match m == 1.0 {
            true => logits,
            false => logits.multiply(f64::from(m)),
        };
        match self.config.final_logit_softcap {
            Some(cap) => nuvec_nn::logit_softcap(&logits, cap),
            None => logits,
        }
    }

    /// The drafter's final hidden states (`[1, B, H]`, normed) for `block`.
    fn hidden(&self, model: &Model, block: &Array) -> Array {
        let c = &self.config;
        let b = block.shape()[1];
        let x = model.draft_embed(block);
        let mut x = match c.input_embedding_scale == 1.0 {
            true => x,
            false => x.multiply(f64::from(c.input_embedding_scale)),
        };
        for (l, layer) in self.layers.iter().enumerate() {
            let n = layer.input_layernorm.forward(&x);
            let (n, kernel) = match &layer.convs {
                Some((conv, _)) => {
                    let (n, k) = conv.prepare(&n);
                    (n, Some(k))
                }
                None => (n, None),
            };
            let q = self.rope(
                &self.heads(&layer.q_proj.forward(&n), c.heads, Some(&layer.q_norm)),
                self.positions,
            );
            let k = self.rope(
                &self.heads(&layer.k_proj.forward(&n), c.kv_heads, Some(&layer.k_norm)),
                self.positions,
            );
            let v = self.heads(&layer.v_proj.forward(&n), c.kv_heads, None);
            let (keys, values, context) = match &self.cache[l] {
                Some((ck, cv)) => (
                    ops::concatenate(&[ck.clone(), k], 2),
                    ops::concatenate(&[cv.clone(), v], 2),
                    ck.shape()[2],
                ),
                None => (k, v, 0),
            };
            // z-lab's masks: none for full layers (the block bidirectional); sliding layers
            // see the context within the window (counted over the cached keys) and the block
            // causally.
            let mask = match (c.windows[l], c.causal[l]) {
                (None, false) => None,
                (window, causal) => {
                    let query = ops::arange(context, context + b, 1).reshape([b, 1]);
                    let key = ops::arange(0, context + b, 1).reshape([1, context + b]);
                    let in_context = key.less(context);
                    let ctx = match window {
                        Some(w) => ops::logical_and(&in_context, (&query - &key).less(w as i32)),
                        None => in_context.clone(),
                    };
                    let mut blk = ops::logical_not(&in_context);
                    if causal {
                        blk = ops::logical_and(&blk, key.less_equal(&query));
                    }
                    Some(ops::logical_or(&ctx, &blk))
                }
            };
            let options = match &mask {
                Some(m) => AttentionOptions::default().mask(m),
                None => AttentionOptions::default(),
            };
            let scale = (c.head_dim as f32).powf(-0.5);
            let o = nuvec::fast::scaled_dot_product_attention(&q, &keys, &values, scale, &options);
            let o = o.transpose([0, 2, 1, 3]).reshape([1, b, -1]);
            let o = layer.o_proj.forward(&o);
            let o = match (&layer.convs, &kernel) {
                (Some((conv, _)), Some(k)) => conv.finish(&o, k),
                _ => o,
            };
            x = x.add(o);
            let n = layer.post_attention_layernorm.forward(&x);
            let (n, kernel) = match &layer.convs {
                Some((_, conv)) => {
                    let (n, k) = conv.prepare(&n);
                    (n, Some(k))
                }
                None => (n, None),
            };
            let mlp = nuvec_nn::swiglu(&layer.gate_proj.forward(&n), &layer.up_proj.forward(&n));
            let mlp = layer.down_proj.forward(&mlp);
            let mlp = match (&layer.convs, &kernel) {
                (Some((_, conv)), Some(k)) => conv.finish(&mlp, k),
                _ => mlp,
            };
            x = x.add(mlp);
        }
        self.norm.forward(&x)
    }

    /// Adds the target's layer outputs at the next positions (`[1, n, ids · hidden]`) to the
    /// context.
    fn extend(&mut self, features: &Array) {
        let c = self.config.clone();
        let h = self.hidden_norm.forward(&self.fc.forward(features));
        for (l, layer) in self.layers.iter().enumerate() {
            let k = self.rope(
                &self.heads(&layer.k_proj.forward(&h), c.kv_heads, Some(&layer.k_norm)),
                self.positions,
            );
            let v = self.heads(&layer.v_proj.forward(&h), c.kv_heads, None);
            let (mut k, mut v) = match self.cache[l].take() {
                Some((ck, cv)) => (ops::concatenate(&[ck, k], 2), ops::concatenate(&[cv, v], 2)),
                None => (k, v),
            };
            if let Some(w) = c.windows[l] {
                let keep = (w as i32 - 1).max(0);
                let s = k.shape()[2];
                if s > keep {
                    let at = |a: &Array| a.i((.., .., s - keep..)).multiply(1.0);
                    (k, v) = (at(&k), at(&v));
                }
            }
            self.cache[l] = Some((k, v));
        }
        self.positions += features.shape()[1];
    }
}

impl Drafter for DFlashDrafter {
    fn draft(&mut self, model: &Model, context: &[u32], max: usize) -> Result<Draft> {
        if self.config.dspark.is_some() {
            let n = max.min(self.config.block_size);
            let Some(&anchor) = context.last().filter(|_| n > 0 && self.positions > 0) else {
                return Ok(Draft::empty());
            };
            return Ok(Draft::new(self.draft_dspark(model, anchor, n)?.check()?));
        }
        let n = max.min(self.config.block_size - 1);
        let Some(&anchor) = context.last().filter(|_| n > 0 && self.positions > 0) else {
            return Ok(Draft::empty());
        };
        // Only the positions a verification can take: the anchor and `n` (a block trained at
        // 16 drafts well at 8, z-lab's Table 8).
        let mut ids = vec![self.config.mask_token_id; n + 1];
        ids[0] = anchor;
        let block = Array::from_slice(&ids, [1, ids.len() as i32]);
        let hidden = self.hidden(model, &block).i((.., 1..1 + n as i32));
        let logits = self.compute_logits(model, &hidden);
        let tokens = match &self.selector {
            // DFlash 2: the selector's greedy path from the anchor.
            Some(selector) => selector.select(&hidden, &logits, anchor),
            None => logits.i(0).argmax(-1).astype(DType::U32),
        };
        Ok(Draft::new(tokens.check()?))
    }

    /// DFlash v1's per-position distributions (the positions are drafted independently);
    /// DFlash 2's selector scores pairs of positions, so it drafts chains.
    fn candidates(
        &mut self,
        model: &Model,
        context: &[u32],
        depth: usize,
        k: usize,
    ) -> Result<Option<Candidates>> {
        let n = depth.min(self.config.block_size - 1);
        let Some(&anchor) = context
            .last()
            .filter(|_| n > 0 && k > 0 && self.positions > 0)
        else {
            return Ok(None);
        };
        // DFlash 2's selector and DSpark's Markov head condition each position on the one
        // before: chains.
        if self.selector.is_some() || self.config.dspark.is_some() {
            return Ok(None);
        }
        let mut ids = vec![self.config.mask_token_id; n + 1];
        ids[0] = anchor;
        let block = Array::from_slice(&ids, [1, ids.len() as i32]);
        let hidden = self.hidden(model, &block).i((.., 1..1 + n as i32));
        let logits = self.compute_logits(model, &hidden).i(0).astype(DType::F32);
        let logprobs = logits.subtract(logits.logsumexp_keepdims(-1));
        let (logprobs, tokens) = crate::speculative::top_k(&logprobs, k);
        Ok(Some(Candidates::new(tokens.check()?, logprobs.check()?)))
    }

    fn observe(&mut self, _: &Model, _: &Array, hidden: Option<&Array>) -> Result<()> {
        let Some(features) = hidden else {
            return Err(Error::InvalidArgument(
                "DFlashDrafter: observed without the target's layer outputs".into(),
            ));
        };
        self.extend(features);
        for (k, v) in self.cache.iter().flatten() {
            let _ = (k.clone().check()?, v.clone().check()?);
        }
        Ok(())
    }

    fn needs_hidden(&self) -> bool {
        true
    }

    fn hidden_layers(&self) -> Option<Vec<usize>> {
        Some(self.config.target_layer_ids.clone())
    }

    /// Drafts of 7 (an 8-row verification, about one decode step's cost with MLX's
    /// few-row kernel): on Qwen3-8B 4-bit (M4 Pro) 2.1× plain decoding, against 1.6× with
    /// the full block of 15, whose extra accepted tokens do not pay for 16 rows.
    fn auto(&self, _: &Model) -> Auto {
        let drafts = match self.config.dspark {
            Some(_) => self.config.block_size,
            None => self.config.block_size - 1,
        };
        Auto {
            len: drafts.min(7),
            break_even: 1.0,
        }
    }
}
