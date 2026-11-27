//! Single-sequence DeepSeek-V3.2 attention, with the floating-point layer layout of
//! mlx-lm 0.32.2. The indexer ranks causal positions once per query, shared by every
//! attention head. The session stores normalized latents, rotary keys and index keys.
//! See THIRD_PARTY_NOTICES for the upstream attribution.
//!
//! Load converted layer parameters with [`SparseMla::new`], then call [`SparseMla::append`]
//! with hidden states shaped `[1, positions, hidden_size]`. Evaluate the output before
//! replacing the old session if execution errors must leave the previous prefix usable.

use std::collections::BTreeMap;
use std::sync::Arc;

use nuvec::{Array, DType, Error, Result, fast, ops, with_stream};
use nuvec_nn::{RopeScaling, Yarn};

/// DeepSeek-V3.2 attention dimensions and rotary parameters. Defaults are those of
/// the public checkpoint. Tiny layers may override them when exporting test weights.
#[derive(Clone, Debug)]
pub struct SparseMlaConfig {
    /// Width of input/output hidden states.
    pub hidden_size: i32,
    /// Number of main attention heads.
    pub num_attention_heads: i32,
    /// Query down-projection width.
    pub q_lora_rank: i32,
    /// Compressed KV width.
    pub kv_lora_rank: i32,
    /// Content-key width per head.
    pub qk_nope_head_dim: i32,
    /// Rotary channels per head.
    pub qk_rope_head_dim: i32,
    /// Per-head value projection width.
    pub v_head_dim: i32,
    /// Number of index query heads.
    pub index_n_heads: i32,
    /// Width of an index key.
    pub index_head_dim: i32,
    /// Maximum number of selected causal positions.
    pub index_topk: i32,
    /// Upper bound on sequence length.
    pub max_position_embeddings: i32,
    /// Rotary wavelength base.
    pub rope_theta: f32,
    /// Optional YaRN scaling parameters.
    pub yarn: Option<Yarn>,
    /// Use interleaved pairs in the indexer. The main attention always interleaves.
    pub indexer_rope_interleave: bool,
}

impl Default for SparseMlaConfig {
    fn default() -> Self {
        Self {
            hidden_size: 7168,
            num_attention_heads: 128,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            index_n_heads: 64,
            index_head_dim: 128,
            index_topk: 2048,
            max_position_embeddings: 163840,
            rope_theta: 10000.0,
            yarn: Some(Yarn {
                factor: 40.0,
                mscale_all_dim: 1.0,
                ..Yarn::default()
            }),
            indexer_rope_interleave: false,
        }
    }
}

fn invalid(msg: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("sparse MLA: {msg}"))
}

/// Immutable weights of one V3.2 layer.
pub struct SparseMla {
    c: SparseMlaConfig,
    weights: BTreeMap<String, Array>,
    id: Arc<()>,
    freqs: Option<Array>,
    mscale: f32,
    scale: f32,
}

/// One layer's persistent state. Copies share storage until an append or trim.
#[derive(Clone, Debug)]
pub struct SparseMlaSession {
    layer: Arc<()>,
    latent: Array,
    rotary: Array,
    index: Array,
}

impl SparseMlaSession {
    /// Number of positions already consumed.
    pub fn offset(&self) -> i32 {
        self.latent.shape()[0]
    }

    /// Logical bytes across the three histories. Excludes spare capacity and temporaries.
    pub fn nbytes(&self) -> usize {
        self.latent.nbytes() + self.rotary.nbytes() + self.index.nbytes()
    }

    /// Evaluate only the session histories (for example, when preparing a saved prefix).
    pub fn eval(&self) -> Result<()> {
        nuvec::eval(&[&self.latent, &self.rotary, &self.index])
    }

    /// Retain the prefix before the last `count` positions.
    pub fn trim(&self, count: i32) -> Result<Self> {
        if !(0..=self.offset()).contains(&count) {
            return Err(invalid("trim exceeds the retained prefix"));
        }
        let end = self.offset() - count;
        Ok(with_stream(self.latent.stream(), || Self {
            layer: self.layer.clone(),
            latent: self.latent.i(..end),
            rotary: self.rotary.i(..end),
            index: self.index.i(..end),
        }))
    }
}

impl SparseMla {
    /// Validate a layer's floating-point weights. Keys are relative to the attention
    /// module, for example `q_a_proj.weight` and `indexer.k_norm.bias`.
    pub fn new(c: SparseMlaConfig, mut weights: BTreeMap<String, Array>) -> Result<Self> {
        let h = c.num_attention_heads;
        let i = c.index_n_heads;
        let d = c.hidden_size;
        let r = c.qk_rope_head_dim;
        let n = c.qk_nope_head_dim;
        let l = c.kv_lora_rank;
        let q = c.q_lora_rank;
        let v = c.v_head_dim;
        let k = c.index_head_dim;
        if [
            h,
            i,
            d,
            r,
            n,
            l,
            q,
            v,
            k,
            c.index_topk,
            c.max_position_embeddings,
        ]
        .iter()
        .any(|&x| x <= 0)
            || r % 2 != 0
            || r > k
            || !c.rope_theta.is_finite()
            || c.rope_theta <= 1.0
        {
            return Err(invalid(
                "invalid layer dimensions, position limit or rope_theta",
            ));
        }
        if let Some(y) = c.yarn
            && (![
                y.factor,
                y.original_max_position_embeddings,
                y.beta_fast,
                y.beta_slow,
            ]
            .iter()
            .all(|p| p.is_finite() && *p > 0.0)
                || !y.mscale.is_finite()
                || !y.mscale_all_dim.is_finite())
        {
            return Err(invalid("invalid YaRN scaling parameters"));
        }
        let overflow = || invalid("weight dimensions exceed the supported size");
        let qd = n.checked_add(r).ok_or_else(overflow)?;
        let shapes = [
            ("q_a_proj.weight", vec![q, d]),
            ("q_a_layernorm.weight", vec![q]),
            (
                "q_b_proj.weight",
                vec![h.checked_mul(qd).ok_or_else(overflow)?, q],
            ),
            (
                "kv_a_proj_with_mqa.weight",
                vec![l.checked_add(r).ok_or_else(overflow)?, d],
            ),
            ("kv_a_layernorm.weight", vec![l]),
            ("embed_q.weight", vec![h, l, n]),
            ("unembed_out.weight", vec![h, v, l]),
            (
                "o_proj.weight",
                vec![d, h.checked_mul(v).ok_or_else(overflow)?],
            ),
            (
                "indexer.wq_b.weight",
                vec![i.checked_mul(k).ok_or_else(overflow)?, q],
            ),
            ("indexer.wk.weight", vec![k, d]),
            ("indexer.k_norm.weight", vec![k]),
            ("indexer.k_norm.bias", vec![k]),
            ("indexer.weights_proj.weight", vec![i, d]),
        ];
        let mut bound = BTreeMap::new();
        let mut placement = None;
        for (name, shape) in shapes {
            let a = weights
                .remove(name)
                .ok_or_else(|| invalid(format!("missing {name}")))?
                .check()?;
            let p = (a.dtype(), a.device());
            if a.shape() != shape
                || !matches!(a.dtype(), DType::F32 | DType::BF16)
                || placement.is_some_and(|previous| previous != p)
            {
                return Err(invalid(format!(
                    "{name}: expected shape {shape:?}, uniform float32 or bfloat16 and device"
                )));
            }
            placement = Some(p);
            bound.insert(name.to_owned(), a);
        }
        let scaling = c.yarn.map_or(RopeScaling::None, RopeScaling::Yarn);
        let freqs = with_stream(bound["q_a_proj.weight"].stream(), || {
            scaling.frequencies(r, c.rope_theta)
        });
        let mscale = scaling.mscale();
        let all_scale = c
            .yarn
            .filter(|y| y.factor > 1.0)
            .map_or(1.0, |y| 0.1 * y.mscale_all_dim * y.factor.ln() + 1.0);
        let scale = all_scale.powi(2) / (qd as f32).sqrt();
        if !mscale.is_finite() || !scale.is_finite() || mscale <= 0.0 || scale <= 0.0 {
            return Err(invalid("nonpositive or nonfinite rotary scale"));
        }
        Ok(Self {
            c,
            weights: bound,
            id: Arc::new(()),
            freqs,
            mscale,
            scale,
        })
    }

    /// Create a session on the weights' device and with their dtype.
    pub fn new_session(&self) -> SparseMlaSession {
        let weight = &self.weights["q_a_proj.weight"];
        with_stream(weight.stream(), || {
            let empty = |width| ops::zeros([0, width], weight.dtype());
            SparseMlaSession {
                layer: self.id.clone(),
                latent: empty(self.c.kv_lora_rank),
                rotary: empty(self.c.qk_rope_head_dim),
                index: empty(self.c.index_head_dim),
            }
        })
    }

    fn project(&self, x: &Array, name: &str) -> Array {
        x.matmul(&self.weights[name].swapaxes(-1, -2))
    }

    fn rotate(&self, x: &Array, offset: i32, interleaved: bool) -> Array {
        let r = self.c.qk_rope_head_dim;
        let x = if self.mscale == 1.0 {
            x.clone()
        } else {
            let rotated = x.i((nuvec::Ellipsis, ..r)).multiply(self.mscale);
            if x.shape().last() == Some(&r) {
                rotated
            } else {
                ops::concatenate(&[rotated, x.i((nuvec::Ellipsis, r..))], -1)
            }
        };
        fast::rope(
            &x,
            r,
            interleaved,
            self.freqs.is_none().then_some(self.c.rope_theta),
            1.0,
            &Array::from(offset),
            self.freqs.as_ref(),
        )
    }

    /// Append hidden states, returning the layer output and an independent candidate session.
    pub fn append(&self, x: &Array, cache: &SparseMlaSession) -> Result<(Array, SparseMlaSession)> {
        let (y, state, _) = self.run(x, cache, false)?;
        Ok((y, state))
    }

    /// Append and expose the selected absolute positions per query for diagnostics.
    /// The `[Q, min(k, end)]` int32 array pads the earliest rows with -1.
    pub fn append_with_selection(
        &self,
        x: &Array,
        cache: &SparseMlaSession,
    ) -> Result<(Array, SparseMlaSession, Array)> {
        let (y, state, picks) = self.run(x, cache, true)?;
        Ok((y, state, picks.ok_or_else(|| invalid("missing selection"))?))
    }

    fn run(
        &self,
        x: &Array,
        cache: &SparseMlaSession,
        diagnostics: bool,
    ) -> Result<(Array, SparseMlaSession, Option<Array>)> {
        let _ = x.clone().check()?;
        let c = &self.c;
        let [1, rows, width] = x.shape_n::<3>().unwrap_or_default() else {
            return Err(invalid(
                "hidden states must have shape [1, positions, hidden_size]",
            ));
        };
        if width != c.hidden_size
            || x.dtype() != cache.latent.dtype()
            || x.device() != cache.latent.device()
            || !Arc::ptr_eq(&self.id, &cache.layer)
        {
            return Err(invalid(
                "hidden width, dtype, device or session layer mismatch",
            ));
        }
        let start = cache.offset();
        let end = start
            .checked_add(rows)
            .filter(|end| *end <= c.max_position_embeddings)
            .ok_or_else(|| invalid("append exceeds the configured position limit"))?;
        with_stream(x.stream(), || {
            if rows == 0 {
                return Ok((
                    x.clone(),
                    cache.clone(),
                    diagnostics.then(|| ops::zeros([0, c.index_topk.min(end)], DType::I32)),
                ));
            }
            let x = x.squeeze(0);
            let qr = fast::rms_norm(
                &self.project(&x, "q_a_proj.weight"),
                &self.weights["q_a_layernorm.weight"],
                1e-6,
            );
            let kv = self.project(&x, "kv_a_proj_with_mqa.weight");
            let latent = fast::rms_norm(
                &kv.i((.., ..c.kv_lora_rank)),
                &self.weights["kv_a_layernorm.weight"],
                1e-6,
            );
            let key = self
                .rotate(&kv.i((.., c.kv_lora_rank..)).expand_dims(0), start, true)
                .squeeze(0);
            let idx = fast::layer_norm(
                &self.project(&x, "indexer.wk.weight"),
                &self.weights["indexer.k_norm.weight"],
                &self.weights["indexer.k_norm.bias"],
                1e-5,
            );
            let idx = self
                .rotate(&idx.expand_dims(0), start, c.indexer_rope_interleave)
                .squeeze(0);
            let join = |old: &Array, new: Array| {
                if start == 0 {
                    new
                } else {
                    ops::concatenate(&[old.clone(), new], 0)
                }
            };
            let state = SparseMlaSession {
                layer: self.id.clone(),
                latent: join(&cache.latent, latent),
                rotary: join(&cache.rotary, key),
                index: join(&cache.index, idx),
            };
            let mut outputs = Vec::new();
            let mut selections = Vec::new();
            // Bound the temporary [index_heads, query_block, keys] score tensor.
            // Each block sees the prefix ending at its own last query.
            for lo in (0..rows).step_by(64) {
                let hi = lo.saturating_add(64).min(rows);
                let prefix = state.trim(rows - hi)?;
                let (output, picks) = self.attend_block(
                    &x.i(lo..hi),
                    &qr.i(lo..hi),
                    &prefix,
                    start + lo,
                    diagnostics,
                );
                outputs.push(output);
                if let Some(picks) = picks {
                    let width = c.index_topk.min(end) - picks.shape()[1];
                    selections.push(if width == 0 {
                        picks
                    } else {
                        ops::concatenate(&[picks, ops::full([hi - lo, width], -1i32)], 1)
                    });
                }
            }
            let join_blocks = |blocks: Vec<Array>, axis| match blocks.as_slice() {
                [one] => one.clone(),
                _ => ops::concatenate(&blocks, axis),
            };
            let out = join_blocks(outputs, 1);
            let shown = diagnostics.then(|| join_blocks(selections, 0));
            Ok((
                ops::depends(&out, std::slice::from_ref(&state.index)).check()?,
                state,
                shown,
            ))
        })
    }
    // Inputs are flattened over the single batch; history ends at this block.
    fn attend_block(
        &self,
        x: &Array,
        qr: &Array,
        state: &SparseMlaSession,
        start: i32,
        diagnostics: bool,
    ) -> (Array, Option<Array>) {
        let c = &self.c;
        let rows = x.shape()[0];
        let end = state.offset();
        let q = self
            .project(qr, "q_b_proj.weight")
            .reshape([rows, c.num_attention_heads, -1])
            .swapaxes(0, 1);
        let qc = self.project(&q.i((.., .., ..c.qk_nope_head_dim)), "embed_q.weight");
        let qp = self.rotate(&q.i((.., .., c.qk_nope_head_dim..)), start, true);
        let positions = ops::arange(start, end, 1).reshape([rows, 1]);
        let axis = ops::arange(0, end, 1);
        let causal = positions.greater_equal(&axis);
        let (scores, values, shown) = if end <= c.index_topk {
            let scores = qc
                .matmul(&state.latent.t())
                .add(qp.matmul(&state.rotary.t()))
                .swapaxes(0, 1);
            (
                scores,
                state.latent.clone(),
                diagnostics.then(|| causal.select(&axis, -1)),
            )
        } else {
            let qi = self
                .project(qr, "indexer.wq_b.weight")
                .reshape([rows, c.index_n_heads, c.index_head_dim])
                .swapaxes(0, 1);
            let qi = self.rotate(&qi, start, c.indexer_rope_interleave);
            let w = self
                .project(x, "indexer.weights_proj.weight")
                .t()
                .expand_dims(-1)
                .multiply(1.0 / ((c.index_n_heads as f32) * (c.index_head_dim as f32)).sqrt());
            let rank = qi.matmul(&state.index.t()).maximum(0).multiply(&w).sum(0);
            let rank = causal.select(rank, f32::NEG_INFINITY);
            let picks = rank
                .argpartition(-c.index_topk, -1)
                .i((.., -c.index_topk..))
                .stop_gradient();
            let latent = state.latent.take(&picks, 0);
            let positional = state.rotary.take(&picks, 0);
            let scores = qc
                .swapaxes(0, 1)
                .matmul(&latent.swapaxes(-1, -2))
                .add(qp.swapaxes(0, 1).matmul(&positional.swapaxes(-1, -2)));
            let visible = positions.greater_equal(picks.astype(DType::I32));
            let masked = visible.expand_dims(1).select(scores, f32::NEG_INFINITY);
            (
                masked,
                latent,
                diagnostics.then(|| visible.select(picks.astype(DType::I32), -1)),
            )
        };
        let scores = if end <= c.index_topk {
            causal.expand_dims(1).select(scores, f32::NEG_INFINITY)
        } else {
            scores
        };
        let attended = scores
            .multiply(self.scale)
            .softmax_precise(-1)
            .matmul(&values);
        let v = self
            .project(&attended.swapaxes(0, 1), "unembed_out.weight")
            .swapaxes(0, 1)
            .reshape([1, rows, -1]);
        (self.project(&v, "o_proj.weight"), shown)
    }
}
