//! Tensor parallelism, as mlx-lm's `model.shard(group)`: every attention, MLP, expert and
//! Gated DeltaNet projection keeps this rank's slice, and each block's two outputs are summed
//! over the group.
//!
//! - **All-to-sharded** projections (q, k, v, gate, up, the Gated DeltaNet's inputs) keep the
//!   rank's output rows, segment by segment for fused projections.
//! - **Sharded-to-all** ones (o, down, out) keep its input columns. Their bias is kept on
//!   rank 0 only, so the group's sum adds it once, exactly. mlx-lm divides an expert bias
//!   by the group size instead, which rounds.
//! - **The output head**, when untied and divisible, keeps the rank's vocabulary rows; its
//!   logits are gathered.
//!
//! Every slice is copied, so the full weight's memory is released.

// `take` takes the segments to keep; one segment is a list of one range.
#![allow(clippy::single_range_in_vec_init)]

use std::ops::Range;

use nuvec::distributed::Group;
use nuvec::{Array, Error, Result};
use nuvec_nn::{Linear, Param, QuantizedLinear};

use super::{
    Attention, Block, BlockMlp, GatedDeltaNet, GptOssMlp, Mixer, Mlp, Model, MoeMlp, Projection,
    SwitchGlu, SwitchProjection,
};

fn invalid(m: String) -> Error {
    Error::InvalidArgument(format!("shard: {m}"))
}

/// Rank `r`'s part `[r·len/n, (r+1)·len/n)` of each segment, in segment order.
fn parts(segments: &[i32], n: i32, r: i32) -> Vec<Range<i32>> {
    let mut at = 0;
    segments
        .iter()
        .map(|&len| {
            let range = at + r * len / n..at + (r + 1) * len / n;
            at += len;
            range
        })
        .collect()
}

/// `a`'s rows `ranges` along `axis` (concatenated), copied.
fn take(a: &Array, axis: usize, ranges: &[Range<i32>]) -> Array {
    let pieces: Vec<Array> = ranges
        .iter()
        .map(|r| match axis {
            0 => a.i(r.clone()),
            1 => a.i((.., r.clone())),
            _ => a.i((.., .., r.clone())),
        })
        .collect();
    let joined = if pieces.len() == 1 {
        pieces[0].clone()
    } else {
        nuvec::ops::concatenate(&pieces, axis as i32)
    };
    nuvec::ops::copy(&joined)
}

fn check(what: &str, len: i32, n: i32, unit: i32) -> Result<()> {
    if len % (n * unit) != 0 {
        return Err(invalid(format!(
            "{what}: {len} do not split across {n} ranks{}",
            if unit > 1 {
                format!(" in whole groups of {unit}")
            } else {
                String::new()
            }
        )));
    }
    Ok(())
}

impl Projection {
    /// The rank's output rows: segment `i` of `segments` (rows) split `n` ways.
    fn rows(&self, what: &str, segments: &[i32], n: i32, r: i32) -> Result<Self> {
        for &s in segments {
            check(what, s, n, 1)?;
        }
        let ranges = parts(segments, n, r);
        Ok(match self {
            Self::Dense(l) => Self::Dense(Linear {
                weight: Param::new(take(l.weight.value(), 0, &ranges)),
                bias: l
                    .bias
                    .as_ref()
                    .map(|b| Param::new(take(b.value(), 0, &ranges))),
            }),
            Self::Quantized(q) => Self::Quantized(QuantizedLinear {
                weight: take(&q.weight, 0, &ranges),
                scales: take(&q.scales, 0, &ranges),
                biases: q.biases.as_ref().map(|b| take(b, 0, &ranges)),
                bias: q
                    .bias
                    .as_ref()
                    .map(|b| Param::new(take(b.value(), 0, &ranges))),
                quant: q.quant,
            }),
            Self::Lora(_) => return Err(invalid(format!("{what}: fuse LoRA adapters first"))),
        })
    }

    /// The rank's input columns, in whole quantization groups; the bias on rank 0 only.
    fn cols(&self, what: &str, n: i32, r: i32) -> Result<Self> {
        let bias = |b: &Option<Param>| {
            b.as_ref().map(|b| {
                let v = b.value();
                Param::new(if r == 0 {
                    v.clone()
                } else {
                    nuvec::ops::zeros_like(v)
                })
            })
        };
        Ok(match self {
            Self::Dense(l) => {
                let input = l.weight.value().shape()[1];
                check(what, input, n, 1)?;
                let cols = parts(&[input], n, r);
                Self::Dense(Linear {
                    weight: Param::new(take(l.weight.value(), 1, &cols)),
                    bias: bias(&l.bias),
                })
            }
            Self::Quantized(q) => {
                let (bits, gs) = (q.quant.bits.max(1), q.quant.group_size.max(1));
                let input = q.weight.shape()[1] * 32 / bits;
                check(what, input, n, gs)?;
                let (a, b) = (r * input / n, (r + 1) * input / n);
                Self::Quantized(QuantizedLinear {
                    weight: take(&q.weight, 1, &[a * bits / 32..b * bits / 32]),
                    scales: take(&q.scales, 1, &[a / gs..b / gs]),
                    biases: q.biases.as_ref().map(|x| take(x, 1, &[a / gs..b / gs])),
                    bias: bias(&q.bias),
                    quant: q.quant,
                })
            }
            Self::Lora(_) => return Err(invalid(format!("{what}: fuse LoRA adapters first"))),
        })
    }
}

impl SwitchProjection {
    /// Every expert's output rows (axis 1 of `[E, out, in]`).
    fn rows(&self, what: &str, n: i32, r: i32) -> Result<Self> {
        Ok(match self {
            Self::Dense { weight, bias } => {
                let out = weight.value().shape()[1];
                check(what, out, n, 1)?;
                let rg = parts(&[out], n, r);
                Self::Dense {
                    weight: Param::new(take(weight.value(), 1, &rg)),
                    bias: bias.as_ref().map(|b| Param::new(take(b.value(), 1, &rg))),
                }
            }
            Self::Quantized {
                bias,
                weight,
                scales,
                biases,
                global_scale,
                quant,
            } => {
                let out = weight.shape()[1];
                check(what, out, n, 1)?;
                let rg = parts(&[out], n, r);
                Self::Quantized {
                    bias: bias.as_ref().map(|b| Param::new(take(b.value(), 1, &rg))),
                    weight: take(weight, 1, &rg),
                    scales: take(scales, 1, &rg),
                    biases: biases.as_ref().map(|x| take(x, 1, &rg)),
                    global_scale: global_scale.clone(),
                    quant: *quant,
                }
            }
        })
    }

    /// Every expert's input columns (the last axis), in whole quantization groups; the
    /// bias on rank 0 only.
    fn cols(&self, what: &str, n: i32, r: i32) -> Result<Self> {
        let keep = |b: &Option<Param>| {
            b.as_ref().map(|b| {
                let v = b.value();
                Param::new(if r == 0 {
                    v.clone()
                } else {
                    nuvec::ops::zeros_like(v)
                })
            })
        };
        Ok(match self {
            Self::Dense { weight, bias } => {
                let input = weight.value().shape()[2];
                check(what, input, n, 1)?;
                Self::Dense {
                    weight: Param::new(take(weight.value(), 2, &parts(&[input], n, r))),
                    bias: keep(bias),
                }
            }
            Self::Quantized {
                bias,
                weight,
                scales,
                biases,
                global_scale,
                quant,
            } => {
                let (bits, gs) = (quant.bits.max(1), quant.group_size.max(1));
                let input = weight.shape()[2] * 32 / bits;
                check(what, input, n, gs)?;
                let (a, b) = (r * input / n, (r + 1) * input / n);
                Self::Quantized {
                    bias: keep(bias),
                    weight: take(weight, 2, &[a * bits / 32..b * bits / 32]),
                    scales: take(scales, 2, &[a / gs..b / gs]),
                    biases: biases.as_ref().map(|x| take(x, 2, &[a / gs..b / gs])),
                    global_scale: global_scale.clone(),
                    quant: *quant,
                }
            }
        })
    }
}

impl SwitchGlu {
    fn shard(&self, path: &str, n: i32, r: i32) -> Result<Self> {
        Ok(Self {
            gate_proj: self.gate_proj.rows(&format!("{path}.gate_proj"), n, r)?,
            up_proj: self.up_proj.rows(&format!("{path}.up_proj"), n, r)?,
            down_proj: self.down_proj.cols(&format!("{path}.down_proj"), n, r)?,
            activation: self.activation,
        })
    }
}

impl Mlp {
    fn shard(&self, path: &str, n: i32, r: i32) -> Result<Self> {
        let hidden = out_rows(&self.gate_proj);
        Ok(Self {
            gate_proj: self
                .gate_proj
                .rows(&format!("{path}.gate_proj"), &[hidden], n, r)?,
            up_proj: self
                .up_proj
                .rows(&format!("{path}.up_proj"), &[hidden], n, r)?,
            down_proj: self.down_proj.cols(&format!("{path}.down_proj"), n, r)?,
            geglu: self.geglu,
        })
    }
}

/// The rows of a projection's output.
fn out_rows(p: &Projection) -> i32 {
    match p {
        Projection::Dense(l) => l.weight.value().shape()[0],
        Projection::Quantized(q) => q.weight.shape()[0],
        Projection::Lora(_) => 0,
    }
}

impl Attention {
    fn shard(&self, path: &str, n: i32, r: i32) -> Result<Self> {
        let at = |s: &str| format!("{path}.{s}");
        if self.heads % n != 0 {
            return Err(invalid(format!(
                "{path}: {} heads do not split across {n} ranks",
                self.heads
            )));
        }
        // Fewer key/value heads than ranks: each rank keeps one, the one its query heads
        // read (mlx-lm repeats them so that every rank has one).
        let (kv_rows, kv_heads) = if self.kv_heads >= n {
            if self.kv_heads % n != 0 {
                return Err(invalid(format!(
                    "{path}: {} key/value heads do not split across {n} ranks",
                    self.kv_heads
                )));
            }
            (None, self.kv_heads / n)
        } else {
            if n % self.kv_heads != 0 {
                return Err(invalid(format!(
                    "{path}: {} key/value heads do not repeat across {n} ranks",
                    self.kv_heads
                )));
            }
            (Some(r / (n / self.kv_heads)), 1)
        };
        let kv = |p: &Projection, name: &str| -> Result<Projection> {
            let rows = out_rows(p);
            match kv_rows {
                None => p.rows(&at(name), &[rows], n, r),
                // Rows of key/value head `h`: the projection split into its heads.
                Some(h) => p.rows(&at(name), &[rows], self.kv_heads, h),
            }
        };
        Ok(Self {
            q_proj: self
                .q_proj
                .rows(&at("q_proj"), &[out_rows(&self.q_proj)], n, r)?,
            k_proj: self.k_proj.as_ref().map(|p| kv(p, "k_proj")).transpose()?,
            v_proj: self.v_proj.as_ref().map(|p| kv(p, "v_proj")).transpose()?,
            o_proj: self.o_proj.cols(&at("o_proj"), n, r)?,
            q_norm: self.q_norm.clone(),
            k_norm: self.k_norm.clone(),
            heads: self.heads / n,
            kv_heads,
            scale: self.scale,
            rope: self.rope.clone(),
            gate: self.gate,
            sinks: self.sinks.as_ref().map(|s| {
                let per = self.heads / n;
                Param::new(take(s.value(), 0, &[r * per..(r + 1) * per]))
            }),
            window: self.window,
            mscale: self.mscale,
            v_norm: self.v_norm,
        })
    }
}

impl GatedDeltaNet {
    fn shard(&self, path: &str, n: i32, r: i32) -> Result<Self> {
        let at = |s: &str| format!("{path}.{s}");
        let s = self.sizes;
        for (what, heads) in [("key heads", s.hk), ("value heads", s.hv)] {
            if heads % n != 0 {
                return Err(invalid(format!(
                    "{path}: {heads} {what} do not split across {n} ranks"
                )));
            }
        }
        let (kd, vd) = (s.hk * s.dk, s.hv * s.dv);
        let qkv = [kd, kd, vd];
        let mut sizes = s;
        sizes.hk /= n;
        sizes.hv /= n;
        let heads = parts(&[s.hv], n, r);
        Ok(Self {
            in_proj_qkv: self.in_proj_qkv.rows(&at("in_proj_qkv"), &qkv, n, r)?,
            in_proj_z: self.in_proj_z.rows(&at("in_proj_z"), &[vd], n, r)?,
            in_proj_b: self.in_proj_b.rows(&at("in_proj_b"), &[s.hv], n, r)?,
            in_proj_a: self.in_proj_a.rows(&at("in_proj_a"), &[s.hv], n, r)?,
            conv1d: Param::new(take(self.conv1d.value(), 0, &parts(&qkv, n, r))),
            dt_bias: Param::new(take(self.dt_bias.value(), 0, &heads)),
            a_log: Param::new(take(self.a_log.value(), 0, &heads)),
            norm: self.norm.clone(),
            out_proj: self.out_proj.cols(&at("out_proj"), n, r)?,
            sizes,
        })
    }
}

impl Block {
    fn shard(&self, path: &str, group: &Group) -> Result<Self> {
        let (n, r) = (group.size() as i32, group.rank() as i32);
        let mixer = match &self.mixer {
            Mixer::Attention(a) => Mixer::Attention(a.shard(&format!("{path}.self_attn"), n, r)?),
            Mixer::Linear(l) => Mixer::Linear(l.shard(&format!("{path}.linear_attn"), n, r)?),
        };
        let p = format!("{path}.mlp");
        let mlp = match &self.mlp {
            BlockMlp::Dense(m) => BlockMlp::Dense(Box::new(m.shard(&p, n, r)?)),
            BlockMlp::Moe(m) => BlockMlp::Moe(Box::new(MoeMlp {
                gate: m.gate.clone(),
                switch_mlp: m.switch_mlp.shard(&format!("{p}.switch_mlp"), n, r)?,
                shared_expert: m
                    .shared_expert
                    .as_ref()
                    .map(|e| e.shard(&format!("{p}.shared_expert"), n, r))
                    .transpose()?,
                shared_expert_gate: m.shared_expert_gate.clone(),
                experts_per_token: m.experts_per_token,
                norm_topk_prob: m.norm_topk_prob,
            })),
            BlockMlp::GptOss(m) => BlockMlp::GptOss(Box::new(GptOssMlp {
                router: m.router.clone(),
                experts: m.experts.shard(&format!("{p}.experts"), n, r)?,
                experts_per_token: m.experts_per_token,
            })),
        };
        Ok(Self {
            mixer,
            mlp,
            input_layernorm: self.input_layernorm.clone(),
            post_attention_layernorm: self.post_attention_layernorm.clone(),
            gemma: None,
            group: Some(group.clone()),
        })
    }
}

impl Model {
    /// Splits the model across `group` (mlx-lm's `model.shard(group)`): every rank keeps its
    /// slice of each attention, MLP, expert and Gated DeltaNet projection (head counts
    /// divided; fewer key/value heads than ranks are shared by the ranks their query heads
    /// are on), and each block's attention and MLP outputs are summed over the group. An
    /// untied output head that divides keeps the rank's vocabulary rows, and its logits are
    /// gathered. Every rank then runs the same calls; only rank 0 samples, and every rank
    /// receives its tokens ([`Model::share_tokens`]).
    ///
    /// Atomic: a size that does not divide is an error naming the layer and the sizes, and
    /// the model is left as it was. A group of size 1 leaves it whole.
    pub fn shard(&mut self, group: &Group) -> Result<()> {
        if group.size() == 1 {
            return Ok(());
        }
        if self.config.gemma4.is_some() {
            return Err(invalid(
                "Gemma 4 is not supported (as mlx-lm, which does not shard it)".into(),
            ));
        }
        if self.stage.is_some() {
            return Err(invalid("a pipelined model cannot also be sharded".into()));
        }
        if self.sharding.is_some() {
            return Err(invalid("the model is already sharded".into()));
        }
        let layers = self
            .model
            .layers
            .iter()
            .enumerate()
            .map(|(i, b)| b.shard(&format!("model.layers.{i}"), group))
            .collect::<Result<Vec<_>>>()?;
        let (n, r) = (group.size() as i32, group.rank() as i32);
        let vocab = self.config.vocab_size as i32;
        let head = match &self.lm_head {
            Some(h) if vocab % n == 0 => Some((h.rows("lm_head", &[vocab], n, r)?, true)),
            Some(h) => Some((h.clone(), false)),
            None => None,
        };
        self.model.layers = layers;
        if let Some((h, split)) = head {
            self.lm_head = Some(h);
            self.vocab_sharded = split;
        }
        self.sharding = Some(group.clone());
        Ok(())
    }

    /// The group a sharded model is split across ([`Model::shard`]).
    pub fn sharding(&self) -> Option<&Group> {
        self.sharding.as_ref()
    }
}
