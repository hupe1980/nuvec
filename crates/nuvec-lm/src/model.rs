//! The Llama-family transformer, as mlx-lm's `llama`, `qwen2` and `qwen3` models, and its
//! hybrid with Gated DeltaNet layers, as mlx-lm's `qwen3_5`, with parameter names matching
//! Hugging Face checkpoints.
//!
//! The forward passes return arrays (a failed array carries the first error, see
//! [`Array::check`]); construction and the paged cache's bookkeeping return `Result`.

use std::sync::LazyLock;

use nuvec::fast::{self, AttentionMask};
use nuvec::{Array, DType, GatherQmm, OpErrorKind, Param, Quant, QuantMode, Result, ops};
use nuvec_nn::{
    Embedding, Linear, LoraLinear, Module, QuantizedEmbedding, QuantizedLinear, RmsNorm, silu,
};

use crate::config::{
    Activation, Config, LayerQuant, LayerType, LinearAttention, MixtureOfExperts, RopeScaling,
};

mod gemma;
mod mtp;
mod shard;

/// A stand-in of `shape` and `dtype` for a weight not loaded yet (no memory): a failed
/// array, so a forward pass or `save_weights` before the weights load reports it.
fn placeholder(shape: &[usize], dtype: DType) -> Array {
    Array::failure(
        "load_weights",
        OpErrorKind::InvalidArgument("this weight has not been loaded".into()),
    )
    .astype(dtype)
    .broadcast_to(shape)
}

/// A placeholder parameter.
fn param(shape: &[usize], dtype: DType) -> Param {
    Param::new(placeholder(shape, dtype))
}

/// A size of the configuration as an array dimension.
fn int(what: &str, v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| nuvec::Error::InvalidArgument(format!("{what}: {v} is too large")))
}

/// A compiled function of two arrays (traced once per input signature, element-wise chains
/// fused), as mlx-lm compiles its activations. On failed inputs it gives failed outputs.
type Compiled2 = Box<dyn Fn(&(Array, Array)) -> Result<Array> + Send + Sync>;
/// [`Compiled2`] of three arrays.
type Compiled3 = Box<dyn Fn(&(Array, Array, Array)) -> Result<Array> + Send + Sync>;

use nuvec_nn::swiglu;

/// A linear layer, full precision or quantized. Variants are transparent in parameter
/// names (`q_proj.weight`).
#[derive(Debug, Clone, Module)]
#[non_exhaustive]
pub enum Projection {
    /// Full precision.
    Dense(Linear),
    /// MLX's affine quantization.
    Quantized(QuantizedLinear),
    /// A LoRA adapter over a full-precision or quantized base (mlx-lm's names:
    /// `linear.weight`, `lora_a`, `lora_b`); see [`crate::lora`].
    Lora(Box<LoraLinear>),
}

impl Projection {
    /// The output features.
    pub(crate) fn outputs(&self) -> usize {
        let rows = match self {
            Self::Dense(l) => l.weight.value().shape()[0],
            Self::Quantized(q) => q.weight.shape()[0],
            Self::Lora(l) => l.lora_b.value().shape()[1],
        };
        rows as usize
    }

    /// `x Wᵀ (+ b)`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
            Self::Lora(l) => l.forward(x),
        }
    }
}

/// A token embedding, full precision or quantized.
#[derive(Debug, Clone, Module)]
pub(crate) enum TokenEmbedding {
    /// Full precision.
    Dense(Embedding),
    /// MLX's affine quantization.
    Quantized(QuantizedEmbedding),
}

impl TokenEmbedding {
    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        match self {
            Self::Dense(e) => e.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }

    #[track_caller]
    fn as_linear(&self, x: &Array) -> Array {
        match self {
            Self::Dense(e) => e.as_linear(x),
            Self::Quantized(q) => q.as_linear(x),
        }
    }
}

/// The groups of `features` features in quantization groups of `group_size` (an error
/// for an empty group).
fn groups(path: &str, features: usize, group_size: usize) -> Result<usize> {
    features
        .checked_div(group_size)
        .ok_or_else(|| nuvec::Error::InvalidArgument(format!("{path}: quantization group size 0")))
}

/// Builds layers with placeholder parameters; `quantized(path)` gives the quantization of
/// quantized layers.
struct Builder<'a> {
    quantized: &'a dyn Fn(&str) -> Option<LayerQuant>,
}

/// Placeholders of a quantized layer with `rows` outputs over `input` values: the codes,
/// the scales (affine: float; floating-point modes: uint8) and, for affine, the biases.
fn quantized_placeholders(
    path: &str,
    batch: &[usize],
    rows: usize,
    input: usize,
    quant: Quant,
) -> Result<(Array, Array, Option<Array>)> {
    let shape = |last: usize| {
        let mut s = batch.to_vec();
        s.extend([rows, last]);
        s
    };
    let bits = usize::try_from(quant.bits).unwrap_or(0);
    let group_size = usize::try_from(quant.group_size).unwrap_or(0);
    let groups = groups(path, input, group_size)?;
    let weight = placeholder(&shape(input * bits / 32), DType::U32);
    if quant.mode == QuantMode::Affine {
        let scales = placeholder(&shape(groups), DType::F32);
        return Ok((weight, scales.clone(), Some(scales)));
    }
    Ok((weight, placeholder(&shape(groups), DType::U8), None))
}

impl Builder<'_> {
    /// The settings of the quantized layer at `path` and whether it has a global scale.
    fn quant(&self, path: &str) -> Option<(Quant, bool)> {
        let l = (self.quantized)(path)?;
        l.quant.map(|q| (q, l.global_scale))
    }

    fn projection(
        &self,
        path: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Projection> {
        let bias = bias.then(|| param(&[output], DType::F32));
        Ok(match self.quant(path) {
            Some((quant, _)) => {
                let (weight, scales, biases) =
                    quantized_placeholders(path, &[], output, input, quant)?;
                Projection::Quantized(QuantizedLinear {
                    weight,
                    scales,
                    biases,
                    bias,
                    quant,
                })
            }
            None => Projection::Dense(Linear {
                weight: param(&[output, input], DType::F32),
                bias,
            }),
        })
    }

    fn switch(
        &self,
        path: &str,
        input: usize,
        output: usize,
        experts: usize,
        bias: bool,
    ) -> Result<SwitchProjection> {
        let bias = bias.then(|| param(&[experts, output], DType::F32));
        Ok(match self.quant(path) {
            Some((quant, global)) => {
                let (weight, scales, biases) =
                    quantized_placeholders(path, &[experts], output, input, quant)?;
                SwitchProjection::Quantized {
                    weight,
                    scales,
                    biases,
                    global_scale: global.then(|| placeholder(&[experts], DType::F32)),
                    quant,
                    bias,
                }
            }
            None => SwitchProjection::Dense {
                weight: param(&[experts, output, input], DType::F32),
                bias,
            },
        })
    }

    fn embedding(&self, path: &str, count: usize, dims: usize) -> Result<TokenEmbedding> {
        Ok(match self.quant(path) {
            Some((quant, _)) => {
                let (weight, scales, biases) =
                    quantized_placeholders(path, &[], count, dims, quant)?;
                TokenEmbedding::Quantized(QuantizedEmbedding {
                    weight,
                    scales,
                    biases,
                    quant,
                })
            }
            None => TokenEmbedding::Dense(Embedding {
                weight: param(&[count, dims], DType::F32),
            }),
        })
    }

    fn norm(dims: usize, eps: f32) -> RmsNorm {
        RmsNorm {
            weight: param(&[dims], DType::F32),
            eps,
        }
    }
}

/// The rotary embedding of every head's features (nuvec-nn's `Rope`, no parameters), with
/// the fused norm, rotation and cache write nuvec-lm uses.
#[derive(Debug, Clone, Module)]
struct Rope {
    rope: nuvec_nn::Rope,
}

impl Rope {
    fn new(dims: usize, config: &Config) -> Result<Self> {
        Self::with(
            dims,
            (config.rope_theta, &config.rope_scaling),
            config.rope_traditional,
        )
    }

    /// Over `dims` features with a base and scaling.
    fn with(dims: usize, (base, scaling): (f32, &RopeScaling), traditional: bool) -> Result<Self> {
        let mut rope = nuvec_nn::Rope::scaled(int("rope dims", dims)?, base, scaling);
        if traditional {
            rope = rope.traditional();
        }
        Ok(Self { rope })
    }

    fn rotation(&self) -> fast::Rotation<'_> {
        self.rope.rotation()
    }

    /// `x` (`[B, H, L, D]`) normalised by `norm` (if any) and rotated at positions
    /// `offset..`, written into `cache` if given ([`fast::norm_rope`]: one kernel from the
    /// projection to the cache).
    #[track_caller]
    fn norm_rope(
        &self,
        x: &Array,
        norm: Option<&RmsNorm>,
        at: i32,
        cache: Option<&Array>,
    ) -> Array {
        let norm = norm.map(|n| (n.weight.value(), n.eps));
        fast::norm_rope(x, norm, self.rotation(), at, cache)
    }

    /// `x` (`[B, H, L, D]`) normalised by `norm` (if any) and rotated at each sequence's
    /// position (`positions`, `[B]`), written at `slots` of a paged pool if given
    /// ([`fast::norm_rope_at`]: one kernel from the projection to the pool).
    #[track_caller]
    fn norm_rope_at(
        &self,
        x: &Array,
        norm: Option<&RmsNorm>,
        positions: &Array,
        pool: Option<(&Array, &Array)>,
    ) -> Array {
        let norm = norm.map(|n| (n.weight.value(), n.eps));
        fast::norm_rope_at(x, norm, self.rotation(), positions, pool)
    }
}

/// `[B, L, H, D]` as `[B, H, L, D]` and back (a transposition of axes 1 and 2).
const HEADS_FIRST: [i32; 4] = [0, 2, 1, 3];

/// One layer's cache. Attention layers keep keys and values, `[B, kv_heads, capacity,
/// head_dim]`, as mlx-lm's `KVCache`: the capacity grows in steps of 256 positions and
/// each step's keys and values are written into it (in place), so buffers keep their
/// size between growths. Gated DeltaNet layers keep the convolution's last inputs and the
/// recurrent state instead, as mlx-lm's `ArraysCache`.
///
/// A sliding-window layer (window `W`) keeps only what later queries read: when its
/// buffer is full (or holds a long prefill), it drops the positions before the last
/// `W − 1` plus a slack of 256, which [`trim`](Self::trim) can still roll back into. Its
/// positions stay in temporal order (no ring, unlike mlx-lm's `RotatingKVCache`), so
/// attention reads one contiguous view of the last `W − 1 + L` positions.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    /// A sliding-window layer's window.
    window: Option<i32>,
    /// The position of the buffers' first entry (dropped positions before it).
    start: i32,
    /// The convolution's last `kernel - 1` inputs, `[B, kernel - 1, channels]`.
    conv: Option<Array>,
    /// The recurrent state, `[B, value_heads, value_head_dim, key_head_dim]` (float32).
    recurrent: Option<Array>,
    /// Positions cached.
    offset: i32,
    /// Whether Gated DeltaNet layers keep every step's state ([`KvCache::keep_steps`]).
    keep_steps: bool,
    /// From the last forward with `keep_steps`: every step's recurrent state
    /// (`[B, T, ...]`) and the convolution's input (`[B, kernel − 1 + T, channels]`).
    steps: Option<(Array, Array)>,
    /// Attention keys and values are returned, not kept ([`Model::forward_uncached`]).
    uncached: bool,
}

/// Positions added to a cache's capacity at a time.
const CACHE_STEP: i32 = 256;

/// Positions a sliding-window cache keeps beyond its window, for [`KvCache::trim`].
const WINDOW_SLACK: i32 = 256;

impl KvCache {
    /// An empty cache for a layer attending to a sliding window of `window` keys.
    pub(crate) fn windowed(window: i32) -> Self {
        Self {
            window: Some(window),
            ..Self::default()
        }
    }

    /// Positions cached.
    pub fn offset(&self) -> usize {
        usize::try_from(self.offset).unwrap_or_default()
    }

    /// The first position the cache still holds (0 but for sliding-window layers).
    pub(crate) fn start(&self) -> usize {
        usize::try_from(self.start).unwrap_or_default()
    }

    /// Writes `keys` and `values` (`[B, kv_heads, L, head_dim]`) after the cached positions,
    /// the keys normalised by `norm` (if any) and rotated as they are written, and returns
    /// all cached positions.
    #[track_caller]
    fn update(
        &mut self,
        rope: &Rope,
        norm: Option<&RmsNorm>,
        keys: &Array,
        values: &Array,
    ) -> (Array, Array) {
        let (prev, start) = (self.offset, self.start);
        let l = values.shape().get(2).copied().unwrap_or_default();
        let end = prev + l;
        if self.uncached {
            self.offset = end;
            return (rope.norm_rope(keys, norm, prev, None), values.clone());
        }
        let fits = self
            .keys
            .as_ref()
            .is_some_and(|k| end - start <= k.shape()[2]);
        // A window keeps the last `W − 1` positions (what the next queries read) and the slack.
        let keep = self.window.map(|w| w - 1 + WINDOW_SLACK);
        let excess = keep.is_some_and(|keep| prev - start > keep + 2 * CACHE_STEP);
        let (k, v) = if fits && !excess {
            // Each write takes the cache's only handle, so it is done in place.
            let all = self.values.take().expect("allocated with the keys");
            let at = prev - start..end - start;
            let k = if start == 0 {
                rope.norm_rope(keys, norm, prev, self.keys.take().as_ref())
            } else {
                let rotated = rope.norm_rope(keys, norm, prev, None);
                let cached = self.keys.take().expect("allocated with the values");
                cached.at((.., .., at.clone())).set(&rotated)
            };
            (k, all.at((.., .., at)).set(values))
        } else {
            // Grown by whole steps (mlx-lm's rule), written in one concatenation: the kept
            // positions, the new ones, then zeros up to the step (no copy of the grown cache
            // to update it). A window drops what no later query reads and keeps a step of
            // room, so it compacts once per step.
            let from = keep.map_or(start, |keep| (prev - keep).max(start));
            let n = prev - from + l;
            let room = if keep.is_some() { CACHE_STEP } else { 0 };
            let pad = (n + CACHE_STEP - 1) / CACHE_STEP * CACHE_STEP - n + room;
            let keys = rope.norm_rope(keys, norm, prev, None);
            let kept = (from - start, prev - start);
            let k = grow(self.keys.take(), kept, &keys, pad);
            self.start = from;
            (k, grow(self.values.take(), kept, values, pad))
        };
        self.offset = end;
        self.keys = Some(k.clone());
        self.values = Some(v.clone());
        // Attention reads the last `W − 1 + L` positions (all of them without a window).
        let first = self
            .window
            .map_or(self.start, |w| (prev - (w - 1)).max(self.start));
        let view = |a: &Array| between(a, first - self.start, end - self.start);
        (view(&k), view(&v))
    }

    /// Makes Gated DeltaNet layers keep the state after every position of the following
    /// forwards ([`nuvec::fast::gated_delta_steps`]), so that [`trim`](Self::trim) can roll
    /// them back. It costs writing each position's state; speculative decoding turns it on
    /// for verification only. Attention layers ignore it.
    pub fn keep_steps(&mut self, on: bool) {
        self.keep_steps = on;
        if !on {
            self.steps = None;
        }
    }

    /// Forgets the last `n` positions (rejected speculative tokens): later writes overwrite
    /// them. A Gated DeltaNet layer rolls back to the state after the position before them,
    /// which needs the last forward to have kept its steps ([`keep_steps`](Self::keep_steps))
    /// and `n` to be less than that forward's length.
    pub fn trim(&mut self, n: usize) -> Result<()> {
        let invalid = |msg: String| nuvec::Error::InvalidArgument(msg);
        if self.conv.is_some() || self.recurrent.is_some() {
            if n == 0 {
                return Ok(());
            }
            let Some((states, input)) = self.steps.take() else {
                return Err(invalid(
                    "a Gated DeltaNet layer rolls back only after a forward that kept its \
                     steps (KvCache::keep_steps)"
                        .into(),
                ));
            };
            let [_, t, ..] = states.shape() else {
                return Err(invalid("a Gated DeltaNet layer without steps".into()));
            };
            let t = *t;
            let Some(kept) = i32::try_from(n).ok().filter(|&n| n < t).map(|n| t - n) else {
                return Err(invalid(format!(
                    "cannot roll back {n} of the last forward's {t} positions"
                )));
            };
            self.recurrent = Some(states.i((.., kept - 1)));
            // The last `kernel − 1` inputs before the rejected positions (a copy: the view
            // would keep the whole input alive).
            let window = input.shape()[1] - t;
            self.conv = Some(input.i((.., kept..kept + window)).multiply(1.0));
            self.offset -= t - kept;
            return Ok(());
        }
        let Some(n) = i32::try_from(n).ok().filter(|&n| n <= self.offset) else {
            return Err(invalid(format!(
                "cannot trim {n} of {} cached positions",
                self.offset
            )));
        };
        // A window's next queries read the `W − 1` positions before the new offset.
        if let Some(w) = self.window
            && self.start > 0
            && self.offset - n - (w - 1) < self.start
        {
            return Err(invalid(format!(
                "cannot trim {n} positions: a sliding-window cache keeps {WINDOW_SLACK} \
                 beyond its window ({} cached from position {})",
                self.offset - self.start,
                self.start
            )));
        }
        self.offset -= n;
        Ok(())
    }

    /// The cached keys and values of an attention layer, `[B, kv_heads, offset − start,
    /// head_dim]` (views of positions [`start`](Self::start)`..offset`), if any.
    pub(crate) fn keys_values(&self) -> Option<(Array, Array)> {
        let n = self.offset - self.start;
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((positions(k, n), positions(v, n))),
            _ => None,
        }
    }

    /// A Gated DeltaNet layer's convolution and recurrent states (`[B, kernel − 1,
    /// channels]` and `[B, Hv, Dv, Dk]`), if any.
    pub(crate) fn linear_state(&self) -> Option<(Array, Array)> {
        Some((self.conv.clone()?, self.recurrent.clone()?))
    }

    /// Sequence `i` of a batched cache (views of its row, the same positions). Kept steps
    /// are not carried.
    #[cfg(test)]
    pub(crate) fn row(&self, i: i32) -> KvCache {
        let row = |a: &Option<Array>| a.as_ref().map(|a| a.i(i..i + 1));
        KvCache {
            keys: row(&self.keys),
            values: row(&self.values),
            window: self.window,
            start: self.start,
            conv: row(&self.conv),
            recurrent: row(&self.recurrent),
            offset: self.offset,
            keep_steps: false,
            steps: None,
            uncached: self.uncached,
        }
    }

    /// The cached arrays (to evaluate them).
    pub fn state(&self) -> Vec<Array> {
        [&self.keys, &self.values, &self.conv, &self.recurrent]
            .into_iter()
            .flatten()
            .cloned()
            .collect()
    }

    /// Makes the state's evaluation evaluate `dep` first (a pipeline stage's send, so that
    /// evaluating the caches, as prefill does, also sends).
    pub(crate) fn attach(&mut self, dep: &Array) {
        for a in [
            &mut self.keys,
            &mut self.values,
            &mut self.conv,
            &mut self.recurrent,
        ]
        .into_iter()
        .flatten()
        {
            *a = nuvec::ops::depends(a, std::slice::from_ref(dep));
        }
    }
}

/// The first `n` positions (axis 2) of a cache array, as a view.
#[track_caller]
fn positions(a: &Array, n: i32) -> Array {
    if a.shape().get(2) == Some(&n) {
        return a.clone();
    }
    a.i((.., .., ..n))
}

/// Positions `from..to` (axis 2) of a cache array, as a view.
#[track_caller]
fn between(a: &Array, from: i32, to: i32) -> Array {
    if from == 0 {
        return positions(a, to);
    }
    a.i((.., .., from..to))
}

/// `old`'s positions `kept.0..kept.1` (axis 2), then `new`, then `pad` positions of zeros,
/// in one concatenation.
#[track_caller]
fn grow(old: Option<Array>, kept: (i32, i32), new: &Array, pad: i32) -> Array {
    let mut parts = Vec::with_capacity(3);
    if let Some(old) = old
        && kept.1 > kept.0
    {
        parts.push(between(&old, kept.0, kept.1));
    }
    parts.push(new.clone());
    if pad > 0 {
        let mut shape = new.shape().to_vec();
        if let Some(d) = shape.get_mut(2) {
            *d = pad;
        }
        // A broadcast zero: the concatenation writes the zeros on the stream's device, as
        // mlx-lm's lazy `mx.zeros`.
        parts.push(ops::broadcast_to(&ops::zeros((), new.dtype()), shape));
    }
    ops::concatenate(&parts, 2)
}

#[derive(Debug, Clone, Module)]
struct Attention {
    q_proj: Projection,
    /// `None` in KV-shared layers (Gemma 4), which read another layer's keys and values.
    k_proj: Option<Projection>,
    /// `None` also with K = V (Gemma 4's full layers: the keys' projection is the values').
    v_proj: Option<Projection>,
    o_proj: Projection,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    #[tree(skip)]
    heads: i32,
    #[tree(skip)]
    kv_heads: i32,
    #[tree(skip)]
    scale: f32,
    rope: Rope,
    /// The query projection also yields a gate of the output (`o · sigmoid(gate)`).
    #[tree(skip)]
    gate: bool,
    /// Learned attention sinks, one logit per query head (gpt-oss).
    sinks: Option<Param>,
    /// A sliding window of this many keys.
    #[tree(skip)]
    window: Option<i32>,
    /// The RoPE scaling's input scale (YaRN's mscale; 1 otherwise), folded into the
    /// attention scale.
    #[tree(skip)]
    mscale: f32,
    /// The values' RMS normalisation without a weight (Gemma 4's `v_norm`), its epsilon.
    #[tree(skip)]
    v_norm: Option<f32>,
}

impl Attention {
    fn new(b: &Builder<'_>, path: &str, c: &Config, window: Option<usize>) -> Result<Self> {
        let (d, hd) = (c.hidden_size, c.head_dim);
        let (q, kv) = (c.num_attention_heads * hd, c.num_key_value_heads * hd);
        let p = |name: &str| format!("{path}.{name}");
        let norm = || c.qk_norm.then(|| Builder::norm(hd, c.rms_norm_eps));
        let gate = c.attn_output_gate;
        let rope = Rope::new(c.rope_dims, c)?;
        // mlx-lm multiplies the rotated features of q and k by the scaling's mscale (YaRN):
        // the scores scale by mscale², folded into the attention scale (both rotations stay
        // one fused kernel each).
        let mscale = rope.rope.mscale();
        if mscale != 1.0 && c.rope_dims != hd {
            return Err(nuvec::Error::InvalidArgument(format!(
                "{path}: a RoPE scaling with an input scale over {} of {hd} features is not \
                 supported",
                c.rope_dims
            )));
        }
        Ok(Self {
            q_proj: b.projection(&p("q_proj"), d, q * (1 + usize::from(gate)), c.qkv_bias)?,
            k_proj: Some(b.projection(&p("k_proj"), d, kv, c.qkv_bias)?),
            v_proj: Some(b.projection(&p("v_proj"), d, kv, c.qkv_bias)?),
            o_proj: b.projection(&p("o_proj"), q, d, c.o_bias)?,
            q_norm: norm(),
            k_norm: norm(),
            heads: int(path, c.num_attention_heads)?,
            kv_heads: int(path, c.num_key_value_heads)?,
            scale: (hd as f32).powf(-0.5),
            mscale,
            rope,
            gate,
            sinks: c
                .attention_sinks
                .then(|| param(&[c.num_attention_heads], DType::F32)),
            window: window.map(|w| int(path, w)).transpose()?,
            v_norm: None,
        })
    }

    /// The key projection (layers that project keys).
    fn k_proj(&self) -> &Projection {
        self.k_proj.as_ref().expect("a layer with its own keys")
    }

    /// Whether the mscale is applied to q and k explicitly (mlx-lm's multiply, rounded per
    /// element in half precision) rather than folded into the attention scale (float32:
    /// the same up to rounding, without the two multiplications).
    fn explicit_mscale(&self, dtype: DType) -> bool {
        self.mscale != 1.0 && dtype != DType::F32
    }

    /// `x` (queries or keys before the rotation) times the mscale, where it is explicit.
    #[track_caller]
    fn input_scale(&self, x: Array) -> Array {
        match self.explicit_mscale(x.dtype()) {
            true => x.multiply(self.mscale),
            false => x,
        }
    }

    /// The attention scale for activations of `dtype` (with the folded mscale²).
    fn scale(&self, dtype: DType) -> f32 {
        match self.mscale == 1.0 || self.explicit_mscale(dtype) {
            true => self.scale,
            false => self.scale * self.mscale * self.mscale,
        }
    }

    /// The attention options: the layer's window (causal) and sinks.
    fn options<'a>(&'a self, mask: AttentionMask<'a>) -> fast::AttentionOptions<'a> {
        let options = match self.window {
            Some(w) => fast::AttentionOptions::causal().window(w),
            None => mask.into(),
        };
        options.sinks(self.sinks.as_ref().map(Param::value))
    }

    /// The queries `[B, L, H, D]` and, with an output gate, the gate `[B, L, H·D]` (each
    /// head's projection is its query, then its gate).
    #[track_caller]
    fn query(&self, x: &Array, [b, l]: [i32; 2]) -> (Array, Option<Array>) {
        let q = self.q_proj.forward(x).reshape([b, l, self.heads, -1]);
        if !self.gate {
            return (q, None);
        }
        let [q, g] = q.split_n::<2>(-1);
        (q, Some(g.reshape([b, l, -1])))
    }

    /// The attention output `o` (`[B, H, L, D]`) through the output projection, gated.
    #[track_caller]
    fn output(&self, o: &Array, gate: Option<Array>, [b, l]: [i32; 2]) -> Array {
        let o = o.transpose(HEADS_FIRST).reshape([b, l, -1]);
        match gate {
            Some(g) => self.o_proj.forward(&o.multiply(g.sigmoid())),
            None => self.o_proj.forward(&o),
        }
    }

    #[track_caller]
    fn forward(&self, x: &Array, mask: AttentionMask<'_>, cache: &mut KvCache) -> Array {
        self.forward_kv(x, mask, cache, None).0
    }

    /// [`Attention::forward`], also giving the keys and values attended to; a KV-shared
    /// layer attends to `shared` (another layer's, with the position of its first query)
    /// and leaves its own cache's arrays empty (its offset advances).
    #[track_caller]
    fn forward_kv(
        &self,
        x: &Array,
        mask: AttentionMask<'_>,
        cache: &mut KvCache,
        shared: Option<(&(Array, Array), i32)>,
    ) -> (Array, (Array, Array)) {
        let [b, l, _] = x.shape_n::<3>().unwrap_or_default();
        let (q, gate) = self.query(x, [b, l]);
        let at = shared.map_or(cache.offset, |(_, at)| at);
        // The query and key (written into the cache) normalised and rotated in one kernel.
        let q = self.input_scale(q.transpose(HEADS_FIRST));
        let q = self.rope.norm_rope(&q, self.q_norm.as_ref(), at, None);
        let (k, v) = match shared {
            Some((kv, _)) => {
                cache.offset += l;
                kv.clone()
            }
            None => {
                let heads = [b, l, self.kv_heads, -1];
                let k = self
                    .k_proj()
                    .forward(x)
                    .reshape(heads)
                    .transpose(HEADS_FIRST);
                let v = match &self.v_proj {
                    Some(v) => v.forward(x).reshape(heads).transpose(HEADS_FIRST),
                    None => k.clone(),
                };
                let v = match self.v_norm {
                    Some(eps) => fast::rms_norm(&v, None, eps),
                    None => v,
                };
                let k = self.input_scale(k);
                cache.update(&self.rope, self.k_norm.as_ref(), &k, &v)
            }
        };
        let scale = self.scale(q.dtype());
        let o = fast::scaled_dot_product_attention(&q, &k, &v, scale, &self.options(mask));
        (self.output(&o, gate, [b, l]), (k, v))
    }
}

/// One attention layer's paged keys and values, `[blocks, kv_heads, block_size,
/// head_dim]` each ([`crate::PagedCache`]).
#[derive(Debug, Clone, Default)]
pub(crate) struct PagedLayer {
    pub(crate) keys: Option<Array>,
    pub(crate) values: Option<Array>,
    /// A Gated DeltaNet layer's states, one slot per sequence: the convolution's last
    /// inputs `[slots, kernel − 1, channels]` and the recurrent states `[slots, Hv, Dv, Dk]`.
    pub(crate) conv: Option<Array>,
    pub(crate) recurrent: Option<Array>,
    /// The pools' blocks and positions per block (attention layers), and state slots
    /// (Gated DeltaNet layers): pools not allocated yet are created as zeros.
    pub(crate) pool_size: (i32, i32),
    pub(crate) state_slots: i32,
    /// The block group of an attention layer (its step's slots and tables).
    pub(crate) group: usize,
}

/// What one batched decoding step writes and reads, per sequence (`[B]`, int32): its
/// position (for RoPE), its length after the step and its Gated DeltaNet state slot; per
/// block group, the slot of its new position and its block table.
#[derive(Debug, Clone)]
pub(crate) struct PagedStep {
    pub(crate) positions: Array,
    pub(crate) lengths: Array,
    pub(crate) state_slots: Array,
    pub(crate) groups: Vec<GroupStep>,
    /// Each sequence's length after the step, and the positions per block, on the host.
    pub(crate) host_lengths: Vec<usize>,
    pub(crate) block_size: usize,
}

/// What a layer keeps in a paged cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PagedKind {
    /// Keys and values: every position (`None`) or a window of `W` keys.
    Keys(Option<usize>),
    /// A Gated DeltaNet layer's states, at the sequence's slot.
    States,
    /// A KV-shared layer: its source layer's keys and values.
    Shared,
}

/// A block group's part of a [`PagedStep`]: each sequence's new slot (`[B]`) and block
/// table (`[B, max_blocks]`).
#[derive(Debug, Clone)]
pub(crate) struct GroupStep {
    pub(crate) slots: Array,
    pub(crate) tables: Array,
    /// Each sequence's block table, on the host.
    pub(crate) host_tables: Vec<Vec<u32>>,
}

impl PagedLayer {
    /// The keys (`[B, kv_heads, 1, head_dim]`) normalised by `k_norm`, rotated at
    /// `positions` and written at `slots` of their pool in one kernel, the values (`[B,
    /// kv_heads, head_dim]`) written at `slots` of theirs; and the pools.
    #[track_caller]
    fn write_kv(
        &mut self,
        (rope, k_norm): (&Rope, Option<&RmsNorm>),
        (k, v): (&Array, &Array),
        positions: &Array,
        slots: &Array,
    ) -> (Array, Array) {
        let size = self.pool_size;
        let pool = take_pool(&mut self.keys, size, k);
        let slots_2d = slots.reshape([-1, 1]);
        let keys = rope.norm_rope_at(k, k_norm, positions, Some((&pool, &slots_2d)));
        self.keys = Some(keys.clone());
        let pool = take_pool(&mut self.values, size, k);
        let values = fast::paged_write(&pool, v, slots);
        self.values = Some(values.clone());
        (keys, values)
    }
}

/// The pool in `pool` (of `size`, blocks and positions per block), taken out so that the
/// write is done in place: zeros shaped for `like`'s heads and head size (`[B, heads, 1,
/// head_dim]`) until a prefilled sequence brought it.
fn take_pool(pool: &mut Option<Array>, size: (i32, i32), like: &Array) -> Array {
    pool.take().unwrap_or_else(|| {
        let [_, heads, _, dims] = like.shape_n::<4>().unwrap_or_default();
        ops::zeros([size.0, heads, size.1, dims], like.dtype())
    })
}

impl Attention {
    /// One new position per sequence (`x` is `[B, 1, D]`) over a paged cache: the query
    /// and key normalised and rotated at each sequence's position, the key and value
    /// written into their slots, then attention through the block tables.
    #[track_caller]
    fn forward_paged(&self, x: &Array, layer: &mut PagedLayer, step: &PagedStep) -> Array {
        self.forward_paged_kv(x, layer, step, None).0
    }

    /// [`Attention::forward_paged`], also giving the pools attended to (and the group of
    /// their tables); a KV-shared layer attends to `shared` (its source layer's).
    #[track_caller]
    fn forward_paged_kv(
        &self,
        x: &Array,
        layer: &mut PagedLayer,
        step: &PagedStep,
        shared: Option<&(Array, Array, usize)>,
    ) -> (Array, (Array, Array, usize)) {
        let [b, l, _] = x.shape_n::<3>().unwrap_or_default();
        let (q, gate) = self.query(x, [b, l]);
        let q = self.input_scale(q).transpose(HEADS_FIRST);
        let q = self
            .rope
            .norm_rope_at(&q, self.q_norm.as_ref(), &step.positions, None);
        let (keys, values, group) = match shared {
            Some(kv) => kv.clone(),
            None => {
                let g = &step.groups[layer.group];
                let k = self.k_proj().forward(x).reshape([b, l, self.kv_heads, -1]);
                // K = V: the values are the keys' projection, before its norm and rotation.
                let v = match &self.v_proj {
                    Some(v) => v.forward(x).reshape([b, l, self.kv_heads, -1]),
                    None => k.clone(),
                };
                let v = match self.v_norm {
                    Some(eps) => fast::rms_norm(&v, None, eps),
                    None => v,
                };
                let k = self.input_scale(k).transpose(HEADS_FIRST);
                let (keys, values) = layer.write_kv(
                    (&self.rope, self.k_norm.as_ref()),
                    (&k, &v.reshape([b, self.kv_heads, -1])),
                    &step.positions,
                    &g.slots,
                );
                (keys, values, layer.group)
            }
        };
        let scale = self.scale(q.dtype());
        let o = fast::paged_attention(
            &q,
            &keys,
            &values,
            &step.groups[group].tables,
            &step.lengths,
            scale,
            &self.options(AttentionMask::Causal),
        );
        let o = self.as_contiguous_decode(o, &q, (&keys, &values), step, group, scale);
        (self.output(&o, gate, [b, l]), (keys, values, group))
    }

    /// At head size 512, one-token decode over the contiguous cache runs MLX's composition
    /// below 1,024 keys (or without 8 query heads per key head), not the vector kernel the
    /// paged kernel equals: row-exact ([`nuvec::with_row_exact`], a batch-invariant
    /// `Batcher`), those rows are computed as that decode computes them, over their keys
    /// gathered from the pools, so a sequence keeps its solo bits.
    #[track_caller]
    fn as_contiguous_decode(
        &self,
        paged: Array,
        q: &Array,
        (keys, values): (&Array, &Array),
        step: &PagedStep,
        group: usize,
        scale: f32,
    ) -> Array {
        let d = q.shape().get(3).copied().unwrap_or(0);
        let vector = |n: usize| self.heads == 8 * self.kv_heads && n >= 1024;
        if !nuvec::row_exact()
            || d != 512
            || q.shape().get(2) != Some(&1)
            || step.host_lengths.iter().all(|&n| vector(n))
        {
            return paged;
        }
        let tables = &step.groups[group].host_tables;
        let rows: Vec<Array> = step
            .host_lengths
            .iter()
            .enumerate()
            .map(|(r, &n)| {
                let r = r as i32;
                if vector(n) {
                    return paged.i(r..r + 1);
                }
                let blocks = &tables[r as usize][..n.div_ceil(step.block_size)];
                let blocks = Array::from_slice(blocks, [blocks.len() as i32]);
                let gather = |pool: &Array| {
                    pool.take(&blocks, 0)
                        .transpose([1, 0, 2, 3])
                        .reshape([1, self.kv_heads, -1, d])
                        .i((.., .., 0..n as i32, ..))
                };
                fast::scaled_dot_product_attention(
                    &q.i(r..r + 1),
                    &gather(keys),
                    &gather(values),
                    scale,
                    &self.options(AttentionMask::None),
                )
            })
            .collect();
        ops::concatenate(&rows, 0)
    }
}

/// `log(1 + exp(x))`, stably (MLX's `nn.softplus`, `logaddexp(x, 0)`): in float32,
/// rounded once to `x`'s dtype, as MLX's kernel.
#[track_caller]
fn softplus(x: &Array) -> Array {
    let xf = x.astype(DType::F32);
    let y = xf
        .maximum(0.0)
        .add(xf.abs().negative().exp().add(1.0).log());
    y.astype(x.dtype())
}

/// `silu(z) · n` in float32, rounded to `n`'s dtype, uncompiled.
#[track_caller]
fn gated_norm_graph(n: &Array, z: &Array) -> Array {
    let (nf, zf) = (n.astype(DType::F32), z.astype(DType::F32));
    silu(&zf).multiply(&nf).astype(n.dtype())
}

/// The gated RMS normalisation of Gated DeltaNet's output (mlx-lm's
/// `Qwen3NextRMSNormGated` with its compiled `precise_swiglu`): `silu(z) · rms_norm(x)`, in
/// float32.
#[track_caller]
fn gated_norm(x: &Array, z: &Array, norm: &RmsNorm) -> Array {
    static GATE: LazyLock<Compiled2> = LazyLock::new(|| {
        Box::new(nuvec::compile(|(n, z): &(Array, Array)| {
            gated_norm_graph(n, z)
        }))
    });
    let n = norm.forward(x);
    GATE(&(n.clone(), z.clone())).unwrap_or_else(|_| gated_norm_graph(&n, z))
}

/// `exp(−exp(A_log) · softplus(a + dt_bias))` in float32, uncompiled.
#[track_caller]
fn decay_graph(a: &Array, dt_bias: &Array, a_log: &Array) -> Array {
    let rate = a_log.astype(DType::F32).exp();
    rate.negative().multiply(softplus(&a.add(dt_bias))).exp()
}

/// Gated DeltaNet's per-head decay, `exp(−exp(A_log) · softplus(a + dt_bias))` in float32
/// (mlx-lm's compiled `compute_g`).
#[track_caller]
fn decay(a: &Array, dt_bias: &Array, a_log: &Array) -> Array {
    static DECAY: LazyLock<Compiled3> = LazyLock::new(|| {
        Box::new(nuvec::compile(|(a, dt, log): &(Array, Array, Array)| {
            decay_graph(a, dt, log)
        }))
    });
    DECAY(&(a.clone(), dt_bias.clone(), a_log.clone()))
        .unwrap_or_else(|_| decay_graph(a, dt_bias, a_log))
}

/// The causal depthwise convolution of `input` (`[B, L + K − 1, C]`) with `weight`
/// (`[C, K, 1]`), then `silu`, as mlx-lm: the convolution is one depthwise kernel (summed in
/// float32 and rounded once, as MLX's).
#[track_caller]
fn causal_conv_silu(input: &Array, weight: &Array) -> Array {
    let channels = weight.shape().first().copied().unwrap_or_default();
    silu(&input.conv1d(weight, 1, 0, 1, channels))
}

/// Gated DeltaNet's head counts and sizes and its convolution's kernel, as dimensions.
#[derive(Debug, Clone, Copy, nuvec::Static)]
struct DeltaSizes {
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
    kernel: i32,
}

/// A Gated DeltaNet layer (mlx-lm's `qwen3_5.GatedDeltaNet`): a causal depthwise
/// convolution over the query, key and value projections, then the gated delta rule
/// ([`nuvec::fast::gated_delta`]) with per-head decays and write strengths, and a gated
/// normalisation of the output.
#[derive(Debug, Clone, Module)]
struct GatedDeltaNet {
    in_proj_qkv: Projection,
    in_proj_z: Projection,
    in_proj_b: Projection,
    in_proj_a: Projection,
    /// The depthwise convolution's weights, `[channels, kernel, 1]`.
    #[tree(rename = "conv1d.weight")]
    conv1d: Param,
    dt_bias: Param,
    #[tree(rename = "A_log")]
    a_log: Param,
    norm: RmsNorm,
    out_proj: Projection,
    #[tree(skip)]
    sizes: DeltaSizes,
}

impl GatedDeltaNet {
    fn new(b: &Builder<'_>, path: &str, c: &Config, sizes: LinearAttention) -> Result<Self> {
        let d = c.hidden_size;
        let key_dim = sizes.key_heads * sizes.key_head_dim;
        let value_dim = sizes.value_heads * sizes.value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let p = |name: &str| format!("{path}.{name}");
        let hv = sizes.value_heads;
        Ok(Self {
            in_proj_qkv: b.projection(&p("in_proj_qkv"), d, conv_dim, false)?,
            in_proj_z: b.projection(&p("in_proj_z"), d, value_dim, false)?,
            in_proj_b: b.projection(&p("in_proj_b"), d, hv, false)?,
            in_proj_a: b.projection(&p("in_proj_a"), d, hv, false)?,
            conv1d: param(&[conv_dim, sizes.conv_kernel, 1], DType::F32),
            dt_bias: param(&[hv], DType::F32),
            a_log: param(&[hv], DType::F32),
            norm: Builder::norm(sizes.value_head_dim, c.rms_norm_eps),
            out_proj: b.projection(&p("out_proj"), value_dim, d, false)?,
            sizes: DeltaSizes {
                hk: int(path, sizes.key_heads)?,
                hv: int(path, hv)?,
                dk: int(path, sizes.key_head_dim)?,
                dv: int(path, sizes.value_head_dim)?,
                kernel: int(path, sizes.conv_kernel)?,
            },
        })
    }

    /// The convolution's output split into queries, keys and values (`[B, L, H, D]`),
    /// normalised as mlx-lm's `normalize_qk`, unit queries and keys with the 1/sqrt(Dk)
    /// scale in the query.
    #[track_caller]
    fn qkv(&self, conv: &Array, [b, l]: [i32; 2]) -> [Array; 3] {
        let DeltaSizes { hk, hv, dk, dv, .. } = self.sizes;
        let parts = conv.split_at(&[hk * dk, 2 * hk * dk], -1);
        let [q, k, v]: [Array; 3] = parts.try_into().expect("three parts");
        let inv = (dk as f32).powf(-0.5);
        let eps = 1e-6 * inv * inv;
        let q =
            fast::rms_norm(&q.reshape([b, l, hk, dk]), None, eps).multiply(f64::from(inv * inv));
        let k = fast::rms_norm(&k.reshape([b, l, hk, dk]), None, eps).multiply(f64::from(inv));
        [q, k, v.reshape([b, l, hv, dv])]
    }

    /// The gated normalisation of the delta rule's output, through the output projection.
    #[track_caller]
    fn output(&self, out: &Array, z: &Array, [b, l]: [i32; 2]) -> Array {
        let DeltaSizes { hv, dv, .. } = self.sizes;
        let out = gated_norm(out, &z.reshape([b, l, hv, dv]), &self.norm);
        self.out_proj.forward(&out.reshape([b, l, -1]))
    }

    #[track_caller]
    fn forward(&self, x: &Array, cache: &mut KvCache) -> Array {
        let DeltaSizes {
            hv, dv, dk, kernel, ..
        } = self.sizes;
        let [b, l, _] = x.shape_n::<3>().unwrap_or_default();
        let qkv = self.in_proj_qkv.forward(x);
        let z = self.in_proj_z.forward(x);
        let (beta, a) = (self.in_proj_b.forward(x), self.in_proj_a.forward(x));
        // The causal convolution sees the previous step's last `kernel - 1` inputs.
        let state = match cache.conv.take() {
            Some(c) => c,
            None => {
                let channels = qkv.shape().get(2).copied().unwrap_or_default();
                ops::zeros([b, kernel - 1, channels], qkv.dtype())
            }
        };
        let input = ops::concatenate(&[state, qkv], 1);
        let kept = input.i((.., l..l + kernel - 1));
        // After a prompt, a copy: the view would keep the whole input alive.
        cache.conv = Some(if l > 1 { kept.multiply(1.0) } else { kept });
        let conv = causal_conv_silu(&input, &self.conv1d);
        let [q, k, v] = self.qkv(&conv, [b, l]);
        // Per-head decay and write strength (mlx-lm's `compute_g`, in float32).
        let g = decay(&a, &self.dt_bias, &self.a_log);
        let beta = beta.sigmoid();
        let state = match cache.recurrent.take() {
            Some(s) => s,
            None => ops::zeros([b, hv, dv, dk], DType::F32),
        };
        let out = if cache.keep_steps {
            let (out, states) = fast::gated_delta_steps(&q, &k, &v, &g, &beta, &state);
            cache.recurrent = Some(states.i((.., l - 1)));
            cache.steps = Some((states, input));
            out
        } else {
            let (out, state) = fast::gated_delta(&q, &k, &v, &g, &beta, &state);
            cache.recurrent = Some(state);
            out
        };
        cache.offset += l;
        self.output(&out, &z, [b, l])
    }
}

impl GatedDeltaNet {
    /// One new position per sequence (`x` is `[B, 1, D]`) with each sequence's states at
    /// its slot of the layer's pools, both updated in place
    /// ([`fast::causal_conv1d_slots`], [`fast::gated_delta_slots`]).
    #[track_caller]
    fn forward_paged(&self, x: &Array, layer: &mut PagedLayer, step: &PagedStep) -> Array {
        let DeltaSizes {
            hv, dv, dk, kernel, ..
        } = self.sizes;
        let [b, l, _] = x.shape_n::<3>().unwrap_or_default();
        let qkv = self.in_proj_qkv.forward(x);
        let z = self.in_proj_z.forward(x);
        let (beta, a) = (self.in_proj_b.forward(x), self.in_proj_a.forward(x));
        // The convolution over each sequence's last inputs and the new ones, which are
        // written back in place (zeros until a prefilled sequence brought the pools).
        let slots = layer.state_slots;
        let conv_pool = match layer.conv.take() {
            Some(p) => p,
            None => {
                let channels = qkv.shape().get(2).copied().unwrap_or_default();
                ops::zeros([slots, kernel - 1, channels], qkv.dtype())
            }
        };
        let (conv, conv_pool) =
            fast::causal_conv1d_slots(&qkv, &self.conv1d, &conv_pool, &step.state_slots, true);
        layer.conv = Some(conv_pool);
        let [q, k, v] = self.qkv(&conv, [b, l]);
        let g = decay(&a, &self.dt_bias, &self.a_log);
        let beta = beta.sigmoid();
        let pool = match layer.recurrent.take() {
            Some(p) => p,
            None => ops::zeros([slots, hv, dv, dk], DType::F32),
        };
        let (out, pool) = fast::gated_delta_slots(&q, &k, &v, &g, &beta, &pool, &step.state_slots);
        layer.recurrent = Some(pool);
        self.output(&out, &z, [b, l])
    }
}

/// A layer's token mixer, named as mlx-lm's (`self_attn`, `linear_attn`).
#[derive(Debug, Clone, Module)]
enum Mixer {
    #[tree(rename = "self_attn")]
    Attention(Attention),
    #[tree(rename = "linear_attn")]
    Linear(GatedDeltaNet),
}

#[derive(Debug, Clone, Module)]
struct Mlp {
    gate_proj: Projection,
    up_proj: Projection,
    down_proj: Projection,
    /// `gelu_approx(gate) · up` (Gemma) instead of SwiGLU.
    #[tree(skip)]
    geglu: bool,
}

impl Mlp {
    fn new(b: &Builder<'_>, path: &str, input: usize, hidden: usize, bias: bool) -> Result<Self> {
        let p = |name: &str| format!("{path}.{name}");
        Ok(Self {
            gate_proj: b.projection(&p("gate_proj"), input, hidden, bias)?,
            up_proj: b.projection(&p("up_proj"), input, hidden, bias)?,
            down_proj: b.projection(&p("down_proj"), hidden, input, bias)?,
            geglu: false,
        })
    }

    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        let (gate, up) = (self.gate_proj.forward(x), self.up_proj.forward(x));
        let gated = match self.geglu {
            true => nuvec_nn::geglu(&gate, &up),
            false => swiglu(&gate, &up),
        };
        self.down_proj.forward(&gated)
    }
}

/// One linear layer per expert, `[experts, out, in]`, applied to each row with its own
/// expert (mlx-lm's `SwitchLinear` and `QuantizedSwitchLinear`).
#[derive(Debug, Clone, Module)]
pub(crate) enum SwitchProjection {
    /// Full precision: [`Array::gather_mm`]; a bias per expert (`[E, out]`, gpt-oss).
    Dense { weight: Param, bias: Option<Param> },
    /// Quantized experts: [`Array::gather_qmm`]; the packed weights are buffers.
    Quantized {
        /// A bias per expert (`[E, out]`, gpt-oss).
        bias: Option<Param>,
        #[tree(buffer)]
        weight: Array,
        #[tree(buffer)]
        scales: Array,
        #[tree(buffer)]
        biases: Option<Array>,
        /// nvfp4: one float32 scale per expert (mlx-lm's `global_scale`).
        #[tree(buffer)]
        global_scale: Option<Array>,
        #[tree(skip)]
        quant: Quant,
    },
}

impl SwitchProjection {
    /// Whether a batch of `b` items of `pairs` (token, expert) pairs each takes the
    /// kernels each item takes alone, so every row keeps its bits: the sorted path from 64
    /// pairs on (`SwitchGlu::forward`) and the grouped gather (quantized experts; dense
    /// ones are not batched).
    fn batches_like_items(&self, b: usize, pairs: usize) -> bool {
        let Self::Quantized { weight, .. } = self else {
            return false;
        };
        let experts = weight.shape().first().copied().unwrap_or(1).max(1) as usize;
        let path = |n: usize| (n >= 64, n >= 64 && GatherQmm::groups(n, experts));
        path(pairs) == path(b * pairs)
    }

    /// `x` (`[..., 1, in]`) through expert `indices` (broadcast with `x`'s batch); rows
    /// sorted by expert if `sorted`.
    #[track_caller]
    fn forward(&self, x: &Array, indices: &Array, sorted: bool) -> Array {
        let (y, bias) = match self {
            Self::Dense { weight, bias } => (
                x.gather_mm(&weight.transpose([0, 2, 1]), None, Some(indices), sorted),
                bias,
            ),
            Self::Quantized {
                weight,
                scales,
                biases,
                global_scale,
                quant,
                bias,
            } => {
                let mut options = GatherQmm::default().sorted_indices(sorted);
                if let Some(g) = global_scale {
                    options = options.global_scale(g);
                }
                let q =
                    nuvec::Quantized::new(weight.clone(), scales.clone(), biases.clone(), *quant);
                (x.gather_qmm(&q, None, Some(indices), options), bias)
            }
        };
        // mlx-lm's `SwitchLinear`: each row's expert's bias (`bias[indices]`, `[.., 1, out]`).
        match bias {
            Some(b) => y.add(b.value().take(indices, 0).expand_dims(-2)),
            None => y,
        }
    }
}

/// The experts' SwiGLU MLPs (mlx-lm's `SwitchGLU`).
#[derive(Debug, Clone, Module)]
struct SwitchGlu {
    gate_proj: SwitchProjection,
    up_proj: SwitchProjection,
    down_proj: SwitchProjection,
    #[tree(skip)]
    activation: Activation,
}

impl SwitchGlu {
    /// Rows `x` through the experts `idx`.
    #[track_caller]
    fn through(&self, x: &Array, idx: &Array, sorted: bool) -> Array {
        let up = self.up_proj.forward(x, idx, sorted);
        let gate = self.gate_proj.forward(x, idx, sorted);
        let h = match self.activation {
            Activation::SwiGluOai { alpha, limit } => {
                nuvec_nn::swiglu_oai(&gate, &up, alpha, limit)
            }
            Activation::GeGlu => nuvec_nn::geglu(&gate, &up),
            _ => swiglu(&gate, &up),
        };
        self.down_proj.forward(&h, idx, sorted)
    }

    /// `x` (`[B, L, D]`) through each token's experts `indices` (`[B, L, k]`):
    /// `[B, L, k, D]`. With many rows, they are sorted by expert first, so that each
    /// expert's weights are read once (mlx-lm's `_gather_sort`).
    #[track_caller]
    fn forward(&self, x: &Array, indices: &Array) -> Array {
        let [b, l, d] = x.shape_n::<3>().unwrap_or_default();
        // Several prompts prefilled together by a batch-invariant `Batcher`: where an item
        // alone would take other kernels (sorting and the grouped kernels depend on the
        // number of rows), each item goes through the experts alone, so it keeps its bits.
        let k = indices.shape().get(2).copied().unwrap_or_default();
        if b > 1
            && l > 1
            && nuvec::row_exact()
            && !self
                .gate_proj
                .batches_like_items(b as usize, (l * k) as usize)
        {
            let items: Vec<Array> = (0..b)
                .map(|i| self.forward(&x.i(i..i + 1), &indices.i(i..i + 1)))
                .collect();
            return ops::concatenate(&items, 0);
        }
        if indices.size() < 64 {
            let x = x.expand_dims([-3, -2]); // [B, L, 1, 1, D]
            return self.through(&x, indices, false).squeeze(-2);
        }
        let flat = indices.reshape([-1]);
        let order = flat.argsort(0);
        let restore = order.argsort(0);
        // Each pair's token: pair `t·k + j` belongs to token `t`.
        let tokens = ops::arange(0, b * l, 1).astype(DType::U32);
        let tokens = tokens
            .reshape([-1, 1])
            .broadcast_to([b * l, k])
            .reshape([-1]);
        let rows = x.reshape([-1, 1, d]).take(&tokens.take(&order, 0), 0);
        let out = self.through(&rows, &flat.take(&order, 0), true);
        out.take(&restore, 0).reshape([b, l, k, d])
    }
}

/// A mixture-of-experts MLP (mlx-lm's `Qwen3NextSparseMoeBlock`): a softmax router picks
/// each token's best experts, whose outputs are summed with the (renormalised) router
/// weights, plus a shared expert gated by a sigmoid.
#[derive(Debug, Clone, Module)]
struct MoeMlp {
    gate: Projection,
    switch_mlp: SwitchGlu,
    /// Qwen 3.5's gated shared expert and its gate.
    shared_expert: Option<Mlp>,
    shared_expert_gate: Option<Projection>,
    #[tree(skip)]
    experts_per_token: i32,
    #[tree(skip)]
    norm_topk_prob: bool,
}

impl MoeMlp {
    fn new(b: &Builder<'_>, path: &str, d: usize, moe: &MixtureOfExperts) -> Result<Self> {
        let p = |name: &str| format!("{path}.{name}");
        let e = moe.experts;
        Ok(Self {
            gate: b.projection(&p("gate"), d, e, false)?,
            switch_mlp: SwitchGlu::new(b, &p("switch_mlp"), d, moe)?,
            shared_expert: moe
                .shared_intermediate_size
                .map(|size| Mlp::new(b, &p("shared_expert"), d, size, false))
                .transpose()?,
            shared_expert_gate: moe
                .shared_intermediate_size
                .map(|_| b.projection(&p("shared_expert_gate"), d, 1, false))
                .transpose()?,
            experts_per_token: int(path, moe.experts_per_token)?,
            norm_topk_prob: moe.norm_topk_prob,
        })
    }

    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        let k = self.experts_per_token;
        // As mlx-lm's MoE blocks (`Qwen3MoeSparseMoeBlock`, `Qwen3NextSparseMoeBlock`): the
        // router's probabilities (a precise softmax, in the activations' type), the best `k`
        // by probability, renormalised over them if configured, and the experts' outputs
        // weighted and summed in the activations' type (`moe_route` and `weighted_sum`: one
        // kernel each, the same bits).
        let gates = self.gate.forward(x).softmax(-1);
        let (top, scores) = fast::moe_route(&gates, k, self.norm_topk_prob);
        #[cfg(feature = "diagnostics")]
        ROUTES.with(|r| {
            if let Some(routes) = r.borrow_mut().as_mut() {
                routes.push(top.clone());
            }
        });
        let experts = self.switch_mlp.forward(x, &top);
        let y = fast::weighted_sum(&experts, &scores);
        let (Some(expert), Some(gate)) = (&self.shared_expert, &self.shared_expert_gate) else {
            return y;
        };
        y.add(gate.forward(x).sigmoid().multiply(expert.forward(x)))
    }
}

impl SwitchGlu {
    fn new(b: &Builder<'_>, path: &str, d: usize, moe: &MixtureOfExperts) -> Result<Self> {
        let p = |name: &str| format!("{path}.{name}");
        let (e, i, bias) = (moe.experts, moe.intermediate_size, moe.expert_bias);
        Ok(Self {
            gate_proj: b.switch(&p("gate_proj"), d, i, e, bias)?,
            up_proj: b.switch(&p("up_proj"), d, i, e, bias)?,
            down_proj: b.switch(&p("down_proj"), i, d, e, bias)?,
            activation: moe.activation,
        })
    }
}

/// gpt-oss's mixture of experts (mlx-lm's `MLPBlock`): a biased router, the softmax of
/// the best `k` logits (precise), biased experts with the clamped SwiGLU, and their outputs
/// weighted and summed in the activations' dtype, as mlx-lm.
#[derive(Debug, Clone, Module)]
struct GptOssMlp {
    router: Projection,
    experts: SwitchGlu,
    #[tree(skip)]
    experts_per_token: i32,
}

impl GptOssMlp {
    fn new(b: &Builder<'_>, path: &str, d: usize, moe: &MixtureOfExperts) -> Result<Self> {
        Ok(Self {
            router: b.projection(&format!("{path}.router"), d, moe.experts, moe.router_bias)?,
            experts: SwitchGlu::new(b, &format!("{path}.experts"), d, moe)?,
            experts_per_token: int(path, moe.experts_per_token)?,
        })
    }

    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        let k = self.experts_per_token;
        let logits = self.router.forward(x);
        let (top, best) = fast::moe_route(&logits, k, false);
        let weights = best.softmax(-1);
        let y = self.experts.forward(x, &top);
        fast::weighted_sum(&y, &weights)
    }
}

/// A layer's MLP.
#[derive(Debug, Clone, Module)]
enum BlockMlp {
    Dense(Box<Mlp>),
    Moe(Box<MoeMlp>),
    GptOss(Box<GptOssMlp>),
}

impl BlockMlp {
    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::Moe(m) => m.forward(x),
            Self::GptOss(m) => m.forward(x),
        }
    }
}

#[derive(Debug, Clone, Module)]
struct Block {
    /// Transparent: the mixer names itself (`self_attn`, `linear_attn`).
    #[tree(rename = "")]
    mixer: Mixer,
    mlp: BlockMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    /// The group the block is split across ([`Model::shard`]): its attention and MLP
    /// outputs are summed over it.
    #[tree(skip, static_unchecked)]
    group: Option<nuvec::distributed::Group>,
    /// Gemma 4's further norms, per-layer input and experts (transparent: named as
    /// mlx-lm's decoder layer).
    #[tree(rename = "")]
    gemma: Option<Box<gemma::Parts>>,
}

/// `x + r` and its normalisation, in one kernel ([`fast::add_rms_norm`]); `x` alone
/// without `r`.
#[track_caller]
fn add_norm(x: &Array, r: Option<&Array>, norm: &RmsNorm) -> (Array, Array) {
    match r {
        Some(r) => fast::add_rms_norm(x, r, norm.weight.value(), norm.eps),
        None => (x.clone(), norm.forward(x)),
    }
}

impl Block {
    /// The block on `x + pending` (the previous block's MLP output, not yet added):
    /// gives the residual stream after attention and this block's MLP output. Each
    /// residual addition is fused with the normalisation that follows it, so a layer has
    /// no dependent dispatch for the additions.
    #[track_caller]
    fn forward(
        &self,
        x: &Array,
        pending: Option<&Array>,
        mask: AttentionMask<'_>,
        cache: &mut KvCache,
    ) -> (Array, Array) {
        let (x, n) = add_norm(x, pending, &self.input_layernorm);
        let mixed = match &self.mixer {
            Mixer::Attention(a) => a.forward(&n, mask, cache),
            Mixer::Linear(l) => l.forward(&n, cache),
        };
        let (h, n) = add_norm(
            &x,
            Some(&self.reduce(mixed)),
            &self.post_attention_layernorm,
        );
        (h, self.reduce(self.mlp.forward(&n)))
    }

    /// A sharded block's partial output summed over its group.
    #[track_caller]
    fn reduce(&self, y: Array) -> Array {
        match &self.group {
            Some(g) => nuvec::distributed::all_sum(&y, Some(g)),
            None => y,
        }
    }

    /// [`Block::forward`] for one new position per sequence over a paged cache.
    #[track_caller]
    fn forward_paged(
        &self,
        x: &Array,
        pending: Option<&Array>,
        layer: &mut PagedLayer,
        step: &PagedStep,
    ) -> (Array, Array) {
        let (x, n) = add_norm(x, pending, &self.input_layernorm);
        let mixed = match &self.mixer {
            Mixer::Attention(a) => a.forward_paged(&n, layer, step),
            Mixer::Linear(l) => l.forward_paged(&n, layer, step),
        };
        let (h, n) = add_norm(
            &x,
            Some(&self.reduce(mixed)),
            &self.post_attention_layernorm,
        );
        (h, self.reduce(self.mlp.forward(&n)))
    }
}

#[derive(Debug, Clone, Module)]
struct Body {
    embed_tokens: TokenEmbedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    /// Gemma 4's per-layer inputs (transparent: named as mlx-lm's text model).
    #[tree(rename = "")]
    per_layer: Option<Box<gemma::PerLayer>>,
}

/// A causal language model of the Llama family. A [`Module`] with Hugging Face's
/// parameter names (`model.layers.0.self_attn.q_proj.weight`).
#[derive(Debug, Clone, Module)]
pub struct Model {
    /// Training or evaluation mode (MLX's `Module.training`): the mode new layers (LoRA
    /// adapters) take; [`crate::load_model`] leaves models in evaluation mode.
    #[tree(mode)]
    training: bool,
    /// The configuration it was built from (not hashed into the structure).
    #[tree(skip, static_unchecked)]
    pub config: Config,
    model: Body,
    lm_head: Option<Projection>,
    /// The multi-token-prediction head (Qwen 3.5–3.8), when its weights are loaded.
    mtp: Option<mtp::MtpHead>,
    /// This rank's stage of a pipeline ([`Model::pipeline`]); `None` runs every layer.
    #[tree(skip, static_unchecked)]
    stage: Option<crate::pipeline::Stage>,
    /// The group a sharded model is split across ([`Model::shard`]).
    #[tree(skip, static_unchecked)]
    sharding: Option<nuvec::distributed::Group>,
    /// Whether the output head keeps this rank's vocabulary rows (its logits are gathered).
    #[tree(skip)]
    vocab_sharded: bool,
}

impl Model {
    /// The model's structure with placeholder parameters (no memory), to be filled by
    /// [`Module::load_weights`] (until then, a forward pass gives a failed array).
    /// `quantized(path)` gives each quantized layer's settings (`None`, or an entry without
    /// settings, for full precision).
    pub fn new(config: &Config, quantized: &dyn Fn(&str) -> Option<LayerQuant>) -> Result<Self> {
        let b = Builder { quantized };
        let c = config;
        if let Some(g) = &c.gemma4 {
            return gemma::model(&b, c, g);
        }
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let path = |s: &str| format!("model.layers.{i}.{s}");
                let mixer = match (c.layer_types[i], c.linear_attention) {
                    (LayerType::Linear, Some(sizes)) => {
                        Mixer::Linear(GatedDeltaNet::new(&b, &path("linear_attn"), c, sizes)?)
                    }
                    _ => Mixer::Attention(Attention::new(
                        &b,
                        &path("self_attn"),
                        c,
                        c.layer_windows.get(i).copied().flatten(),
                    )?),
                };
                Ok(Block {
                    mixer,
                    mlp: match c.moe.as_ref().filter(|m| m.is_sparse(i)) {
                        Some(moe) if c.model_type == "gpt_oss" => BlockMlp::GptOss(Box::new(
                            GptOssMlp::new(&b, &path("mlp"), c.hidden_size, moe)?,
                        )),
                        Some(moe) => BlockMlp::Moe(Box::new(MoeMlp::new(
                            &b,
                            &path("mlp"),
                            c.hidden_size,
                            moe,
                        )?)),
                        None => BlockMlp::Dense(Box::new(Mlp::new(
                            &b,
                            &path("mlp"),
                            c.hidden_size,
                            c.intermediate_size,
                            c.mlp_bias,
                        )?)),
                    },
                    input_layernorm: Builder::norm(c.hidden_size, c.rms_norm_eps),
                    post_attention_layernorm: Builder::norm(c.hidden_size, c.rms_norm_eps),
                    group: None,
                    gemma: None,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            training: true,
            config: config.clone(),
            model: Body {
                embed_tokens: b.embedding("model.embed_tokens", c.vocab_size, c.hidden_size)?,
                layers,
                norm: Builder::norm(c.hidden_size, c.rms_norm_eps),
                per_layer: None,
            },
            lm_head: (!c.tie_word_embeddings)
                .then(|| b.projection("lm_head", c.hidden_size, c.vocab_size, false))
                .transpose()?,
            mtp: None,
            stage: None,
            sharding: None,
            vocab_sharded: false,
        })
    }

    /// Runs one token through the model (and discards the result), so that the first
    /// prompt does not pay for first use: on the GPU, memory-mapped weights are made
    /// resident when a command buffer first reads them, and kernels are compiled when
    /// first dispatched. [`load_model`](crate::load_model) does not call it (as mlx-lm's
    /// `load`, it only loads the weights): a server can, before its first request. Its
    /// KV cache and activations add to the peak memory.
    pub fn warm_up(&self) -> Result<()> {
        let token = Array::from_slice(&[0u32], [1, 1]);
        self.forward(&token, &mut self.new_cache()).eval()
    }

    /// Per layer, what a paged cache keeps.
    pub(crate) fn paged_layers(&self) -> Vec<PagedKind> {
        let range = self.stage.as_ref().map(|s| s.layers());
        self.model
            .layers
            .iter()
            .enumerate()
            .map(|(i, b)| match &b.mixer {
                // Another stage's layer: nothing of its own here (as a KV-shared layer).
                _ if range.as_ref().is_some_and(|r| !r.contains(&i)) => PagedKind::Shared,
                Mixer::Attention(a) if a.k_proj.is_some() => {
                    PagedKind::Keys(a.window.map(|w| usize::try_from(w).unwrap_or(0)))
                }
                Mixer::Attention(_) => PagedKind::Shared,
                Mixer::Linear(_) => PagedKind::States,
            })
            .collect()
    }

    /// The bytes one position of a sequence takes in a [`crate::PagedCache`]: the keys
    /// and values of this stage's full-attention layers with their own keys, in the
    /// activations' dtype. Sliding-window layers hold a fixed window per sequence instead.
    pub fn kv_bytes_per_position(&self) -> usize {
        let dtype = match self.stage() {
            Some(stage) => stage.activations(),
            None => self.embed(&Array::from_slice(&[0u32], [1, 1])).dtype(),
        };
        self.paged_layers()
            .iter()
            .zip(&self.model.layers)
            .map(|(kind, b)| match (kind, &b.mixer) {
                (PagedKind::Keys(None), Mixer::Attention(a)) => {
                    2 * a.k_proj().outputs() * dtype.size()
                }
                _ => 0,
            })
            .sum()
    }

    /// An empty cache per layer.
    pub fn new_cache(&self) -> Vec<KvCache> {
        self.model
            .layers
            .iter()
            .map(|b| match &b.mixer {
                Mixer::Attention(a) => a.window.map_or_else(KvCache::default, KvCache::windowed),
                Mixer::Linear(_) => KvCache::default(),
            })
            .collect()
    }

    /// The logits `[B, L, vocab]` for tokens `[B, L]` with no cache, as mlx-lm's
    /// `model(inputs)`: for training and scoring, the keys and values are not kept.
    #[track_caller]
    pub fn forward_uncached(&self, tokens: &Array) -> Array {
        let mut cache = self.new_cache();
        for c in &mut cache {
            c.uncached = true;
        }
        self.forward(tokens, &mut cache)
    }

    /// The logits `[B, L, vocab]` for tokens `[B, L]` (integers) following the cached
    /// positions, which are extended. `cache` has one entry per layer
    /// ([`Model::new_cache`]); another length gives a failed array (and leaves the cache
    /// alone).
    ///
    /// Nothing is evaluated: errors surface at [`Array::check`] or [`Array::eval`].
    #[track_caller]
    pub fn forward(&self, tokens: &Array, cache: &mut [KvCache]) -> Array {
        if cache.len() != self.model.layers.len() {
            return mismatched_layers(cache.len(), self.model.layers.len());
        }
        if self.config.gemma4.is_some() {
            return gemma::forward(self, tokens, cache);
        }
        self.forward_hidden(tokens, cache).0
    }

    /// [`Model::forward`], also giving the final hidden states `[B, L, hidden_size]` (after
    /// the final norm: what the output head reads, and what the MTP head takes). Not for
    /// Gemma 4 (a failed array).
    #[track_caller]
    pub fn forward_hidden(&self, tokens: &Array, cache: &mut [KvCache]) -> (Array, Array) {
        if cache.len() != self.model.layers.len() {
            let e = mismatched_layers(cache.len(), self.model.layers.len());
            return (e.clone(), e);
        }
        if self.config.gemma4.is_some() {
            let e = Array::failure(
                "forward_hidden",
                OpErrorKind::InvalidArgument("not available for Gemma 4".into()),
            );
            return (e.clone(), e);
        }
        // Causal (aligned to the last positions) for several tokens; one token sees all.
        let mask = match tokens.shape().get(1) {
            Some(&l) if l > 1 => AttentionMask::Causal,
            _ => AttentionMask::None,
        };
        if let Some(stage) = &self.stage {
            return self.forward_stage(stage, tokens, mask, cache);
        }
        let mut h = self.model.embed_tokens.forward(tokens);
        let mut pending = None;
        for (layer, c) in self.model.layers.iter().zip(cache) {
            let (next, mlp) = layer.forward(&h, pending.as_ref(), mask, c);
            (h, pending) = (next, Some(mlp));
        }
        let (_, h) = add_norm(&h, pending.as_ref(), &self.model.norm);
        (self.head(&h), h)
    }

    /// One stage's part of [`Model::forward_hidden`]: the first stage embeds, the others
    /// receive the residual stream and the last layer's MLP output not yet added
    /// (`[B, L, 2·hidden]`, so the next layer fuses the addition as the whole model
    /// does); the local layers run; the stage then sends both on, or (rank 0) runs the norm
    /// and the head. A stage that sends returns zero logits of the head's shape, evaluated
    /// after its send; the send is attached to its last layer's cache, so prefill (which
    /// evaluates the caches) sends too.
    #[track_caller]
    fn forward_stage(
        &self,
        stage: &crate::pipeline::Stage,
        tokens: &Array,
        mask: AttentionMask<'_>,
        cache: &mut [KvCache],
    ) -> (Array, Array) {
        let (mut h, mut pending) = self.stage_input(stage, tokens);
        let layers = stage.layers();
        for (layer, c) in self.model.layers[layers.clone()]
            .iter()
            .zip(&mut cache[layers.clone()])
        {
            let (next, mlp) = layer.forward(&h, pending.as_ref(), mask, c);
            (h, pending) = (next, Some(mlp));
        }
        if stage.is_last() {
            let (_, h) = add_norm(&h, pending.as_ref(), &self.model.norm);
            return (self.head(&h), h);
        }
        let (logits, h, sent) = self.stage_output(stage, tokens, &h, pending);
        if let Some(c) = cache.get_mut(layers.end - 1) {
            c.attach(&sent);
        }
        (logits, h)
    }

    /// A stage's input: the first stage's embeddings, or the residual stream and the
    /// previous layer's MLP output received from the stage before (`[B, L, 2·hidden]`).
    #[track_caller]
    fn stage_input(
        &self,
        stage: &crate::pipeline::Stage,
        tokens: &Array,
    ) -> (Array, Option<Array>) {
        if stage.is_first() {
            return (self.model.embed_tokens.forward(tokens), None);
        }
        let shape = tokens.shape();
        let hidden = self.config.hidden_size as i32;
        let wire = [shape[0], shape.get(1).copied().unwrap_or(1), 2 * hidden];
        let got = nuvec::distributed::recv(
            wire,
            stage.activations(),
            stage.previous(),
            Some(stage.group()),
        );
        let parts = got.split(2, -1);
        (parts[0].clone(), Some(parts[1].clone()))
    }

    /// A non-last stage's output: sends the residual stream and the last layer's MLP output
    /// on, and gives zero logits of the head's shape and the residual stream, both
    /// evaluated after the send, and the send.
    #[track_caller]
    fn stage_output(
        &self,
        stage: &crate::pipeline::Stage,
        tokens: &Array,
        h: &Array,
        pending: Option<Array>,
    ) -> (Array, Array, Array) {
        let pending = pending.expect("a stage runs at least one layer");
        let wire = nuvec::ops::concatenate(&[h.clone(), pending], -1);
        let sent = nuvec::distributed::send(&wire, stage.next(), Some(stage.group()));
        let shape = tokens.shape();
        let logits = nuvec::ops::zeros(
            [
                shape[0],
                shape.get(1).copied().unwrap_or(1),
                self.config.vocab_size as i32,
            ],
            stage.activations(),
        );
        let after = |a: &Array| nuvec::ops::depends(a, std::slice::from_ref(&sent));
        (after(&logits), after(h), sent)
    }

    /// The token embeddings of `tokens` (not evaluated).
    pub(crate) fn embed(&self, tokens: &Array) -> Array {
        self.model.embed_tokens.forward(tokens)
    }

    pub(crate) fn set_stage(&mut self, stage: crate::pipeline::Stage) {
        self.stage = Some(stage);
    }

    pub(crate) fn stage_ref(&self) -> Option<&crate::pipeline::Stage> {
        self.stage.as_ref()
    }

    /// Replaces every weight outside `stage` by an unloaded placeholder of its shape and
    /// dtype (no memory).
    pub(crate) fn drop_outside(&mut self, stage: &crate::pipeline::Stage) -> Result<()> {
        let tied = self.config.tie_word_embeddings;
        let blank: std::collections::BTreeMap<String, Array> = self
            .parameters()
            .into_iter()
            .filter(|(name, _)| !crate::pipeline::in_stage(name, stage, tied))
            .map(|(name, a)| {
                let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
                (name, placeholder(&shape, a.dtype()))
            })
            .collect();
        nuvec_nn::Module::load_weights_partial(self, blank)
    }

    /// The logits of final hidden states `[..., hidden_size]` ([`Model::forward_hidden`],
    /// [`Model::mtp_forward`]): the output head alone.
    #[track_caller]
    pub fn logits(&self, hidden: &Array) -> Array {
        self.head(hidden)
    }

    /// The output head on the final hidden states.
    #[track_caller]
    fn head(&self, h: &Array) -> Array {
        let logits = match &self.lm_head {
            Some(head) => head.forward(h),
            None => self.model.embed_tokens.as_linear(h),
        };
        match (&self.sharding, self.vocab_sharded) {
            // Every rank's vocabulary rows, joined along the last axis (the gather joins
            // along the first).
            (Some(g), true) => {
                let d = logits.ndim();
                let front: Vec<usize> = std::iter::once(d - 1).chain(0..d - 1).collect();
                let back: Vec<usize> = (1..d).chain(std::iter::once(0)).collect();
                let gathered = nuvec::distributed::all_gather(&logits.transpose(front), Some(g));
                gathered.transpose(back)
            }
            _ => logits,
        }
    }
}

impl Model {
    /// The logits `[B, 1, vocab]` for one new token per sequence (`tokens` `[B, 1]`) over
    /// a paged cache (one [`PagedLayer`] per layer), checked (not evaluated).
    ///
    /// The layers' pools are taken out while the step is built (so that the writes are
    /// done in place); when the step fails, they are put back as they were, so a failed
    /// step leaves the cache unchanged.
    pub(crate) fn forward_paged(
        &self,
        tokens: &Array,
        layers: &mut [PagedLayer],
        step: &PagedStep,
    ) -> Result<Array> {
        check_layers(layers.len(), self.model.layers.len())?;
        // Handles to the pools (not copies): dropped before anything is evaluated, so
        // the in-place writes still own their buffers.
        let saved = layers.to_vec();
        let mut logits = self.forward_paged_unguarded(tokens, layers, step).check();
        #[cfg(test)]
        if let Some(i) = tests::FAIL_PAGED_AT.with(std::cell::Cell::get)
            && i < layers.len()
        {
            // An injected failure, after every layer replaced its pools.
            logits = Err(nuvec::Error::InvalidArgument(format!(
                "injected failure at layer {i}"
            )));
        }
        if logits.is_ok() {
            // A pool write that failed without failing the logits fails the step too.
            for layer in layers.iter() {
                for pool in [&layer.keys, &layer.values, &layer.conv, &layer.recurrent] {
                    if let Some(e) = pool.as_ref().and_then(Array::error) {
                        logits = Err(e.clone());
                    }
                }
            }
        }
        if logits.is_err() {
            layers.clone_from_slice(&saved);
        }
        logits
    }

    #[track_caller]
    fn forward_paged_unguarded(
        &self,
        tokens: &Array,
        layers: &mut [PagedLayer],
        step: &PagedStep,
    ) -> Array {
        if self.config.gemma4.is_some() {
            return gemma::forward_paged(self, tokens, layers, step);
        }
        let (mut h, mut pending, range) = match &self.stage {
            Some(stage) => {
                let (h, pending) = self.stage_input(stage, tokens);
                (h, pending, stage.layers())
            }
            None => (
                self.model.embed_tokens.forward(tokens),
                None,
                0..layers.len(),
            ),
        };
        for (layer, pages) in self.model.layers[range.clone()]
            .iter()
            .zip(&mut layers[range])
        {
            let (next, mlp) = layer.forward_paged(&h, pending.as_ref(), pages, step);
            (h, pending) = (next, Some(mlp));
        }
        match &self.stage {
            Some(stage) if !stage.is_last() => self.stage_output(stage, tokens, &h, pending).0,
            _ => {
                let (_, h) = add_norm(&h, pending.as_ref(), &self.model.norm);
                self.head(&h)
            }
        }
    }
}

/// A cache of `got` layers for a model of `layers` (a shorter one would skip layers).
fn check_layers(got: usize, layers: usize) -> Result<()> {
    if got != layers {
        return Err(nuvec::Error::InvalidArgument(format!(
            "a cache of {got} layers for a model of {layers}"
        )));
    }
    Ok(())
}

/// The failed array of a forward pass given a cache of `got` layers for a model of
/// `layers`, located at the caller of `forward`.
#[track_caller]
fn mismatched_layers(got: usize, layers: usize) -> Array {
    Array::failure(
        "forward",
        nuvec::OpErrorKind::InvalidArgument(format!(
            "a cache of {got} layers for a model of {layers}"
        )),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use super::*;

    thread_local! {
        /// A fault hook: `forward_paged` fails at this layer (on this thread).
        pub(crate) static FAIL_PAGED_AT: Cell<Option<usize>> = const { Cell::new(None) };
    }

    /// Sets [`FAIL_PAGED_AT`] until the guard is dropped.
    pub(crate) struct Fault;

    impl Drop for Fault {
        fn drop(&mut self) {
            FAIL_PAGED_AT.with(|f| f.set(None));
        }
    }

    /// `forward_paged` fails at `layer` until the guard is dropped.
    pub(crate) fn fail_paged_at(layer: usize) -> Fault {
        FAIL_PAGED_AT.with(|f| f.set(Some(layer)));
        Fault
    }

    /// Growing a cache keeps `old`'s kept positions, appends `new` and pads with zeros, in
    /// every dtype, on the CPU and the GPU (the padding is a broadcast zero, written by the
    /// concatenation).
    #[test]
    fn grow_keeps_appends_and_pads_with_zeros() {
        let devices = [
            Some(nuvec::Device::cpu()),
            nuvec::metal::is_available().then(|| nuvec::Device::gpu(0)),
        ];
        for device in devices.into_iter().flatten() {
            nuvec::with_stream(nuvec::Stream::default_for(device), || {
                for dtype in [DType::F32, DType::F16, DType::BF16] {
                    let old = ops::arange(0.0, 24.0, 1.0)
                        .reshape([1, 2, 6, 2])
                        .astype(dtype);
                    let new = ops::arange(100.0, 104.0, 1.0)
                        .reshape([1, 2, 1, 2])
                        .astype(dtype);
                    let grown = grow(Some(old), (1, 4), &new, 3);
                    assert_eq!(grown.shape(), [1, 2, 7, 2]);
                    assert_eq!(grown.dtype(), dtype);
                    let got = grown.astype(DType::F32).to_vec::<f32>().unwrap();
                    let head = |h: f32| {
                        let base = h * 12.0;
                        let mut row: Vec<f32> = (2..8).map(|i| base + i as f32).collect();
                        row.extend([100.0 + 2.0 * h, 101.0 + 2.0 * h]);
                        row.extend([0.0; 6]);
                        row
                    };
                    let want: Vec<f32> = [head(0.0), head(1.0)].concat();
                    assert_eq!(got, want, "{dtype:?} on {device:?}");
                    // No kept positions and no padding: `new` alone.
                    let alone = grow(None, (0, 0), &new, 0);
                    assert_eq!(alone.shape(), new.shape());
                }
            });
        }
    }

    /// A model of the `qwen3_f32` fixture, or `None` without a Metal device.
    /// A position's bytes are the full-attention pools' bytes per position once a
    /// sequence brought them (windows, K = V and Gated DeltaNet layers included).
    #[test]
    fn kv_bytes_per_position_match_the_pools() {
        for name in [
            "qwen3_f32",
            "llama_bf16",
            "qwen3_5_q4",
            "gemma4_e_bf16",
            "gpt_oss_q4",
        ] {
            let Some(model) = fixture_model(name) else {
                return;
            };
            let mut kv = model.new_cache();
            let _ = model.forward(&Array::from_slice(&[1u32, 2, 3], [1, 3]), &mut kv);
            let mut cache = crate::PagedCache::new(&model, 8, 16, 1).unwrap();
            cache.insert(&kv).unwrap();
            let pools: usize = model
                .paged_layers()
                .iter()
                .zip(&cache.layers)
                .filter(|(k, _)| **k == PagedKind::Keys(None))
                .map(|(_, l)| {
                    [&l.keys, &l.values]
                        .iter()
                        .map(|a| a.as_ref().map_or(0, |a| a.nbytes()))
                        .sum::<usize>()
                })
                .sum();
            assert_eq!(pools, 8 * 16 * model.kv_bytes_per_position(), "{name}");
            assert!(model.kv_bytes_per_position() > 0, "{name}");
        }
    }

    pub(crate) fn fixture_model(name: &str) -> Option<Model> {
        nuvec::set_default_stream(nuvec::Stream::default_for(nuvec::Device::gpu(0)));
        if (&Array::from(1.0f32) + 1.0).eval().is_err() {
            eprintln!("skipped: no Metal device");
            return None;
        }
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        Some(crate::load_model(dir).unwrap())
    }

    /// `forward_paged` rejects a layer slice of another length, and a failed step leaves
    /// the pools as they were.
    #[test]
    fn forward_paged_checks_layers_and_restores_pools() {
        let Some(model) = fixture_model("qwen3_f32") else {
            return;
        };
        let mut cache = crate::PagedCache::new(&model, 8, 4, 2).unwrap();
        let mut kv = model.new_cache();
        let prompt = Array::from_slice(&[1u32, 2, 3], [1, 3]);
        let _ = model.forward(&prompt, &mut kv).check().unwrap();
        let id = cache.insert(&kv).unwrap();
        let reservation = cache.reserve(&[id]).unwrap();
        let tokens = Array::from_slice(&[4u32], [1, 1]);
        let layers = cache.layers.len();
        assert!(
            model
                .forward_paged(&tokens, &mut cache.layers[..layers - 1], &reservation.step)
                .is_err()
        );
        let before: Vec<Vec<f32>> = cache
            .state()
            .iter()
            .map(|a| a.to_vec::<f32>().unwrap())
            .collect();
        FAIL_PAGED_AT.with(|f| f.set(Some(1)));
        let failed = model.forward_paged(&tokens, &mut cache.layers, &reservation.step);
        FAIL_PAGED_AT.with(|f| f.set(None));
        assert!(failed.is_err());
        let after: Vec<Vec<f32>> = cache
            .state()
            .iter()
            .map(|a| a.to_vec::<f32>().unwrap())
            .collect();
        assert_eq!(before, after, "pools restored");
        cache.rollback(reservation);
        assert_eq!(cache.free_blocks(), 7);
    }
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// The experts each mixture-of-experts layer chose, while [`record_routes`] runs.
    static ROUTES: std::cell::RefCell<Option<Vec<Array>>> = const { std::cell::RefCell::new(None) };
}

/// Runs `f`, recording on this thread the experts every mixture-of-experts layer chose
/// (`[B, L, k]` indices per layer call, in order), for routing studies
/// (`examples/route_overlap.rs`). Only with the `diagnostics` feature.
#[cfg(feature = "diagnostics")]
pub fn record_routes<R>(f: impl FnOnce() -> R) -> (R, Vec<Array>) {
    ROUTES.with(|r| *r.borrow_mut() = Some(Vec::new()));
    let out = f();
    let routes = ROUTES.with(|r| r.borrow_mut().take()).unwrap_or_default();
    (out, routes)
}
