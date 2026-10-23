//! The Llama-family transformer, as mlx-lm's `llama`, `qwen2` and `qwen3` models, and its
//! hybrid with Gated DeltaNet layers, as mlx-lm's `qwen3_5`, with parameter names matching
//! Hugging Face checkpoints.
//!
//! The forward passes return arrays (a failed array carries the first error, see
//! [`Array::check`]); construction and the paged cache's bookkeeping return `Result`.

use std::sync::LazyLock;

use nuvec::fast::{self, AttentionMask};
use nuvec::{Array, DType, OpErrorKind, Param, Result, ops};
use nuvec_nn::{
    Embedding, Linear, LoraLinear, Module, QuantizedEmbedding, QuantizedLinear, RmsNorm, silu,
};

use crate::config::{Config, LayerType, LinearAttention, MixtureOfExperts, RopeScaling};

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

/// `silu(gate) · x`, uncompiled.
#[track_caller]
fn swiglu_graph(gate: &Array, x: &Array) -> Array {
    silu(gate).multiply(x)
}

/// `silu(gate) · x` in one kernel, as mlx-lm's compiled `swiglu`.
#[track_caller]
fn swiglu(gate: &Array, x: &Array) -> Array {
    static SWIGLU: LazyLock<Compiled2> =
        LazyLock::new(|| Box::new(nuvec::compile(|(g, x): &(Array, Array)| swiglu_graph(g, x))));
    // A failed replay (not expected for valid inputs) falls back to the graph, which
    // carries the error.
    SWIGLU(&(gate.clone(), x.clone())).unwrap_or_else(|_| swiglu_graph(gate, x))
}

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

/// Builds layers with placeholder parameters; `quantized(path)` gives the group size and
/// bits of quantized layers.
struct Builder<'a> {
    quantized: &'a dyn Fn(&str) -> Option<(usize, u32)>,
}

impl Builder<'_> {
    fn projection(
        &self,
        path: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Projection> {
        let bias = bias.then(|| param(&[output], DType::F32));
        Ok(match (self.quantized)(path) {
            Some((group_size, bits)) => Projection::Quantized(QuantizedLinear {
                weight: placeholder(&[output, input * bits as usize / 32], DType::U32),
                scales: placeholder(&[output, groups(path, input, group_size)?], DType::F32),
                biases: placeholder(&[output, groups(path, input, group_size)?], DType::F32),
                bias,
                group_size: int(path, group_size)?,
                bits: int(path, bits as usize)?,
            }),
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
    ) -> Result<SwitchProjection> {
        Ok(match (self.quantized)(path) {
            Some((group_size, bits)) => {
                let groups = groups(path, input, group_size)?;
                SwitchProjection::Quantized {
                    weight: placeholder(&[experts, output, input * bits as usize / 32], DType::U32),
                    scales: placeholder(&[experts, output, groups], DType::F32),
                    biases: placeholder(&[experts, output, groups], DType::F32),
                    group_size: int(path, group_size)?,
                    bits: int(path, bits as usize)?,
                }
            }
            None => SwitchProjection::Dense {
                weight: param(&[experts, output, input], DType::F32),
            },
        })
    }

    fn embedding(&self, path: &str, count: usize, dims: usize) -> Result<TokenEmbedding> {
        Ok(match (self.quantized)(path) {
            Some((group_size, bits)) => TokenEmbedding::Quantized(QuantizedEmbedding {
                weight: placeholder(&[count, dims * bits as usize / 32], DType::U32),
                scales: placeholder(&[count, groups(path, dims, group_size)?], DType::F32),
                biases: placeholder(&[count, groups(path, dims, group_size)?], DType::F32),
                group_size: int(path, group_size)?,
                bits: int(path, bits as usize)?,
            }),
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

/// The rotary embedding of every head's features (no parameters).
#[derive(Debug, Clone, Module)]
struct Rope {
    #[tree(skip)]
    dims: i32,
    #[tree(skip)]
    traditional: bool,
    #[tree(skip)]
    base: f32,
    #[tree(skip)]
    scale: f32,
    /// Per-pair frequencies (Llama 3 scaling) instead of `base`: computed from the
    /// configuration, not in checkpoints (mlx-lm's `_freqs`).
    #[tree(skip)]
    freqs: Option<Array>,
}

impl Rope {
    fn new(dims: usize, config: &Config) -> Result<Self> {
        let base = config.rope_theta;
        let mut rope = Self {
            dims: int("rope dims", dims)?,
            traditional: config.rope_traditional,
            base,
            scale: 1.0,
            freqs: None,
        };
        match config.rope_scaling {
            RopeScaling::None => {}
            RopeScaling::Linear { factor } => rope.scale = 1.0 / factor,
            RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings: old,
            } => {
                // mlx-lm's `Llama3RoPE`.
                let (low_wavelen, high_wavelen) = (old / low_freq_factor, old / high_freq_factor);
                let freqs: Vec<f32> = (0..dims)
                    .step_by(2)
                    .map(|i| {
                        let freq = base.powf(i as f32 / dims as f32);
                        let wavelen = 2.0 * std::f32::consts::PI * freq;
                        let scaled = if wavelen > low_wavelen {
                            freq * factor
                        } else {
                            freq
                        };
                        if wavelen > high_wavelen && wavelen < low_wavelen {
                            let smooth = (old / wavelen - low_freq_factor)
                                / (high_freq_factor - low_freq_factor);
                            scaled / ((1.0 - smooth) / factor + smooth)
                        } else {
                            scaled
                        }
                    })
                    .collect();
                rope.freqs = Some(Array::from(freqs));
            }
        }
        Ok(rope)
    }

    fn rotation(&self) -> fast::Rotation<'_> {
        fast::Rotation {
            dims: self.dims,
            traditional: self.traditional,
            base: self.freqs.is_none().then_some(self.base),
            scale: self.scale,
            freqs: self.freqs.as_ref(),
        }
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

    /// `x` (`[B, H, L, D]`) rotated at each sequence's position (`positions`, `[B]`).
    #[track_caller]
    fn rope_at(&self, x: &Array, positions: &Array) -> Array {
        let r = self.rotation();
        fast::rope(
            x,
            r.dims,
            r.traditional,
            r.base,
            r.scale,
            positions,
            r.freqs,
        )
    }
}

/// `[B, L, H, D]` as `[B, H, L, D]` and back (a transposition of axes 1 and 2).
const HEADS_FIRST: [i32; 4] = [0, 2, 1, 3];

/// One layer's cache. Attention layers keep keys and values, `[B, kv_heads, capacity,
/// head_dim]`, as mlx-lm's `KVCache`: the capacity grows in steps of 256 positions and
/// each step's keys and values are written into it (in place), so buffers keep their
/// size between growths. Gated DeltaNet layers keep the convolution's last inputs and the
/// recurrent state instead, as mlx-lm's `ArraysCache`.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
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
}

/// Positions added to a cache's capacity at a time.
const CACHE_STEP: i32 = 256;

impl KvCache {
    /// Positions cached.
    pub fn offset(&self) -> usize {
        usize::try_from(self.offset).unwrap_or_default()
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
        let prev = self.offset;
        let end = prev + values.shape().get(2).copied().unwrap_or_default();
        let fits = self.keys.as_ref().is_some_and(|k| end <= k.shape()[2]);
        let (k, v) = if fits {
            // Each write takes the cache's only handle, so it is done in place.
            let k = rope.norm_rope(keys, norm, prev, self.keys.take().as_ref());
            let all = self.values.take().expect("allocated with the keys");
            (k, all.at((.., .., prev..end)).set(values))
        } else {
            // Grown by whole steps (mlx-lm's rule), written in one concatenation: the cached
            // positions, the new ones, then zeros up to the step (no copy of the grown cache
            // to update it).
            let l = end - prev;
            let pad = (l + CACHE_STEP - 1) / CACHE_STEP * CACHE_STEP - l;
            let keys = rope.norm_rope(keys, norm, prev, None);
            let k = grow(self.keys.take(), prev, &keys, pad);
            (k, grow(self.values.take(), prev, values, pad))
        };
        self.offset = end;
        self.keys = Some(k.clone());
        self.values = Some(v.clone());
        (positions(&k, end), positions(&v, end))
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
        match i32::try_from(n).ok().filter(|&n| n <= self.offset) {
            Some(n) => {
                self.offset -= n;
                Ok(())
            }
            None => Err(invalid(format!(
                "cannot trim {n} of {} cached positions",
                self.offset
            ))),
        }
    }

    /// The cached keys and values of an attention layer, `[B, kv_heads, offset, head_dim]`
    /// (views), if any.
    pub(crate) fn keys_values(&self) -> Option<(Array, Array)> {
        match (&self.keys, &self.values) {
            (Some(k), Some(v)) => Some((positions(k, self.offset), positions(v, self.offset))),
            _ => None,
        }
    }

    /// A Gated DeltaNet layer's convolution and recurrent states (`[B, kernel − 1,
    /// channels]` and `[B, Hv, Dv, Dk]`), if any.
    pub(crate) fn linear_state(&self) -> Option<(Array, Array)> {
        Some((self.conv.clone()?, self.recurrent.clone()?))
    }

    /// Sequence `i` of a batched cache (views of its row, the same positions), for
    /// [`PagedCache::insert`](crate::PagedCache::insert). Kept steps are not carried.
    pub(crate) fn row(&self, i: i32) -> KvCache {
        let row = |a: &Option<Array>| a.as_ref().map(|a| a.i(i..i + 1));
        KvCache {
            keys: row(&self.keys),
            values: row(&self.values),
            conv: row(&self.conv),
            recurrent: row(&self.recurrent),
            offset: self.offset,
            keep_steps: false,
            steps: None,
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
}

/// The first `n` positions (axis 2) of a cache array, as a view.
#[track_caller]
fn positions(a: &Array, n: i32) -> Array {
    if a.shape().get(2) == Some(&n) {
        return a.clone();
    }
    a.i((.., .., ..n))
}

/// `old`'s first `prev` positions (axis 2), then `new`, then `pad` positions of zeros, in
/// one concatenation.
#[track_caller]
fn grow(old: Option<Array>, prev: i32, new: &Array, pad: i32) -> Array {
    let mut parts = Vec::with_capacity(3);
    if let Some(old) = old {
        parts.push(positions(&old, prev));
    }
    parts.push(new.clone());
    if pad > 0 {
        let mut shape = new.shape().to_vec();
        if let Some(d) = shape.get_mut(2) {
            *d = pad;
        }
        parts.push(ops::zeros(shape, new.dtype()));
    }
    ops::concatenate(&parts, 2)
}

#[derive(Debug, Clone, Module)]
struct Attention {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,
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
}

impl Attention {
    fn new(b: &Builder<'_>, path: &str, c: &Config) -> Result<Self> {
        let (d, hd) = (c.hidden_size, c.head_dim);
        let (q, kv) = (c.num_attention_heads * hd, c.num_key_value_heads * hd);
        let p = |name: &str| format!("{path}.{name}");
        let norm = || c.qk_norm.then(|| Builder::norm(hd, c.rms_norm_eps));
        let gate = c.attn_output_gate;
        Ok(Self {
            q_proj: b.projection(&p("q_proj"), d, q * (1 + usize::from(gate)), c.qkv_bias)?,
            k_proj: b.projection(&p("k_proj"), d, kv, c.qkv_bias)?,
            v_proj: b.projection(&p("v_proj"), d, kv, c.qkv_bias)?,
            o_proj: b.projection(&p("o_proj"), q, d, c.o_bias)?,
            q_norm: norm(),
            k_norm: norm(),
            heads: int(path, c.num_attention_heads)?,
            kv_heads: int(path, c.num_key_value_heads)?,
            scale: (hd as f32).powf(-0.5),
            rope: Rope::new(c.rope_dims, c)?,
            gate,
        })
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
        let [b, l, _] = x.shape_n::<3>().unwrap_or_default();
        let (q, gate) = self.query(x, [b, l]);
        // The query and key (written into the cache) normalised and rotated in one kernel.
        let q = q.transpose(HEADS_FIRST);
        let q = self
            .rope
            .norm_rope(&q, self.q_norm.as_ref(), cache.offset, None);
        let heads = [b, l, self.kv_heads, -1];
        let k = self.k_proj.forward(x).reshape(heads).transpose(HEADS_FIRST);
        let v = self.v_proj.forward(x).reshape(heads).transpose(HEADS_FIRST);
        let (k, v) = cache.update(&self.rope, self.k_norm.as_ref(), &k, &v);
        let o = fast::scaled_dot_product_attention(&q, &k, &v, self.scale, &mask.into());
        self.output(&o, gate, [b, l])
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
}

/// What one batched decoding step writes and reads, per sequence (`[B]`, int32): the slot
/// of its new position, its position (for RoPE), its length after the step, its block
/// table (`[B, max_blocks]`) and its Gated DeltaNet state slot.
#[derive(Debug, Clone)]
pub(crate) struct PagedStep {
    pub(crate) slots: Array,
    pub(crate) positions: Array,
    pub(crate) lengths: Array,
    pub(crate) tables: Array,
    pub(crate) state_slots: Array,
}

impl PagedLayer {
    /// Keys and values (`[B, kv_heads, head_dim]`) written at `slots` of their pools, and
    /// the pools.
    #[track_caller]
    fn write_kv(&mut self, k: &Array, v: &Array, slots: &Array) -> (Array, Array) {
        let keys = write_pool(&mut self.keys, self.pool_size, k, slots);
        (keys, write_pool(&mut self.values, self.pool_size, v, slots))
    }
}

/// `rows` (`[B, kv_heads, head_dim]`) written at `slots` of the pool in `pool` (of `size`,
/// blocks and positions per block), taken out first so that the write is done in place
/// (zeros until a prefilled sequence brought the pool); put back and returned.
#[track_caller]
fn write_pool(pool: &mut Option<Array>, size: (i32, i32), rows: &Array, slots: &Array) -> Array {
    let all = match pool.take() {
        Some(p) => p,
        None => {
            let [_, heads, dims] = rows.shape_n::<3>().unwrap_or_default();
            ops::zeros([size.0, heads, size.1, dims], rows.dtype())
        }
    };
    let written = fast::paged_write(&all, rows, slots);
    *pool = Some(written.clone());
    written
}

impl Attention {
    /// `h` (`[B, 1, H, D]`) normalised per head, then as `[B, H, 1, D]` rotated at each
    /// sequence's position (as `norm_rope`, unfused: the positions differ).
    #[track_caller]
    fn rotate_at(&self, h: Array, norm: Option<&RmsNorm>, positions: &Array) -> Array {
        let h = match norm {
            Some(norm) => norm.forward(&h),
            None => h,
        };
        self.rope.rope_at(&h.transpose(HEADS_FIRST), positions)
    }

    /// One new position per sequence (`x` is `[B, 1, D]`) over a paged cache: the query
    /// and key normalised and rotated at each sequence's position, the key and value
    /// written into their slots, then attention through the block tables.
    #[track_caller]
    fn forward_paged(&self, x: &Array, layer: &mut PagedLayer, step: &PagedStep) -> Array {
        let [b, l, _] = x.shape_n::<3>().unwrap_or_default();
        let (q, gate) = self.query(x, [b, l]);
        let q = self.rotate_at(q, self.q_norm.as_ref(), &step.positions);
        let k = self.k_proj.forward(x).reshape([b, l, self.kv_heads, -1]);
        let k = self.rotate_at(k, self.k_norm.as_ref(), &step.positions);
        let v = self.v_proj.forward(x);
        let rows = [b, self.kv_heads, -1];
        let (keys, values) = layer.write_kv(&k.reshape(rows), &v.reshape(rows), &step.slots);
        let o = fast::paged_attention(&q, &keys, &values, &step.tables, &step.lengths, self.scale);
        self.output(&o, gate, [b, l])
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
    /// its slot of the layer's pools: the convolution's inputs gathered and written back,
    /// the recurrent states updated in place ([`fast::gated_delta_slots`]).
    #[track_caller]
    fn forward_paged(&self, x: &Array, layer: &mut PagedLayer, step: &PagedStep) -> Array {
        let DeltaSizes {
            hv, dv, dk, kernel, ..
        } = self.sizes;
        let [b, l, _] = x.shape_n::<3>().unwrap_or_default();
        let qkv = self.in_proj_qkv.forward(x);
        let z = self.in_proj_z.forward(x);
        let (beta, a) = (self.in_proj_b.forward(x), self.in_proj_a.forward(x));
        // The convolution's last inputs of each sequence, then the new ones written back
        // (zeros until a prefilled sequence brought the pools).
        let slots = layer.state_slots;
        let conv_pool = match layer.conv.take() {
            Some(p) => p,
            None => {
                let channels = qkv.shape().get(2).copied().unwrap_or_default();
                ops::zeros([slots, kernel - 1, channels], qkv.dtype())
            }
        };
        let state = conv_pool.take(&step.state_slots, 0);
        let input = ops::concatenate(&[state, qkv], 1);
        let kept = input.i((.., 1..kernel));
        let at = step
            .state_slots
            .reshape([-1, 1, 1])
            .broadcast_to(kept.shape());
        layer.conv = Some(conv_pool.put_along_axis(&at, &kept, 0));
        let conv = causal_conv_silu(&input, &self.conv1d);
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
}

impl Mlp {
    fn new(b: &Builder<'_>, path: &str, input: usize, hidden: usize, bias: bool) -> Result<Self> {
        let p = |name: &str| format!("{path}.{name}");
        Ok(Self {
            gate_proj: b.projection(&p("gate_proj"), input, hidden, bias)?,
            up_proj: b.projection(&p("up_proj"), input, hidden, bias)?,
            down_proj: b.projection(&p("down_proj"), hidden, input, bias)?,
        })
    }

    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        let gated = swiglu(&self.gate_proj.forward(x), &self.up_proj.forward(x));
        self.down_proj.forward(&gated)
    }
}

/// One linear layer per expert, `[experts, out, in]`, applied to each row with its own
/// expert (mlx-lm's `SwitchLinear` and `QuantizedSwitchLinear`).
#[derive(Debug, Clone, Module)]
pub(crate) enum SwitchProjection {
    /// Full precision: [`Array::gather_mm`].
    Dense { weight: Param },
    /// MLX's affine quantization: [`Array::gather_qmm`]; the packed weights are buffers.
    Quantized {
        #[tree(buffer)]
        weight: Array,
        #[tree(buffer)]
        scales: Array,
        #[tree(buffer)]
        biases: Array,
        #[tree(skip)]
        group_size: i32,
        #[tree(skip)]
        bits: i32,
    },
}

impl SwitchProjection {
    /// `x` (`[..., 1, in]`) through expert `indices` (broadcast with `x`'s batch); rows
    /// sorted by expert if `sorted`.
    #[track_caller]
    fn forward(&self, x: &Array, indices: &Array, sorted: bool) -> Array {
        match self {
            Self::Dense { weight } => {
                x.gather_mm(&weight.transpose([0, 2, 1]), None, Some(indices), sorted)
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => x.gather_qmm(
                weight,
                scales,
                biases,
                None,
                Some(indices),
                true,
                *group_size,
                *bits,
                sorted,
            ),
        }
    }
}

/// The experts' SwiGLU MLPs (mlx-lm's `SwitchGLU`).
#[derive(Debug, Clone, Module)]
struct SwitchGlu {
    gate_proj: SwitchProjection,
    up_proj: SwitchProjection,
    down_proj: SwitchProjection,
}

impl SwitchGlu {
    /// Rows `x` through the experts `idx`.
    #[track_caller]
    fn through(&self, x: &Array, idx: &Array, sorted: bool) -> Array {
        let up = self.up_proj.forward(x, idx, sorted);
        let gate = self.gate_proj.forward(x, idx, sorted);
        self.down_proj.forward(&swiglu(&gate, &up), idx, sorted)
    }

    /// `x` (`[B, L, D]`) through each token's experts `indices` (`[B, L, k]`):
    /// `[B, L, k, D]`. With many rows, they are sorted by expert first, so that each
    /// expert's weights are read once (mlx-lm's `_gather_sort`).
    #[track_caller]
    fn forward(&self, x: &Array, indices: &Array) -> Array {
        let [b, l, d] = x.shape_n::<3>().unwrap_or_default();
        let k = indices.shape().get(2).copied().unwrap_or_default();
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
        let (e, i) = (moe.experts, moe.intermediate_size);
        Ok(Self {
            gate: b.projection(&p("gate"), d, e, false)?,
            switch_mlp: SwitchGlu {
                gate_proj: b.switch(&p("switch_mlp.gate_proj"), d, i, e)?,
                up_proj: b.switch(&p("switch_mlp.up_proj"), d, i, e)?,
                down_proj: b.switch(&p("switch_mlp.down_proj"), i, d, e)?,
            },
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
        // The best `k` experts by router logit (as by probability: softmax is monotonic) and
        // their weights: the probabilities, or renormalised over the chosen experts, the
        // softmax of their logits (`p_i / Σ_top p_j` with `p = softmax(logits)`).
        let logits = self.gate.forward(x);
        let top = logits.argpartition(-k, -1).i((.., .., -k..));
        // In float32, as mlx-lm's `softmax(…, precise=True)`.
        let logits = logits.astype(DType::F32);
        let scores = match self.norm_topk_prob {
            true => logits.take_along_axis(&top, -1).softmax(-1),
            false => logits.softmax(-1).take_along_axis(&top, -1),
        };
        // The weighted sum of the experts' outputs as one product (`[1, k] · [k, D]` per
        // token) in float32, rounded once to the activations' type, as mlx-lm.
        let experts = self.switch_mlp.forward(x, &top);
        let y = scores.expand_dims(-2).matmul(&experts).squeeze(-2);
        let y = y.astype(experts.dtype());
        let (Some(expert), Some(gate)) = (&self.shared_expert, &self.shared_expert_gate) else {
            return y;
        };
        y.add(gate.forward(x).sigmoid().multiply(expert.forward(x)))
    }
}

/// A layer's MLP.
#[derive(Debug, Clone, Module)]
enum BlockMlp {
    Dense(Box<Mlp>),
    Moe(Box<MoeMlp>),
}

impl BlockMlp {
    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::Moe(m) => m.forward(x),
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
        let (h, n) = add_norm(&x, Some(&mixed), &self.post_attention_layernorm);
        (h, self.mlp.forward(&n))
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
        let (h, n) = add_norm(&x, Some(&mixed), &self.post_attention_layernorm);
        (h, self.mlp.forward(&n))
    }
}

#[derive(Debug, Clone, Module)]
struct Body {
    embed_tokens: TokenEmbedding,
    layers: Vec<Block>,
    norm: RmsNorm,
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
}

impl Model {
    /// The model's structure with placeholder parameters (no memory), to be filled by
    /// [`Module::load_weights`] (until then, a forward pass gives a failed array). `quantized(path)` gives the group size and bits of each
    /// quantized layer (`None` for full precision).
    pub fn new(config: &Config, quantized: &dyn Fn(&str) -> Option<(usize, u32)>) -> Result<Self> {
        let b = Builder { quantized };
        let c = config;
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let path = |s: &str| format!("model.layers.{i}.{s}");
                let mixer = match (c.layer_types[i], c.linear_attention) {
                    (LayerType::Linear, Some(sizes)) => {
                        Mixer::Linear(GatedDeltaNet::new(&b, &path("linear_attn"), c, sizes)?)
                    }
                    _ => Mixer::Attention(Attention::new(&b, &path("self_attn"), c)?),
                };
                Ok(Block {
                    mixer,
                    mlp: match c.moe.as_ref().filter(|m| m.is_sparse(i)) {
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
            },
            lm_head: (!c.tie_word_embeddings)
                .then(|| b.projection("lm_head", c.hidden_size, c.vocab_size, false))
                .transpose()?,
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

    /// An empty cache per layer.
    pub fn new_cache(&self) -> Vec<KvCache> {
        vec![KvCache::default(); self.model.layers.len()]
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
        let mut h = self.model.embed_tokens.forward(tokens);
        // Causal (aligned to the last positions) for several tokens; one token sees all.
        let mask = match tokens.shape().get(1) {
            Some(&l) if l > 1 => AttentionMask::Causal,
            _ => AttentionMask::None,
        };
        let mut pending = None;
        for (layer, c) in self.model.layers.iter().zip(cache) {
            let (next, mlp) = layer.forward(&h, pending.as_ref(), mask, c);
            (h, pending) = (next, Some(mlp));
        }
        let (_, h) = add_norm(&h, pending.as_ref(), &self.model.norm);
        self.head(&h)
    }

    /// The output head on the final hidden states.
    #[track_caller]
    fn head(&self, h: &Array) -> Array {
        match &self.lm_head {
            Some(head) => head.forward(h),
            None => self.model.embed_tokens.as_linear(h),
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
        let mut h = self.model.embed_tokens.forward(tokens);
        let mut pending = None;
        for (layer, pages) in self.model.layers.iter().zip(layers) {
            let (next, mlp) = layer.forward_paged(&h, pending.as_ref(), pages, step);
            (h, pending) = (next, Some(mlp));
        }
        let (_, h) = add_norm(&h, pending.as_ref(), &self.model.norm);
        self.head(&h)
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

    /// A model of the `qwen3_f32` fixture, or `None` without a Metal device.
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
