//! The Llama-family transformer, as mlx-lm's `llama`, `qwen2` and `qwen3` models, and its
//! hybrid with Gated DeltaNet layers, as mlx-lm's `qwen3_5`, with parameter names matching
//! Hugging Face checkpoints.

use std::sync::LazyLock;

use nuvec::fast::{self, AttentionMask};
use nuvec::{
    Array, DType, Result, SliceArg, abs, add, argpartition, argsort, broadcast_to, concatenate,
    exp, expand_dims, gather_mm, gather_qmm, log, matmul, maximum, multiply, negative, sigmoid,
    slice, slice_update, softmax, split_at, squeeze, take, take_along_axis, transpose,
};
use nuvec_nn::{
    Embedding, Linear, Module, Parameters, QuantizedEmbedding, QuantizedLinear, RmsNorm, silu,
};

use crate::config::{Config, LayerType, LinearAttention, MixtureOfExperts, RopeScaling};

/// A zero-cost stand-in of `shape` (a broadcast scalar), replaced when weights load.
fn placeholder(shape: &[usize], dtype: DType) -> Result<Array> {
    broadcast_to(&Array::scalar(0u8).astype(dtype)?, shape)
}

/// `silu(gate) · x` in one kernel, as mlx-lm's compiled `swiglu`.
fn swiglu(gate: &Array, x: &Array) -> Result<Array> {
    static SWIGLU: LazyLock<Compiled> = LazyLock::new(|| {
        Box::new(nuvec::compile(|a: &[Array]| {
            Ok(vec![multiply(silu(&a[0])?, &a[1])?])
        }))
    });
    Ok(SWIGLU(&[gate.clone(), x.clone()])?.swap_remove(0))
}

/// A linear layer, full precision or quantized.
#[derive(Debug, Clone)]
pub enum Projection {
    /// Full precision.
    Dense(Linear),
    /// MLX's affine quantization.
    Quantized(QuantizedLinear),
}

impl Projection {
    /// `x Wᵀ (+ b)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }
}

impl Module for Projection {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        match self {
            Self::Dense(l) => l.visit(p),
            Self::Quantized(q) => q.visit(p),
        }
    }
}

/// A token embedding, full precision or quantized.
#[derive(Debug, Clone)]
pub enum TokenEmbedding {
    /// Full precision.
    Dense(Embedding),
    /// MLX's affine quantization.
    Quantized(QuantizedEmbedding),
}

impl TokenEmbedding {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(e) => e.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }

    fn as_linear(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(e) => e.as_linear(x),
            Self::Quantized(q) => q.as_linear(x),
        }
    }
}

impl Module for TokenEmbedding {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        match self {
            Self::Dense(e) => e.visit(p),
            Self::Quantized(q) => q.visit(p),
        }
    }
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
        let bias = bias
            .then(|| placeholder(&[output], DType::F32))
            .transpose()?;
        Ok(match (self.quantized)(path) {
            Some((group_size, bits)) => Projection::Quantized(QuantizedLinear {
                weight: placeholder(&[output, input * bits as usize / 32], DType::U32)?,
                scales: placeholder(&[output, input / group_size], DType::F32)?,
                biases: placeholder(&[output, input / group_size], DType::F32)?,
                bias,
                group_size,
                bits,
            }),
            None => Projection::Dense(Linear {
                weight: placeholder(&[output, input], DType::F32)?,
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
            Some((group_size, bits)) => SwitchProjection::Quantized {
                weight: placeholder(&[experts, output, input * bits as usize / 32], DType::U32)?,
                scales: placeholder(&[experts, output, input / group_size], DType::F32)?,
                biases: placeholder(&[experts, output, input / group_size], DType::F32)?,
                group_size,
                bits,
            },
            None => SwitchProjection::Dense {
                weight: placeholder(&[experts, output, input], DType::F32)?,
            },
        })
    }

    fn embedding(&self, path: &str, count: usize, dims: usize) -> Result<TokenEmbedding> {
        Ok(match (self.quantized)(path) {
            Some((group_size, bits)) => TokenEmbedding::Quantized(QuantizedEmbedding {
                weight: placeholder(&[count, dims * bits as usize / 32], DType::U32)?,
                scales: placeholder(&[count, dims / group_size], DType::F32)?,
                biases: placeholder(&[count, dims / group_size], DType::F32)?,
                group_size,
                bits,
            }),
            None => TokenEmbedding::Dense(Embedding {
                weight: placeholder(&[count, dims], DType::F32)?,
            }),
        })
    }

    fn norm(dims: usize, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: placeholder(&[dims], DType::F32)?,
            eps,
        })
    }
}

/// The rotary embedding of every head's features.
#[derive(Debug, Clone)]
struct Rope {
    dims: usize,
    traditional: bool,
    base: f32,
    scale: f32,
    /// Per-pair frequencies (Llama 3 scaling) instead of `base`.
    freqs: Option<Array>,
}

impl Rope {
    fn new(dims: usize, config: &Config) -> Result<Self> {
        let base = config.rope_theta;
        let mut rope = Self {
            dims,
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
                rope.freqs = Some(Array::from_slice(&freqs, &[freqs.len()])?);
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
}

/// Keys before they are cached: `[B, kv_heads, L, head_dim]`, to be normalised (optionally)
/// and rotated at the cache's offset as they are written ([`fast::norm_rope`]: one kernel
/// from the projection to the cache).
struct Keys<'a> {
    x: Array,
    norm: Option<&'a RmsNorm>,
    rope: &'a Rope,
}

impl Keys<'_> {
    /// The keys at positions `offset..`, written into `cache` if given.
    fn rotate(&self, offset: usize, cache: Option<&Array>) -> Result<Array> {
        let norm = self.norm.map(|n| (&n.weight, n.eps));
        fast::norm_rope(&self.x, norm, self.rope.rotation(), offset, cache)
    }
}

/// One layer's cache. Attention layers keep keys and values, `[B, kv_heads, capacity,
/// head_dim]`, as mlx-lm's `KVCache`: the capacity grows in steps of 256 positions and
/// each step's keys and values are written into it (`slice_update`), so buffers keep their
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
    offset: usize,
    /// Whether Gated DeltaNet layers keep every step's state ([`KvCache::keep_steps`]).
    keep_steps: bool,
    /// From the last forward with `keep_steps`: every step's recurrent state
    /// (`[B, T, ...]`) and the convolution's input (`[B, kernel − 1 + T, channels]`).
    steps: Option<(Array, Array)>,
}

/// Positions added to a cache's capacity at a time.
const CACHE_STEP: usize = 256;

impl KvCache {
    /// Positions cached.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Writes `keys` and `values` (`[B, kv_heads, L, head_dim]`) after the cached positions
    /// and returns all cached positions.
    fn update(&mut self, keys: Keys<'_>, values: Array) -> Result<(Array, Array)> {
        let (prev, l) = (self.offset, values.shape()[2]);
        let capacity = self.keys.as_ref().map_or(0, |k| k.shape()[2]);
        if self.keys.is_none() || prev + l > capacity {
            // Grown by whole steps (mlx-lm's rule), written in one concatenation: the cached
            // positions, the new ones, then zeros up to the step (no copy of the grown cache
            // to update it).
            let pad = l.div_ceil(CACHE_STEP) * CACHE_STEP - l;
            let grow = |old: &Option<Array>, new: &Array| -> Result<Array> {
                let mut parts = Vec::with_capacity(3);
                if let Some(old) = old {
                    parts.push(positions(old, prev)?);
                }
                parts.push(new.clone());
                if pad > 0 {
                    let mut shape = new.shape().to_vec();
                    shape[2] = pad;
                    parts.push(nuvec::zeros(&shape, new.dtype())?);
                }
                concatenate(&parts, 2)
            };
            let keys = keys.rotate(prev, None)?;
            let (k, v) = (grow(&self.keys, &keys)?, grow(&self.values, &values)?);
            self.offset = prev + l;
            let (pk, pv) = (positions(&k, self.offset)?, positions(&v, self.offset)?);
            self.keys = Some(k);
            self.values = Some(v);
            return Ok((pk, pv));
        }
        self.offset = prev + l;
        // Each write takes the cache's only handle, so it is done in place.
        let all = self.keys.take().expect("allocated above");
        let k = keys.rotate(prev, Some(&all))?;
        self.keys = Some(k.clone());
        let all = self.values.take().expect("allocated above");
        let region = [
            SliceArg::full(),
            SliceArg::full(),
            SliceArg::new(Some(prev as isize), Some(self.offset as isize), 1),
            SliceArg::full(),
        ];
        let v = slice_update(&all, &values, &region)?;
        self.values = Some(v.clone());
        Ok((positions(&k, self.offset)?, positions(&v, self.offset)?))
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
        if self.conv.is_some() || self.recurrent.is_some() {
            if n == 0 {
                return Ok(());
            }
            let Some((states, input)) = self.steps.take() else {
                return Err(nuvec::Error::InvalidArgument(
                    "a Gated DeltaNet layer rolls back only after a forward that kept its \
                     steps (KvCache::keep_steps)"
                        .into(),
                ));
            };
            let t = states.shape()[1];
            if n >= t {
                return Err(nuvec::Error::InvalidArgument(format!(
                    "cannot roll back {n} of the last forward's {t} positions"
                )));
            }
            let along = |a: &Array, start: usize, len: usize| -> Result<Array> {
                let mut args = vec![SliceArg::full(); a.ndim()];
                args[1] = SliceArg::new(Some(start as isize), Some((start + len) as isize), 1);
                slice(a, &args)
            };
            let kept = t - n;
            self.recurrent = Some(squeeze(&along(&states, kept - 1, 1)?, &[1])?);
            // The last `kernel − 1` inputs before the rejected positions (a copy: the view
            // would keep the whole input alive).
            let window = input.shape()[1] - t;
            self.conv = Some(multiply(&along(&input, kept, window)?, 1.0)?);
            self.offset -= n;
            return Ok(());
        }
        if n > self.offset {
            return Err(nuvec::Error::InvalidArgument(format!(
                "cannot trim {n} of {} cached positions",
                self.offset
            )));
        }
        self.offset -= n;
        Ok(())
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
fn positions(a: &Array, n: usize) -> Result<Array> {
    if a.shape()[2] == n {
        return Ok(a.clone());
    }
    slice(
        a,
        &[
            SliceArg::full(),
            SliceArg::full(),
            SliceArg::new(Some(0), Some(n as isize), 1),
            SliceArg::full(),
        ],
    )
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,
    o_proj: Projection,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    heads: usize,
    kv_heads: usize,
    scale: f32,
    rope: Rope,
    /// The query projection also yields a gate of the output (`o · sigmoid(gate)`).
    gate: bool,
}

impl Attention {
    fn new(b: &Builder<'_>, path: &str, c: &Config) -> Result<Self> {
        let (d, hd) = (c.hidden_size, c.head_dim);
        let (q, kv) = (c.num_attention_heads * hd, c.num_key_value_heads * hd);
        let p = |name: &str| format!("{path}.{name}");
        let norm = || {
            c.qk_norm
                .then(|| Builder::norm(hd, c.rms_norm_eps))
                .transpose()
        };
        let gate = c.attn_output_gate;
        Ok(Self {
            q_proj: b.projection(&p("q_proj"), d, q * (1 + usize::from(gate)), c.qkv_bias)?,
            k_proj: b.projection(&p("k_proj"), d, kv, c.qkv_bias)?,
            v_proj: b.projection(&p("v_proj"), d, kv, c.qkv_bias)?,
            o_proj: b.projection(&p("o_proj"), q, d, c.o_bias)?,
            q_norm: norm()?,
            k_norm: norm()?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            scale: (hd as f32).powf(-0.5),
            rope: Rope::new(c.rope_dims, c)?,
            gate,
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask<'_>, cache: &mut KvCache) -> Result<Array> {
        let (b, l) = (x.shape()[0] as isize, x.shape()[1] as isize);
        // `[B, L, H·D]` as `[B, H, L, D]` (a view).
        let heads = |h: Array, n: usize| -> Result<Array> {
            transpose(&h.reshape(&[b, l, n as isize, -1])?, &[0, 2, 1, 3])
        };
        let offset = cache.offset();
        // With an output gate, each head's projection is its query, then its gate.
        let (q, gate) = match self.gate {
            false => (self.q_proj.forward(x)?, None),
            true => {
                let qg = self
                    .q_proj
                    .forward(x)?
                    .reshape(&[b, l, self.heads as isize, -1])?;
                let half = qg.shape()[3] / 2;
                let mut parts = split_at(&qg, &[half], -1)?;
                let g = parts.pop().expect("two parts").reshape(&[b, l, -1])?;
                (parts.pop().expect("two parts"), Some(g))
            }
        };
        // The query (and, written into the cache, the key) normalised and rotated in one
        // kernel each.
        let q = fast::norm_rope(
            &heads(q, self.heads)?,
            self.q_norm.as_ref().map(|n| (&n.weight, n.eps)),
            self.rope.rotation(),
            offset,
            None,
        )?;
        let keys = Keys {
            x: heads(self.k_proj.forward(x)?, self.kv_heads)?,
            norm: self.k_norm.as_ref(),
            rope: &self.rope,
        };
        let v = heads(self.v_proj.forward(x)?, self.kv_heads)?;
        let (k, v) = cache.update(keys, v)?;
        let o = fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let o = transpose(&o, &[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        match gate {
            Some(g) => self.o_proj.forward(&multiply(&o, sigmoid(&g)?)?),
            None => self.o_proj.forward(&o),
        }
    }
}

impl Module for Attention {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("q_proj", &mut self.q_proj);
        p.module("k_proj", &mut self.k_proj);
        p.module("v_proj", &mut self.v_proj);
        p.module("o_proj", &mut self.o_proj);
        if let Some(n) = &mut self.q_norm {
            p.module("q_norm", n);
        }
        if let Some(n) = &mut self.k_norm {
            p.module("k_norm", n);
        }
    }
}

/// `log(1 + exp(x))`, stably (MLX's `nn.softplus`, `logaddexp(x, 0)`): in float32,
/// rounded once to `x`'s dtype, as MLX's kernel.
fn softplus(x: &Array) -> Result<Array> {
    let xf = x.astype(DType::F32)?;
    let y = add(
        maximum(&xf, 0.0)?,
        log(&add(exp(&negative(&abs(&xf)?)?)?, 1.0)?)?,
    )?;
    y.astype(x.dtype())
}

/// A compiled function of arrays (traced once per input signature, element-wise chains
/// fused), as mlx-lm compiles its activations.
type Compiled = Box<dyn Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync>;

/// The gated RMS normalisation of Gated DeltaNet's output (mlx-lm's
/// `Qwen3NextRMSNormGated` with its compiled `precise_swiglu`): `silu(z) · rms_norm(x)`, in
/// float32.
fn gated_norm(x: &Array, z: &Array, norm: &RmsNorm) -> Result<Array> {
    static GATE: LazyLock<Compiled> = LazyLock::new(|| {
        Box::new(nuvec::compile(|a: &[Array]| {
            let (n, z) = (a[0].astype(DType::F32)?, a[1].astype(DType::F32)?);
            Ok(vec![multiply(silu(&z)?, &n)?.astype(a[0].dtype())?])
        }))
    });
    Ok(GATE(&[norm.forward(x)?, z.clone()])?.swap_remove(0))
}

/// Gated DeltaNet's per-head decay, `exp(−exp(A_log) · softplus(a + dt_bias))` in float32
/// (mlx-lm's compiled `compute_g`).
fn decay(a: &Array, dt_bias: &Array, a_log: &Array) -> Result<Array> {
    static DECAY: LazyLock<Compiled> = LazyLock::new(|| {
        Box::new(nuvec::compile(|x: &[Array]| {
            let rate = exp(&x[2].astype(DType::F32)?)?;
            let g = exp(&multiply(negative(&rate)?, softplus(&add(&x[0], &x[1])?)?)?)?;
            Ok(vec![g])
        }))
    });
    Ok(DECAY(&[a.clone(), dt_bias.clone(), a_log.clone()])?.swap_remove(0))
}

/// The causal depthwise convolution of `input` (`[B, L + K − 1, C]`) with `weight`
/// (`[C, K, 1]`), then `silu`, as mlx-lm: the convolution is one depthwise kernel (summed in
/// float32 and rounded once, as MLX's).
fn causal_conv_silu(input: &Array, weight: &Array) -> Result<Array> {
    let channels = weight.shape()[0];
    silu(&nuvec::conv1d(input, weight, 1, 0, 1, channels)?)
}

/// A Gated DeltaNet layer (mlx-lm's `qwen3_5.GatedDeltaNet`): a causal depthwise
/// convolution over the query, key and value projections, then the gated delta rule
/// ([`nuvec::fast::gated_delta`]) with per-head decays and write strengths, and a gated
/// normalisation of the output.
#[derive(Debug, Clone)]
struct GatedDeltaNet {
    in_proj_qkv: Projection,
    in_proj_z: Projection,
    in_proj_b: Projection,
    in_proj_a: Projection,
    /// The depthwise convolution's weights, `[channels, kernel, 1]`.
    conv1d: Array,
    dt_bias: Array,
    a_log: Array,
    norm: RmsNorm,
    out_proj: Projection,
    sizes: LinearAttention,
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
            conv1d: placeholder(&[conv_dim, sizes.conv_kernel, 1], DType::F32)?,
            dt_bias: placeholder(&[hv], DType::F32)?,
            a_log: placeholder(&[hv], DType::F32)?,
            norm: Builder::norm(sizes.value_head_dim, c.rms_norm_eps)?,
            out_proj: b.projection(&p("out_proj"), value_dim, d, false)?,
            sizes,
        })
    }

    fn forward(&self, x: &Array, cache: &mut KvCache) -> Result<Array> {
        let LinearAttention {
            key_heads: hk,
            value_heads: hv,
            key_head_dim: dk,
            value_head_dim: dv,
            conv_kernel: kernel,
        } = self.sizes;
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let qkv = self.in_proj_qkv.forward(x)?;
        let channels = qkv.shape()[2];
        let z = self.in_proj_z.forward(x)?;
        let (beta, a) = (self.in_proj_b.forward(x)?, self.in_proj_a.forward(x)?);
        // The causal convolution sees the previous step's last `kernel - 1` inputs.
        let state = match cache.conv.take() {
            Some(c) => c,
            None => nuvec::zeros(&[b, kernel - 1, channels], qkv.dtype())?,
        };
        let input = concatenate(&[state, qkv], 1)?;
        let rows = |start: usize, len: usize| {
            slice(
                &input,
                &[
                    SliceArg::full(),
                    SliceArg::new(Some(start as isize), Some((start + len) as isize), 1),
                    SliceArg::full(),
                ],
            )
        };
        let kept = rows(l, kernel - 1)?;
        // After a prompt, a copy: the view would keep the whole input alive.
        cache.conv = Some(if l > 1 { multiply(&kept, 1.0)? } else { kept });
        let conv = causal_conv_silu(&input, &self.conv1d)?;
        let mut parts = split_at(&conv, &[hk * dk, 2 * hk * dk], -1)?;
        let shaped = |a: Array, h: usize, d: usize| {
            a.reshape(&[b as isize, l as isize, h as isize, d as isize])
        };
        let v = shaped(parts.pop().expect("three parts"), hv, dv)?;
        let k = shaped(parts.pop().expect("three parts"), hk, dk)?;
        let q = shaped(parts.pop().expect("three parts"), hk, dk)?;
        // mlx-lm's `normalize_qk`: unit queries and keys, the 1/sqrt(Dk) scale in the query.
        let inv = (dk as f32).powf(-0.5);
        let eps = 1e-6 * inv * inv;
        let q = multiply(&nuvec::fast::rms_norm(&q, None, eps)?, f64::from(inv * inv))?;
        let k = multiply(&nuvec::fast::rms_norm(&k, None, eps)?, f64::from(inv))?;
        // Per-head decay and write strength (mlx-lm's `compute_g`, in float32).
        let g = decay(&a, &self.dt_bias, &self.a_log)?;
        let beta = sigmoid(&beta)?;
        let state = match cache.recurrent.take() {
            Some(s) => s,
            None => nuvec::zeros(&[b, hv, dv, dk], DType::F32)?,
        };
        let out = if cache.keep_steps {
            let (out, states) = nuvec::fast::gated_delta_steps(&q, &k, &v, &g, &beta, &state)?;
            let mut last = vec![SliceArg::full(); states.ndim()];
            last[1] = SliceArg::new(Some(l as isize - 1), Some(l as isize), 1);
            let last = slice(&states, &last)?;
            cache.recurrent = Some(squeeze(&last, &[1])?);
            cache.steps = Some((states, input));
            out
        } else {
            let (out, state) = nuvec::fast::gated_delta(&q, &k, &v, &g, &beta, &state)?;
            cache.recurrent = Some(state);
            out
        };
        cache.offset += l;
        let z = z.reshape(&[b as isize, l as isize, hv as isize, dv as isize])?;
        let out = gated_norm(&out, &z, &self.norm)?;
        self.out_proj
            .forward(&out.reshape(&[b as isize, l as isize, -1])?)
    }
}

impl Module for GatedDeltaNet {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("in_proj_qkv", &mut self.in_proj_qkv);
        p.module("in_proj_z", &mut self.in_proj_z);
        p.module("in_proj_b", &mut self.in_proj_b);
        p.module("in_proj_a", &mut self.in_proj_a);
        p.param("conv1d.weight", &mut self.conv1d);
        p.param("dt_bias", &mut self.dt_bias);
        p.param("A_log", &mut self.a_log);
        p.module("norm", &mut self.norm);
        p.module("out_proj", &mut self.out_proj);
    }
}

/// A layer's token mixer.
#[derive(Debug, Clone)]
enum Mixer {
    Attention(Attention),
    Linear(GatedDeltaNet),
}

#[derive(Debug, Clone)]
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

    fn forward(&self, x: &Array) -> Result<Array> {
        let gated = swiglu(&self.gate_proj.forward(x)?, &self.up_proj.forward(x)?)?;
        self.down_proj.forward(&gated)
    }
}

impl Module for Mlp {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("gate_proj", &mut self.gate_proj);
        p.module("up_proj", &mut self.up_proj);
        p.module("down_proj", &mut self.down_proj);
    }
}

/// One linear layer per expert, `[experts, out, in]`, applied to each row with its own
/// expert (mlx-lm's `SwitchLinear` and `QuantizedSwitchLinear`).
#[derive(Debug, Clone)]
enum SwitchProjection {
    /// Full precision: [`gather_mm`].
    Dense { weight: Array },
    /// MLX's affine quantization: [`gather_qmm`].
    Quantized {
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: usize,
        bits: u32,
    },
}

impl SwitchProjection {
    /// `x` (`[..., 1, in]`) through expert `indices` (broadcast with `x`'s batch); rows
    /// sorted by expert if `sorted`.
    fn forward(&self, x: &Array, indices: &Array, sorted: bool) -> Result<Array> {
        match self {
            Self::Dense { weight } => {
                let nd = weight.ndim();
                let mut axes: Vec<usize> = (0..nd).collect();
                axes.swap(nd - 1, nd - 2);
                gather_mm(x, &transpose(weight, &axes)?, None, Some(indices), sorted)
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
            } => gather_qmm(
                x,
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

impl Module for SwitchProjection {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        match self {
            Self::Dense { weight } => p.param("weight", weight),
            Self::Quantized {
                weight,
                scales,
                biases,
                ..
            } => {
                p.param("weight", weight);
                p.param("scales", scales);
                p.param("biases", biases);
            }
        }
    }
}

/// The experts' SwiGLU MLPs (mlx-lm's `SwitchGLU`).
#[derive(Debug, Clone)]
struct SwitchGlu {
    gate_proj: SwitchProjection,
    up_proj: SwitchProjection,
    down_proj: SwitchProjection,
}

impl SwitchGlu {
    /// `x` (`[B, L, D]`) through each token's experts `indices` (`[B, L, k]`):
    /// `[B, L, k, D]`. With many rows, they are sorted by expert first, so that each
    /// expert's weights are read once (mlx-lm's `_gather_sort`).
    fn forward(&self, x: &Array, indices: &Array) -> Result<Array> {
        let (b, l, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let k = indices.shape()[2];
        let through = |x: &Array, idx: &Array, sorted: bool| -> Result<Array> {
            let up = self.up_proj.forward(x, idx, sorted)?;
            let gate = self.gate_proj.forward(x, idx, sorted)?;
            self.down_proj.forward(&swiglu(&gate, &up)?, idx, sorted)
        };
        if indices.size() < 64 {
            let x = expand_dims(&expand_dims(x, -2)?, -2)?; // [B, L, 1, 1, D]
            return squeeze(&through(&x, indices, false)?, &[-2]);
        }
        let flat = indices.reshape(&[-1])?;
        let order = argsort(&flat, Some(0))?;
        let restore = argsort(&order, Some(0))?;
        // Each pair's token: pair `t·k + j` belongs to token `t`.
        let tokens = broadcast_to(
            &nuvec::arange(0.0, (b * l) as f64, 1.0, DType::U32)?.reshape(&[-1, 1])?,
            &[b * l, k],
        )?
        .reshape(&[-1])?;
        let rows = take(
            &x.reshape(&[-1, 1, d as isize])?,
            &take(&tokens, &order, Some(0))?,
            Some(0),
        )?;
        let out = through(&rows, &take(&flat, &order, Some(0))?, true)?;
        take(&out, &restore, Some(0))?.reshape(&[b as isize, l as isize, k as isize, d as isize])
    }
}

impl Module for SwitchGlu {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("gate_proj", &mut self.gate_proj);
        p.module("up_proj", &mut self.up_proj);
        p.module("down_proj", &mut self.down_proj);
    }
}

/// A mixture-of-experts MLP (mlx-lm's `Qwen3NextSparseMoeBlock`): a softmax router picks
/// each token's best experts, whose outputs are summed with the (renormalised) router
/// weights, plus a shared expert gated by a sigmoid.
#[derive(Debug, Clone)]
struct MoeMlp {
    gate: Projection,
    switch_mlp: SwitchGlu,
    /// Qwen 3.5's gated shared expert.
    shared: Option<(Mlp, Projection)>,
    experts_per_token: usize,
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
            shared: match moe.shared_intermediate_size {
                Some(size) => Some((
                    Mlp::new(b, &p("shared_expert"), d, size, false)?,
                    b.projection(&p("shared_expert_gate"), d, 1, false)?,
                )),
                None => None,
            },
            experts_per_token: moe.experts_per_token,
            norm_topk_prob: moe.norm_topk_prob,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let k = self.experts_per_token as isize;
        // The best `k` experts by router logit (as by probability: softmax is monotonic) and
        // their weights: the probabilities, or renormalised over the chosen experts, the
        // softmax of their logits (`p_i / Σ_top p_j` with `p = softmax(logits)`).
        let logits = self.gate.forward(x)?;
        let order = argpartition(&logits, -k, Some(-1))?;
        let e = order.shape()[2] as isize;
        let top = slice(
            &order,
            &[
                SliceArg::full(),
                SliceArg::full(),
                SliceArg::new(Some(e - k), None, 1),
            ],
        )?;
        // In float32, as mlx-lm's `softmax(…, precise=True)`.
        let logits = logits.astype(DType::F32)?;
        let scores = match self.norm_topk_prob {
            true => softmax(&take_along_axis(&logits, &top, -1)?, &[-1])?,
            false => take_along_axis(&softmax(&logits, &[-1])?, &top, -1)?,
        };
        // The weighted sum of the experts' outputs as one product (`[1, k] · [k, D]` per
        // token) in float32, rounded once to the activations' type, as mlx-lm.
        let experts = self.switch_mlp.forward(x, &top)?;
        let y = squeeze(&matmul(&expand_dims(&scores, -2)?, &experts)?, &[-2])?;
        let y = y.astype(experts.dtype())?;
        let Some((expert, gate)) = &self.shared else {
            return Ok(y);
        };
        add(
            &y,
            multiply(sigmoid(&gate.forward(x)?)?, expert.forward(x)?)?,
        )
    }
}

impl Module for MoeMlp {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("gate", &mut self.gate);
        p.module("switch_mlp", &mut self.switch_mlp);
        if let Some((expert, gate)) = &mut self.shared {
            p.module("shared_expert", expert);
            p.module("shared_expert_gate", gate);
        }
    }
}

/// A layer's MLP.
#[derive(Debug, Clone)]
enum BlockMlp {
    Dense(Box<Mlp>),
    Moe(Box<MoeMlp>),
}

impl Module for BlockMlp {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        match self {
            Self::Dense(m) => m.visit(p),
            Self::Moe(m) => m.visit(p),
        }
    }
}

#[derive(Debug, Clone)]
struct Block {
    mixer: Mixer,
    mlp: BlockMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// `x + r` and its normalisation, in one kernel ([`fast::add_rms_norm`]); `x` alone
/// without `r`.
fn add_norm(x: &Array, r: Option<&Array>, norm: &RmsNorm) -> Result<(Array, Array)> {
    match r {
        Some(r) => fast::add_rms_norm(x, r, Some(&norm.weight), norm.eps),
        None => Ok((x.clone(), norm.forward(x)?)),
    }
}

impl Block {
    /// The block on `x + pending` (the previous block's MLP output, not yet added):
    /// gives the residual stream after attention and this block's MLP output. Each
    /// residual addition is fused with the normalisation that follows it, so a layer has
    /// no dependent dispatch for the additions.
    fn forward(
        &self,
        x: &Array,
        pending: Option<&Array>,
        mask: AttentionMask<'_>,
        cache: &mut KvCache,
    ) -> Result<(Array, Array)> {
        let (x, n) = add_norm(x, pending, &self.input_layernorm)?;
        let mixed = match &self.mixer {
            Mixer::Attention(a) => a.forward(&n, mask, cache)?,
            Mixer::Linear(l) => l.forward(&n, cache)?,
        };
        let (h, n) = add_norm(&x, Some(&mixed), &self.post_attention_layernorm)?;
        let mlp = match &self.mlp {
            BlockMlp::Dense(m) => m.forward(&n)?,
            BlockMlp::Moe(m) => m.forward(&n)?,
        };
        Ok((h, mlp))
    }
}

impl Module for Block {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        match &mut self.mixer {
            Mixer::Attention(a) => p.module("self_attn", a),
            Mixer::Linear(l) => p.module("linear_attn", l),
        }
        p.module("mlp", &mut self.mlp);
        p.module("input_layernorm", &mut self.input_layernorm);
        p.module(
            "post_attention_layernorm",
            &mut self.post_attention_layernorm,
        );
    }
}

#[derive(Debug, Clone)]
struct Body {
    embed_tokens: TokenEmbedding,
    layers: Vec<Block>,
    norm: RmsNorm,
}

impl Module for Body {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("embed_tokens", &mut self.embed_tokens);
        p.module("layers", &mut self.layers);
        p.module("norm", &mut self.norm);
    }
}

/// A causal language model of the Llama family.
#[derive(Debug, Clone)]
pub struct Model {
    /// The configuration it was built from.
    pub config: Config,
    model: Body,
    lm_head: Option<Projection>,
}

impl Model {
    /// The model's structure with placeholder parameters (no memory), to be filled by
    /// [`Module::load_weights`]. `quantized(path)` gives the group size and bits of each
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
                    input_layernorm: Builder::norm(c.hidden_size, c.rms_norm_eps)?,
                    post_attention_layernorm: Builder::norm(c.hidden_size, c.rms_norm_eps)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            config: config.clone(),
            model: Body {
                embed_tokens: b.embedding("model.embed_tokens", c.vocab_size, c.hidden_size)?,
                layers,
                norm: Builder::norm(c.hidden_size, c.rms_norm_eps)?,
            },
            lm_head: (!c.tie_word_embeddings)
                .then(|| b.projection("lm_head", c.hidden_size, c.vocab_size, false))
                .transpose()?,
        })
    }

    /// Runs one token through the model (and discards the result), so that the first
    /// prompt does not pay for first use: on the GPU, memory-mapped weights are made
    /// resident when a command buffer first reads them, and kernels are compiled when
    /// first dispatched. [`load_model`](crate::load_model) calls it.
    pub fn warm_up(&self) -> Result<()> {
        let token = Array::from_slice(&[0u32], &[1, 1])?;
        nuvec::eval(&[self.forward(&token, &mut self.new_cache())?])
    }

    /// An empty cache per layer.
    pub fn new_cache(&self) -> Vec<KvCache> {
        vec![KvCache::default(); self.model.layers.len()]
    }

    /// The logits `[B, L, vocab]` for tokens `[B, L]` (integers) following the cached
    /// positions, which are extended.
    pub fn forward(&self, tokens: &Array, cache: &mut [KvCache]) -> Result<Array> {
        let mut h = self.model.embed_tokens.forward(tokens)?;
        // Causal (aligned to the last positions) for several tokens; one token sees all.
        let mask = if tokens.shape()[1] > 1 {
            AttentionMask::Causal
        } else {
            AttentionMask::None
        };
        let mut pending = None;
        for (layer, c) in self.model.layers.iter().zip(cache) {
            let (next, mlp) = layer.forward(&h, pending.as_ref(), mask, c)?;
            (h, pending) = (next, Some(mlp));
        }
        let (_, h) = add_norm(&h, pending.as_ref(), &self.model.norm)?;
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.model.embed_tokens.as_linear(&h),
        }
    }
}

impl Module for Model {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("model", &mut self.model);
        if let Some(head) = &mut self.lm_head {
            p.module("lm_head", head);
        }
    }
}
