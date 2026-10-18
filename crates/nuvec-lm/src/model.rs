//! The Llama-family transformer, as mlx-lm's `llama`, `qwen2` and `qwen3` models, with
//! parameter names matching Hugging Face checkpoints.

use nuvec::fast::{self, AttentionMask};
use nuvec::{
    Array, DType, Result, SliceArg, add, broadcast_to, concatenate, multiply, slice, slice_update,
    transpose,
};
use nuvec_nn::{
    Embedding, Linear, Module, Parameters, QuantizedEmbedding, QuantizedLinear, RmsNorm, silu,
};

use crate::config::{Config, RopeScaling};

/// A zero-cost stand-in of `shape` (a broadcast scalar), replaced when weights load.
fn placeholder(shape: &[usize], dtype: DType) -> Result<Array> {
    broadcast_to(&Array::scalar(0u8).astype(dtype)?, shape)
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

    fn forward(&self, x: &Array, offset: usize) -> Result<Array> {
        fast::rope(
            x,
            self.dims,
            self.traditional,
            self.freqs.is_none().then_some(self.base),
            self.scale,
            &Array::scalar(offset as i32),
            self.freqs.as_ref(),
        )
    }
}

/// One layer's keys and values, `[B, kv_heads, capacity, head_dim]`, as mlx-lm's
/// `KVCache`: the capacity grows in steps of 256 positions and each step's keys and
/// values are written into it (`slice_update`), so buffers keep their size between
/// growths.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: usize,
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
    fn update(&mut self, keys: Array, values: Array) -> Result<(Array, Array)> {
        let (prev, l) = (self.offset, keys.shape()[2]);
        let capacity = self.keys.as_ref().map_or(0, |k| k.shape()[2]);
        if self.keys.is_none() || prev + l > capacity {
            let grow = |old: &Option<Array>, like: &Array| -> Result<Array> {
                let mut shape = like.shape().to_vec();
                shape[2] = l.div_ceil(CACHE_STEP) * CACHE_STEP;
                let fresh = nuvec::zeros(&shape, like.dtype())?;
                match old {
                    Some(old) => concatenate(&[positions(old, prev)?, fresh], 2),
                    None => Ok(fresh),
                }
            };
            self.keys = Some(grow(&self.keys, &keys)?);
            self.values = Some(grow(&self.values, &values)?);
        }
        self.offset = prev + l;
        let write = |cache: &mut Option<Array>, new: &Array| -> Result<Array> {
            let all = cache.take().expect("allocated above");
            let region = [
                SliceArg::full(),
                SliceArg::full(),
                SliceArg::new(Some(prev as isize), Some(self.offset as isize), 1),
                SliceArg::full(),
            ];
            let updated = slice_update(&all, new, &region)?;
            *cache = Some(updated.clone());
            positions(&updated, self.offset)
        };
        let k = write(&mut self.keys, &keys)?;
        let v = write(&mut self.values, &values)?;
        Ok((k, v))
    }

    /// The cached arrays (to evaluate them).
    pub fn state(&self) -> Vec<Array> {
        self.keys.iter().chain(&self.values).cloned().collect()
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
        Ok(Self {
            q_proj: b.projection(&p("q_proj"), d, q, c.qkv_bias)?,
            k_proj: b.projection(&p("k_proj"), d, kv, c.qkv_bias)?,
            v_proj: b.projection(&p("v_proj"), d, kv, c.qkv_bias)?,
            o_proj: b.projection(&p("o_proj"), q, d, c.o_bias)?,
            q_norm: norm()?,
            k_norm: norm()?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            scale: (hd as f32).powf(-0.5),
            rope: Rope::new(hd, c)?,
        })
    }

    fn forward(&self, x: &Array, mask: AttentionMask<'_>, cache: &mut KvCache) -> Result<Array> {
        let (b, l) = (x.shape()[0] as isize, x.shape()[1] as isize);
        let heads = |proj: &Projection, n: usize, norm: &Option<RmsNorm>| -> Result<Array> {
            let h = proj.forward(x)?.reshape(&[b, l, n as isize, -1])?;
            let h = match norm {
                Some(norm) => norm.forward(&h)?,
                None => h,
            };
            transpose(&h, &[0, 2, 1, 3])
        };
        let offset = cache.offset();
        let q = self
            .rope
            .forward(&heads(&self.q_proj, self.heads, &self.q_norm)?, offset)?;
        let k = self
            .rope
            .forward(&heads(&self.k_proj, self.kv_heads, &self.k_norm)?, offset)?;
        let v = heads(&self.v_proj, self.kv_heads, &None)?;
        let (k, v) = cache.update(k, v)?;
        let o = fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let o = transpose(&o, &[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
        self.o_proj.forward(&o)
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

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Projection,
    up_proj: Projection,
    down_proj: Projection,
}

impl Module for Mlp {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("gate_proj", &mut self.gate_proj);
        p.module("up_proj", &mut self.up_proj);
        p.module("down_proj", &mut self.down_proj);
    }
}

#[derive(Debug, Clone)]
struct Block {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Block {
    fn forward(&self, x: &Array, mask: AttentionMask<'_>, cache: &mut KvCache) -> Result<Array> {
        let h = add(
            x,
            self.self_attn
                .forward(&self.input_layernorm.forward(x)?, mask, cache)?,
        )?;
        let n = self.post_attention_layernorm.forward(&h)?;
        let m = &self.mlp;
        let gated = multiply(silu(&m.gate_proj.forward(&n)?)?, m.up_proj.forward(&n)?)?;
        add(&h, m.down_proj.forward(&gated)?)
    }
}

impl Module for Block {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("self_attn", &mut self.self_attn);
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
                Ok(Block {
                    self_attn: Attention::new(&b, &path("self_attn"), c)?,
                    mlp: Mlp {
                        gate_proj: b.projection(
                            &path("mlp.gate_proj"),
                            c.hidden_size,
                            c.intermediate_size,
                            c.mlp_bias,
                        )?,
                        up_proj: b.projection(
                            &path("mlp.up_proj"),
                            c.hidden_size,
                            c.intermediate_size,
                            c.mlp_bias,
                        )?,
                        down_proj: b.projection(
                            &path("mlp.down_proj"),
                            c.intermediate_size,
                            c.hidden_size,
                            c.mlp_bias,
                        )?,
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
        for (layer, c) in self.model.layers.iter().zip(cache) {
            h = layer.forward(&h, mask, c)?;
        }
        let h = self.model.norm.forward(&h)?;
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
