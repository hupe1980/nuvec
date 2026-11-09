//! Layers, as in MLX's `mlx.nn` (and mlx-lm's LoRA layer). Initialisation draws from this
//! thread's key sequence in MLX's order, so the same seed gives MLX's values.
//!
//! Constructors take no `bool` and return no `Result`: they build lazy arrays, and an
//! invalid size gives failed parameters whose error surfaces at the first boundary. Every
//! `forward` returns an [`Array`] and is `#[track_caller]`: an invalid input gives a failed
//! array whose error (`check()`, `eval()`, …) reports the line that called `forward`.

use std::cell::RefCell;

use nuvec::{Array, DType, OpErrorKind, Param, Quant, QuantMode, Quantized, fast, ops, random};

use crate::Module;

mod conv;
pub use conv::{
    AvgPool1d, AvgPool2d, AvgPool3d, Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d,
    ConvTranspose3d, MaxPool1d, MaxPool2d, MaxPool3d,
};

thread_local! {
    /// The key sequence before and after the bias draw of the last `Linear::new` on this
    /// thread, and the bias it drew: `without_bias` rewinds the sequence when nothing was
    /// drawn since, so `Linear::new(i, o).without_bias()` draws exactly MLX's
    /// `nn.Linear(i, o, bias=False)`.
    static BIAS_DRAW: RefCell<Option<(Array, Array, usize)>> = const { RefCell::new(None) };
}

/// A uniform draw in `[-1/√input, 1/√input)` (MLX's `nn.Linear` initialisation).
#[track_caller]
fn uniform_init(input: i32, shape: &[i32]) -> Array {
    let scale = (1.0 / f64::from(input.max(1))).sqrt();
    random::uniform(-scale, scale, shape, DType::F32, None)
}

/// An affine layer `y = x Wᵀ + b`, with `W` of shape `[output, input]` and `b` of shape
/// `[output]` (MLX's `nn.Linear`).
#[derive(Debug, Clone, Module)]
pub struct Linear {
    /// The weight, `[output_dims, input_dims]`.
    pub weight: Param,
    /// The bias, `[output_dims]`.
    pub bias: Option<Param>,
}

impl Linear {
    /// A layer with a bias, initialised as MLX's `nn.Linear`: weight, then bias, drawn
    /// uniformly from `[-1/√input, 1/√input)`.
    #[track_caller]
    pub fn new(input_dims: i32, output_dims: i32) -> Self {
        let weight = uniform_init(input_dims, &[output_dims, input_dims]);
        let before = random::sequence_key();
        let bias = uniform_init(input_dims, &[output_dims]);
        let after = random::sequence_key();
        BIAS_DRAW.set(Some((before, after, nuvec::graph::id(&bias))));
        Self {
            weight: Param::new(weight),
            bias: Some(Param::new(bias)),
        }
    }

    /// The layer without its bias. Right after [`Linear::new`] it also undoes the bias's
    /// draw from the key sequence, so the initialisation equals MLX's
    /// `nn.Linear(i, o, bias=False)`.
    pub fn without_bias(mut self) -> Self {
        if let Some(bias) = self.bias.take() {
            BIAS_DRAW.with_borrow_mut(|draw| {
                if let Some((before, after, id)) = draw.take()
                    && id == nuvec::graph::id(&bias)
                    && nuvec::graph::id(&random::sequence_key()) == nuvec::graph::id(&after)
                {
                    random::set_sequence_key(before);
                }
            });
        }
        self
    }

    /// `x Wᵀ + b` over the last axis of `x`; with a bias it is one fused `addmm`, as in
    /// MLX.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        match &self.bias {
            Some(b) => b.addmm(x, &self.weight.t(), 1.0, 1.0),
            None => x.matmul(&self.weight.t()),
        }
    }
}

/// A [`Linear`] layer with its weight quantized (MLX's `nn.QuantizedLinear`): `x Wᵀ + b`
/// computed by `quantized_matmul` from the packed weight and its scales (and, for affine,
/// biases), in any [`QuantMode`].
///
/// The packed weight, scales and biases are buffers (never trained, also after
/// `unfreeze`); the bias is a trainable parameter.
#[derive(Debug, Clone, Module)]
pub struct QuantizedLinear {
    /// The packed weight (uint32), `[output_dims, input_dims · bits / 32]`.
    #[tree(buffer)]
    pub weight: Array,
    /// One scale per group of input values, `[output_dims, input_dims / group_size]`
    /// (uint8 codes for the floating-point modes).
    #[tree(buffer)]
    pub scales: Array,
    /// One bias per group of input values, shaped like `scales`: affine only.
    #[tree(buffer)]
    pub biases: Option<Array>,
    /// The layer's bias, `[output_dims]`.
    pub bias: Option<Param>,
    /// How the weight is quantized.
    #[tree(skip)]
    pub quant: Quant,
}

impl QuantizedLinear {
    /// Quantizes `linear`'s weight with MLX's defaults (affine, groups of 64 values, 4
    /// bits); the bias is kept.
    #[track_caller]
    pub fn from_linear(linear: &Linear) -> Self {
        Self::from_linear_with(linear, Quant::default())
    }

    /// Quantizes `linear`'s weight with `quant` (a mode such as `QuantMode::Mxfp4`, or
    /// `Quant::affine(group_size, bits)`); the bias is kept.
    #[track_caller]
    pub fn from_linear_with(linear: &Linear, quant: impl Into<Quant>) -> Self {
        let q = linear.weight.quantize(quant);
        Self {
            weight: q.weight,
            scales: q.scales,
            biases: q.biases,
            bias: linear.bias.clone(),
            quant: q.quant,
        }
    }

    /// The quantized weight.
    pub fn quantized(&self) -> Quantized {
        Quantized::new(
            self.weight.clone(),
            self.scales.clone(),
            self.biases.clone(),
            self.quant,
        )
    }

    /// The dequantized weight, `[output_dims, input_dims]`, in `dtype` (default: the
    /// scales' dtype for affine, bfloat16 for the floating-point modes).
    #[track_caller]
    pub fn dequantized_weight(&self, dtype: impl Into<Option<DType>>) -> Array {
        self.quantized().dequantize(dtype)
    }

    /// `x Wᵀ + b` over the last axis of `x`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        let y = x.quantized_matmul(&self.quantized(), true);
        match &self.bias {
            Some(b) => y.add(b),
            None => y,
        }
    }
}

/// A lookup table of `dims`-dimensional vectors for integer ids (MLX's `nn.Embedding`).
#[derive(Debug, Clone, Module)]
pub struct Embedding {
    /// The table, `[num_embeddings, dims]`.
    pub weight: Param,
}

impl Embedding {
    /// A table drawn from a normal distribution with standard deviation `1/√dims`, as
    /// MLX.
    #[track_caller]
    pub fn new(num_embeddings: i32, dims: i32) -> Self {
        let scale = (1.0 / f64::from(dims.max(1))).sqrt();
        let weight = random::normal([num_embeddings, dims], DType::F32, 0.0, scale, None);
        Self {
            weight: Param::new(weight),
        }
    }

    /// The rows for the ids in `x`: shape `x.shape + [dims]`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        self.weight.take(x, 0)
    }

    /// `x Wᵀ`: the table used as an output projection (tied embeddings).
    #[track_caller]
    pub fn as_linear(&self, x: &Array) -> Array {
        x.matmul(&self.weight.t())
    }
}

/// An [`Embedding`] with its table quantized (MLX's `nn.QuantizedEmbedding`): rows are
/// gathered packed and dequantized; the packed table, scales and biases are buffers. In the
/// floating-point modes the rows come out in bfloat16, as MLX's.
#[derive(Debug, Clone, Module)]
pub struct QuantizedEmbedding {
    /// The packed table (uint32), `[num_embeddings, dims · bits / 32]`.
    #[tree(buffer)]
    pub weight: Array,
    /// One scale per group of values, `[num_embeddings, dims / group_size]`.
    #[tree(buffer)]
    pub scales: Array,
    /// One bias per group of values, shaped like `scales`: affine only.
    #[tree(buffer)]
    pub biases: Option<Array>,
    /// How the table is quantized.
    #[tree(skip)]
    pub quant: Quant,
}

impl QuantizedEmbedding {
    /// Quantizes `embedding`'s table with MLX's defaults (affine, groups of 64, 4 bits).
    #[track_caller]
    pub fn from_embedding(embedding: &Embedding) -> Self {
        Self::from_embedding_with(embedding, Quant::default())
    }

    /// Quantizes `embedding`'s table with `quant`.
    #[track_caller]
    pub fn from_embedding_with(embedding: &Embedding, quant: impl Into<Quant>) -> Self {
        let q = embedding.weight.quantize(quant);
        Self {
            weight: q.weight,
            scales: q.scales,
            biases: q.biases,
            quant: q.quant,
        }
    }

    /// The quantized table.
    pub fn quantized(&self) -> Quantized {
        Quantized::new(
            self.weight.clone(),
            self.scales.clone(),
            self.biases.clone(),
            self.quant,
        )
    }

    /// The rows for the ids in `x`, dequantized: shape `x.shape + [dims]`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        let rows = Quantized::new(
            self.weight.take(x, 0),
            self.scales.take(x, 0),
            self.biases.as_ref().map(|b| b.take(x, 0)),
            self.quant,
        );
        rows.dequantize(None)
    }

    /// `x Wᵀ` with the quantized table (tied embeddings).
    #[track_caller]
    pub fn as_linear(&self, x: &Array) -> Array {
        x.quantized_matmul(&self.quantized(), true)
    }
}

/// Root-mean-square normalisation over the last axis with a learned scale (MLX's
/// `nn.RMSNorm`, the fused `fast::rms_norm`).
#[derive(Debug, Clone, Module)]
pub struct RmsNorm {
    /// The scale, `[dims]`, initially ones.
    pub weight: Param,
    /// Added to the mean square (MLX's default `1e-5`).
    #[tree(skip)]
    pub eps: f32,
}

impl RmsNorm {
    /// A layer over `dims` features with MLX's default `eps` (`1e-5`).
    #[track_caller]
    pub fn new(dims: i32) -> Self {
        Self {
            weight: Param::new(ops::ones([dims], DType::F32)),
            eps: 1e-5,
        }
    }

    /// Sets `eps`.
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// The normalised `x`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        fast::rms_norm(x, &self.weight, self.eps)
    }
}

/// Layer normalisation over the last axis, optionally with a learned scale and bias
/// (MLX's `nn.LayerNorm`, the fused `fast::layer_norm`).
#[derive(Debug, Clone, Module)]
pub struct LayerNorm {
    /// The scale, `[dims]` (ones), if affine.
    pub weight: Option<Param>,
    /// The shift, `[dims]` (zeros), if affine with a bias.
    pub bias: Option<Param>,
    /// Added to the variance (MLX's default `1e-5`).
    #[tree(skip)]
    pub eps: f32,
}

impl LayerNorm {
    /// A layer over `dims` features with MLX's defaults: affine with a bias, `eps = 1e-5`.
    #[track_caller]
    pub fn new(dims: i32) -> Self {
        Self {
            weight: Some(Param::new(ops::ones([dims], DType::F32))),
            bias: Some(Param::new(ops::zeros([dims], DType::F32))),
            eps: 1e-5,
        }
    }

    /// Sets `eps`.
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// With (`true`, the default) or without a learned scale and shift (MLX's `affine`).
    pub fn affine(mut self, affine: bool) -> Self {
        if !affine {
            self.weight = None;
            self.bias = None;
        }
        self
    }

    /// Without the learned shift (MLX's `bias=False`).
    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }

    /// The normalised `x`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        fast::layer_norm(
            x,
            self.weight.as_ref().map(Param::value),
            self.bias.as_ref().map(Param::value),
            self.eps,
        )
    }
}

/// Rotary position embedding of the first `dims` features (MLX's `nn.RoPE`, the fused
/// `fast::rope`), optionally with a frequency scaling ([`Rope::scaled`]: mlx-lm's
/// `Llama3RoPE`, `YarnRoPE`, `SuScaledRoPE`, `ProportionalRoPE`). No parameters: the
/// frequencies are computed from the settings, never loaded or saved.
#[derive(Debug, Clone, Module)]
pub struct Rope {
    /// Rotated features.
    #[tree(skip)]
    pub dims: i32,
    /// Rotate consecutive pairs (`true`) or the two halves (`false`, MLX's default).
    #[tree(skip)]
    pub traditional: bool,
    /// The frequency base (MLX's default 10000).
    #[tree(skip)]
    pub base: f32,
    /// Scale applied to positions.
    #[tree(skip)]
    pub scale: f32,
    /// The frequency scaling.
    #[tree(skip)]
    scaling: crate::RopeScaling,
    /// The scaling's per-pair frequencies (mlx-lm's `_freqs`), if it has them.
    #[tree(skip)]
    freqs: Option<Array>,
}

impl Rope {
    /// RoPE over `dims` features with MLX's defaults (halves, base 10000, scale 1).
    pub fn new(dims: i32) -> Self {
        Self {
            dims,
            traditional: false,
            base: 10000.0,
            scale: 1.0,
            scaling: crate::RopeScaling::None,
            freqs: None,
        }
    }

    /// RoPE over `dims` features with `base` and a frequency `scaling`, as mlx-lm's scaled
    /// RoPE layers (for [`RopeScaling::Proportional`](crate::RopeScaling::Proportional)
    /// `dims` is the whole head's).
    ///
    /// ```
    /// use nuvec_nn::{Rope, RopeScaling, Yarn};
    ///
    /// // gpt-oss: YaRN over 64 features, base 150000, factor 32 over 4096 positions.
    /// let yarn = Yarn { factor: 32.0, original_max_position_embeddings: 4096.0, ..Yarn::default() };
    /// let rope = Rope::scaled(64, 150_000.0, &RopeScaling::Yarn(yarn));
    /// assert!((rope.mscale() - 1.3466).abs() < 1e-4);
    /// ```
    pub fn scaled(dims: i32, base: f32, scaling: &crate::RopeScaling) -> Self {
        Self {
            dims,
            traditional: false,
            base,
            scale: scaling.position_scale(),
            scaling: scaling.clone(),
            freqs: scaling.frequencies(dims, base),
        }
    }

    /// The input scale the scaling applies to the rotated features (YaRN, LongRoPE; else
    /// 1). Callers that rotate queries and keys with [`Rope::rotation`] (one fused kernel
    /// each) instead multiply the attention scale by `mscale()²`, which is the same up to
    /// rounding when the rotated features are the whole head. In half precision mlx-lm
    /// multiplies q and k by the mscale rounded to their dtype, which the fold does not
    /// reproduce: nuvec-lm multiplies explicitly there and folds only in float32.
    pub fn mscale(&self) -> f32 {
        self.scaling.mscale()
    }

    /// The rotation for the fused ops (`fast::rope`, `fast::norm_rope`), without the input
    /// scale ([`Rope::mscale`]).
    pub fn rotation(&self) -> fast::Rotation<'_> {
        fast::Rotation {
            dims: self.dims,
            traditional: self.traditional,
            base: self.freqs.is_none().then_some(self.base),
            scale: self.scale,
            freqs: self.freqs.as_ref(),
        }
    }

    /// Rotates consecutive pairs of features (MLX's `traditional=True`).
    pub fn traditional(mut self) -> Self {
        self.traditional = true;
        self
    }

    /// Sets the frequency base (recomputing a scaling's frequencies).
    pub fn base(mut self, base: f32) -> Self {
        self.base = base;
        self.freqs = self.scaling.frequencies(self.dims, base);
        self
    }

    /// Sets the scale applied to positions.
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// `x` (`[..., sequence, features]`) rotated for positions `offset`, `offset + 1`, …,
    /// its rotated features first multiplied by [`Rope::mscale`] (as mlx-lm).
    #[track_caller]
    pub fn forward(&self, x: &Array, offset: i32) -> Array {
        let m = self.mscale();
        let x = if m == 1.0 {
            x.clone()
        } else if x.shape().last() == Some(&self.dims) {
            x.multiply(m)
        } else {
            let head = (.., ..self.dims);
            x.clone().at(head).set(x.i(head).multiply(m))
        };
        let r = self.rotation();
        fast::rope(
            &x,
            r.dims,
            r.traditional,
            r.base,
            r.scale,
            &Array::from(offset),
            r.freqs,
        )
    }
}

/// Randomly zeroes elements with probability `p` during training and scales the rest by
/// `1/(1 − p)`; the identity in evaluation mode (MLX's `nn.Dropout`). The mask is
/// `bernoulli(1 − p)`, drawn from the thread's key sequence (bit for bit MLX's for a seed)
/// or from an explicit key ([`Dropout::forward_with_key`]).
///
/// The mode is switched with [`Module::train`] on the whole model, and is part of the
/// model's structure, so a compiled step retraces when it changes.
#[derive(Debug, Clone, Module)]
pub struct Dropout {
    /// The probability of zeroing an element, `0 ≤ p < 1`.
    #[tree(skip)]
    pub p: f32,
    /// Training (`true`, the default) or evaluation mode.
    #[tree(mode)]
    pub training: bool,
}

impl Dropout {
    /// Dropout with probability `p` (`0 ≤ p < 1`; another value makes `forward` return a
    /// failed array), in training mode.
    pub fn new(p: f32) -> Self {
        Self { p, training: true }
    }

    /// `x` with dropout (training mode) or unchanged (evaluation mode).
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        self.apply(x, None)
    }

    /// [`Dropout::forward`] with the mask drawn from `key`.
    #[track_caller]
    pub fn forward_with_key(&self, x: &Array, key: &Array) -> Array {
        self.apply(x, Some(key))
    }

    #[track_caller]
    fn apply(&self, x: &Array, key: Option<&Array>) -> Array {
        if !(0.0..1.0).contains(&self.p) {
            return Array::failure(
                "dropout",
                OpErrorKind::InvalidArgument(format!(
                    "the probability must be in [0, 1), got {}",
                    self.p
                )),
            );
        }
        if !self.training || self.p == 0.0 {
            return x.clone();
        }
        // MLX computes `p_1 = 1 - p` and `1 / p_1` on Python floats: use `p`'s shortest
        // decimal value as an `f64` (0.1f32 is 0.1), so both round as MLX's.
        let p: f64 = self.p.to_string().parse().unwrap_or(f64::from(self.p));
        let keep = 1.0 - p;
        // MLX: `(mask * x) * (1 / p_1)`.
        let mask = random::bernoulli(keep, x.shape(), key);
        mask.multiply(x).multiply(1.0 / keep)
    }
}

/// The layer a [`LoraLinear`] adapts.
#[derive(Debug, Clone, Module)]
pub enum LoraBase {
    /// A full-precision layer.
    Linear(Linear),
    /// A quantized layer (QLoRA).
    Quantized(QuantizedLinear),
}

impl LoraBase {
    /// `[output_dims, input_dims]` of the adapted layer.
    fn dims(&self) -> (i32, i32) {
        match self {
            Self::Linear(l) => {
                let s = l.weight.shape();
                (
                    s.first().copied().unwrap_or(0),
                    s.get(1).copied().unwrap_or(0),
                )
            }
            Self::Quantized(q) => {
                let s = q.weight.shape();
                let packed = s.get(1).copied().unwrap_or(0);
                (
                    s.first().copied().unwrap_or(0),
                    packed * 32 / q.quant.bits.max(1),
                )
            }
        }
    }

    #[track_caller]
    fn forward(&self, x: &Array) -> Array {
        match self {
            Self::Linear(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }
}

/// A frozen linear layer plus a trainable low-rank update (mlx-lm's `LoRALinear`):
/// `y = base(x) + scale · (dropout(x) A) B`, with `A` of shape `[input, r]` and `B` of shape
/// `[r, output]` (zeros at first, so the layer starts as its base). Names follow mlx-lm:
/// `linear.weight`, `lora_a`, `lora_b`.
///
/// With a magnitude `m` ([`LoraLinear::dora`]), it is mlx-lm's `DoRALinear`: the output is
/// rescaled per output feature by `m / ‖W + scale · Bᵀ Aᵀ‖` (the norm without gradient).
#[derive(Debug, Clone, Module)]
pub struct LoraLinear {
    /// The adapted layer (frozen).
    #[tree(rename = "linear")]
    pub base: LoraBase,
    /// `[input_dims, r]`.
    pub lora_a: Param,
    /// `[r, output_dims]`.
    pub lora_b: Param,
    /// The update's scale (mlx-lm's default 20).
    #[tree(skip)]
    pub scale: f32,
    /// Dropout on the update's input (mlx-lm's LoRA dropout; 0 by default).
    pub dropout: Dropout,
    /// DoRA's learned magnitude per output feature, `[output_dims]` (float32).
    pub m: Option<Param>,
}

impl LoraLinear {
    /// Adapts `base` with rank `r` and mlx-lm's default scale 20, as mlx-lm's
    /// `LoRALinear.from_base` (including its random draws: `A` is uniform in
    /// `[-1/√input, 1/√input)`). The base is frozen.
    #[track_caller]
    pub fn from_base(base: impl Into<LoraBase>, r: i32) -> Self {
        let mut base = base.into();
        let (output_dims, input_dims) = base.dims();
        // mlx-lm builds (and replaces) a fresh `nn.Linear(input, output, bias=False)`
        // first: draw its weight so the key sequence matches. Nothing is computed for it.
        let _ = uniform_init(input_dims, &[output_dims, input_dims]);
        let lora_a = uniform_init(input_dims, &[input_dims, r]);
        let lora_b = ops::zeros([r, output_dims], DType::F32);
        // Every parameter of the base is frozen (buffers are never trained).
        let _ = base.freeze(|_| true);
        Self {
            base,
            lora_a: Param::new(lora_a),
            lora_b: Param::new(lora_b),
            scale: 20.0,
            dropout: Dropout::new(0.0),
            m: None,
        }
    }

    /// Weight-decomposed adaptation (DoRA, mlx-lm's `DoRALinear`): adds the magnitude `m`,
    /// initialised to the row norms of the base weight (dequantized) in float32.
    #[track_caller]
    pub fn dora(mut self) -> Self {
        let w = self.base_weight().astype(DType::F32);
        self.m = Some(Param::new(row_norms(&w)));
        self
    }

    /// The base weight `[output, input]`, dequantized.
    #[track_caller]
    fn base_weight(&self) -> Array {
        match &self.base {
            LoraBase::Linear(l) => l.weight.value().clone(),
            LoraBase::Quantized(q) => q.dequantized_weight(None),
        }
    }

    /// `scale · Bᵀ Aᵀ`, `[output, input]`.
    #[track_caller]
    fn delta(&self) -> Array {
        self.lora_b
            .t()
            .multiply(f64::from(self.scale))
            .matmul(&self.lora_a.t())
    }

    /// Sets the update's scale.
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Drops the update's input with probability `p` in training mode (mlx-lm's LoRA
    /// dropout; the base sees the input unchanged).
    pub fn dropout(mut self, p: f32) -> Self {
        self.dropout = Dropout::new(p);
        self
    }

    /// Rank-stabilised scaling (rsLoRA): treating `scale` as `α/r`, the update is scaled
    /// by `α/√r = scale · √r` instead.
    pub fn rslora(mut self) -> Self {
        let r = self.lora_a.shape().get(1).copied().unwrap_or(1).max(1);
        self.scale *= (r as f32).sqrt();
        self
    }

    /// `base(x) + scale · (dropout(x) A) B`, the update cast to `x`'s dtype; with DoRA,
    /// `(m / stop_gradient(‖W + scale · Bᵀ Aᵀ‖)) · (x Wᵀ + scale · (dropout(x) A) B) + bias`.
    #[track_caller]
    pub fn forward(&self, x: &Array) -> Array {
        let update = lora_update(
            &self.dropout.forward(x),
            self.lora_a.value(),
            self.lora_b.value(),
            self.scale,
        );
        let Some(m) = &self.m else {
            return self.base.forward(x).add(update);
        };
        let w = self.base_weight();
        let (y, bias) = match &self.base {
            LoraBase::Linear(l) => (x.matmul(&w.t()), l.bias.as_ref()),
            LoraBase::Quantized(q) => (x.quantized_matmul(&q.quantized(), true), q.bias.as_ref()),
        };
        let adapted = w.add(self.delta());
        let denom = row_norms(&adapted).stop_gradient();
        let out = m
            .value()
            .divide(&denom)
            .astype(x.dtype())
            .multiply(y.add(update));
        match bias {
            Some(b) => out.add(b.value()),
            None => out,
        }
    }

    /// The base layer with the update merged into its weight (mlx-lm's `fuse` with
    /// `dequantize`): `W + scale · Bᵀ Aᵀ`. A floating-point quantized base is dequantized to
    /// the adapter's dtype (not MLX's bfloat16 default), an affine one to its scales' dtype.
    #[track_caller]
    pub fn fuse(&self) -> Linear {
        let (weight, bias) = match &self.base {
            LoraBase::Linear(l) => (l.weight.value().clone(), l.bias.clone()),
            LoraBase::Quantized(q) => {
                let dtype = (q.quant.mode != QuantMode::Affine).then(|| self.lora_a.dtype());
                (q.dequantized_weight(dtype), q.bias.clone())
            }
        };
        let delta = self.delta().astype(weight.dtype());
        let weight = weight.add(&delta);
        // DoRA: each row rescaled to its learned magnitude (mlx-lm's `DoRALinear.fuse`).
        let weight = match &self.m {
            Some(m) => m
                .value()
                .divide(row_norms(&weight))
                .expand_dims(-1)
                .multiply(&weight)
                .astype(weight.dtype()),
            None => weight,
        };
        Linear {
            weight: Param::new(weight),
            bias: bias.map(|b| Param::new(b.into_inner())),
        }
    }
}

/// LoRA's update `(scale · (x A) B)` in `x`'s dtype, `x` promoted to the adapters' dtype as
/// `x.matmul(a)` does, with its own backward pass:
/// - `x` is kept in its dtype (promoted again in the backward pass), not its promoted copy,
///   and `x A` is recomputed from it (referring to the forward pass's results would keep
///   the update itself alive);
/// - the input's gradient is evaluated after the adapters' gradients, so each layer's
///   `[.., output]` cotangent is freed before the backward pass moves to the layer below
///   (otherwise every layer's would be alive until its adapters' gradients are reached).
///
/// The values and gradients equal the composition's.
#[track_caller]
fn lora_update(x: &Array, a: &Array, b: &Array, scale: f32) -> Array {
    if x.ndim() == 0 || a.ndim() != 2 || b.ndim() != 2 {
        let z = x.matmul(a).matmul(b);
        return z.multiply(f64::from(scale)).astype(x.dtype());
    }
    let scale = f64::from(scale);
    let f = nuvec::custom_function_flat(move |xs| {
        let (x, a, b) = (&xs[0], &xs[1], &xs[2]);
        Ok(vec![
            x.matmul(a).matmul(b).multiply(scale).astype(x.dtype()),
        ])
    })
    .with_vjp(move |p, _, c| {
        let (x, a, b, g) = (&p[0], &p[1], &p[2], &c[0]);
        let (d, out) = (x.shape()[x.ndim() - 1], b.shape()[1]);
        let wide = a.dtype();
        let g = g.astype(wide).multiply(scale).reshape([-1, out]);
        // Promoted after the cotangent exists, so `compile` cannot merge it with the forward
        // pass's promotion.
        let xw = ops::depends(x, std::slice::from_ref(&g))
            .reshape([-1, d])
            .astype(wide);
        let db = xw.matmul(a).t().matmul(&g);
        let ddown = g.matmul(&b.t());
        let da = xw.t().matmul(&ddown);
        let mut shape = x.shape().to_vec();
        shape[x.ndim() - 1] = d;
        let dx = ddown.matmul(&a.t()).astype(x.dtype()).reshape(shape);
        let dx = ops::depends(&dx, &[da.clone(), db.clone()]);
        Ok(vec![dx, da, db])
    });
    match f.call(&[x.clone(), a.clone(), b.clone()]) {
        Ok(mut out) => out.swap_remove(0),
        Err(e) => Array::failure("lora", OpErrorKind::InvalidArgument(e.to_string())),
    }
}

/// The Euclidean norm of each row of `w` (`mx.linalg.norm(w, axis=1)`).
#[track_caller]
fn row_norms(w: &Array) -> Array {
    w.multiply(w).sum(1).sqrt()
}

impl From<Linear> for LoraBase {
    fn from(l: Linear) -> Self {
        Self::Linear(l)
    }
}

impl From<QuantizedLinear> for LoraBase {
    fn from(q: QuantizedLinear) -> Self {
        Self::Quantized(q)
    }
}
