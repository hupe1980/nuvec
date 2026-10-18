//! Layers, as in MLX's `mlx.nn` (and mlx-lm's LoRA layer). Initialisation draws from this
//! thread's key sequence in MLX's order, so the same seed gives MLX's values.

use nuvec::{
    Array, DType, Result, add, dequantize, fast, matmul, multiply, quantize, quantized_matmul,
    random, take,
};

use crate::{Module, Parameters};

/// An affine layer `y = x Wᵀ + b`, with `W` of shape `[output, input]` and `b` of shape
/// `[output]` (MLX's `nn.Linear`).
#[derive(Debug, Clone)]
pub struct Linear {
    /// The weight, `[output_dims, input_dims]`.
    pub weight: Array,
    /// The bias, `[output_dims]`.
    pub bias: Option<Array>,
}

impl Linear {
    /// A layer initialised as MLX's `nn.Linear`: weight, then bias, drawn uniformly from
    /// `[-1/√input, 1/√input)`.
    pub fn new(input_dims: usize, output_dims: usize, bias: bool) -> Result<Self> {
        let scale = (1.0 / input_dims as f64).sqrt();
        let draw = |shape: &[usize]| random::uniform(-scale, scale, shape, DType::F32, None);
        let weight = draw(&[output_dims, input_dims])?;
        let bias = if bias {
            Some(draw(&[output_dims])?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// `x Wᵀ + b` over the last axis of `x`; with a bias it is one fused `addmm`, as in
    /// MLX.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match &self.bias {
            Some(b) => nuvec::addmm(b, x, &self.weight.t()?, 1.0, 1.0),
            None => matmul(x, &self.weight.t()?),
        }
    }
}

impl Module for Linear {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.param("weight", &mut self.weight);
        p.optional("bias", &mut self.bias);
    }
}

/// A [`Linear`] layer with its weight quantized (MLX's `nn.QuantizedLinear`): `x Wᵀ + b`
/// computed by `quantized_matmul` from the packed weight, its scales and biases.
///
/// The packed weight, scales and biases are frozen; the bias stays trainable, as in MLX.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// The packed weight (uint32), `[output_dims, input_dims · bits / 32]`.
    pub weight: Array,
    /// One scale per group of input values, `[output_dims, input_dims / group_size]`.
    pub scales: Array,
    /// One bias per group of input values, shaped like `scales`.
    pub biases: Array,
    /// The layer's bias, `[output_dims]`.
    pub bias: Option<Array>,
    /// Input values per scale and bias.
    pub group_size: usize,
    /// Bits per weight.
    pub bits: u32,
}

impl QuantizedLinear {
    /// Quantizes `linear`'s weight in groups of `group_size` (32, 64 or 128) input values
    /// with `bits` (1–6 or 8) bits each; the bias is kept. MLX's defaults are 64 and 4.
    pub fn from_linear(linear: &Linear, group_size: usize, bits: u32) -> Result<Self> {
        let (weight, scales, biases) = quantize(&linear.weight, group_size, bits)?;
        Ok(Self {
            weight,
            scales,
            biases,
            bias: linear.bias.clone(),
            group_size,
            bits,
        })
    }

    /// The dequantized weight, `[output_dims, input_dims]`.
    pub fn dequantized_weight(&self) -> Result<Array> {
        dequantize(
            &self.weight,
            &self.scales,
            &self.biases,
            self.group_size,
            self.bits,
        )
    }

    /// `x Wᵀ + b` over the last axis of `x`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )?;
        match &self.bias {
            Some(b) => add(&y, b),
            None => Ok(y),
        }
    }
}

impl Module for QuantizedLinear {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.frozen("weight", &mut self.weight);
        p.frozen("scales", &mut self.scales);
        p.frozen("biases", &mut self.biases);
        p.optional("bias", &mut self.bias);
    }
}

/// A lookup table of `dims`-dimensional vectors for integer ids (MLX's `nn.Embedding`).
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The table, `[num_embeddings, dims]`.
    pub weight: Array,
}

impl Embedding {
    /// A table drawn from a normal distribution with standard deviation `1/√dims`, as
    /// MLX.
    pub fn new(num_embeddings: usize, dims: usize) -> Result<Self> {
        let scale = (1.0 / dims as f64).sqrt();
        let weight = random::normal(&[num_embeddings, dims], DType::F32, None, Some(scale), None)?;
        Ok(Self { weight })
    }

    /// The rows for the ids in `x`: shape `x.shape + [dims]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        take(&self.weight, x, Some(0))
    }

    /// `x Wᵀ`: the table used as an output projection (tied embeddings).
    pub fn as_linear(&self, x: &Array) -> Result<Array> {
        matmul(x, &self.weight.t()?)
    }
}

impl Module for Embedding {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.param("weight", &mut self.weight);
    }
}

/// An [`Embedding`] with its table quantized (MLX's `nn.QuantizedEmbedding`): rows are
/// gathered packed and dequantized; the packed table, scales and biases are frozen.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    /// The packed table (uint32), `[num_embeddings, dims · bits / 32]`.
    pub weight: Array,
    /// One scale per group of values, `[num_embeddings, dims / group_size]`.
    pub scales: Array,
    /// One bias per group of values, shaped like `scales`.
    pub biases: Array,
    /// Values per scale and bias.
    pub group_size: usize,
    /// Bits per value.
    pub bits: u32,
}

impl QuantizedEmbedding {
    /// Quantizes `embedding`'s table in groups of `group_size` values with `bits` bits.
    pub fn from_embedding(embedding: &Embedding, group_size: usize, bits: u32) -> Result<Self> {
        let (weight, scales, biases) = quantize(&embedding.weight, group_size, bits)?;
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits,
        })
    }

    /// The rows for the ids in `x`, dequantized: shape `x.shape + [dims]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        dequantize(
            &take(&self.weight, x, Some(0))?,
            &take(&self.scales, x, Some(0))?,
            &take(&self.biases, x, Some(0))?,
            self.group_size,
            self.bits,
        )
    }

    /// `x Wᵀ` with the quantized table (tied embeddings).
    pub fn as_linear(&self, x: &Array) -> Result<Array> {
        quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )
    }
}

impl Module for QuantizedEmbedding {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.frozen("weight", &mut self.weight);
        p.frozen("scales", &mut self.scales);
        p.frozen("biases", &mut self.biases);
    }
}

/// Root-mean-square normalisation over the last axis with a learned scale (MLX's
/// `nn.RMSNorm`, the fused `fast::rms_norm`).
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// The scale, `[dims]`, initially ones.
    pub weight: Array,
    /// Added to the mean square.
    pub eps: f32,
}

impl RmsNorm {
    /// A layer over `dims` features (MLX's default `eps` is `1e-5`).
    pub fn new(dims: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            weight: nuvec::ones(&[dims], DType::F32)?,
            eps,
        })
    }

    /// The normalised `x`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        fast::rms_norm(x, Some(&self.weight), self.eps)
    }
}

impl Module for RmsNorm {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.param("weight", &mut self.weight);
    }
}

/// Layer normalisation over the last axis, optionally with a learned scale and bias
/// (MLX's `nn.LayerNorm`, the fused `fast::layer_norm`).
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// The scale, `[dims]` (ones), if affine.
    pub weight: Option<Array>,
    /// The shift, `[dims]` (zeros), if affine with a bias.
    pub bias: Option<Array>,
    /// Added to the variance.
    pub eps: f32,
}

impl LayerNorm {
    /// A layer over `dims` features: `affine` adds a scale, and `bias` (with `affine`) a
    /// shift. MLX's defaults: `eps = 1e-5`, affine with a bias.
    pub fn new(dims: usize, eps: f32, affine: bool, bias: bool) -> Result<Self> {
        let weight = affine
            .then(|| nuvec::ones(&[dims], DType::F32))
            .transpose()?;
        let bias = (affine && bias)
            .then(|| nuvec::zeros(&[dims], DType::F32))
            .transpose()?;
        Ok(Self { weight, bias, eps })
    }

    /// The normalised `x`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        fast::layer_norm(x, self.weight.as_ref(), self.bias.as_ref(), self.eps)
    }
}

impl Module for LayerNorm {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.optional("weight", &mut self.weight);
        p.optional("bias", &mut self.bias);
    }
}

/// Rotary position embedding of the first `dims` features (MLX's `nn.RoPE`, the fused
/// `fast::rope`). No parameters.
#[derive(Debug, Clone)]
pub struct Rope {
    /// Rotated features.
    pub dims: usize,
    /// Rotate consecutive pairs (`true`) or the two halves (`false`, MLX's default).
    pub traditional: bool,
    /// The frequency base (MLX's default 10000).
    pub base: f32,
    /// Scale applied to positions.
    pub scale: f32,
}

impl Rope {
    /// RoPE over `dims` features with MLX's defaults (halves, base 10000, scale 1).
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            traditional: false,
            base: 10000.0,
            scale: 1.0,
        }
    }

    /// `x` (`[..., sequence, features]`) rotated for positions `offset`, `offset + 1`, ….
    pub fn forward(&self, x: &Array, offset: usize) -> Result<Array> {
        fast::rope(
            x,
            self.dims,
            self.traditional,
            Some(self.base),
            self.scale,
            &Array::scalar(offset as i32),
            None,
        )
    }
}

impl Module for Rope {
    fn visit(&mut self, _: &mut Parameters<'_>) {}
}

/// The layer a [`LoraLinear`] adapts.
#[derive(Debug, Clone)]
pub enum LoraBase {
    /// A full-precision layer.
    Linear(Linear),
    /// A quantized layer (QLoRA).
    Quantized(QuantizedLinear),
}

/// A frozen linear layer plus a trainable low-rank update (mlx-lm's `LoRALinear`):
/// `y = base(x) + scale · (x A) B`, with `A` of shape `[input, r]` and `B` of shape
/// `[r, output]` (zeros at first, so the layer starts as its base).
#[derive(Debug, Clone)]
pub struct LoraLinear {
    /// The adapted layer (frozen).
    pub base: LoraBase,
    /// `[input_dims, r]`.
    pub lora_a: Array,
    /// `[r, output_dims]`.
    pub lora_b: Array,
    /// The update's scale (mlx-lm's default 20).
    pub scale: f32,
}

impl LoraLinear {
    /// Adapts `base` with rank `r`, as mlx-lm's `LoRALinear.from_base` (including its
    /// random draws: `A` is uniform in `[-1/√input, 1/√input)`).
    pub fn from_base(base: LoraBase, r: usize, scale: f32) -> Result<Self> {
        let (output_dims, input_dims) = match &base {
            LoraBase::Linear(l) => (l.weight.shape()[0], l.weight.shape()[1]),
            LoraBase::Quantized(q) => (
                q.weight.shape()[0],
                q.weight.shape()[1] * 32 / q.bits as usize,
            ),
        };
        // mlx-lm builds (and replaces) a fresh linear layer first: draw its weight so the
        // key sequence matches. Nothing is computed for it.
        Linear::new(input_dims, output_dims, false)?;
        let bound = 1.0 / (input_dims as f64).sqrt();
        let lora_a = random::uniform(-bound, bound, &[input_dims, r], DType::F32, None)?;
        let lora_b = nuvec::zeros(&[r, output_dims], DType::F32)?;
        Ok(Self {
            base,
            lora_a,
            lora_b,
            scale,
        })
    }

    /// `base(x) + scale · (x A) B`, the update cast to `x`'s dtype.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = match &self.base {
            LoraBase::Linear(l) => l.forward(x)?,
            LoraBase::Quantized(q) => q.forward(x)?,
        };
        let z = matmul(&matmul(x, &self.lora_a)?, &self.lora_b)?;
        add(&y, multiply(z, f64::from(self.scale))?.astype(x.dtype())?)
    }

    /// The base layer with the update merged into its weight (mlx-lm's `fuse` with
    /// `dequantize`): `W + scale · Bᵀ Aᵀ`.
    pub fn fuse(&self) -> Result<Linear> {
        let (weight, bias) = match &self.base {
            LoraBase::Linear(l) => (l.weight.clone(), l.bias.clone()),
            LoraBase::Quantized(q) => (q.dequantized_weight()?, q.bias.clone()),
        };
        let delta = matmul(
            &multiply(f64::from(self.scale), self.lora_b.t()?)?,
            &self.lora_a.t()?,
        )?
        .astype(weight.dtype())?;
        Ok(Linear {
            weight: add(&weight, &delta)?,
            bias,
        })
    }
}

impl Module for LoraLinear {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        // mlx-lm's names: `linear.*` for the base, frozen.
        let base = &mut self.base;
        p.freeze(|p| match base {
            LoraBase::Linear(l) => p.module("linear", l),
            LoraBase::Quantized(q) => p.module("linear", q),
        });
        p.param("lora_a", &mut self.lora_a);
        p.param("lora_b", &mut self.lora_b);
    }
}
