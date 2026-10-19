//! Fused operations with dedicated GPU kernels, as MLX's `mx.fast`: [`rms_norm`],
//! [`layer_norm`], [`rope`], [`cross_entropy`], [`scaled_dot_product_attention`] and
//! [`gated_delta`] (mlx-lm's Gated DeltaNet kernels) (and,
//! through [`crate::softmax`] and
//! [`crate::logsumexp`] over the last axis, MLX's softmax and log-sum-exp kernels).
//!
//! Each operation is one GPU kernel (MLX's, ported) with an equivalent graph of ordinary
//! operations, its *fallback*. As in MLX, the fallback runs on the CPU stream, and it
//! defines the forward-mode derivative and batching; reverse mode uses dedicated kernels
//! where MLX has them.
//!
//! ```
//! use nuvec_core::{Array, Result, fast};
//!
//! # fn main() -> Result<()> {
//! let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
//! let y = fast::rms_norm(&x, None, 1e-5)?; // x / sqrt(mean(x²) + eps), per row
//! let v = y.to_vec::<f32>()?;
//! assert!((v[0] - 1.0 / 2.5f32.sqrt()).abs() < 1e-5);
//! # Ok(())
//! # }
//! ```

use crate::cast::Canon;
use crate::kernels::Operand;
use crate::ops::{
    self, add, arange, astype, broadcast_to, cos, divide, equal, exp, expand_dims, filled, less,
    log, max, mean, multiply, reshape, sin, slice, sqrt, squeeze, stop_gradient, subtract, sum,
    where_,
};
use crate::primitive::Primitive;
use crate::{
    Array, Category, DType, Device, DeviceKind, Error, Result, SliceArg, Stream, default_stream,
    set_default_stream,
};
use crate::{stack, zeros};

pub use crate::kernel::{KernelCall, KernelConfig, MetalKernel, TemplateArg};

/// A fused operation (see the [module documentation](self)).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FastOp {
    /// `(x, w)`: `x / sqrt(mean(x²) + eps) · w` over the last axis; `w` is 1-D or a scalar 1.
    RmsNorm {
        /// `f32::to_bits` of epsilon.
        eps: u32,
    },
    /// `(x, w, g)`: the gradients of [`FastOp::RmsNorm`] for `x` and `w`.
    RmsNormVjp {
        /// `f32::to_bits` of epsilon.
        eps: u32,
    },
    /// `(x, w, b)`: `(x - mean) / sqrt(var + eps) · w + b` over the last axis; `w` and `b`
    /// are 1-D or the scalars 1 and 0.
    LayerNorm {
        /// `f32::to_bits` of epsilon.
        eps: u32,
    },
    /// `(x, w, b, g)`: the gradients of [`FastOp::LayerNorm`] for `x`, `w` and `b`.
    LayerNormVjp {
        /// `f32::to_bits` of epsilon.
        eps: u32,
    },
    /// `(x, offset[, freqs])`: rotary position embeddings of the first `dims` features.
    Rope {
        /// Rotated features (even).
        dims: usize,
        /// Rotate interleaved pairs instead of the two halves.
        traditional: bool,
        /// `f32::to_bits` of the frequency base (unused with `freqs`).
        base: u32,
        /// `f32::to_bits` of the position scale.
        scale: u32,
        /// Rotate forwards (`false`: the inverse rotation, the gradient).
        forward: bool,
    },
    /// `(x)`: softmax over the last axis, accumulated in float32.
    Softmax,
    /// `(x)`: `log(sum(exp(x)))` over the last axis, which becomes size 1.
    LogSumExp,
    /// `(logits, targets)`: `logsumexp(logits) - logits[target]` per row, in float32.
    CrossEntropy,
    /// `(logits, targets, loss, g)`: the gradient of [`FastOp::CrossEntropy`] for the logits.
    CrossEntropyVjp,
    /// `(q, k, v[, mask][, sinks])`: scaled dot-product attention (see
    /// [`scaled_dot_product_attention`]).
    Attention {
        /// `f32::to_bits` of the score scale.
        scale: u32,
        /// Causal masking (keys after the query's position, aligned at the end, are hidden).
        causal: bool,
        /// Whether an array mask (bool or additive, `[B, Hq, Lq, Lk]`) is the fourth input.
        mask: bool,
        /// Whether per-head attention sinks (`[Hq]`) are the last input.
        sinks: bool,
    },
    /// `(q, k, v, g, beta, state)`: the gated delta rule (see [`gated_delta`]), with the
    /// outputs and the final state.
    GatedDelta,
    /// `(x, w, scales, biases, lhs_indices, rhs_indices[, offsets])`: quantized matrix
    /// products of gathered batches (see [`crate::gather_qmm`]); the indices are uint32
    /// and broadcast to the output's batch shape. `offsets` (int32, one per batch of `w`)
    /// marks where each weight batch's rows start when the rows are sorted by it.
    GatherQmm {
        /// Values per scale and bias.
        group_size: usize,
        /// Bits per value.
        bits: u8,
        /// `x · wᵀ` (else `x · w`).
        transpose: bool,
    },
}

/// The mask of [`scaled_dot_product_attention`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum AttentionMask<'a> {
    /// Every query attends to every key.
    None,
    /// Query `i` attends to keys `0..=i + Lk - Lq` (the queries are the last `Lq` positions).
    Causal,
    /// A boolean mask (true attends) or an additive mask, broadcastable to
    /// `[B, Hq, Lq, Lk]`.
    Array(&'a Array),
}

fn float(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn check_inputs(inputs: &[&Array]) -> Result<()> {
    match inputs.iter().find_map(|a| a.error()) {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// A 0-D array holding `value` in `dtype`.
fn scalar(value: f64, dtype: DType) -> Result<Array> {
    filled(&[], dtype, |_| Canon::Float(value))
}

/// Whether the kernels cover `dtype` on the current stream.
pub(crate) fn on_gpu(dtype: DType, size: usize) -> bool {
    default_stream().device.kind == DeviceKind::Gpu
        && matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        && size > 0
        && size < u32::MAX as usize
}

/// Records `op` as one GPU kernel, or builds its fallback graph (CPU stream, other dtypes,
/// empty arrays).
pub(crate) fn record_or_fallback(op: FastOp, inputs: Vec<Array>) -> Result<Vec<Array>> {
    if on_gpu(inputs[0].dtype(), inputs[0].size()) {
        Array::from_op(Primitive::Fast(op), inputs)
    } else {
        op.fallback(&inputs)
    }
}

fn one(op: FastOp, inputs: Vec<Array>) -> Result<Array> {
    Ok(record_or_fallback(op, inputs)?.swap_remove(0))
}

fn floating(op: &'static str, dtype: DType) -> Result<()> {
    match dtype.category() {
        Category::Float => Ok(()),
        _ => Err(Error::UnsupportedDType { op, dtype }),
    }
}

/// A 1-D parameter of the last axis's size.
fn check_param(op: &'static str, what: &str, p: &Array, x: &Array) -> Result<()> {
    if p.ndim() != 1 || p.size() != x.shape()[x.ndim() - 1] {
        return Err(Error::InvalidArgument(format!(
            "{op}: {what} must be 1-D with {} elements, got shape {:?}",
            x.shape()[x.ndim() - 1],
            p.shape()
        )));
    }
    Ok(())
}

/// Root-mean-square normalisation over the last axis, `x / sqrt(mean(x²) + eps) · weight`,
/// as MLX's `fast.rms_norm`: statistics in float32, the result in the promoted type of `x`
/// and `weight` (1-D, the last axis's size).
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
    check_inputs(&[x])?;
    if x.ndim() == 0 {
        return Err(Error::InvalidArgument(
            "rms_norm: the input needs at least one dimension".into(),
        ));
    }
    let dtype = match weight {
        Some(w) => {
            check_inputs(&[w])?;
            check_param("rms_norm", "weight", w, x)?;
            crate::promote_types(x.dtype(), w.dtype())
        }
        None => x.dtype(),
    };
    floating("rms_norm", dtype)?;
    let w = match weight {
        Some(w) => astype(w, dtype)?,
        None => scalar(1.0, dtype)?,
    };
    let op = FastOp::RmsNorm { eps: eps.to_bits() };
    one(op, vec![astype(x, dtype)?, w])
}

/// Layer normalisation over the last axis, `(x - mean) / sqrt(var + eps) · weight + bias`,
/// as MLX's `fast.layer_norm`: statistics in float32, the result in the promoted type.
pub fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
) -> Result<Array> {
    check_inputs(&[x])?;
    if x.ndim() == 0 {
        return Err(Error::InvalidArgument(
            "layer_norm: the input needs at least one dimension".into(),
        ));
    }
    let mut dtype = x.dtype();
    for (p, what) in [(weight, "weight"), (bias, "bias")] {
        if let Some(p) = p {
            check_inputs(&[p])?;
            check_param("layer_norm", what, p, x)?;
            dtype = crate::promote_types(dtype, p.dtype());
        }
    }
    floating("layer_norm", dtype)?;
    let w = match weight {
        Some(w) => astype(w, dtype)?,
        None => scalar(1.0, dtype)?,
    };
    let b = match bias {
        Some(b) => astype(b, dtype)?,
        None => scalar(0.0, dtype)?,
    };
    let op = FastOp::LayerNorm { eps: eps.to_bits() };
    one(op, vec![astype(x, dtype)?, w, b])
}

/// Rotary position embeddings, as MLX's `fast.rope`, of `x` (at least 3-D: `[B, ...,
/// T, D]`, positions along `T`): the first `dims` features (even) are rotated in pairs,
/// the two halves (or interleaved pairs, `traditional`) by angle `(offset + t) · scale ·
/// θᵢ`, with `θᵢ = base^(-2i/dims)` or `1 / freqs[i]`. Exactly one of `base` and `freqs`
/// is given. `offset` is an integer scalar or one per batch entry.
pub fn rope(
    x: &Array,
    dims: usize,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &Array,
    freqs: Option<&Array>,
) -> Result<Array> {
    check_inputs(&[x, offset])?;
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("rope: {msg}")));
    if x.ndim() < 3 {
        return invalid(format!(
            "the input needs at least 3 dimensions, got {:?}",
            x.shape()
        ));
    }
    floating("rope", x.dtype())?;
    if offset.ndim() > 1 || (offset.size() != 1 && offset.size() != x.shape()[0]) {
        return invalid(format!(
            "offset must be a scalar or have {} elements, got shape {:?}",
            x.shape()[0],
            offset.shape()
        ));
    }
    if !matches!(
        offset.dtype().category(),
        Category::Signed | Category::Unsigned
    ) {
        return invalid(format!("offset must be an integer, got {}", offset.dtype()));
    }
    if dims == 0 || !dims.is_multiple_of(2) || dims > x.shape()[x.ndim() - 1] {
        return invalid(format!(
            "dims must be even, positive and at most {}, got {dims}",
            x.shape()[x.ndim() - 1]
        ));
    }
    let offset = match offset.dtype().size() {
        4 => offset.clone(),
        _ => astype(offset, DType::I32)?,
    };
    let mut inputs = vec![x.clone(), offset];
    let base = match (base, freqs) {
        (Some(b), None) => b,
        (None, Some(f)) => {
            check_inputs(&[f])?;
            if f.ndim() != 1 || f.size() != dims / 2 {
                return invalid(format!(
                    "freqs must be 1-D with {} elements, got shape {:?}",
                    dims / 2,
                    f.shape()
                ));
            }
            inputs.push(astype(f, DType::F32)?);
            1.0
        }
        _ => return invalid("exactly one of base and freqs must be given".into()),
    };
    let op = FastOp::Rope {
        dims,
        traditional,
        base: base.to_bits(),
        scale: scale.to_bits(),
        forward: true,
    };
    one(op, inputs)
}

/// Cross entropy over the last axis of `logits` against integer class indices `targets`
/// (the logits' shape without the last axis; negative indices count from the end), as
/// MLX's `fast.cross_entropy`: `logsumexp(logits) - logits[target]` per row, in float32.
pub fn cross_entropy(logits: &Array, targets: &Array) -> Result<Array> {
    check_inputs(&[logits, targets])?;
    if logits.ndim() == 0 || targets.shape() != &logits.shape()[..logits.ndim() - 1] {
        return Err(Error::InvalidArgument(format!(
            "cross_entropy: targets of shape {:?} do not match logits of shape {:?}",
            targets.shape(),
            logits.shape()
        )));
    }
    floating("cross_entropy", logits.dtype())?;
    if !matches!(
        targets.dtype().category(),
        Category::Signed | Category::Unsigned
    ) {
        return Err(Error::InvalidArgument(format!(
            "cross_entropy: targets must be integer class indices, got {}",
            targets.dtype()
        )));
    }
    one(
        FastOp::CrossEntropy,
        vec![logits.clone(), astype(targets, DType::I32)?],
    )
}

/// Records a `SliceUpdate` of `base`'s last axis at `start, start + step, …` (`len` values)
/// with `update`.
fn update_last(base: &Array, update: &Array, start: usize, step: isize) -> Result<Array> {
    let n = base.ndim();
    let mut starts = vec![0; n];
    starts[n - 1] = start;
    let mut steps = vec![1; n];
    steps[n - 1] = step;
    let primitive = Primitive::SliceUpdate {
        start: starts,
        step: steps,
        len: update.shape().to_vec(),
    };
    Ok(Array::from_op(primitive, vec![base.clone(), update.clone()])?.swap_remove(0))
}

/// The last axis of `x` from `start` in steps of `step`, `len` values.
fn last(x: &Array, start: usize, step: isize, len: usize) -> Result<Array> {
    let mut args = vec![SliceArg::full(); x.ndim()];
    let stop = start as isize + step * len as isize;
    args[x.ndim() - 1] = SliceArg::new(Some(start as isize), Some(stop), step);
    slice(x, &args)
}

/// `1 / sqrt(x)`.
fn rsqrt(x: &Array) -> Result<Array> {
    divide(1.0, sqrt(x)?)
}

/// `softmax` over the last axis from ordinary operations, in float32 for half types.
fn softmax_graph(x: &Array) -> Result<Array> {
    let xf = if x.dtype() == DType::F32 {
        x.clone()
    } else {
        astype(x, DType::F32)?
    };
    let m = stop_gradient(&max(&xf, &[-1], true)?)?;
    let e = exp(&subtract(&xf, &m)?)?;
    astype(&divide(&e, sum(&e, &[-1], true)?)?, x.dtype())
}

/// `log(sum(exp(x)))` over the last axis (kept), from ordinary operations.
fn logsumexp_graph(x: &Array) -> Result<Array> {
    let m = stop_gradient(&max(x, &[-1], true)?)?;
    let out = add(log(&sum(&exp(&subtract(x, &m)?)?, &[-1], true)?)?, &m)?;
    where_(equal(&ops::abs(&m)?, f64::INFINITY)?, &m, &out)
}

/// `[..., C]` one-hot rows of the (non-negative) class indices `y`, in `dtype`.
fn one_hot(y: &Array, classes: usize, dtype: DType) -> Result<Array> {
    let positions = arange(0.0, classes as f64, 1.0, y.dtype())?;
    equal(&expand_dims(y, -1)?, &positions)?.astype(dtype)
}

/// Class indices with negative values counted from the end.
fn wrap(y: &Array, classes: usize) -> Result<Array> {
    where_(less(y, 0)?, add(y, classes as f64)?, y)
}

impl FastOp {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::RmsNorm { .. } => "rms_norm",
            Self::RmsNormVjp { .. } => "rms_norm_vjp",
            Self::LayerNorm { .. } => "layer_norm",
            Self::LayerNormVjp { .. } => "layer_norm_vjp",
            Self::Rope { .. } => "rope",
            Self::Softmax => "softmax",
            Self::LogSumExp => "logsumexp",
            Self::CrossEntropy => "cross_entropy",
            Self::CrossEntropyVjp => "cross_entropy_vjp",
            Self::Attention { .. } => "scaled_dot_product_attention",
            Self::GatedDelta => "gated_delta",
            Self::GatherQmm { .. } => "gather_qmm",
        }
    }

    /// `(dtype, shape)` per output, after checking the input count.
    pub(crate) fn outputs(&self, inputs: &[Array]) -> Result<Vec<(DType, Vec<usize>)>> {
        let arity = match self {
            Self::Softmax | Self::LogSumExp => 1,
            Self::RmsNorm { .. } | Self::CrossEntropy => 2,
            Self::RmsNormVjp { .. } | Self::LayerNorm { .. } => 3,
            Self::LayerNormVjp { .. } | Self::CrossEntropyVjp => 4,
            Self::Rope { .. } => inputs.len().clamp(2, 3),
            Self::Attention { mask, sinks, .. } => 3 + usize::from(*mask) + usize::from(*sinks),
            Self::GatedDelta => 6,
            Self::GatherQmm { .. } => inputs.len().clamp(6, 7),
        };
        if inputs.len() != arity || inputs[0].ndim() == 0 {
            return Err(Error::InvalidArgument(format!(
                "{}: {arity} inputs expected, got {}",
                self.name(),
                inputs.len()
            )));
        }
        let x = &inputs[0];
        let like = |a: &Array| (a.dtype(), a.shape().to_vec());
        Ok(match self {
            Self::RmsNorm { .. } | Self::LayerNorm { .. } | Self::Rope { .. } | Self::Softmax => {
                vec![like(x)]
            }
            Self::LogSumExp => {
                let mut shape = x.shape().to_vec();
                *shape.last_mut().expect("1-D") = 1;
                vec![(x.dtype(), shape)]
            }
            Self::RmsNormVjp { .. } => vec![like(x), like(&inputs[1])],
            Self::LayerNormVjp { .. } => vec![like(x), like(&inputs[1]), like(&inputs[2])],
            Self::CrossEntropy => vec![(DType::F32, inputs[1].shape().to_vec())],
            Self::CrossEntropyVjp => vec![like(x)],
            Self::Attention { .. } => {
                let mut shape = x.shape().to_vec();
                shape[3] = inputs[2].shape()[3];
                vec![(x.dtype(), shape)]
            }
            Self::GatedDelta => vec![(x.dtype(), inputs[2].shape().to_vec()), like(&inputs[5])],
            Self::GatherQmm {
                bits, transpose, ..
            } => {
                let w = inputs[1].shape();
                let n = match transpose {
                    true => w[w.len() - 2],
                    false => w[w.len() - 1] * 32 / usize::from(*bits),
                };
                let mut shape = inputs[4].shape().to_vec();
                shape.extend([x.shape()[x.ndim() - 2], n]);
                vec![(x.dtype(), shape)]
            }
        })
    }

    /// The same computation from ordinary operations (MLX's fallbacks).
    pub(crate) fn fallback(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let x = &inputs[0];
        let f32x = || astype(x, DType::F32);
        let out = match self {
            Self::RmsNorm { eps } => {
                let xf = f32x()?;
                let n = rsqrt(&add(mean(&multiply(&xf, &xf)?, &[-1], true)?, float(*eps))?)?;
                let y = multiply(&xf, &n)?.astype(x.dtype())?;
                let w = &inputs[1];
                vec![if w.ndim() == 0 { y } else { multiply(&y, w)? }]
            }
            Self::LayerNorm { eps } => {
                let xf = f32x()?;
                let xc = subtract(&xf, mean(&xf, &[-1], true)?)?;
                let v = mean(&multiply(&xc, &xc)?, &[-1], true)?;
                let mut y = multiply(&xc, rsqrt(&add(&v, float(*eps))?)?)?.astype(x.dtype())?;
                if inputs[1].ndim() != 0 {
                    y = multiply(&y, &inputs[1])?;
                }
                if inputs[2].ndim() != 0 {
                    y = add(&y, &inputs[2])?;
                }
                vec![y]
            }
            Self::RmsNormVjp { eps } => {
                let (w, g) = (&inputs[1], &inputs[2]);
                let n = rsqrt(&add(mean(&multiply(x, x)?, &[-1], true)?, float(*eps))?)?;
                let n3 = multiply(&multiply(&n, &n)?, &n)?;
                let gw = multiply(g, w)?;
                let t = mean(&multiply(&gw, x)?, &[-1], true)?;
                let dx = subtract(multiply(&gw, &n)?, multiply(multiply(x, &t)?, &n3)?)?;
                let dw = match w.ndim() {
                    0 => scalar(0.0, w.dtype())?,
                    _ => sum_rows(&multiply(g, multiply(x, &n)?)?)?,
                };
                vec![dx, dw]
            }
            Self::LayerNormVjp { eps } => {
                let (w, b, g) = (&inputs[1], &inputs[2], &inputs[3]);
                let mu = mean(x, &[-1], true)?;
                let xc = subtract(x, &mu)?;
                let var = mean(&multiply(&xc, &xc)?, &[-1], true)?;
                let n = rsqrt(&add(&var, float(*eps))?)?;
                let n3 = multiply(&multiply(&n, &n)?, &n)?;
                let wg = multiply(w, g)?;
                let mwg = mean(&wg, &[-1], true)?;
                let mwgxc = mean(&multiply(&wg, &xc)?, &[-1], true)?;
                let dx = subtract(
                    multiply(subtract(&wg, &mwg)?, &n)?,
                    multiply(multiply(&xc, &mwgxc)?, &n3)?,
                )?;
                let dw = match w.ndim() {
                    0 => scalar(0.0, w.dtype())?,
                    _ => sum_rows(&multiply(g, multiply(&xc, &n)?)?)?,
                };
                let db = match b.ndim() {
                    0 => scalar(0.0, b.dtype())?,
                    _ => sum_rows(g)?,
                };
                vec![dx, dw, db]
            }
            Self::Attention {
                scale,
                causal,
                mask,
                sinks,
            } => vec![attention_graph(
                inputs,
                float(*scale),
                *causal,
                *mask,
                *sinks,
            )?],
            Self::Rope {
                dims,
                traditional,
                base,
                scale,
                forward,
            } => vec![rope_graph(
                inputs,
                *dims,
                *traditional,
                float(*base),
                float(*scale),
                *forward,
            )?],
            Self::Softmax => vec![softmax_graph(x)?],
            Self::GatedDelta => gated_delta_graph(inputs)?,
            Self::GatherQmm {
                group_size,
                bits,
                transpose,
            } => vec![gather_qmm_graph(
                inputs,
                *group_size,
                u32::from(*bits),
                *transpose,
            )?],
            Self::LogSumExp => vec![logsumexp_graph(x)?],
            Self::CrossEntropy => {
                let classes = x.shape()[x.ndim() - 1];
                let y = wrap(&inputs[1], classes)?;
                let hit = one_hot(&y, classes, DType::Bool)?;
                let score = sum(&where_(&hit, x, 0.0)?, &[-1], false)?;
                let lse = squeeze(&logsumexp_graph(x)?, &[-1])?;
                vec![subtract(&lse, &score)?.astype(DType::F32)?]
            }
            Self::CrossEntropyVjp => {
                let classes = x.shape()[x.ndim() - 1];
                let (y, loss, g) = (wrap(&inputs[1], classes)?, &inputs[2], &inputs[3]);
                let hit = one_hot(&y, classes, DType::Bool)?;
                let score = sum(&where_(&hit, x, 0.0)?, &[-1], false)?;
                let lse = add(loss, score.astype(DType::F32)?)?;
                let p = exp(&subtract(&f32x()?, expand_dims(&lse, -1)?)?)?;
                let gx = multiply(
                    expand_dims(g, -1)?,
                    subtract(&p, one_hot(&y, classes, DType::F32)?)?,
                )?;
                vec![gx.astype(x.dtype())?]
            }
        };
        Ok(out)
    }

    /// Reverse-mode rule: one cotangent per input (`None` where not wanted).
    pub(crate) fn vjp(
        &self,
        inputs: &[Array],
        outputs: &[Array],
        cotangents: &[Array],
        wanted: &[bool],
    ) -> Result<Vec<Option<Array>>> {
        let g = &cotangents[0];
        let pick = |all: Vec<Array>| -> Vec<Option<Array>> {
            all.into_iter()
                .zip(wanted)
                .map(|(a, &w)| w.then_some(a))
                .collect()
        };
        match self {
            Self::RmsNorm { eps } => {
                let op = Self::RmsNormVjp { eps: *eps };
                let grads =
                    record_or_fallback(op, vec![inputs[0].clone(), inputs[1].clone(), g.clone()])?;
                Ok(pick(grads))
            }
            Self::LayerNorm { eps } => {
                let op = Self::LayerNormVjp { eps: *eps };
                let mut all = inputs.to_vec();
                all.push(g.clone());
                Ok(pick(record_or_fallback(op, all)?))
            }
            Self::Rope { forward, .. } => {
                if wanted[1..].iter().any(|&w| w) {
                    return Err(Error::InvalidArgument(
                        "rope: no gradient with respect to the offset or frequencies".into(),
                    ));
                }
                let Self::Rope {
                    dims,
                    traditional,
                    base,
                    scale,
                    ..
                } = *self
                else {
                    unreachable!()
                };
                let inverse = Self::Rope {
                    dims,
                    traditional,
                    base,
                    scale,
                    forward: !forward,
                };
                let mut all = inputs.to_vec();
                all[0] = g.clone();
                let mut out = vec![Some(one(inverse, all)?)];
                out.extend(std::iter::repeat_n(None, inputs.len() - 1));
                Ok(out)
            }
            Self::Softmax => {
                let s = &outputs[0];
                let sg = multiply(s, g)?;
                let dx = subtract(&sg, multiply(s, sum(&sg, &[-1], true)?)?)?;
                Ok(vec![Some(dx)])
            }
            Self::LogSumExp => {
                let s = one(Self::Softmax, vec![inputs[0].clone()])?;
                Ok(vec![Some(multiply(g, &s)?)])
            }
            Self::CrossEntropy => {
                if wanted[1] {
                    return Err(Error::InvalidArgument(
                        "cross_entropy: no gradient with respect to the targets".into(),
                    ));
                }
                let all = vec![
                    inputs[0].clone(),
                    inputs[1].clone(),
                    outputs[0].clone(),
                    g.clone(),
                ];
                Ok(vec![Some(one(Self::CrossEntropyVjp, all)?), None])
            }
            Self::RmsNormVjp { .. }
            | Self::LayerNormVjp { .. }
            | Self::CrossEntropyVjp
            | Self::Attention { .. }
            | Self::GatedDelta
            | Self::GatherQmm { .. } => self.fallback_vjp(inputs, cotangents, wanted),
        }
    }

    /// Reverse mode through the fallback graph, for the wanted floating-point inputs.
    fn fallback_vjp(
        &self,
        inputs: &[Array],
        cotangents: &[Array],
        wanted: &[bool],
    ) -> Result<Vec<Option<Array>>> {
        let active: Vec<usize> = (0..inputs.len())
            .filter(|&i| wanted[i] && inputs[i].dtype().category() == Category::Float)
            .collect();
        let primals: Vec<Array> = active.iter().map(|&i| inputs[i].clone()).collect();
        let f = |ps: &[Array]| {
            let mut all = inputs.to_vec();
            for (p, &i) in ps.iter().zip(&active) {
                all[i] = p.clone();
            }
            self.fallback(&all)
        };
        let (_, grads) = crate::vjp(f, &primals, cotangents)?;
        let mut out = vec![None; inputs.len()];
        for (g, &i) in grads.into_iter().zip(&active) {
            out[i] = Some(g);
        }
        Ok(out)
    }

    /// Forward-mode rule through the fallback graph (integer inputs are constants).
    pub(crate) fn jvp(&self, inputs: &[Array], tangents: &[Option<Array>]) -> Result<Vec<Array>> {
        let active: Vec<usize> = (0..inputs.len())
            .filter(|&i| tangents[i].is_some() && inputs[i].dtype().category() == Category::Float)
            .collect();
        let primals: Vec<Array> = active.iter().map(|&i| inputs[i].clone()).collect();
        let ts: Vec<Array> = active
            .iter()
            .map(|&i| tangents[i].clone().expect("active"))
            .collect();
        let f = |ps: &[Array]| {
            let mut all = inputs.to_vec();
            for (p, &i) in ps.iter().zip(&active) {
                all[i] = p.clone();
            }
            self.fallback(&all)
        };
        Ok(crate::jvp(f, &primals, &ts)?.1)
    }

    /// Batching rule: row-wise operations with only the first input batched (and cross
    /// entropy) keep their kernel with the batch axis first; otherwise the fallback graph
    /// is batched.
    pub(crate) fn vmap(
        &self,
        inputs: &[Array],
        axes: &[Option<usize>],
    ) -> Result<(Vec<Array>, usize)> {
        let rest_unbatched = axes[1..].iter().all(Option::is_none);
        let front = |i: usize| -> Result<Array> {
            let a = &inputs[i];
            match axes[i] {
                Some(ax) => crate::batching::move_axis(a, ax, 0),
                None => {
                    let b = inputs
                        .iter()
                        .zip(axes)
                        .find_map(|(a, ax)| ax.map(|ax| a.shape()[ax]))
                        .expect("one batched input");
                    let mut shape = vec![b];
                    shape.extend_from_slice(a.shape());
                    broadcast_to(&expand_dims(a, 0)?, &shape)
                }
            }
        };
        let simple = match self {
            Self::RmsNorm { .. } | Self::LayerNorm { .. } | Self::Softmax | Self::LogSumExp => {
                rest_unbatched
            }
            Self::Rope { .. } => rest_unbatched && inputs[1].size() == 1,
            Self::CrossEntropy => true,
            _ => false,
        };
        if simple {
            let mut all = inputs.to_vec();
            all[0] = front(0)?;
            if *self == Self::CrossEntropy {
                all[1] = front(1)?;
            }
            return Ok((record_or_fallback(self.clone(), all)?, 0));
        }
        let f = |xs: &[Array]| self.fallback(xs);
        Ok((crate::transforms::batch(&f, inputs, axes)?, 0))
    }

    /// Runs the fallback graph on the CPU over evaluated operands, for work no kernel takes.
    pub(crate) fn eval_cpu(&self, operands: &[Operand<'_>]) -> Result<Vec<crate::array::Data>> {
        let leaves = operands
            .iter()
            .map(|(meta, data)| {
                let buffer = crate::kernels::contiguous(meta, data)?;
                Array::from_buffer(buffer, meta.dtype, &meta.shape)
            })
            .collect::<Result<Vec<_>>>()?;
        let previous = default_stream();
        set_default_stream(Stream::default_for(Device::cpu()));
        let outputs = self.fallback(&leaves);
        set_default_stream(previous);
        let outputs = outputs?;
        crate::eval(&outputs)?;
        outputs
            .iter()
            .map(|a| {
                let data = a.data().expect("evaluated");
                Ok(crate::array::Data::new(crate::kernels::contiguous(
                    a.out(),
                    data,
                )?))
            })
            .collect()
    }
}

/// `g` summed over every axis but the last.
fn sum_rows(g: &Array) -> Result<Array> {
    let axes: Vec<isize> = (0..g.ndim() as isize - 1).collect();
    sum(g, &axes, false)
}

/// MLX's RoPE fallback: the rotated features replace the input's in place.
fn rope_graph(
    inputs: &[Array],
    dims: usize,
    traditional: bool,
    base: f32,
    scale: f32,
    forward: bool,
) -> Result<Array> {
    let x = &inputs[0];
    let shape: Vec<isize> = x.shape().iter().map(|&d| d as isize).collect();
    let nd = x.ndim();
    let (b, t, d) = (x.shape()[0], x.shape()[nd - 2], x.shape()[nd - 1]);
    let x4 = reshape(x, &[b as isize, -1, t as isize, d as isize])?;
    let dtype = x.dtype();
    let half = dims / 2;
    let offset = &inputs[1];
    let offset = if offset.size() > 1 {
        reshape(offset, &[-1, 1, 1])?
    } else {
        reshape(offset, &[])?
    };
    let positions = multiply(
        add(arange(0.0, t as f64, 1.0, DType::F32)?, &offset)?,
        f64::from(scale),
    )?;
    let inv_freqs = match inputs.get(2) {
        Some(f) => divide(1.0, f)?,
        None => exp(&multiply(
            arange(0.0, -(half as f64), -1.0, DType::F32)?,
            f64::from(base.ln() / half as f32),
        )?)?,
    };
    let theta = multiply(expand_dims(&positions, -1)?, &inv_freqs)?;
    let (c, s) = (cos(&theta)?.astype(dtype)?, sin(&theta)?.astype(dtype)?);
    let (start2, step) = if traditional { (1, 2) } else { (half, 1) };
    let x1 = last(&x4, 0, step, half)?;
    let x2 = last(&x4, start2, step, half)?;
    let (r1, r2) = if forward {
        (
            subtract(multiply(&x1, &c)?, multiply(&x2, &s)?)?,
            add(multiply(&x1, &s)?, multiply(&x2, &c)?)?,
        )
    } else {
        (
            add(multiply(&x2, &s)?, multiply(&x1, &c)?)?,
            subtract(multiply(&x2, &c)?, multiply(&x1, &s)?)?,
        )
    };
    let full = |r: Array| broadcast_to(&r, x1.shape());
    let out = update_last(&x4, &full(r1)?, 0, step)?;
    let out = update_last(&out, &full(r2)?, start2, step)?;
    reshape(&out, &shape)
}

/// Scaled dot-product attention, as MLX's `fast.scaled_dot_product_attention`:
/// `softmax(scale · q kᵀ + mask) v` over `[B, H, L, D]` arrays.
///
/// `k` and `v` may have fewer heads than `q` (grouped-query attention; `Hq` a multiple of
/// `Hkv`), and `v` its own head size. Optional `sinks` (`[Hq]`) add one logit per head that
/// takes part in the softmax but attends to nothing. On the GPU, short queries (at most 8)
/// use MLX's vector kernels and longer ones its tiled kernel, for the head sizes MLX
/// supports; other cases run the equivalent composition (as MLX does).
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: AttentionMask<'_>,
    sinks: Option<&Array>,
) -> Result<Array> {
    check_inputs(&[q, k, v])?;
    let invalid = |msg: String| {
        Err(Error::InvalidArgument(format!(
            "scaled_dot_product_attention: {msg}"
        )))
    };
    if [q, k, v].iter().any(|a| a.ndim() != 4) {
        return invalid(format!(
            "q, k and v must be 4-D, got {:?}, {:?} and {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        ));
    }
    let (qs, ks, vs) = (q.shape(), k.shape(), v.shape());
    if ks[0] != qs[0] || vs[0] != qs[0] || qs[3] != ks[3] || ks[1] != vs[1] || ks[2] != vs[2] {
        return invalid(format!("incompatible shapes {qs:?}, {ks:?} and {vs:?}"));
    }
    if ks[1] == 0 || qs[1] % ks[1] != 0 {
        return invalid(format!(
            "the query heads ({}) must be a multiple of the key heads ({})",
            qs[1], ks[1]
        ));
    }
    let dtype = crate::promote_types(crate::promote_types(q.dtype(), k.dtype()), v.dtype());
    floating("scaled_dot_product_attention", dtype)?;
    let mut inputs = vec![astype(q, dtype)?, astype(k, dtype)?, astype(v, dtype)?];
    let (causal, has_mask) = match mask {
        AttentionMask::None => (false, false),
        AttentionMask::Causal => (true, false),
        AttentionMask::Array(m) => {
            check_inputs(&[m])?;
            if m.ndim() > 4 {
                return invalid(format!(
                    "the mask must have at most 4 dimensions, got {:?}",
                    m.shape()
                ));
            }
            let m = if m.dtype() == DType::Bool {
                m.clone()
            } else if crate::promote_types(m.dtype(), dtype) == dtype {
                astype(m, dtype)?
            } else {
                return invalid(format!("a {} mask does not promote to {dtype}", m.dtype()));
            };
            inputs.push(broadcast_to(&m, &[qs[0], qs[1], qs[2], ks[2]])?);
            (false, true)
        }
    };
    if let Some(s) = sinks {
        check_inputs(&[s])?;
        if s.ndim() != 1 || s.size() != qs[1] || crate::promote_types(s.dtype(), dtype) != dtype {
            return invalid(format!(
                "sinks must be [{}] of a type promoting to {dtype}",
                qs[1]
            ));
        }
        inputs.push(astype(s, dtype)?);
    }
    let op = FastOp::Attention {
        scale: scale.to_bits(),
        causal,
        mask: has_mask,
        sinks: sinks.is_some(),
    };
    if on_gpu(dtype, q.size()) && attention_kernel(&inputs, causal, has_mask) {
        Ok(Array::from_op(Primitive::Fast(op), inputs)?.swap_remove(0))
    } else {
        Ok(op.fallback(&inputs)?.swap_remove(0))
    }
}

/// Whether MLX routes these shapes to a fused kernel (`has_fused_kernel` and
/// `use_fallback`, without the NAX and blocked head-size-512 kernels).
fn attention_kernel(inputs: &[Array], causal: bool, has_mask: bool) -> bool {
    let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
    let (lq, lk, d, dv) = (q.shape()[2], k.shape()[2], q.shape()[3], v.shape()[3]);
    let gqa = q.shape()[1] / k.shape()[1];
    if lk == 0 || lq == 0 {
        return false;
    }
    if lq > 8 {
        let head =
            (d == dv && matches!(d, 64 | 72 | 80 | 96 | 128 | 192 | 256)) || (d, dv) == (96, 64);
        let causal_ok = !causal || lq <= lk;
        // MLX: the composition is faster for these head sizes, except long causal
        // self-attention in half precision.
        let slow = match d {
            192 => true,
            256 => !(q.dtype() != DType::F32 && causal && lq >= 2048 && lq == lk),
            _ => false,
        };
        head && causal_ok && !slow
    } else {
        let head = (d == dv && matches!(d, 64 | 96 | 128 | 256 | 512))
            || (d, dv) == (192, 128)
            || (d, dv) == (96, 64);
        let d512 = d != 512 || (lq == 1 && gqa == 8 && !has_mask && lk >= 1024);
        head && lq <= lk && lq * gqa <= 32 && d512
    }
}

/// MLX's attention fallback, with sinks folded into the softmax.
fn attention_graph(
    inputs: &[Array],
    scale: f32,
    causal: bool,
    has_mask: bool,
    has_sinks: bool,
) -> Result<Array> {
    let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
    let dtype = q.dtype();
    let (b, hq, lq) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let (hkv, lk) = (k.shape()[1], k.shape()[2]);
    let rep = hq / hkv;
    let i = |n: usize| n as isize;
    let mut q = multiply(q, f64::from(scale))?;
    let (mut k, mut v) = (k.clone(), v.clone());
    if rep > 1 {
        q = reshape(&q, &[i(b), i(hkv), i(rep), i(lq), i(q.shape()[3])])?;
        k = expand_dims(&k, 2)?;
        v = expand_dims(&v, 2)?;
    }
    let n = k.ndim();
    let mut axes: Vec<usize> = (0..n).collect();
    axes.swap(n - 2, n - 1);
    let mut scores = crate::matmul(&q, &crate::transpose(&k, &axes)?)?;
    let mask = if causal {
        let offset = lk as f64 - lq as f64;
        let qi = expand_dims(&arange(offset, offset + lq as f64, 1.0, DType::I32)?, 1)?;
        let ki = expand_dims(&arange(0.0, lk as f64, 1.0, DType::I32)?, 0)?;
        Some(crate::greater_equal(&qi, &ki)?)
    } else if has_mask {
        let m = &inputs[3];
        Some(if rep > 1 {
            reshape(m, &[i(b), i(hkv), i(rep), i(lq), i(lk)])?
        } else {
            m.clone()
        })
    } else {
        None
    };
    if let Some(m) = mask {
        scores = if m.dtype() == DType::Bool {
            let lowest = match dtype {
                DType::F16 => f64::from(crate::f16::MIN),
                DType::BF16 => f64::from(crate::bf16::MIN),
                DType::F64 => f64::MIN,
                _ => f64::from(f32::MIN),
            };
            where_(&m, &scores, lowest)?
        } else {
            add(&scores, &m)?
        };
    }
    let probs = if has_sinks {
        // softmax over [sink, scores...], without the sink's column.
        let s = astype(&scores, DType::F32)?;
        let sink = astype(inputs.last().expect("sinks"), DType::F32)?;
        let shape: Vec<isize> = if rep > 1 {
            vec![1, i(hkv), i(rep), 1, 1]
        } else {
            vec![1, i(hq), 1, 1]
        };
        let sink = reshape(&sink, &shape)?;
        let m = stop_gradient(&crate::maximum(&max(&s, &[-1], true)?, &sink)?)?;
        let e = exp(&subtract(&s, &m)?)?;
        let denom = add(sum(&e, &[-1], true)?, exp(&subtract(&sink, &m)?)?)?;
        astype(&divide(&e, &denom)?, dtype)?
    } else if on_gpu(dtype, scores.size()) {
        crate::softmax(&scores, &[-1])?
    } else {
        softmax_graph(&scores)?
    };
    let out = crate::matmul(&probs, &v)?;
    if rep > 1 {
        reshape(&out, &[i(b), i(hq), i(lq), i(v.shape()[n - 1])])
    } else {
        Ok(out)
    }
}

/// The gated delta rule, Gated DeltaNet's recurrence (as mlx-lm's `gated_delta_update`
/// once its gates are computed). For each step `t` and value head, with `S` its state
/// (`[Dv, Dk]`):
///
/// `S ← g_t · S`, then `S ← S + β_t (v_t − S k_t) k_tᵀ`, and `y_t = S q_t`.
///
/// Shapes: `q` and `k` are `[B, T, Hk, Dk]`, `v` is `[B, T, Hv, Dv]` (value head `h` reads
/// key head `h / (Hv / Hk)`), `g` and `beta` are `[B, T, Hv]` and `state` is
/// `[B, Hv, Dv, Dk]`. Returns `y` (`[B, T, Hv, Dv]`, in `q`'s dtype) and the final state
/// (in `state`'s dtype); the recurrence runs in float32. On the GPU it is one kernel
/// (mlx-lm's, for `Dk` a multiple of 32), else the steps as ordinary operations, which also
/// define the derivatives.
///
/// ```
/// use nuvec_core::{Array, DType, Result, fast, ones, zeros};
///
/// # fn main() -> Result<()> {
/// let q = ones(&[1, 2, 1, 32], DType::F32)?;
/// let v = ones(&[1, 2, 1, 4], DType::F32)?;
/// let (g, beta) = (ones(&[1, 2, 1], DType::F32)?, ones(&[1, 2, 1], DType::F32)?);
/// let state = zeros(&[1, 1, 4, 32], DType::F32)?;
/// let (y, s) = fast::gated_delta(&(&q * 0.25), &(&q * 0.125), &v, &g, &(&beta * 0.5), &state)?;
/// // k · k = 0.5 and k · q = 1. Step 1: S = 0.5 · v kᵀ, so y = 0.5. Step 2: S k = 0.25,
/// // so S gains (1 − 0.25) · 0.5 · v kᵀ, and y = 0.875.
/// assert_eq!(y.to_vec::<f32>()?, [0.5, 0.5, 0.5, 0.5, 0.875, 0.875, 0.875, 0.875]);
/// assert_eq!(s.shape(), [1, 1, 4, 32]);
/// # Ok(())
/// # }
/// ```
pub fn gated_delta(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    check_inputs(&[q, k, v, g, beta, state])?;
    let op = "gated_delta";
    for a in [q, k, v, g, beta, state] {
        floating(op, a.dtype())?;
    }
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("{op}: {msg}")));
    if q.ndim() != 4 || q.shape() != k.shape() || v.ndim() != 4 {
        return invalid(format!(
            "q and k must be [B, T, Hk, Dk] and v [B, T, Hv, Dv], got {:?}, {:?}, {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        ));
    }
    let (b, t, hk, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (hv, dv) = (v.shape()[2], v.shape()[3]);
    if v.shape()[..2] != [b, t] || hk == 0 || hv % hk != 0 {
        return invalid(format!(
            "v must be [{b}, {t}, Hv, Dv] with Hv a multiple of {hk}, got {:?}",
            v.shape()
        ));
    }
    if g.shape() != [b, t, hv] || beta.shape() != [b, t, hv] {
        return invalid(format!(
            "g and beta must be [{b}, {t}, {hv}], got {:?} and {:?}",
            g.shape(),
            beta.shape()
        ));
    }
    if state.shape() != [b, hv, dv, dk] {
        return invalid(format!(
            "state must be [{b}, {hv}, {dv}, {dk}], got {:?}",
            state.shape()
        ));
    }
    let inputs = vec![
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        state.clone(),
    ];
    let kernel = on_gpu(q.dtype(), q.size())
        && q.dtype() == v.dtype()
        && dk % 32 == 0
        && [g, beta, state]
            .iter()
            .all(|a| matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32));
    let mut out = if kernel {
        Array::from_op(Primitive::Fast(FastOp::GatedDelta), inputs)?
    } else {
        FastOp::GatedDelta.fallback(&inputs)?
    };
    let s = out.pop().expect("two outputs");
    Ok((out.pop().expect("two outputs"), s))
}

/// The gated delta rule step by step from ordinary operations (mlx-lm's
/// `gated_delta_ops`).
fn gated_delta_graph(inputs: &[Array]) -> Result<Vec<Array>> {
    let [q, k, v, g, beta, state] = inputs else {
        unreachable!("six inputs")
    };
    let (b, t, hk, dk) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (hv, dv) = (v.shape()[2], v.shape()[3]);
    // Each key head serves `Hv / Hk` consecutive value heads.
    let heads = |x: &Array| -> Result<Array> {
        if hv == hk {
            return Ok(x.clone());
        }
        let wide = broadcast_to(&expand_dims(x, 3)?, &[b, t, hk, hv / hk, dk])?;
        reshape(&wide, &[b as isize, t as isize, hv as isize, dk as isize])
    };
    // In float32 at least, as the kernels.
    let wide = |x: &Array| match x.dtype() {
        DType::F64 => Ok(x.clone()),
        _ => astype(x, DType::F32),
    };
    let (qh, kh) = (heads(&wide(q)?)?, heads(&wide(k)?)?);
    let (v, g, beta) = (&wide(v)?, &wide(g)?, &wide(beta)?);
    let at = |x: &Array, i: usize| -> Result<Array> {
        let mut args = vec![SliceArg::full(); x.ndim()];
        args[1] = SliceArg::new(Some(i as isize), Some(i as isize + 1), 1);
        squeeze(&slice(x, &args)?, &[1])
    };
    let mut s = wide(state)?;
    let mut ys = Vec::with_capacity(t);
    for i in 0..t {
        let (qt, kt, vt) = (at(&qh, i)?, expand_dims(&at(&kh, i)?, 2)?, at(v, i)?);
        let decay = reshape(&at(g, i)?, &[b as isize, hv as isize, 1, 1])?;
        s = multiply(&s, &decay)?;
        let kv_mem = sum(&multiply(&s, &kt)?, &[-1], false)?;
        let delta = multiply(subtract(&vt, &kv_mem)?, expand_dims(&at(beta, i)?, -1)?)?;
        s = add(&s, multiply(&kt, expand_dims(&delta, -1)?)?)?;
        let y = sum(&multiply(&s, expand_dims(&qt, 2)?)?, &[-1], false)?;
        ys.push(astype(&y, q.dtype())?);
    }
    let y = match ys.is_empty() {
        true => zeros(&[b, 0, hv, dv], q.dtype())?,
        false => stack(&ys, 1)?,
    };
    Ok(vec![y, astype(&s, state.dtype())?])
}

/// The gathered quantized products from ordinary operations: the indexed batches of `x` and
/// of the weights, then one batched quantized product.
fn gather_qmm_graph(
    inputs: &[Array],
    group_size: usize,
    bits: u32,
    transpose: bool,
) -> Result<Array> {
    let (lhs, rhs) = (&inputs[4], &inputs[5]);
    let rows = |a: &Array, indices: &Array| -> Result<Array> {
        let nd = a.ndim();
        let flat = reshape(
            a,
            &[-1, a.shape()[nd - 2] as isize, a.shape()[nd - 1] as isize],
        )?;
        crate::take(&flat, indices, Some(0))
    };
    crate::quantized_matmul(
        &rows(&inputs[0], lhs)?,
        &rows(&inputs[1], rhs)?,
        &rows(&inputs[2], rhs)?,
        &rows(&inputs[3], rhs)?,
        transpose,
        group_size,
        bits,
    )
}
