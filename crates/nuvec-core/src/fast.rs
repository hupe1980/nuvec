//! Fused operations with dedicated GPU kernels, as MLX's `mx.fast`: [`rms_norm`],
//! [`layer_norm`], [`rope`] and [`cross_entropy`] (and, through [`crate::softmax`] and
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
            Self::RmsNormVjp { .. } | Self::LayerNormVjp { .. } | Self::CrossEntropyVjp => {
                self.fallback_vjp(inputs, cotangents, wanted)
            }
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
