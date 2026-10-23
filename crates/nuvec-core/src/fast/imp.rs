//! The implementation of the fused operations (see [`crate::fast`]): `Result`-returning
//! bodies, kernels' records and fallback graphs.

use crate::cast::Canon;
use crate::indexing::stack;
use crate::kernels::Operand;
use crate::ops::imp::zeros;
use crate::ops::imp::{
    self as ops, add, arange, astype, broadcast_to, cos, divide, equal, exp, expand_dims, filled,
    less, log, max, mean, multiply, reshape, sin, slice, sqrt, squeeze, stop_gradient, subtract,
    sum, where_,
};
use crate::primitive::Primitive;

use super::plan::{self, GroupKind, MaskKind, Route, RouteKey, Target};
use crate::{
    Array, Category, DType, Device, DeviceKind, Error, Result, SliceArg, Stream, default_stream,
};

/// A fused operation (see [`crate::fast`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FastOp {
    /// `(x, w)`: `x / sqrt(mean(x²) + eps) · w` over the last axis; `w` is 1-D or a scalar 1.
    RmsNorm {
        /// `f32::to_bits` of epsilon.
        eps: u32,
    },
    /// `(x, r, w)`: `h = x + r` and [`FastOp::RmsNorm`] of `h` (see [`add_rms_norm`](crate::fast::add_rms_norm)).
    AddRmsNorm {
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
    /// `([cache,] x, w[, freqs])`: [`norm_rope`](crate::fast::norm_rope): each `[b, h, l]` row of `x` RMS-normalised
    /// (with `eps`) and rotated as [`FastOp::Rope`] at position `offset + l`; written into
    /// `cache` at the same positions, or returned as a new `[B, H, L, D]` array.
    NormRope {
        /// `f32::to_bits` of epsilon, or no normalisation.
        eps: Option<u32>,
        /// Rotated features (even).
        dims: usize,
        /// Rotate interleaved pairs instead of the two halves.
        traditional: bool,
        /// `f32::to_bits` of the frequency base (unused with `freqs`).
        base: u32,
        /// `f32::to_bits` of the position scale.
        scale: u32,
        /// The position of the first row.
        offset: usize,
        /// Whether the first input is a cache written at the rows' positions.
        cache: bool,
    },
    /// `(x, w)`: [`conv1d`](crate::Array::conv1d) with `groups` equal to the channels (depthwise).
    DepthwiseConv1d {
        /// Positions between outputs.
        stride: usize,
        /// Zeros before and after the input.
        padding: usize,
        /// Positions between taps.
        dilation: usize,
    },
    /// `(q, k_cache, v_cache, block_tables, lengths)`: [`paged_attention`](crate::fast::paged_attention).
    PagedAttention {
        /// `f32::to_bits` of the score scale.
        scale: u32,
    },
    /// `(cache, new, slots)`: [`paged_write`](crate::fast::paged_write).
    PagedWrite,
    /// `(x)`: softmax over the last axis, accumulated in float32.
    Softmax,
    /// `(x)`: `log(sum(exp(x)))` over the last axis, which becomes size 1.
    LogSumExp,
    /// `(logits, targets)`: `logsumexp(logits) - logits[target]` per row, in float32.
    CrossEntropy,
    /// `(logits, targets, loss, g)`: the gradient of [`FastOp::CrossEntropy`] for the logits.
    CrossEntropyVjp,
    /// `(q, k, v[, mask][, sinks])`: scaled dot-product attention (see
    /// [`scaled_dot_product_attention`](crate::fast::scaled_dot_product_attention)).
    Attention {
        /// `f32::to_bits` of the score scale.
        scale: u32,
        /// Causal masking (keys after the query's position, aligned at the end, are hidden).
        causal: bool,
        /// Whether an array mask (bool or additive, `[B, Hq, Lq, Lk]`) is the fourth input.
        mask: bool,
        /// Whether per-head attention sinks (`[Hq]`) are the last input.
        sinks: bool,
        /// The vector kernels' passes over the keys (1 or 2; at most 8 queries), or 0 for
        /// the tiled kernel. Set by the attention plan.
        passes: u8,
        /// Key partitions of the first of two vector passes.
        blocks: u32,
        /// The decode-only grouped-query two-pass kernel (one query, no mask or sinks).
        gqa_kernel: bool,
        /// The one-key-per-step variant of the two-pass kernel.
        tk1: bool,
    },
    /// `(q, k, v, g, beta, state)`: the gated delta rule (see [`gated_delta`](crate::fast::gated_delta)), with the
    /// outputs and the final state (or every step's, [`gated_delta_steps`](crate::fast::gated_delta_steps)).
    GatedDelta {
        /// The state after every step (`[B, T, Hv, Dv, Dk]`) instead of the final one.
        steps: bool,
    },
    /// `(q, k, v, g, beta, pool, slots)`: [`gated_delta_slots`](crate::fast::gated_delta_slots), with the outputs and the
    /// updated pool.
    GatedDeltaSlots,
    /// `(a, b, lhs_indices, rhs_indices[, offsets])`: matrix products of gathered batches
    /// (see [`gather_mm`](crate::ops::gather_mm)), as [`FastOp::GatherQmm`] in full precision.
    GatherMm,
    /// `(x, w, scales, biases, lhs_indices, rhs_indices[, offsets])`: quantized matrix
    /// products of gathered batches (see [`gather_qmm`](crate::ops::gather_qmm)); the indices are uint32
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

/// The mask of [`scaled_dot_product_attention`](crate::fast::scaled_dot_product_attention).
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub enum AttentionMask<'a> {
    /// Every query attends to every key.
    #[default]
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
        _ => Err(Error::unsupported_dtype(op, dtype)),
    }
}

/// A 1-D parameter of the last axis's size.
fn check_param(op: &'static str, what: &str, p: &Array, x: &Array) -> Result<()> {
    if p.ndim() != 1 || p.size() != x.dims()[x.ndim() - 1] {
        return Err(Error::InvalidArgument(format!(
            "{op}: {what} must be 1-D with {} elements, got shape {:?}",
            x.dims()[x.ndim() - 1],
            p.dims()
        )));
    }
    Ok(())
}

/// The body of [`crate::fast::rms_norm`].
pub(crate) fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
    if x.ndim() == 0 {
        return Err(Error::InvalidArgument(
            "rms_norm: the input needs at least one dimension".into(),
        ));
    }
    let dtype = match weight {
        Some(w) => {
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

/// The body of [`crate::fast::add_rms_norm`].
pub(crate) fn add_rms_norm(
    x: &Array,
    residual: &Array,
    weight: Option<&Array>,
    eps: f32,
) -> Result<(Array, Array)> {
    let dtype = crate::promote_types(x.dtype(), residual.dtype());
    let fused = x.dims() == residual.dims()
        && x.ndim() > 0
        && dtype.category() == Category::Float
        && weight.is_none_or(|w| {
            w.slot_error().is_none()
                && w.ndim() == 1
                && w.size() == x.dims()[x.ndim() - 1]
                && crate::promote_types(dtype, w.dtype()) == dtype
        });
    if !fused {
        let h = add(x, residual)?;
        let y = rms_norm(&h, weight, eps)?;
        return Ok((h, y));
    }
    let w = match weight {
        Some(w) => astype(w, dtype)?,
        None => scalar(1.0, dtype)?,
    };
    let op = FastOp::AddRmsNorm { eps: eps.to_bits() };
    let mut out = record_or_fallback(op, vec![astype(x, dtype)?, astype(residual, dtype)?, w])?;
    let y = out.pop().expect("two outputs");
    Ok((out.pop().expect("two outputs"), y))
}

/// The body of [`crate::fast::layer_norm`].
pub(crate) fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
) -> Result<Array> {
    if x.ndim() == 0 {
        return Err(Error::InvalidArgument(
            "layer_norm: the input needs at least one dimension".into(),
        ));
    }
    let mut dtype = x.dtype();
    for (p, what) in [(weight, "weight"), (bias, "bias")] {
        if let Some(p) = p {
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

/// The body of [`crate::fast::rope`].
pub(crate) fn rope(
    x: &Array,
    dims: usize,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: &Array,
    freqs: Option<&Array>,
) -> Result<Array> {
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("rope: {msg}")));
    if x.ndim() < 3 {
        return invalid(format!(
            "the input needs at least 3 dimensions, got {:?}",
            x.dims()
        ));
    }
    floating("rope", x.dtype())?;
    if offset.ndim() > 1 || (offset.size() != 1 && offset.size() != x.dims()[0]) {
        return invalid(format!(
            "offset must be a scalar or have {} elements, got shape {:?}",
            x.dims()[0],
            offset.dims()
        ));
    }
    if !matches!(
        offset.dtype().category(),
        Category::Signed | Category::Unsigned
    ) {
        return invalid(format!("offset must be an integer, got {}", offset.dtype()));
    }
    if dims == 0 || !dims.is_multiple_of(2) || dims > x.dims()[x.ndim() - 1] {
        return invalid(format!(
            "dims must be even, positive and at most {}, got {dims}",
            x.dims()[x.ndim() - 1]
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
            if f.ndim() != 1 || f.size() != dims / 2 {
                return invalid(format!(
                    "freqs must be 1-D with {} elements, got shape {:?}",
                    dims / 2,
                    f.dims()
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

/// The body of [`crate::fast::cross_entropy`].
pub(crate) fn cross_entropy(logits: &Array, targets: &Array) -> Result<Array> {
    if logits.ndim() == 0 || targets.dims() != &logits.dims()[..logits.ndim() - 1] {
        return Err(Error::InvalidArgument(format!(
            "cross_entropy: targets of shape {:?} do not match logits of shape {:?}",
            targets.dims(),
            logits.dims()
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
        len: update.dims().to_vec(),
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
    equal(&expand_dims(y, -1)?, &positions)?.try_astype(dtype)
}

/// Class indices with negative values counted from the end.
fn wrap(y: &Array, classes: usize) -> Result<Array> {
    where_(less(y, 0)?, add(y, classes as f64)?, y)
}

impl FastOp {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::RmsNorm { .. } => "rms_norm",
            Self::AddRmsNorm { .. } => "add_rms_norm",
            Self::RmsNormVjp { .. } => "rms_norm_vjp",
            Self::LayerNorm { .. } => "layer_norm",
            Self::LayerNormVjp { .. } => "layer_norm_vjp",
            Self::Rope { .. } => "rope",
            Self::NormRope { .. } => "norm_rope",
            Self::DepthwiseConv1d { .. } => "conv1d",
            Self::PagedAttention { .. } => "paged_attention",
            Self::PagedWrite => "paged_write",
            Self::Softmax => "softmax",
            Self::LogSumExp => "logsumexp",
            Self::CrossEntropy => "cross_entropy",
            Self::CrossEntropyVjp => "cross_entropy_vjp",
            Self::Attention { .. } => "scaled_dot_product_attention",
            Self::GatedDelta { .. } | Self::GatedDeltaSlots => "gated_delta",
            Self::GatherQmm { .. } => "gather_qmm",
            Self::GatherMm => "gather_mm",
        }
    }

    /// `(dtype, shape)` per output, after checking the input count.
    pub(crate) fn outputs(&self, inputs: &[Array]) -> Result<Vec<(DType, Vec<usize>)>> {
        let arity = match self {
            Self::Softmax | Self::LogSumExp => 1,
            Self::RmsNorm { .. } | Self::CrossEntropy | Self::DepthwiseConv1d { .. } => 2,
            Self::PagedAttention { .. } => 5,
            Self::PagedWrite => 3,
            Self::AddRmsNorm { .. } | Self::RmsNormVjp { .. } | Self::LayerNorm { .. } => 3,
            Self::LayerNormVjp { .. } | Self::CrossEntropyVjp => 4,
            Self::Rope { .. } => inputs.len().clamp(2, 3),
            Self::NormRope { cache, .. } => inputs
                .len()
                .clamp(2 + usize::from(*cache), 3 + usize::from(*cache)),
            Self::Attention { mask, sinks, .. } => 3 + usize::from(*mask) + usize::from(*sinks),
            Self::GatedDelta { .. } => 6,
            Self::GatedDeltaSlots => 7,
            Self::GatherQmm { .. } => inputs.len().clamp(6, 7),
            Self::GatherMm => inputs.len().clamp(4, 5),
        };
        if inputs.len() != arity || inputs[0].ndim() == 0 {
            return Err(Error::InvalidArgument(format!(
                "{}: {arity} inputs expected, got {}",
                self.name(),
                inputs.len()
            )));
        }
        self.check(inputs)?;
        let x = &inputs[0];
        let like = |a: &Array| (a.dtype(), a.dims().to_vec());
        Ok(match self {
            Self::RmsNorm { .. }
            | Self::LayerNorm { .. }
            | Self::Rope { .. }
            | Self::NormRope { .. }
            | Self::Softmax => {
                vec![like(x)]
            }
            Self::LogSumExp => {
                let mut shape = x.dims().to_vec();
                *shape.last_mut().expect("1-D") = 1;
                vec![(x.dtype(), shape)]
            }
            Self::AddRmsNorm { .. } => vec![like(x), like(x)],
            Self::PagedAttention { .. } | Self::PagedWrite => vec![like(x)],
            Self::DepthwiseConv1d {
                stride,
                padding,
                dilation,
            } => {
                let (l, k) = (x.dims()[1], inputs[1].dims()[1]);
                let l_out = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
                vec![(x.dtype(), vec![x.dims()[0], l_out, x.dims()[2]])]
            }
            Self::RmsNormVjp { .. } => vec![like(x), like(&inputs[1])],
            Self::LayerNormVjp { .. } => vec![like(x), like(&inputs[1]), like(&inputs[2])],
            Self::CrossEntropy => vec![(DType::F32, inputs[1].dims().to_vec())],
            Self::CrossEntropyVjp => vec![like(x)],
            Self::Attention { .. } => {
                let mut shape = x.dims().to_vec();
                shape[3] = inputs[2].dims()[3];
                vec![(x.dtype(), shape)]
            }
            Self::GatedDeltaSlots => {
                vec![(x.dtype(), inputs[2].dims().to_vec()), like(&inputs[5])]
            }
            Self::GatedDelta { steps } => {
                let state = match steps {
                    false => inputs[5].dims().to_vec(),
                    true => {
                        let mut shape = inputs[5].dims().to_vec();
                        shape.insert(1, x.dims()[1]);
                        shape
                    }
                };
                vec![
                    (x.dtype(), inputs[2].dims().to_vec()),
                    (inputs[5].dtype(), state),
                ]
            }
            Self::GatherMm => {
                let b = inputs[1].dims();
                let mut shape = inputs[2].dims().to_vec();
                shape.extend([x.dims()[x.ndim() - 2], b[b.len() - 1]]);
                vec![(x.dtype(), shape)]
            }
            Self::GatherQmm {
                bits, transpose, ..
            } => {
                let w = inputs[1].dims();
                let n = match transpose {
                    true => w[w.len() - 2],
                    false => w[w.len() - 1] * 32 / usize::from(*bits),
                };
                let mut shape = inputs[4].dims().to_vec();
                shape.extend([x.dims()[x.ndim() - 2], n]);
                vec![(x.dtype(), shape)]
            }
        })
    }

    /// Validates the inputs of a recorded fast operation (the input count is checked
    /// already): what the public functions check before they record one, including the
    /// conditions under which they choose the kernel, so that a primitive built directly
    /// (`Array::from_primitive`) is either rejected or one the kernels and fallbacks take.
    fn check(&self, inputs: &[Array]) -> Result<()> {
        let name = self.name();
        let invalid = |msg: String| Err(Error::InvalidArgument(format!("{name}: {msg}")));
        let x = &inputs[0];
        let kernel_float = |a: &Array| matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32);
        let integer = |a: &Array| matches!(a.dtype(), DType::I32 | DType::U32);
        let last = |a: &Array| a.dims().last().copied().unwrap_or(0);
        // A scalar or a 1-D parameter of the last axis's size, in `x`'s dtype.
        let param = |p: &Array| {
            p.dtype() == x.dtype() && (p.ndim() == 0 || (p.ndim() == 1 && p.size() == last(x)))
        };
        let like = |a: &Array| a.dtype() == x.dtype() && a.dims() == x.dims();
        if !kernel_float(x) {
            return Err(Error::unsupported_dtype(name, x.dtype()));
        }
        let rotation = |dims: usize, x: &Array, freqs: Option<&Array>| {
            let d = last(x);
            dims > 0
                && dims.is_multiple_of(2)
                && dims <= d
                && freqs.is_none_or(|f| {
                    f.dtype() == DType::F32 && f.ndim() == 1 && f.size() == dims / 2
                })
        };
        let ok = match self {
            Self::RmsNorm { .. } | Self::LayerNorm { .. } => inputs[1..].iter().all(param),
            Self::AddRmsNorm { .. } => like(&inputs[1]) && param(&inputs[2]),
            Self::RmsNormVjp { .. } => param(&inputs[1]) && like(&inputs[2]),
            Self::LayerNormVjp { .. } => param(&inputs[1]) && param(&inputs[2]) && like(&inputs[3]),
            Self::Softmax | Self::LogSumExp => true,
            Self::Rope { dims, .. } => {
                let offset = &inputs[1];
                x.ndim() >= 3
                    && integer(offset)
                    && offset.ndim() <= 1
                    && (offset.size() == 1 || offset.size() == x.dims()[0])
                    && rotation(*dims, x, inputs.get(2))
            }
            Self::NormRope {
                eps,
                dims,
                offset,
                cache,
                ..
            } => {
                let at = usize::from(*cache);
                let (y, w) = (&inputs[at], &inputs[at + 1]);
                let s = y.dims();
                let Some(end) = offset.checked_add(s.get(2).copied().unwrap_or(0)) else {
                    return invalid(format!("position {offset} overflows"));
                };
                let fits = y.ndim() == 4
                    && y.dtype() == x.dtype()
                    && s[3] <= NORM_ROPE_MAX
                    && end <= i32::MAX as usize;
                let cache_fits = !cache
                    || (fits
                        && x.ndim() == 4
                        && x.dims()[..2] == s[..2]
                        && x.dims()[3] == s[3]
                        && x.dims()[2] >= end);
                let weight = || {
                    w.dtype() == x.dtype()
                        && match eps {
                            Some(_) => w.ndim() == 1 && w.size() == s[3],
                            None => w.ndim() == 0,
                        }
                };
                fits && cache_fits && weight() && rotation(*dims, y, inputs.get(at + 2))
            }
            Self::DepthwiseConv1d {
                stride,
                padding,
                dilation,
            } => {
                let w = &inputs[1];
                let span = w
                    .dims()
                    .get(1)
                    .and_then(|&k| k.checked_sub(1))
                    .and_then(|k| dilation.checked_mul(k)?.checked_add(1));
                let padded = padding
                    .checked_mul(2)
                    .and_then(|p| p.checked_add(x.dims().get(1).copied()?));
                x.ndim() == 3
                    && w.ndim() == 3
                    && w.dtype() == x.dtype()
                    && w.dims()[0] == x.dims()[2]
                    && w.dims()[2] == 1
                    && *stride > 0
                    && *dilation > 0
                    && matches!((span, padded), (Some(s), Some(p)) if p >= s)
            }
            Self::PagedAttention { .. } => {
                let [q, k, v, tables, lengths] = inputs else {
                    unreachable!("five inputs")
                };
                let (b, hq, d) = (q.dims().first(), q.dims().get(1), q.dims().get(3));
                q.ndim() == 4
                    && k.ndim() == 4
                    && k.dims() == v.dims()
                    && k.dtype() == q.dtype()
                    && v.dtype() == q.dtype()
                    && Some(&k.dims()[3]) == d
                    && k.dims()[1] > 0
                    && hq.is_some_and(|h| h % k.dims()[1] == 0)
                    && matches!(d, Some(64 | 128 | 256))
                    && k.dims()[2].is_power_of_two()
                    && k.size() < u32::MAX as usize
                    && tables.ndim() == 2
                    && tables.dims().first() == b
                    && tables.dtype() == DType::I32
                    && lengths.ndim() == 1
                    && lengths.dims().first() == b
                    && lengths.dtype() == DType::I32
            }
            Self::PagedWrite => {
                let [cache, new, slots] = inputs else {
                    unreachable!("three inputs")
                };
                cache.ndim() == 4
                    && new.ndim() == 3
                    && new.dtype() == cache.dtype()
                    && new.dims()[1] == cache.dims()[1]
                    && new.dims()[2] == cache.dims()[3]
                    && slots.dims() == [new.dims()[0]]
                    && slots.dtype() == DType::I32
            }
            Self::CrossEntropy | Self::CrossEntropyVjp => {
                let rows = &x.dims()[..x.ndim() - 1];
                let targets = &inputs[1];
                let rest = inputs[2..]
                    .iter()
                    .all(|a| a.dtype() == DType::F32 && a.dims() == rows);
                targets.dtype() == DType::I32 && targets.dims() == rows && rest
            }
            Self::Attention { mask, sinks, .. } => {
                let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
                if [q, k, v]
                    .iter()
                    .any(|a| a.ndim() != 4 || a.dtype() != x.dtype())
                {
                    return invalid("q, k and v must be 4-D arrays of one dtype".into());
                }
                let (qs, ks, vs) = (q.dims(), k.dims(), v.dims());
                let shapes = ks[0] == qs[0]
                    && vs[0] == qs[0]
                    && qs[3] == ks[3]
                    && ks[1] == vs[1]
                    && ks[2] == vs[2]
                    && ks[1] > 0
                    && qs[1] % ks[1] == 0;
                let mask_ok = !mask
                    || (inputs[3].dims() == [qs[0], qs[1], qs[2], ks[2]]
                        && matches!(inputs[3].dtype(), d if d == DType::Bool || d == x.dtype()));
                let sinks_ok = !sinks || {
                    let s = inputs.last().expect("sinks");
                    s.ndim() == 1 && s.size() == qs[1] && s.dtype() == x.dtype()
                };
                shapes && mask_ok && sinks_ok && attention_config(self, inputs)
            }
            Self::GatedDelta { .. } | Self::GatedDeltaSlots => {
                let (q, k, v, g, beta) =
                    (&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]);
                if q.ndim() != 4 || v.ndim() != 4 || q.dims() != k.dims() {
                    return invalid("q and k must be [B, T, Hk, Dk] and v [B, T, Hv, Dv]".into());
                }
                let (b, t, hk, dk) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
                let (hv, dv) = (v.dims()[2], v.dims()[3]);
                let state = match self {
                    Self::GatedDeltaSlots => {
                        let (pool, slots) = (&inputs[5], &inputs[6]);
                        pool.ndim() == 4
                            && pool.dims()[1..] == [hv, dv, dk]
                            && slots.dims() == [b]
                            && slots.dtype() == DType::I32
                            && kernel_float(pool)
                    }
                    _ => inputs[5].dims() == [b, hv, dv, dk] && kernel_float(&inputs[5]),
                };
                k.dtype() == q.dtype()
                    && v.dtype() == q.dtype()
                    && v.dims()[..2] == [b, t]
                    && hk > 0
                    && hv % hk == 0
                    && dk.is_multiple_of(32)
                    && g.dims() == [b, t, hv]
                    && beta.dims() == [b, t, hv]
                    && kernel_float(g)
                    && kernel_float(beta)
                    && state
            }
            Self::GatherMm | Self::GatherQmm { .. } => {
                let (w, lhs, rhs) = match self {
                    Self::GatherMm => (&inputs[1], &inputs[2], &inputs[3]),
                    _ => (&inputs[1], &inputs[4], &inputs[5]),
                };
                if x.ndim() < 2 || w.ndim() < 2 {
                    return invalid("operands must be [..., M, K] and [..., K, N]".into());
                }
                let (m, k) = (x.dims()[x.ndim() - 2], last(x));
                let inner = match self {
                    Self::GatherQmm {
                        group_size,
                        bits,
                        transpose,
                    } => {
                        crate::quantized::check_params(name, *group_size, u32::from(*bits))?;
                        let (scales, biases) = (&inputs[2], &inputs[3]);
                        let n = w.ndim();
                        let packing = scales.dtype() == x.dtype()
                            && biases.dtype() == x.dtype()
                            && biases.dims() == scales.dims()
                            && w.dtype() == DType::U32
                            && scales.ndim() == n
                            && w.dims()[..n - 1] == scales.dims()[..n - 1]
                            && last(w).checked_mul(32)
                                == last(scales)
                                    .checked_mul(*group_size)
                                    .and_then(|c| c.checked_mul(usize::from(*bits)));
                        // The kernel takes transposed weights (`x · wᵀ`) only.
                        let columns = last(scales) * group_size;
                        (packing && *transpose).then_some(columns)
                    }
                    _ => (w.dtype() == x.dtype()).then_some(w.dims()[w.ndim() - 2]),
                };
                let offsets = match inputs.get(if *self == Self::GatherMm { 4 } else { 6 }) {
                    None => true,
                    Some(o) => {
                        let experts: usize = w.dims()[..w.ndim() - 2].iter().product();
                        x.ndim() == 3
                            && m == 1
                            && rhs.ndim() == 1
                            && o.dtype() == DType::I32
                            && o.dims() == [experts]
                    }
                };
                inner == Some(k)
                    && lhs.dtype() == DType::U32
                    && rhs.dtype() == DType::U32
                    && lhs.dims() == rhs.dims()
                    && offsets
            }
        };
        if ok {
            Ok(())
        } else {
            invalid(format!(
                "invalid inputs {:?}",
                inputs
                    .iter()
                    .map(|a| (a.dtype(), a.dims()))
                    .collect::<Vec<_>>()
            ))
        }
    }

    /// The same computation from ordinary operations (MLX's fallbacks).
    pub(crate) fn fallback(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let x = &inputs[0];
        let f32x = || astype(x, DType::F32);
        let out = match self {
            Self::RmsNorm { eps } => {
                let xf = f32x()?;
                let n = rsqrt(&add(mean(&multiply(&xf, &xf)?, &[-1], true)?, float(*eps))?)?;
                let y = multiply(&xf, &n)?.try_astype(x.dtype())?;
                let w = &inputs[1];
                vec![if w.ndim() == 0 { y } else { multiply(&y, w)? }]
            }
            Self::AddRmsNorm { eps } => {
                let h = add(x, &inputs[1])?;
                let op = Self::RmsNorm { eps: *eps };
                let y = record_or_fallback(op, vec![h.clone(), inputs[2].clone()])?;
                vec![h, y.into_iter().next().expect("one output")]
            }
            Self::LayerNorm { eps } => {
                let xf = f32x()?;
                let xc = subtract(&xf, mean(&xf, &[-1], true)?)?;
                let v = mean(&multiply(&xc, &xc)?, &[-1], true)?;
                let mut y = multiply(&xc, rsqrt(&add(&v, float(*eps))?)?)?.try_astype(x.dtype())?;
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
                ..
            } => vec![attention_graph(
                inputs,
                float(*scale),
                *causal,
                *mask,
                *sinks,
                None,
                None,
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
            Self::NormRope {
                eps,
                dims,
                traditional,
                base,
                scale,
                offset,
                cache,
            } => {
                let at = usize::from(*cache);
                let (x, w) = (&inputs[at], &inputs[at + 1]);
                let mut h = x.clone();
                if let Some(eps) = eps {
                    let op = Self::RmsNorm { eps: *eps };
                    h = record_or_fallback(op, vec![h, w.clone()])?.swap_remove(0);
                }
                let mut rope_inputs = vec![h, Array::from(*offset as i32)];
                rope_inputs.extend(inputs.get(at + 2).cloned());
                let op = Self::Rope {
                    dims: *dims,
                    traditional: *traditional,
                    base: *base,
                    scale: *scale,
                    forward: true,
                };
                let r = record_or_fallback(op, rope_inputs)?.swap_remove(0);
                vec![match cache {
                    true => {
                        let mut region = vec![SliceArg::full(); 4];
                        let end = (offset + r.dims()[2]) as isize;
                        region[2] = SliceArg::new(Some(*offset as isize), Some(end), 1);
                        crate::ops::imp::slice_update(&inputs[0], &r, &region)?
                    }
                    false => r,
                }]
            }
            Self::DepthwiseConv1d {
                stride,
                padding,
                dilation,
            } => {
                let c = x.dims()[2];
                vec![crate::conv::general(
                    x,
                    &inputs[1],
                    (*stride, *padding, *dilation, c),
                )?]
            }
            Self::PagedAttention { scale } => vec![paged_attention_graph(inputs, float(*scale))?],
            Self::PagedWrite => vec![paged_write_graph(inputs)?],
            Self::Softmax => vec![softmax_graph(x)?],
            Self::GatedDelta { steps } => gated_delta_graph(inputs, *steps)?,
            Self::GatedDeltaSlots => {
                // The states at the slots, the rule, then the new states back at the slots.
                let (pool, slots) = (&inputs[5], &inputs[6]);
                let states = crate::indexing::take(pool, slots, Some(0))?;
                let mut args = inputs[..5].to_vec();
                args.push(states);
                let mut out = gated_delta_graph(&args, false)?;
                let new = out.pop().expect("two outputs");
                let at = broadcast_to(&reshape(slots, &[-1, 1, 1, 1])?, new.dims())?;
                let pool = crate::indexing::put_along_axis(pool, &at, &new, 0)?;
                vec![out.pop().expect("two outputs"), pool]
            }
            Self::GatherMm => {
                let rows = |a: &Array, indices: &Array| -> Result<Array> {
                    let nd = a.ndim();
                    let shape = [-1, a.dims()[nd - 2] as isize, a.dims()[nd - 1] as isize];
                    crate::indexing::take(&reshape(a, &shape)?, indices, Some(0))
                };
                vec![crate::ops::imp::matmul(
                    &rows(x, &inputs[2])?,
                    &rows(&inputs[1], &inputs[3])?,
                )?]
            }
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
                let classes = x.dims()[x.ndim() - 1];
                let y = wrap(&inputs[1], classes)?;
                let hit = one_hot(&y, classes, DType::Bool)?;
                let score = sum(&where_(&hit, x, 0.0)?, &[-1], false)?;
                let lse = squeeze(&logsumexp_graph(x)?, &[-1])?;
                vec![subtract(&lse, &score)?.try_astype(DType::F32)?]
            }
            Self::CrossEntropyVjp => {
                let classes = x.dims()[x.ndim() - 1];
                let (y, loss, g) = (wrap(&inputs[1], classes)?, &inputs[2], &inputs[3]);
                let hit = one_hot(&y, classes, DType::Bool)?;
                let score = sum(&where_(&hit, x, 0.0)?, &[-1], false)?;
                let lse = add(loss, score.try_astype(DType::F32)?)?;
                let p = exp(&subtract(&f32x()?, expand_dims(&lse, -1)?)?)?;
                let gx = multiply(
                    expand_dims(g, -1)?,
                    subtract(&p, one_hot(&y, classes, DType::F32)?)?,
                )?;
                vec![gx.try_astype(x.dtype())?]
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
            Self::AddRmsNorm { .. }
            | Self::NormRope { .. }
            | Self::PagedAttention { .. }
            | Self::PagedWrite
            | Self::DepthwiseConv1d { .. }
            | Self::RmsNormVjp { .. }
            | Self::LayerNormVjp { .. }
            | Self::CrossEntropyVjp
            | Self::Attention { .. }
            | Self::GatedDelta { .. }
            | Self::GatedDeltaSlots
            | Self::GatherQmm { .. }
            | Self::GatherMm => self.fallback_vjp(inputs, cotangents, wanted),
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
        let (_, grads) = crate::vjp_flat(f, &primals, cotangents)?;
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
        Ok(crate::jvp_flat(f, &primals, &ts)?.1)
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
                        .find_map(|(a, ax)| ax.map(|ax| a.dims()[ax]))
                        .expect("one batched input");
                    let mut shape = vec![b];
                    shape.extend_from_slice(a.dims());
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
        run_on_cpu(operands, |leaves| self.fallback(leaves))
    }
}

/// Builds a graph with `f` over evaluated `operands` (as contiguous leaves) on the CPU
/// stream, evaluates it and returns its outputs' data (contiguous): the CPU fallback of
/// fast operations and fused nodes. The thread's default stream is restored afterwards,
/// even if `f` or the evaluation panics. The caller holds no node lock (`eval` drops it
/// while computing), so the nested evaluation cannot deadlock with the outer one.
pub(crate) fn run_on_cpu(
    operands: &[Operand<'_>],
    f: impl FnOnce(&[Array]) -> Result<Vec<Array>>,
) -> Result<Vec<crate::array::Data>> {
    let leaves = operands
        .iter()
        .map(|(meta, data)| {
            let buffer = crate::kernels::contiguous(meta, data)?;
            Array::from_buffer(buffer, meta.dtype, &meta.shape)
        })
        .collect::<Result<Vec<_>>>()?;
    let outputs = {
        let _cpu = crate::device::StreamGuard::set(Stream::default_for(Device::cpu()));
        f(&leaves)?
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panicking_cpu_fallback_restores_the_default_stream() {
        let gpu = Stream::default_for(Device::gpu(0));
        crate::set_default_stream(gpu);
        let x = Array::from(1.0f32);
        let data = x.data().expect("host data").clone();
        let operands = [(x.out(), &data)];
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_on_cpu(&operands, |_| -> Result<Vec<Array>> {
                assert_eq!(default_stream().device, Device::cpu(), "runs on the CPU");
                panic!("fallback hook panics");
            })
        }));
        assert!(caught.is_err());
        assert_eq!(default_stream(), gpu, "the default stream is restored");
        crate::set_default_stream(Stream::default_for(Device::cpu()));
    }
}

/// `g` summed over every axis but the last.
fn sum_rows(g: &Array) -> Result<Array> {
    if g.ndim() <= 1 {
        // No axis to sum over (`sum` with no axes would sum over all of them).
        return Ok(g.clone());
    }
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
    let shape: Vec<isize> = x.dims().iter().map(|&d| d as isize).collect();
    let nd = x.ndim();
    let (b, t, d) = (x.dims()[0], x.dims()[nd - 2], x.dims()[nd - 1]);
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
    let (c, s) = (
        cos(&theta)?.try_astype(dtype)?,
        sin(&theta)?.try_astype(dtype)?,
    );
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
    let full = |r: Array| broadcast_to(&r, x1.dims());
    let out = update_last(&x4, &full(r1)?, 0, step)?;
    let out = update_last(&out, &full(r2)?, start2, step)?;
    reshape(&out, &shape)
}

/// The rotation of [`norm_rope`](crate::fast::norm_rope): [`rope`](crate::fast::rope)'s parameters.
#[derive(Debug, Clone, Copy)]
pub struct Rotation<'a> {
    /// Rotated features (even, at most the head size).
    pub dims: i32,
    /// Rotate interleaved pairs instead of the two halves.
    pub traditional: bool,
    /// The frequency base (exactly one of `base` and `freqs`).
    pub base: Option<f32>,
    /// The position scale.
    pub scale: f32,
    /// Per-pair frequencies (`[dims / 2]`).
    pub freqs: Option<&'a Array>,
}

/// The largest head size [`norm_rope`]'s kernel takes.
const NORM_ROPE_MAX: usize = 1024;

/// The body of [`crate::fast::norm_rope`].
pub(crate) fn norm_rope(
    x: &Array,
    norm: Option<(&Array, f32)>,
    rotation: Rotation<'_>,
    offset: usize,
    cache: Option<&Array>,
) -> Result<Array> {
    let rdims = usize::try_from(rotation.dims).map_err(|_| {
        Error::InvalidArgument(format!(
            "norm_rope: dims must not be negative, got {}",
            rotation.dims
        ))
    })?;
    let mut inputs = Vec::with_capacity(4);
    let mut fused = x.ndim() == 4
        && x.dtype().category() == Category::Float
        && x.dims()[3] <= NORM_ROPE_MAX
        && offset + x.dims()[2] <= i32::MAX as usize;
    if let Some(c) = cache {
        let s = x.dims();
        let fits = c.ndim() == 4
            && c.dims()[..2] == s[..2]
            && c.dims()[3] == s[3]
            && c.dims()[2] >= offset + s[2];
        if !fits {
            return Err(Error::InvalidArgument(format!(
                "norm_rope: a cache of shape {:?} cannot take {:?} at position {offset}",
                c.dims(),
                s
            )));
        }
        fused &= c.dtype() == x.dtype();
        inputs.push(c.clone());
    }
    if let Some((w, _)) = norm {
        fused &= w.dtype() == x.dtype() && w.ndim() == 1 && w.size() == x.dims()[x.ndim() - 1];
    }
    let freqs_valid = rotation
        .freqs
        .is_none_or(|f| f.slot_error().is_none() && f.ndim() == 1 && f.size() == rdims / 2);
    let dims_valid =
        rdims > 0 && rdims.is_multiple_of(2) && x.ndim() > 0 && rdims <= x.dims()[x.ndim() - 1];
    if !fused || !freqs_valid || !dims_valid || rotation.base.is_some() == rotation.freqs.is_some()
    {
        // The composition (which also reports invalid arguments).
        return norm_rope_graph(x, norm, rotation, offset, cache);
    }
    inputs.push(x.clone());
    inputs.push(match norm {
        Some((w, _)) => w.clone(),
        None => scalar(1.0, x.dtype())?,
    });
    if let Some(f) = rotation.freqs {
        inputs.push(astype(f, DType::F32)?);
    }
    let op = FastOp::NormRope {
        eps: norm.map(|(_, eps)| eps.to_bits()),
        dims: rdims,
        traditional: rotation.traditional,
        base: rotation.base.unwrap_or(1.0).to_bits(),
        scale: rotation.scale.to_bits(),
        offset,
        cache: cache.is_some(),
    };
    one(op, inputs)
}

/// [`norm_rope`] composed of [`rms_norm`], [`rope`] and [`crate::slice_update`].
fn norm_rope_graph(
    x: &Array,
    norm: Option<(&Array, f32)>,
    rotation: Rotation<'_>,
    offset: usize,
    cache: Option<&Array>,
) -> Result<Array> {
    let rdims = usize::try_from(rotation.dims).map_err(|_| {
        Error::InvalidArgument(format!(
            "norm_rope: dims must not be negative, got {}",
            rotation.dims
        ))
    })?;
    let x = match norm {
        Some((w, eps)) => rms_norm(x, Some(w), eps)?,
        None => x.clone(),
    };
    let r = rope(
        &x,
        rdims,
        rotation.traditional,
        rotation.base,
        rotation.scale,
        &Array::from(offset as i32),
        rotation.freqs,
    )?;
    match cache {
        Some(c) => {
            let mut region = vec![SliceArg::full(); 4];
            region[2] = SliceArg::new(
                Some(offset as isize),
                Some((offset + r.dims()[2]) as isize),
                1,
            );
            crate::ops::imp::slice_update(c, &r, &region)
        }
        None => Ok(r),
    }
}

/// The body of [`crate::fast::paged_attention`].
pub(crate) fn paged_attention(
    q: &Array,
    k_cache: &Array,
    v_cache: &Array,
    block_tables: &Array,
    lengths: &Array,
    scale: f32,
) -> Result<Array> {
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("paged_attention: {msg}")));
    floating("paged_attention", q.dtype())?;
    if q.ndim() != 4 || k_cache.ndim() != 4 || k_cache.dims() != v_cache.dims() {
        return invalid(format!(
            "q must be [B, Hq, L, D] and the caches [blocks, Hkv, block_size, D], got {:?}, \
             {:?}, {:?}",
            q.dims(),
            k_cache.dims(),
            v_cache.dims()
        ));
    }
    let (b, hq, d) = (q.dims()[0], q.dims()[1], q.dims()[3]);
    let hkv = k_cache.dims()[1];
    if k_cache.dims()[3] != d || hkv == 0 || hq % hkv != 0 {
        return invalid(format!(
            "q's heads ({hq}) must be a multiple of the caches' ({hkv}), with head size {d}"
        ));
    }
    let integer = |a: &Array| matches!(a.dtype().category(), Category::Signed | Category::Unsigned);
    if block_tables.ndim() != 2 || block_tables.dims()[0] != b || !integer(block_tables) {
        return invalid(format!(
            "block_tables must be [{b}, max_blocks] integers, got {} {:?}",
            block_tables.dtype(),
            block_tables.dims()
        ));
    }
    if lengths.dims() != [b] || !integer(lengths) {
        return invalid(format!(
            "lengths must be [{b}] integers, got {} {:?}",
            lengths.dtype(),
            lengths.dims()
        ));
    }
    if k_cache.dims()[0] == 0 {
        return invalid("the caches hold no blocks".into());
    }
    let dtype = q.dtype();
    let inputs = vec![
        q.clone(),
        astype(k_cache, dtype)?,
        astype(v_cache, dtype)?,
        astype(block_tables, DType::I32)?,
        astype(lengths, DType::I32)?,
    ];
    let op = FastOp::PagedAttention {
        scale: scale.to_bits(),
    };
    // The kernels use 32-bit offsets into the caches.
    let small = k_cache.size() < u32::MAX as usize;
    if matches!(d, 64 | 128 | 256) && k_cache.dims()[2].is_power_of_two() && small {
        one(op, inputs)
    } else {
        Ok(op.fallback(&inputs)?.swap_remove(0))
    }
}

/// The body of [`crate::fast::paged_write`].
pub(crate) fn paged_write(cache: &Array, new: &Array, slots: &Array) -> Result<Array> {
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("paged_write: {msg}")));
    let fits = cache.ndim() == 4
        && new.ndim() == 3
        && new.dims()[1] == cache.dims()[1]
        && new.dims()[2] == cache.dims()[3];
    if !fits || slots.dims() != [new.dims().first().copied().unwrap_or(0)] {
        return invalid(format!(
            "cache must be [blocks, Hkv, block_size, D], new [T, Hkv, D] and slots [T], got \
             {:?}, {:?}, {:?}",
            cache.dims(),
            new.dims(),
            slots.dims()
        ));
    }
    if !matches!(
        slots.dtype().category(),
        Category::Signed | Category::Unsigned
    ) {
        return invalid(format!("slots must be integers, got {}", slots.dtype()));
    }
    let inputs = vec![
        cache.clone(),
        astype(new, cache.dtype())?,
        astype(slots, DType::I32)?,
    ];
    one(FastOp::PagedWrite, inputs)
}

/// The keys of every listed block per sequence, `[B, Hkv, max_blocks · block_size, D]`.
/// Block ids are clamped to `0..blocks`, as the kernels do.
fn paged_keys(cache: &Array, tables: &Array) -> Result<Array> {
    let (b, max_blocks) = (tables.dims()[0], tables.dims()[1]);
    let (blocks, hkv, bs, d) = (
        cache.dims()[0],
        cache.dims()[1],
        cache.dims()[2],
        cache.dims()[3],
    );
    let last = i32::try_from(blocks.saturating_sub(1)).unwrap_or(i32::MAX);
    let ids = ops::minimum(ops::maximum(reshape(tables, &[-1])?, 0i32)?, last)?;
    let gathered = crate::indexing::take(cache, &ids, Some(0))?; // [B·M, Hkv, bs, D]
    let gathered = reshape(
        &gathered,
        &[
            b as isize,
            max_blocks as isize,
            hkv as isize,
            bs as isize,
            d as isize,
        ],
    )?;
    let gathered = crate::ops::imp::transpose(&gathered, &[0, 2, 1, 3, 4])?;
    reshape(
        &gathered,
        &[
            b as isize,
            hkv as isize,
            (max_blocks * bs) as isize,
            d as isize,
        ],
    )
}

/// [`paged_attention`] from ordinary operations: every listed block gathered, then
/// attention with a mask of each sequence's positions.
fn paged_attention_graph(inputs: &[Array], scale: f32) -> Result<Array> {
    let [q, k_cache, v_cache, tables, lengths] = inputs else {
        unreachable!("five inputs")
    };
    let (b, lq) = (q.dims()[0], q.dims()[2]);
    let (k, v) = (paged_keys(k_cache, tables)?, paged_keys(v_cache, tables)?);
    let s = k.dims()[2];
    // Lengths past the capacity see every listed position (clamped first, so that the
    // arithmetic below cannot wrap).
    let capacity = i32::try_from(s).unwrap_or(i32::MAX);
    let lengths = &ops::minimum(ops::maximum(lengths, 0i32)?, capacity)?;
    // Query `l` sees positions up to `lengths[b] − L + l`.
    let positions = reshape(
        &arange(0.0, s as f64, 1.0, DType::I32)?,
        &[1, 1, 1, s as isize],
    )?;
    let queries = reshape(
        &arange(0.0, lq as f64, 1.0, DType::I32)?,
        &[1, 1, lq as isize, 1],
    )?;
    let end = add(
        reshape(lengths, &[b as isize, 1, 1, 1])?,
        subtract(&queries, (lq - 1) as f64)?,
    )?;
    let mask = less(&positions, &end)?;
    let options = crate::fast::AttentionOptions::default().mask(&mask);
    scaled_dot_product_attention(q, &k, &v, scale, &options)
}

/// [`paged_write`] from ordinary operations: the cache as one row per slot, the rows of
/// `slots` replaced. Out-of-range slots write a scratch row past the end, dropped after.
fn paged_write_graph(inputs: &[Array]) -> Result<Array> {
    let [cache, new, slots] = inputs else {
        unreachable!("three inputs")
    };
    let (blocks, hkv, bs, d) = (
        cache.dims()[0],
        cache.dims()[1],
        cache.dims()[2],
        cache.dims()[3],
    );
    let rows = reshape(
        &crate::ops::imp::transpose(cache, &[0, 2, 1, 3])?,
        &[(blocks * bs) as isize, hkv as isize, d as isize],
    )?;
    let end = i32::try_from(blocks * bs)
        .map_err(|_| Error::InvalidArgument("paged_write: more than 2^31 slots".into()))?;
    let slots = where_(
        less(slots, 0i32)?,
        end,
        where_(less(slots, end)?, slots, end)?,
    )?;
    let scratch = zeros(&[1, hkv, d], rows.dtype())?;
    let rows = crate::indexing::concatenate(&[rows, scratch], 0)?;
    let at = broadcast_to(&reshape(&slots, &[-1, 1, 1])?, new.dims())?;
    let rows = crate::indexing::put_along_axis(&rows, &at, new, 0)?;
    let rows = slice(
        &rows,
        &[
            SliceArg::new(Some(0), Some(end as isize), 1),
            SliceArg::full(),
            SliceArg::full(),
        ],
    )?;
    let rows = reshape(
        &rows,
        &[blocks as isize, bs as isize, hkv as isize, d as isize],
    )?;
    crate::ops::imp::transpose(&rows, &[0, 2, 1, 3])
}

/// The routing key of an attention call from its shapes, dtype and options, validated as
/// the op validates them (the mask's and the sinks' shapes and dtypes).
pub(crate) fn attention_key(
    qs: &[usize],
    ks: &[usize],
    vs: &[usize],
    dtype: DType,
    options: &crate::fast::AttentionOptions<'_>,
    target: Target,
) -> Result<RouteKey> {
    let invalid = |msg: String| {
        Err(Error::InvalidArgument(format!(
            "scaled_dot_product_attention: {msg}"
        )))
    };
    if [qs, ks, vs].iter().any(|s| s.len() != 4) {
        return invalid(format!(
            "q, k and v must be 4-D, got {qs:?}, {ks:?} and {vs:?}"
        ));
    }
    if ks[0] != qs[0] || vs[0] != qs[0] || qs[3] != ks[3] || ks[1] != vs[1] || ks[2] != vs[2] {
        return invalid(format!("incompatible shapes {qs:?}, {ks:?} and {vs:?}"));
    }
    if ks[1] == 0 || !qs[1].is_multiple_of(ks[1]) {
        return invalid(format!(
            "the query heads ({}) must be a multiple of the key heads ({})",
            qs[1], ks[1]
        ));
    }
    floating("scaled_dot_product_attention", dtype)?;
    let (mask, bool_mask) = match options.mask {
        AttentionMask::None => (MaskKind::None, false),
        AttentionMask::Causal => (MaskKind::Causal, false),
        AttentionMask::Array(m) => {
            if m.ndim() > 4 {
                return invalid(format!(
                    "the mask must have at most 4 dimensions, got {:?}",
                    m.dims()
                ));
            }
            if m.dtype() != DType::Bool && crate::promote_types(m.dtype(), dtype) != dtype {
                return invalid(format!("a {} mask does not promote to {dtype}", m.dtype()));
            }
            let full = [qs[0], qs[1], qs[2], ks[2]];
            let fits = m
                .dims()
                .iter()
                .rev()
                .zip(full.iter().rev())
                .all(|(&a, &b)| a == b || a == 1);
            if !fits {
                return invalid(format!(
                    "the mask {:?} does not broadcast to {full:?}",
                    m.dims()
                ));
            }
            (MaskKind::Array, m.dtype() == DType::Bool)
        }
    };
    if let Some(s) = options.sinks
        && (s.ndim() != 1 || s.size() != qs[1] || crate::promote_types(s.dtype(), dtype) != dtype)
    {
        return invalid(format!(
            "sinks must be [{}] of a type promoting to {dtype}",
            qs[1]
        ));
    }
    let window = match options.window {
        Some(w) if w < 1 => return invalid(format!("the window must be at least 1, got {w}")),
        Some(w) => Some(w as usize),
        None => None,
    };
    Ok(RouteKey {
        b: qs[0],
        hq: qs[1],
        hkv: ks[1],
        lq: qs[2],
        lk: ks[2],
        d: qs[3],
        dv: vs[3],
        dtype,
        mask,
        bool_mask,
        sinks: options.sinks.is_some(),
        window,
        cap: options.max_score_bytes,
        target,
    })
}

/// Where attention over `dtype` runs on the current stream.
fn attention_target(dtype: DType, size: usize) -> Target {
    if on_gpu(dtype, size.max(1)) {
        Target::Gpu {
            class: crate::gpu::class(),
        }
    } else {
        Target::Cpu
    }
}

/// The body of [`crate::fast::plan_attention`].
pub(crate) fn plan_attention(
    q: &[i32],
    k: &[i32],
    v: &[i32],
    dtype: DType,
    options: &crate::fast::AttentionOptions<'_>,
) -> Result<crate::fast::AttentionPlan> {
    let dims = |s: &[i32]| -> Result<Vec<usize>> {
        s.iter()
            .map(|&d| {
                usize::try_from(d).map_err(|_| {
                    Error::InvalidArgument(format!(
                        "plan_attention: dimensions must not be negative, got {s:?}"
                    ))
                })
            })
            .collect()
    };
    let (qs, ks, vs) = (dims(q)?, dims(k)?, dims(v)?);
    let size = qs.iter().product();
    let key = attention_key(&qs, &ks, &vs, dtype, options, attention_target(dtype, size))?;
    let plan = plan::plan(&key)?;
    let as_i32 = |n: usize| i32::try_from(n).unwrap_or(i32::MAX);
    let kernel = match &plan.route {
        Route::Vector(groups) => {
            let (mut passes, mut blocks) = (1u8, 1usize);
            for g in groups {
                if let GroupKind::Vector {
                    passes: p,
                    blocks: b,
                    ..
                } = g.kind
                {
                    passes = passes.max(p);
                    blocks = blocks.max(b);
                }
            }
            crate::fast::AttentionKernel::Vector {
                passes,
                blocks: as_i32(blocks),
                dispatches: as_i32(groups.len()),
            }
        }
        Route::Tiled => crate::fast::AttentionKernel::Tiled,
        Route::Composed { rows } if *rows < key.lq => {
            crate::fast::AttentionKernel::ComposedBlocks {
                rows: as_i32(*rows),
            }
        }
        Route::Composed { .. } => crate::fast::AttentionKernel::Composed,
    };
    Ok(crate::fast::AttentionPlan {
        kernel,
        scratch_bytes: plan.scratch,
        max_intermediate_bytes: plan.max_intermediate,
        capped: plan.capped,
    })
}

/// The body of [`crate::fast::scaled_dot_product_attention`]: validates, plans, and runs
/// the plan.
pub(crate) fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    options: &crate::fast::AttentionOptions<'_>,
) -> Result<Array> {
    if [q, k, v].iter().any(|a| a.ndim() != 4) {
        return Err(Error::InvalidArgument(format!(
            "scaled_dot_product_attention: q, k and v must be 4-D, got {:?}, {:?} and {:?}",
            q.dims(),
            k.dims(),
            v.dims()
        )));
    }
    let dtype = crate::promote_types(crate::promote_types(q.dtype(), k.dtype()), v.dtype());
    let key = attention_key(
        q.dims(),
        k.dims(),
        v.dims(),
        dtype,
        options,
        attention_target(dtype, q.size()),
    )?;
    let plan = plan::plan(&key)?;
    let (qs, ks) = (q.dims(), k.dims());
    let (q, k, v) = (astype(q, dtype)?, astype(k, dtype)?, astype(v, dtype)?);
    let mask = match options.mask {
        AttentionMask::Array(m) => {
            let m = if m.dtype() == DType::Bool {
                m.clone()
            } else {
                astype(m, dtype)?
            };
            Some(broadcast_to(&m, &[qs[0], qs[1], qs[2], ks[2]])?)
        }
        _ => None,
    };
    let sinks = match options.sinks {
        Some(s) => Some(astype(s, dtype)?),
        None => None,
    };
    let call = Call {
        key: &key,
        scale,
        q: &q,
        k: &k,
        v: &v,
        mask: mask.as_ref(),
        sinks: sinks.as_ref(),
    };
    match &plan.route {
        Route::Tiled => call.fused(0, key.lq, key.lk, Kernel::Tiled),
        Route::Vector(groups) => {
            if let [g] = groups.as_slice()
                && g.start == 0
                && g.end == key.lq
                && g.keys == key.lk
            {
                return call.group(g);
            }
            let outs = groups
                .iter()
                .map(|g| call.group(g))
                .collect::<Result<Vec<_>>>()?;
            crate::indexing::concatenate(&outs, 2)
        }
        Route::Composed { rows } => {
            let rows = (*rows).max(1);
            let mut outs: Vec<Array> = Vec::with_capacity(key.lq.div_ceil(rows));
            let mut start = 0;
            while start < key.lq {
                let end = (start + rows).min(key.lq);
                // Each block's queries wait for the previous block's output, so one
                // block's scores are live at a time.
                let after = outs.last().cloned();
                outs.push(call.composed(start, end, after)?);
                start = end;
            }
            if outs.len() == 1 {
                Ok(outs.pop().expect("one block"))
            } else {
                crate::indexing::concatenate(&outs, 2)
            }
        }
    }
}

/// A kernel choice for [`Call::fused`].
#[derive(Clone, Copy)]
enum Kernel {
    Tiled,
    Vector {
        passes: u8,
        blocks: usize,
        gqa_kernel: bool,
        tk1: bool,
    },
}

/// One validated attention call, ready to run parts of.
struct Call<'a> {
    key: &'a RouteKey,
    scale: f32,
    q: &'a Array,
    k: &'a Array,
    v: &'a Array,
    /// Broadcast to `[B, Hq, Lq, Lk]`.
    mask: Option<&'a Array>,
    sinks: Option<&'a Array>,
}

impl Call<'_> {
    /// Queries `rows`, keys `keys` and the mask's matching part.
    fn slices(
        &self,
        rows: (usize, usize),
        keys: (usize, usize),
    ) -> Result<(Array, Array, Array, Option<Array>)> {
        let range = |(a, b): (usize, usize)| SliceArg::new(Some(a as isize), Some(b as isize), 1);
        let whole_q = rows == (0, self.key.lq);
        let whole_k = keys == (0, self.key.lk);
        let full = SliceArg::full;
        let q = match whole_q {
            true => self.q.clone(),
            false => slice(self.q, &[full(), full(), range(rows), full()])?,
        };
        let (k, v) = match whole_k {
            true => (self.k.clone(), self.v.clone()),
            false => (
                slice(self.k, &[full(), full(), range(keys), full()])?,
                slice(self.v, &[full(), full(), range(keys), full()])?,
            ),
        };
        let mask = match self.mask {
            Some(m) if !(whole_q && whole_k) => {
                Some(slice(m, &[full(), full(), range(rows), range(keys)])?)
            }
            other => other.cloned(),
        };
        Ok((q, k, v, mask))
    }

    /// The fused kernel over query rows `start..start + rows` and the first `keys` keys
    /// (`rows`, `keys` the whole call when `start == 0` and they are `Lq`, `Lk`).
    fn fused(&self, start: usize, end: usize, keys: usize, kernel: Kernel) -> Result<Array> {
        let (q, k, v, mask) = self.slices((start, end), (0, keys))?;
        let (passes, blocks, gqa_kernel, tk1) = match kernel {
            Kernel::Tiled => (0, 1, false, false),
            Kernel::Vector {
                passes,
                blocks,
                gqa_kernel,
                tk1,
            } => (passes, blocks, gqa_kernel, tk1),
        };
        let op = FastOp::Attention {
            scale: self.scale.to_bits(),
            causal: self.key.mask == MaskKind::Causal,
            mask: mask.is_some(),
            sinks: self.sinks.is_some(),
            passes,
            blocks: blocks as u32,
            gqa_kernel,
            tk1,
        };
        let mut inputs = vec![q, k, v];
        inputs.extend(mask);
        inputs.extend(self.sinks.cloned());
        Ok(Array::from_op(Primitive::Fast(op), inputs)?.swap_remove(0))
    }

    /// One dispatch group of the vector schedule.
    fn group(&self, g: &plan::Group) -> Result<Array> {
        match g.kind {
            GroupKind::Vector {
                passes,
                blocks,
                gqa_kernel,
                tk1,
            } => self.fused(
                g.start,
                g.end,
                g.keys,
                Kernel::Vector {
                    passes,
                    blocks,
                    gqa_kernel,
                    tk1,
                },
            ),
            GroupKind::Composed => {
                let (q, k, v, mask) = self.slices((g.start, g.end), (0, g.keys))?;
                let mut inputs = vec![q, k, v];
                let has_mask = mask.is_some();
                inputs.extend(mask);
                inputs.extend(self.sinks.cloned());
                attention_graph(
                    &inputs,
                    self.scale,
                    self.key.mask == MaskKind::Causal,
                    has_mask,
                    self.sinks.is_some(),
                    None,
                    None,
                )
            }
        }
    }

    /// The composition over query rows `start..end`, with the keys those rows see (the
    /// causal prefix, cut to the window), evaluated after `after` when given.
    fn composed(&self, start: usize, end: usize, after: Option<Array>) -> Result<Array> {
        let key = self.key;
        let causal = key.mask == MaskKind::Causal;
        // Under a causal mask, row `i` sees keys up to `i + Lk − Lq`.
        let hi = match causal {
            true => (key.lk + end).saturating_sub(key.lq).clamp(1, key.lk),
            false => key.lk,
        };
        // The window of row `start` begins at key `start + Lk − Lq − window + 1`.
        let lo = match key.window {
            Some(w) => (start + key.lk).saturating_sub(key.lq + w - 1).min(hi - 1),
            None => 0,
        };
        let offset = (start + key.lk) as isize - key.lq as isize - lo as isize;
        let (mut q, k, v, mask) = self.slices((start, end), (lo, hi))?;
        if let Some(prev) = after {
            q = crate::primitive::depends(&q, &[prev])?;
        }
        let mut inputs = vec![q, k, v];
        let has_mask = mask.is_some();
        inputs.extend(mask);
        inputs.extend(self.sinks.cloned());
        attention_graph(
            &inputs,
            self.scale,
            causal,
            has_mask,
            self.sinks.is_some(),
            Some(offset),
            key.window,
        )
    }
}

/// Whether a recorded attention node's kernel choice fits its inputs (the planner's
/// choices always do; this guards nodes built through `graph::from_primitive`).
fn attention_config(op: &FastOp, inputs: &[Array]) -> bool {
    let FastOp::Attention {
        causal,
        mask,
        sinks,
        passes,
        blocks,
        gqa_kernel,
        tk1,
        ..
    } = *op
    else {
        return false;
    };
    let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
    let (lq, lk, d, dv) = (q.dims()[2], k.dims()[2], q.dims()[3], v.dims()[3]);
    let gqa = q.dims()[1] / k.dims()[1];
    if lq == 0 || lk == 0 {
        return false;
    }
    match passes {
        0 => plan::tiled_head(d, dv) && (!causal || lq <= lk) && !gqa_kernel && !tk1,
        1 | 2 => {
            let vector = lq <= 8 && lq <= lk && plan::vector_head(d, dv) && (d != 512 || lq == 1);
            let two = passes == 1
                || ((1..=1024).contains(&blocks)
                    && 32 * gqa * lq <= 1024
                    && (!gqa_kernel
                        || (lq == 1
                            && !mask
                            && !sinks
                            && d == dv
                            && ((gqa == 8 && (d == 64 || d == 128))
                                || ((gqa == 12 || gqa == 16) && d == 128))))
                    && (!tk1 || (!gqa_kernel && plan::tk1_variant(q.dtype(), d, dv))));
            vector && two && (passes == 2 || (!gqa_kernel && !tk1))
        }
        _ => false,
    }
}

/// MLX's attention fallback, with sinks folded into the softmax.
///
/// Under `causal`, query `i` sees keys up to `i + offset` (`offset` defaults to `Lk − Lq`,
/// the queries being the last positions), and with a `window` only the `window` keys up
/// to that one.
fn attention_graph(
    inputs: &[Array],
    scale: f32,
    causal: bool,
    has_mask: bool,
    has_sinks: bool,
    offset: Option<isize>,
    window: Option<usize>,
) -> Result<Array> {
    let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
    let dtype = q.dtype();
    let (b, hq, lq) = (q.dims()[0], q.dims()[1], q.dims()[2]);
    let (hkv, lk) = (k.dims()[1], k.dims()[2]);
    let rep = hq / hkv;
    let i = |n: usize| n as isize;
    let mut q = multiply(q, f64::from(scale))?;
    let (mut k, mut v) = (k.clone(), v.clone());
    if rep > 1 {
        q = reshape(&q, &[i(b), i(hkv), i(rep), i(lq), i(q.dims()[3])])?;
        k = expand_dims(&k, 2)?;
        v = expand_dims(&v, 2)?;
    }
    let n = k.ndim();
    let mut axes: Vec<usize> = (0..n).collect();
    axes.swap(n - 2, n - 1);
    let mut scores = crate::ops::imp::matmul(&q, &crate::ops::imp::transpose(&k, &axes)?)?;
    let mask = if causal {
        let offset = offset.unwrap_or(lk as isize - lq as isize) as f64;
        let qi = expand_dims(&arange(offset, offset + lq as f64, 1.0, DType::I32)?, 1)?;
        let ki = expand_dims(&arange(0.0, lk as f64, 1.0, DType::I32)?, 0)?;
        let seen = crate::ops::imp::greater_equal(&qi, &ki)?;
        Some(match window {
            // Key `j` is in the window of query position `p` when `j > p − window`.
            Some(w) => multiply(
                &seen,
                &crate::ops::imp::greater(&ki, &subtract(&qi, w as f64)?)?,
            )?,
            None => seen,
        })
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
        let m = stop_gradient(&crate::ops::imp::maximum(&max(&s, &[-1], true)?, &sink)?)?;
        let e = exp(&subtract(&s, &m)?)?;
        let denom = add(sum(&e, &[-1], true)?, exp(&subtract(&sink, &m)?)?)?;
        astype(&divide(&e, &denom)?, dtype)?
    } else if on_gpu(dtype, scores.size()) {
        crate::ops::imp::softmax(&scores, &[-1])?
    } else {
        softmax_graph(&scores)?
    };
    let out = crate::ops::imp::matmul(&probs, &v)?;
    if rep > 1 {
        reshape(&out, &[i(b), i(hq), i(lq), i(v.dims()[n - 1])])
    } else {
        Ok(out)
    }
}

/// The body of [`crate::fast::gated_delta`].
pub(crate) fn gated_delta(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    gated_delta_op(q, k, v, g, beta, state, false)
}

/// The body of [`crate::fast::gated_delta_steps`].
pub(crate) fn gated_delta_steps(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    gated_delta_op(q, k, v, g, beta, state, true)
}

/// The body of [`crate::fast::gated_delta_slots`].
pub(crate) fn gated_delta_slots(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    pool: &Array,
    slots: &Array,
) -> Result<(Array, Array)> {
    let b = q.dims().first().copied().unwrap_or(0);
    let fits = pool.ndim() == 4
        && slots.dims() == [b]
        && matches!(
            slots.dtype().category(),
            Category::Signed | Category::Unsigned
        );
    if !fits {
        return Err(Error::InvalidArgument(format!(
            "gated_delta_slots: pool must be [slots, Hv, Dv, Dk] and slots [{b}] integers, got \
             {:?} and {} {:?}",
            pool.dims(),
            slots.dtype(),
            slots.dims()
        )));
    }
    // The checks of `gated_delta`, on one slot's state.
    let one = crate::indexing::take(pool, &reshape(&astype(slots, DType::I32)?, &[-1])?, Some(0))?;
    let _validated = gated_delta_op(q, k, v, g, beta, &one, false)?;
    let inputs = vec![
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        pool.clone(),
        astype(slots, DType::I32)?,
    ];
    let kernel = on_gpu(q.dtype(), q.size())
        && q.dtype() == v.dtype()
        && q.dims()[3].is_multiple_of(32)
        && [g, beta, pool]
            .iter()
            .all(|a| matches!(a.dtype(), DType::F16 | DType::BF16 | DType::F32));
    let op = FastOp::GatedDeltaSlots;
    let mut out = if kernel {
        Array::from_op(Primitive::Fast(op), inputs)?
    } else {
        op.fallback(&inputs)?
    };
    let pool = out.pop().expect("two outputs");
    Ok((out.pop().expect("two outputs"), pool))
}

fn gated_delta_op(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    steps: bool,
) -> Result<(Array, Array)> {
    let op = "gated_delta";
    for a in [q, k, v, g, beta, state] {
        floating(op, a.dtype())?;
    }
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("{op}: {msg}")));
    if q.ndim() != 4 || q.dims() != k.dims() || v.ndim() != 4 {
        return invalid(format!(
            "q and k must be [B, T, Hk, Dk] and v [B, T, Hv, Dv], got {:?}, {:?}, {:?}",
            q.dims(),
            k.dims(),
            v.dims()
        ));
    }
    let (b, t, hk, dk) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let (hv, dv) = (v.dims()[2], v.dims()[3]);
    if v.dims()[..2] != [b, t] || hk == 0 || hv % hk != 0 {
        return invalid(format!(
            "v must be [{b}, {t}, Hv, Dv] with Hv a multiple of {hk}, got {:?}",
            v.dims()
        ));
    }
    if g.dims() != [b, t, hv] || beta.dims() != [b, t, hv] {
        return invalid(format!(
            "g and beta must be [{b}, {t}, {hv}], got {:?} and {:?}",
            g.dims(),
            beta.dims()
        ));
    }
    if state.dims() != [b, hv, dv, dk] {
        return invalid(format!(
            "state must be [{b}, {hv}, {dv}, {dk}], got {:?}",
            state.dims()
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
    let op = FastOp::GatedDelta { steps };
    let mut out = if kernel {
        Array::from_op(Primitive::Fast(op), inputs)?
    } else {
        op.fallback(&inputs)?
    };
    let s = out.pop().expect("two outputs");
    Ok((out.pop().expect("two outputs"), s))
}

/// The gated delta rule step by step from ordinary operations (mlx-lm's
/// `gated_delta_ops`).
fn gated_delta_graph(inputs: &[Array], steps: bool) -> Result<Vec<Array>> {
    let [q, k, v, g, beta, state] = inputs else {
        unreachable!("six inputs")
    };
    let (b, t, hk, dk) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let (hv, dv) = (v.dims()[2], v.dims()[3]);
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
    let mut states = Vec::with_capacity(if steps { t } else { 0 });
    for i in 0..t {
        let (qt, kt, vt) = (at(&qh, i)?, expand_dims(&at(&kh, i)?, 2)?, at(v, i)?);
        let decay = reshape(&at(g, i)?, &[b as isize, hv as isize, 1, 1])?;
        s = multiply(&s, &decay)?;
        let kv_mem = sum(&multiply(&s, &kt)?, &[-1], false)?;
        let delta = multiply(subtract(&vt, &kv_mem)?, expand_dims(&at(beta, i)?, -1)?)?;
        s = add(&s, multiply(&kt, expand_dims(&delta, -1)?)?)?;
        let y = sum(&multiply(&s, expand_dims(&qt, 2)?)?, &[-1], false)?;
        ys.push(astype(&y, q.dtype())?);
        if steps {
            states.push(astype(&s, state.dtype())?);
        }
    }
    let y = match ys.is_empty() {
        true => zeros(&[b, 0, hv, dv], q.dtype())?,
        false => stack(&ys, 1)?,
    };
    let s = match (steps, states.is_empty()) {
        (false, _) => astype(&s, state.dtype())?,
        (true, false) => stack(&states, 1)?,
        (true, true) => zeros(&[b, 0, hv, dv, dk], state.dtype())?,
    };
    Ok(vec![y, s])
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
            &[-1, a.dims()[nd - 2] as isize, a.dims()[nd - 1] as isize],
        )?;
        crate::indexing::take(&flat, indices, Some(0))
    };
    crate::quantized::quantized_matmul(
        &rows(&inputs[0], lhs)?,
        &rows(&inputs[1], rhs)?,
        &rows(&inputs[2], rhs)?,
        &rows(&inputs[3], rhs)?,
        transpose,
        group_size,
        bits,
    )
}
