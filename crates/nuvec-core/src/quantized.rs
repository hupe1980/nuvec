//! Affine quantization, as in MLX: weights are stored with `bits` bits per value and one
//! scale and bias per group of `group_size` values along the last axis.
//!
//! The packed form is MLX's, so weights quantized by either library load in the other:
//! codes form a little-endian bit stream per row, stored as uint32; scales and biases keep
//! the weights' dtype. A value `w` is stored as `round((w - bias) / scale)`, clamped to
//! `[0, 2^bits - 1]`.

use crate::ops::imp::{astype, broadcast_to, expand_dims, squeeze};
use crate::primitive::Primitive;
use crate::{Array, Category, Error, Result, promote_types, view};

/// The supported group sizes and bit widths (MLX's).
pub(crate) fn check_params(op: &str, group_size: usize, bits: u32) -> Result<u8> {
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(Error::InvalidArgument(format!(
            "{op}: group size {group_size} is not supported (32, 64 or 128)"
        )));
    }
    match bits {
        1..=6 | 8 => Ok(bits as u8),
        _ => Err(Error::InvalidArgument(format!(
            "{op}: {bits} bits are not supported (1, 2, 3, 4, 5, 6 or 8)"
        ))),
    }
}

/// Quantizes the last axis of `w` (float16, bfloat16 or float32, at least 2-D, the last
/// dimension divisible by `group_size`) in groups of `group_size` (32, 64 or 128) values with
/// `bits` (1–6 or 8) bits each. Returns the packed codes (uint32, last dimension
/// `· bits / 32`), the scales and the biases (`w`'s dtype, last dimension `/ group_size`),
/// equal to MLX's bit for bit.
pub(crate) fn quantize(w: &Array, group_size: usize, bits: u32) -> Result<(Array, Array, Array)> {
    let bits = check_params("quantize", group_size, bits)?;
    let mut out = Array::from_op(Primitive::Quantize { group_size, bits }, vec![w.clone()])?;
    let biases = out.pop().expect("three outputs");
    let scales = out.pop().expect("three outputs");
    Ok((out.pop().expect("three outputs"), scales, biases))
}

/// Restores the values packed by [`quantize`]: `code · scale + bias`, in the scales' dtype.
pub(crate) fn dequantize(
    w: &Array,
    scales: &Array,
    biases: &Array,
    group_size: usize,
    bits: u32,
) -> Result<Array> {
    let bits = check_params("dequantize", group_size, bits)?;
    let inputs = vec![w.clone(), scales.clone(), biases.clone()];
    Ok(Array::from_op(Primitive::Dequantize { group_size, bits }, inputs)?.swap_remove(0))
}

/// The matrix product of `x` with quantized weights, `x · dequantize(w)ᵀ` if `transpose`
/// (MLX's default, for weights stored as `[out, in]`) or `x · dequantize(w)`, without
/// materialising the dequantized matrix.
///
/// As in MLX, `x`, `scales` and `biases` are promoted to one floating-point type, and batch
/// dimensions broadcast when `x` and `w` both have more than two. A one-dimensional `x` is a
/// row vector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    transpose: bool,
    group_size: usize,
    bits: u32,
) -> Result<Array> {
    let bits = check_params("quantized_matmul", group_size, bits)?;
    let dtype = promote_types(promote_types(x.dtype(), scales.dtype()), biases.dtype());
    if dtype.category() != Category::Float || dtype.size() > 4 {
        return Err(Error::unsupported_dtype("quantized_matmul", dtype));
    }
    let (x, scales, biases) = (
        astype(x, dtype)?,
        astype(scales, dtype)?,
        astype(biases, dtype)?,
    );
    let vector = x.ndim() == 1;
    let mut x = if vector { expand_dims(&x, 0)? } else { x };
    let (mut w, mut scales, mut biases) = (w.clone(), scales, biases);
    let (xn, wn) = (x.ndim(), w.ndim());
    if wn > 2 && xn >= 2 {
        // Broadcast the batch dimensions of every operand together (MLX broadcasts them only
        // when both have batch dimensions; nuvec also lifts a 2-D `x` to `w`'s batch).
        let batch =
            view::broadcast_shapes("quantized_matmul", &x.dims()[..xn - 2], &w.dims()[..wn - 2])?;
        let full = |a: &Array| -> Result<Array> {
            let mut shape = batch.to_vec();
            shape.extend_from_slice(&a.dims()[a.ndim() - 2..]);
            broadcast_to(a, &shape)
        };
        x = full(&x)?;
        w = full(&w)?;
        scales = full(&scales)?;
        biases = full(&biases)?;
    }
    let primitive = Primitive::QuantizedMatmul {
        group_size,
        bits,
        transpose,
    };
    let out = Array::from_op(primitive, vec![x, w, scales, biases])?.swap_remove(0);
    if vector {
        squeeze(&out, &[-2])
    } else {
        Ok(out)
    }
}

/// Quantized matrix products of gathered batches, as MLX's `gather_qmm` (mixture-of-experts
/// layers): output batch `i` is `x[lhs_indices[i]] · dequantize(w)[rhs_indices[i]]ᵀ` (or
/// without the transpose), indexing the flattened batch dimensions of `x` (`[..., M, K]`)
/// and of `w`, `scales` and `biases`. The indices (integers) broadcast together; either
/// defaults to every batch of its operand in order. `sorted_indices` promises that
/// `rhs_indices` is sorted (as after grouping rows by expert), which lets one kernel walk
/// each weight batch once.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    transpose: bool,
    group_size: usize,
    bits: u32,
    sorted_indices: bool,
) -> Result<Array> {
    let bits = check_params("gather_qmm", group_size, bits)?;
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("gather_qmm: {msg}")));
    let dtype = promote_types(promote_types(x.dtype(), scales.dtype()), biases.dtype());
    if dtype.category() != Category::Float || dtype.size() > 4 {
        return Err(Error::unsupported_dtype("gather_qmm", dtype));
    }
    if x.ndim() < 2 || w.ndim() < 2 || scales.dims() != biases.dims() {
        return invalid(format!(
            "x must be [..., M, K] and w [..., N, K'], got {:?} and {:?}",
            x.dims(),
            w.dims()
        ));
    }
    let (x, scales, biases) = (
        astype(x, dtype)?,
        astype(scales, dtype)?,
        astype(biases, dtype)?,
    );
    let (lhs, rhs) = gather_indices("gather_qmm", &x, w, lhs_indices, rhs_indices)?;
    let mut inputs = vec![x.clone(), w.clone(), scales, biases, lhs, rhs.clone()];
    // MLX's rule (`GatherQMM::eval_gpu`): the sorted kernel tiles each expert's rows, so it
    // pays only with at least 16 rows and 4 per expert on average; fewer take one product
    // per row. nuvec also keeps one product per row up to `ROW_INDEPENDENT_ROWS` tokens of
    // top-8 routing, so rows stay independent (the row-independence contract).
    let experts: usize = w.dims()[..w.ndim() - 2].iter().product();
    let grouped = rhs.size() > ROW_TOKENS && rhs.size() >= 16 && rhs.size() / experts.max(1) >= 4;
    let sorted = lhs_indices.is_none() && sorted_indices && grouped;
    if let Some(offsets) = sorted_offsets(&x, w, &rhs, sorted)? {
        inputs.push(offsets);
    }
    let op = crate::fast::imp::FastOp::GatherQmm {
        group_size,
        bits,
        transpose,
    };
    let kernel = transpose && crate::fast::imp::on_gpu(dtype, x.size());
    let out = if kernel {
        Array::from_op(Primitive::Fast(op), inputs)?.swap_remove(0)
    } else {
        op.fallback(&inputs)?.swap_remove(0)
    };
    Ok(out)
}

/// Gathered rows up to which every row runs as its own product: `ROW_INDEPENDENT_ROWS`
/// tokens with up to 8 experts each.
pub(crate) const ROW_TOKENS: usize = crate::ROW_INDEPENDENT_ROWS * 8;

/// The batch indices of a gathered product, broadcast together; either defaults to every
/// batch of its operand, in order.
fn gather_indices(
    op: &'static str,
    a: &Array,
    b: &Array,
    lhs: Option<&Array>,
    rhs: Option<&Array>,
) -> Result<(Array, Array)> {
    let indices = |x: &Array, given: Option<&Array>| -> Result<Array> {
        let batch = &x.dims()[..x.ndim() - 2];
        match given {
            Some(i)
                if i.dtype().category() == Category::Float || i.dtype() == crate::DType::Bool =>
            {
                Err(Error::InvalidArgument(format!(
                    "{op}: indices must be integers"
                )))
            }
            Some(i) => astype(i, crate::DType::U32),
            None => {
                let n: usize = batch.iter().product();
                let all = crate::ops::imp::arange(0.0, n as f64, 1.0, crate::DType::U32)?;
                all.try_reshape(&batch.iter().map(|&d| d as isize).collect::<Vec<_>>())
            }
        }
    };
    let (l, r) = (indices(a, lhs)?, indices(b, rhs)?);
    let batch = view::broadcast_shapes(op, l.dims(), r.dims())?;
    Ok((broadcast_to(&l, &batch)?, broadcast_to(&r, &batch)?))
}

/// For rows sorted by weight batch (`a` is `[rows, 1, K]`, one row per output batch), where
/// each batch's rows start: the rows per batch (scatter-added ones), then their exclusive
/// running sum (MLX's grouped kernels walk every weight batch once).
fn sorted_offsets(a: &Array, b: &Array, rhs: &Array, sorted: bool) -> Result<Option<Array>> {
    let m = a.dims()[a.ndim() - 2];
    if !(sorted && m == 1 && a.ndim() == 3 && rhs.ndim() == 1) {
        return Ok(None);
    }
    let experts: usize = b.dims()[..b.ndim() - 2].iter().product();
    let ones = crate::ops::imp::ones(rhs.dims(), crate::DType::I32)?;
    let zeros = crate::ops::imp::zeros(&[experts], crate::DType::I32)?;
    let counts = crate::indexing::scatter_add_along_axis(&zeros, rhs, &ones, 0)?;
    Ok(Some(crate::sorting::cumsum(
        &counts,
        Some(0),
        false,
        false,
    )?))
}

/// Matrix products of gathered batches, as MLX's `gather_mm` (mixture-of-experts layers in
/// full precision): output batch `i` is `a[lhs_indices[i]] · b[rhs_indices[i]]` over the
/// flattened batch dimensions of `a` (`[..., M, K]`) and `b` (`[..., K, N]`), with the
/// indices broadcast together and either defaulting to every batch. `sorted_indices`
/// promises sorted `rhs_indices`, which lets one kernel walk each batch of `b` once.
pub(crate) fn gather_mm(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
) -> Result<Array> {
    let dtype = promote_types(a.dtype(), b.dtype());
    if dtype.category() != Category::Float {
        return Err(Error::unsupported_dtype("gather_mm", dtype));
    }
    if a.ndim() < 2 || b.ndim() < 2 || a.dims()[a.ndim() - 1] != b.dims()[b.ndim() - 2] {
        return Err(Error::InvalidArgument(format!(
            "gather_mm: a must be [..., M, K] and b [..., K, N], got {:?} and {:?}",
            a.dims(),
            b.dims()
        )));
    }
    let (a, b) = (astype(a, dtype)?, astype(b, dtype)?);
    let (lhs, rhs) = gather_indices("gather_mm", &a, &b, lhs_indices, rhs_indices)?;
    let mut inputs = vec![a.clone(), b.clone(), lhs, rhs.clone()];
    let grouped = rhs.size() > ROW_TOKENS;
    if let Some(offsets) = sorted_offsets(
        &a,
        &b,
        &rhs,
        lhs_indices.is_none() && sorted_indices && grouped,
    )? {
        inputs.push(offsets);
    }
    Ok(
        crate::fast::imp::record_or_fallback(crate::fast::imp::FastOp::GatherMm, inputs)?
            .swap_remove(0),
    )
}
