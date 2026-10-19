//! Affine quantization, as in MLX: weights are stored with `bits` bits per value and one
//! scale and bias per group of `group_size` values along the last axis.
//!
//! The packed form is MLX's, so weights quantized by either library load in the other:
//! codes form a little-endian bit stream per row, stored as uint32; scales and biases keep
//! the weights' dtype. A value `w` is stored as `round((w - bias) / scale)`, clamped to
//! `[0, 2^bits - 1]`.
//!
//! ```
//! use nuvec_core::{Array, Result, dequantize, quantize, quantized_matmul};
//!
//! # fn main() -> Result<()> {
//! let w = Array::from_slice(&(0..256).map(|i| i as f32 / 64.0).collect::<Vec<_>>(), &[4, 64])?;
//! let (wq, scales, biases) = quantize(&w, 64, 4)?;
//! assert_eq!(wq.shape(), &[4, 8]); // 64 values · 4 bits = 8 words per row
//! let restored = dequantize(&wq, &scales, &biases, 64, 4)?;
//! assert_eq!(restored.shape(), w.shape());
//!
//! let x = Array::full(&[2, 64], 1.0f32)?;
//! let y = quantized_matmul(&x, &wq, &scales, &biases, true, 64, 4)?; // x · Wᵀ
//! assert_eq!(y.shape(), &[2, 4]);
//! # Ok(())
//! # }
//! ```

use crate::ops::{astype, broadcast_to, expand_dims, squeeze};
use crate::primitive::Primitive;
use crate::{Array, Category, Error, Result, promote_types, view};

/// The supported group sizes and bit widths (MLX's).
fn check_params(op: &str, group_size: usize, bits: u32) -> Result<u8> {
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

fn check_inputs(inputs: &[&Array]) -> Result<()> {
    match inputs.iter().find_map(|a| a.error()) {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Quantizes the last axis of `w` (float16, bfloat16 or float32, at least 2-D, the last
/// dimension divisible by `group_size`) in groups of `group_size` (32, 64 or 128) values with
/// `bits` (1–6 or 8) bits each. Returns the packed codes (uint32, last dimension
/// `· bits / 32`), the scales and the biases (`w`'s dtype, last dimension `/ group_size`),
/// equal to MLX's bit for bit.
pub fn quantize(w: &Array, group_size: usize, bits: u32) -> Result<(Array, Array, Array)> {
    check_inputs(&[w])?;
    let bits = check_params("quantize", group_size, bits)?;
    let mut out = Array::from_op(Primitive::Quantize { group_size, bits }, vec![w.clone()])?;
    let biases = out.pop().expect("three outputs");
    let scales = out.pop().expect("three outputs");
    Ok((out.pop().expect("three outputs"), scales, biases))
}

/// Restores the values packed by [`quantize`]: `code · scale + bias`, in the scales' dtype.
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: &Array,
    group_size: usize,
    bits: u32,
) -> Result<Array> {
    check_inputs(&[w, scales, biases])?;
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
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    transpose: bool,
    group_size: usize,
    bits: u32,
) -> Result<Array> {
    check_inputs(&[x, w, scales, biases])?;
    let bits = check_params("quantized_matmul", group_size, bits)?;
    let dtype = promote_types(promote_types(x.dtype(), scales.dtype()), biases.dtype());
    if dtype.category() != Category::Float || dtype.size() > 4 {
        return Err(Error::UnsupportedDType {
            op: "quantized_matmul",
            dtype,
        });
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
        let batch = view::broadcast_shapes(
            "quantized_matmul",
            &x.shape()[..xn - 2],
            &w.shape()[..wn - 2],
        )?;
        let full = |a: &Array| -> Result<Array> {
            let mut shape = batch.to_vec();
            shape.extend_from_slice(&a.shape()[a.ndim() - 2..]);
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
///
/// ```
/// use nuvec_core::{Array, Result, gather_qmm, quantize};
///
/// # fn main() -> Result<()> {
/// // Two "experts" of shape [4, 64]: expert e has every weight equal to e + 1.
/// let w = Array::from_slice(
///     &(0..512).map(|i| if i < 256 { 1.0f32 } else { 2.0 }).collect::<Vec<_>>(),
///     &[2, 4, 64],
/// )?;
/// let (wq, scales, biases) = quantize(&w, 64, 4)?;
/// let x = Array::full(&[3, 1, 64], 1.0f32)?; // three tokens
/// let experts = Array::from_slice(&[1u32, 0, 1], &[3])?;
/// let y = gather_qmm(&x, &wq, &scales, &biases, None, Some(&experts), true, 64, 4, false)?;
/// assert_eq!(y.shape(), &[3, 1, 4]);
/// assert_eq!(y.to_vec::<f32>()?[..5], [128.0, 128.0, 128.0, 128.0, 64.0]);
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
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
    check_inputs(&[x, w, scales, biases])?;
    let bits = check_params("gather_qmm", group_size, bits)?;
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("gather_qmm: {msg}")));
    let dtype = promote_types(promote_types(x.dtype(), scales.dtype()), biases.dtype());
    if dtype.category() != Category::Float || dtype.size() > 4 {
        return Err(Error::UnsupportedDType {
            op: "gather_qmm",
            dtype,
        });
    }
    if x.ndim() < 2 || w.ndim() < 2 || scales.shape() != biases.shape() {
        return invalid(format!(
            "x must be [..., M, K] and w [..., N, K'], got {:?} and {:?}",
            x.shape(),
            w.shape()
        ));
    }
    let (x, scales, biases) = (
        astype(x, dtype)?,
        astype(scales, dtype)?,
        astype(biases, dtype)?,
    );
    let (lhs, rhs) = gather_indices("gather_qmm", &x, w, lhs_indices, rhs_indices)?;
    let mut inputs = vec![x.clone(), w.clone(), scales, biases, lhs, rhs.clone()];
    if let Some(offsets) = sorted_offsets(&x, w, &rhs, lhs_indices.is_none() && sorted_indices)? {
        inputs.push(offsets);
    }
    let op = crate::fast::FastOp::GatherQmm {
        group_size,
        bits,
        transpose,
    };
    let kernel = transpose && crate::fast::on_gpu(dtype, x.size());
    let out = if kernel {
        Array::from_op(Primitive::Fast(op), inputs)?.swap_remove(0)
    } else {
        op.fallback(&inputs)?.swap_remove(0)
    };
    Ok(out)
}

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
        let batch = &x.shape()[..x.ndim() - 2];
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
                let all = crate::arange(0.0, n as f64, 1.0, crate::DType::U32)?;
                all.reshape(&batch.iter().map(|&d| d as isize).collect::<Vec<_>>())
            }
        }
    };
    let (l, r) = (indices(a, lhs)?, indices(b, rhs)?);
    let batch = view::broadcast_shapes(op, l.shape(), r.shape())?;
    Ok((broadcast_to(&l, &batch)?, broadcast_to(&r, &batch)?))
}

/// For rows sorted by weight batch (`a` is `[rows, 1, K]`, one row per output batch), where
/// each batch's rows start: the rows per batch (scatter-added ones), then their exclusive
/// running sum (MLX's grouped kernels walk every weight batch once).
fn sorted_offsets(a: &Array, b: &Array, rhs: &Array, sorted: bool) -> Result<Option<Array>> {
    let m = a.shape()[a.ndim() - 2];
    if !(sorted && m == 1 && a.ndim() == 3 && rhs.ndim() == 1) {
        return Ok(None);
    }
    let experts: usize = b.shape()[..b.ndim() - 2].iter().product();
    let ones = crate::ones(rhs.shape(), crate::DType::I32)?;
    let zeros = crate::zeros(&[experts], crate::DType::I32)?;
    let counts = crate::scatter_add_along_axis(&zeros, rhs, &ones, 0)?;
    Ok(Some(crate::cumsum(&counts, Some(0), false, false)?))
}

/// Matrix products of gathered batches, as MLX's `gather_mm` (mixture-of-experts layers in
/// full precision): output batch `i` is `a[lhs_indices[i]] · b[rhs_indices[i]]` over the
/// flattened batch dimensions of `a` (`[..., M, K]`) and `b` (`[..., K, N]`), with the
/// indices broadcast together and either defaulting to every batch. `sorted_indices`
/// promises sorted `rhs_indices`, which lets one kernel walk each batch of `b` once.
///
/// ```
/// use nuvec_core::{Array, Result, gather_mm};
///
/// # fn main() -> Result<()> {
/// // Two 2×2 "experts": the identity and twice the identity.
/// let b = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2])?;
/// let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 1, 2])?; // two rows
/// let experts = Array::from_slice(&[1u32, 0], &[2])?;
/// let y = gather_mm(&a, &b, None, Some(&experts), false)?;
/// assert_eq!(y.to_vec::<f32>()?, [2.0, 4.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub fn gather_mm(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
) -> Result<Array> {
    check_inputs(&[a, b])?;
    let dtype = promote_types(a.dtype(), b.dtype());
    if dtype.category() != Category::Float {
        return Err(Error::UnsupportedDType {
            op: "gather_mm",
            dtype,
        });
    }
    if a.ndim() < 2 || b.ndim() < 2 || a.shape()[a.ndim() - 1] != b.shape()[b.ndim() - 2] {
        return Err(Error::InvalidArgument(format!(
            "gather_mm: a must be [..., M, K] and b [..., K, N], got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let (a, b) = (astype(a, dtype)?, astype(b, dtype)?);
    let (lhs, rhs) = gather_indices("gather_mm", &a, &b, lhs_indices, rhs_indices)?;
    let mut inputs = vec![a.clone(), b.clone(), lhs, rhs.clone()];
    if let Some(offsets) = sorted_offsets(&a, &b, &rhs, lhs_indices.is_none() && sorted_indices)? {
        inputs.push(offsets);
    }
    Ok(crate::fast::record_or_fallback(crate::fast::FastOp::GatherMm, inputs)?.swap_remove(0))
}
