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
