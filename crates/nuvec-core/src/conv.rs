//! Convolutions, as MLX's `conv1d`.

use crate::ops::{add, matmul, multiply, reshape, slice, transpose};
use crate::{
    Array, Category, DType, Error, Result, SliceArg, concatenate, promote_types, stack, zeros,
};

/// The 1-D convolution (cross-correlation) of `input` (`[N, L, C_in]`) with `weight`
/// (`[C_out, K, C_in / groups]`), as MLX's `conv1d`: output `[N, L_out, C_out]` with
/// `L_out = (L + 2·padding − dilation·(K − 1) − 1) / stride + 1`. Channels split into
/// `groups` independent groups (`groups = C_in = C_out` is a depthwise convolution).
/// Products accumulate in float32 and round once, as MLX's kernels.
///
/// ```
/// use nuvec_core::{Array, Result, conv1d};
///
/// # fn main() -> Result<()> {
/// // A moving sum of three, one channel.
/// let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 5, 1])?;
/// let w = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1])?;
/// assert_eq!(conv1d(&x, &w, 1, 0, 1, 1)?.to_vec::<f32>()?, [6.0, 9.0, 12.0]);
/// assert_eq!(conv1d(&x, &w, 2, 1, 1, 1)?.to_vec::<f32>()?, [3.0, 9.0, 9.0]);
/// # Ok(())
/// # }
/// ```
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Array> {
    let invalid = |msg: String| Err(Error::InvalidArgument(format!("conv1d: {msg}")));
    if input.ndim() != 3 || weight.ndim() != 3 {
        return invalid(format!(
            "input must be [N, L, C_in] and weight [C_out, K, C_in / groups], got {:?} and {:?}",
            input.shape(),
            weight.shape()
        ));
    }
    let (l, c_in) = (input.shape()[1], input.shape()[2]);
    let (c_out, k, c_group) = (weight.shape()[0], weight.shape()[1], weight.shape()[2]);
    if stride == 0 || dilation == 0 || groups == 0 {
        return invalid("stride, dilation and groups must be positive".into());
    }
    if c_in % groups != 0 || c_out % groups != 0 || c_group * groups != c_in {
        return invalid(format!(
            "{c_in} input and {c_out} output channels in {groups} groups need weights \
             [C_out, K, {}], got {:?}",
            c_in / groups.max(1),
            weight.shape()
        ));
    }
    let span = dilation * (k.max(1) - 1) + 1;
    if k == 0 || l + 2 * padding < span {
        return invalid(format!(
            "a kernel spanning {span} positions does not fit {l} positions padded by {padding}"
        ));
    }
    let dtype = promote_types(input.dtype(), weight.dtype());
    if dtype.category() != Category::Float {
        return Err(Error::UnsupportedDType {
            op: "conv1d",
            dtype,
        });
    }
    if groups == c_in && c_out == c_in {
        // Depthwise: one GPU kernel, with `general` as its fallback.
        let op = crate::fast::FastOp::DepthwiseConv1d {
            stride,
            padding,
            dilation,
        };
        let inputs = vec![input.astype(dtype)?, weight.astype(dtype)?];
        return Ok(crate::fast::record_or_fallback(op, inputs)?.swap_remove(0));
    }
    general(input, weight, (stride, padding, dilation, groups))
}

/// [`conv1d`] composed of ordinary operations (the fallback of the depthwise kernel, and
/// every grouped convolution): taps of the padded input in float32, weighted and summed
/// in tap order, rounded once.
pub(crate) fn general(
    input: &Array,
    weight: &Array,
    (stride, padding, dilation, groups): (usize, usize, usize, usize),
) -> Result<Array> {
    let (n, l, c_in) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let (c_out, k, c_group) = (weight.shape()[0], weight.shape()[1], weight.shape()[2]);
    let span = dilation * (k - 1) + 1;
    let dtype = promote_types(input.dtype(), weight.dtype());
    let acc = if dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let l_out = (l + 2 * padding - span) / stride + 1;
    let mut x = input.astype(acc)?;
    if padding > 0 {
        let pad = zeros(&[n, padding, c_in], acc)?;
        x = concatenate(&[pad.clone(), x, pad], 1)?;
    }
    // Tap `j` of every output position: positions `j·dilation + i·stride`.
    let tap = |j: usize| -> Result<Array> {
        let start = (j * dilation) as isize;
        let stop = start + ((l_out - 1) * stride + 1) as isize;
        slice(
            &x,
            &[
                SliceArg::full(),
                SliceArg::new(Some(start), Some(stop), stride as isize),
                SliceArg::full(),
            ],
        )
    };
    let w = weight.astype(acc)?;
    let out = if groups == c_in && c_out == c_in {
        // Depthwise: each channel's taps weighted by its own kernel.
        let mut sum: Option<Array> = None;
        for j in 0..k {
            let wj = reshape(
                &slice(
                    &w,
                    &[
                        SliceArg::full(),
                        SliceArg::new(Some(j as isize), Some(j as isize + 1), 1),
                        SliceArg::full(),
                    ],
                )?,
                &[c_in as isize],
            )?;
            let term = multiply(&tap(j)?, &wj)?;
            sum = Some(match sum {
                Some(s) => add(&s, &term)?,
                None => term,
            });
        }
        sum.expect("at least one tap")
    } else {
        // `[N·L_out, K·C_in]` columns, one product per group.
        let cols = stack(&(0..k).map(tap).collect::<Result<Vec<_>>>()?, 2)?;
        let per_out = c_out / groups;
        let mut outs = Vec::with_capacity(groups);
        for g in 0..groups {
            let cg = slice(
                &cols,
                &[
                    SliceArg::full(),
                    SliceArg::full(),
                    SliceArg::full(),
                    SliceArg::new(
                        Some((g * c_group) as isize),
                        Some(((g + 1) * c_group) as isize),
                        1,
                    ),
                ],
            )?;
            let cg = reshape(&cg, &[(n * l_out) as isize, (k * c_group) as isize])?;
            let wg = slice(
                &w,
                &[
                    SliceArg::new(
                        Some((g * per_out) as isize),
                        Some(((g + 1) * per_out) as isize),
                        1,
                    ),
                    SliceArg::full(),
                    SliceArg::full(),
                ],
            )?;
            let wg = reshape(&wg, &[per_out as isize, (k * c_group) as isize])?;
            outs.push(matmul(&cg, &transpose(&wg, &[1, 0])?)?);
        }
        let y = match groups {
            1 => outs.pop().expect("one group"),
            _ => concatenate(&outs, 1)?,
        };
        reshape(&y, &[n as isize, l_out as isize, c_out as isize])?
    };
    out.astype(dtype)
}
