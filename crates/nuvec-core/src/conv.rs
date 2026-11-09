//! Convolutions, as MLX's `conv_general` (and `conv1d`, `conv2d`, `conv3d`, the transposed
//! convolutions and `convolve` on top): channels last, input `[N, spatial…, C_in]`, weight
//! `[C_out, kernel…, C_in / groups]`. On the GPU, MLX's kernels where MLX routes to them
//! (Winograd, depthwise, implicit GEMM), else a depthwise kernel or an unfold (im2col) and a
//! GEMM; elsewhere a composition of strided taps and matrix products, whose derivatives give
//! the gradients. Products accumulate in float32
//! (float64 for float64) and round once, as MLX's kernels.

use crate::indexing::{concatenate, stack};
use crate::ops::imp::{add, matmul, multiply, reshape, slice, transpose, zeros};
use crate::{Array, Category, DType, Error, Result, SliceArg, promote_types};

/// The parameters of a general convolution (per spatial axis).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Conv {
    pub stride: Vec<usize>,
    pub pad_lo: Vec<isize>,
    pub pad_hi: Vec<isize>,
    pub kernel_dilation: Vec<usize>,
    pub input_dilation: Vec<usize>,
    pub groups: usize,
    pub flip: bool,
}

impl Conv {
    /// Plain convolutions: stride, symmetric padding and kernel dilation per axis.
    pub(crate) fn plain(
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
    ) -> Self {
        Self {
            stride: stride.to_vec(),
            pad_lo: padding.iter().map(|&p| p as isize).collect(),
            pad_hi: padding.iter().map(|&p| p as isize).collect(),
            kernel_dilation: dilation.to_vec(),
            input_dilation: vec![1; stride.len()],
            groups,
            flip: false,
        }
    }
}

fn invalid(name: &str, msg: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("{name}: {msg}"))
}

/// The 1-D convolution of `input` (`[N, L, C_in]`) with `weight` (`[C_out, K, C_in / groups]`),
/// as MLX's `conv1d`.
pub(crate) fn conv1d(
    input: &Array,
    weight: &Array,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Array> {
    conv_general(
        "conv1d",
        input,
        weight,
        &Conv::plain(&[stride], &[padding], &[dilation], groups),
    )
}

/// Checks `input`, `weight` and `conv` against each other (MLX's messages' substance).
fn check(name: &'static str, input: &Array, weight: &Array, conv: &Conv) -> Result<()> {
    let d = conv.stride.len();
    if input.ndim() != d + 2 || weight.ndim() != d + 2 {
        return Err(invalid(
            name,
            format!(
                "input must be [N, {d} spatial, C_in] and weight [C_out, {d} kernel, C_in / groups], \
                 got {:?} and {:?}",
                input.dims(),
                weight.dims()
            ),
        ));
    }
    for v in [&conv.pad_lo, &conv.pad_hi] {
        if v.len() != d {
            return Err(invalid(name, "one padding per spatial axis"));
        }
    }
    if conv.kernel_dilation.len() != d || conv.input_dilation.len() != d {
        return Err(invalid(name, "one dilation per spatial axis"));
    }
    if conv.stride.contains(&0)
        || conv.kernel_dilation.contains(&0)
        || conv.input_dilation.contains(&0)
        || conv.groups == 0
    {
        return Err(invalid(
            name,
            "stride, dilation and groups must be positive",
        ));
    }
    let c_in = input.dims()[d + 1];
    let (c_out, c_group) = (weight.dims()[0], weight.dims()[d + 1]);
    let g = conv.groups;
    if !c_in.is_multiple_of(g) || !c_out.is_multiple_of(g) || c_group * g != c_in {
        return Err(invalid(
            name,
            format!(
                "{c_in} input and {c_out} output channels in {g} groups need weights \
                 [C_out, kernel…, {}], got {:?}",
                c_in / g.max(1),
                weight.dims()
            ),
        ));
    }
    for i in 0..d {
        let k = weight.dims()[1 + i];
        let n = input.dims()[1 + i];
        let span = conv.kernel_dilation[i]
            .checked_mul(k.saturating_sub(1))
            .and_then(|s| isize::try_from(s).ok()?.checked_add(1));
        let size = n
            .saturating_sub(1)
            .checked_mul(conv.input_dilation[i])
            .and_then(|s| isize::try_from(s + usize::from(n > 0)).ok())
            .and_then(|s| s.checked_add(conv.pad_lo[i])?.checked_add(conv.pad_hi[i]));
        let (Some(span), Some(size)) = (span, size) else {
            return Err(invalid(name, format!("the sizes overflow on axis {i}")));
        };
        if k == 0 || size < span {
            return Err(invalid(
                name,
                format!(
                    "a kernel spanning {span} positions does not fit {size} padded positions on axis {i}"
                ),
            ));
        }
    }
    let dtype = promote_types(input.dtype(), weight.dtype());
    if dtype.category() != Category::Float {
        return Err(Error::unsupported_dtype(name, dtype));
    }
    Ok(())
}

/// Whether `check` accepts the arguments.
pub(crate) fn valid(input: &Array, weight: &Array, conv: &Conv) -> bool {
    check("conv", input, weight, conv).is_ok()
}

fn dilated(n: usize, d: usize) -> usize {
    if n == 0 { 0 } else { (n - 1) * d + 1 }
}

/// The output's spatial sizes.
pub(crate) fn out_dims(input: &[usize], weight: &[usize], conv: &Conv) -> Vec<usize> {
    let d = conv.stride.len();
    (0..d)
        .map(|i| {
            let span = conv.kernel_dilation[i] * (weight[1 + i] - 1) + 1;
            let size = dilated(input[1 + i], conv.input_dilation[i]) as isize
                + conv.pad_lo[i]
                + conv.pad_hi[i];
            (size as usize - span) / conv.stride[i] + 1
        })
        .collect()
}

/// MLX's `conv_general`: `input` `[N, spatial…, C_in]` with `weight` `[C_out, kernel…, C_in /
/// groups]`. On the GPU MLX's kernels, a depthwise kernel or an unfold and a GEMM; elsewhere
/// (and for the derivatives) [`compose`].
pub(crate) fn conv_general(
    name: &'static str,
    input: &Array,
    weight: &Array,
    conv: &Conv,
) -> Result<Array> {
    // Negative padding slices the input (before its dilation, as MLX), then pads nothing.
    let d = conv.stride.len();
    let mut input = input.clone();
    let mut conv = conv.clone();
    if input.ndim() == d + 2
        && conv.pad_lo.len() == d
        && conv.pad_hi.len() == d
        && conv.pad_lo.iter().chain(&conv.pad_hi).any(|&p| p < 0)
    {
        let mut args = vec![SliceArg::full(); d + 2];
        for i in 0..d {
            let (lo, hi) = ((-conv.pad_lo[i]).max(0), (-conv.pad_hi[i]).max(0));
            let n = input.dims()[1 + i] as isize;
            args[1 + i] = SliceArg::new(Some(lo.min(n)), Some((n - hi).max(lo.min(n))), 1);
            conv.pad_lo[i] = conv.pad_lo[i].max(0);
            conv.pad_hi[i] = conv.pad_hi[i].max(0);
        }
        input = slice(&input, &args)?;
    }
    let conv = &conv;
    check(name, &input, weight, conv)?;
    let dtype = promote_types(input.dtype(), weight.dtype());
    let (pad_lo, pad_hi) = (conv.pad_lo.clone(), conv.pad_hi.clone());
    let op = crate::fast::imp::FastOp::Conv {
        stride: conv.stride.clone(),
        pad_lo,
        pad_hi,
        kernel_dilation: conv.kernel_dilation.clone(),
        input_dilation: conv.input_dilation.clone(),
        groups: conv.groups,
        flip: conv.flip,
    };
    let inputs = vec![input.try_astype(dtype)?, weight.try_astype(dtype)?];
    Ok(crate::fast::imp::record_or_fallback(op, inputs)?.swap_remove(0))
}

/// [`conv_general`] composed of strided taps and matrix products (checked arguments), with
/// fast paths for patch embeddings, 1×1 and depthwise convolutions. Products accumulate in
/// float32 (float64 for float64) and round once.
pub(crate) fn compose(input: &Array, weight: &Array, conv: &Conv) -> Result<Array> {
    let d = conv.stride.len();
    let dtype = promote_types(input.dtype(), weight.dtype());
    let acc = if dtype == DType::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let mut x = input.try_astype(acc)?;
    let mut w = weight.try_astype(acc)?;
    if conv.flip {
        let mut args = vec![SliceArg::full(); d + 2];
        for a in &mut args[1..=d] {
            *a = SliceArg::new(None, None, -1);
        }
        w = slice(&w, &args)?;
    }
    // Input dilation, then padding (negative padding crops).
    for i in 0..d {
        if conv.input_dilation[i] > 1 {
            x = dilate(&x, 1 + i, conv.input_dilation[i])?;
        }
        x = pad_axis(&x, 1 + i, conv.pad_lo[i], conv.pad_hi[i])?;
    }
    let n = x.dims()[0];
    let c_in = x.dims()[d + 1];
    let (c_out, c_group) = (w.dims()[0], w.dims()[d + 1]);
    let kernel: Vec<usize> = w.dims()[1..=d].to_vec();
    let out: Vec<usize> = (0..d)
        .map(|i| {
            let span = conv.kernel_dilation[i] * (kernel[i] - 1) + 1;
            (x.dims()[1 + i] - span) / conv.stride[i] + 1
        })
        .collect();
    let rows: usize = n * out.iter().product::<usize>();
    let taps: usize = kernel.iter().product();
    let mut out_shape: Vec<isize> = vec![n as isize];
    out_shape.extend(out.iter().map(|&o| o as isize));
    out_shape.push(c_out as isize);

    // Patch embedding: stride = kernel, no dilation, the input tiled exactly.
    let patches = conv.groups == 1
        && (0..d).all(|i| {
            conv.stride[i] == kernel[i]
                && conv.kernel_dilation[i] == 1
                && x.dims()[1 + i] == out[i] * kernel[i]
        });
    if patches {
        // [N, o0, k0, o1, k1, …, C] → [N, o0, o1, …, k0, k1, …, C] → [rows, K·C].
        let mut split: Vec<isize> = vec![n as isize];
        for i in 0..d {
            split.extend([out[i] as isize, kernel[i] as isize]);
        }
        split.push(c_in as isize);
        let xs = reshape(&x, &split)?;
        let mut order: Vec<usize> = vec![0];
        order.extend((0..d).map(|i| 1 + 2 * i));
        order.extend((0..d).map(|i| 2 + 2 * i));
        order.push(2 * d + 1);
        let cols = reshape(
            &transpose(&xs, &order)?,
            &[rows as isize, (taps * c_in) as isize],
        )?;
        let wm = reshape(&w, &[c_out as isize, (taps * c_in) as isize])?;
        let y = matmul(&cols, &transpose(&wm, &[1, 0])?)?;
        return reshape(&y, &out_shape)?.try_astype(dtype);
    }

    // Tap `t` (a kernel offset): positions `offset·dilation + j·stride` per axis.
    let offsets: Vec<Vec<usize>> = (0..taps)
        .map(|t| {
            let mut rem = t;
            let mut v = vec![0; d];
            for i in (0..d).rev() {
                v[i] = rem % kernel[i];
                rem /= kernel[i];
            }
            v
        })
        .collect();
    let tap = |o: &[usize]| -> Result<Array> {
        let mut args = vec![SliceArg::full(); d + 2];
        for i in 0..d {
            let start = (o[i] * conv.kernel_dilation[i]) as isize;
            let stop = start + ((out[i] - 1) * conv.stride[i] + 1) as isize;
            args[1 + i] = SliceArg::new(Some(start), Some(stop), conv.stride[i] as isize);
        }
        slice(&x, &args)
    };
    let weight_tap = |t: &[usize]| -> Result<Array> {
        let mut args = vec![SliceArg::full(); d + 2];
        for i in 0..d {
            args[1 + i] = SliceArg::new(Some(t[i] as isize), Some(t[i] as isize + 1), 1);
        }
        reshape(&slice(&w, &args)?, &[c_out as isize, c_group as isize])
    };

    let y = if conv.groups == c_in && c_group == 1 {
        // Depthwise (with a depth multiplier m = C_out / C_in): per channel, taps weighted
        // by its kernels and summed in tap order.
        let m = c_out / c_in;
        let mut sum: Option<Array> = None;
        for o in &offsets {
            // [C_out] weights of this tap, as [C_in, m] → broadcast over the tap's channels.
            let wt = reshape(&weight_tap(o)?, &[c_in as isize, m as isize])?;
            let xt = crate::ops::imp::expand_dims(&tap(o)?, (d + 2) as isize)?;
            let term = multiply(&xt, &wt)?;
            sum = Some(match sum {
                Some(s) => add(&s, &term)?,
                None => term,
            });
        }
        reshape(&sum.expect("at least one tap"), &out_shape)?
    } else if taps == 1 && conv.groups == 1 {
        // 1×1 (or a single tap): one matrix product.
        let t = tap(&offsets[0])?;
        let cols = reshape(&t, &[rows as isize, c_in as isize])?;
        let y = matmul(&cols, &transpose(&weight_tap(&offsets[0])?, &[1, 0])?)?;
        reshape(&y, &out_shape)?
    } else {
        // [rows, taps · C_in] columns, one product per group.
        let cols = stack(
            &offsets.iter().map(|o| tap(o)).collect::<Result<Vec<_>>>()?,
            (d + 1) as isize,
        )?;
        let per_out = c_out / conv.groups;
        let mut outs = Vec::with_capacity(conv.groups);
        for g in 0..conv.groups {
            let mut cargs = vec![SliceArg::full(); d + 3];
            cargs[d + 2] = SliceArg::new(
                Some((g * c_group) as isize),
                Some(((g + 1) * c_group) as isize),
                1,
            );
            let cg = reshape(
                &slice(&cols, &cargs)?,
                &[rows as isize, (taps * c_group) as isize],
            )?;
            let mut wargs = vec![SliceArg::full(); d + 2];
            wargs[0] = SliceArg::new(
                Some((g * per_out) as isize),
                Some(((g + 1) * per_out) as isize),
                1,
            );
            let wg = reshape(
                &slice(&w, &wargs)?,
                &[per_out as isize, (taps * c_group) as isize],
            )?;
            outs.push(matmul(&cg, &transpose(&wg, &[1, 0])?)?);
        }
        let y = match conv.groups {
            1 => outs.pop().expect("one group"),
            _ => concatenate(&outs, 1)?,
        };
        reshape(&y, &out_shape)?
    };
    y.try_astype(dtype)
}

/// `x` with `factor − 1` zeros between consecutive elements of `axis`.
fn dilate(x: &Array, axis: usize, factor: usize) -> Result<Array> {
    let len = x.dims()[axis];
    if len <= 1 {
        return Ok(x.clone());
    }
    // [.., L, ..] → [.., L, factor, ..] with zeros after each element, then cut the tail.
    let expanded = crate::ops::imp::expand_dims(x, (axis + 1) as isize)?;
    let mut zshape = expanded.dims().to_vec();
    zshape[axis + 1] = factor - 1;
    let z = zeros(&zshape, x.dtype())?;
    let both = concatenate(&[expanded, z], (axis + 1) as isize)?;
    let mut merged: Vec<isize> = both.dims().iter().map(|&v| v as isize).collect();
    merged[axis] = (len * factor) as isize;
    merged.remove(axis + 1);
    let flat = reshape(&both, &merged)?;
    let mut args = vec![SliceArg::full(); x.ndim()];
    args[axis] = SliceArg::new(Some(0), Some(((len - 1) * factor + 1) as isize), 1);
    slice(&flat, &args)
}

/// `x` padded with zeros (or cropped, for negative amounts) on `axis`.
fn pad_axis(x: &Array, axis: usize, lo: isize, hi: isize) -> Result<Array> {
    let mut x = x.clone();
    let len = x.dims()[axis] as isize;
    let (crop_lo, crop_hi) = ((-lo).max(0), (-hi).max(0));
    if crop_lo > 0 || crop_hi > 0 {
        let mut args = vec![SliceArg::full(); x.ndim()];
        args[axis] = SliceArg::new(Some(crop_lo), Some(len - crop_hi), 1);
        x = slice(&x, &args)?;
    }
    let side = |k: isize| -> Result<Array> {
        let mut s = x.dims().to_vec();
        s[axis] = k as usize;
        zeros(&s, x.dtype())
    };
    match (lo.max(0), hi.max(0)) {
        (0, 0) => Ok(x),
        (a, 0) => concatenate(&[side(a)?, x], axis as isize),
        (0, b) => concatenate(&[x.clone(), side(b)?], axis as isize),
        (a, b) => concatenate(&[side(a)?, x.clone(), side(b)?], axis as isize),
    }
}

/// MLX's `conv_transpose_general`: a flipped convolution of the stride-dilated input.
/// Per spatial axis: `[stride, padding, dilation, output_padding]`.
pub(crate) fn conv_transpose(
    name: &'static str,
    input: &Array,
    weight: &Array,
    [stride, padding, dilation, output_padding]: [&[usize]; 4],
    groups: usize,
) -> Result<Array> {
    let d = stride.len();
    if input.ndim() != d + 2 || weight.ndim() != d + 2 {
        return Err(invalid(
            name,
            format!("input must have {} dimensions and weight {}", d + 2, d + 2),
        ));
    }
    let mut lo = Vec::with_capacity(d);
    let mut hi = Vec::with_capacity(d);
    for i in 0..d {
        let wt = 1 + dilation[i] * (weight.dims()[1 + i].max(1) - 1);
        let side = wt as isize - padding[i] as isize - 1;
        lo.push(side);
        hi.push(side + output_padding[i] as isize);
    }
    let conv = Conv {
        stride: vec![1; d],
        pad_lo: lo,
        pad_hi: hi,
        kernel_dilation: dilation.to_vec(),
        input_dilation: stride.to_vec(),
        groups,
        flip: true,
    };
    conv_general(name, input, weight, &conv)
}
