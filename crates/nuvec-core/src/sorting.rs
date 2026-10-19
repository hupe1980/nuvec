//! Sorting and cumulative scans: `sort`, `argsort`, `partition`, `argpartition`, `topk`,
//! `cumsum`, `cumprod`, `cummax`, `cummin` and `logcumsumexp`, as in MLX.
//!
//! Sorting is ascending and stable, with NaN last. `partition` is a full sort, which is a
//! valid partition (MLX's GPU does the same; its CPU leaves the order unspecified).

use crate::ops::{reshape, slice};
use crate::primitive::{Primitive, ScanOp};
use crate::view::normalize_axis;
use crate::{Array, DType, Error, Result, SliceArg, at_least_float};

fn single(primitive: Primitive, a: Array) -> Result<Array> {
    Ok(Array::from_op(primitive, vec![a])?.swap_remove(0))
}

/// `a` and the axis to work on: `axis`, or 0 of the flattened array.
fn along(a: &Array, axis: Option<isize>) -> Result<(Array, usize)> {
    match axis {
        Some(axis) => Ok((a.clone(), normalize_axis(axis, a.ndim())?)),
        None => Ok((reshape(a, &[-1])?, 0)),
    }
}

/// `a` sorted along `axis` (flattened without one): ascending, stable, NaN last.
///
/// ```
/// use nuvec_core::{Array, Result, argsort, sort};
///
/// # fn main() -> Result<()> {
/// let a = Array::from_slice(&[3.0f32, f32::NAN, 1.0, 2.0], &[4])?;
/// assert_eq!(sort(&a, Some(0))?.to_vec::<f32>()?[..3], [1.0, 2.0, 3.0]);
/// assert_eq!(argsort(&a, Some(0))?.to_vec::<u32>()?, [2, 3, 0, 1]);
/// # Ok(())
/// # }
/// ```
pub fn sort(a: &Array, axis: Option<isize>) -> Result<Array> {
    let (a, axis) = along(a, axis)?;
    single(Primitive::Sort { axis }, a)
}

/// The indices (uint32) that sort `a` along `axis`, as [`sort`].
pub fn argsort(a: &Array, axis: Option<isize>) -> Result<Array> {
    let (a, axis) = along(a, axis)?;
    single(Primitive::ArgSort { axis }, a)
}

/// `kth` (negative counts from the end) validated against the axis length `n`.
fn kth_index(op: &str, kth: isize, n: usize) -> Result<isize> {
    let k = if kth < 0 { kth + n as isize } else { kth };
    if (0..n as isize).contains(&k) {
        Ok(k)
    } else {
        Err(Error::InvalidArgument(format!(
            "{op}: kth {kth} is out of range for an axis of length {n}"
        )))
    }
}

/// `a` partitioned along `axis` (flattened without one): the element at position `kth`
/// is the one a sort would put there, smaller ones come before it and larger ones after
/// (MLX's `partition`).
pub fn partition(a: &Array, kth: isize, axis: Option<isize>) -> Result<Array> {
    let (a, axis) = along(a, axis)?;
    kth_index("partition", kth, a.shape()[axis])?;
    single(Primitive::Sort { axis }, a)
}

/// The indices (uint32) that partition `a`, as [`partition`].
pub fn argpartition(a: &Array, kth: isize, axis: Option<isize>) -> Result<Array> {
    let (a, axis) = along(a, axis)?;
    kth_index("argpartition", kth, a.shape()[axis])?;
    single(Primitive::ArgSort { axis }, a)
}

/// The `k` largest elements of `a` along `axis` (flattened without one), in ascending
/// order (MLX's `topk`; MLX leaves the order unspecified).
///
/// ```
/// use nuvec_core::{Array, Result, topk};
///
/// # fn main() -> Result<()> {
/// let a = Array::from_slice(&[5i32, 1, 9, 3, 7], &[5])?;
/// assert_eq!(topk(&a, 2, Some(0))?.to_vec::<i32>()?, [7, 9]);
/// # Ok(())
/// # }
/// ```
pub fn topk(a: &Array, k: usize, axis: Option<isize>) -> Result<Array> {
    let (a, axis) = along(a, axis)?;
    let n = a.shape()[axis];
    if k > n {
        return Err(Error::InvalidArgument(format!(
            "topk: k = {k} exceeds the axis length {n}"
        )));
    }
    if k == n {
        return Ok(a);
    }
    let sorted = single(Primitive::Sort { axis }, a.clone())?;
    let mut args = vec![SliceArg::full(); a.ndim()];
    args[axis] = SliceArg::new(Some((n - k) as isize), None, 1);
    slice(&sorted, &args)
}

fn scan(
    op: ScanOp,
    a: &Array,
    axis: Option<isize>,
    reverse: bool,
    inclusive: bool,
) -> Result<Array> {
    let (a, axis) = along(a, axis)?;
    let a = match op {
        // As MLX: booleans sum as int32; log-sum-exp needs floating point.
        ScanOp::Sum if a.dtype() == DType::Bool => a.astype(DType::I32)?,
        ScanOp::LogAddExp => a.astype(at_least_float(a.dtype()))?,
        _ => a,
    };
    // 8- and 16-bit integers scan in 32 bits and wrap back (the same modular results):
    // the reverse exclusive scan of 8-bit integers was wrong on a virtualized Apple GPU
    // (GitHub's macOS runners), while 32-bit scans are right everywhere.
    let narrow = a.dtype();
    let wide = match narrow {
        DType::U8 | DType::U16 => Some(DType::U32),
        DType::I8 | DType::I16 => Some(DType::I32),
        _ => None,
    };
    let a = match wide {
        Some(w) => a.astype(w)?,
        None => a,
    };
    let out = single(
        Primitive::Scan {
            op,
            axis,
            reverse,
            inclusive,
        },
        a,
    )?;
    match wide {
        Some(_) => out.astype(narrow),
        None => Ok(out),
    }
}

/// The cumulative sum along `axis` (flattened without one): `reverse` scans from the
/// end; an exclusive scan (`inclusive = false`) leaves each element out of its own
/// output. Booleans sum as int32.
///
/// ```
/// use nuvec_core::{Array, Result, cumsum};
///
/// # fn main() -> Result<()> {
/// let a = Array::from_slice(&[1i32, 2, 3, 4], &[4])?;
/// assert_eq!(cumsum(&a, None, false, true)?.to_vec::<i32>()?, [1, 3, 6, 10]);
/// assert_eq!(cumsum(&a, None, true, false)?.to_vec::<i32>()?, [9, 7, 4, 0]);
/// # Ok(())
/// # }
/// ```
pub fn cumsum(a: &Array, axis: Option<isize>, reverse: bool, inclusive: bool) -> Result<Array> {
    scan(ScanOp::Sum, a, axis, reverse, inclusive)
}

/// The cumulative product, as [`cumsum`].
pub fn cumprod(a: &Array, axis: Option<isize>, reverse: bool, inclusive: bool) -> Result<Array> {
    scan(ScanOp::Prod, a, axis, reverse, inclusive)
}

/// The running maximum (NaN propagates), as [`cumsum`].
pub fn cummax(a: &Array, axis: Option<isize>, reverse: bool, inclusive: bool) -> Result<Array> {
    scan(ScanOp::Max, a, axis, reverse, inclusive)
}

/// The running minimum (NaN propagates), as [`cumsum`].
pub fn cummin(a: &Array, axis: Option<isize>, reverse: bool, inclusive: bool) -> Result<Array> {
    scan(ScanOp::Min, a, axis, reverse, inclusive)
}

/// The running `log(Σ exp)`, as [`cumsum`] (computed in floating point).
pub fn logcumsumexp(
    a: &Array,
    axis: Option<isize>,
    reverse: bool,
    inclusive: bool,
) -> Result<Array> {
    scan(ScanOp::LogAddExp, a, axis, reverse, inclusive)
}
