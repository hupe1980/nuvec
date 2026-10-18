//! Indexing and joining: `take`, `take_along_axis`, `put_along_axis`, `concatenate`,
//! `stack` and `split`, as in MLX.
//!
//! Indices are integer arrays. As in MLX, negative indices count from the end of the
//! axis; out-of-range indices (which MLX leaves unspecified) are clamped to the axis.

use crate::ops::{broadcast_to, expand_dims, reshape, slice};
use crate::primitive::{Primitive, ScatterReduce};
use crate::view::normalize_axis;
use crate::{Array, Error, Result, SliceArg, promote_types};

fn single(primitive: Primitive, inputs: Vec<Array>) -> Result<Array> {
    Ok(Array::from_op(primitive, inputs)?.swap_remove(0))
}

fn isize_shape(shape: &[usize]) -> Vec<isize> {
    shape.iter().map(|&d| d as isize).collect()
}

/// `a` and `indices` broadcast against each other on every axis but `axis` (which keeps
/// each one's own length), as MLX's `take_along_axis` and `put_along_axis` require.
fn broadcast_except(
    a: &Array,
    indices: &Array,
    axis: usize,
    a_too: bool,
) -> Result<(Array, Array)> {
    if indices.ndim() != a.ndim() {
        return Err(Error::InvalidArgument(format!(
            "indices of {} dimensions for an array of {}",
            indices.ndim(),
            a.ndim()
        )));
    }
    let mut common = Vec::with_capacity(a.ndim());
    for (d, (&x, &i)) in a.shape().iter().zip(indices.shape()).enumerate() {
        common.push(match (d == axis, x, i) {
            (true, _, _) => 0,
            (_, x, i) if x == i || i == 1 => x,
            (_, 1, i) if a_too => i,
            _ => {
                return Err(Error::ShapeMismatch {
                    op: "take_along_axis",
                    lhs: a.shape().to_vec(),
                    rhs: indices.shape().to_vec(),
                });
            }
        });
    }
    let with = |len: usize| {
        let mut s = common.clone();
        s[axis] = len;
        s
    };
    let a = if a_too {
        broadcast_to(a, &with(a.shape()[axis]))?
    } else {
        a.clone()
    };
    let indices = broadcast_to(indices, &with(indices.shape()[axis]))?;
    Ok((a, indices))
}

/// The elements of `a` at `indices` along `axis` (MLX's `take_along_axis`): `indices`
/// has `a`'s number of dimensions, and both broadcast on the other axes.
///
/// ```
/// use nuvec_core::{Array, Result, take_along_axis};
///
/// # fn main() -> Result<()> {
/// let a = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 2])?;
/// let i = Array::from_slice(&[1i32, 0], &[2, 1])?;
/// assert_eq!(take_along_axis(&a, &i, 1)?.to_vec::<f32>()?, [20.0, 30.0]);
/// # Ok(())
/// # }
/// ```
pub fn take_along_axis(a: &Array, indices: &Array, axis: isize) -> Result<Array> {
    let axis = normalize_axis(axis, a.ndim())?;
    let (a, indices) = broadcast_except(a, indices, axis, true)?;
    single(Primitive::GatherAxis { axis }, vec![a, indices])
}

/// The elements of `a` at `indices` along `axis` (MLX's `take`): the axis is replaced by
/// `indices`' dimensions. Without an axis, `a` is flattened first.
///
/// ```
/// use nuvec_core::{Array, Result, take};
///
/// # fn main() -> Result<()> {
/// // An embedding lookup: rows of a table.
/// let table = Array::from_slice(&[0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5], &[3, 2])?;
/// let tokens = Array::from_slice(&[2u32, 0, 2], &[3])?;
/// let rows = take(&table, &tokens, Some(0))?;
/// assert_eq!(rows.shape(), [3, 2]);
/// assert_eq!(rows.to_vec::<f32>()?, [2.0, 2.5, 0.0, 0.5, 2.0, 2.5]);
/// # Ok(())
/// # }
/// ```
pub fn take(a: &Array, indices: &Array, axis: Option<isize>) -> Result<Array> {
    let (a, axis) = match axis {
        Some(axis) => (a.clone(), normalize_axis(axis, a.ndim())?),
        None => (reshape(a, &[-1])?, 0),
    };
    let m = indices.size();
    // Flat indices along `axis`, broadcast over the other axes.
    let mut along = vec![1; a.ndim()];
    along[axis] = m as isize;
    let mut full = a.shape().to_vec();
    full[axis] = m;
    let flat = broadcast_to(&reshape(indices, &along)?, &full)?;
    let out = single(Primitive::GatherAxis { axis }, vec![a.clone(), flat])?;
    let mut shape = isize_shape(&a.shape()[..axis]);
    shape.extend(isize_shape(indices.shape()));
    shape.extend(isize_shape(&a.shape()[axis + 1..]));
    reshape(&out, &shape)
}

fn scatter(
    a: &Array,
    indices: &Array,
    values: &Array,
    axis: isize,
    reduce: ScatterReduce,
) -> Result<Array> {
    let axis = normalize_axis(axis, a.ndim())?;
    let (_, indices) = broadcast_except(a, indices, axis, false)?;
    let values = broadcast_to(&values.astype(a.dtype())?, indices.shape())?;
    single(
        Primitive::ScatterAxis { axis, reduce },
        vec![a.clone(), indices, values],
    )
}

/// A copy of `a` with `values` placed at `indices` along `axis` (MLX's `put_along_axis`).
/// `indices` broadcasts against `a` on the other axes, `values` against `indices`; with
/// repeated indices, one of the values wins.
///
/// ```
/// use nuvec_core::{Array, Result, put_along_axis};
///
/// # fn main() -> Result<()> {
/// let a = Array::from_slice(&[0i32; 4], &[4])?;
/// let i = Array::from_slice(&[3i32, -4], &[2])?;
/// let v = Array::from_slice(&[7i32, 9], &[2])?;
/// assert_eq!(put_along_axis(&a, &i, &v, 0)?.to_vec::<i32>()?, [9, 0, 0, 7]);
/// # Ok(())
/// # }
/// ```
pub fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: isize) -> Result<Array> {
    scatter(a, indices, values, axis, ScatterReduce::None)
}

/// A copy of `a` with `values` added at `indices` along `axis`; repeated indices
/// accumulate (MLX's `scatter_add_axis`, the gradient of `take_along_axis`).
pub fn scatter_add_along_axis(
    a: &Array,
    indices: &Array,
    values: &Array,
    axis: isize,
) -> Result<Array> {
    scatter(a, indices, values, axis, ScatterReduce::Sum)
}

/// `arrays` joined along `axis` (MLX's `concatenate`): equal shapes except along `axis`,
/// with dtypes promoted to a common one.
///
/// ```
/// use nuvec_core::{Array, Result, concatenate};
///
/// # fn main() -> Result<()> {
/// let a = Array::from_slice(&[1.0f32, 2.0], &[1, 2])?;
/// let b = Array::from_slice(&[3i32, 4, 5, 6], &[2, 2])?;
/// let c = concatenate(&[a, b], 0)?;
/// assert_eq!(c.shape(), [3, 2]);
/// assert_eq!(c.to_vec::<f32>()?, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub fn concatenate(arrays: &[Array], axis: isize) -> Result<Array> {
    let first = arrays
        .first()
        .ok_or_else(|| Error::InvalidArgument("concatenate: no arrays to join".into()))?;
    let axis = normalize_axis(axis, first.ndim())?;
    let dtype = arrays
        .iter()
        .skip(1)
        .fold(first.dtype(), |d, a| promote_types(d, a.dtype()));
    let arrays = arrays
        .iter()
        .map(|a| a.astype(dtype))
        .collect::<Result<Vec<_>>>()?;
    single(Primitive::Concatenate { axis }, arrays)
}

/// `arrays` (one shape) stacked along a new axis `axis` (MLX's `stack`).
pub fn stack(arrays: &[Array], axis: isize) -> Result<Array> {
    let first = arrays
        .first()
        .ok_or_else(|| Error::InvalidArgument("stack: no arrays to stack".into()))?;
    let axis = normalize_axis(axis, first.ndim() + 1)? as isize;
    if let Some(other) = arrays.iter().find(|a| a.shape() != first.shape()) {
        return Err(Error::ShapeMismatch {
            op: "stack",
            lhs: first.shape().to_vec(),
            rhs: other.shape().to_vec(),
        });
    }
    let expanded = arrays
        .iter()
        .map(|a| expand_dims(a, axis))
        .collect::<Result<Vec<_>>>()?;
    concatenate(&expanded, axis)
}

/// `a` split into `parts` equal parts along `axis` (MLX's `split` with a count); views.
pub fn split(a: &Array, parts: usize, axis: isize) -> Result<Vec<Array>> {
    let d = normalize_axis(axis, a.ndim())?;
    let n = a.shape()[d];
    if parts == 0 || !n.is_multiple_of(parts) {
        return Err(Error::InvalidArgument(format!(
            "split: an axis of length {n} cannot be split into {parts} equal parts"
        )));
    }
    let size = n / parts;
    let at: Vec<usize> = (1..parts).map(|k| k * size).collect();
    split_at(a, &at, axis)
}

/// `a` split along `axis` before each of the (non-decreasing) positions `indices`, as
/// MLX's `split` with indices: `indices.len() + 1` views.
pub fn split_at(a: &Array, indices: &[usize], axis: isize) -> Result<Vec<Array>> {
    let d = normalize_axis(axis, a.ndim())?;
    let n = a.shape()[d];
    let mut bounds = vec![0];
    bounds.extend(indices.iter().map(|&i| i.min(n)));
    bounds.push(n);
    bounds
        .windows(2)
        .map(|w| {
            let mut args = vec![SliceArg::full(); a.ndim()];
            args[d] = SliceArg::new(Some(w[0] as isize), Some(w[1].max(w[0]) as isize), 1);
            slice(a, &args)
        })
        .collect()
}
