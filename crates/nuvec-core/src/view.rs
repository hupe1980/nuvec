//! View geometry: shapes, strides and offsets of arrays that share another array's data.
//!
//! Pure functions of `(shape, strides)`, used both when a view is recorded (to know its
//! layout without evaluating) and when it is evaluated.

use smallvec::SmallVec;

use crate::shape::{self, Shape, Strides};
use crate::{Error, Result};

/// One slice specification per dimension: `start:stop:step` with Python semantics.
///
/// `None` bounds mean "from the beginning/end" in the step's direction. Negative indices
/// count from the end, and out-of-range bounds are clamped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceArg {
    /// First index, or `None` for the start in the step's direction.
    pub start: Option<isize>,
    /// One past the last index, or `None` for the end in the step's direction.
    pub stop: Option<isize>,
    /// Distance between selected indices; must not be zero.
    pub step: isize,
}

impl SliceArg {
    /// The whole dimension (`::`).
    pub const fn full() -> Self {
        Self {
            start: None,
            stop: None,
            step: 1,
        }
    }

    /// `start:stop:step`.
    pub const fn new(start: Option<isize>, stop: Option<isize>, step: isize) -> Self {
        Self { start, stop, step }
    }
}

impl From<std::ops::Range<isize>> for SliceArg {
    fn from(r: std::ops::Range<isize>) -> Self {
        Self::new(Some(r.start), Some(r.end), 1)
    }
}

impl From<std::ops::RangeFrom<isize>> for SliceArg {
    fn from(r: std::ops::RangeFrom<isize>) -> Self {
        Self::new(Some(r.start), None, 1)
    }
}

impl From<std::ops::RangeTo<isize>> for SliceArg {
    fn from(r: std::ops::RangeTo<isize>) -> Self {
        Self::new(None, Some(r.end), 1)
    }
}

impl From<std::ops::RangeFull> for SliceArg {
    fn from(_: std::ops::RangeFull) -> Self {
        Self::full()
    }
}

/// A single index `i`, keeping the dimension with size 1 (`i:i+1`, or the last element for
/// `-1`).
impl From<isize> for SliceArg {
    fn from(i: isize) -> Self {
        // `i + 1` past `isize::MAX` is the end, as `-1 + 1` is.
        Self::new(Some(i), if i == -1 { None } else { i.checked_add(1) }, 1)
    }
}

/// A normalised slice of one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DimSlice {
    pub(crate) start: usize,
    pub(crate) step: isize,
    pub(crate) len: usize,
}

/// Normalises one slice per dimension like Python's `slice.indices`.
pub(crate) fn normalize_slice(shape: &[usize], args: &[SliceArg]) -> Result<Vec<DimSlice>> {
    if args.len() != shape.len() {
        return Err(Error::InvalidArgument(format!(
            "slice needs one specification per dimension: got {} for shape {shape:?}",
            args.len()
        )));
    }
    shape
        .iter()
        .zip(args)
        .map(|(&n, arg)| {
            if arg.step == 0 {
                return Err(Error::InvalidArgument("slice step must not be zero".into()));
            }
            let n = n as isize;
            let (lower, upper) = if arg.step > 0 { (0, n) } else { (-1, n - 1) };
            let clamp = |i: isize| {
                let i = if i < 0 { i + n } else { i };
                i.clamp(lower, upper)
            };
            let start = arg
                .start
                .map_or(if arg.step > 0 { lower } else { upper }, clamp);
            let stop = arg
                .stop
                .map_or(if arg.step > 0 { upper } else { lower }, clamp);
            let span = if arg.step > 0 {
                stop - start
            } else {
                start - stop
            };
            let len = if span > 0 {
                (span + arg.step.abs() - 1) / arg.step.abs()
            } else {
                0
            };
            Ok(if len == 0 {
                DimSlice {
                    start: 0,
                    step: 1,
                    len: 0,
                }
            } else {
                DimSlice {
                    start: start as usize,
                    step: arg.step,
                    len: len as usize,
                }
            })
        })
        .collect()
}

/// Strides and offset delta of a slice of a layout with `strides`.
pub(crate) fn slice_geometry(strides: &[isize], slices: &[DimSlice]) -> (Strides, isize) {
    // The step of a dimension of at most one index is never applied, and may be any
    // value (a validated slice of more indices stays within the buffer).
    let out = strides
        .iter()
        .zip(slices)
        .map(|(&s, d)| if d.len <= 1 { s } else { s * d.step })
        .collect();
    let delta = strides
        .iter()
        .zip(slices)
        .map(|(&s, d)| s * d.start as isize)
        .sum();
    (out, delta)
}

/// The broadcast shape of `a` and `b` (NumPy rules), or a shape mismatch for `op`.
pub(crate) fn broadcast_shapes(op: &'static str, a: &[usize], b: &[usize]) -> Result<Shape> {
    let ndim = a.len().max(b.len());
    let dim = |s: &[usize], i: usize| {
        if i + s.len() < ndim {
            1
        } else {
            s[i + s.len() - ndim]
        }
    };
    (0..ndim)
        .map(|i| match (dim(a, i), dim(b, i)) {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => Err(Error::shape_mismatch(op, a, b)),
        })
        .collect()
}

/// Strides that present a layout of `shape`/`strides` as the larger shape `out`, or an
/// error if `shape` cannot broadcast to `out`.
pub(crate) fn broadcast_strides(
    shape: &[usize],
    strides: &[isize],
    out: &[usize],
) -> Result<Strides> {
    let mismatch = || Error::shape_mismatch("broadcast_to", shape, out);
    if shape.len() > out.len() {
        return Err(mismatch());
    }
    let lead = out.len() - shape.len();
    (0..out.len())
        .map(|i| {
            if i < lead {
                return Ok(0);
            }
            match (shape[i - lead], out[i]) {
                (x, y) if x == y => Ok(strides[i - lead]),
                (1, _) => Ok(0),
                _ => Err(mismatch()),
            }
        })
        .collect()
}

/// Strides for viewing a layout of `shape`/`strides` as `new_shape` without copying, or
/// `None` if the data must be copied. NumPy's `_attempt_nocopy_reshape` (C order).
///
/// Requires both shapes to have the same element count.
pub(crate) fn nocopy_reshape(
    shape: &[usize],
    strides: &[isize],
    new_shape: &[usize],
) -> Option<Strides> {
    if shape.contains(&0) || new_shape.iter().all(|&d| d == 1) {
        return Some(shape::row_contiguous_strides(new_shape));
    }
    let (old_dims, old_strides): (Vec<usize>, Vec<isize>) = shape
        .iter()
        .zip(strides)
        .filter(|(d, _)| **d != 1)
        .map(|(&d, &s)| (d, s))
        .unzip();
    let mut out: Strides = SmallVec::from_elem(1, new_shape.len());
    let (mut oi, mut oj, mut ni, mut nj) = (0, 1, 0, 1);
    while ni < new_shape.len() && oi < old_dims.len() {
        let mut np = new_shape[ni];
        let mut op = old_dims[oi];
        while np != op {
            if np < op {
                np *= new_shape[nj];
                nj += 1;
            } else {
                op *= old_dims[oj];
                oj += 1;
            }
        }
        // The merged old dimensions must be contiguous among themselves.
        for ok in oi..oj - 1 {
            if old_strides[ok] != old_dims[ok + 1] as isize * old_strides[ok + 1] {
                return None;
            }
        }
        out[nj - 1] = old_strides[oj - 1];
        for nk in (ni + 1..nj).rev() {
            out[nk - 1] = out[nk] * new_shape[nk] as isize;
        }
        ni = nj;
        nj += 1;
        oi = oj;
        oj += 1;
    }
    Some(out)
}

/// Element count of `requested` with at most one `-1` inferred to give `size` elements.
pub(crate) fn infer_shape(size: usize, requested: &[isize]) -> Result<Shape> {
    let invalid = || Error::invalid_shape("reshape", requested.iter().map(|&d| d as i64));
    if requested.iter().filter(|&&d| d == -1).count() > 1 || requested.iter().any(|&d| d < -1) {
        return Err(invalid());
    }
    let known = requested
        .iter()
        .filter(|&&d| d >= 0)
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or_else(invalid)?;
    let inferred = if requested.contains(&-1) {
        if known == 0 || !size.is_multiple_of(known) {
            return Err(invalid());
        }
        size / known
    } else if known == size {
        0
    } else {
        return Err(invalid());
    };
    Ok(requested
        .iter()
        .map(|&d| if d == -1 { inferred } else { d as usize })
        .collect())
}

/// Shape and strides with axes reordered: output axis `i` is input axis `axes[i]`.
pub(crate) fn permute(
    shape: &[usize],
    strides: &[isize],
    axes: &[usize],
) -> Result<(Shape, Strides)> {
    let mut seen = vec![false; shape.len()];
    let valid = axes.len() == shape.len()
        && axes
            .iter()
            .all(|&a| a < shape.len() && !std::mem::replace(&mut seen[a], true));
    if !valid {
        return Err(Error::InvalidArgument(format!(
            "transpose axes {axes:?} are not a permutation of {} dimensions",
            shape.len()
        )));
    }
    Ok((
        axes.iter().map(|&a| shape[a]).collect(),
        axes.iter().map(|&a| strides[a]).collect(),
    ))
}

/// `axis` as an index into `ndim` dimensions; negative axes count from the end.
pub(crate) fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    let a = if axis < 0 { axis + ndim as isize } else { axis };
    if (0..ndim as isize).contains(&a) {
        Ok(a as usize)
    } else {
        Err(Error::invalid_axis("axis", axis as i64, ndim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slices(n: usize, start: Option<isize>, stop: Option<isize>, step: isize) -> DimSlice {
        normalize_slice(&[n], &[SliceArg::new(start, stop, step)]).unwrap()[0]
    }

    #[test]
    fn python_slice_semantics() {
        let d = |start, step, len| DimSlice { start, step, len };
        assert_eq!(slices(5, None, None, 1), d(0, 1, 5));
        assert_eq!(slices(5, None, None, -1), d(4, -1, 5));
        assert_eq!(slices(5, Some(-2), None, 1), d(3, 1, 2));
        assert_eq!(slices(5, Some(1), Some(100), 2), d(1, 2, 2));
        assert_eq!(slices(5, None, Some(-6), -2), d(4, -2, 3));
        assert_eq!(slices(5, Some(3), Some(1), 1).len, 0);
        assert_eq!(slices(5, Some(0), Some(5), 10), d(0, 10, 1));
        assert!(normalize_slice(&[5], &[SliceArg::new(None, None, 0)]).is_err());
        assert!(normalize_slice(&[5, 2], &[SliceArg::full()]).is_err());
    }

    #[test]
    fn broadcasting() {
        assert_eq!(
            broadcast_shapes("t", &[3, 1], &[4]).unwrap().as_slice(),
            &[3, 4]
        );
        assert_eq!(
            broadcast_shapes("t", &[], &[2, 0]).unwrap().as_slice(),
            &[2, 0]
        );
        assert!(broadcast_shapes("t", &[2, 3], &[4]).is_err());
        assert_eq!(
            broadcast_strides(&[3, 1], &[1, 1], &[2, 3, 4])
                .unwrap()
                .as_slice(),
            &[0, 1, 0]
        );
        assert!(broadcast_strides(&[3], &[1], &[4]).is_err());
    }

    #[test]
    fn reshape_views() {
        // Contiguous: always a view.
        assert_eq!(
            nocopy_reshape(&[2, 3], &[3, 1], &[3, 2])
                .unwrap()
                .as_slice(),
            &[2, 1]
        );
        assert_eq!(
            nocopy_reshape(&[2, 3, 4], &[12, 4, 1], &[6, 4])
                .unwrap()
                .as_slice(),
            &[4, 1]
        );
        // Transposed 2x3 -> 6: needs a copy.
        assert!(nocopy_reshape(&[3, 2], &[1, 3], &[6]).is_none());
        // Transposed, but splitting one dimension is fine.
        assert_eq!(
            nocopy_reshape(&[4, 2], &[1, 4], &[2, 2, 2])
                .unwrap()
                .as_slice(),
            &[2, 1, 4]
        );
        // Unit dimensions are free (the stride of a size-1 dimension is arbitrary).
        let s = nocopy_reshape(&[3, 2], &[1, 3], &[3, 1, 2]).unwrap();
        assert_eq!((s[0], s[2]), (1, 3));
    }

    #[test]
    fn shape_inference_and_axes() {
        assert_eq!(infer_shape(6, &[-1, 2]).unwrap().as_slice(), &[3, 2]);
        assert!(infer_shape(6, &[-1, -1]).is_err());
        assert!(infer_shape(6, &[4, -1]).is_err());
        assert!(infer_shape(0, &[0, -1]).is_err());
        assert!(infer_shape(6, &[5]).is_err());
        assert_eq!(normalize_axis(-1, 3).unwrap(), 2);
        assert!(normalize_axis(3, 3).is_err());
        assert!(permute(&[2, 3], &[3, 1], &[0, 0]).is_err());
        let (s, st) = permute(&[2, 3, 4], &[12, 4, 1], &[2, 0, 1]).unwrap();
        assert_eq!(
            (s.as_slice(), st.as_slice()),
            (&[4, 2, 3][..], &[1, 12, 4][..])
        );
    }
}
