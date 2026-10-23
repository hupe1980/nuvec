//! Axes and shapes as arguments: the sealed [`Axes`] and [`IntoShape`] traits, and the
//! option structs of the reductions and scans that have rare knobs.

use std::ops::{RangeFrom, RangeFull};

use smallvec::SmallVec;

use crate::shape::Shape;
use crate::{Error, Result};

mod sealed {
    pub trait Sealed {}
    impl<T: Sealed + ?Sized> Sealed for &T {}
}

/// `v` as an `i64` (saturating: an absurd `usize` fails validation, never wraps).
pub(crate) fn wide(v: usize) -> i64 {
    i64::try_from(v).unwrap_or(i64::MAX)
}

/// Which axes an operation works on, before it is resolved against an array.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxesSpec {
    /// Every axis (`..`, MLX's `axis=None`).
    All,
    /// The listed axes, negative counting from the end; an empty list means none.
    List(SmallVec<[i64; 4]>),
    /// Axes `k..ndim` (`k..`), `k` negative counting from the end.
    From(i64),
}

/// Axes argument of reductions and shape ops: one axis (`-1`, `x.ndim() - 1`), a list
/// (`[0, 2]`, `&axes[..]`, `vec![…]`, or a reference to any of these), all axes (`..`,
/// MLX's `axis=None`) or the trailing axes from one on (`1..`).
///
/// An empty list means *no* axes, as in MLX C++ (`sum(a, {})`). Negative axes count from the
/// end. Out-of-range or repeated axes make the operation fail with
/// [`crate::OpErrorKind::InvalidAxis`].
///
/// ```
/// use nuvec_core::Array;
///
/// let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]);
/// assert_eq!(x.sum(-1).shape(), [1, 2]);
/// assert_eq!(x.sum([0, 2]).shape(), [2]);
/// assert_eq!(x.sum(..).shape(), [0i32; 0]);
/// assert_eq!(x.sum(1..).shape(), [1]);
/// assert_eq!(x.sum([]).shape(), [1, 2, 3]); // no axis reduced
/// assert_eq!(x.sum(x.ndim() - 1).shape(), [1, 2]);
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an axes argument",
    label = "expected an axis (`-1`), a list (`[0, 2]`, `&v[..]`), all axes (`..`) or trailing axes (`1..`)"
)]
pub trait Axes: sealed::Sealed {
    #[doc(hidden)]
    fn axes_spec(&self) -> AxesSpec;
}

#[diagnostic::do_not_recommend]
impl<T: Axes + ?Sized> Axes for &T {
    fn axes_spec(&self) -> AxesSpec {
        (**self).axes_spec()
    }
}

impl sealed::Sealed for i32 {}
impl Axes for i32 {
    fn axes_spec(&self) -> AxesSpec {
        AxesSpec::List(SmallVec::from_elem(i64::from(*self), 1))
    }
}

impl sealed::Sealed for usize {}
impl Axes for usize {
    fn axes_spec(&self) -> AxesSpec {
        AxesSpec::List(SmallVec::from_elem(wide(*self), 1))
    }
}

impl sealed::Sealed for [i32] {}
impl Axes for [i32] {
    fn axes_spec(&self) -> AxesSpec {
        AxesSpec::List(self.iter().map(|&a| i64::from(a)).collect())
    }
}

impl sealed::Sealed for [usize] {}
impl Axes for [usize] {
    fn axes_spec(&self) -> AxesSpec {
        AxesSpec::List(self.iter().map(|&a| wide(a)).collect())
    }
}

impl sealed::Sealed for Vec<i32> {}
impl Axes for Vec<i32> {
    fn axes_spec(&self) -> AxesSpec {
        self.as_slice().axes_spec()
    }
}

impl sealed::Sealed for Vec<usize> {}
impl Axes for Vec<usize> {
    fn axes_spec(&self) -> AxesSpec {
        self.as_slice().axes_spec()
    }
}

impl<const N: usize> sealed::Sealed for [i32; N] {}
impl<const N: usize> Axes for [i32; N] {
    fn axes_spec(&self) -> AxesSpec {
        self.as_slice().axes_spec()
    }
}

impl sealed::Sealed for RangeFull {}
impl Axes for RangeFull {
    fn axes_spec(&self) -> AxesSpec {
        AxesSpec::All
    }
}

impl sealed::Sealed for RangeFrom<i32> {}
impl Axes for RangeFrom<i32> {
    fn axes_spec(&self) -> AxesSpec {
        AxesSpec::From(i64::from(self.start))
    }
}

impl sealed::Sealed for RangeFrom<usize> {}
impl Axes for RangeFrom<usize> {
    fn axes_spec(&self) -> AxesSpec {
        AxesSpec::From(wide(self.start))
    }
}

/// The axis argument of operations that work along one axis (`argmax`, `sort`, the scans,
/// `take`, `split`, `concatenate`, …): an `i32` or `usize` axis (negative counts from the
/// end), or `..` for "the flattened array" where the operation has that meaning (as MLX's
/// `axis=None`); elsewhere `..` fails with [`crate::OpErrorKind::InvalidAxis`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not an axis argument",
    label = "expected one axis (`-1`, `x.ndim() - 1`) or `..` (the flattened array)"
)]
pub trait Axis: sealed::Sealed {
    /// The axis, or `None` for `..`.
    #[doc(hidden)]
    fn axis_spec(&self) -> Option<i64>;
}

#[diagnostic::do_not_recommend]
impl<T: Axis + ?Sized> Axis for &T {
    fn axis_spec(&self) -> Option<i64> {
        (**self).axis_spec()
    }
}

impl Axis for i32 {
    fn axis_spec(&self) -> Option<i64> {
        Some(i64::from(*self))
    }
}

impl Axis for usize {
    fn axis_spec(&self) -> Option<i64> {
        Some(wide(*self))
    }
}

impl Axis for RangeFull {
    fn axis_spec(&self) -> Option<i64> {
        None
    }
}

/// One axis normalised against `ndim` dimensions, or `None` for `..` (flattened).
pub(crate) fn one_axis(spec: Option<i64>, ndim: usize) -> Result<Option<usize>> {
    spec.map(|a| normalize(a, ndim)).transpose()
}

/// An axis where `..` has no meaning: the axis as given (negative counts from the end, the
/// operation normalises it), or an invalid-axis error for `..`.
pub(crate) fn given_axis(spec: Option<i64>, ndim: usize) -> Result<isize> {
    match spec {
        Some(a) => Ok(a.clamp(isize::MIN as i64, isize::MAX as i64) as isize),
        None => Err(Error::invalid_axis("axis", wide(ndim), ndim)),
    }
}

/// `axis` as an index into `ndim` dimensions, or an invalid-axis error.
pub(crate) fn normalize(axis: i64, ndim: usize) -> Result<usize> {
    let n = ndim as i64;
    let a = if axis < 0 { axis + n } else { axis };
    if (0..n).contains(&a) {
        Ok(a as usize)
    } else {
        Err(Error::invalid_axis("axis", axis, ndim))
    }
}

impl AxesSpec {
    /// The axes in the given order, normalised against `ndim`: every axis for `All`, `k..` for
    /// `From(k)`. Out-of-range and repeated axes are errors.
    pub(crate) fn list(&self, ndim: usize) -> Result<SmallVec<[usize; 4]>> {
        match self {
            Self::All => Ok((0..ndim).collect()),
            Self::From(k) => {
                // `ndim..` is valid and empty, like a slice.
                let n = ndim as i64;
                let start = if *k < 0 { k + n } else { *k };
                if !(0..=n).contains(&start) {
                    return Err(Error::invalid_axis("axis", *k, ndim));
                }
                Ok((start as usize..ndim).collect())
            }
            Self::List(axes) => {
                let mut out = SmallVec::<[usize; 4]>::with_capacity(axes.len());
                for &axis in axes {
                    let a = normalize(axis, ndim)?;
                    if out.contains(&a) {
                        return Err(Error::invalid_axis("axis", axis, ndim));
                    }
                    out.push(a);
                }
                Ok(out)
            }
        }
    }

    /// The axes sorted (as reductions want them).
    pub(crate) fn sorted(&self, ndim: usize) -> Result<SmallVec<[usize; 4]>> {
        let mut axes = self.list(ndim)?;
        axes.sort_unstable();
        Ok(axes)
    }
}

/// A shape argument: `[2, 3]` (of `i32` or `usize`), a slice, a vector, a reference to any of
/// these, or `()` for a 0-D array. Every dimension must be in `0..=i32::MAX`; `reshape` also
/// accepts one `-1` (inferred). Anything else makes the operation fail with
/// [`crate::OpErrorKind::InvalidShape`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a shape argument",
    label = "expected a shape such as `[2, 3]`, `&dims[..]`, `vec![2, 3]` or `()`"
)]
pub trait IntoShape: sealed::Sealed {
    #[doc(hidden)]
    fn shape_spec(&self) -> SmallVec<[i64; 4]>;
}

#[diagnostic::do_not_recommend]
impl<T: IntoShape + ?Sized> IntoShape for &T {
    fn shape_spec(&self) -> SmallVec<[i64; 4]> {
        (**self).shape_spec()
    }
}

impl IntoShape for [i32] {
    fn shape_spec(&self) -> SmallVec<[i64; 4]> {
        self.iter().map(|&d| i64::from(d)).collect()
    }
}

impl IntoShape for [usize] {
    fn shape_spec(&self) -> SmallVec<[i64; 4]> {
        self.iter().map(|&d| wide(d)).collect()
    }
}

impl<const N: usize> IntoShape for [i32; N] {
    fn shape_spec(&self) -> SmallVec<[i64; 4]> {
        self.as_slice().shape_spec()
    }
}

impl<const N: usize> sealed::Sealed for [usize; N] {}
impl<const N: usize> IntoShape for [usize; N] {
    fn shape_spec(&self) -> SmallVec<[i64; 4]> {
        self.as_slice().shape_spec()
    }
}

impl IntoShape for Vec<i32> {
    fn shape_spec(&self) -> SmallVec<[i64; 4]> {
        self.as_slice().shape_spec()
    }
}

impl IntoShape for Vec<usize> {
    fn shape_spec(&self) -> SmallVec<[i64; 4]> {
        self.as_slice().shape_spec()
    }
}

impl sealed::Sealed for () {}
impl IntoShape for () {
    fn shape_spec(&self) -> SmallVec<[i64; 4]> {
        SmallVec::new()
    }
}

/// A requested shape as dimension sizes: every dimension in `0..=i32::MAX`.
pub(crate) fn dims(spec: &[i64]) -> Result<Shape> {
    spec.iter()
        .map(|&d| {
            if (0..=i64::from(i32::MAX)).contains(&d) {
                Ok(d as usize)
            } else {
                Err(Error::invalid_shape("shape", spec.iter().copied()))
            }
        })
        .collect()
}

/// A requested `reshape` target for `size` elements: dimensions in `0..=i32::MAX` and at
/// most one `-1`, inferred.
pub(crate) fn reshape_dims(spec: &[i64], size: usize) -> Result<Shape> {
    let invalid = || Error::invalid_shape("reshape", spec.iter().copied());
    if spec.iter().any(|&d| d < -1 || d > i64::from(i32::MAX)) {
        return Err(invalid());
    }
    let requested: Vec<isize> = spec.iter().map(|&d| d as isize).collect();
    crate::view::infer_shape(size, &requested)
}

/// Rare knobs of [`crate::Array::var_with`] and [`crate::Array::std_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VarOptions {
    /// Delta degrees of freedom: the divisor is `n - ddof` (clamped at 0). 0 by default.
    pub ddof: i32,
    /// Keep the reduced axes as dimensions of size 1.
    pub keepdims: bool,
}

/// Rare knobs of the scans ([`crate::Array::cumsum_with`] and the other `*_with` scans).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanOptions {
    /// Scan from the end of the axis towards its start.
    pub reverse: bool,
    /// Leave each element out of its own output (MLX's `inclusive=False`).
    pub exclusive: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution() {
        assert_eq!(AxesSpec::All.list(3).unwrap().as_slice(), [0, 1, 2]);
        assert_eq!((-1).axes_spec().list(3).unwrap().as_slice(), [2]);
        assert_eq!((1..).axes_spec().list(3).unwrap().as_slice(), [1, 2]);
        assert_eq!((-1..).axes_spec().list(3).unwrap().as_slice(), [2]);
        assert!((3..).axes_spec().list(3).unwrap().is_empty());
        assert!((4..).axes_spec().list(3).is_err());
        assert!([0, -3].axes_spec().list(3).is_err(), "repeated");
        assert!(3.axes_spec().list(3).is_err());
        assert_eq!([2, 0].axes_spec().sorted(3).unwrap().as_slice(), [0, 2]);
        assert!(
            <[i32; 0]>::default()
                .axes_spec()
                .list(3)
                .unwrap()
                .is_empty()
        );
        assert_eq!(one_axis((..).axis_spec(), 3).unwrap(), None);
        assert_eq!(one_axis((-2).axis_spec(), 3).unwrap(), Some(1));
        assert!(one_axis(usize::MAX.axis_spec(), 3).is_err(), "no wrap");
        assert!(usize::MAX.axes_spec().list(3).is_err(), "no wrap");
        assert!(given_axis((..).axis_spec(), 3).is_err());
        assert_eq!((&&[0, 2]).axes_spec().list(3).unwrap().as_slice(), [0, 2]);
        assert_eq!((1usize..).axes_spec().list(3).unwrap().as_slice(), [1, 2]);
    }

    #[test]
    fn shapes() {
        assert_eq!(dims(&[2, 3]).unwrap().as_slice(), [2, 3]);
        assert!(dims(&[-1]).is_err());
        assert!(dims(&[i64::from(i32::MAX) + 1]).is_err());
        assert_eq!(reshape_dims(&[-1, 2], 6).unwrap().as_slice(), [3, 2]);
        assert!(reshape_dims(&[-2, 3], 6).is_err());
        assert!(reshape_dims(&[-1, -1], 6).is_err());
        assert_eq!(().shape_spec().len(), 0);
        assert_eq!([2usize, 3].shape_spec().as_slice(), [2, 3]);
        let r: &[usize; 2] = &[2, 3];
        assert_eq!(r.shape_spec().as_slice(), [2, 3]);
        assert!(dims(&[usize::MAX].shape_spec()).is_err(), "no wrap");
    }
}
