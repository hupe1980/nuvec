//! MLX's element-wise compositions (`ops.cpp`): functions defined over the primitives,
//! with MLX's dtypes and gradients (they inherit the primitives' rules).

use std::panic::Location;

use super::{arrays_of, imp, isize_axes, resolve_pair};
use crate::axes::Axes;
use crate::scalar::ScalarOrArray;
use crate::{Array, Category, DType, Result, at_least_float};

/// The tolerances of [`isclose`] and [`allclose`] (MLX's defaults: `rtol = 1e-5`,
/// `atol = 1e-8`, NaN unequal).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CloseOptions {
    /// The relative tolerance, times `|b|`.
    pub rtol: f64,
    /// The absolute tolerance.
    pub atol: f64,
    /// Whether NaN equals NaN.
    pub equal_nan: bool,
}

impl Default for CloseOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-5,
            atol: 1e-8,
            equal_nan: false,
        }
    }
}

impl CloseOptions {
    /// MLX's defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the relative tolerance.
    pub fn rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the absolute tolerance.
    pub fn atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }

    /// Sets whether NaN equals NaN.
    pub fn equal_nan(mut self, equal_nan: bool) -> Self {
        self.equal_nan = equal_nan;
        self
    }
}

/// The values [`nan_to_num`] writes (MLX's defaults: 0, and the dtype's largest and
/// smallest finite values).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct NanToNum {
    /// The value for NaN.
    pub nan: f64,
    /// The value for +infinity; the dtype's largest finite value when unset.
    pub posinf: Option<f64>,
    /// The value for −infinity; the dtype's smallest finite value when unset.
    pub neginf: Option<f64>,
}

impl NanToNum {
    /// MLX's defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the value for NaN.
    pub fn nan(mut self, value: f64) -> Self {
        self.nan = value;
        self
    }

    /// Sets the value for +infinity.
    pub fn posinf(mut self, value: f64) -> Self {
        self.posinf = Some(value);
        self
    }

    /// Sets the value for −infinity.
    pub fn neginf(mut self, value: f64) -> Self {
        self.neginf = Some(value);
        self
    }
}

#[track_caller]
fn sticky1(name: &'static str, a: &Array, f: impl FnOnce() -> Result<Array>) -> Array {
    Array::sticky(name, Location::caller(), &[a], f)
}

/// All of `a`'s axes.
fn every_axis(a: &Array) -> Vec<isize> {
    (0..a.ndim() as isize).collect()
}

fn is_float(d: DType) -> bool {
    d.category() == Category::Float
}

fn is_real_or_complex_float(d: DType) -> bool {
    matches!(d.category(), Category::Float | Category::Complex)
}

/// Element-wise `a · a`, in `a`'s dtype.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// assert_eq!(ops::square(&array![-3i8, 2]).to_vec::<i8>().unwrap(), [9, 4]);
/// ```
#[track_caller]
pub fn square(a: &Array) -> Array {
    sticky1("square", a, || imp::multiply(a, a))
}

/// Element-wise `1 / a`, in at least float32.
#[track_caller]
pub fn reciprocal(a: &Array) -> Array {
    sticky1("reciprocal", a, || imp::divide(1.0, a))
}

/// Element-wise conversion from radians to degrees, in at least float32.
#[track_caller]
pub fn degrees(a: &Array) -> Array {
    sticky1("degrees", a, || {
        imp::multiply(
            &imp::astype(a, at_least_float(a.dtype()))?,
            180.0 / std::f64::consts::PI,
        )
    })
}

/// Element-wise conversion from degrees to radians, in at least float32.
#[track_caller]
pub fn radians(a: &Array) -> Array {
    sticky1("radians", a, || {
        imp::multiply(
            &imp::astype(a, at_least_float(a.dtype()))?,
            std::f64::consts::PI / 180.0,
        )
    })
}

/// `a` unchanged (MLX's `positive`).
#[track_caller]
pub fn positive(a: &Array) -> Array {
    a.clone()
}

/// Element-wise truncation toward zero; integers and booleans are returned unchanged.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// assert_eq!(ops::trunc(&array![-1.5f32, 1.5]).to_vec::<f32>().unwrap(), [-1.0, 1.0]);
/// ```
#[track_caller]
pub fn trunc(a: &Array) -> Array {
    sticky1("trunc", a, || {
        if !is_float(a.dtype()) {
            return Ok(a.clone());
        }
        imp::where_(&imp::less(a, 0.0)?, &imp::ceil(a)?, &imp::floor(a)?)
    })
}

/// The real part of a complex array (float32); real arrays as float32.
#[track_caller]
pub fn real(a: &Array) -> Array {
    sticky1("real", a, || imp::astype(a, DType::F32))
}

/// The imaginary part of a complex array (float32); zeros for real arrays.
#[track_caller]
pub fn imag(a: &Array) -> Array {
    sticky1("imag", a, || {
        if a.dtype() != DType::Complex64 {
            return imp::multiply(&imp::astype(a, DType::F32)?, 0.0);
        }
        // imag(z) = re(−i·z)
        let minus_i = crate::creation::from_data(&[crate::Complex64::new(0.0, -1.0)], &[])?;
        imp::astype(&imp::multiply(a, &minus_i)?, DType::F32)
    })
}

/// The complex conjugate (MLX's `conj`, the same as [`conjugate`](super::conjugate)).
#[track_caller]
pub fn conj(a: &Array) -> Array {
    super::conjugate(a)
}

/// Element-wise logical negation (`a == 0`), as booleans.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// assert_eq!(ops::logical_not(&array![0.0f32, 2.0]).to_vec::<bool>().unwrap(), [true, false]);
/// ```
#[track_caller]
pub fn logical_not(a: &Array) -> Array {
    sticky1("logical_not", a, || imp::equal(a, 0))
}

#[track_caller]
fn logical(
    name: &'static str,
    a: impl Into<ScalarOrArray>,
    b: impl Into<ScalarOrArray>,
    f: fn(&Array, &Array) -> Result<Array>,
) -> Array {
    let (a, b) = (a.into(), b.into());
    let inputs = arrays_of(&a, &b);
    Array::sticky(name, Location::caller(), &inputs, || {
        let (a, b) = resolve_pair(&a, &b)?;
        f(
            &imp::astype(&a, DType::Bool)?,
            &imp::astype(&b, DType::Bool)?,
        )
    })
}

/// Element-wise logical AND of the operands' truth values (non-zero is true).
#[track_caller]
pub fn logical_and(a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> Array {
    logical("logical_and", a, b, |a, b| imp::multiply(a, b))
}

/// Element-wise logical OR of the operands' truth values (non-zero is true).
#[track_caller]
pub fn logical_or(a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> Array {
    logical("logical_or", a, b, |a, b| imp::add(a, b))
}

/// Element-wise logical XOR of the operands' truth values (non-zero is true).
#[track_caller]
pub fn logical_xor(a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> Array {
    logical("logical_xor", a, b, |a, b| imp::not_equal(a, b))
}

/// `(floor_divide(a, b), remainder(a, b))` (MLX's `divmod`).
///
/// ```
/// use nuvec_core::{array, ops};
///
/// let [q, r] = ops::divmod(&array![7, -7], 2);
/// assert_eq!((q.to_vec::<i32>().unwrap(), r.to_vec::<i32>().unwrap()), (vec![3, -4], vec![1, 1]));
/// ```
#[track_caller]
pub fn divmod(a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> [Array; 2] {
    let (a, b) = (a.into(), b.into());
    [
        super::floor_divide(a.clone(), b.clone()),
        super::remainder(a, b),
    ]
}

/// Whether each element is NaN (false for integers and booleans).
#[track_caller]
pub fn isnan(a: &Array) -> Array {
    sticky1("isnan", a, || imp::not_equal(a, a))
}

/// Whether each element is ±infinity (false for integers and booleans).
#[track_caller]
pub fn isinf(a: &Array) -> Array {
    sticky1("isinf", a, || inf(a))
}

/// Whether each element is finite (true for integers and booleans).
#[track_caller]
pub fn isfinite(a: &Array) -> Array {
    sticky1("isfinite", a, || {
        if !is_real_or_complex_float(a.dtype()) {
            return imp::equal(a, a);
        }
        // Finite values have a finite difference with themselves (NaN and ±inf give NaN).
        let d = imp::subtract(a, a)?;
        imp::equal(&d, &d)
    })
}

/// Whether each element is +infinity.
#[track_caller]
pub fn isposinf(a: &Array) -> Array {
    sticky1("isposinf", a, || {
        if !is_float(a.dtype()) {
            return imp::not_equal(a, a);
        }
        imp::equal(a, f64::INFINITY)
    })
}

/// Whether each element is −infinity.
#[track_caller]
pub fn isneginf(a: &Array) -> Array {
    sticky1("isneginf", a, || {
        if !is_float(a.dtype()) {
            return imp::not_equal(a, a);
        }
        imp::equal(a, f64::NEG_INFINITY)
    })
}

/// NaN, +infinity and −infinity replaced by finite values ([`NanToNum`]), in `a`'s dtype;
/// other types unchanged.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// let x = array![f32::NAN, f32::INFINITY, 1.0];
/// assert_eq!(ops::nan_to_num(&x, ops::NanToNum::new()).to_vec::<f32>().unwrap(), [0.0, f32::MAX, 1.0]);
/// assert_eq!(ops::nan_to_num(&x, ops::NanToNum::new().nan(-1.0).posinf(9.0)).to_vec::<f32>().unwrap(), [-1.0, 9.0, 1.0]);
/// ```
#[track_caller]
pub fn nan_to_num(a: &Array, values: NanToNum) -> Array {
    sticky1("nan_to_num", a, || {
        let d = a.dtype();
        if !is_float(d) {
            return Ok(a.clone());
        }
        let (max, min) = match d {
            DType::F16 => (f64::from(half::f16::MAX), f64::from(half::f16::MIN)),
            DType::BF16 => (f64::from(half::bf16::MAX), f64::from(half::bf16::MIN)),
            DType::F32 => (f64::from(f32::MAX), f64::from(f32::MIN)),
            _ => (f64::MAX, f64::MIN),
        };
        let fill = |v: f64| imp::astype(&crate::creation::from_data(&[v], &[])?, d);
        let out = imp::where_(&imp::not_equal(a, a)?, &fill(values.nan)?, a)?;
        let out = imp::where_(
            &imp::equal(&out, f64::INFINITY)?,
            &fill(values.posinf.unwrap_or(max))?,
            &out,
        )?;
        imp::where_(
            &imp::equal(&out, f64::NEG_INFINITY)?,
            &fill(values.neginf.unwrap_or(min))?,
            &out,
        )
    })
}

/// `|a − b| ≤ atol + rtol·|b|` element-wise ([`CloseOptions`], MLX's formula with float32
/// tolerances); infinities are close only to themselves, NaN to NaN only with `equal_nan`.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// let c = ops::isclose(&array![1.0f32, 2.0], &array![1.0f32 + 1e-6, 2.1], ops::CloseOptions::new());
/// assert_eq!(c.to_vec::<bool>().unwrap(), [true, false]);
/// ```
#[track_caller]
pub fn isclose(a: &Array, b: &Array, options: CloseOptions) -> Array {
    Array::sticky("isclose", Location::caller(), &[a, b], || {
        close(a, b, options)
    })
}

fn close(a: &Array, b: &Array, options: CloseOptions) -> Result<Array> {
    // MLX's `isclose`: the tolerances are float32 arrays (not weak numbers), and only
    // infinities are corrected (equal infinities are close, others are not).
    let f32 = |v: f64| crate::creation::from_data(&[v as f32], &[]);
    let rhs = imp::add(
        &f32(options.atol)?,
        &imp::multiply(&f32(options.rtol)?, &imp::abs(b)?)?,
    )?;
    let lhs = imp::abs(&imp::subtract(a, b)?)?;
    let out = imp::less_equal(&lhs, &rhs)?;
    let any_inf = imp::add(&inf(a)?, &inf(b)?)?;
    let both_inf = imp::add(
        &imp::multiply(
            &imp::equal(a, f64::INFINITY)?,
            &imp::equal(b, f64::INFINITY)?,
        )?,
        &imp::multiply(
            &imp::equal(a, f64::NEG_INFINITY)?,
            &imp::equal(b, f64::NEG_INFINITY)?,
        )?,
    )?;
    let out = imp::add(
        &imp::multiply(&out, &imp::equal(&any_inf, false)?)?,
        &both_inf,
    )?;
    if options.equal_nan {
        let both = imp::multiply(&imp::not_equal(a, a)?, &imp::not_equal(b, b)?)?;
        return imp::add(&out, &both);
    }
    Ok(out)
}

/// Whether each element is ±infinity (false for integers and booleans).
fn inf(a: &Array) -> Result<Array> {
    if !is_real_or_complex_float(a.dtype()) {
        return imp::not_equal(a, a);
    }
    imp::equal(&imp::abs(a)?, f64::INFINITY)
}

/// Whether every element pair is close ([`isclose`]), as a 0-d boolean.
#[track_caller]
pub fn allclose(a: &Array, b: &Array, options: CloseOptions) -> Array {
    Array::sticky("allclose", Location::caller(), &[a, b], || {
        let c = close(a, b, options)?;
        imp::all(&c, &every_axis(&c), false)
    })
}

/// Whether `a` and `b` have the same shape and equal values (NaN equal with `equal_nan`), as
/// a 0-d boolean; dtypes may differ.
#[track_caller]
pub fn array_equal(a: &Array, b: &Array, equal_nan: bool) -> Array {
    Array::sticky("array_equal", Location::caller(), &[a, b], || {
        if a.dims() != b.dims() {
            return crate::creation::from_data(&[false], &[]);
        }
        let mut eq = imp::equal(a, b)?;
        if equal_nan {
            eq = imp::add(
                &eq,
                &imp::multiply(&imp::not_equal(a, a)?, &imp::not_equal(b, b)?)?,
            )?;
        }
        imp::all(&eq, &every_axis(&eq), false)
    })
}

/// `a` limited to `[min, max]`: `minimum(maximum(a, min), max)` (MLX's `clip`). One-sided
/// clipping is [`maximum`](super::maximum) or [`minimum`](super::minimum).
///
/// ```
/// use nuvec_core::{array, ops};
///
/// assert_eq!(ops::clip(&array![-2.0f32, 0.5, 3.0], 0.0, 1.0).to_vec::<f32>().unwrap(), [0.0, 0.5, 1.0]);
/// ```
#[track_caller]
pub fn clip(a: &Array, min: impl Into<ScalarOrArray>, max: impl Into<ScalarOrArray>) -> Array {
    let lower = super::maximum(a, min);
    super::minimum(&lower, max)
}

/// The number of non-zero elements over `axes`, as int32 (MLX's `count_nonzero`).
#[track_caller]
pub fn count_nonzero(a: &Array, axes: impl Axes, keepdims: bool) -> Array {
    let spec = axes.axes_spec();
    sticky1("count_nonzero", a, || {
        let axes = isize_axes(&spec.sorted(a.ndim())?);
        imp::astype(
            &imp::sum(&imp::not_equal(a, 0)?, &axes, keepdims)?,
            DType::I32,
        )
    })
}

/// `(min, max)` over `axes` (MLX's `minmax`).
#[track_caller]
pub fn minmax(a: &Array, axes: impl Axes, keepdims: bool) -> [Array; 2] {
    let spec = axes.axes_spec();
    let axes = || spec.sorted(a.ndim()).map(|x| isize_axes(&x));
    [
        sticky1("minmax", a, || imp::min(a, &axes()?, keepdims)),
        sticky1("minmax", a, || imp::max(a, &axes()?, keepdims)),
    ]
}

/// # Element-wise compositions (MLX's array methods)
impl Array {
    /// Element-wise `self · self`; see [`square`].
    #[track_caller]
    pub fn square(&self) -> Array {
        square(self)
    }

    /// Element-wise `1 / self`; see [`reciprocal`].
    #[track_caller]
    pub fn reciprocal(&self) -> Array {
        reciprocal(self)
    }

    /// The complex conjugate; see [`conj`].
    #[track_caller]
    pub fn conj(&self) -> Array {
        conj(self)
    }

    /// The real part; see [`real`].
    #[track_caller]
    pub fn real(&self) -> Array {
        real(self)
    }

    /// The imaginary part; see [`imag`].
    #[track_caller]
    pub fn imag(&self) -> Array {
        imag(self)
    }
}

// MLX's NumPy-style aliases.

/// [`arccos`](super::arccos) (MLX's `acos`).
#[track_caller]
pub fn acos(a: &Array) -> Array {
    super::arccos(a)
}

/// [`arccosh`](super::arccosh) (MLX's `acosh`).
#[track_caller]
pub fn acosh(a: &Array) -> Array {
    super::arccosh(a)
}

/// [`arcsin`](super::arcsin) (MLX's `asin`).
#[track_caller]
pub fn asin(a: &Array) -> Array {
    super::arcsin(a)
}

/// [`arcsinh`](super::arcsinh) (MLX's `asinh`).
#[track_caller]
pub fn asinh(a: &Array) -> Array {
    super::arcsinh(a)
}

/// [`arctan`](super::arctan) (MLX's `atan`).
#[track_caller]
pub fn atan(a: &Array) -> Array {
    super::arctan(a)
}

/// [`arctanh`](super::arctanh) (MLX's `atanh`).
#[track_caller]
pub fn atanh(a: &Array) -> Array {
    super::arctanh(a)
}

/// [`arctan2`](super::arctan2) (MLX's `atan2`).
#[track_caller]
pub fn atan2(a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> Array {
    super::arctan2(a, b)
}

/// [`left_shift`](super::left_shift) (MLX's `bitwise_left_shift`).
#[track_caller]
pub fn bitwise_left_shift(a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> Array {
    super::left_shift(a, b)
}

/// [`right_shift`](super::right_shift) (MLX's `bitwise_right_shift`).
#[track_caller]
pub fn bitwise_right_shift(a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> Array {
    super::right_shift(a, b)
}
