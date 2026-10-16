//! Array operations.
//!
//! Following MLX, operations promote operand types and broadcast shapes here, by inserting
//! `AsType` and `Broadcast` nodes, so primitives always see operands of equal shape and
//! dtype (D-019). Views (`reshape`, `transpose`, `slice`, `broadcast_to`, `expand_dims`,
//! `squeeze`) share their input's data whenever the layout permits.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::buffer::Buffer;
use crate::cast::{Canon, Scalar, with_type};
use crate::primitive::{ArgReduceOp, BinaryOp, Primitive, ReduceOp, UnaryOp};
use crate::shape::{self, Shape};
use crate::view::{self, SliceArg};
use crate::{Array, Category, DType, Error, Result, at_least_float, promote_types};

fn single(primitive: Primitive, inputs: Vec<Array>) -> Result<Array> {
    let mut outputs = Array::from_op(primitive, inputs)?;
    Ok(outputs.swap_remove(0))
}

/// Returns the error of the first input that has already failed, so that chained
/// operations report the original cause.
fn check_inputs(inputs: &[&Array]) -> Result<()> {
    match inputs.iter().find_map(|a| a.error()) {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

// ── Element-wise binary ─────────────────────────────────────────────────────

fn binary(op: BinaryOp, a: &Array, b: &Array) -> Result<Array> {
    check_inputs(&[a, b])?;
    let mut dtype = promote_types(a.dtype(), b.dtype());
    if op == BinaryOp::Divide {
        dtype = at_least_float(dtype);
    }
    let ordered = matches!(
        op,
        BinaryOp::Maximum
            | BinaryOp::Minimum
            | BinaryOp::Less
            | BinaryOp::LessEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterEqual
    );
    if ordered && dtype == DType::Complex64 {
        return Err(Error::UnsupportedDType {
            op: op.name(),
            dtype,
        });
    }
    let shape = view::broadcast_shapes(op.name(), a.shape(), b.shape())?;
    let a = broadcast_to(&astype(a, dtype)?, &shape)?;
    let b = broadcast_to(&astype(b, dtype)?, &shape)?;
    single(Primitive::Binary(op), vec![a, b])
}

macro_rules! binary_fn {
    ($($(#[$doc:meta])* $name:ident => $op:ident;)*) => {$(
        $(#[$doc])*
        pub fn $name(a: &Array, b: &Array) -> Result<Array> {
            binary(BinaryOp::$op, a, b)
        }
    )*};
}

binary_fn! {
    /// Element-wise `a + b`, with promotion and broadcasting. Integers wrap; for booleans
    /// this is logical OR.
    add => Add;
    /// Element-wise `a - b`, with promotion and broadcasting. Integers wrap; for booleans
    /// this is logical XOR.
    subtract => Subtract;
    /// Element-wise `a * b`, with promotion and broadcasting. Integers wrap; for booleans
    /// this is logical AND.
    multiply => Multiply;
    /// Element-wise true division `a / b`. Integer operands are promoted to floating point.
    divide => Divide;
    /// Element-wise maximum; NaN propagates. Not defined for complex numbers.
    maximum => Maximum;
    /// Element-wise minimum; NaN propagates. Not defined for complex numbers.
    minimum => Minimum;
    /// Element-wise `a == b` as a bool array.
    equal => Equal;
    /// Element-wise `a != b` as a bool array.
    not_equal => NotEqual;
    /// Element-wise `a < b` as a bool array. Not defined for complex numbers.
    less => Less;
    /// Element-wise `a <= b` as a bool array. Not defined for complex numbers.
    less_equal => LessEqual;
    /// Element-wise `a > b` as a bool array. Not defined for complex numbers.
    greater => Greater;
    /// Element-wise `a >= b` as a bool array. Not defined for complex numbers.
    greater_equal => GreaterEqual;
}

// ── Element-wise unary ──────────────────────────────────────────────────────

fn unary(op: UnaryOp, a: &Array) -> Result<Array> {
    check_inputs(&[a])?;
    let dtype = if op.is_transcendental() {
        at_least_float(a.dtype())
    } else {
        a.dtype()
    };
    let unsupported = match op {
        UnaryOp::Negative => dtype == DType::Bool,
        UnaryOp::Abs => false,
        _ => dtype == DType::Complex64,
    };
    if unsupported {
        return Err(Error::UnsupportedDType {
            op: op.name(),
            dtype,
        });
    }
    single(Primitive::Unary(op), vec![astype(a, dtype)?])
}

macro_rules! unary_fn {
    ($($(#[$doc:meta])* $name:ident => $op:ident;)*) => {$(
        $(#[$doc])*
        pub fn $name(a: &Array) -> Result<Array> {
            unary(UnaryOp::$op, a)
        }
    )*};
}

unary_fn! {
    /// Element-wise `-a`. Integers wrap. Not defined for booleans.
    negative => Negative;
    /// Element-wise absolute value; the float32 magnitude for complex numbers.
    abs => Abs;
    /// Element-wise `e^a`, in at least float32.
    exp => Exp;
    /// Element-wise natural logarithm, in at least float32.
    log => Log;
    /// Element-wise square root, in at least float32.
    sqrt => Sqrt;
    /// Element-wise sine, in at least float32.
    sin => Sin;
    /// Element-wise cosine, in at least float32.
    cos => Cos;
    /// Element-wise hyperbolic tangent, in at least float32.
    tanh => Tanh;
    /// Element-wise logistic function `1 / (1 + e^-a)`, in at least float32.
    sigmoid => Sigmoid;
}

/// Element-wise selection: `a` where `condition` is true, otherwise `b`.
///
/// `condition` is converted to bool (non-zero is true); `a` and `b` are promoted to a
/// common type; all three are broadcast together.
pub fn where_(condition: &Array, a: &Array, b: &Array) -> Result<Array> {
    check_inputs(&[condition, a, b])?;
    let dtype = promote_types(a.dtype(), b.dtype());
    let shape = view::broadcast_shapes("where", condition.shape(), a.shape())?;
    let shape = view::broadcast_shapes("where", &shape, b.shape())?;
    let condition = broadcast_to(&astype(condition, DType::Bool)?, &shape)?;
    let a = broadcast_to(&astype(a, dtype)?, &shape)?;
    let b = broadcast_to(&astype(b, dtype)?, &shape)?;
    single(Primitive::Select, vec![condition, a, b])
}

/// Converts the elements to `dtype`. Float-to-integer conversion saturates (NaN becomes
/// 0); conversion to bool tests for non-zero; complex-to-real keeps the real part.
pub fn astype(a: &Array, dtype: DType) -> Result<Array> {
    if a.dtype() == dtype {
        return Ok(a.clone());
    }
    single(Primitive::AsType(dtype), vec![a.clone()])
}

// ── Views ───────────────────────────────────────────────────────────────────

/// The same elements in another shape; one dimension may be `-1` (inferred). Shares the
/// data when the layout permits, otherwise copies.
pub fn reshape(a: &Array, shape: &[isize]) -> Result<Array> {
    let shape = view::infer_shape(a.size(), shape)?;
    reshape_to(a, shape)
}

fn reshape_to(a: &Array, shape: Shape) -> Result<Array> {
    if a.shape() == shape.as_slice() {
        return Ok(a.clone());
    }
    single(Primitive::Reshape(shape.to_vec()), vec![a.clone()])
}

/// Reorders the axes: output axis `i` is input axis `axes[i]`. Shares the data.
pub fn transpose(a: &Array, axes: &[usize]) -> Result<Array> {
    view::permute(a.shape(), a.strides(), axes)?;
    if axes.iter().enumerate().all(|(i, &axis)| i == axis) {
        return Ok(a.clone());
    }
    single(Primitive::Transpose(axes.to_vec()), vec![a.clone()])
}

/// Selects `start:stop:step` in every dimension, with Python semantics (negative indices
/// count from the end, bounds are clamped, steps may be negative). Shares the data.
pub fn slice(a: &Array, args: &[SliceArg]) -> Result<Array> {
    let dims = view::normalize_slice(a.shape(), args)?;
    let full = dims
        .iter()
        .zip(a.shape())
        .all(|(d, &n)| d.start == 0 && d.step == 1 && d.len == n);
    if full {
        return Ok(a.clone());
    }
    let primitive = Primitive::Slice {
        start: dims.iter().map(|d| d.start).collect(),
        step: dims.iter().map(|d| d.step).collect(),
        len: dims.iter().map(|d| d.len).collect(),
    };
    single(primitive, vec![a.clone()])
}

/// Presents `a` as the larger `shape` (NumPy broadcasting rules). Shares the data.
pub fn broadcast_to(a: &Array, shape: &[usize]) -> Result<Array> {
    if a.shape() == shape {
        return Ok(a.clone());
    }
    single(Primitive::Broadcast(shape.to_vec()), vec![a.clone()])
}

/// Inserts a dimension of size 1 at `axis` (negative counts from the end of the result).
/// Shares the data.
pub fn expand_dims(a: &Array, axis: isize) -> Result<Array> {
    let axis = view::normalize_axis(axis, a.ndim() + 1)?;
    let mut shape: Shape = a.shape().into();
    shape.insert(axis, 1);
    reshape_to(a, shape)
}

/// Removes dimensions of size 1: the given `axes`, or all of them if `axes` is empty.
/// Shares the data.
pub fn squeeze(a: &Array, axes: &[isize]) -> Result<Array> {
    let mut remove = vec![axes.is_empty(); a.ndim()];
    for &axis in axes {
        let i = view::normalize_axis(axis, a.ndim())?;
        if a.shape()[i] != 1 {
            return Err(Error::InvalidArgument(format!(
                "cannot squeeze axis {axis} of size {} in shape {:?}",
                a.shape()[i],
                a.shape()
            )));
        }
        remove[i] = true;
    }
    let shape = a
        .shape()
        .iter()
        .zip(&remove)
        .filter(|&(&d, &r)| !(r && d == 1))
        .map(|(&d, _)| d)
        .collect();
    reshape_to(a, shape)
}

// ── Reductions ──────────────────────────────────────────────────────────────

/// Sorted, unique axes; an empty list means every axis.
fn normalize_axes(a: &Array, axes: &[isize]) -> Result<Vec<usize>> {
    if axes.is_empty() {
        return Ok((0..a.ndim()).collect());
    }
    let mut out = axes
        .iter()
        .map(|&axis| view::normalize_axis(axis, a.ndim()))
        .collect::<Result<Vec<_>>>()?;
    out.sort_unstable();
    if out.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::InvalidArgument(format!(
            "duplicate axes in {axes:?}"
        )));
    }
    Ok(out)
}

/// `out` with the (size-1) `axes` removed, unless `keepdims`.
fn finish_reduction(out: Array, axes: &[usize], keepdims: bool) -> Result<Array> {
    if keepdims {
        return Ok(out);
    }
    let shape = out
        .shape()
        .iter()
        .enumerate()
        .filter(|(d, _)| !axes.contains(d))
        .map(|(_, &n)| n)
        .collect();
    reshape_to(&out, shape)
}

fn reduction(
    op: ReduceOp,
    a: &Array,
    axes: &[isize],
    keepdims: bool,
    dtype: DType,
) -> Result<Array> {
    check_inputs(&[a])?;
    let axes = normalize_axes(a, axes)?;
    if matches!(op, ReduceOp::Max | ReduceOp::Min) {
        if a.dtype() == DType::Complex64 {
            return Err(Error::UnsupportedDType {
                op: op.name(),
                dtype: a.dtype(),
            });
        }
        if axes.iter().any(|&d| a.shape()[d] == 0) {
            return Err(Error::InvalidArgument(format!(
                "{}: cannot reduce a zero-length axis (no identity)",
                op.name()
            )));
        }
    }
    let a = astype(a, dtype)?;
    if axes.is_empty() {
        return Ok(a); // a 0-D array: nothing to reduce
    }
    let out = single(
        Primitive::Reduce {
            op,
            axes: axes.clone(),
        },
        vec![a],
    )?;
    finish_reduction(out, &axes, keepdims)
}

/// The type sums and products accumulate into, as in MLX: small integers widen.
fn widened(dtype: DType) -> DType {
    match (dtype.category(), dtype.size()) {
        (Category::Bool, _) => DType::I32,
        (Category::Signed, 1..=4) => DType::I32,
        (Category::Signed, _) => DType::I64,
        (Category::Unsigned, 1..=4) => DType::U32,
        (Category::Unsigned, _) => DType::U64,
        _ => dtype,
    }
}

/// Sum over `axes` (empty = all). Integers and bools widen to 32 or 64 bits (as in MLX);
/// half and single precision accumulate in double precision.
pub fn sum(a: &Array, axes: &[isize], keepdims: bool) -> Result<Array> {
    reduction(ReduceOp::Sum, a, axes, keepdims, widened(a.dtype()))
}

/// Product over `axes` (empty = all), with the same widening as [`sum`].
pub fn prod(a: &Array, axes: &[isize], keepdims: bool) -> Result<Array> {
    reduction(ReduceOp::Prod, a, axes, keepdims, widened(a.dtype()))
}

/// Maximum over `axes` (empty = all); NaN propagates. Zero-length axes are an error.
pub fn max(a: &Array, axes: &[isize], keepdims: bool) -> Result<Array> {
    reduction(ReduceOp::Max, a, axes, keepdims, a.dtype())
}

/// Minimum over `axes` (empty = all); NaN propagates. Zero-length axes are an error.
pub fn min(a: &Array, axes: &[isize], keepdims: bool) -> Result<Array> {
    reduction(ReduceOp::Min, a, axes, keepdims, a.dtype())
}

/// Whether every element over `axes` (empty = all) is non-zero.
pub fn all(a: &Array, axes: &[isize], keepdims: bool) -> Result<Array> {
    reduction(ReduceOp::And, a, axes, keepdims, DType::Bool)
}

/// Whether any element over `axes` (empty = all) is non-zero.
pub fn any(a: &Array, axes: &[isize], keepdims: bool) -> Result<Array> {
    reduction(ReduceOp::Or, a, axes, keepdims, DType::Bool)
}

/// Mean over `axes` (empty = all), in at least float32. Empty reductions are NaN.
pub fn mean(a: &Array, axes: &[isize], keepdims: bool) -> Result<Array> {
    let dtype = at_least_float(a.dtype());
    let count: usize = normalize_axes(a, axes)?
        .iter()
        .map(|&d| a.shape()[d])
        .product();
    let scale = astype(&Array::scalar(1.0 / count as f64), dtype)?;
    multiply(&sum(a, axes, keepdims)?, &scale)
}

fn arg_reduction(op: ArgReduceOp, a: &Array, axis: Option<isize>, keepdims: bool) -> Result<Array> {
    check_inputs(&[a])?;
    if a.dtype() == DType::Complex64 {
        return Err(Error::UnsupportedDType {
            op: op.name(),
            dtype: a.dtype(),
        });
    }
    let Some(axis) = axis else {
        // Over the flattened array; with keepdims the result has one size-1 dimension per
        // input dimension.
        let flat = arg_reduction(op, &reshape(a, &[-1])?, Some(0), false)?;
        return if keepdims {
            reshape_to(&flat, smallvec::smallvec![1; a.ndim()])
        } else {
            Ok(flat)
        };
    };
    let axis = view::normalize_axis(axis, a.ndim())?;
    if a.shape()[axis] == 0 {
        return Err(Error::InvalidArgument(format!(
            "{}: cannot reduce a zero-length axis",
            op.name()
        )));
    }
    let out = single(Primitive::ArgReduce { op, axis }, vec![a.clone()])?;
    finish_reduction(out, &[axis], keepdims)
}

/// Index (uint32) of the maximum along `axis`, or over the flattened array if `None`.
/// The first occurrence wins; the first NaN wins over any number.
pub fn argmax(a: &Array, axis: Option<isize>, keepdims: bool) -> Result<Array> {
    arg_reduction(ArgReduceOp::Max, a, axis, keepdims)
}

/// Index (uint32) of the minimum along `axis`, or over the flattened array if `None`.
/// The first occurrence wins; the first NaN wins over any number.
pub fn argmin(a: &Array, axis: Option<isize>, keepdims: bool) -> Result<Array> {
    arg_reduction(ArgReduceOp::Min, a, axis, keepdims)
}

// ── Matrix multiplication ───────────────────────────────────────────────────

/// Matrix product of `[..., M, K]` and `[..., K, N]`, broadcasting batch dimensions.
///
/// A 1-D `a` is a row vector and a 1-D `b` a column vector; the inserted dimension is
/// removed from the result. The promoted type must be floating point or complex.
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
    check_inputs(&[a, b])?;
    if a.ndim() == 0 || b.ndim() == 0 {
        return Err(Error::InvalidArgument(
            "matmul: operands need at least one dimension".into(),
        ));
    }
    let dtype = promote_types(a.dtype(), b.dtype());
    if !matches!(dtype.category(), Category::Float | Category::Complex) {
        return Err(Error::UnsupportedDType {
            op: "matmul",
            dtype,
        });
    }
    let a2 = if a.ndim() == 1 {
        expand_dims(a, 0)?
    } else {
        a.clone()
    };
    let b2 = if b.ndim() == 1 {
        expand_dims(b, -1)?
    } else {
        b.clone()
    };
    let (sa, sb) = (a2.shape(), b2.shape());
    if sa[sa.len() - 1] != sb[sb.len() - 2] {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        });
    }
    let (a2, b2) = (astype(&a2, dtype)?, astype(&b2, dtype)?);
    let out = if a2.ndim() > 2 && b2.ndim() == 2 {
        // One large product instead of a batch of small ones (as in MLX).
        let k = a2.shape()[a2.ndim() - 1];
        let rows = a2
            .size()
            .checked_div(k)
            .unwrap_or_else(|| a2.shape()[..a2.ndim() - 1].iter().product());
        let flat = single(
            Primitive::Matmul,
            vec![reshape_to(&a2, smallvec::smallvec![rows, k])?, b2.clone()],
        )?;
        let mut shape: Shape = a2.shape().into();
        *shape.last_mut().expect("ndim > 2") = b2.shape()[1];
        reshape_to(&flat, shape)?
    } else {
        let (na, nb) = (a2.ndim(), b2.ndim());
        let batch = view::broadcast_shapes("matmul", &a2.shape()[..na - 2], &b2.shape()[..nb - 2])?;
        let full = |x: &Array, nd: usize| -> Result<Array> {
            let mut shape = batch.clone();
            shape.extend_from_slice(&x.shape()[nd - 2..]);
            broadcast_to(x, &shape)
        };
        single(Primitive::Matmul, vec![full(&a2, na)?, full(&b2, nb)?])?
    };
    // Remove the dimensions inserted for 1-D operands.
    let mut drop = Vec::new();
    if a.ndim() == 1 {
        drop.push(out.ndim() as isize - 2);
    }
    if b.ndim() == 1 {
        drop.push(-1);
    }
    if drop.is_empty() {
        Ok(out)
    } else {
        squeeze(&out, &drop)
    }
}

// ── Creation ────────────────────────────────────────────────────────────────

/// An array of zeros.
pub fn zeros(shape: &[usize], dtype: DType) -> Result<Array> {
    let buffer = Buffer::try_zeroed(shape::checked_nbytes(shape, dtype.size())?)?;
    Array::from_buffer(buffer, dtype, shape)
}

/// An array of ones.
pub fn ones(shape: &[usize], dtype: DType) -> Result<Array> {
    filled(shape, dtype, |_| Canon::Int(1))
}

/// Evenly spaced values `start, start + step, …` below `stop` (above it for negative
/// steps), converted to `dtype`. The length is `ceil((stop - start) / step)`, as in MLX.
pub fn arange(start: f64, stop: f64, step: f64, dtype: DType) -> Result<Array> {
    let invalid = |msg: &str| Err(Error::InvalidArgument(format!("arange: {msg}")));
    if dtype == DType::Bool {
        return Err(Error::UnsupportedDType {
            op: "arange",
            dtype,
        });
    }
    if start.is_nan() || stop.is_nan() || step.is_nan() || start.is_infinite() || stop.is_infinite()
    {
        return invalid("cannot compute the length from NaN or infinite bounds");
    }
    if step == 0.0 {
        return invalid("step must be non-zero");
    }
    let len = if step.is_infinite() {
        usize::from((step > 0.0 && start < stop) || (step < 0.0 && start > stop))
    } else {
        let real = ((stop - start) / step).ceil();
        if real > i32::MAX as f64 {
            return invalid("maximum size exceeded");
        }
        real.max(0.0) as usize
    };
    // `i * step` would be NaN for i = 0 and an infinite step.
    filled(&[len], dtype, |i| {
        Canon::Float(if i == 0 {
            start
        } else {
            start + i as f64 * step
        })
    })
}

/// An array whose element `i` (row-major) is `value(i)` converted to `dtype`.
fn filled(shape: &[usize], dtype: DType, value: impl Fn(usize) -> Canon) -> Result<Array> {
    let mut buffer = Buffer::try_zeroed(shape::checked_nbytes(shape, dtype.size())?)?;
    with_type!(dtype, T => {
        for (i, x) in buffer.typed_mut::<T>().iter_mut().enumerate() {
            *x = T::from_canon(value(i));
        }
    });
    Array::from_buffer(buffer, dtype, shape)
}

// ── Methods ─────────────────────────────────────────────────────────────────

impl Array {
    /// Converts the elements to `dtype`; see [`astype`].
    pub fn astype(&self, dtype: DType) -> Result<Array> {
        astype(self, dtype)
    }

    /// The same elements in another shape; see [`reshape`].
    pub fn reshape(&self, shape: &[isize]) -> Result<Array> {
        reshape(self, shape)
    }

    /// The array with all axes reversed (the matrix transpose for 2-D arrays).
    pub fn t(&self) -> Result<Array> {
        let axes: Vec<usize> = (0..self.ndim()).rev().collect();
        transpose(self, &axes)
    }
}

// ── Operators ───────────────────────────────────────────────────────────────

/// Operators cannot return a `Result`. An invalid expression yields an array whose
/// evaluation (or readback) returns the error, instead of panicking.
fn or_failed(result: Result<Array>, lhs: &Array) -> Array {
    result.unwrap_or_else(|err| Array::failed(err, lhs.dtype(), lhs.shape()))
}

macro_rules! operator {
    ($trait:ident, $method:ident, $op:ident) => {
        impl $trait<&Array> for &Array {
            type Output = Array;
            fn $method(self, rhs: &Array) -> Array {
                or_failed($op(self, rhs), self)
            }
        }
        impl $trait<Array> for &Array {
            type Output = Array;
            fn $method(self, rhs: Array) -> Array {
                or_failed($op(self, &rhs), self)
            }
        }
        impl $trait<&Array> for Array {
            type Output = Array;
            fn $method(self, rhs: &Array) -> Array {
                or_failed($op(&self, rhs), &self)
            }
        }
        impl $trait<Array> for Array {
            type Output = Array;
            fn $method(self, rhs: Array) -> Array {
                or_failed($op(&self, &rhs), &self)
            }
        }
    };
}

operator!(Add, add, add);
operator!(Sub, sub, subtract);
operator!(Mul, mul, multiply);
operator!(Div, div, divide);

impl Neg for &Array {
    type Output = Array;
    fn neg(self) -> Array {
        or_failed(negative(self), self)
    }
}

impl Neg for Array {
    type Output = Array;
    fn neg(self) -> Array {
        or_failed(negative(&self), &self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::live;

    /// Live buffers created while running `f` (after evaluating its result).
    fn allocations(f: impl FnOnce() -> Array) -> isize {
        let before = live::count();
        let out = f();
        out.eval().unwrap();
        live::count() - before
    }

    #[test]
    fn views_share_data() {
        let base = arange(0.0, 24.0, 1.0, DType::F32)
            .unwrap()
            .reshape(&[2, 3, 4])
            .unwrap();
        base.eval().unwrap();
        assert_eq!(allocations(|| base.reshape(&[6, 4]).unwrap()), 0);
        assert_eq!(allocations(|| base.t().unwrap()), 0);
        assert_eq!(
            allocations(|| slice(
                &base,
                &[(..).into(), SliceArg::new(None, None, -2), (1..3).into()]
            )
            .unwrap()),
            0
        );
        assert_eq!(
            allocations(|| broadcast_to(&base, &[5, 2, 3, 4]).unwrap()),
            0
        );
        assert_eq!(allocations(|| expand_dims(&base, 1).unwrap()), 0);
        assert_eq!(
            allocations(|| squeeze(&expand_dims(&base, -1).unwrap(), &[]).unwrap()),
            0
        );
        // A transposed layout cannot be flattened without copying.
        assert_eq!(allocations(|| base.t().unwrap().reshape(&[-1]).unwrap()), 1);
    }

    #[test]
    fn noop_views_record_nothing() {
        let a = Array::from_slice(&[1u8, 2, 3, 4], &[2, 2]).unwrap();
        assert!(a.reshape(&[2, 2]).unwrap().primitive().is_none());
        assert!(transpose(&a, &[0, 1]).unwrap().primitive().is_none());
        assert!(
            slice(&a, &[SliceArg::full(), SliceArg::full()])
                .unwrap()
                .primitive()
                .is_none()
        );
        assert!(astype(&a, DType::U8).unwrap().primitive().is_none());
        assert!(broadcast_to(&a, &[2, 2]).unwrap().primitive().is_none());
    }

    #[test]
    fn view_flags() {
        let a = zeros(&[2, 3], DType::F32).unwrap();
        let t = a.t().unwrap();
        assert!(t.flags().col_contiguous && !t.flags().row_contiguous);
        assert_eq!(t.strides(), &[1, 3]);
        let b = broadcast_to(&a, &[4, 2, 3]).unwrap();
        assert_eq!(b.strides(), &[0, 3, 1]);
        assert!(!b.flags().contiguous);
    }
}
