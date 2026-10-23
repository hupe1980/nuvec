//! Array operations as free functions with MLX's names: the mirrors of the [`Array`]
//! methods.
//!
//! Every operation whose first argument is an array is a method, and the method is the
//! canonical, documented form: `x.sum(-1)`, `x.matmul(&w.t())`, `m.select(&a, 0.0)`. This
//! module holds the same operations as free functions with MLX's names and the same
//! parameters (the receiver becomes the first argument), so MLX code ports line by line:
//! `sum(&x, -1)`, `matmul(&x, &w.t())`, `where_(&m, &a, 0.0)`. A few operations exist only
//! here because their first argument is not one array: [`concatenate`], [`stack`],
//! [`where_`] and the constructors ([`zeros`], [`ones`], [`full`], [`arange`]).
//!
//! Operations return [`Array`], never `Result`: invalid arguments give a *failed array*
//! carrying [`crate::Error::Op`] with the operation's name and the caller's source location,
//! and every array computed from it fails with the same error, surfacing at
//! [`Array::check`], [`Array::eval`], [`Array::item`] or [`Array::to_vec`].
//!
//! ```
//! use nuvec_core::{Array, array, ops};
//!
//! let x = array![[1.0, 2.0], [3.0, 4.0]];
//! let a = x.exp().sum(-1);
//! let b = ops::sum(&ops::exp(&x), -1);
//! assert_eq!(a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
//! ```

// The implementation functions behind the public operations (they return `Result`).
pub(crate) mod imp;

use std::panic::Location;

use smallvec::SmallVec;

use crate::axes::{self, Axes, AxesSpec, Axis, IntoShape, ScanOptions, VarOptions};
use crate::primitive::BinaryOp;
use crate::scalar::{Scalar, ScalarOrArray};
use crate::{Array, Category, DType, Error, Result, at_least_float};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Axes in the `isize` form of the implementation functions.
fn isize_axes(axes: &[usize]) -> Vec<isize> {
    axes.iter().map(|&a| a as isize).collect()
}

/// A weak operand as an array next to `like` (its dtype adapts).
fn operand(value: ScalarOrArray, like: DType) -> Result<Array> {
    value.into_array(Some(like))
}

/// The arrays among two operands, for failure reporting.
fn arrays_of<'a>(a: &'a ScalarOrArray, b: &'a ScalarOrArray) -> SmallVec<[&'a Array; 2]> {
    [a, b]
        .into_iter()
        .filter_map(|x| match x {
            ScalarOrArray::Array(a) => Some(a),
            ScalarOrArray::Scalar(_) => None,
        })
        .collect()
}

/// Both operands of a binary operation as arrays (MLX's `to_arrays`): a weak scalar adapts
/// to the other operand's dtype; two scalars take their default dtypes.
fn resolve_pair(a: &ScalarOrArray, b: &ScalarOrArray) -> Result<(Array, Array)> {
    let like = |x: &ScalarOrArray| match x {
        ScalarOrArray::Array(a) => Some(a.dtype()),
        ScalarOrArray::Scalar(_) => None,
    };
    let (la, lb) = (like(a), like(b));
    let conv = |x: &ScalarOrArray, other: Option<DType>| match x {
        ScalarOrArray::Array(a) => Ok(a.clone()),
        ScalarOrArray::Scalar(s) => s.to_array(other),
    };
    Ok((conv(a, lb)?, conv(b, la)?))
}

#[inline]
fn binary_op(
    name: &'static str,
    at: &'static Location<'static>,
    op: BinaryOp,
    a: ScalarOrArray,
    b: ScalarOrArray,
) -> Array {
    match (a, b) {
        (ScalarOrArray::Array(a), ScalarOrArray::Array(b)) => {
            Array::sticky(name, at, &[&a, &b], || imp::binary(op, &a, &b))
        }
        (a, b) => {
            let ins = arrays_of(&a, &b);
            Array::sticky(name, at, &ins, || {
                let (x, y) = resolve_pair(&a, &b)?;
                imp::binary(op, &x, &y)
            })
        }
    }
}

// ── Element-wise binary ─────────────────────────────────────────────────────

macro_rules! binary_ops {
    ($($name:ident => $op:ident),* $(,)?) => {$(
        #[doc = concat!("Element-wise `", stringify!($name), "`; see [`Array::", stringify!($name), "`]. Either operand may be a weak Rust number.")]
        #[track_caller]
        pub fn $name(a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> Array {
            binary_op(stringify!($name), Location::caller(), BinaryOp::$op, a.into(), b.into())
        }
    )*};
}

binary_ops! {
    add => Add,
    subtract => Subtract,
    multiply => Multiply,
    divide => Divide,
    maximum => Maximum,
    minimum => Minimum,
    logaddexp => LogAddExp,
    equal => Equal,
    not_equal => NotEqual,
    less => Less,
    less_equal => LessEqual,
    greater => Greater,
    greater_equal => GreaterEqual,
}

// ── Element-wise unary ──────────────────────────────────────────────────────

macro_rules! unary_ops {
    ($($name:ident),* $(,)?) => {$(
        #[doc = concat!("Element-wise `", stringify!($name), "`; see [`Array::", stringify!($name), "`].")]
        #[track_caller]
        pub fn $name(a: &Array) -> Array {
            Array::sticky(stringify!($name), Location::caller(), &[a], || imp::$name(a))
        }
    )*};
}

unary_ops!(
    negative,
    abs,
    exp,
    log,
    sqrt,
    sin,
    cos,
    tanh,
    sigmoid,
    erfinv,
    sign,
    conjugate,
    stop_gradient,
    contiguous,
    copy
);

/// Element-wise selection: `a` where `condition` is true, else `b` (MLX's `where`); see
/// [`Array::select`]. Any operand may be a weak Rust number.
#[track_caller]
pub fn where_(
    condition: impl Into<ScalarOrArray>,
    a: impl Into<ScalarOrArray>,
    b: impl Into<ScalarOrArray>,
) -> Array {
    let (condition, a, b) = (condition.into(), a.into(), b.into());
    let mut ins: SmallVec<[&Array; 3]> = SmallVec::new();
    for x in [&condition, &a, &b] {
        if let ScalarOrArray::Array(x) = x {
            ins.push(x);
        }
    }
    Array::sticky("where", Location::caller(), &ins, || {
        let (a, b) = resolve_pair(&a, &b)?;
        let condition = condition.clone().into_array(None)?;
        imp::where_(condition, a, b)
    })
}

/// The elements converted to `dtype`; see [`Array::astype`].
#[track_caller]
pub fn astype(a: &Array, dtype: DType) -> Array {
    Array::sticky("astype", Location::caller(), &[a], || imp::astype(a, dtype))
}

/// Matrix product; see [`Array::matmul`].
#[track_caller]
pub fn matmul(a: &Array, b: &Array) -> Array {
    Array::sticky("matmul", Location::caller(), &[a, b], || imp::matmul(a, b))
}

/// `alpha · (a @ b) + beta · c`; see [`Array::addmm`].
#[track_caller]
pub fn addmm(c: &Array, a: &Array, b: &Array, alpha: f32, beta: f32) -> Array {
    Array::sticky("addmm", Location::caller(), &[c, a, b], || {
        imp::addmm(c, a, b, alpha, beta)
    })
}

// ── Reductions ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Reduce {
    Sum,
    Prod,
    Max,
    Min,
    All,
    Any,
    Mean,
    LogSumExp,
    Var(i32),
    Std(i32),
}

impl Reduce {
    fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Max => "max",
            Self::Min => "min",
            Self::All => "all",
            Self::Any => "any",
            Self::Mean => "mean",
            Self::LogSumExp => "logsumexp",
            Self::Var(_) => "var",
            Self::Std(_) => "std",
        }
    }
}

/// The body of every reduction (non-generic: the public functions only convert their
/// arguments).
fn reduce_op(
    at: &'static Location<'static>,
    a: &Array,
    axes: AxesSpec,
    keepdims: bool,
    op: Reduce,
) -> Array {
    Array::sticky(op.name(), at, &[a], || reduce(a, &axes, keepdims, op))
}

fn reduce(a: &Array, axes: &AxesSpec, keepdims: bool, op: Reduce) -> Result<Array> {
    let axes = axes.sorted(a.ndim())?;
    let list = isize_axes(&axes);
    if axes.is_empty() {
        // No axis is reduced (MLX C++'s `sum(a, {})`).
        return match op {
            Reduce::Sum | Reduce::Prod | Reduce::Max | Reduce::Min => Ok(a.clone()),
            Reduce::All | Reduce::Any => imp::astype(a, DType::Bool),
            Reduce::Mean => imp::astype(a, at_least_float(a.dtype())),
            Reduce::LogSumExp => logsumexp_none(a),
            Reduce::Var(ddof) => variance(a, &axes, keepdims, ddof),
            Reduce::Std(ddof) => imp::sqrt(&variance(a, &axes, keepdims, ddof)?),
        };
    }
    match op {
        Reduce::Sum => imp::sum(a, &list, keepdims),
        Reduce::Prod => imp::prod(a, &list, keepdims),
        Reduce::Max => imp::max(a, &list, keepdims),
        Reduce::Min => imp::min(a, &list, keepdims),
        Reduce::All => imp::all(a, &list, keepdims),
        Reduce::Any => imp::any(a, &list, keepdims),
        Reduce::Mean => imp::mean(a, &list, keepdims),
        Reduce::LogSumExp => imp::logsumexp(a, &list, keepdims),
        Reduce::Var(ddof) => variance(a, &axes, keepdims, ddof),
        Reduce::Std(ddof) => imp::sqrt(&variance(a, &axes, keepdims, ddof)?),
    }
}

/// `log(exp(a))` element-wise, as MLX's `logsumexp` over no axes: `a`, with the same
/// handling of infinities.
fn logsumexp_none(a: &Array) -> Result<Array> {
    let a = imp::astype(a, at_least_float(a.dtype()))?;
    let maxval = imp::stop_gradient(&a)?;
    let out = imp::log(&imp::exp(&imp::subtract(&a, &maxval)?)?)?;
    let out = imp::add(&out, &maxval)?;
    imp::where_(
        imp::equal(&imp::abs(&maxval)?, f64::INFINITY)?,
        &maxval,
        &out,
    )
}

/// The mean over `axes` (sorted; empty = none) of a floating-point `a`.
fn mean_over(a: &Array, axes: &[usize], keepdims: bool) -> Result<Array> {
    if axes.is_empty() {
        return Ok(a.clone());
    }
    imp::mean(a, &isize_axes(axes), keepdims)
}

/// The variance over `axes`, in at least float32 (the input is converted first, so integers
/// never wrap): `mean((a − mean(a))²)`, scaled by `n / max(n − ddof, 0)` when `ddof` is not 0.
fn variance(a: &Array, axes: &[usize], keepdims: bool, ddof: i32) -> Result<Array> {
    let dtype = at_least_float(a.dtype());
    let a = imp::astype(a, dtype)?;
    // Two passes, `mean((a − mean(a))²)`: no cancellation for large means.
    let mu = if axes.is_empty() {
        a.clone()
    } else {
        imp::mean(&a, &isize_axes(axes), true)?
    };
    let d = imp::subtract(&a, &mu)?;
    let v = mean_over(&imp::multiply(&d, &d)?, axes, keepdims)?;
    if ddof == 0 {
        return Ok(v);
    }
    let n: f64 = axes.iter().map(|&d| a.dims()[d] as f64).product();
    let n = imp::astype(&crate::creation::from_data(&[n], &[])?, dtype)?;
    let rest = imp::maximum(&imp::subtract(&n, f64::from(ddof))?, 0.0)?;
    imp::multiply(&v, &imp::divide(&n, &rest)?)
}

macro_rules! reductions {
    ($($name:ident, $keep:ident => $op:expr, $doc:literal;)*) => {$(
        #[doc = concat!($doc, " over `axes`; see [`Array::", stringify!($name), "`].")]
        #[track_caller]
        pub fn $name(a: &Array, axes: impl Axes) -> Array {
            reduce_op(Location::caller(), a, axes.axes_spec(), false, $op)
        }
        #[doc = concat!($doc, " over `axes`, keeping them as size-1 dimensions; see [`Array::", stringify!($keep), "`].")]
        #[track_caller]
        pub fn $keep(a: &Array, axes: impl Axes) -> Array {
            reduce_op(Location::caller(), a, axes.axes_spec(), true, $op)
        }
    )*};
}

reductions! {
    sum, sum_keepdims => Reduce::Sum, "The sum";
    prod, prod_keepdims => Reduce::Prod, "The product";
    max, max_keepdims => Reduce::Max, "The maximum";
    min, min_keepdims => Reduce::Min, "The minimum";
    all, all_keepdims => Reduce::All, "Whether every element is non-zero";
    any, any_keepdims => Reduce::Any, "Whether any element is non-zero";
    mean, mean_keepdims => Reduce::Mean, "The mean";
    logsumexp, logsumexp_keepdims => Reduce::LogSumExp, "`log(sum(exp(a)))`, computed stably,";
    var, var_keepdims => Reduce::Var(0), "The variance";
    std, std_keepdims => Reduce::Std(0), "The standard deviation";
}

/// The variance over `axes` with [`VarOptions`]; see [`Array::var_with`].
#[track_caller]
pub fn var_with(a: &Array, axes: impl Axes, options: VarOptions) -> Array {
    reduce_op(
        Location::caller(),
        a,
        axes.axes_spec(),
        options.keepdims,
        Reduce::Var(options.ddof),
    )
}

/// The standard deviation over `axes` with [`VarOptions`]; see [`Array::std_with`].
#[track_caller]
pub fn std_with(a: &Array, axes: impl Axes, options: VarOptions) -> Array {
    reduce_op(
        Location::caller(),
        a,
        axes.axes_spec(),
        options.keepdims,
        Reduce::Std(options.ddof),
    )
}

fn arg_reduce_op(
    at: &'static Location<'static>,
    a: &Array,
    axis: Option<i64>,
    keepdims: bool,
    max: bool,
) -> Array {
    let name = if max { "argmax" } else { "argmin" };
    Array::sticky(name, at, &[a], || {
        let axis = axes::one_axis(axis, a.ndim())?.map(|d| d as isize);
        if max {
            imp::argmax(a, axis, keepdims)
        } else {
            imp::argmin(a, axis, keepdims)
        }
    })
}

/// Indices (uint32) of the maxima along one axis (`..`: the flattened array); see
/// [`Array::argmax`].
#[track_caller]
pub fn argmax(a: &Array, axis: impl Axis) -> Array {
    arg_reduce_op(Location::caller(), a, axis.axis_spec(), false, true)
}

/// [`argmax`] keeping the reduced axis; see [`Array::argmax_keepdims`].
#[track_caller]
pub fn argmax_keepdims(a: &Array, axis: impl Axis) -> Array {
    arg_reduce_op(Location::caller(), a, axis.axis_spec(), true, true)
}

/// Indices (uint32) of the minima along one axis (`..`: the flattened array); see
/// [`Array::argmin`].
#[track_caller]
pub fn argmin(a: &Array, axis: impl Axis) -> Array {
    arg_reduce_op(Location::caller(), a, axis.axis_spec(), false, false)
}

/// [`argmin`] keeping the reduced axis; see [`Array::argmin_keepdims`].
#[track_caller]
pub fn argmin_keepdims(a: &Array, axis: impl Axis) -> Array {
    arg_reduce_op(Location::caller(), a, axis.axis_spec(), true, false)
}

fn softmax_op(at: &'static Location<'static>, a: &Array, spec: AxesSpec) -> Array {
    Array::sticky("softmax", at, &[a], || {
        let axes = spec.sorted(a.ndim())?;
        if axes.is_empty() {
            // Over no axes: every element is its own group (1, or NaN where `a` is not
            // finite), as MLX.
            let a = imp::astype(a, at_least_float(a.dtype()))?;
            let ex = imp::exp(&imp::subtract(&a, &imp::stop_gradient(&a)?)?)?;
            return imp::divide(&ex, &ex);
        }
        imp::softmax(a, &isize_axes(&axes))
    })
}

/// `exp(a) / sum(exp(a))` over `axes`; see [`Array::softmax`].
#[track_caller]
pub fn softmax(a: &Array, axes: impl Axes) -> Array {
    softmax_op(Location::caller(), a, axes.axes_spec())
}

// ── Scans and sorting ───────────────────────────────────────────────────────

type ScanImpl = fn(&Array, Option<isize>, bool, bool) -> Result<Array>;

fn scan_op(
    name: &'static str,
    at: &'static Location<'static>,
    a: &Array,
    axis: Option<i64>,
    options: ScanOptions,
    f: ScanImpl,
) -> Array {
    Array::sticky(name, at, &[a], || {
        let axis = axes::one_axis(axis, a.ndim())?.map(|d| d as isize);
        f(a, axis, options.reverse, !options.exclusive)
    })
}

macro_rules! scans {
    ($($name:ident, $with:ident => $imp:path, $doc:literal;)*) => {$(
        #[doc = concat!($doc, " along one axis (`..`: the flattened array); see [`Array::", stringify!($name), "`].")]
        #[track_caller]
        pub fn $name(a: &Array, axis: impl Axis) -> Array {
            scan_op(stringify!($name), Location::caller(), a, axis.axis_spec(), ScanOptions::default(), $imp)
        }
        #[doc = concat!($doc, " with [`ScanOptions`]; see [`Array::", stringify!($with), "`].")]
        #[track_caller]
        pub fn $with(a: &Array, axis: impl Axis, options: ScanOptions) -> Array {
            scan_op(stringify!($name), Location::caller(), a, axis.axis_spec(), options, $imp)
        }
    )*};
}

scans! {
    cumsum, cumsum_with => crate::sorting::cumsum, "The cumulative sum";
    cumprod, cumprod_with => crate::sorting::cumprod, "The cumulative product";
    cummax, cummax_with => crate::sorting::cummax, "The running maximum";
    cummin, cummin_with => crate::sorting::cummin, "The running minimum";
    logcumsumexp, logcumsumexp_with => crate::sorting::logcumsumexp, "The running `log(Σ exp)`";
}

#[derive(Clone, Copy)]
enum Sorting {
    Sort,
    ArgSort,
    Partition(i32),
    ArgPartition(i32),
    TopK(i32),
}

fn sorting_op(at: &'static Location<'static>, a: &Array, axis: Option<i64>, op: Sorting) -> Array {
    let name = match op {
        Sorting::Sort => "sort",
        Sorting::ArgSort => "argsort",
        Sorting::Partition(_) => "partition",
        Sorting::ArgPartition(_) => "argpartition",
        Sorting::TopK(_) => "topk",
    };
    Array::sticky(name, at, &[a], || {
        let axis = axes::one_axis(axis, a.ndim())?.map(|d| d as isize);
        match op {
            Sorting::Sort => crate::sorting::sort(a, axis),
            Sorting::ArgSort => crate::sorting::argsort(a, axis),
            Sorting::Partition(kth) => crate::sorting::partition(a, kth as isize, axis),
            Sorting::ArgPartition(kth) => crate::sorting::argpartition(a, kth as isize, axis),
            Sorting::TopK(k) => {
                let k = usize::try_from(k).map_err(|_| {
                    Error::InvalidArgument(format!("topk: k must not be negative, got {k}"))
                })?;
                crate::sorting::topk(a, k, axis)
            }
        }
    })
}

/// The array sorted along one axis; see [`Array::sort`].
#[track_caller]
pub fn sort(a: &Array, axis: impl Axis) -> Array {
    sorting_op(Location::caller(), a, axis.axis_spec(), Sorting::Sort)
}

/// The indices (uint32) that sort the array along one axis; see [`Array::argsort`].
#[track_caller]
pub fn argsort(a: &Array, axis: impl Axis) -> Array {
    sorting_op(Location::caller(), a, axis.axis_spec(), Sorting::ArgSort)
}

/// The array partitioned around position `kth` along one axis; see [`Array::partition`].
#[track_caller]
pub fn partition(a: &Array, kth: i32, axis: impl Axis) -> Array {
    sorting_op(
        Location::caller(),
        a,
        axis.axis_spec(),
        Sorting::Partition(kth),
    )
}

/// The indices (uint32) that partition the array; see [`Array::argpartition`].
#[track_caller]
pub fn argpartition(a: &Array, kth: i32, axis: impl Axis) -> Array {
    sorting_op(
        Location::caller(),
        a,
        axis.axis_spec(),
        Sorting::ArgPartition(kth),
    )
}

/// The `k` largest elements along one axis; see [`Array::topk`].
#[track_caller]
pub fn topk(a: &Array, k: i32, axis: impl Axis) -> Array {
    sorting_op(Location::caller(), a, axis.axis_spec(), Sorting::TopK(k))
}

// ── Shapes and views ────────────────────────────────────────────────────────

fn reshape_op(at: &'static Location<'static>, a: &Array, spec: &[i64]) -> Array {
    Array::sticky("reshape", at, &[a], || {
        imp::reshape_to(a, axes::reshape_dims(spec, a.size())?)
    })
}

/// The same elements in another shape (one `-1` inferred); see [`Array::reshape`].
#[track_caller]
pub fn reshape(a: &Array, shape: impl IntoShape) -> Array {
    reshape_op(Location::caller(), a, &shape.shape_spec())
}

fn transpose_op(at: &'static Location<'static>, a: &Array, spec: AxesSpec) -> Array {
    Array::sticky("transpose", at, &[a], || {
        let perm: SmallVec<[usize; 4]> = match spec {
            AxesSpec::All => (0..a.ndim()).rev().collect(),
            _ => spec.list(a.ndim())?,
        };
        if perm.len() != a.ndim() {
            return Err(Error::InvalidArgument(format!(
                "transpose: {} axes for an array of {} dimensions",
                perm.len(),
                a.ndim()
            )));
        }
        imp::transpose(a, &perm)
    })
}

/// The axes reordered (`..`: reversed); see [`Array::transpose`].
#[track_caller]
pub fn transpose(a: &Array, axes: impl Axes) -> Array {
    transpose_op(Location::caller(), a, axes.axes_spec())
}

fn broadcast_op(at: &'static Location<'static>, a: &Array, spec: &[i64]) -> Array {
    Array::sticky("broadcast_to", at, &[a], || {
        imp::broadcast_to(a, &axes::dims(spec)?)
    })
}

/// The array broadcast to `shape`; see [`Array::broadcast_to`].
#[track_caller]
pub fn broadcast_to(a: &Array, shape: impl IntoShape) -> Array {
    broadcast_op(Location::caller(), a, &shape.shape_spec())
}

fn expand_dims_op(at: &'static Location<'static>, a: &Array, spec: AxesSpec) -> Array {
    Array::sticky("expand_dims", at, &[a], || {
        let AxesSpec::List(list) = &spec else {
            return Err(Error::InvalidArgument(
                "expand_dims: the new axes must be listed".into(),
            ));
        };
        let out_ndim = a.ndim() + list.len();
        let positions = spec.sorted(out_ndim)?;
        let mut shape: crate::shape::Shape = a.dims().into();
        for &p in &positions {
            shape.insert(p, 1);
        }
        imp::reshape_to(a, shape)
    })
}

/// Size-1 dimensions inserted at `axes` (positions in the result); see
/// [`Array::expand_dims`].
#[track_caller]
pub fn expand_dims(a: &Array, axes: impl Axes) -> Array {
    expand_dims_op(Location::caller(), a, axes.axes_spec())
}

fn squeeze_op(at: &'static Location<'static>, a: &Array, spec: AxesSpec) -> Array {
    Array::sticky("squeeze", at, &[a], || {
        if spec == AxesSpec::All {
            return imp::squeeze(a, &[]);
        }
        let list = spec.list(a.ndim())?;
        if list.is_empty() {
            return Ok(a.clone());
        }
        imp::squeeze(a, &isize_axes(&list))
    })
}

/// Size-1 dimensions removed (`..`: all of them); see [`Array::squeeze`].
#[track_caller]
pub fn squeeze(a: &Array, axes: impl Axes) -> Array {
    squeeze_op(Location::caller(), a, axes.axes_spec())
}

fn repeat_op(at: &'static Location<'static>, a: &Array, repeats: i32, axis: Option<i64>) -> Array {
    Array::sticky("repeat", at, &[a], || {
        let repeats = usize::try_from(repeats).map_err(|_| {
            Error::InvalidArgument(format!(
                "repeat: repeats must not be negative, got {repeats}"
            ))
        })?;
        imp::repeat(
            a,
            repeats,
            axes::one_axis(axis, a.ndim())?.map(|d| d as isize),
        )
    })
}

/// Each element repeated `repeats` times along one axis; see [`Array::repeat`].
#[track_caller]
pub fn repeat(a: &Array, repeats: i32, axis: impl Axis) -> Array {
    repeat_op(Location::caller(), a, repeats, axis.axis_spec())
}

fn split_op(
    at: &'static Location<'static>,
    a: &Array,
    parts: i32,
    axis: Option<i64>,
) -> Vec<Array> {
    let count = usize::try_from(parts).unwrap_or(0).max(1);
    Array::sticky_vec("split", at, &[a], count, || {
        let parts = usize::try_from(parts).map_err(|_| {
            Error::InvalidArgument(format!("split: parts must not be negative, got {parts}"))
        })?;
        crate::indexing::split(a, parts, axes::given_axis(axis, a.ndim())?)
    })
}

/// The array split into `parts` equal parts along `axis`; see [`Array::split`].
#[track_caller]
pub fn split(a: &Array, parts: i32, axis: impl Axis) -> Vec<Array> {
    split_op(Location::caller(), a, parts, axis.axis_spec())
}

fn split_at_op(
    at: &'static Location<'static>,
    a: &Array,
    indices: &[i32],
    axis: Option<i64>,
) -> Vec<Array> {
    Array::sticky_vec("split", at, &[a], indices.len() + 1, || {
        let indices = indices
            .iter()
            .map(|&i| {
                usize::try_from(i).map_err(|_| {
                    Error::InvalidArgument(format!("split: index {i} must not be negative"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        crate::indexing::split_at(a, &indices, axes::given_axis(axis, a.ndim())?)
    })
}

/// The array split before each of `indices` along `axis`; see [`Array::split_at`].
#[track_caller]
pub fn split_at(a: &Array, indices: &[i32], axis: impl Axis) -> Vec<Array> {
    split_at_op(Location::caller(), a, indices, axis.axis_spec())
}

/// The array split into `N` equal parts along `axis`; see [`Array::split_n`].
#[track_caller]
pub fn split_n<const N: usize>(a: &Array, axis: impl Axis) -> [Array; N] {
    let parts = split_op(
        Location::caller(),
        a,
        i32::try_from(N).unwrap_or(i32::MAX),
        axis.axis_spec(),
    );
    parts
        .try_into()
        .unwrap_or_else(|v: Vec<Array>| std::array::from_fn(|i| v[i % v.len()].clone()))
}

fn join_op(
    name: &'static str,
    at: &'static Location<'static>,
    arrays: &[Array],
    axis: Option<i64>,
) -> Array {
    let ins: SmallVec<[&Array; 4]> = arrays.iter().collect();
    Array::sticky(name, at, &ins, || {
        let ndim = arrays.first().map_or(0, Array::ndim);
        let axis = axes::given_axis(axis, ndim)?;
        if name == "stack" {
            crate::indexing::stack(arrays, axis)
        } else {
            crate::indexing::concatenate(arrays, axis)
        }
    })
}

/// `arrays` joined along `axis` (MLX's `concatenate`); dtypes are promoted.
///
/// ```
/// use nuvec_core::{array, ops::concatenate};
///
/// let c = concatenate(&[array![[1.0, 2.0]], array![[3.0, 4.0]]], 0);
/// assert_eq!(c.shape(), [2, 2]);
/// ```
#[doc(alias = "concat")]
#[track_caller]
pub fn concatenate(arrays: &[Array], axis: impl Axis) -> Array {
    join_op("concatenate", Location::caller(), arrays, axis.axis_spec())
}

/// `arrays` (one shape) stacked along a new axis `axis` (MLX's `stack`).
///
/// ```
/// use nuvec_core::{array, ops::stack};
///
/// assert_eq!(stack(&[array![1, 2], array![3, 4]], -1).shape(), [2, 2]);
/// ```
#[track_caller]
pub fn stack(arrays: &[Array], axis: impl Axis) -> Array {
    join_op("stack", Location::caller(), arrays, axis.axis_spec())
}

// ── Gathers and scatters ────────────────────────────────────────────────────

fn take_op(at: &'static Location<'static>, a: &Array, indices: &Array, axis: Option<i64>) -> Array {
    Array::sticky("take", at, &[a, indices], || {
        crate::indexing::take(
            a,
            indices,
            axes::one_axis(axis, a.ndim())?.map(|d| d as isize),
        )
    })
}

/// The elements at `indices` along one axis (`..`: the flattened array); see
/// [`Array::take`].
#[track_caller]
pub fn take(a: &Array, indices: &Array, axis: impl Axis) -> Array {
    take_op(Location::caller(), a, indices, axis.axis_spec())
}

fn take_along_op(
    at: &'static Location<'static>,
    a: &Array,
    indices: &Array,
    axis: Option<i64>,
) -> Array {
    Array::sticky("take_along_axis", at, &[a, indices], || {
        crate::indexing::take_along_axis(a, indices, axes::given_axis(axis, a.ndim())?)
    })
}

/// The elements at `indices` along `axis`; see [`Array::take_along_axis`].
#[track_caller]
pub fn take_along_axis(a: &Array, indices: &Array, axis: impl Axis) -> Array {
    take_along_op(Location::caller(), a, indices, axis.axis_spec())
}

fn scatter_op(
    at: &'static Location<'static>,
    a: &Array,
    indices: &Array,
    values: ScalarOrArray,
    axis: Option<i64>,
    add: bool,
) -> Array {
    let name = if add {
        "scatter_add_along_axis"
    } else {
        "put_along_axis"
    };
    let mut ins: SmallVec<[&Array; 3]> = SmallVec::from_slice(&[a, indices]);
    if let ScalarOrArray::Array(v) = &values {
        ins.push(v);
    }
    Array::sticky(name, at, &ins, || {
        let values = operand(values.clone(), a.dtype())?;
        let axis = axes::given_axis(axis, a.ndim())?;
        if add {
            crate::indexing::scatter_add_along_axis(a, indices, &values, axis)
        } else {
            crate::indexing::put_along_axis(a, indices, &values, axis)
        }
    })
}

/// A copy with `values` placed at `indices` along `axis`; see [`Array::put_along_axis`].
#[track_caller]
pub fn put_along_axis(
    a: &Array,
    indices: &Array,
    values: impl Into<ScalarOrArray>,
    axis: impl Axis,
) -> Array {
    scatter_op(
        Location::caller(),
        a,
        indices,
        values.into(),
        axis.axis_spec(),
        false,
    )
}

/// A copy with `values` added at `indices` along `axis` (repeated indices accumulate); see
/// [`Array::scatter_add_along_axis`].
#[track_caller]
pub fn scatter_add_along_axis(
    a: &Array,
    indices: &Array,
    values: impl Into<ScalarOrArray>,
    axis: impl Axis,
) -> Array {
    scatter_op(
        Location::caller(),
        a,
        indices,
        values.into(),
        axis.axis_spec(),
        true,
    )
}

/// A copy of `a` with the region at `index` replaced by `update`; see
/// [`Array::slice_update`].
#[track_caller]
pub fn slice_update(a: &Array, update: &Array, index: impl crate::ArrayIndex) -> Array {
    a.slice_update(update, index)
}

// ── Quantization and convolution ────────────────────────────────────────────

fn group_bits(op: &'static str, group_size: i32, bits: i32) -> Result<(usize, u32)> {
    match (usize::try_from(group_size), u32::try_from(bits)) {
        (Ok(g), Ok(b)) => Ok((g, b)),
        _ => Err(Error::InvalidArgument(format!(
            "{op}: group size {group_size} and bits {bits} must not be negative"
        ))),
    }
}

/// `w` quantized along its last axis: `(w_q, scales, biases)`; see [`Array::quantize`].
#[track_caller]
pub fn quantize(w: &Array, group_size: i32, bits: i32) -> (Array, Array, Array) {
    let [q, s, b] = Array::sticky_n("quantize", Location::caller(), &[w], || {
        let (g, n) = group_bits("quantize", group_size, bits)?;
        let (q, s, b) = crate::quantized::quantize(w, g, n)?;
        Ok([q, s, b])
    });
    (q, s, b)
}

/// The weights of [`quantize`] restored; see [`Array::dequantize`].
#[track_caller]
pub fn dequantize(w: &Array, scales: &Array, biases: &Array, group_size: i32, bits: i32) -> Array {
    Array::sticky(
        "dequantize",
        Location::caller(),
        &[w, scales, biases],
        || {
            let (g, n) = group_bits("dequantize", group_size, bits)?;
            crate::quantized::dequantize(w, scales, biases, g, n)
        },
    )
}

/// `x · dequantize(w)ᵀ` (or without the transpose); see [`Array::quantized_matmul`].
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    transpose: bool,
    group_size: i32,
    bits: i32,
) -> Array {
    Array::sticky(
        "quantized_matmul",
        Location::caller(),
        &[x, w, scales, biases],
        || {
            let (g, n) = group_bits("quantized_matmul", group_size, bits)?;
            crate::quantized::quantized_matmul(x, w, scales, biases, transpose, g, n)
        },
    )
}

/// Quantized products of gathered matrices (mixture-of-experts layers); see
/// [`Array::gather_qmm`].
#[track_caller]
#[allow(clippy::too_many_arguments)]
pub fn gather_qmm(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    sorted_indices: bool,
) -> Array {
    let mut ins: SmallVec<[&Array; 6]> = SmallVec::from_slice(&[x, w, scales, biases]);
    ins.extend(lhs_indices);
    ins.extend(rhs_indices);
    Array::sticky("gather_qmm", Location::caller(), &ins, || {
        let (g, n) = group_bits("gather_qmm", group_size, bits)?;
        crate::quantized::gather_qmm(
            x,
            w,
            scales,
            biases,
            lhs_indices,
            rhs_indices,
            transpose,
            g,
            n,
            sorted_indices,
        )
    })
}

/// Products of gathered matrices; see [`Array::gather_mm`].
#[track_caller]
pub fn gather_mm(
    a: &Array,
    b: &Array,
    lhs_indices: Option<&Array>,
    rhs_indices: Option<&Array>,
    sorted_indices: bool,
) -> Array {
    let mut ins: SmallVec<[&Array; 4]> = SmallVec::from_slice(&[a, b]);
    ins.extend(lhs_indices);
    ins.extend(rhs_indices);
    Array::sticky("gather_mm", Location::caller(), &ins, || {
        crate::quantized::gather_mm(a, b, lhs_indices, rhs_indices, sorted_indices)
    })
}

/// A 1-D convolution of `[N, L, C_in]` with `[C_out, K, C_in / groups]`; see
/// [`Array::conv1d`].
#[track_caller]
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Array {
    Array::sticky("conv1d", Location::caller(), &[input, weight], || {
        let p = |name: &str, v: i32| {
            usize::try_from(v).map_err(|_| {
                Error::InvalidArgument(format!("conv1d: {name} must not be negative, got {v}"))
            })
        };
        crate::conv::conv1d(
            input,
            weight,
            p("stride", stride)?,
            p("padding", padding)?,
            p("dilation", dilation)?,
            p("groups", groups)?,
        )
    })
}

// ── Creation ────────────────────────────────────────────────────────────────

/// An array of zeros of `shape`.
///
/// ```
/// use nuvec_core::{DType, ops::zeros};
///
/// assert_eq!(zeros([2, 3], DType::F32).shape(), [2, 3]);
/// assert_eq!(zeros((), DType::I32).ndim(), 0);
/// ```
#[track_caller]
pub fn zeros(shape: impl IntoShape, dtype: DType) -> Array {
    let spec = shape.shape_spec();
    Array::constructor("zeros", Location::caller(), dtype, &spec, |shape| {
        imp::zeros(shape, dtype)
    })
}

/// An array of ones of `shape`.
#[track_caller]
pub fn ones(shape: impl IntoShape, dtype: DType) -> Array {
    let spec = shape.shape_spec();
    Array::constructor("ones", Location::caller(), dtype, &spec, |shape| {
        imp::ones(shape, dtype)
    })
}

/// An array of `shape` with every element `value`: a Rust number is weak, as in MLX (`0.5`
/// gives float32, `1` int32), and an array keeps its dtype (`Array::from(7u8)` gives uint8).
///
/// ```
/// use nuvec_core::{Array, DType, ops::full};
///
/// assert_eq!(full([2, 2], 0.5).dtype(), DType::F32);
/// let a = full([2, 2], Array::from(7u8));
/// assert_eq!((a.dtype(), a.to_vec::<u8>().unwrap()), (DType::U8, vec![7; 4]));
/// ```
#[track_caller]
pub fn full(shape: impl IntoShape, value: impl Into<ScalarOrArray>) -> Array {
    full_op(Location::caller(), &shape.shape_spec(), value.into())
}

fn full_op(at: &'static Location<'static>, spec: &[i64], value: ScalarOrArray) -> Array {
    let dtype = match &value {
        ScalarOrArray::Array(a) => a.dtype(),
        ScalarOrArray::Scalar(s) => s.default_dtype(),
    };
    if let ScalarOrArray::Array(v) = &value
        && let Some(e) = v.slot_error()
    {
        return Array::failed(e, dtype, &axes::dims(spec).unwrap_or_default());
    }
    Array::constructor("full", at, dtype, spec, |shape| match &value {
        // A number: the buffer is filled now, on the host (unified memory), as `zeros`.
        ScalarOrArray::Scalar(s) => imp::filled(shape, dtype, |_| s.canon()),
        ScalarOrArray::Array(v) => imp::contiguous(&imp::broadcast_to(v, shape)?),
    })
}

/// Evenly spaced values `start, start + step, …` below `stop` (above it for a negative
/// step), as MLX: int32 when every argument is an integer (int64 when they do not fit),
/// float32 when any is a float. Use [`Array::astype`] for another dtype.
///
/// ```
/// use nuvec_core::{DType, ops::arange};
///
/// assert_eq!(arange(0, 5, 2).to_vec::<i32>().unwrap(), [0, 2, 4]);
/// assert_eq!(arange(0.0, 1.0, 0.25).dtype(), DType::F32);
/// ```
#[track_caller]
pub fn arange(start: impl Into<Scalar>, stop: impl Into<Scalar>, step: impl Into<Scalar>) -> Array {
    let (start, stop, step) = (start.into(), stop.into(), step.into());
    let at = Location::caller();
    let bounds = [start, stop, step];
    let dtype = if bounds.iter().any(|s| matches!(s, Scalar::Float(_))) {
        DType::F32
    } else if bounds
        .iter()
        .all(|s| s.default_dtype().category() != Category::Complex)
        && bounds.iter().all(|s| match s {
            Scalar::Int(v) => i32::try_from(*v).is_ok(),
            _ => true,
        })
    {
        DType::I32
    } else {
        DType::I64
    };
    Array::sticky("arange", at, &[], || {
        let f = |s: Scalar| match s {
            Scalar::Bool(b) => Ok(f64::from(u8::from(b))),
            Scalar::Int(v) => Ok(v as f64),
            Scalar::Float(x) => Ok(x),
            Scalar::Complex(_) => Err(Error::unsupported_dtype("arange", DType::Complex64)),
        };
        imp::arange(f(start)?, f(stop)?, f(step)?, dtype)
    })
}

/// Zeros of `a`'s shape and dtype; see [`Array::zeros_like`].
#[track_caller]
pub fn zeros_like(a: &Array) -> Array {
    Array::sticky("zeros_like", Location::caller(), &[a], || {
        imp::broadcast_to(&imp::zeros(&[], a.dtype())?, a.dims())
    })
}

/// Ones of `a`'s shape and dtype; see [`Array::ones_like`].
#[track_caller]
pub fn ones_like(a: &Array) -> Array {
    Array::sticky("ones_like", Location::caller(), &[a], || {
        imp::broadcast_to(&imp::ones(&[], a.dtype())?, a.dims())
    })
}

/// `value` (a weak scalar takes `a`'s dtype) in `a`'s shape; see [`Array::full_like`].
#[track_caller]
pub fn full_like(a: &Array, value: impl Into<ScalarOrArray>) -> Array {
    let value = value.into();
    Array::sticky("full_like", Location::caller(), &[a], || {
        let v = imp::astype(&operand(value.clone(), a.dtype())?, a.dtype())?;
        imp::contiguous(&imp::broadcast_to(&v, a.dims())?)
    })
}

// ── Methods ─────────────────────────────────────────────────────────────────

/// The methods of [`Array`], one `impl` block per family, each method delegating to the
/// free function of the same name (or the one given after `=`). The list is also the table
/// the surface tests check.
macro_rules! methods {
    ($(
        $(#[$gdoc:meta])*
        impl {
            $(
                $(#[$doc:meta])*
                fn $name:ident(&self $(, $p:ident: $t:ty)*) -> $ret:ty $(= $target:ident)?;
            )*
        }
    )*) => {
        $(
            $(#[$gdoc])*
            #[allow(clippy::should_implement_trait, clippy::too_many_arguments)]
            impl Array {
                $(
                    $(#[$doc])*
                    #[track_caller]
                    pub fn $name(&self $(, $p: $t)*) -> $ret {
                        methods!(@call $name [$($target)?] (self $(, $p)*))
                    }
                )*
            }
        )*

        /// The names of the operations that exist as a method and as a free function of
        /// the same parameters (for the surface tests).
        #[doc(hidden)]
        pub fn __op_names() -> &'static [&'static str] {
            &[$($(stringify!($name),)*)*]
        }
    };
    (@call $name:ident [] ($($args:tt)*)) => { $name($($args)*) };
    (@call $name:ident [$target:ident] ($($args:tt)*)) => { $target($($args)*) };
}

methods! {
    /// # Element-wise binary operations
    ///
    /// The other operand is an array or a weak Rust number (it adapts to this array's dtype);
    /// dtypes are promoted and shapes broadcast as in MLX. The operators `+ - * /`
    /// are the same operations.
    impl {
        /// Element-wise `self + other`. Integers wrap; for booleans this is logical OR.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let x = array![1.0, 2.0];
        /// assert_eq!(x.add(1.0).to_vec::<f32>().unwrap(), [2.0, 3.0]);
        /// assert_eq!(x.add(&x).to_vec::<f32>().unwrap(), [2.0, 4.0]);
        /// ```
        fn add(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `self - other`. Integers wrap; for booleans this is logical XOR.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![3, 5].subtract(1).to_vec::<i32>().unwrap(), [2, 4]);
        /// ```
        fn subtract(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `self * other`. Integers wrap; for booleans this is logical AND.
        ///
        /// ```
        /// use nuvec_core::{DType, array};
        ///
        /// let y = array![1u8, 2].multiply(3);
        /// assert_eq!((y.dtype(), y.to_vec::<u8>().unwrap()), (DType::U8, vec![3, 6]));
        /// ```
        fn multiply(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise true division; integers are promoted to floating point.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1, 3].divide(2).to_vec::<f32>().unwrap(), [0.5, 1.5]);
        /// ```
        fn divide(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise maximum; NaN propagates. Not defined for complex numbers.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![-1.0, 2.0].maximum(0.0).to_vec::<f32>().unwrap(), [0.0, 2.0]);
        /// ```
        fn maximum(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise minimum; NaN propagates. Not defined for complex numbers.
        fn minimum(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `log(exp(self) + exp(other))`, computed stably, in floating point.
        fn logaddexp(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `self == other` as a bool array.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1, 2].equal(2).to_vec::<bool>().unwrap(), [false, true]);
        /// ```
        fn equal(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `self != other` as a bool array.
        fn not_equal(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `self < other` as a bool array.
        fn less(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `self <= other` as a bool array.
        fn less_equal(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `self > other` as a bool array.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1.0, 3.0].greater(2.0).to_vec::<bool>().unwrap(), [false, true]);
        /// ```
        fn greater(&self, other: impl Into<ScalarOrArray>) -> Array;
        /// Element-wise `self >= other` as a bool array.
        fn greater_equal(&self, other: impl Into<ScalarOrArray>) -> Array;
    }

    /// # Element-wise unary operations
    impl {
        /// Element-wise `-self`. Integers wrap. Not defined for booleans.
        fn negative(&self) -> Array;
        /// Element-wise absolute value; the float32 magnitude for complex numbers.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![-2, 3].abs().to_vec::<i32>().unwrap(), [2, 3]);
        /// ```
        fn abs(&self) -> Array;
        /// Element-wise `e^self`, in at least float32.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![0.0].exp().item::<f32>().unwrap(), 1.0);
        /// ```
        fn exp(&self) -> Array;
        /// Element-wise natural logarithm, in at least float32.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1.0].log().item::<f32>().unwrap(), 0.0);
        /// ```
        fn log(&self) -> Array;
        /// Element-wise square root, in at least float32.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![4, 9].sqrt().to_vec::<f32>().unwrap(), [2.0, 3.0]);
        /// ```
        fn sqrt(&self) -> Array;
        /// Element-wise sine, in at least float32.
        fn sin(&self) -> Array;
        /// Element-wise cosine, in at least float32.
        fn cos(&self) -> Array;
        /// Element-wise hyperbolic tangent, in at least float32.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![0.0].tanh().item::<f32>().unwrap(), 0.0);
        /// ```
        fn tanh(&self) -> Array;
        /// Element-wise logistic function `1 / (1 + e^-self)`, in at least float32.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![0.0].sigmoid().item::<f32>().unwrap(), 0.5);
        /// ```
        fn sigmoid(&self) -> Array;
        /// Element-wise inverse error function, in at least float32.
        fn erfinv(&self) -> Array;
        /// Element-wise sign: `-1`, `0` or `1` in the array's dtype (NaN gives 0).
        fn sign(&self) -> Array;
        /// Element-wise complex conjugate; real arrays are returned unchanged.
        fn conjugate(&self) -> Array;
        /// The array unchanged, but no gradient flows through it. Shares the data.
        ///
        /// ```
        /// use nuvec_core::{Array, array, grad};
        ///
        /// let g = grad(|x: &Array| x.multiply(&x.stop_gradient()).sum(..))(&array![3.0]).unwrap();
        /// assert_eq!(g.item::<f32>().unwrap(), 3.0); // d(x·c)/dx with c = x held constant
        /// ```
        fn stop_gradient(&self) -> Array;
        /// The array in row-major memory: itself if it already is, else a copy.
        fn contiguous(&self) -> Array;
        /// An independent copy in a buffer of its own (MLX's `mx.copy`), for snapshots: no
        /// `compile` simplification, constant folding or in-place update removes it or
        /// makes it share memory with `self`, so updating `self` in place afterwards leaves
        /// the copy unchanged. Derivatives pass through unchanged.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let cache = array![1.0, 2.0, 3.0];
        /// let snapshot = cache.copy();
        /// let cache = cache.at(0).set(9.0); // in place where `cache` owns its buffer
        /// assert_eq!(snapshot.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
        /// assert_eq!(cache.to_vec::<f32>().unwrap(), [9.0, 2.0, 3.0]);
        /// ```
        ///
        /// # Memory
        ///
        /// The copy's own bytes; no scratch.
        fn copy(&self) -> Array;
    }

    /// # Selection, conversion and matrix products
    impl {
        /// `a` where `self` (the condition, converted to bool) is true, else `b` (MLX's
        /// `where`); `a` and `b` may be weak Rust numbers.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let m = array![true, false];
        /// let a = array![1.0, 2.0];
        /// assert_eq!(m.select(&a, 0.0).to_vec::<f32>().unwrap(), [1.0, 0.0]);
        /// ```
        #[doc(alias = "where")]
        fn select(&self, a: impl Into<ScalarOrArray>, b: impl Into<ScalarOrArray>) -> Array = where_;
        /// The elements converted to `dtype`. Float-to-integer conversion saturates (NaN
        /// becomes 0); conversion to bool tests for non-zero; complex-to-real keeps the real
        /// part.
        ///
        /// ```
        /// use nuvec_core::{DType, array};
        ///
        /// assert_eq!(array![1.7, -2.2].astype(DType::I32).to_vec::<i32>().unwrap(), [1, -2]);
        /// ```
        fn astype(&self, dtype: DType) -> Array;
        /// The matrix product `self @ other` (`[..., M, K] @ [..., K, N]`, batch dimensions
        /// broadcast; 1-D operands as in NumPy). The promoted type must be floating point.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let a = array![[1.0, 2.0], [3.0, 4.0]];
        /// assert_eq!(a.matmul(&a.t()).to_vec::<f32>().unwrap(), [5.0, 11.0, 11.0, 25.0]);
        /// let bad = a.matmul(&array![1.0, 2.0, 3.0]);
        /// assert!(bad.is_failed());
        /// ```
        ///
        /// With at most [`ROW_INDEPENDENT_ROWS`](crate::ROW_INDEPENDENT_ROWS) rows, each
        /// row of the result has the bits of that row's product alone (multi-row
        /// matrix-vector kernels on the GPU).
        ///
        /// # Memory
        ///
        /// No scratch, except split-K GEMM on the GPU (few output tiles over a long K):
        /// `partitions × M × N × 4` bytes of float32 partial products, `partitions` being
        /// `K / 16 / tiles` rounded up to a power of two in `2..=32` (`tiles = ⌈M/32⌉·⌈N/32⌉`).
        /// Operands that are neither row- nor column-major are copied first.
        fn matmul(&self, other: &Array) -> Array;
        /// `alpha · (a @ b) + beta · self`, with `self` broadcast to the product's shape
        /// (MLX's `addmm`; one GEMM with an epilogue on the GPU).
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let bias = array![1.0, 1.0];
        /// let y = bias.addmm(&array![[1.0, 2.0]], &array![[1.0, 0.0], [0.0, 1.0]], 1.0, 1.0);
        /// assert_eq!(y.to_vec::<f32>().unwrap(), [2.0, 3.0]);
        /// ```
        fn addmm(&self, a: &Array, b: &Array, alpha: f32, beta: f32) -> Array;
    }

    /// # Reductions
    ///
    /// Over [`Axes`]: `-1`, `[0, 2]`, `..` (all), `1..` (trailing); an empty list reduces
    /// nothing. Each has a `*_keepdims` twin that keeps the reduced axes as size-1 dimensions.
    impl {
        /// The sum over `axes`. Integers and bools widen to 32 or 64 bits, as in MLX.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let x = array![[1, 2, 3], [4, 5, 6]];
        /// assert_eq!(x.sum(-1).to_vec::<i32>().unwrap(), [6, 15]);
        /// assert_eq!(x.sum(..).item::<i32>().unwrap(), 21);
        /// ```
        fn sum(&self, axes: impl Axes) -> Array;
        /// [`Array::sum`] keeping the reduced axes as size-1 dimensions.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![[1, 2, 3], [4, 5, 6]].sum_keepdims(0).shape(), [1, 3]);
        /// ```
        fn sum_keepdims(&self, axes: impl Axes) -> Array;
        /// The product over `axes`, widened as [`Array::sum`].
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![2, 3, 4].prod(..).item::<i32>().unwrap(), 24);
        /// ```
        fn prod(&self, axes: impl Axes) -> Array;
        /// [`Array::prod`] keeping the reduced axes.
        fn prod_keepdims(&self, axes: impl Axes) -> Array;
        /// The maximum over `axes`; NaN propagates; zero-length axes fail.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![[1, 5], [7, 2]].max(1).to_vec::<i32>().unwrap(), [5, 7]);
        /// ```
        fn max(&self, axes: impl Axes) -> Array;
        /// [`Array::max`] keeping the reduced axes.
        fn max_keepdims(&self, axes: impl Axes) -> Array;
        /// The minimum over `axes`; NaN propagates; zero-length axes fail.
        fn min(&self, axes: impl Axes) -> Array;
        /// [`Array::min`] keeping the reduced axes.
        fn min_keepdims(&self, axes: impl Axes) -> Array;
        /// Whether every element over `axes` is non-zero.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert!(!array![true, false].all(..).item::<bool>().unwrap());
        /// ```
        fn all(&self, axes: impl Axes) -> Array;
        /// [`Array::all`] keeping the reduced axes.
        fn all_keepdims(&self, axes: impl Axes) -> Array;
        /// Whether any element over `axes` is non-zero.
        fn any(&self, axes: impl Axes) -> Array;
        /// [`Array::any`] keeping the reduced axes.
        fn any_keepdims(&self, axes: impl Axes) -> Array;
        /// The mean over `axes`, in at least float32. Empty reductions are NaN.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1, 2, 3, 4].mean(..).item::<f32>().unwrap(), 2.5);
        /// ```
        fn mean(&self, axes: impl Axes) -> Array;
        /// [`Array::mean`] keeping the reduced axes.
        fn mean_keepdims(&self, axes: impl Axes) -> Array;
        /// `log(sum(exp(self)))` over `axes`, computed stably (the maximum subtracted first).
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let l = array![0.0, 0.0].logsumexp(..).item::<f32>().unwrap();
        /// assert!((l - 2f32.ln()).abs() < 1e-6);
        /// ```
        fn logsumexp(&self, axes: impl Axes) -> Array;
        /// [`Array::logsumexp`] keeping the reduced axes.
        fn logsumexp_keepdims(&self, axes: impl Axes) -> Array;
        /// The variance over `axes` (`mean((x − mean(x))²)`), in at least float32.
        ///
        /// ```
        /// use nuvec_core::{VarOptions, array};
        ///
        /// let x = array![1.0, 2.0, 3.0, 4.0];
        /// assert_eq!(x.var(..).item::<f32>().unwrap(), 1.25);
        /// let sample = x.var_with(.., VarOptions { ddof: 1, ..Default::default() });
        /// assert!((sample.item::<f32>().unwrap() - 5.0 / 3.0).abs() < 1e-6);
        /// ```
        fn var(&self, axes: impl Axes) -> Array;
        /// [`Array::var`] keeping the reduced axes.
        fn var_keepdims(&self, axes: impl Axes) -> Array;
        /// [`Array::var`] with [`VarOptions`] (`ddof`, `keepdims`).
        fn var_with(&self, axes: impl Axes, options: VarOptions) -> Array;
        /// The standard deviation over `axes` (the square root of [`Array::var`]).
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1.0, 3.0].std(..).item::<f32>().unwrap(), 1.0);
        /// ```
        fn std(&self, axes: impl Axes) -> Array;
        /// [`Array::std`] keeping the reduced axes.
        fn std_keepdims(&self, axes: impl Axes) -> Array;
        /// [`Array::std`] with [`VarOptions`] (`ddof`, `keepdims`).
        fn std_with(&self, axes: impl Axes, options: VarOptions) -> Array;
        /// The index (uint32) of the maximum along one [`Axis`] (`..`: of the flattened
        /// array). The first occurrence wins; the first NaN wins over any number.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let x = array![[1, 9, 3], [7, 2, 8]];
        /// assert_eq!(x.argmax(-1).to_vec::<u32>().unwrap(), [1, 2]);
        /// assert_eq!(x.argmax(..).item::<u32>().unwrap(), 1);
        /// ```
        fn argmax(&self, axis: impl Axis) -> Array;
        /// [`Array::argmax`] keeping the reduced axis (every axis for `..`).
        fn argmax_keepdims(&self, axis: impl Axis) -> Array;
        /// The index (uint32) of the minimum along one axis (`..`: of the flattened array).
        fn argmin(&self, axis: impl Axis) -> Array;
        /// [`Array::argmin`] keeping the reduced axis (every axis for `..`).
        fn argmin_keepdims(&self, axis: impl Axis) -> Array;
        /// `exp(self) / sum(exp(self))` over `axes`, with the maximum subtracted first.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1.0, 1.0].softmax(-1).to_vec::<f32>().unwrap(), [0.5, 0.5]);
        /// ```
        fn softmax(&self, axes: impl Axes) -> Array;
    }

    /// # Scans and sorting
    ///
    /// Along one [`Axis`]; `..` works on the flattened array.
    impl {
        /// The cumulative sum along one axis. Booleans sum as int32.
        ///
        /// ```
        /// use nuvec_core::{ScanOptions, array};
        ///
        /// let a = array![1, 2, 3, 4];
        /// assert_eq!(a.cumsum(0).to_vec::<i32>().unwrap(), [1, 3, 6, 10]);
        /// let opts = ScanOptions { reverse: true, exclusive: true };
        /// assert_eq!(a.cumsum_with(0, opts).to_vec::<i32>().unwrap(), [9, 7, 4, 0]);
        /// ```
        #[doc(alias = "cumulative_sum")]
        fn cumsum(&self, axis: impl Axis) -> Array;
        /// [`Array::cumsum`] with [`ScanOptions`] (`reverse`, `exclusive`).
        fn cumsum_with(&self, axis: impl Axis, options: ScanOptions) -> Array;
        /// The cumulative product along one axis.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1, 2, 3].cumprod(..).to_vec::<i32>().unwrap(), [1, 2, 6]);
        /// ```
        #[doc(alias = "cumulative_prod")]
        fn cumprod(&self, axis: impl Axis) -> Array;
        /// [`Array::cumprod`] with [`ScanOptions`].
        fn cumprod_with(&self, axis: impl Axis, options: ScanOptions) -> Array;
        /// The running maximum along one axis (NaN propagates).
        fn cummax(&self, axis: impl Axis) -> Array;
        /// [`Array::cummax`] with [`ScanOptions`].
        fn cummax_with(&self, axis: impl Axis, options: ScanOptions) -> Array;
        /// The running minimum along one axis (NaN propagates).
        fn cummin(&self, axis: impl Axis) -> Array;
        /// [`Array::cummin`] with [`ScanOptions`].
        fn cummin_with(&self, axis: impl Axis, options: ScanOptions) -> Array;
        /// The running `log(Σ exp)` along one axis, in floating point.
        fn logcumsumexp(&self, axis: impl Axis) -> Array;
        /// [`Array::logcumsumexp`] with [`ScanOptions`].
        fn logcumsumexp_with(&self, axis: impl Axis, options: ScanOptions) -> Array;
        /// The array sorted along one axis: ascending, stable, NaN last.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![3, 1, 2].sort(-1).to_vec::<i32>().unwrap(), [1, 2, 3]);
        /// ```
        fn sort(&self, axis: impl Axis) -> Array;
        /// The indices (uint32) that sort the array along one axis, as [`Array::sort`].
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![3, 1, 2].argsort(0).to_vec::<u32>().unwrap(), [1, 2, 0]);
        /// ```
        fn argsort(&self, axis: impl Axis) -> Array;
        /// The array partitioned along one axis: the element at `kth` (negative counts from
        /// the end) is the one a sort would put there, smaller ones before, larger ones after.
        fn partition(&self, kth: i32, axis: impl Axis) -> Array;
        /// The indices (uint32) that partition the array, as [`Array::partition`].
        fn argpartition(&self, kth: i32, axis: impl Axis) -> Array;
        /// The `k` largest elements along one axis, in ascending order.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![5, 1, 9, 3, 7].topk(2, 0).to_vec::<i32>().unwrap(), [7, 9]);
        /// ```
        fn topk(&self, k: i32, axis: impl Axis) -> Array;
    }

    /// # Shapes and views
    impl {
        /// The same elements in another shape ([`IntoShape`]: `[b, l, -1]`, one `-1`
        /// inferred). Shares the data when the layout permits.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let x = array![[1, 2, 3], [4, 5, 6]];
        /// let [b, n] = x.shape_n::<2>().unwrap();
        /// assert_eq!(x.reshape([n, b]).shape(), [3, 2]);
        /// assert_eq!(x.reshape([-1]).shape(), [6]);
        /// ```
        fn reshape(&self, shape: impl IntoShape) -> Array;
        /// The axes reordered: output axis `i` is input axis `axes[i]` (`..` reverses them, as
        /// MLX's default). Shares the data.
        ///
        /// ```
        /// use nuvec_core::{DType, ops::zeros};
        ///
        /// let x = zeros([2, 3, 4], DType::F32);
        /// assert_eq!(x.transpose([0, 2, 1]).shape(), [2, 4, 3]);
        /// assert_eq!(x.transpose(..).shape(), [4, 3, 2]);
        /// ```
        #[doc(alias = "permute_dims")]
        fn transpose(&self, axes: impl Axes) -> Array;
        /// The array presented as the larger `shape` (NumPy broadcasting). Shares the data.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1, 2].broadcast_to([3, 2]).shape(), [3, 2]);
        /// ```
        fn broadcast_to(&self, shape: impl IntoShape) -> Array;
        /// Size-1 dimensions inserted at `axes` (positions in the result). Shares the data.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1, 2].expand_dims([0, -1]).shape(), [1, 2, 1]);
        /// ```
        fn expand_dims(&self, axes: impl Axes) -> Array;
        /// Size-1 dimensions removed: the given `axes`, or all of them for `..`. Shares the
        /// data.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![[[1], [2]]].squeeze(..).shape(), [2]);
        /// ```
        fn squeeze(&self, axes: impl Axes) -> Array;
        /// Each element repeated `repeats` times along one axis (`..`: the flattened array).
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1, 2].repeat(2, 0).to_vec::<i32>().unwrap(), [1, 1, 2, 2]);
        /// ```
        fn repeat(&self, repeats: i32, axis: impl Axis) -> Array;
        /// The array split into `parts` equal parts along `axis` (views); `parts` failed
        /// arrays if the axis does not divide.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let parts = array![1, 2, 3, 4, 5, 6].split(3, 0);
        /// assert_eq!(parts[2].to_vec::<i32>().unwrap(), [5, 6]);
        /// ```
        fn split(&self, parts: i32, axis: impl Axis) -> Vec<Array>;
        /// The array split before each of the (non-decreasing) positions `indices` along
        /// `axis`: `indices.len() + 1` views.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let parts = array![1, 2, 3, 4].split_at(&[1, 3], 0);
        /// assert_eq!(parts[1].to_vec::<i32>().unwrap(), [2, 3]);
        /// ```
        fn split_at(&self, indices: &[i32], axis: impl Axis) -> Vec<Array>;
    }

    /// # Gathers and scatters
    ///
    /// See also [`Array::i`] and [`Array::at`].
    impl {
        /// The elements at `indices` along one axis (`..`: the flattened array): the axis is
        /// replaced by `indices`' dimensions (MLX's `take`). Negative indices count from the
        /// end.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let table = array![[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]];
        /// let rows = table.take(&array![2, 0], 0);
        /// assert_eq!(rows.to_vec::<f32>().unwrap(), [2.0, 2.5, 0.0, 0.5]);
        /// ```
        fn take(&self, indices: &Array, axis: impl Axis) -> Array;
        /// The elements at `indices` along `axis` (MLX's `take_along_axis`).
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let a = array![[10.0, 20.0], [30.0, 40.0]];
        /// let i = array![[1], [0]];
        /// assert_eq!(a.take_along_axis(&i, 1).to_vec::<f32>().unwrap(), [20.0, 30.0]);
        /// ```
        fn take_along_axis(&self, indices: &Array, axis: impl Axis) -> Array;
        /// A copy with `values` placed at `indices` along `axis` (MLX's `put_along_axis`).
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let a = array![0, 0, 0, 0];
        /// let b = a.put_along_axis(&array![3, -4], &array![7, 9], 0);
        /// assert_eq!(b.to_vec::<i32>().unwrap(), [9, 0, 0, 7]);
        /// ```
        fn put_along_axis(&self, indices: &Array, values: impl Into<ScalarOrArray>, axis: impl Axis) -> Array;
        /// A copy with `values` added at `indices` along `axis`; repeated indices accumulate.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let a = array![0, 0];
        /// assert_eq!(a.scatter_add_along_axis(&array![1, 1], 5, 0).to_vec::<i32>().unwrap(), [0, 10]);
        /// ```
        fn scatter_add_along_axis(&self, indices: &Array, values: impl Into<ScalarOrArray>, axis: impl Axis) -> Array;
    }

    /// # Quantization and convolution
    impl {
        /// The matrix quantized along its last axis as MLX's affine quantization:
        /// `(w_q, scales, biases)`.
        ///
        /// ```
        /// use nuvec_core::{DType, ops::ones};
        ///
        /// let (wq, scales, biases) = ones([2, 64], DType::F32).quantize(64, 4);
        /// assert_eq!((wq.shape(), scales.shape(), biases.shape()), (&[2, 8][..], &[2, 1][..], &[2, 1][..]));
        /// ```
        fn quantize(&self, group_size: i32, bits: i32) -> (Array, Array, Array);
        /// The weights of [`Array::quantize`] restored from `self` (the packed weights).
        fn dequantize(&self, scales: &Array, biases: &Array, group_size: i32, bits: i32) -> Array;
        /// `self · dequantize(w)ᵀ` (`transpose`) or `self · dequantize(w)`. Up to
        /// [`ROW_INDEPENDENT_ROWS`](crate::ROW_INDEPENDENT_ROWS) rows run as
        /// matrix-vector products, each row with the bits of its own product.
        ///
        /// # Memory
        ///
        /// No scratch, except quantized split-K (many rows, transposed weights, one batch):
        /// `split × M × N × 4` bytes, `split = min(512 / tiles, K / max(group, 32))`
        /// reduced until it divides `K` (`tiles = ⌈M/32⌉·⌈N/32⌉`).
        fn quantized_matmul(&self, w: &Array, scales: &Array, biases: &Array, transpose: bool, group_size: i32, bits: i32) -> Array;
        /// Quantized products of gathered matrices (MLX's `gather_qmm`). Up to
        /// `ROW_INDEPENDENT_ROWS` tokens of up to 8 experts run one product per row, sorted
        /// or not, each with the bits of its own product.
        ///
        /// # Memory
        ///
        /// With sorted indices over many rows: the experts' row counts and offsets (int32,
        /// two per expert) and an int32 per row; otherwise none.
        fn gather_qmm(&self, w: &Array, scales: &Array, biases: &Array, lhs_indices: Option<&Array>, rhs_indices: Option<&Array>, transpose: bool, group_size: i32, bits: i32, sorted_indices: bool) -> Array;
        /// Products of gathered matrices (MLX's `gather_mm`). One row per output batch
        /// (up to `ROW_INDEPENDENT_ROWS` tokens of up to 8 experts) runs MLX's gathered
        /// matrix-vector kernels, each row with the bits of its own product.
        ///
        /// # Memory
        ///
        /// As [`Array::gather_qmm`]: the sorted path's counts and offsets; otherwise none
        /// (operands that are not row-major are copied first).
        fn gather_mm(&self, b: &Array, lhs_indices: Option<&Array>, rhs_indices: Option<&Array>, sorted_indices: bool) -> Array;
        /// A 1-D convolution of `self` (`[N, L, C_in]`) with `weight` (`[C_out, K, C_in /
        /// groups]`).
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// let x = array![[[1.0], [2.0], [3.0]]]; // [1, 3, 1]
        /// let w = array![[[1.0], [1.0]]]; // [1, 2, 1]
        /// assert_eq!(x.conv1d(&w, 1, 0, 1, 1).to_vec::<f32>().unwrap(), [3.0, 5.0]);
        /// ```
        fn conv1d(&self, weight: &Array, stride: i32, padding: i32, dilation: i32, groups: i32) -> Array;
    }

    /// # Creation from an existing array
    impl {
        /// Zeros of this array's shape and dtype.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1u8, 2].zeros_like().to_vec::<u8>().unwrap(), [0, 0]);
        /// ```
        fn zeros_like(&self) -> Array;
        /// Ones of this array's shape and dtype.
        fn ones_like(&self) -> Array;
        /// `value` (a weak scalar takes this array's dtype) in this array's shape and dtype.
        ///
        /// ```
        /// use nuvec_core::array;
        ///
        /// assert_eq!(array![1.0, 2.0].full_like(7).to_vec::<f32>().unwrap(), [7.0, 7.0]);
        /// ```
        fn full_like(&self, value: impl Into<ScalarOrArray>) -> Array;
    }
}

/// # Shortcuts
impl Array {
    /// The array with all axes reversed (the matrix transpose for 2-D arrays). Shares the
    /// data.
    ///
    /// ```
    /// use nuvec_core::array;
    ///
    /// assert_eq!(array![[1, 2, 3]].t().shape(), [3, 1]);
    /// ```
    #[track_caller]
    pub fn t(&self) -> Array {
        transpose(self, ..)
    }

    /// The array split into `N` equal parts along `axis`, for destructuring:
    /// `let [q, gate] = qg.split_n::<2>(-1);`. `N` failed arrays if the axis does not divide.
    ///
    /// ```
    /// use nuvec_core::array;
    ///
    /// let qg = array![[1.0, 2.0, 3.0, 4.0]];
    /// let [q, g] = qg.split_n::<2>(-1);
    /// assert_eq!((q.shape(), g.to_vec::<f32>().unwrap()), (&[1, 2][..], vec![3.0, 4.0]));
    /// ```
    #[track_caller]
    pub fn split_n<const N: usize>(&self, axis: impl Axis) -> [Array; N] {
        split_n::<N>(self, axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn located_failures() {
        let a = zeros([2, 3], DType::F32);
        let b = zeros([4, 5], DType::F32);
        let line = line!() + 1;
        let c = a.matmul(&b);
        let err = c.clone().check().unwrap_err();
        assert_eq!(err.op_name(), Some("matmul"));
        let at = err.location().unwrap();
        assert!(at.file().ends_with("ops.rs") && at.line() == line, "{err}");
        // Sticky: the first error wins.
        let d = (c.exp() + 1.0).sum(-1);
        assert_eq!(d.check().unwrap_err(), err);
    }

    #[test]
    fn empty_axes_reduce_nothing() {
        let x = Array::from_slice(&[1i8, 2, 3, 4], [2, 2]);
        let s = x.sum([]);
        assert_eq!((s.shape(), s.dtype()), (&[2, 2][..], DType::I8));
        assert_eq!(x.all(Vec::<i32>::new()).dtype(), DType::Bool);
        assert_eq!(x.mean([]).dtype(), DType::F32);
        assert_eq!(x.sum(..).item::<i32>().unwrap(), 10);
        // Integers never wrap in the variance (computed in float).
        let v = Array::from([100i8, -100, 50])
            .var(..)
            .item::<f32>()
            .unwrap();
        assert!((v - 7222.222).abs() < 0.01, "{v}");
    }
}
