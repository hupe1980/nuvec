//! MLX's shape, creation and linear-algebra conveniences (`ops.cpp`, `linalg.cpp`),
//! composed from the primitives: their values, dtypes and gradients are MLX's.

use std::panic::Location;

use super::imp;
use crate::view::SliceArg;
use crate::{Array, DType, Error, Result};

fn invalid(name: &str, message: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("{name}: {message}"))
}

/// `axis` (negative from the end) as an index into `ndim` dimensions.
fn axis_of(name: &str, axis: i32, ndim: usize) -> Result<usize> {
    let n = ndim as i32;
    let a = if axis < 0 { axis + n } else { axis };
    if a < 0 || a >= n.max(0) {
        return Err(invalid(
            name,
            format!("axis {axis} is out of range for {ndim} dimensions"),
        ));
    }
    Ok(a as usize)
}

fn axes_of(name: &str, axes: Option<&[i32]>, ndim: usize) -> Result<Vec<usize>> {
    match axes {
        None => Ok((0..ndim).collect()),
        Some(list) => list.iter().map(|&a| axis_of(name, a, ndim)).collect(),
    }
}

#[track_caller]
fn sticky(name: &'static str, inputs: &[&Array], f: impl FnOnce() -> Result<Array>) -> Array {
    Array::sticky(name, Location::caller(), inputs, f)
}

fn int_array(values: &[i32], shape: &[usize]) -> Result<Array> {
    crate::creation::from_data(values, shape)
}

fn slice_axis(a: &Array, axis: usize, start: isize, stop: isize, step: isize) -> Result<Array> {
    let mut args = vec![SliceArg::full(); a.ndim()];
    args[axis] = SliceArg {
        start: Some(start),
        stop: Some(stop),
        step,
    };
    imp::slice(a, &args)
}

fn take(a: &Array, indices: &[i32], axis: usize) -> Result<Array> {
    super::take(a, &int_array(indices, &[indices.len()])?, axis as i32).into_result()
}

use crate::IntoResult;

// ── Reordering ───────────────────────────────────────────────────────────────

/// The elements in reverse order along `axes` (all axes for `None`), as a view.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// assert_eq!(ops::flip(&array![[1, 2], [3, 4]], Some(&[1])).to_vec::<i32>().unwrap(), [2, 1, 4, 3]);
/// ```
#[track_caller]
pub fn flip(a: &Array, axes: Option<&[i32]>) -> Array {
    sticky("flip", &[a], || {
        let axes = axes_of("flip", axes, a.ndim())?;
        let mut args = vec![SliceArg::full(); a.ndim()];
        for &d in &axes {
            args[d] = SliceArg {
                start: None,
                stop: None,
                step: -1,
            };
        }
        imp::slice(a, &args)
    })
}

/// The elements shifted by `shift` along `axes` (MLX's `roll`): elements that fall off the
/// end come back at the start. With `axes = None` the flattened array is rolled.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// assert_eq!(ops::roll(&array![1, 2, 3, 4], &[1], None).to_vec::<i32>().unwrap(), [4, 1, 2, 3]);
/// ```
#[track_caller]
pub fn roll(a: &Array, shift: &[i32], axes: Option<&[i32]>) -> Array {
    sticky("roll", &[a], || {
        let Some(axes) = axes else {
            if shift.len() != 1 {
                return Err(invalid("roll", "one shift for the flattened array"));
            }
            let flat = imp::reshape(a, &[a.size() as isize])?;
            let rolled = roll(&flat, shift, Some(&[0])).into_result()?;
            return imp::reshape(
                &rolled,
                &a.dims().iter().map(|&d| d as isize).collect::<Vec<_>>(),
            );
        };
        let axes = axes_of("roll", Some(axes), a.ndim())?;
        let shifts: Vec<i32> = match (shift.len(), axes.len()) {
            (1, n) => vec![shift[0]; n],
            (s, n) if s == n => shift.to_vec(),
            _ => return Err(invalid("roll", "the shifts and axes differ in number")),
        };
        let mut out = a.clone();
        for (&d, &s) in axes.iter().zip(&shifts) {
            let n = a.dims()[d] as i64;
            if n == 0 {
                continue;
            }
            let s = (i64::from(s)).rem_euclid(n) as isize;
            if s == 0 {
                continue;
            }
            let n = n as isize;
            let tail = slice_axis(&out, d, n - s, n, 1)?;
            let head = slice_axis(&out, d, 0, n - s, 1)?;
            out = super::concatenate(&[tail, head], d as i32).into_result()?;
        }
        Ok(out)
    })
}

/// `a` repeated `reps` times along each axis (NumPy's `tile`: `reps` and `a`'s shape are
/// right-aligned, the shorter padded with ones).
///
/// ```
/// use nuvec_core::{array, ops};
///
/// assert_eq!(ops::tile(&array![1, 2], &[2, 2]).to_vec::<i32>().unwrap(), [1, 2, 1, 2, 1, 2, 1, 2]);
/// ```
#[track_caller]
pub fn tile(a: &Array, reps: &[i32]) -> Array {
    sticky("tile", &[a], || {
        if reps.iter().any(|&r| r < 0) {
            return Err(invalid("tile", "negative repetitions"));
        }
        let n = a.ndim().max(reps.len());
        let mut dims = vec![1isize; n - a.ndim()];
        dims.extend(a.dims().iter().map(|&d| d as isize));
        let mut r = vec![1isize; n - reps.len()];
        r.extend(reps.iter().map(|&x| x as isize));
        // [d0, d1, ...] -> [1, d0, 1, d1, ...] -> [r0, d0, r1, d1, ...] -> [r0·d0, r1·d1, ...]
        let spread: Vec<isize> = dims.iter().flat_map(|&d| [1, d]).collect();
        let wide: Vec<usize> = r
            .iter()
            .zip(&dims)
            .flat_map(|(&k, &d)| [k as usize, d as usize])
            .collect();
        let x = imp::reshape(a, &spread)?;
        let x = imp::broadcast_to(&x, &wide)?;
        imp::reshape(
            &x,
            &r.iter()
                .zip(&dims)
                .map(|(&k, &d)| k * d)
                .collect::<Vec<_>>(),
        )
    })
}

/// `a` with axis `source` moved to position `destination`, as a view.
#[track_caller]
pub fn moveaxis(a: &Array, source: i32, destination: i32) -> Array {
    sticky("moveaxis", &[a], || {
        let n = a.ndim();
        let (s, d) = (
            axis_of("moveaxis", source, n)?,
            axis_of("moveaxis", destination, n)?,
        );
        let mut order: Vec<usize> = (0..n).filter(|&i| i != s).collect();
        order.insert(d, s);
        imp::transpose(a, &order)
    })
}

/// `a` with axes `axis1` and `axis2` exchanged, as a view.
#[track_caller]
pub fn swapaxes(a: &Array, axis1: i32, axis2: i32) -> Array {
    sticky("swapaxes", &[a], || {
        let n = a.ndim();
        let (x, y) = (
            axis_of("swapaxes", axis1, n)?,
            axis_of("swapaxes", axis2, n)?,
        );
        let mut order: Vec<usize> = (0..n).collect();
        order.swap(x, y);
        imp::transpose(a, &order)
    })
}

/// The last two axes exchanged (MLX's `matrix_transpose`).
#[track_caller]
pub fn matrix_transpose(a: &Array) -> Array {
    if a.ndim() < 2 {
        return sticky("matrix_transpose", &[a], || {
            Err(invalid(
                "matrix_transpose",
                "the array has fewer than 2 dimensions",
            ))
        });
    }
    swapaxes(a, -1, -2)
}

fn at_least(name: &'static str, a: &Array, n: usize) -> Array {
    sticky(name, &[a], || {
        Ok(match (n, a.ndim()) {
            (_, d) if d >= n => a.clone(),
            (1, _) => imp::reshape(a, &[1])?,
            (2, 0) => imp::reshape(a, &[1, 1])?,
            (2, _) => imp::reshape(a, &[1, a.dims()[0] as isize])?,
            (_, 0) => imp::reshape(a, &[1, 1, 1])?,
            (_, 1) => imp::reshape(a, &[1, a.dims()[0] as isize, 1])?,
            (_, _) => imp::reshape(a, &[a.dims()[0] as isize, a.dims()[1] as isize, 1])?,
        })
    })
}

/// `a` with at least one dimension.
#[track_caller]
pub fn atleast_1d(a: &Array) -> Array {
    at_least("atleast_1d", a, 1)
}

/// `a` with at least two dimensions (a vector becomes a row).
#[track_caller]
pub fn atleast_2d(a: &Array) -> Array {
    at_least("atleast_2d", a, 2)
}

/// `a` with at least three dimensions (NumPy's placement: `[n]` → `[1, n, 1]`, `[m, n]` →
/// `[m, n, 1]`).
#[track_caller]
pub fn atleast_3d(a: &Array) -> Array {
    at_least("atleast_3d", a, 3)
}

/// `a` with axis `axis` split into `shape` (one entry may be −1).
#[track_caller]
pub fn unflatten(a: &Array, axis: i32, shape: &[i32]) -> Array {
    sticky("unflatten", &[a], || {
        let d = axis_of("unflatten", axis, a.ndim())?;
        let mut dims: Vec<isize> = a.dims()[..d].iter().map(|&x| x as isize).collect();
        dims.extend(shape.iter().map(|&x| x as isize));
        dims.extend(a.dims()[d + 1..].iter().map(|&x| x as isize));
        imp::reshape(a, &dims)
    })
}

/// The slices of `a` along `axis`, each without that axis.
#[track_caller]
pub fn unstack(a: &Array, axis: i32) -> Vec<Array> {
    let d = match axis_of("unstack", axis, a.ndim()) {
        Ok(d) => d,
        Err(e) => return vec![sticky("unstack", &[a], || Err(e))],
    };
    (0..a.dims()[d] as isize)
        .map(|i| {
            sticky("unstack", &[a], || {
                imp::squeeze(&slice_axis(a, d, i, i + 1, 1)?, &[d as isize])
            })
        })
        .collect()
}

/// The arrays broadcast to their common shape (views).
#[track_caller]
pub fn broadcast_arrays(arrays: &[Array]) -> Vec<Array> {
    let shape = arrays.iter().try_fold(Vec::<usize>::new(), |s, x| {
        crate::view::broadcast_shapes("broadcast_arrays", &s, x.dims()).map(|v| v.to_vec())
    });
    arrays
        .iter()
        .map(|x| {
            sticky("broadcast_arrays", &[x], || {
                imp::broadcast_to(x, shape.as_ref().map_err(Clone::clone)?)
            })
        })
        .collect()
}

// ── Padding ──────────────────────────────────────────────────────────────────

/// How [`pad`] fills the new elements (MLX's modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PadMode {
    /// A constant value.
    #[default]
    Constant,
    /// The edge element repeated.
    Edge,
    /// A mirror image without the edge element (`[3, 2 | 1, 2, 3 | 2, 1]`).
    Reflect,
    /// A mirror image with the edge element (`[2, 1 | 1, 2, 3 | 3, 2]`).
    Symmetric,
}

/// `a` padded with `widths[d] = (before, after)` elements per axis (one pair for every
/// axis), filled as `mode` says (`value` for [`PadMode::Constant`]).
///
/// ```
/// use nuvec_core::{array, ops};
///
/// let x = array![1, 2, 3];
/// assert_eq!(ops::pad(&x, &[(1, 2)], ops::PadMode::Constant, 0).to_vec::<i32>().unwrap(), [0, 1, 2, 3, 0, 0]);
/// assert_eq!(ops::pad(&x, &[(2, 2)], ops::PadMode::Reflect, 0).to_vec::<i32>().unwrap(), [3, 2, 1, 2, 3, 2, 1]);
/// ```
#[track_caller]
pub fn pad(
    a: &Array,
    widths: &[(i32, i32)],
    mode: PadMode,
    value: impl Into<crate::scalar::ScalarOrArray>,
) -> Array {
    let value = value.into();
    sticky("pad", &[a], || {
        let n = a.ndim();
        let widths: Vec<(i32, i32)> = match widths.len() {
            1 => vec![widths[0]; n],
            w if w == n => widths.to_vec(),
            w => {
                return Err(invalid("pad", format!("{w} pad widths for {n} dimensions")));
            }
        };
        if widths.iter().any(|&(b, e)| b < 0 || e < 0) {
            return Err(invalid("pad", "negative pad widths"));
        }
        let mut out = a.clone();
        for (d, &(before, after)) in widths.iter().enumerate() {
            if before == 0 && after == 0 {
                continue;
            }
            let size = out.dims()[d] as i64;
            out = match mode {
                PadMode::Constant => {
                    let fill = value.clone().into_array(Some(out.dtype()))?;
                    let fill = imp::astype(&fill, out.dtype())?;
                    let side = |k: i32| -> Result<Array> {
                        let mut dims = out.dims().to_vec();
                        dims[d] = k as usize;
                        imp::broadcast_to(&fill, &dims)
                    };
                    super::concatenate(&[side(before)?, out.clone(), side(after)?], d as i32)
                        .into_result()?
                }
                _ => {
                    if size == 0 {
                        return Err(invalid("pad", "a non-constant pad of an empty axis"));
                    }
                    if mode == PadMode::Reflect && size == 1 {
                        // NumPy and MLX: a reflection of one element repeats it.
                        let idx = vec![0; (i64::from(before) + 1 + i64::from(after)) as usize];
                        out = take(&out, &idx, d)?;
                        continue;
                    }
                    let idx: Vec<i32> = (-i64::from(before)..size + i64::from(after))
                        .map(|i| pad_index(mode, i, size) as i32)
                        .collect();
                    take(&out, &idx, d)?
                }
            };
        }
        Ok(out)
    })
}

/// The source index of position `i` (−before ≤ i < size + after) of a padded axis.
fn pad_index(mode: PadMode, i: i64, n: i64) -> i64 {
    match mode {
        PadMode::Edge => i.clamp(0, n - 1),
        PadMode::Reflect => {
            let period = 2 * (n - 1);
            let j = i.rem_euclid(period);
            if j >= n { period - j } else { j }
        }
        _ => {
            let period = 2 * n;
            let j = i.rem_euclid(period);
            if j >= n { period - 1 - j } else { j }
        }
    }
}

// ── Diagonals and triangles ──────────────────────────────────────────────────

/// `[n, m]` ones on and below the `k`-th diagonal, zeros above (MLX's `tri`), in `dtype`.
#[track_caller]
pub fn tri(n: i32, m: Option<i32>, k: i32, dtype: DType) -> Array {
    sticky("tri", &[], || {
        let m = m.unwrap_or(n);
        if n < 0 || m < 0 {
            return Err(invalid("tri", "negative size"));
        }
        imp::astype(&lower_mask(n as usize, m as usize, k)?, dtype)
    })
}

/// `rows[:, None] + k >= cols[None, :]` for an `[n, m]` matrix.
fn lower_mask(n: usize, m: usize, k: i32) -> Result<Array> {
    let rows: Vec<i32> = (0..n as i32).map(|i| i + k).collect();
    let cols: Vec<i32> = (0..m as i32).collect();
    imp::greater_equal(&int_array(&rows, &[n, 1])?, &int_array(&cols, &[1, m])?)
}

fn triangle(name: &'static str, a: &Array, k: i32, lower: bool) -> Array {
    sticky(name, &[a], || {
        if a.ndim() < 2 {
            return Err(invalid(name, "the array has fewer than 2 dimensions"));
        }
        let (n, m) = (a.dims()[a.ndim() - 2], a.dims()[a.ndim() - 1]);
        // tril keeps j <= i + k; triu keeps j >= i + k, the complement of tril(k − 1).
        let mask = if lower {
            lower_mask(n, m, k)?
        } else {
            imp::equal(&lower_mask(n, m, k - 1)?, false)?
        };
        imp::where_(&mask, a, &imp::zeros(&[], a.dtype())?)
    })
}

/// `a`'s elements on and below the `k`-th diagonal of its last two axes, zeros above.
#[track_caller]
pub fn tril(a: &Array, k: i32) -> Array {
    triangle("tril", a, k, true)
}

/// `a`'s elements on and above the `k`-th diagonal of its last two axes, zeros below.
#[track_caller]
pub fn triu(a: &Array, k: i32) -> Array {
    triangle("triu", a, k, false)
}

/// The `[n, m]` matrix with ones on the `k`-th diagonal, in `dtype`.
#[track_caller]
pub fn eye(n: i32, m: Option<i32>, k: i32, dtype: DType) -> Array {
    sticky("eye", &[], || {
        let m = m.unwrap_or(n);
        if n < 0 || m < 0 {
            return Err(invalid("eye", "negative size"));
        }
        let rows: Vec<i32> = (0..n).map(|i| i + k).collect();
        let cols: Vec<i32> = (0..m).collect();
        let on = imp::equal(
            &int_array(&rows, &[n as usize, 1])?,
            &int_array(&cols, &[1, m as usize])?,
        )?;
        imp::astype(&on, dtype)
    })
}

/// The `[n, n]` identity matrix in `dtype`.
#[track_caller]
pub fn identity(n: i32, dtype: DType) -> Array {
    eye(n, None, 0, dtype)
}

/// The `offset`-th diagonal of the plane of `axis1` and `axis2`, as the last axis.
#[track_caller]
pub fn diagonal(a: &Array, offset: i32, axis1: i32, axis2: i32) -> Array {
    sticky("diagonal", &[a], || {
        let n = a.ndim();
        if n < 2 {
            return Err(invalid("diagonal", "the array has fewer than 2 dimensions"));
        }
        let (x, y) = (
            axis_of("diagonal", axis1, n)?,
            axis_of("diagonal", axis2, n)?,
        );
        if x == y {
            return Err(invalid("diagonal", "the two axes are the same"));
        }
        let mut order: Vec<usize> = (0..n).filter(|&i| i != x && i != y).collect();
        order.extend([x, y]);
        let t = imp::transpose(a, &order)?;
        let (rows, cols) = (a.dims()[x] as i64, a.dims()[y] as i64);
        let (r0, c0) = if offset >= 0 {
            (0, i64::from(offset))
        } else {
            (-i64::from(offset), 0)
        };
        let len = (rows - r0).min(cols - c0).max(0);
        let idx: Vec<i32> = (0..len)
            .map(|i| ((r0 + i) * cols + c0 + i) as i32)
            .collect();
        let mut flat: Vec<isize> = t.dims()[..n - 2].iter().map(|&d| d as isize).collect();
        flat.push((rows * cols) as isize);
        take(&imp::reshape(&t, &flat)?, &idx, n - 2)
    })
}

/// MLX's `diag`: the `k`-th diagonal of a matrix, or the matrix with a vector on its `k`-th
/// diagonal.
#[track_caller]
pub fn diag(a: &Array, k: i32) -> Array {
    match a.ndim() {
        2 => diagonal(a, k, 0, 1),
        1 => sticky("diag", &[a], || {
            let n = a.dims()[0] as i64;
            let size = (n + i64::from(k).abs()) as usize;
            let (r0, c0) = if k >= 0 {
                (0, i64::from(k))
            } else {
                (-i64::from(k), 0)
            };
            // Scatter the vector into the flat matrix (a gather from [0, a] keeps the
            // gradient: position 0 is the zero).
            let mut idx = vec![0i32; size * size];
            for i in 0..n {
                idx[((r0 + i) * size as i64 + c0 + i) as usize] = (i + 1) as i32;
            }
            let padded =
                super::concatenate(&[imp::zeros(&[1], a.dtype())?, a.clone()], 0).into_result()?;
            imp::reshape(&take(&padded, &idx, 0)?, &[size as isize, size as isize])
        }),
        _ => sticky("diag", &[a], || {
            Err(invalid("diag", "the array must have 1 or 2 dimensions"))
        }),
    }
}

/// The sum of the `offset`-th diagonal of the plane of `axis1` and `axis2`, in `a`'s dtype.
#[track_caller]
pub fn trace(a: &Array, offset: i32, axis1: i32, axis2: i32) -> Array {
    let d = diagonal(a, offset, axis1, axis2);
    sticky("trace", &[a], || {
        let d = d.into_result()?;
        let last = d.ndim() as isize - 1;
        imp::astype(&imp::sum(&d, &[last], false)?, a.dtype())
    })
}

// ── Ranges and grids ─────────────────────────────────────────────────────────

/// `num` evenly spaced values from `start` to `stop` (both included), as MLX computes them
/// (`(1 − t)·start + t·stop` in float32, `t = i / (num − 1)`), in `dtype`.
///
/// ```
/// use nuvec_core::{DType, ops};
///
/// assert_eq!(ops::linspace(0.0, 1.0, 5, DType::F32).to_vec::<f32>().unwrap(), [0.0, 0.25, 0.5, 0.75, 1.0]);
/// ```
#[track_caller]
pub fn linspace(start: f64, stop: f64, num: i32, dtype: DType) -> Array {
    sticky("linspace", &[], || {
        if num < 0 {
            return Err(invalid("linspace", "a negative number of samples"));
        }
        if num == 1 {
            return imp::astype(&crate::creation::from_data(&[start as f32], &[1])?, dtype);
        }
        let i: Vec<f32> = (0..num).map(|i| i as f32).collect();
        let f = |v: f32| crate::creation::from_data(&[v], &[]);
        let t = imp::divide(
            &crate::creation::from_data(&i, &[num as usize])?,
            &f((num - 1) as f32)?,
        )?;
        let t_bar = imp::subtract(&f(1.0)?, &t)?;
        let v = imp::add(
            &imp::multiply(&t_bar, &f(start as f32)?)?,
            &imp::multiply(&t, &f(stop as f32)?)?,
        )?;
        imp::astype(&v, dtype)
    })
}

/// Coordinate grids of the 1-D `arrays` (NumPy's `meshgrid`): `indexing` `"xy"` (Cartesian,
/// the first two axes swapped) or `"ij"` (matrix); `sparse` keeps singleton axes.
#[track_caller]
pub fn meshgrid(arrays: &[Array], sparse: bool, indexing: &str) -> Vec<Array> {
    let n = arrays.len();
    let xy = match indexing {
        "xy" => true,
        "ij" => false,
        other => {
            let e = invalid(
                "meshgrid",
                format!("indexing must be \"xy\" or \"ij\", not {other:?}"),
            );
            return vec![sticky("meshgrid", &[], || Err(e))];
        }
    };
    let lens: Vec<usize> = arrays.iter().map(Array::size).collect();
    let placed: Vec<Array> = arrays
        .iter()
        .enumerate()
        .map(|(i, x)| {
            sticky("meshgrid", &[x], || {
                let mut dims = vec![1isize; n];
                let at = if xy && n >= 2 && i < 2 { 1 - i } else { i };
                dims[at] = x.size() as isize;
                imp::reshape(x, &dims)
            })
        })
        .collect();
    if sparse {
        return placed;
    }
    let mut full = lens;
    if xy && n >= 2 {
        full.swap(0, 1);
    }
    placed
        .iter()
        .map(|x| sticky("meshgrid", &[x], || imp::broadcast_to(x, &full)))
        .collect()
}

fn window(name: &'static str, m: i32, f: impl Fn(f64) -> f64) -> Array {
    sticky(name, &[], || {
        if m < 0 {
            return Err(invalid(name, "a negative length"));
        }
        if m == 1 {
            return crate::creation::from_data(&[1.0f32], &[1]);
        }
        let v: Vec<f32> = (0..m)
            .map(|i| f(f64::from(i) / f64::from(m - 1)) as f32)
            .collect();
        crate::creation::from_data(&v, &[m as usize])
    })
}

/// The Hanning window of `m` points (NumPy's), as float32.
#[track_caller]
pub fn hanning(m: i32) -> Array {
    window("hanning", m, |t| {
        0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos()
    })
}

/// The Hamming window of `m` points (NumPy's), as float32.
#[track_caller]
pub fn hamming(m: i32) -> Array {
    window("hamming", m, |t| {
        0.54 - 0.46 * (2.0 * std::f64::consts::PI * t).cos()
    })
}

/// The Blackman window of `m` points (NumPy's), as float32.
#[track_caller]
pub fn blackman(m: i32) -> Array {
    let tau = 2.0 * std::f64::consts::PI;
    window("blackman", m, move |t| {
        0.42 - 0.5 * (tau * t).cos() + 0.08 * (2.0 * tau * t).cos()
    })
}

/// The Bartlett (triangular) window of `m` points (NumPy's), as float32.
#[track_caller]
pub fn bartlett(m: i32) -> Array {
    window("bartlett", m, |t| 1.0 - (2.0 * t - 1.0).abs())
}

// ── Products ─────────────────────────────────────────────────────────────────

/// The sum of products over `a`'s axes `axes_a` and `b`'s axes `axes_b` (paired in order),
/// as one matrix product: the result has `a`'s other axes, then `b`'s.
#[track_caller]
pub fn tensordot(a: &Array, b: &Array, axes_a: &[i32], axes_b: &[i32]) -> Array {
    sticky("tensordot", &[a, b], || {
        if axes_a.len() != axes_b.len() {
            return Err(invalid(
                "tensordot",
                "the two lists of axes differ in length",
            ));
        }
        let xa = axes_of("tensordot", Some(axes_a), a.ndim())?;
        let xb = axes_of("tensordot", Some(axes_b), b.ndim())?;
        for (&i, &j) in xa.iter().zip(&xb) {
            if a.dims()[i] != b.dims()[j] {
                return Err(invalid("tensordot", "contracted axes differ in size"));
            }
        }
        let free_a: Vec<usize> = (0..a.ndim()).filter(|d| !xa.contains(d)).collect();
        let free_b: Vec<usize> = (0..b.ndim()).filter(|d| !xb.contains(d)).collect();
        let k: usize = xa.iter().map(|&d| a.dims()[d]).product();
        let m: usize = free_a.iter().map(|&d| a.dims()[d]).product();
        let n: usize = free_b.iter().map(|&d| b.dims()[d]).product();
        let at = imp::transpose(a, &[free_a.clone(), xa.clone()].concat())?;
        let bt = imp::transpose(b, &[xb.clone(), free_b.clone()].concat())?;
        let p = imp::matmul(
            &imp::reshape(&at, &[m as isize, k as isize])?,
            &imp::reshape(&bt, &[k as isize, n as isize])?,
        )?;
        let mut dims: Vec<isize> = free_a.iter().map(|&d| a.dims()[d] as isize).collect();
        dims.extend(free_b.iter().map(|&d| b.dims()[d] as isize));
        imp::reshape(&p, &dims)
    })
}

/// [`tensordot`] over the last `n` axes of `a` and the first `n` of `b` (MLX's
/// `tensordot(a, b, axes=n)`).
#[track_caller]
pub fn tensordot_n(a: &Array, b: &Array, n: i32) -> Array {
    let n = n.max(0);
    let xa: Vec<i32> = (a.ndim() as i32 - n..a.ndim() as i32).collect();
    let xb: Vec<i32> = (0..n).collect();
    tensordot(a, b, &xa, &xb)
}

/// The inner product over the last axes (MLX's `inner`).
#[track_caller]
pub fn inner(a: &Array, b: &Array) -> Array {
    if a.ndim() == 0 || b.ndim() == 0 {
        return super::multiply(a, b);
    }
    tensordot(a, b, &[-1], &[-1])
}

/// The outer product of the flattened arrays.
#[track_caller]
pub fn outer(a: &Array, b: &Array) -> Array {
    sticky("outer", &[a, b], || {
        imp::multiply(
            &imp::reshape(a, &[a.size() as isize, 1])?,
            &imp::reshape(b, &[1, b.size() as isize])?,
        )
    })
}

/// The Kronecker product.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// let k = ops::kron(&array![[1, 2]], &array![[1], [10]]);
/// assert_eq!((k.shape(), k.to_vec::<i32>().unwrap()), (&[2, 2][..], vec![1, 2, 10, 20]));
/// ```
#[track_caller]
pub fn kron(a: &Array, b: &Array) -> Array {
    sticky("kron", &[a, b], || {
        let n = a.ndim().max(b.ndim());
        let pad = |x: &Array| -> Vec<isize> {
            let mut d = vec![1isize; n - x.ndim()];
            d.extend(x.dims().iter().map(|&v| v as isize));
            d
        };
        let (da, db) = (pad(a), pad(b));
        // a: [a0, 1, a1, 1, ...] · b: [1, b0, 1, b1, ...] -> [a0·b0, a1·b1, ...]
        let sa: Vec<isize> = da.iter().flat_map(|&d| [d, 1]).collect();
        let sb: Vec<isize> = db.iter().flat_map(|&d| [1, d]).collect();
        let p = imp::multiply(&imp::reshape(a, &sa)?, &imp::reshape(b, &sb)?)?;
        imp::reshape(
            &p,
            &da.iter().zip(&db).map(|(x, y)| x * y).collect::<Vec<_>>(),
        )
    })
}

/// `sum(conj(a) · b, axis)` (MLX's `vecdot`).
#[track_caller]
pub fn vecdot(a: &Array, b: &Array, axis: i32) -> Array {
    sticky("vecdot", &[a, b], || {
        let p = imp::multiply(&imp::conjugate(a)?, b)?;
        let d = axis_of("vecdot", axis, p.ndim())?;
        imp::sum(&p, &[d as isize], false)
    })
}

/// Einstein summation over `operands` (`"ij,jk->ik"`, with `...` and implicit outputs, as
/// NumPy's and MLX's `einsum`): pairs are contracted left to right with matrix products;
/// indices repeated within one operand take its diagonal.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// let a = array![[1.0f32, 2.0], [3.0, 4.0]];
/// assert_eq!(ops::einsum("ij,jk->ik", &[a.clone(), a.clone()]).to_vec::<f32>().unwrap(), [7.0, 10.0, 15.0, 22.0]);
/// assert_eq!(ops::einsum("ii", &[a]).item::<f32>().unwrap(), 5.0);
/// ```
#[track_caller]
pub fn einsum(subscripts: &str, operands: &[Array]) -> Array {
    let refs: Vec<&Array> = operands.iter().collect();
    sticky("einsum", &refs, || einsum_impl(subscripts, operands))
}

/// An einsum's labelled terms, output labels and label sizes.
type Parsed = (
    Vec<Vec<char>>,
    Vec<char>,
    std::collections::HashMap<char, usize>,
);

/// MLX's `einsum_path`: the contraction order `einsum` takes (the positions contracted at
/// each step, the result appended last; opt_einsum's greedy path for three operands or
/// more) and MLX's report of the naive and optimized scaling and FLOP counts.
///
/// ```
/// use nuvec_core::{ops, DType};
/// let a = ops::ones([2, 300], DType::F32);
/// let b = ops::ones([300, 400], DType::F32);
/// let c = ops::ones([400, 5], DType::F32);
/// let (path, report) = ops::einsum_path("ij,jk,kl->il", &[a, b, c]).unwrap();
/// assert_eq!(path, [vec![1, 2], vec![0, 1]]); // b · c first: the smaller intermediate
/// assert!(report.contains("Optimized FLOP count"));
/// ```
pub fn einsum_path(subscripts: &str, operands: &[Array]) -> Result<(Vec<Vec<usize>>, String)> {
    let (terms, out, size) = einsum_parse("einsum_path", subscripts, operands)?;
    let inputs: Vec<Labels> = terms.iter().map(|t| t.iter().copied().collect()).collect();
    let output: Labels = out.iter().copied().collect();
    let (naive_cost, naive_scaling) = cost_and_scaling(&inputs, &output, &size);
    let (path, cost, scaling) = if terms.len() <= 2 {
        (vec![(0..terms.len()).collect()], naive_cost, naive_scaling)
    } else {
        greedy_path(&terms, &out, &size)
    };
    let report = format!(
        "  Complete contraction:  {subscripts}\n         Naive scaling:  {naive_scaling}\n     \
         Optimized scaling:  {scaling}\n      Naive FLOP count:  {naive_cost}\n  \
         Optimized FLOP count:  {cost}\n"
    );
    Ok((path, report))
}

fn einsum_impl(subscripts: &str, operands: &[Array]) -> Result<Array> {
    let err = |m: &str| invalid("einsum", m);
    let (terms, out, size) = einsum_parse("einsum", subscripts, operands)?;
    // Each operand: take repeated indices' diagonals, sum indices nobody else needs.
    let mut items: Vec<(Vec<char>, Array)> = Vec::new();
    for (k, (t, x)) in terms.iter().zip(operands).enumerate() {
        let (mut t, mut x) = (t.clone(), x.clone());
        while let Some((i, j)) = repeated(&t) {
            x = diagonal(&x, 0, i as i32, j as i32).into_result()?;
            let c = t[i];
            t.remove(j);
            t.remove(i);
            t.push(c);
        }
        let needed = |c: char| {
            out.contains(&c)
                || terms
                    .iter()
                    .enumerate()
                    .any(|(o, u)| o != k && u.contains(&c))
        };
        let drop: Vec<isize> = (0..t.len())
            .filter(|&i| !needed(t[i]))
            .map(|i| i as isize)
            .collect();
        if !drop.is_empty() {
            x = imp::sum(&x, &drop, false)?;
            t.retain(|&c| needed(c));
        }
        items.push((t, x));
    }
    // Contract in MLX's order (its greedy path for three operands or more).
    let path = if items.len() <= 2 {
        vec![(0..items.len()).collect::<Vec<_>>()]
    } else {
        greedy_path(&terms, &out, &size).0
    };
    for positions in path {
        let mut picked = Vec::with_capacity(positions.len());
        for &p in positions.iter().rev() {
            picked.push(items.remove(p));
        }
        picked.reverse();
        let mut picked = picked.into_iter();
        let (mut t, mut x) = picked.next().ok_or_else(|| err("no operands"))?;
        let rest: Vec<(Vec<char>, Array)> = picked.collect();
        for (k, (u, y)) in rest.iter().enumerate() {
            let later = |c: char| {
                out.contains(&c)
                    || rest[k + 1..].iter().any(|(v, _)| v.contains(&c))
                    || items.iter().any(|(v, _)| v.contains(&c))
            };
            (t, x) = contract_pair((&t, &x), (u, y), later, &size)?;
        }
        items.push((t, x));
    }
    let (mut t, mut x) = items.pop().ok_or_else(|| err("no operands"))?;
    // Sum what is left over and order the output.
    let drop: Vec<isize> = (0..t.len())
        .filter(|&i| !out.contains(&t[i]))
        .map(|i| i as isize)
        .collect();
    if !drop.is_empty() {
        x = imp::sum(&x, &drop, false)?;
        t.retain(|c| out.contains(c));
    }
    let perm: Vec<usize> = out
        .iter()
        .map(|c| t.iter().position(|l| l == c))
        .collect::<Option<_>>()
        .ok_or_else(|| err("an output index was contracted"))?;
    let x = imp::transpose(&x, &perm)?;
    imp::broadcast_to(&x, &out.iter().map(|c| size[c]).collect::<Vec<_>>())
}

/// The product of two labelled operands: labels in both that `later` needs are batch
/// axes, the others are contracted; one batched matrix product.
fn contract_pair(
    (t, x): (&[char], &Array),
    (u, y): (&[char], &Array),
    later: impl Fn(char) -> bool,
    size: &std::collections::HashMap<char, usize>,
) -> Result<(Vec<char>, Array)> {
    let batch: Vec<char> = t
        .iter()
        .copied()
        .filter(|c| u.contains(c) && later(*c))
        .collect();
    let contract: Vec<char> = t
        .iter()
        .copied()
        .filter(|c| u.contains(c) && !later(*c))
        .collect();
    let only_a: Vec<char> = t.iter().copied().filter(|c| !u.contains(c)).collect();
    let only_b: Vec<char> = u.iter().copied().filter(|c| !t.contains(c)).collect();
    let place = |labels: &[char], x: &Array, order: &[&[char]]| -> Result<Array> {
        let perm: Vec<usize> = order
            .iter()
            .flat_map(|g| g.iter())
            .map(|c| labels.iter().position(|l| l == c).unwrap())
            .collect();
        let x = imp::transpose(x, &perm)?;
        let dims: Vec<usize> = order
            .iter()
            .flat_map(|g| g.iter())
            .map(|c| size[c])
            .collect();
        let x = imp::broadcast_to(&x, &dims)?;
        let groups: Vec<isize> = order
            .iter()
            .map(|g| g.iter().map(|c| size[c]).product::<usize>() as isize)
            .collect();
        imp::reshape(&x, &groups)
    };
    let a3 = place(t, x, &[&batch, &only_a, &contract])?;
    let b3 = place(u, y, &[&batch, &contract, &only_b])?;
    let p = imp::matmul(&a3, &b3)?;
    let labels: Vec<char> = [batch, only_a, only_b].concat();
    let dims: Vec<isize> = labels.iter().map(|c| size[c] as isize).collect();
    Ok((labels.clone(), imp::reshape(&p, &dims)?))
}

type Labels = std::collections::BTreeSet<char>;

/// The product of the sizes of `term`'s labels (wrapping, as MLX's `size_t`).
fn term_size<'a>(
    term: impl IntoIterator<Item = &'a char>,
    size: &std::collections::HashMap<char, usize>,
) -> usize {
    term.into_iter()
        .fold(1usize, |p, c| p.wrapping_mul(size[c]))
}

/// opt_einsum's FLOP count, as MLX's `flop_count`.
fn flop_count(
    term: &Labels,
    inner: bool,
    terms: usize,
    size: &std::collections::HashMap<char, usize>,
) -> usize {
    let factor = (terms.saturating_sub(1)).max(1) + usize::from(inner);
    term_size(term, size).wrapping_mul(factor)
}

/// The cost and scaling of contracting `inputs` at once into `output`.
fn cost_and_scaling(
    inputs: &[Labels],
    output: &Labels,
    size: &std::collections::HashMap<char, usize>,
) -> (usize, usize) {
    let all: Labels = inputs.iter().flatten().copied().collect();
    let inner = all.iter().any(|c| !output.contains(c));
    (flop_count(&all, inner, inputs.len(), size), all.len())
}

/// MLX's (opt_einsum's) greedy contraction path for `terms` into `out`: the positions
/// contracted at each step (the result appended last), its cost and its scaling.
fn greedy_path(
    terms: &[Vec<char>],
    out: &[char],
    size: &std::collections::HashMap<char, usize>,
) -> (Vec<Vec<usize>>, usize, usize) {
    let output: Labels = out.iter().copied().collect();
    let mut inputs: Vec<Labels> = terms.iter().map(|t| t.iter().copied().collect()).collect();
    let (cost_limit, _) = cost_and_scaling(&inputs, &output, size);
    let memory_limit = terms
        .iter()
        .map(|t| term_size(t, size))
        .chain([term_size(out, size)])
        .max()
        .unwrap_or(1);
    struct Candidate {
        removed: i128,
        cost: usize,
        output: Labels,
        dims: usize,
        x: usize,
        y: usize,
    }
    let mut pairs: Vec<(usize, usize)> = (0..inputs.len())
        .flat_map(|i| (i + 1..inputs.len()).map(move |j| (i, j)))
        .collect();
    let (mut path, mut cost, mut scaling) = (Vec::new(), 0usize, 0usize);
    let mut candidates: Vec<Candidate> = Vec::new();
    let n = inputs.len();
    for _ in 0..n.saturating_sub(1) {
        let add = |p1: usize,
                   p2: usize,
                   inputs: &[Labels],
                   candidates: &mut Vec<Candidate>,
                   cost: usize| {
            let both: Labels = inputs[p1].union(&inputs[p2]).copied().collect();
            let kept: Labels = both
                .iter()
                .copied()
                .filter(|c| {
                    output.contains(c)
                        || inputs
                            .iter()
                            .enumerate()
                            .any(|(i, t)| i != p1 && i != p2 && t.contains(c))
                })
                .collect();
            let new_size = term_size(&kept, size);
            if new_size > memory_limit {
                return;
            }
            let removed = term_size(&inputs[p1], size) as i128
                + term_size(&inputs[p2], size) as i128
                - new_size as i128;
            let step = flop_count(&both, both.len() > kept.len(), 2, size);
            if cost.wrapping_add(step) > cost_limit {
                return;
            }
            candidates.push(Candidate {
                removed,
                cost: step,
                output: kept,
                dims: both.len(),
                x: p1,
                y: p2,
            });
        };
        for &(p1, p2) in &pairs {
            if !inputs[p1].is_disjoint(&inputs[p2]) {
                add(p1, p2, &inputs, &mut candidates, cost);
            }
        }
        if candidates.is_empty() {
            for &(p1, p2) in &pairs {
                add(p1, p2, &inputs, &mut candidates, cost);
            }
        }
        if candidates.is_empty() {
            // The rest at once.
            let (c, sc) = cost_and_scaling(&inputs, &output, size);
            path.push((0..inputs.len()).collect());
            cost = cost.wrapping_add(c);
            scaling = scaling.max(sc);
            break;
        }
        // The largest size reduction, then the lowest cost (the first such).
        let mut best = 0;
        for (i, c) in candidates.iter().enumerate() {
            let b = &candidates[best];
            if c.removed > b.removed || (c.removed == b.removed && c.cost < b.cost) {
                best = i;
            }
        }
        let best = candidates.remove(best);
        scaling = scaling.max(best.dims);
        path.push(vec![best.x, best.y]);
        inputs.remove(best.y);
        inputs.remove(best.x);
        inputs.push(best.output);
        candidates = candidates
            .into_iter()
            .filter(|c| c.x != best.x && c.x != best.y && c.y != best.x && c.y != best.y)
            .map(|mut c| {
                c.x -= usize::from(c.x > best.x) + usize::from(c.x > best.y);
                c.y -= usize::from(c.y > best.x) + usize::from(c.y > best.y);
                c
            })
            .collect();
        pairs = (0..inputs.len() - 1)
            .map(|i| (i, inputs.len() - 1))
            .collect();
        cost = cost.wrapping_add(best.cost);
    }
    (path, cost, scaling)
}

fn einsum_parse(name: &'static str, subscripts: &str, operands: &[Array]) -> Result<Parsed> {
    let err = |m: &str| invalid(name, m);
    let s: String = subscripts.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs) = match s.split_once("->") {
        Some((l, r)) => (l.to_owned(), Some(r.to_owned())),
        None => (s.clone(), None),
    };
    let inputs: Vec<&str> = lhs.split(',').collect();
    if inputs.len() != operands.len() {
        return Err(err("the subscripts and the operands differ in number"));
    }
    // Ellipses become uppercase letters that cannot appear otherwise (subscripts are a–z).
    let mut ell_rank = 0usize;
    let mut terms: Vec<Vec<char>> = Vec::new();
    for (t, x) in inputs.iter().zip(operands) {
        let named = t.replace("...", "").chars().count();
        if t.contains("...") {
            ell_rank = ell_rank.max(x.ndim().saturating_sub(named));
        }
    }
    let ell: Vec<char> = (0..ell_rank).map(|i| (b'A' + i as u8) as char).collect();
    for (t, x) in inputs.iter().zip(operands) {
        let named = t.replace("...", "").chars().count();
        let mut v = Vec::new();
        if let Some((a, b)) = t.split_once("...") {
            v.extend(a.chars());
            let r = x.ndim().saturating_sub(named);
            v.extend(&ell[ell_rank - r..]);
            v.extend(b.chars());
        } else {
            v.extend(t.chars());
        }
        if v.len() != x.ndim() || v.iter().any(|c| !c.is_ascii_alphabetic()) {
            return Err(err("a term does not match its operand's dimensions"));
        }
        terms.push(v);
    }
    let out: Vec<char> = match &rhs {
        Some(r) => {
            let mut v = Vec::new();
            if let Some((a, b)) = r.split_once("...") {
                v.extend(a.chars());
                v.extend(&ell);
                v.extend(b.chars());
            } else {
                v.extend(r.chars());
            }
            v
        }
        None => {
            // Implicit: the ellipsis, then the indices that appear once, in order.
            let mut counts = std::collections::BTreeMap::new();
            for t in &terms {
                for &c in t {
                    if c.is_ascii_lowercase() {
                        *counts.entry(c).or_insert(0) += 1;
                    }
                }
            }
            let mut v = ell.clone();
            v.extend(counts.into_iter().filter(|&(_, n)| n == 1).map(|(c, _)| c));
            v
        }
    };
    // The size of each index, checked across operands (size-1 dimensions broadcast).
    let mut size: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (t, x) in terms.iter().zip(operands) {
        for (&c, &d) in t.iter().zip(x.dims()) {
            let e = size.entry(c).or_insert(d);
            if *e != d && *e != 1 && d != 1 {
                return Err(err("an index has two different sizes"));
            }
            *e = (*e).max(d);
        }
    }
    if out.iter().any(|c| !size.contains_key(c)) {
        return Err(err("an output index appears in no input"));
    }
    Ok((terms, out, size))
}

/// The first pair of positions holding the same label.
fn repeated(t: &[char]) -> Option<(usize, usize)> {
    (0..t.len()).find_map(|i| (i + 1..t.len()).find(|&j| t[j] == t[i]).map(|j| (i, j)))
}

// ── Statistics and search ────────────────────────────────────────────────────

/// The `n`-th discrete difference along `axis` (`a[1:] − a[:-1]`, `n` times).
#[track_caller]
pub fn diff(a: &Array, n: i32, axis: i32) -> Array {
    sticky("diff", &[a], || {
        if n < 0 {
            return Err(invalid("diff", "a negative order"));
        }
        let d = axis_of("diff", axis, a.ndim())?;
        let mut x = a.clone();
        for _ in 0..n {
            let len = x.dims()[d] as isize;
            if len == 0 {
                break;
            }
            x = imp::subtract(
                &slice_axis(&x, d, 1, len, 1)?,
                &slice_axis(&x, d, 0, len - 1, 1)?,
            )?;
        }
        Ok(x)
    })
}

/// The median over `axes` (all axes for `None`): the middle element, or the mean of the
/// two middle ones, in at least float32.
#[track_caller]
pub fn median(a: &Array, axes: Option<&[i32]>, keepdims: bool) -> Array {
    sticky("median", &[a], || {
        let ax = axes_of("median", axes, a.ndim())?;
        let n = a.ndim();
        let kept: Vec<usize> = (0..n).filter(|d| !ax.contains(d)).collect();
        let order: Vec<usize> = kept.iter().chain(&ax).copied().collect();
        let t = imp::transpose(a, &order)?;
        let mut dims: Vec<isize> = kept.iter().map(|&d| a.dims()[d] as isize).collect();
        let m: usize = ax.iter().map(|&d| a.dims()[d]).product();
        dims.push(m as isize);
        let flat = imp::reshape(&t, &dims)?;
        let last = kept.len() as i32;
        let sorted = super::sort(&flat, last).into_result()?;
        let lo = slice_axis(
            &sorted,
            kept.len(),
            ((m - 1) / 2) as isize,
            ((m - 1) / 2 + 1) as isize,
            1,
        )?;
        let hi = slice_axis(
            &sorted,
            kept.len(),
            (m / 2) as isize,
            (m / 2 + 1) as isize,
            1,
        )?;
        let mid = imp::multiply(
            &imp::add(
                &imp::astype(&lo, crate::at_least_float(a.dtype()))?,
                &imp::astype(&hi, crate::at_least_float(a.dtype()))?,
            )?,
            0.5,
        )?;
        let mid = imp::squeeze(&mid, &[kept.len() as isize])?;
        if !keepdims {
            return Ok(mid);
        }
        let shape: Vec<isize> = (0..n)
            .map(|d| {
                if ax.contains(&d) {
                    1
                } else {
                    a.dims()[d] as isize
                }
            })
            .collect();
        imp::reshape(&mid, &shape)
    })
}

/// The insertion indices of `values` into the sorted 1-D `a` (MLX's `searchsorted`): before
/// equal elements (`right = false`) or after them, as uint32.
#[track_caller]
pub fn searchsorted(a: &Array, values: &Array, right: bool) -> Array {
    sticky("searchsorted", &[a, values], || {
        if a.ndim() != 1 {
            return Err(invalid(
                "searchsorted",
                "the sorted array must have one dimension",
            ));
        }
        let mut dims: Vec<isize> = values.dims().iter().map(|&d| d as isize).collect();
        dims.push(1);
        let v = imp::reshape(values, &dims)?;
        let below = if right {
            imp::less_equal(a, &v)?
        } else {
            imp::less(a, &v)?
        };
        let last = values.ndim() as isize;
        imp::astype(&imp::sum(&below, &[last], false)?, DType::U32)
    })
}

/// The outputs of [`unique`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Unique {
    /// The sorted unique values, then `fill_value` up to `size`.
    pub values: Array,
    /// The index in the flattened input of each value's first occurrence (the first value's
    /// past the end), as uint32.
    pub index: Array,
    /// For each input element, the position of its value in `values`, as uint32.
    pub inverse: Array,
    /// How often each value occurs (0 past the end), as int32.
    pub counts: Array,
}

/// The sorted unique elements of the flattened `a` in a fixed `size` (MLX's `unique`):
/// nothing is evaluated early. Positions past the last unique value hold `fill_value` (the
/// smallest value when `None`); a smaller `size` keeps the smallest values.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// let u = ops::unique(&array![3, 1, 3, 2], 5, None);
/// assert_eq!(u.values.to_vec::<i32>().unwrap(), [1, 2, 3, 1, 1]);
/// assert_eq!(u.counts.to_vec::<i32>().unwrap(), [1, 1, 2, 0, 0]);
/// assert_eq!(u.inverse.to_vec::<u32>().unwrap(), [2, 0, 2, 1]);
/// ```
#[track_caller]
pub fn unique(a: &Array, size: i32, fill_value: Option<f64>) -> Unique {
    let run = || -> Result<[Array; 4]> {
        if size < 0 {
            return Err(invalid("unique", "a negative size"));
        }
        let n = a.size();
        let size = size as usize;
        let flat = imp::reshape(a, &[n as isize])?;
        let order = super::argsort(&flat, 0).into_result()?;
        let order = imp::astype(&order, DType::I32)?;
        let sorted = super::take_along_axis(&flat, &order, 0).into_result()?;
        // first[i]: sorted[i] starts a new value; group[i]: its value's position.
        let first = if n == 0 {
            imp::astype(&imp::zeros(&[0], DType::Bool)?, DType::Bool)?
        } else {
            let head = crate::creation::from_data(&[true], &[1])?;
            let next = imp::not_equal(
                &slice_axis(&sorted, 0, 1, n as isize, 1)?,
                &slice_axis(&sorted, 0, 0, n as isize - 1, 1)?,
            )?;
            super::concatenate(&[head, next], 0).into_result()?
        };
        let group = imp::subtract(
            &super::cumsum(&imp::astype(&first, DType::I32)?, 0).into_result()?,
            1,
        )?;
        // Positions past `size` go to a spare slot that is cut off.
        let slot = imp::minimum(&group, size as f64)?;
        let target = imp::where_(&first, &slot, size as f64)?;
        let target = imp::astype(&target, DType::I32)?;
        let fill = match fill_value {
            Some(v) => imp::astype(&crate::creation::from_data(&[v], &[1])?, a.dtype())?,
            None if n > 0 => slice_axis(&sorted, 0, 0, 1, 1)?,
            None => imp::zeros(&[1], a.dtype())?,
        };
        let values = imp::broadcast_to(&fill, &[size + 1])?;
        let values = super::put_along_axis(&values, &target, &sorted, 0).into_result()?;
        // Past the end: the first value's index (MLX).
        let first_index = if n > 0 {
            slice_axis(&order, 0, 0, 1, 1)?
        } else {
            imp::zeros(&[1], DType::I32)?
        };
        let index = imp::broadcast_to(&first_index, &[size + 1])?;
        let index = super::put_along_axis(&index, &target, &order, 0).into_result()?;
        let counts = imp::zeros(&[size + 1], DType::I32)?;
        let ones = imp::astype(&imp::ones(&[n], DType::I32)?, DType::I32)?;
        let counts =
            super::scatter_add_along_axis(&counts, &imp::astype(&slot, DType::I32)?, &ones, 0)
                .into_result()?;
        let cut = |x: &Array| slice_axis(x, 0, 0, size as isize, 1);
        // inverse[order[i]] = group[i], clamped to the last kept entry.
        let clamped = imp::minimum(&group, (size.max(1) - 1) as f64)?;
        let inverse = imp::zeros(&[n], DType::I32)?;
        let inverse =
            super::put_along_axis(&inverse, &order, &imp::astype(&clamped, DType::I32)?, 0)
                .into_result()?;
        let inverse = imp::reshape(
            &inverse,
            &a.dims().iter().map(|&d| d as isize).collect::<Vec<_>>(),
        )?;
        Ok([
            cut(&values)?,
            imp::astype(&cut(&index)?, DType::U32)?,
            imp::astype(&inverse, DType::U32)?,
            cut(&counts)?,
        ])
    };
    let [values, index, inverse, counts] = match run() {
        Ok(v) => v,
        Err(e) => {
            let fail = || sticky("unique", &[a], || Err(e.clone()));
            [fail(), fail(), fail(), fail()]
        }
    };
    Unique {
        values,
        index,
        inverse,
        counts,
    }
}

/// A view of `a`'s elements (in row-major order) with `shape`, element `strides` and
/// `offset` (MLX's `as_strided`), gathered so that it is safe for any strides.
#[track_caller]
pub fn as_strided(a: &Array, shape: &[usize], strides: &[i64], offset: i64) -> Array {
    sticky("as_strided", &[a], || {
        if shape.len() != strides.len() {
            return Err(invalid(
                "as_strided",
                "the shape and the strides differ in length",
            ));
        }
        let count: usize = shape.iter().product();
        let mut idx = vec![0i64; count];
        for (k, slot) in idx.iter_mut().enumerate() {
            let mut rem = k;
            let mut at = offset;
            for (d, &n) in shape.iter().enumerate().rev() {
                at += (rem % n.max(1)) as i64 * strides[d];
                rem /= n.max(1);
            }
            *slot = at;
        }
        if idx.iter().any(|&i| i < 0 || i as usize >= a.size().max(1)) && count > 0 {
            return Err(invalid("as_strided", "the view reaches outside the array"));
        }
        let idx: Vec<i32> = idx.into_iter().map(|i| i as i32).collect();
        let flat = imp::reshape(a, &[a.size() as isize])?;
        imp::reshape(
            &take(&flat, &idx, 0)?,
            &shape.iter().map(|&d| d as isize).collect::<Vec<_>>(),
        )
    })
}

/// An array of `shape` and `dtype` (MLX's `empty`; nuvec's are zeros).
#[track_caller]
pub fn empty(shape: &[usize], dtype: DType) -> Array {
    sticky("empty", &[], || imp::zeros(shape, dtype))
}

/// An array of `a`'s shape and dtype (MLX's `empty_like`; zeros).
#[track_caller]
pub fn empty_like(a: &Array) -> Array {
    sticky("empty_like", &[a], || imp::zeros(a.dims(), a.dtype()))
}

/// # Shape conveniences (MLX's array methods)
impl Array {
    /// `self` with axis `source` moved to `destination`; see [`moveaxis`].
    #[track_caller]
    pub fn moveaxis(&self, source: i32, destination: i32) -> Array {
        moveaxis(self, source, destination)
    }

    /// `self` with two axes exchanged; see [`swapaxes`].
    #[track_caller]
    pub fn swapaxes(&self, axis1: i32, axis2: i32) -> Array {
        swapaxes(self, axis1, axis2)
    }

    /// The `offset`-th diagonal of a plane; see [`diagonal`].
    #[track_caller]
    pub fn diagonal(&self, offset: i32, axis1: i32, axis2: i32) -> Array {
        diagonal(self, offset, axis1, axis2)
    }

    /// MLX's `diag`; see [`diag`].
    #[track_caller]
    pub fn diag(&self, k: i32) -> Array {
        diag(self, k)
    }

    /// The elements reversed along `axes`; see [`flip`].
    #[track_caller]
    pub fn flip(&self, axes: Option<&[i32]>) -> Array {
        flip(self, axes)
    }
}

/// MLX's dtype categories (`mx.floating`, `mx.integer`, …), for [`issubdtype`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DTypeCategory {
    /// Every dtype.
    Generic,
    /// Every dtype but bool.
    Number,
    /// Integers.
    Integer,
    /// Signed integers.
    SignedInteger,
    /// Unsigned integers.
    UnsignedInteger,
    /// Floating point and complex.
    Inexact,
    /// Floating point.
    Floating,
    /// Complex.
    ComplexFloating,
}

/// Whether `dtype` belongs to `category` (MLX's `issubdtype`).
pub fn issubdtype(dtype: DType, category: DTypeCategory) -> bool {
    use crate::Category as C;
    let c = dtype.category();
    match category {
        DTypeCategory::Generic => true,
        DTypeCategory::Number => c != C::Bool,
        DTypeCategory::Integer => matches!(c, C::Signed | C::Unsigned),
        DTypeCategory::SignedInteger => c == C::Signed,
        DTypeCategory::UnsignedInteger => c == C::Unsigned,
        DTypeCategory::Inexact => matches!(c, C::Float | C::Complex),
        DTypeCategory::Floating => c == C::Float,
        DTypeCategory::ComplexFloating => c == C::Complex,
    }
}

/// The dtype the operands promote to (MLX's `result_type` over dtypes).
pub fn result_type(dtypes: &[DType]) -> Option<DType> {
    dtypes.iter().copied().reduce(crate::promote_types)
}

/// Whether `from` promotes to `to` without changing `to` (MLX's `can_cast`).
pub fn can_cast(from: DType, to: DType) -> bool {
    crate::promote_types(from, to) == to
}

/// `a` with axes `start_axis..=end_axis` merged into one (MLX's `flatten`), as a view where
/// the layout permits.
///
/// ```
/// use nuvec_core::{ops, random, DType};
///
/// let x = random::uniform(0.0, 1.0, [2, 3, 4], DType::F32, None);
/// assert_eq!(ops::flatten(&x, 1, -1).shape(), [2, 12]);
/// ```
#[track_caller]
pub fn flatten(a: &Array, start_axis: i32, end_axis: i32) -> Array {
    sticky("flatten", &[a], || {
        if a.ndim() == 0 {
            return imp::reshape(a, &[1]);
        }
        let n = a.ndim();
        let (s, e) = (
            axis_of("flatten", start_axis, n)?,
            axis_of("flatten", end_axis, n)?,
        );
        if s > e {
            return Err(invalid("flatten", "the start axis is after the end axis"));
        }
        let mut dims: Vec<isize> = a.dims()[..s].iter().map(|&d| d as isize).collect();
        dims.push(a.dims()[s..=e].iter().product::<usize>() as isize);
        dims.extend(a.dims()[e + 1..].iter().map(|&d| d as isize));
        imp::reshape(a, &dims)
    })
}

/// The order of a [`matrix_norm`] (MLX's `ord`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Norm {
    /// MLX's `None`: the 2-norm of vectors, the Frobenius norm of matrices.
    Default,
    /// A vector `p`-norm (`f64::INFINITY` for the largest magnitude, `0` for the count of
    /// non-zeros), or the matrix norms 1, −1, ∞ and −∞ (column and row sums).
    P(f64),
    /// The Frobenius norm.
    Fro,
    /// The nuclear norm (the sum of the singular values).
    Nuc,
}

/// Vector or matrix norms (MLX's `matrix_norm`, `linalg.norm`): over `axes` (one axis: a
/// vector norm; two: a matrix norm; `None`: the whole 1-D or 2-D array, or the 2-norm of
/// the flattened array for [`Norm::Default`]), in at least float32.
///
/// ```
/// use nuvec_core::{array, ops};
///
/// let m = array![[3.0f32, -4.0], [0.0, 0.0]];
/// assert_eq!(ops::matrix_norm(&m, ops::Norm::Default, None, false).item::<f32>().unwrap(), 5.0);
/// assert_eq!(ops::matrix_norm(&m, ops::Norm::P(1.0), None, false).item::<f32>().unwrap(), 4.0);
/// ```
#[track_caller]
pub fn matrix_norm(a: &Array, ord: Norm, axes: Option<&[i32]>, keepdims: bool) -> Array {
    sticky("matrix_norm", &[a], || {
        let err = |m: &str| invalid("matrix_norm", m);
        let x = imp::abs(a)?;
        let x = imp::astype(&x, crate::at_least_float(x.dtype()))?;
        let axes: Vec<usize> = match axes {
            Some(list) => axes_of("matrix_norm", Some(list), a.ndim())?,
            None if ord == Norm::Default => {
                let all: Vec<isize> = (0..a.ndim() as isize).collect();
                return imp::sqrt(&imp::sum(&imp::multiply(&x, &x)?, &all, keepdims)?);
            }
            None if matches!(a.ndim(), 1 | 2) => (0..a.ndim()).collect(),
            None => return Err(err("without axes, the array must have 1 or 2 dimensions")),
        };
        let sax: Vec<isize> = axes.iter().map(|&d| d as isize).collect();
        match (axes.len(), ord) {
            (1, Norm::Default) => imp::sqrt(&imp::sum(&imp::multiply(&x, &x)?, &sax, keepdims)?),
            (1, Norm::P(p)) if p == f64::INFINITY => imp::max(&x, &sax, keepdims),
            (1, Norm::P(p)) if p == f64::NEG_INFINITY => imp::min(&x, &sax, keepdims),
            (1, Norm::P(0.0)) => imp::astype(
                &imp::sum(&imp::not_equal(&x, 0)?, &sax, keepdims)?,
                x.dtype(),
            ),
            (1, Norm::P(1.0)) => imp::sum(&x, &sax, keepdims),
            (1, Norm::P(p)) => {
                let s = imp::sum(
                    &imp::binary(
                        crate::primitive::BinaryOp::Power,
                        &x,
                        &imp::astype(&crate::creation::from_data(&[p], &[])?, x.dtype())?,
                    )?,
                    &sax,
                    keepdims,
                )?;
                imp::binary(
                    crate::primitive::BinaryOp::Power,
                    &s,
                    &imp::astype(&crate::creation::from_data(&[1.0 / p], &[])?, x.dtype())?,
                )
            }
            (1, _) => Err(err("Frobenius and nuclear norms are matrix norms")),
            (2, Norm::Default | Norm::Fro) => {
                imp::sqrt(&imp::sum(&imp::multiply(&x, &x)?, &sax, keepdims)?)
            }
            (2, Norm::P(p)) if p.abs() == 1.0 || p.is_infinite() => {
                // 1: the column sums (over the rows axis); ∞: the row sums.
                let (sum_over, then) = if p.abs() == 1.0 {
                    (axes[0], axes[1])
                } else {
                    (axes[1], axes[0])
                };
                let s = imp::sum(&x, &[sum_over as isize], true)?;
                let r = if p > 0.0 {
                    imp::max(&s, &[then as isize], true)?
                } else {
                    imp::min(&s, &[then as isize], true)?
                };
                if keepdims {
                    Ok(r)
                } else {
                    imp::squeeze(
                        &r,
                        &[axes[0].min(axes[1]) as isize, axes[0].max(axes[1]) as isize],
                    )
                }
            }
            (2, Norm::P(_) | Norm::Nuc) => {
                // The singular values of the matrices in the plane of the two axes.
                let n = a.ndim();
                let (r, c) = (axes[0], axes[1]);
                let mut order: Vec<usize> = (0..n).filter(|&d| d != r && d != c).collect();
                order.extend([r, c]);
                let t = imp::transpose(&imp::astype(a, crate::at_least_float(a.dtype()))?, &order)?;
                let s = crate::linalg::svd_values(&t).into_result()?;
                let last = s.ndim() as isize - 1;
                let v = match ord {
                    Norm::Nuc => imp::sum(&s, &[last], false)?,
                    Norm::P(2.0) => imp::max(&s, &[last], false)?,
                    Norm::P(-2.0) => imp::min(&s, &[last], false)?,
                    _ => {
                        return Err(err(
                            "matrix norms are 1, −1, 2, −2, ∞, −∞, Frobenius or nuclear",
                        ));
                    }
                };
                if !keepdims {
                    return Ok(v);
                }
                let shape: Vec<isize> = (0..n)
                    .map(|d| {
                        if axes.contains(&d) {
                            1
                        } else {
                            a.dims()[d] as isize
                        }
                    })
                    .collect();
                imp::reshape(&v, &shape)
            }
            _ => Err(err("one or two axes")),
        }
    })
}

/// Whether `dtype` is of `kind` (MLX's `isdtype`): `"bool"`, `"signed integer"`,
/// `"unsigned integer"`, `"integral"`, `"real floating"`, `"complex floating"` or
/// `"numeric"`. An unknown kind is false.
pub fn isdtype(dtype: DType, kind: &str) -> bool {
    use crate::Category as C;
    let c = dtype.category();
    match kind {
        "bool" => c == C::Bool,
        "signed integer" => c == C::Signed,
        "unsigned integer" => c == C::Unsigned,
        "integral" => matches!(c, C::Signed | C::Unsigned),
        "real floating" => c == C::Float,
        "complex floating" => c == C::Complex,
        "numeric" => c != C::Bool,
        _ => false,
    }
}
