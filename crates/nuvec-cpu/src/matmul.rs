//! Matrix multiplication over strided operands, backed by the pure-Rust `gemm` crate.

use std::fmt;

use half::f16;

/// Why [`matmul`] refused its operands (before touching any memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatmulError {
    /// `m·n`, or a view's extent, overflows.
    Overflow,
    /// The output does not hold `m·n` elements.
    OutputLength,
    /// A view reaches outside its data.
    OutOfBounds,
}

impl fmt::Display for MatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Overflow => "matmul: sizes overflow",
            Self::OutputLength => "matmul: output length does not match m·n",
            Self::OutOfBounds => "matmul: matrix view reaches outside its data",
        })
    }
}

impl std::error::Error for MatmulError {}

/// A strided matrix: element `(i, j)` lives at `data[offset + i·row_stride + j·col_stride]`.
#[derive(Debug, Clone, Copy)]
pub struct MatView<'a, T> {
    /// The underlying storage.
    pub data: &'a [T],
    /// Position of element `(0, 0)`, in elements.
    pub offset: isize,
    /// Step between rows, in elements. May be zero or negative.
    pub row_stride: isize,
    /// Step between columns, in elements. May be zero or negative.
    pub col_stride: isize,
}

impl<T> MatView<'_, T> {
    /// Checks that every element of a `rows × cols` matrix lies inside `data`, with
    /// checked arithmetic throughout.
    fn check_in_bounds(&self, rows: usize, cols: usize) -> Result<(), MatmulError> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let span = |n: usize, stride: isize| {
            isize::try_from(n - 1)
                .ok()
                .and_then(|n| n.checked_mul(stride))
                .ok_or(MatmulError::Overflow)
        };
        let (r, c) = (span(rows, self.row_stride)?, span(cols, self.col_stride)?);
        let reach = |a: isize, b: isize| {
            self.offset
                .checked_add(a)
                .and_then(|x| x.checked_add(b))
                .ok_or(MatmulError::Overflow)
        };
        let lo = reach(r.min(0), c.min(0))?;
        let hi = reach(r.max(0), c.max(0))?;
        if lo >= 0 && (hi as usize) < self.data.len() {
            Ok(())
        } else {
            Err(MatmulError::OutOfBounds)
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An element type with a GEMM kernel: `f32`, `f64`, `f16`, and `[f32; 2]` (complex64 as
/// `(re, im)`).
pub trait Gemm: Copy + Default + sealed::Sealed + 'static {
    /// `acc + a·b`, used by the reference loop (and under Miri).
    fn mul_add(acc: Self, a: Self, b: Self) -> Self;

    /// Calls `gemm::gemm` for `dst (row-major m×n) = lhs · rhs`.
    ///
    /// # Safety
    ///
    /// Every element reachable through the views must lie inside their data, and `dst` must
    /// hold `m·n` elements.
    #[cfg(not(miri))]
    unsafe fn gemm(
        m: usize,
        n: usize,
        k: usize,
        dst: &mut [Self],
        a: MatView<'_, Self>,
        b: MatView<'_, Self>,
    );
}

macro_rules! impl_gemm {
    ($t:ty, $g:ty, $mul_add:expr, $one:expr) => {
        impl sealed::Sealed for $t {}
        impl Gemm for $t {
            fn mul_add(acc: Self, a: Self, b: Self) -> Self {
                let f: fn(Self, Self, Self) -> Self = $mul_add;
                f(acc, a, b)
            }

            #[cfg(not(miri))]
            unsafe fn gemm(
                m: usize,
                n: usize,
                k: usize,
                dst: &mut [Self],
                a: MatView<'_, Self>,
                b: MatView<'_, Self>,
            ) {
                let lhs = a.data.as_ptr().wrapping_offset(a.offset).cast::<$g>();
                let rhs = b.data.as_ptr().wrapping_offset(b.offset).cast::<$g>();
                // SAFETY: the caller guarantees that every element reachable through `a`
                // and `b` is in bounds and that `dst` holds `m·n` elements, written with
                // row stride `n` and column stride 1. `$t` and `$g` have the same size and
                // layout. `read_dst = false`, so `dst` is only written.
                unsafe {
                    gemm::gemm::<$g>(
                        m,
                        n,
                        k,
                        dst.as_mut_ptr().cast::<$g>(),
                        1,
                        n as isize,
                        false,
                        lhs,
                        a.col_stride,
                        a.row_stride,
                        rhs,
                        b.col_stride,
                        b.row_stride,
                        $one,
                        $one,
                        false,
                        false,
                        false,
                        gemm::Parallelism::Rayon(0),
                    );
                }
            }
        }
    };
}

impl_gemm!(f32, f32, |acc, a, b| acc + a * b, 1.0);
impl_gemm!(f64, f64, |acc, a, b| acc + a * b, 1.0);

impl sealed::Sealed for f16 {}
impl Gemm for f16 {
    fn mul_add(acc: Self, a: Self, b: Self) -> Self {
        f16::from_f32(acc.to_f32() + a.to_f32() * b.to_f32())
    }

    /// As MLX: widened to `f32`, multiplied with `f32` accumulation, rounded once (the
    /// `gemm` crate's f16 kernel accumulates in f16, losing bits over a long `k`).
    #[cfg(not(miri))]
    unsafe fn gemm(
        m: usize,
        n: usize,
        k: usize,
        dst: &mut [Self],
        a: MatView<'_, Self>,
        b: MatView<'_, Self>,
    ) {
        let widen = |v: MatView<'_, Self>, rows: usize, cols: usize| -> Vec<f32> {
            (0..rows)
                .flat_map(|i| {
                    (0..cols).map(move |j| {
                        let at = v.offset + i as isize * v.row_stride + j as isize * v.col_stride;
                        v.data[at as usize].to_f32()
                    })
                })
                .collect()
        };
        let (a32, b32) = (widen(a, m, k), widen(b, k, n));
        let view = |data, cols: usize| MatView {
            data,
            offset: 0,
            row_stride: cols as isize,
            col_stride: 1,
        };
        let mut out = vec![0.0f32; m * n];
        // SAFETY: `a32` (`m × k`) and `b32` (`k × n`) are row-contiguous and hold exactly
        // the elements the views reach, and `out` holds `m·n` elements.
        unsafe { f32::gemm(m, n, k, &mut out, view(&a32, k), view(&b32, n)) };
        for (d, v) in dst.iter_mut().zip(out) {
            *d = f16::from_f32(v);
        }
    }
}
impl_gemm!(
    [f32; 2],
    gemm::c32,
    |acc, a, b| [
        acc[0] + a[0] * b[0] - a[1] * b[1],
        acc[1] + a[0] * b[1] + a[1] * b[0]
    ],
    gemm::c32::new(1.0, 0.0)
);

/// `out = a · b` for an `m × k` matrix `a` and a `k × n` matrix `b`; `out` is row-major
/// `m × n`. Operands may have any strides (transposed, reversed, broadcast).
///
/// # Errors
///
/// [`MatmulError`] if `m·n` or a view's extent overflows, `out.len() != m·n`, or a view
/// reaches outside its data; nothing is written then.
pub fn matmul<T: Gemm>(
    m: usize,
    n: usize,
    k: usize,
    a: MatView<'_, T>,
    b: MatView<'_, T>,
    out: &mut [T],
) -> Result<(), MatmulError> {
    let size = m.checked_mul(n).ok_or(MatmulError::Overflow)?;
    if out.len() != size {
        return Err(MatmulError::OutputLength);
    }
    a.check_in_bounds(m, k)?;
    b.check_in_bounds(k, n)?;
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        out.fill(T::default());
        return Ok(());
    }
    #[cfg(not(miri))]
    // SAFETY: both views were checked to be in bounds for their extents above (with
    // checked arithmetic), and `out` holds exactly `m·n` elements.
    unsafe {
        T::gemm(m, n, k, out, a, b);
    }
    #[cfg(miri)]
    naive(m, n, k, a, b, out);
    Ok(())
}

/// The textbook triple loop: the reference for tests, and the kernel under Miri (which
/// cannot execute SIMD intrinsics).
#[cfg_attr(not(any(miri, test)), allow(dead_code))]
fn naive<T: Gemm>(
    m: usize,
    n: usize,
    k: usize,
    a: MatView<'_, T>,
    b: MatView<'_, T>,
    out: &mut [T],
) {
    let at = |v: &MatView<'_, T>, i: usize, j: usize| {
        v.data[(v.offset + i as isize * v.row_stride + j as isize * v.col_stride) as usize]
    };
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).fold(T::default(), |acc, p| {
                T::mul_add(acc, at(&a, i, p), at(&b, p, j))
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive_on_strided_operands() {
        let (m, n, k) = (7, 5, 9);
        let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.37).sin()).collect();
        // `a` read transposed from the middle, `b` with a negative row stride.
        let a = MatView {
            data: &data,
            offset: 10,
            row_stride: 1,
            col_stride: 11,
        };
        let b = MatView {
            data: &data,
            offset: 150,
            row_stride: -12,
            col_stride: 2,
        };
        let (mut fast, mut slow) = (vec![0.0; m * n], vec![0.0; m * n]);
        matmul(m, n, k, a, b, &mut fast).unwrap();
        naive(m, n, k, a, b, &mut slow);
        for (x, y) in fast.iter().zip(&slow) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    #[test]
    fn complex_and_zero_k() {
        let a = [[1.0f32, 2.0], [0.0, 1.0]]; // 1x2: [1+2i, i]
        let b = [[3.0f32, 0.0], [0.0, -1.0]]; // 2x1: [3, -i]
        let mut out = [[0.0f32; 2]; 1];
        let av = MatView {
            data: &a,
            offset: 0,
            row_stride: 2,
            col_stride: 1,
        };
        let bv = MatView {
            data: &b,
            offset: 0,
            row_stride: 1,
            col_stride: 1,
        };
        matmul(1, 1, 2, av, bv, &mut out).unwrap();
        assert_eq!(out, [[4.0, 6.0]]); // (1+2i)·3 + i·(-i) = 4 + 6i
        let mut zeros = [9.0f32; 4];
        let empty = MatView {
            data: &[0.0f32; 0],
            offset: 0,
            row_stride: 0,
            col_stride: 0,
        };
        matmul(2, 2, 0, empty, empty, &mut zeros).unwrap();
        assert_eq!(zeros, [0.0; 4]);
    }

    #[test]
    fn rejects_out_of_bounds_views() {
        let data = [0.0f32; 4];
        let v = MatView {
            data: &data,
            offset: 0,
            row_stride: 2,
            col_stride: 1,
        };
        let mut out = [0.0f32; 9];
        assert_eq!(
            matmul(3, 3, 2, v, v, &mut out),
            Err(MatmulError::OutOfBounds)
        );
    }

    #[test]
    fn overflowing_sizes_and_wrong_outputs_are_errors() {
        // Zero strides keep every view in bounds, so only `m·n` can catch this: it wraps to
        // 0 = `out.len()`, and the kernel would write 2^64 elements.
        let data = [1.0f32];
        let v = MatView {
            data: &data,
            offset: 0,
            row_stride: 0,
            col_stride: 0,
        };
        let mut out = [7.0f32; 0];
        assert_eq!(
            matmul(1 << 33, 1 << 31, 1, v, v, &mut out),
            Err(MatmulError::Overflow)
        );
        let mut out = [7.0f32; 5];
        assert_eq!(
            matmul(2, 3, 1, v, v, &mut out),
            Err(MatmulError::OutputLength)
        );
        assert_eq!(out, [7.0; 5], "written before the check");
        // A stride span past `isize`.
        let far = MatView {
            data: &data,
            offset: 0,
            row_stride: isize::MAX,
            col_stride: 0,
        };
        let mut out = [0.0f32; 3];
        assert_eq!(
            matmul(3, 1, 1, far, v, &mut out),
            Err(MatmulError::Overflow)
        );
        let negative = MatView {
            data: &data,
            offset: 0,
            row_stride: isize::MIN,
            col_stride: 0,
        };
        assert_eq!(
            matmul(3, 1, 1, negative, v, &mut out),
            Err(MatmulError::Overflow)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "half's f16 conversions use inline assembly")]
    fn f16_accumulates_like_f32() {
        // k = 4096: f16 accumulation would lose about log2(4096) bits; MLX accumulates in
        // f32 and rounds once.
        let (m, n, k) = (4, 3, 4096);
        let a: Vec<f16> = (0..m * k)
            .map(|i| f16::from_f32(((i as f32) * 0.618).sin()))
            .collect();
        let b: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32(((i as f32) * 0.377).cos()))
            .collect();
        let av = MatView {
            data: &a,
            offset: 0,
            row_stride: k as isize,
            col_stride: 1,
        };
        let bv = MatView {
            data: &b,
            offset: 0,
            row_stride: n as isize,
            col_stride: 1,
        };
        let mut out = vec![f16::ZERO; m * n];
        matmul(m, n, k, av, bv, &mut out).unwrap();
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k)
                    .map(|p| a[i * k + p].to_f32() * b[p * n + j].to_f32())
                    .sum();
                let got = out[i * n + j].to_f32();
                // One f16 rounding of the result, plus f32 summation-order noise.
                let tol = want.abs() * 2.0f32.powi(-10) + 1e-3;
                assert!((got - want).abs() <= tol, "({i}, {j}): {got} vs {want}");
            }
        }
    }
}
