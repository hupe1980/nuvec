//! Matrix multiplication over strided operands, backed by the pure-Rust `gemm` crate.

use half::f16;

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
    /// Asserts that every element of a `rows × cols` matrix lies inside `data`.
    fn assert_in_bounds(&self, rows: usize, cols: usize) {
        if rows == 0 || cols == 0 {
            return;
        }
        let span = |n: usize, stride: isize| (n as isize - 1) * stride;
        let (r, c) = (span(rows, self.row_stride), span(cols, self.col_stride));
        let lo = self.offset + r.min(0) + c.min(0);
        let hi = self.offset + r.max(0) + c.max(0);
        assert!(
            lo >= 0 && (hi as usize) < self.data.len(),
            "matrix view reaches outside its data"
        );
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
impl_gemm!(
    f16,
    gemm::f16,
    |acc, a, b| f16::from_f32(acc.to_f32() + a.to_f32() * b.to_f32()),
    gemm::f16::ONE
);
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
/// # Panics
///
/// If `out.len() != m·n` or a view reaches outside its data.
pub fn matmul<T: Gemm>(
    m: usize,
    n: usize,
    k: usize,
    a: MatView<'_, T>,
    b: MatView<'_, T>,
    out: &mut [T],
) {
    assert_eq!(out.len(), m * n, "output length does not match m·n");
    a.assert_in_bounds(m, k);
    b.assert_in_bounds(k, n);
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        out.fill(T::default());
        return;
    }
    #[cfg(not(miri))]
    // SAFETY: both views were checked to be in bounds for their extents above, and `out`
    // holds exactly `m·n` elements.
    unsafe {
        T::gemm(m, n, k, out, a, b);
    }
    #[cfg(miri)]
    naive(m, n, k, a, b, out);
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
        matmul(m, n, k, a, b, &mut fast);
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
        matmul(1, 1, 2, av, bv, &mut out);
        assert_eq!(out, [[4.0, 6.0]]); // (1+2i)·3 + i·(-i) = 4 + 6i
        let mut zeros = [9.0f32; 4];
        let empty = MatView {
            data: &[0.0f32; 0],
            offset: 0,
            row_stride: 0,
            col_stride: 0,
        };
        matmul(2, 2, 0, empty, empty, &mut zeros);
        assert_eq!(zeros, [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "outside its data")]
    fn rejects_out_of_bounds_views() {
        let data = [0.0f32; 4];
        let v = MatView {
            data: &data,
            offset: 0,
            row_stride: 2,
            col_stride: 1,
        };
        let mut out = [0.0f32; 9];
        matmul(3, 3, 2, v, v, &mut out);
    }
}
