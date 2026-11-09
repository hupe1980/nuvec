//! Dense decompositions, inverses and solves of one matrix (`faer`), in MLX's conventions.
//!
//! Every function takes a row-major `m × n` matrix and writes row-major outputs; batching is
//! the caller's (one call per matrix). Errors are messages for the caller to wrap.

use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Mat, MatRef, Par, Side};

#[cfg(all(feature = "accelerate", target_vendor = "apple"))]
use crate::lapack;

/// With `accelerate`, Accelerate's LAPACK; otherwise none (the `faer` path).
#[cfg(all(feature = "accelerate", target_vendor = "apple"))]
trait Backend: lapack::Lapack {}
#[cfg(all(feature = "accelerate", target_vendor = "apple"))]
impl<T: lapack::Lapack> Backend for T {}
#[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
trait Backend {}
#[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
impl<T> Backend for T {}

/// Whether the decompositions call Accelerate's LAPACK (feature `accelerate`).
pub const ACCELERATE: bool = cfg!(all(feature = "accelerate", target_vendor = "apple"));

/// The element types the decompositions accept.
#[allow(private_bounds)]
pub trait Real: faer::traits::RealField + Copy + Backend {
    /// The value as float32 (the complex outputs are complex64, as MLX's).
    fn to_f32(self) -> f32;
}
impl Real for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}
impl Real for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

fn view<T: Real>(a: &[T], m: usize, n: usize) -> MatRef<'_, T> {
    MatRef::from_row_major_slice(a, m, n)
}

fn write<T: Real>(src: MatRef<'_, T>, out: &mut [T]) {
    let n = src.ncols();
    for i in 0..src.nrows() {
        for j in 0..n {
            out[i * n + j] = src[(i, j)];
        }
    }
}

/// Large matrices use the thread pool; small ones would only pay its overhead.
fn par(n: usize) -> Par {
    if n >= 192 { Par::rayon(0) } else { Par::Seq }
}

/// Sets `faer`'s parallelism for the decompositions that read it (only speed depends on it).
fn threads(n: usize) {
    faer::set_global_parallelism(par(n));
}

/// `A = Q R` with the reduced `Q` (`m × k`) and `R` (`k × n`), `k = min(m, n)`.
pub fn qr<T: Real>(a: &[T], m: usize, n: usize, q: &mut [T], r: &mut [T]) {
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    return lapack::qr(a, m, n, q, r);
    #[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
    qr_faer(a, m, n, q, r)
}

#[cfg_attr(all(feature = "accelerate", target_vendor = "apple"), allow(dead_code))]
fn qr_faer<T: Real>(a: &[T], m: usize, n: usize, q: &mut [T], r: &mut [T]) {
    threads(m.max(n));
    let k = m.min(n);
    let f = view(a, m, n).qr();
    let full_q = f.compute_thin_Q();
    write(full_q.as_ref().subcols(0, k), q);
    let rr = f.thin_R();
    // `thin_R` is `k × n` (upper triangular).
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = if j >= i { rr[(i, j)] } else { T::zero_impl() };
        }
    }
}

/// The singular values (descending) and, with `uv`, the full `U` (`m × m`) and `Vᵀ`
/// (`n × n`).
pub fn svd<T: Real>(
    a: &[T],
    m: usize,
    n: usize,
    uv: Option<(&mut [T], &mut [T])>,
    s: &mut [T],
) -> Result<(), String> {
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    return lapack::svd(a, m, n, uv, s);
    #[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
    svd_faer(a, m, n, uv, s)
}

#[cfg_attr(all(feature = "accelerate", target_vendor = "apple"), allow(dead_code))]
fn svd_faer<T: Real>(
    a: &[T],
    m: usize,
    n: usize,
    uv: Option<(&mut [T], &mut [T])>,
    s: &mut [T],
) -> Result<(), String> {
    threads(m.max(n));
    let a = view(a, m, n);
    match uv {
        None => {
            let v = a.singular_values().map_err(|e| format!("{e:?}"))?;
            s.copy_from_slice(&v);
        }
        Some((u, vt)) => {
            let f = a.svd().map_err(|e| format!("{e:?}"))?;
            write(f.U(), u);
            let v = f.V();
            for i in 0..n {
                for j in 0..n {
                    vt[i * n + j] = v[(j, i)];
                }
            }
            let d = f.S().column_vector();
            for (i, x) in s.iter_mut().enumerate() {
                *x = d[i];
            }
        }
    }
    Ok(())
}

/// The eigenvalues (ascending) and, with `v`, the eigenvectors (columns) of the symmetric
/// matrix whose `upper` or lower triangle is given.
pub fn eigh<T: Real>(
    a: &[T],
    n: usize,
    upper: bool,
    w: &mut [T],
    v: Option<&mut [T]>,
) -> Result<(), String> {
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    return lapack::eigh(a, n, upper, w, v);
    #[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
    eigh_faer(a, n, upper, w, v)
}

#[cfg_attr(all(feature = "accelerate", target_vendor = "apple"), allow(dead_code))]
fn eigh_faer<T: Real>(
    a: &[T],
    n: usize,
    upper: bool,
    w: &mut [T],
    v: Option<&mut [T]>,
) -> Result<(), String> {
    threads(n);
    // Symmetrise from the triangle read (the other is ignored, as LAPACK's).
    let a = view(a, n, n);
    let sym = Mat::from_fn(n, n, |i, j| {
        let (r, c) = if (i <= j) == upper { (i, j) } else { (j, i) };
        a[(r, c)]
    });
    let side = if upper { Side::Upper } else { Side::Lower };
    match v {
        None => {
            let e = sym
                .self_adjoint_eigenvalues(side)
                .map_err(|e| format!("{e:?}"))?;
            w.copy_from_slice(&e);
        }
        Some(v) => {
            let f = sym.self_adjoint_eigen(side).map_err(|e| format!("{e:?}"))?;
            let d = f.S().column_vector();
            for (i, x) in w.iter_mut().enumerate() {
                *x = d[i];
            }
            write(f.U(), v);
        }
    }
    Ok(())
}

/// The eigenvalues and (with `v`) the unit-norm right eigenvectors of a general matrix, as
/// interleaved complex numbers (`re, im`).
pub fn eig<T: Real>(a: &[T], n: usize, w: &mut [T], v: Option<&mut [T]>) -> Result<(), String> {
    threads(n);
    let a = view(a, n, n);
    match v {
        None => {
            let e = a.eigenvalues().map_err(|e| format!("{e:?}"))?;
            for (i, z) in e.iter().enumerate() {
                w[2 * i] = z.re;
                w[2 * i + 1] = z.im;
            }
        }
        Some(v) => {
            let f = a.eigen().map_err(|e| format!("{e:?}"))?;
            let d = f.S().column_vector();
            for i in 0..n {
                w[2 * i] = d[i].re;
                w[2 * i + 1] = d[i].im;
            }
            let u = f.U();
            for i in 0..n {
                for j in 0..n {
                    v[2 * (i * n + j)] = u[(i, j)].re;
                    v[2 * (i * n + j) + 1] = u[(i, j)].im;
                }
            }
        }
    }
    Ok(())
}

/// The partial-pivoting LU factorisation of a square matrix: `lu` holds `L` (unit lower,
/// below the diagonal) and `U` of `P A`, and `rows[k]` is the row of `A` at row `k` of
/// `P A`.
pub fn lu<T: Real>(a: &[T], n: usize, lu: &mut [T], rows: &mut [usize]) {
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    {
        let mut pivots = vec![0usize; n];
        lapack::lu(a, n, lu, &mut pivots);
        // The row order from the swaps.
        for (i, r) in rows.iter_mut().enumerate() {
            *r = i;
        }
        for (j, &p) in pivots.iter().enumerate() {
            rows.swap(j, p);
        }
    }
    #[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
    lu_faer(a, n, lu, rows)
}

#[cfg_attr(all(feature = "accelerate", target_vendor = "apple"), allow(dead_code))]
fn lu_faer<T: Real>(a: &[T], n: usize, lu: &mut [T], rows: &mut [usize]) {
    threads(n);
    let f = view(a, n, n).partial_piv_lu();
    let (l, u) = (f.L(), f.U());
    for i in 0..n {
        for j in 0..n {
            lu[i * n + j] = if j < i { l[(i, j)] } else { u[(i, j)] };
        }
    }
    let (forward, _) = f.P().arrays();
    rows.copy_from_slice(forward);
}

/// The Cholesky factor (`L` with `A = L Lᵀ`, or `U` with `A = Uᵀ U`) of a symmetric
/// positive-definite matrix, reading its lower (or `upper`) triangle.
pub fn cholesky<T: Real>(a: &[T], n: usize, upper: bool, out: &mut [T]) -> Result<(), String> {
    out.copy_from_slice(a);
    cholesky_in_place(out, n, upper)
}

/// [`cholesky`] in place: `a` holds the matrix and receives the factor.
pub fn cholesky_in_place<T: Real>(a: &mut [T], n: usize, upper: bool) -> Result<(), String> {
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    return lapack::cholesky_in_place(a, n, upper);
    #[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
    {
        let src = a.to_vec();
        cholesky_faer(&src, n, upper, a)
    }
}

#[cfg_attr(all(feature = "accelerate", target_vendor = "apple"), allow(dead_code))]
fn cholesky_faer<T: Real>(a: &[T], n: usize, upper: bool, out: &mut [T]) -> Result<(), String> {
    threads(n);
    let a = view(a, n, n);
    // Read the lower triangle: for `upper`, the transpose of the given upper triangle.
    let lower = Mat::from_fn(n, n, |i, j| {
        if j > i {
            T::zero_impl()
        } else if upper {
            a[(j, i)]
        } else {
            a[(i, j)]
        }
    });
    let f = lower
        .llt(Side::Lower)
        .map_err(|_| "the matrix is not positive definite".to_owned())?;
    let l = f.L();
    for i in 0..n {
        for j in 0..n {
            let (r, c) = if upper { (j, i) } else { (i, j) };
            out[i * n + j] = if c <= r { l[(r, c)] } else { T::zero_impl() };
        }
    }
    Ok(())
}

/// `A⁻¹` by partial-pivoting LU (a singular matrix gives non-finite values, as LAPACK's).
pub fn inv<T: Real>(a: &[T], n: usize, out: &mut [T]) {
    out.copy_from_slice(a);
    inv_in_place(out, n);
}

/// [`inv`] in place.
pub fn inv_in_place<T: Real>(a: &mut [T], n: usize) {
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    lapack::inv_in_place(a, n);
    #[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
    {
        let src = a.to_vec();
        inv_faer(&src, n, a);
    }
}

#[cfg_attr(all(feature = "accelerate", target_vendor = "apple"), allow(dead_code))]
fn inv_faer<T: Real>(a: &[T], n: usize, out: &mut [T]) {
    threads(n);
    let x = view(a, n, n).partial_piv_lu().inverse();
    write(x.as_ref(), out);
}

/// The inverse of the lower (or `upper`) triangle of `a`.
pub fn tri_inv<T: Real>(a: &[T], n: usize, upper: bool, out: &mut [T]) {
    out.copy_from_slice(a);
    tri_inv_in_place(out, n, upper);
}

/// [`tri_inv`] in place.
pub fn tri_inv_in_place<T: Real>(a: &mut [T], n: usize, upper: bool) {
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    lapack::tri_inv_in_place(a, n, upper);
    #[cfg(not(all(feature = "accelerate", target_vendor = "apple")))]
    {
        let src = a.to_vec();
        tri_inv_faer(&src, n, upper, a);
    }
}

#[cfg_attr(all(feature = "accelerate", target_vendor = "apple"), allow(dead_code))]
fn tri_inv_faer<T: Real>(a: &[T], n: usize, upper: bool, out: &mut [T]) {
    let a = view(a, n, n);
    let mut dst = Mat::<T>::zeros(n, n);
    if upper {
        faer::linalg::triangular_inverse::invert_upper_triangular(dst.as_mut(), a, par(n));
    } else {
        faer::linalg::triangular_inverse::invert_lower_triangular(dst.as_mut(), a, par(n));
    }
    for i in 0..n {
        for j in 0..n {
            let keep = if upper { j >= i } else { j <= i };
            out[i * n + j] = if keep { dst[(i, j)] } else { T::zero_impl() };
        }
    }
}

/// `X = A⁻¹ B` for a square `A` (`n × n`) and `B` (`n × k`).
pub fn solve<T: Real>(a: &[T], n: usize, b: &[T], k: usize, out: &mut [T]) {
    // Small systems are faster in `faer` (no transposes, no framework call).
    #[cfg(all(feature = "accelerate", target_vendor = "apple"))]
    if n >= 128 {
        return lapack::solve(a, n, b, k, out);
    }
    solve_faer(a, n, b, k, out)
}

fn solve_faer<T: Real>(a: &[T], n: usize, b: &[T], k: usize, out: &mut [T]) {
    threads(n);
    let x = view(a, n, n).partial_piv_lu().solve(view(b, n, k));
    write(x.as_ref(), out);
}

/// `X = A⁻¹ B` for the lower (or `upper`) triangle of `A`.
pub fn solve_triangular<T: Real>(a: &[T], n: usize, upper: bool, b: &[T], k: usize, out: &mut [T]) {
    let a = view(a, n, n);
    let mut x = view(b, n, k).to_owned();
    if upper {
        faer::linalg::triangular_solve::solve_upper_triangular_in_place(a, x.as_mut(), par(n));
    } else {
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(a, x.as_mut(), par(n));
    }
    write(x.as_ref(), out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
            }
        }
        c
    }

    fn close(a: &[f64], b: &[f64]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-10, "{a:?} vs {b:?}");
        }
    }

    const A: [f64; 9] = [2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0];

    #[test]
    #[cfg_attr(
        miri,
        ignore = "faer's SIMD kernels call NEON intrinsics Miri cannot run"
    )]
    fn factorisations_reconstruct() {
        let (mut q, mut r) = (vec![0.0; 9], vec![0.0; 9]);
        qr(&A, 3, 3, &mut q, &mut r);
        close(&matmul(&q, &r, 3, 3, 3), &A);
        let (mut u, mut vt, mut s) = (vec![0.0; 9], vec![0.0; 9], vec![0.0; 3]);
        svd(&A, 3, 3, Some((&mut u, &mut vt)), &mut s).unwrap();
        let us: Vec<f64> = (0..9).map(|i| u[i] * s[i % 3]).collect();
        close(&matmul(&us, &vt, 3, 3, 3), &A);
        assert!(s[0] >= s[1] && s[1] >= s[2]);
        let (mut f, mut rows) = (vec![0.0; 9], vec![0; 3]);
        lu(&A, 3, &mut f, &mut rows);
        assert_eq!(rows, [2, 0, 1]);
        let mut x = vec![0.0; 9];
        inv(&A, 3, &mut x);
        close(
            &matmul(&A, &x, 3, 3, 3),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        let spd = [4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0];
        let mut l = vec![0.0; 9];
        cholesky(&spd, 3, false, &mut l).unwrap();
        let lt: Vec<f64> = (0..9).map(|i| l[(i % 3) * 3 + i / 3]).collect();
        close(&matmul(&l, &lt, 3, 3, 3), &spd);
        assert!(cholesky(&A, 3, false, &mut l).is_err());
        let (mut w, mut v) = (vec![0.0; 3], vec![0.0; 9]);
        eigh(&spd, 3, false, &mut w, Some(&mut v)).unwrap();
        assert!(w[0] <= w[1] && w[1] <= w[2]);
    }
}
