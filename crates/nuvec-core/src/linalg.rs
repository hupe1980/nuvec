//! Linear algebra, as MLX's `mx.linalg`: decompositions, inverses, solves, determinants,
//! norms and the cross product, over the last two axes and batched over the others, in
//! float32 and float64.
//!
//! - **Devices:** the decompositions run on the CPU (pure Rust, `faer`; Accelerate's LAPACK
//!   with the feature `accelerate`), also inside a GPU computation: on unified memory the
//!   kernels read and write the GPU graph's buffers, so a GPU stream needs no copy (MLX asks
//!   for a CPU stream instead).
//! - **Gradients:** every function but `eig`, `eigvals` and the LU factors' pivots is
//!   differentiable (`grad`, `vjp`, `jvp`, `vmap`), with the published rules (Giles 2008,
//!   Murray 2016, Walter and Lehmann 2018, Townsend 2016). MLX's `inv`, `solve`, `qr`,
//!   `svd`, `eigh` and `cholesky` have none. `eigh` and `cholesky` take the input as
//!   symmetric (their gradients are symmetric); `svd`'s vectors need distinct singular
//!   values, and the columns of `U` past `min(m, n)` carry no gradient.
//!
//! ```
//! use nuvec_core::{array, linalg};
//!
//! let a = array![[4.0f32, 1.0], [2.0, 3.0]];
//! let x = linalg::solve(&a, &array![1.0f32, 2.0]);
//! let close = |v: Vec<f32>, w: [f32; 2]| v.iter().zip(w).all(|(a, b)| (a - b).abs() < 1e-6);
//! assert!(close(x.to_vec::<f32>().unwrap(), [0.1, 0.6]));
//! let [q, r] = linalg::qr(&a);
//! assert!(q.matmul(&r).subtract(&a).abs().max(..).item::<f32>().unwrap() < 1e-6);
//! ```

pub(crate) mod imp;

use std::panic::Location;

use imp::{LinalgOp, record};

use crate::custom::custom_function_flat;
use crate::{Array, DType, Error, IntoResult, Result, ops};

#[track_caller]
fn sticky(name: &'static str, inputs: &[&Array], f: impl FnOnce() -> Result<Array>) -> Array {
    Array::sticky(name, Location::caller(), inputs, f)
}

#[track_caller]
fn sticky_n<const N: usize>(
    name: &'static str,
    inputs: &[&Array],
    f: impl FnOnce() -> Result<Vec<Array>>,
) -> [Array; N] {
    match f() {
        Ok(v) => v
            .try_into()
            .unwrap_or_else(|v: Vec<Array>| panic!("{name}: {} outputs", v.len())),
        Err(e) => std::array::from_fn(|_| sticky(name, inputs, || Err(e.clone()))),
    }
}

/// The reduced QR factorisation `A = Q R`: `Q [.., m, k]` with orthonormal columns and an
/// upper triangular `R [.., k, n]`, `k = min(m, n)`.
#[track_caller]
pub fn qr(a: &Array) -> [Array; 2] {
    sticky_n("linalg::qr", &[a], || record(LinalgOp::Qr, vec![a.clone()]))
}

/// The singular value decomposition `A = U diag(S) Vt`: `U [.., m, m]`, `S [.., k]`
/// (descending) and `Vt [.., n, n]` (MLX's `svd(a)`).
#[track_caller]
pub fn svd(a: &Array) -> [Array; 3] {
    sticky_n("linalg::svd", &[a], || {
        record(LinalgOp::Svd { uv: true }, vec![a.clone()])
    })
}

/// The singular values, descending (MLX's `svd(a, compute_uv=False)`).
#[track_caller]
pub fn svd_values(a: &Array) -> Array {
    sticky("linalg::svd", &[a], || {
        Ok(record(LinalgOp::Svd { uv: false }, vec![a.clone()])?.remove(0))
    })
}

/// The eigenvalues (ascending) and eigenvectors (columns) of symmetric matrices, reading
/// the lower (or `upper`, MLX's `UPLO="U"`) triangle.
#[track_caller]
pub fn eigh(a: &Array, upper: bool) -> [Array; 2] {
    sticky_n("linalg::eigh", &[a], || {
        record(
            LinalgOp::Eigh {
                upper,
                vectors: true,
            },
            vec![a.clone()],
        )
    })
}

/// The eigenvalues (ascending) of symmetric matrices (see [`eigh`]).
#[track_caller]
pub fn eigvalsh(a: &Array, upper: bool) -> Array {
    sticky("linalg::eigvalsh", &[a], || {
        Ok(record(
            LinalgOp::Eigh {
                upper,
                vectors: false,
            },
            vec![a.clone()],
        )?
        .remove(0))
    })
}

/// The eigenvalues and unit-norm right eigenvectors (columns) of general matrices, as
/// complex64. No derivative.
#[track_caller]
pub fn eig(a: &Array) -> [Array; 2] {
    sticky_n("linalg::eig", &[a], || {
        record(LinalgOp::Eig { vectors: true }, vec![a.clone()])
    })
}

/// The eigenvalues of general matrices, as complex64. No derivative.
#[track_caller]
pub fn eigvals(a: &Array) -> Array {
    sticky("linalg::eigvals", &[a], || {
        Ok(record(LinalgOp::Eig { vectors: false }, vec![a.clone()])?.remove(0))
    })
}

/// The LU factorisation with partial pivoting, as MLX's `lu_factor`: the packed `LU` of
/// `P A` and LAPACK's 0-based pivots (uint32: row `i` was swapped with `pivots[i]`). No
/// derivative.
#[track_caller]
pub fn lu_factor(a: &Array) -> [Array; 2] {
    sticky_n("linalg::lu_factor", &[a], || {
        let mut out = record(LinalgOp::LuFactor, vec![a.clone()])?;
        let pivots = out.remove(2);
        Ok(vec![out.remove(0), pivots])
    })
}

/// `(P, L, U)` with `A = L[P, :] @ U` (MLX's `lu`): `P` (uint32) a row permutation, `L`
/// unit lower and `U` upper triangular.
#[track_caller]
pub fn lu(a: &Array) -> [Array; 3] {
    sticky_n("linalg::lu", &[a], || {
        let mut out = record(LinalgOp::LuFactor, vec![a.clone()])?;
        let packed = ops::stop_gradient(&out.remove(0));
        let rows = out.remove(0);
        let n = a.dims()[a.ndim() - 1] as i32;
        let eye = ops::eye(n, None, 0, a.dtype());
        let l = ops::tril(&packed, -1).add(&eye);
        let u = ops::triu(&packed, 0);
        // P[rows[k]] = k: the inverse permutation.
        let p = ops::argsort(&rows, -1).astype(DType::U32);
        Ok(vec![p.into_result()?, l.into_result()?, u.into_result()?])
    })
}

/// The Cholesky factor of symmetric positive-definite matrices: lower `L` with `A = L Lᵀ`,
/// or `upper` `U` with `A = Uᵀ U`. A matrix that is not positive definite is an error.
#[track_caller]
pub fn cholesky(a: &Array, upper: bool) -> Array {
    sticky("linalg::cholesky", &[a], || {
        Ok(record(LinalgOp::Cholesky { upper }, vec![a.clone()])?.remove(0))
    })
}

/// `A⁻¹` (a singular matrix gives non-finite values).
#[track_caller]
pub fn inv(a: &Array) -> Array {
    sticky("linalg::inv", &[a], || {
        Ok(record(LinalgOp::Inverse, vec![a.clone()])?.remove(0))
    })
}

/// The inverse of the lower (or `upper`) triangle of `a`.
#[track_caller]
pub fn tri_inv(a: &Array, upper: bool) -> Array {
    sticky("linalg::tri_inv", &[a], || {
        Ok(record(LinalgOp::TriInverse { upper }, vec![a.clone()])?.remove(0))
    })
}

/// `A⁻¹` from its Cholesky factor (MLX's `cholesky_inv`): `L` (or `upper` `U`) with
/// `A = L Lᵀ` (`A = Uᵀ U`).
#[track_caller]
pub fn cholesky_inv(factor: &Array, upper: bool) -> Array {
    sticky("linalg::cholesky_inv", &[factor], || {
        let x = tri_inv(factor, upper);
        let xt = ops::swapaxes(&x, -1, -2);
        // L⁻ᵀ L⁻¹, or U⁻¹ U⁻ᵀ.
        if upper { x.matmul(&xt) } else { xt.matmul(&x) }.into_result()
    })
}

/// The Moore–Penrose pseudo-inverse, from the SVD: singular values at most
/// `max(m, n) · ε · s_max` are treated as zero.
#[track_caller]
pub fn pinv(a: &Array) -> Array {
    sticky("linalg::pinv", &[a], || {
        let nd = a.ndim();
        if nd < 2 {
            return Err(Error::InvalidArgument(
                "linalg::pinv: arrays must have at least 2 dimensions".into(),
            ));
        }
        let (m, n) = (a.dims()[nd - 2] as i32, a.dims()[nd - 1] as i32);
        let k = m.min(n);
        let [u, s, vt] = svd(a);
        let eps = if a.dtype() == DType::F64 {
            f64::EPSILON
        } else {
            f64::from(f32::EPSILON)
        };
        let cutoff = s.max_keepdims(-1).multiply(f64::from(m.max(n)) * eps);
        let keep = s.greater(&cutoff);
        let sinv = ops::where_(&keep, ops::reciprocal(&ops::where_(&keep, &s, 1.0)), 0.0);
        let uk = u.i((crate::Ellipsis, ..k));
        let vk = ops::swapaxes(&vt.i((crate::Ellipsis, ..k, ..)), -1, -2);
        vk.multiply(sinv.expand_dims(-2))
            .matmul(&ops::swapaxes(&uk, -1, -2))
            .into_result()
    })
}

/// `X = A⁻¹ B`: `b` is `[.., n]` (vectors) or `[.., n, k]`, broadcast against `a`'s batch.
#[track_caller]
pub fn solve(a: &Array, b: &Array) -> Array {
    sticky("linalg::solve", &[a, b], || {
        solve_with(LinalgOp::Solve, a, b)
    })
}

/// `X = A⁻¹ B` for the lower (or `upper`) triangle of `A`.
#[track_caller]
pub fn solve_triangular(a: &Array, b: &Array, upper: bool) -> Array {
    sticky("linalg::solve_triangular", &[a, b], || {
        solve_with(LinalgOp::SolveTriangular { upper }, a, b)
    })
}

fn solve_with(op: LinalgOp, a: &Array, b: &Array) -> Result<Array> {
    if a.ndim() < 2 {
        return Err(Error::InvalidArgument(format!(
            "{}: arrays must have at least 2 dimensions",
            op_name(op)
        )));
    }
    // The operands promote to a common dtype, as MLX's.
    let dtype = crate::promote_types(a.dtype(), b.dtype());
    let (a, b) = (&a.astype(dtype), &b.astype(dtype));
    // A vector right-hand side has one dimension fewer than the matrices.
    let vector = b.ndim() == a.ndim() - 1;
    let b2 = if vector { b.expand_dims(-1) } else { b.clone() };
    let nd = a.ndim().max(b2.ndim());
    let batch_a = &a.dims()[..a.ndim() - 2];
    let batch_b = &b2.dims()[..b2.ndim() - 2];
    let batch = crate::view::broadcast_shapes(op_name(op), batch_a, batch_b)?.to_vec();
    let _ = nd;
    let with = |x: &Array| -> Result<Array> {
        let mut shape = batch.clone();
        shape.extend_from_slice(&x.dims()[x.ndim() - 2..]);
        crate::ops::imp::broadcast_to(x, &shape)
    };
    let x = record(op, vec![with(a)?, with(&b2)?])?.remove(0);
    if vector {
        x.squeeze(-1).into_result()
    } else {
        Ok(x)
    }
}

fn op_name(op: LinalgOp) -> &'static str {
    op.name()
}

/// The determinant (from the LU factors), with the derivative `det · A⁻ᵀ`.
#[track_caller]
pub fn det(a: &Array) -> Array {
    sticky("linalg::det", &[a], || {
        let f = custom_function_flat(|xs: &[Array]| {
            let [sign, logabs] = signed_log_det(&xs[0])?;
            Ok(vec![sign.multiply(logabs.exp()).into_result()?])
        })
        .with_vjp(|p: &[Array], o: &[Array], c: &[Array]| {
            let inv_t = ops::swapaxes(&inv(&p[0]), -1, -2);
            let scale = c[0].multiply(&o[0]).expand_dims(-1).expand_dims(-1);
            Ok(vec![inv_t.multiply(scale).into_result()?])
        })
        .with_jvp(|p: &[Array], t: &[Array]| {
            // d det = det · tr(A⁻¹ dA)
            let [sign, logabs] = signed_log_det(&p[0])?;
            let d = sign.multiply(logabs.exp());
            Ok(vec![
                d.multiply(trace_of_solve(&p[0], &t[0])).into_result()?,
            ])
        });
        Ok(f.call(std::slice::from_ref(a))?.remove(0))
    })
}

/// `(sign, log|det|)` (MLX's `slogdet`): `sign` is −1, 0 or 1, and `log|det|` is −∞ for a
/// singular matrix. The derivative of `log|det|` is `A⁻ᵀ`.
#[track_caller]
pub fn slogdet(a: &Array) -> [Array; 2] {
    sticky_n("linalg::slogdet", &[a], || {
        let f = custom_function_flat(|xs: &[Array]| Ok(signed_log_det(&xs[0])?.to_vec()))
            .with_vjp(|p: &[Array], _o: &[Array], c: &[Array]| {
                let inv_t = ops::swapaxes(&inv(&p[0]), -1, -2);
                Ok(vec![
                    inv_t
                        .multiply(c[1].expand_dims(-1).expand_dims(-1))
                        .into_result()?,
                ])
            })
            .with_jvp(|p: &[Array], t: &[Array]| {
                // d log|det| = tr(A⁻¹ dA); the sign is locally constant.
                let [sign, _] = signed_log_det(&p[0])?;
                Ok(vec![
                    ops::zeros_like(&sign).into_result()?,
                    trace_of_solve(&p[0], &t[0]).into_result()?,
                ])
            });
        f.call(std::slice::from_ref(a))
    })
}

/// `tr(A⁻¹ T)` over the last two axes.
fn trace_of_solve(a: &Array, t: &Array) -> Array {
    ops::diagonal(&solve(a, t), 0, -2, -1).sum(-1)
}

/// `(sign, log|det|)` from the LU factors.
fn signed_log_det(a: &Array) -> Result<[Array; 2]> {
    let mut out = record(LinalgOp::LuFactor, vec![a.clone()])?;
    let packed = out.remove(0);
    let pivots = out.remove(1);
    let d = ops::diagonal(&packed, 0, -2, -1);
    let n = a.dims()[a.ndim() - 1] as i32;
    let idx = ops::arange(0, n, 1).astype(DType::U32);
    // Each pivot that moves a row flips the sign.
    let swaps = pivots.not_equal(&idx).astype(DType::I32).sum(-1);
    let parity = ops::remainder(&swaps, 2)
        .multiply(-2)
        .add(1)
        .astype(a.dtype());
    let sign = d.sign().prod(-1).multiply(parity);
    let logabs = d.abs().log().sum(-1);
    Ok([sign.into_result()?, logabs.into_result()?])
}

/// The cross product of 3-vectors (or 2-vectors, with `z = 0`, giving 3-vectors) along
/// `axis`, broadcast (MLX's `cross`).
#[track_caller]
pub fn cross(a: &Array, b: &Array, axis: i32) -> Array {
    sticky("linalg::cross", &[a, b], || {
        let pick = |x: &Array, i: i32| -> Array { ops::take(x, &Array::from(i), axis) };
        let parts = |x: &Array| -> Result<[Array; 3]> {
            let ax = if axis < 0 {
                axis + x.ndim() as i32
            } else {
                axis
            };
            match x.dims().get(ax as usize) {
                Some(3) => Ok([pick(x, 0), pick(x, 1), pick(x, 2)]),
                Some(2) => Ok([pick(x, 0), pick(x, 1), ops::zeros_like(&pick(x, 0))]),
                _ => Err(Error::InvalidArgument(
                    "linalg::cross: the axis must have 2 or 3 elements".into(),
                )),
            }
        };
        let ([a0, a1, a2], [b0, b1, b2]) = (parts(a)?, parts(b)?);
        let c0 = a1.multiply(&b2).subtract(a2.multiply(&b1));
        let c1 = a2.multiply(&b0).subtract(a0.multiply(&b2));
        let c2 = a0.multiply(&b1).subtract(a1.multiply(&b0));
        ops::stack(&[c0, c1, c2], axis).into_result()
    })
}

/// Vector and matrix norms, as MLX's `linalg.norm` ([`ops::matrix_norm`]).
#[track_caller]
pub fn norm(a: &Array, ord: ops::Norm, axes: Option<&[i32]>, keepdims: bool) -> Array {
    ops::matrix_norm(a, ord, axes, keepdims)
}
