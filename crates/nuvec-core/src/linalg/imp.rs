//! The decomposition primitive: shapes and dtypes, CPU kernels (`nuvec_cpu::linalg`, on
//! `faer` or Accelerate's LAPACK), batching, and reverse-mode rules (forward mode is the
//! transpose of the reverse rule, which is linear in the cotangents).

use crate::array::Data;
use crate::buffer::Buffer;
use crate::kernels::{Operand, contiguous};
use crate::{Array, DType, Ellipsis, Error, IntoResult, Result, ops};

/// A decomposition, inverse or solve over the last two axes (batched over the others).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LinalgOp {
    /// `[.., m, n]` → `Q [.., m, k]`, `R [.., k, n]`.
    Qr,
    /// → `U [.., m, m]`, `S [.., k]`, `Vt [.., n, n]`; or `S` alone.
    Svd { uv: bool },
    /// Symmetric `[.., n, n]` → `w [.., n]` (ascending) and `V [.., n, n]`, or `w` alone.
    Eigh { upper: bool, vectors: bool },
    /// General `[.., n, n]` → complex `w [.., n]` and `V [.., n, n]`, or `w` alone.
    Eig { vectors: bool },
    /// → `LU [.., n, n]` (of `P A`), the rows of `A` in `P A` and LAPACK's pivots (uint32).
    LuFactor,
    /// → the Cholesky factor.
    Cholesky { upper: bool },
    /// → `A⁻¹`.
    Inverse,
    /// → the inverse of a triangle.
    TriInverse { upper: bool },
    /// `(A [.., n, n], B [.., n, k])` → `A⁻¹ B`.
    Solve,
    /// The same for a triangle of `A`.
    SolveTriangular { upper: bool },
}

fn invalid(name: &str, m: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("{name}: {m}"))
}

impl LinalgOp {
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Qr => "linalg::qr",
            Self::Svd { .. } => "linalg::svd",
            Self::Eigh { .. } => "linalg::eigh",
            Self::Eig { .. } => "linalg::eig",
            Self::LuFactor => "linalg::lu_factor",
            Self::Cholesky { .. } => "linalg::cholesky",
            Self::Inverse => "linalg::inv",
            Self::TriInverse { .. } => "linalg::tri_inv",
            Self::Solve => "linalg::solve",
            Self::SolveTriangular { .. } => "linalg::solve_triangular",
        }
    }

    fn square(&self) -> bool {
        !matches!(self, Self::Qr | Self::Svd { .. })
    }

    /// The outputs' dtypes and shapes, after checking the inputs.
    pub(crate) fn outputs(&self, inputs: &[Array]) -> Result<Vec<(DType, Vec<usize>)>> {
        let name = self.name();
        let a = &inputs[0];
        let d = a.dtype();
        if !matches!(d, DType::F32 | DType::F64) {
            return Err(invalid(
                name,
                format!("arrays must have type float32 or float64, not {}", d.name()),
            ));
        }
        if a.ndim() < 2 {
            return Err(invalid(name, "arrays must have at least 2 dimensions"));
        }
        let n = a.ndim();
        let (rows, cols) = (a.dims()[n - 2], a.dims()[n - 1]);
        if self.square() && rows != cols {
            return Err(invalid(
                name,
                format!("the matrices must be square, not {rows}×{cols}"),
            ));
        }
        let batch = &a.dims()[..n - 2];
        let with = |tail: &[usize]| [batch, tail].concat();
        let k = rows.min(cols);
        Ok(match self {
            Self::Qr => vec![(d, with(&[rows, k])), (d, with(&[k, cols]))],
            Self::Svd { uv: true } => {
                vec![
                    (d, with(&[rows, rows])),
                    (d, with(&[k])),
                    (d, with(&[cols, cols])),
                ]
            }
            Self::Svd { uv: false } => vec![(d, with(&[k]))],
            Self::Eigh { vectors: true, .. } => vec![(d, with(&[rows])), (d, with(&[rows, rows]))],
            Self::Eigh { vectors: false, .. } => vec![(d, with(&[rows]))],
            Self::Eig { vectors } => {
                let c = DType::Complex64;
                let mut v = vec![(c, with(&[rows]))];
                if *vectors {
                    v.push((c, with(&[rows, rows])));
                }
                v
            }
            Self::LuFactor => vec![
                (d, with(&[rows, rows])),
                (DType::U32, with(&[rows])),
                (DType::U32, with(&[rows])),
            ],
            Self::Cholesky { .. } | Self::Inverse | Self::TriInverse { .. } => {
                vec![(d, a.dims().to_vec())]
            }
            Self::Solve | Self::SolveTriangular { .. } => {
                let b = &inputs[1];
                if b.dtype() != d {
                    return Err(invalid(name, "the operands differ in dtype"));
                }
                let nb = b.ndim();
                if nb != n || b.dims()[..n - 2] != *batch || b.dims()[n - 2] != rows {
                    return Err(invalid(
                        name,
                        format!(
                            "the right-hand side {:?} does not match the matrices {:?}",
                            b.dims(),
                            a.dims()
                        ),
                    ));
                }
                vec![(d, b.dims().to_vec())]
            }
        })
    }

    /// Evaluates the operation on the CPU, one matrix at a time.
    pub(crate) fn eval_cpu(&self, operands: &[Operand<'_>]) -> Result<Vec<Data>> {
        match operands[0].0.dtype {
            DType::F32 => self.run::<f32>(operands),
            _ => self.run::<f64>(operands),
        }
    }

    fn run<T: nuvec_cpu::linalg::Real + bytemuck::Pod>(
        &self,
        operands: &[Operand<'_>],
    ) -> Result<Vec<Data>> {
        let name = self.name();
        let meta = operands[0].0;
        let a = contiguous(meta, operands[0].1)?;
        let a = a.typed::<T>();
        let nd = meta.shape.len();
        let (m, n) = (meta.shape[nd - 2], meta.shape[nd - 1]);
        let batch: usize = meta.shape[..nd - 2].iter().product();
        let k = m.min(n);
        let alloc = |count: usize| Buffer::try_alloc(count * size_of::<T>());
        let fail = |e: String| invalid(name, e);
        let mat = |i: usize| &a[i * m * n..(i + 1) * m * n];
        let mut outs: Vec<Buffer> = Vec::new();
        match *self {
            Self::Qr => {
                let (mut q, mut r) = (alloc(batch * m * k)?, alloc(batch * k * n)?);
                for i in 0..batch {
                    nuvec_cpu::linalg::qr(
                        mat(i),
                        m,
                        n,
                        &mut q.typed_mut::<T>()[i * m * k..(i + 1) * m * k],
                        &mut r.typed_mut::<T>()[i * k * n..(i + 1) * k * n],
                    );
                }
                outs = vec![q, r];
            }
            Self::Svd { uv } => {
                let mut s = alloc(batch * k)?;
                if uv {
                    let (mut u, mut vt) = (alloc(batch * m * m)?, alloc(batch * n * n)?);
                    for i in 0..batch {
                        nuvec_cpu::linalg::svd(
                            mat(i),
                            m,
                            n,
                            Some((
                                &mut u.typed_mut::<T>()[i * m * m..(i + 1) * m * m],
                                &mut vt.typed_mut::<T>()[i * n * n..(i + 1) * n * n],
                            )),
                            &mut s.typed_mut::<T>()[i * k..(i + 1) * k],
                        )
                        .map_err(fail)?;
                    }
                    outs = vec![u, s, vt];
                } else {
                    for i in 0..batch {
                        nuvec_cpu::linalg::svd(
                            mat(i),
                            m,
                            n,
                            None,
                            &mut s.typed_mut::<T>()[i * k..(i + 1) * k],
                        )
                        .map_err(fail)?;
                    }
                    outs = vec![s];
                }
            }
            Self::Eigh { upper, vectors } => {
                let mut w = alloc(batch * n)?;
                let mut v = if vectors {
                    Some(alloc(batch * n * n)?)
                } else {
                    None
                };
                for i in 0..batch {
                    let vi = v
                        .as_mut()
                        .map(|v| &mut v.typed_mut::<T>()[i * n * n..(i + 1) * n * n]);
                    nuvec_cpu::linalg::eigh(
                        mat(i),
                        n,
                        upper,
                        &mut w.typed_mut::<T>()[i * n..(i + 1) * n],
                        vi,
                    )
                    .map_err(fail)?;
                }
                outs.push(w);
                outs.extend(v);
            }
            Self::Eig { vectors } => {
                let mut w = vec![T::zero_impl(); 2 * n];
                let mut v = vec![T::zero_impl(); 2 * n * n];
                let mut wo = Buffer::try_alloc(batch * n * 8)?;
                let mut vo = if vectors {
                    Some(Buffer::try_alloc(batch * n * n * 8)?)
                } else {
                    None
                };
                for i in 0..batch {
                    nuvec_cpu::linalg::eig(mat(i), n, &mut w, vectors.then_some(&mut v[..]))
                        .map_err(fail)?;
                    let to32 = |x: T| x.to_f32();
                    let wf = wo.typed_mut::<f32>();
                    for (j, &x) in w.iter().enumerate() {
                        wf[i * 2 * n + j] = to32(x);
                    }
                    if let Some(vo) = vo.as_mut() {
                        let vf = vo.typed_mut::<f32>();
                        for (j, &x) in v.iter().enumerate() {
                            vf[i * 2 * n * n + j] = to32(x);
                        }
                    }
                }
                outs.push(wo);
                outs.extend(vo);
            }
            Self::LuFactor => {
                let mut f = alloc(batch * n * n)?;
                let mut rows_out = Buffer::try_alloc(batch * n * 4)?;
                let mut pivots = Buffer::try_alloc(batch * n * 4)?;
                let mut rows = vec![0usize; n];
                for i in 0..batch {
                    nuvec_cpu::linalg::lu(
                        mat(i),
                        n,
                        &mut f.typed_mut::<T>()[i * n * n..(i + 1) * n * n],
                        &mut rows,
                    );
                    let r = rows_out.typed_mut::<u32>();
                    for (j, &x) in rows.iter().enumerate() {
                        r[i * n + j] = x as u32;
                    }
                    // LAPACK's pivots: the row swapped into position j at step j.
                    let mut at: Vec<usize> = (0..n).collect();
                    let mut pos: Vec<usize> = (0..n).collect();
                    let p = pivots.typed_mut::<u32>();
                    for j in 0..n {
                        let s = pos[rows[j]];
                        p[i * n + j] = s as u32;
                        let (rj, rs) = (at[j], at[s]);
                        at.swap(j, s);
                        pos[rj] = s;
                        pos[rs] = j;
                    }
                }
                outs = vec![f, rows_out, pivots];
            }
            Self::Cholesky { .. } | Self::Inverse | Self::TriInverse { .. } => {
                // One copy, as MLX's: the contiguous input becomes the output in place.
                let mut o = contiguous(meta, operands[0].1)?;
                for i in 0..batch {
                    let x = &mut o.typed_mut::<T>()[i * n * n..(i + 1) * n * n];
                    match *self {
                        Self::Cholesky { upper } => {
                            nuvec_cpu::linalg::cholesky_in_place(x, n, upper).map_err(fail)?;
                        }
                        Self::Inverse => nuvec_cpu::linalg::inv_in_place(x, n),
                        Self::TriInverse { upper } => {
                            nuvec_cpu::linalg::tri_inv_in_place(x, n, upper)
                        }
                        _ => unreachable!(),
                    }
                }
                outs = vec![o];
            }
            Self::Solve | Self::SolveTriangular { .. } => {
                let bm = operands[1].0;
                let b = contiguous(bm, operands[1].1)?;
                let b = b.typed::<T>();
                let kk = bm.shape[nd - 1];
                let mut o = alloc(batch * n * kk)?;
                for i in 0..batch {
                    let (bi, oi) = (
                        &b[i * n * kk..(i + 1) * n * kk],
                        i * n * kk..(i + 1) * n * kk,
                    );
                    let out = &mut o.typed_mut::<T>()[oi];
                    match *self {
                        Self::Solve => nuvec_cpu::linalg::solve(mat(i), n, bi, kk, out),
                        Self::SolveTriangular { upper } => {
                            nuvec_cpu::linalg::solve_triangular(mat(i), n, upper, bi, kk, out);
                        }
                        _ => unreachable!(),
                    }
                }
                outs = vec![o];
            }
        }
        Ok(outs.into_iter().map(Data::new).collect())
    }

    /// Batching rule: the batch axis first (an unbatched operand of a solve broadcast).
    pub(crate) fn vmap(
        &self,
        inputs: &[Array],
        axes: &[Option<usize>],
    ) -> Result<(Vec<Array>, usize)> {
        let b = inputs
            .iter()
            .zip(axes)
            .find_map(|(a, ax)| ax.map(|ax| a.dims()[ax]))
            .expect("one batched input");
        let front = |i: usize| -> Result<Array> {
            match axes[i] {
                Some(ax) => crate::batching::move_axis(&inputs[i], ax, 0),
                None => {
                    let a = &inputs[i];
                    let mut shape = vec![b];
                    shape.extend_from_slice(a.dims());
                    crate::ops::imp::broadcast_to(&crate::ops::imp::expand_dims(a, 0)?, &shape)
                }
            }
        };
        let all: Vec<Array> = (0..inputs.len()).map(front).collect::<Result<_>>()?;
        Ok((record(*self, all)?, 0))
    }

    /// Reverse-mode rule: one cotangent per input.
    pub(crate) fn vjp(
        &self,
        inputs: &[Array],
        outputs: &[Array],
        cotangents: &[Array],
        wanted: &[bool],
    ) -> Result<Vec<Option<Array>>> {
        let name = self.name();
        let a = &inputs[0];
        let g = cotangents;
        let mt = |x: &Array| ops::swapaxes(x, -1, -2);
        let mm = |x: &Array, y: &Array| ops::matmul(x, y);
        let r = |x: Array| x.into_result();
        let da: Array = match *self {
            Self::Inverse => {
                let xt = mt(&outputs[0]);
                r(mm(&mm(&xt, &g[0]), &xt).negative())?
            }
            Self::TriInverse { upper } => {
                let xt = mt(&outputs[0]);
                let full = mm(&mm(&xt, &g[0]), &xt).negative();
                r(if upper {
                    ops::triu(&full, 0)
                } else {
                    ops::tril(&full, 0)
                })?
            }
            Self::Solve | Self::SolveTriangular { .. } => {
                let at = mt(a);
                let gb = match *self {
                    Self::Solve => record(Self::Solve, vec![at, g[0].clone()])?.remove(0),
                    Self::SolveTriangular { upper } => record(
                        Self::SolveTriangular { upper: !upper },
                        vec![at, g[0].clone()],
                    )?
                    .remove(0),
                    _ => unreachable!(),
                };
                let ga = mm(&gb, &mt(&outputs[0])).negative();
                let ga = match *self {
                    Self::SolveTriangular { upper: true } => ops::triu(&ga, 0),
                    Self::SolveTriangular { upper: false } => ops::tril(&ga, 0),
                    _ => ga,
                };
                return Ok(vec![
                    wanted[0].then(|| r(ga)).transpose()?,
                    wanted[1].then_some(gb),
                ]);
            }
            Self::Cholesky { upper } => {
                // Murray 2016, over the lower factor.
                let (l, gl) = if upper {
                    (mt(&outputs[0]), mt(&g[0]))
                } else {
                    (outputs[0].clone(), g[0].clone())
                };
                let p = mm(&mt(&l), &gl);
                let p = ops::tril(&p, 0).subtract(half_diag(&p));
                let lt = mt(&l);
                // S = L⁻ᵀ P L⁻¹
                let t =
                    record(Self::SolveTriangular { upper: true }, vec![lt.clone(), p])?.remove(0);
                let s =
                    mt(&record(Self::SolveTriangular { upper: true }, vec![lt, mt(&t)])?.remove(0));
                r(s.add(mt(&s)).multiply(0.5))?
            }
            Self::Qr => {
                let (m, n) = (a.dims()[a.ndim() - 2], a.dims()[a.ndim() - 1]);
                if m < n {
                    return Err(invalid(
                        name,
                        "the derivative of a wide matrix's QR is not defined here",
                    ));
                }
                let (q, rr) = (&outputs[0], &outputs[1]);
                let mmat = mm(rr, &mt(&g[1])).subtract(mm(&mt(&g[0]), q));
                let low = ops::tril(&mmat, -1);
                let copyltu = low.add(mt(&low)).add(diag_part(&mmat));
                let b = g[0].add(mm(q, &copyltu));
                // Ā = B R⁻ᵀ, so R Āᵀ = Bᵀ.
                let x = record(
                    Self::SolveTriangular { upper: true },
                    vec![rr.clone(), mt(&b)],
                )?
                .remove(0);
                r(mt(&x))?
            }
            Self::Eigh { vectors, .. } => {
                let w = &outputs[0];
                if !vectors {
                    // eigvalsh: the vectors are needed for the gradient.
                    let both = record(
                        Self::Eigh {
                            upper: matches!(self, Self::Eigh { upper: true, .. }),
                            vectors: true,
                        },
                        vec![a.clone()],
                    )?;
                    let v = &both[1];
                    return Ok(vec![Some(r(mm(&mm(v, &diag_embed(&g[0])), &mt(v)))?)]);
                }
                let v = &outputs[1];
                let f = gap_inverse(w)?;
                let inner = f.multiply(mm(&mt(v), &g[1]));
                let mid = diag_embed(&g[0]).add(inner);
                let x = mm(&mm(v, &mid), &mt(v));
                r(x.add(mt(&x)).multiply(0.5))?
            }
            Self::Svd { uv } => {
                let (m, n) = (a.dims()[a.ndim() - 2], a.dims()[a.ndim() - 1]);
                let k = m.min(n) as i32;
                if !uv {
                    let both = record(Self::Svd { uv: true }, vec![a.clone()])?;
                    let (uk, vhk) = (cols(&both[0], k), rows_of(&both[2], k));
                    return Ok(vec![Some(r(mm(&mm(&uk, &diag_embed(&g[0])), &vhk))?)]);
                }
                svd_vjp(
                    &outputs[0],
                    &outputs[1],
                    &outputs[2],
                    &g[0],
                    &g[1],
                    &g[2],
                    m,
                    n,
                )?
            }
            Self::Eig { .. } | Self::LuFactor => {
                return Err(invalid(name, "no derivative (as in MLX)"));
            }
        };
        let mut out = vec![wanted[0].then_some(da)];
        out.resize(inputs.len(), None);
        Ok(out)
    }

    /// Forward-mode rule: the reverse rule is linear in the cotangents, so `J t` is the
    /// reverse rule's own reverse derivative applied to `t`.
    pub(crate) fn jvp(&self, inputs: &[Array], tangents: &[Option<Array>]) -> Result<Vec<Array>> {
        let outputs = record(*self, inputs.to_vec())?;
        let zeros: Vec<Array> = outputs
            .iter()
            .map(|o| ops::zeros_like(o).into_result())
            .collect::<Result<_>>()?;
        let ts: Vec<Array> = inputs
            .iter()
            .zip(tangents)
            .map(|(x, t)| {
                t.clone()
                    .map_or_else(|| ops::zeros_like(x).into_result(), Ok)
            })
            .collect::<Result<_>>()?;
        let wanted = vec![true; inputs.len()];
        let op = *self;
        let (ins, outs) = (inputs.to_vec(), outputs.clone());
        let rule = move |cot: &[Array]| -> Result<Vec<Array>> {
            let gs = op.vjp(&ins, &outs, cot, &wanted)?;
            Ok(gs
                .into_iter()
                .zip(&ins)
                .map(|(g, x)| g.unwrap_or_else(|| ops::zeros_like(x)))
                .collect())
        };
        Ok(crate::vjp_flat(rule, &zeros, &ts)?.1)
    }
}

/// Records the primitive on `inputs`.
pub(crate) fn record(op: LinalgOp, inputs: Vec<Array>) -> Result<Vec<Array>> {
    Array::from_op(crate::primitive::Primitive::Linalg(op), inputs)
}

/// The diagonal of `x` times one half, as a matrix.
fn half_diag(x: &Array) -> Array {
    diag_part(x).multiply(0.5)
}

/// The diagonal of the last two axes, as a matrix (zeros elsewhere).
fn diag_part(x: &Array) -> Array {
    let n = x.dims()[x.ndim() - 1] as i32;
    x.multiply(ops::eye(n, None, 0, x.dtype()))
}

/// `[.., n]` → `[.., n, n]` with the vector on the diagonal.
fn diag_embed(v: &Array) -> Array {
    let n = v.dims()[v.ndim() - 1] as i32;
    v.expand_dims(-2).multiply(ops::eye(n, None, 0, v.dtype()))
}

/// `F_ij = 1 / (w_j − w_i)` off the diagonal, 0 on it.
fn gap_inverse(w: &Array) -> Result<Array> {
    let n = w.dims()[w.ndim() - 1] as i32;
    let gap = w.expand_dims(-2).subtract(w.expand_dims(-1));
    let eye = ops::eye(n, None, 0, w.dtype());
    // Off the diagonal 1/gap; the diagonal (where gap is 0) is masked to 0.
    let safe = gap.add(&eye);
    ops::reciprocal(&safe)
        .multiply(eye.negative().add(1.0))
        .into_result()
}

fn cols(x: &Array, k: i32) -> Array {
    x.i((Ellipsis, ..k))
}

fn rows_of(x: &Array, k: i32) -> Array {
    x.i((Ellipsis, ..k, ..))
}

/// PyTorch's `svd_backward` (real matrices), over the thin factors.
#[allow(clippy::too_many_arguments)]
fn svd_vjp(
    u: &Array,
    s: &Array,
    vt: &Array,
    gu: &Array,
    gs: &Array,
    gvt: &Array,
    m: usize,
    n: usize,
) -> Result<Array> {
    let k = m.min(n) as i32;
    let mt = |x: &Array| ops::swapaxes(x, -1, -2);
    let mm = |x: &Array, y: &Array| ops::matmul(x, y);
    let last2 = |x: &Array, r: i32, c: i32| x.i((Ellipsis, ..r, ..c));
    let uk = last2(u, m as i32, k);
    let vhk = last2(vt, k, n as i32);
    let guk = last2(gu, m as i32, k);
    let gv = mt(&last2(gvt, k, n as i32)); // [n, k]
    let v = mt(&vhk);
    let s2 = s.multiply(s);
    let e = s2.expand_dims(-2).subtract(s2.expand_dims(-1));
    let eye = ops::eye(k, None, 0, s.dtype());
    let e = e.add(&eye);
    let skew = |x: Array| x.subtract(mt(&x));
    let uhgu = skew(mm(&mt(&uk), &guk));
    let vhgv = skew(mm(&mt(&v), &gv));
    let mid = uhgu
        .multiply(s.expand_dims(-2))
        .add(s.expand_dims(-1).multiply(&vhgv))
        .divide(&e)
        .add(diag_embed(gs));
    let mut ga = mm(&mm(&uk, &mid), &vhk);
    if m > k as usize {
        let gu_sinv = guk.divide(s.expand_dims(-2));
        let proj = gu_sinv.subtract(mm(&uk, &mm(&mt(&uk), &gu_sinv)));
        ga = ga.add(mm(&proj, &vhk));
    }
    if n > k as usize {
        let gv_sinv = gv.divide(s.expand_dims(-2));
        let proj = gv_sinv.subtract(mm(&v, &mm(&mt(&v), &gv_sinv)));
        ga = ga.add(mm(&uk, &mt(&proj)));
    }
    ga.into_result()
}
