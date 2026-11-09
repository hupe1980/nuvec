//! Accelerate's LAPACK (feature `accelerate`, Apple targets): the routines MLX calls, on
//! Apple's AMX units. Accelerate is a system framework; the declarations are LAPACK's,
//! hand-written (`$NEWLAPACK`, 32-bit integers). LAPACK is column-major: a row-major buffer
//! is the matrix's transpose, which each routine below accounts for.

use std::os::raw::c_char;

type Int = i32;

macro_rules! lapack {
    ($($sym:literal => fn $name:ident($($arg:ident: $t:ty),*);)*) => {
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            $(
                #[link_name = $sym]
                fn $name($($arg: $t),*);
            )*
        }
    };
}

lapack! {
    "spotrf$NEWLAPACK" => fn spotrf(uplo: *const c_char, n: *const Int, a: *mut f32, lda: *const Int, info: *mut Int);
    "dpotrf$NEWLAPACK" => fn dpotrf(uplo: *const c_char, n: *const Int, a: *mut f64, lda: *const Int, info: *mut Int);
    "sgetrf$NEWLAPACK" => fn sgetrf(m: *const Int, n: *const Int, a: *mut f32, lda: *const Int, ipiv: *mut Int, info: *mut Int);
    "dgetrf$NEWLAPACK" => fn dgetrf(m: *const Int, n: *const Int, a: *mut f64, lda: *const Int, ipiv: *mut Int, info: *mut Int);
    "sgetrs$NEWLAPACK" => fn sgetrs(trans: *const c_char, n: *const Int, nrhs: *const Int, a: *const f32, lda: *const Int, ipiv: *const Int, b: *mut f32, ldb: *const Int, info: *mut Int);
    "dgetrs$NEWLAPACK" => fn dgetrs(trans: *const c_char, n: *const Int, nrhs: *const Int, a: *const f64, lda: *const Int, ipiv: *const Int, b: *mut f64, ldb: *const Int, info: *mut Int);
    "sgetri$NEWLAPACK" => fn sgetri(n: *const Int, a: *mut f32, lda: *const Int, ipiv: *const Int, work: *mut f32, lwork: *const Int, info: *mut Int);
    "dgetri$NEWLAPACK" => fn dgetri(n: *const Int, a: *mut f64, lda: *const Int, ipiv: *const Int, work: *mut f64, lwork: *const Int, info: *mut Int);
    "sgeqrf$NEWLAPACK" => fn sgeqrf(m: *const Int, n: *const Int, a: *mut f32, lda: *const Int, tau: *mut f32, work: *mut f32, lwork: *const Int, info: *mut Int);
    "dgeqrf$NEWLAPACK" => fn dgeqrf(m: *const Int, n: *const Int, a: *mut f64, lda: *const Int, tau: *mut f64, work: *mut f64, lwork: *const Int, info: *mut Int);
    "sorgqr$NEWLAPACK" => fn sorgqr(m: *const Int, n: *const Int, k: *const Int, a: *mut f32, lda: *const Int, tau: *const f32, work: *mut f32, lwork: *const Int, info: *mut Int);
    "dorgqr$NEWLAPACK" => fn dorgqr(m: *const Int, n: *const Int, k: *const Int, a: *mut f64, lda: *const Int, tau: *const f64, work: *mut f64, lwork: *const Int, info: *mut Int);
    "sgesdd$NEWLAPACK" => fn sgesdd(jobz: *const c_char, m: *const Int, n: *const Int, a: *mut f32, lda: *const Int, s: *mut f32, u: *mut f32, ldu: *const Int, vt: *mut f32, ldvt: *const Int, work: *mut f32, lwork: *const Int, iwork: *mut Int, info: *mut Int);
    "dgesdd$NEWLAPACK" => fn dgesdd(jobz: *const c_char, m: *const Int, n: *const Int, a: *mut f64, lda: *const Int, s: *mut f64, u: *mut f64, ldu: *const Int, vt: *mut f64, ldvt: *const Int, work: *mut f64, lwork: *const Int, iwork: *mut Int, info: *mut Int);
    "ssyevd$NEWLAPACK" => fn ssyevd(jobz: *const c_char, uplo: *const c_char, n: *const Int, a: *mut f32, lda: *const Int, w: *mut f32, work: *mut f32, lwork: *const Int, iwork: *mut Int, liwork: *const Int, info: *mut Int);
    "dsyevd$NEWLAPACK" => fn dsyevd(jobz: *const c_char, uplo: *const c_char, n: *const Int, a: *mut f64, lda: *const Int, w: *mut f64, work: *mut f64, lwork: *const Int, iwork: *mut Int, liwork: *const Int, info: *mut Int);
    "strtri$NEWLAPACK" => fn strtri(uplo: *const c_char, diag: *const c_char, n: *const Int, a: *mut f32, lda: *const Int, info: *mut Int);
    "dtrtri$NEWLAPACK" => fn dtrtri(uplo: *const c_char, diag: *const c_char, n: *const Int, a: *mut f64, lda: *const Int, info: *mut Int);
}

/// The element types with LAPACK routines, dispatching to the `s` or `d` routine.
pub(crate) trait Lapack: Copy + Default + PartialOrd {
    /// −1: a workspace query's marker in `work[0]`.
    const NEG_ONE: Self;
    fn to_usize(self) -> usize;
    /// Cholesky (`potrf`).
    fn potrf(uplo: u8, n: Int, a: &mut [Self]) -> Int;
    /// LU (`getrf`).
    fn getrf(m: Int, n: Int, a: &mut [Self], ipiv: &mut [Int]) -> Int;
    /// A solve from LU (`getrs`).
    fn getrs(trans: u8, n: Int, nrhs: Int, a: &[Self], ipiv: &[Int], b: &mut [Self]) -> Int;
    /// The inverse from LU (`getri`).
    fn getri(n: Int, a: &mut [Self], ipiv: &[Int], work: &mut [Self]) -> Int;
    /// QR (`geqrf`).
    fn geqrf(m: Int, n: Int, a: &mut [Self], tau: &mut [Self], work: &mut [Self]) -> Int;
    /// `Q` from QR (`orgqr`).
    fn orgqr(m: Int, n: Int, k: Int, a: &mut [Self], tau: &[Self], work: &mut [Self]) -> Int;
    /// SVD (`gesdd`).
    #[allow(clippy::too_many_arguments)]
    fn gesdd(
        jobz: u8,
        m: Int,
        n: Int,
        a: &mut [Self],
        s: &mut [Self],
        u: &mut [Self],
        ldu: Int,
        vt: &mut [Self],
        ldvt: Int,
        work: &mut [Self],
        iwork: &mut [Int],
    ) -> Int;
    /// Symmetric eigendecomposition (`syevd`).
    fn syevd(
        jobz: u8,
        uplo: u8,
        n: Int,
        a: &mut [Self],
        w: &mut [Self],
        work: &mut [Self],
        iwork: &mut [Int],
    ) -> Int;
    /// Triangular inverse (`trtri`).
    fn trtri(uplo: u8, n: Int, a: &mut [Self]) -> Int;
}

macro_rules! impl_lapack {
    ($t:ty, $potrf:ident, $getrf:ident, $getrs:ident, $getri:ident, $geqrf:ident, $orgqr:ident, $gesdd:ident, $syevd:ident, $trtri:ident) => {
        impl Lapack for $t {
            const NEG_ONE: Self = -1.0;
            fn to_usize(self) -> usize {
                self as usize
            }
            fn potrf(uplo: u8, n: Int, a: &mut [Self]) -> Int {
                let mut info = 0;
                // SAFETY: `a` holds `n · n` elements (lda = n); the pointers outlive the call.
                unsafe {
                    $potrf(
                        (&uplo as *const u8).cast(),
                        &n,
                        a.as_mut_ptr(),
                        &n.max(1),
                        &mut info,
                    )
                };
                info
            }
            fn getrf(m: Int, n: Int, a: &mut [Self], ipiv: &mut [Int]) -> Int {
                let mut info = 0;
                // SAFETY: `a` holds `m · n` elements (lda = m), `ipiv` min(m, n).
                unsafe {
                    $getrf(
                        &m,
                        &n,
                        a.as_mut_ptr(),
                        &m.max(1),
                        ipiv.as_mut_ptr(),
                        &mut info,
                    )
                };
                info
            }
            fn getrs(
                trans: u8,
                n: Int,
                nrhs: Int,
                a: &[Self],
                ipiv: &[Int],
                b: &mut [Self],
            ) -> Int {
                let mut info = 0;
                // SAFETY: `a` holds `n · n` elements, `ipiv` n, `b` n · nrhs (ldb = n).
                unsafe {
                    $getrs(
                        (&trans as *const u8).cast(),
                        &n,
                        &nrhs,
                        a.as_ptr(),
                        &n.max(1),
                        ipiv.as_ptr(),
                        b.as_mut_ptr(),
                        &n.max(1),
                        &mut info,
                    )
                };
                info
            }
            fn getri(n: Int, a: &mut [Self], ipiv: &[Int], work: &mut [Self]) -> Int {
                let (mut info, lwork) = (0, work.len() as Int);
                // SAFETY: `a` holds `n · n` elements, `ipiv` n, `work` lwork (≥ 1; −1 queries).
                unsafe {
                    $getri(
                        &n,
                        a.as_mut_ptr(),
                        &n.max(1),
                        ipiv.as_ptr(),
                        work.as_mut_ptr(),
                        &if lwork == 1 && work[0] < 0.0 {
                            -1
                        } else {
                            lwork
                        },
                        &mut info,
                    )
                };
                info
            }
            fn geqrf(m: Int, n: Int, a: &mut [Self], tau: &mut [Self], work: &mut [Self]) -> Int {
                let (mut info, lwork) = (0, work.len() as Int);
                // SAFETY: `a` holds `m · n` elements, `tau` min(m, n), `work` lwork.
                unsafe {
                    $geqrf(
                        &m,
                        &n,
                        a.as_mut_ptr(),
                        &m.max(1),
                        tau.as_mut_ptr(),
                        work.as_mut_ptr(),
                        &if lwork == 1 && work[0] < 0.0 {
                            -1
                        } else {
                            lwork
                        },
                        &mut info,
                    )
                };
                info
            }
            fn orgqr(
                m: Int,
                n: Int,
                k: Int,
                a: &mut [Self],
                tau: &[Self],
                work: &mut [Self],
            ) -> Int {
                let (mut info, lwork) = (0, work.len() as Int);
                // SAFETY: `a` holds `m · n` elements, `tau` k, `work` lwork.
                unsafe {
                    $orgqr(
                        &m,
                        &n,
                        &k,
                        a.as_mut_ptr(),
                        &m.max(1),
                        tau.as_ptr(),
                        work.as_mut_ptr(),
                        &if lwork == 1 && work[0] < 0.0 {
                            -1
                        } else {
                            lwork
                        },
                        &mut info,
                    )
                };
                info
            }
            fn gesdd(
                jobz: u8,
                m: Int,
                n: Int,
                a: &mut [Self],
                s: &mut [Self],
                u: &mut [Self],
                ldu: Int,
                vt: &mut [Self],
                ldvt: Int,
                work: &mut [Self],
                iwork: &mut [Int],
            ) -> Int {
                let (mut info, lwork) = (0, work.len() as Int);
                // SAFETY: `a` holds `m · n` elements, `s` min(m, n), `u` ldu · m, `vt`
                // ldvt · n, `work` lwork, `iwork` 8 · min(m, n).
                unsafe {
                    $gesdd(
                        (&jobz as *const u8).cast(),
                        &m,
                        &n,
                        a.as_mut_ptr(),
                        &m.max(1),
                        s.as_mut_ptr(),
                        u.as_mut_ptr(),
                        &ldu,
                        vt.as_mut_ptr(),
                        &ldvt,
                        work.as_mut_ptr(),
                        &if lwork == 1 && work[0] < 0.0 {
                            -1
                        } else {
                            lwork
                        },
                        iwork.as_mut_ptr(),
                        &mut info,
                    )
                };
                info
            }
            fn syevd(
                jobz: u8,
                uplo: u8,
                n: Int,
                a: &mut [Self],
                w: &mut [Self],
                work: &mut [Self],
                iwork: &mut [Int],
            ) -> Int {
                let mut info = 0;
                let query = work.len() == 1 && work[0] < 0.0;
                let (lwork, liwork) = if query {
                    (-1, -1)
                } else {
                    (work.len() as Int, iwork.len() as Int)
                };
                // SAFETY: `a` holds `n · n` elements, `w` n, `work` lwork, `iwork` liwork.
                unsafe {
                    $syevd(
                        (&jobz as *const u8).cast(),
                        (&uplo as *const u8).cast(),
                        &n,
                        a.as_mut_ptr(),
                        &n.max(1),
                        w.as_mut_ptr(),
                        work.as_mut_ptr(),
                        &lwork,
                        iwork.as_mut_ptr(),
                        &liwork,
                        &mut info,
                    )
                };
                info
            }
            fn trtri(uplo: u8, n: Int, a: &mut [Self]) -> Int {
                let mut info = 0;
                let diag = b'N';
                // SAFETY: `a` holds `n · n` elements.
                unsafe {
                    $trtri(
                        (&uplo as *const u8).cast(),
                        (&diag as *const u8).cast(),
                        &n,
                        a.as_mut_ptr(),
                        &n.max(1),
                        &mut info,
                    )
                };
                info
            }
        }
    };
}

impl_lapack!(
    f32, spotrf, sgetrf, sgetrs, sgetri, sgeqrf, sorgqr, sgesdd, ssyevd, strtri
);
impl_lapack!(
    f64, dpotrf, dgetrf, dgetrs, dgetri, dgeqrf, dorgqr, dgesdd, dsyevd, dtrtri
);

/// A workspace query's size (LAPACK writes it to `work[0]`).
fn query<T: Lapack>(f: impl FnOnce(&mut [T]) -> Int) -> usize {
    let mut w = [T::default(); 1];
    // A negative first element marks the query (lwork = −1).
    w[0] = T::NEG_ONE;
    f(&mut w);
    w[0].to_usize().max(1)
}

fn transpose<T: Copy>(a: &[T], m: usize, n: usize) -> Vec<T> {
    let mut t = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            t.push(a[i * n + j]);
        }
    }
    t
}

/// Zeros the triangle a factor does not use (row-major), one row slice at a time.
fn clear_triangle<T: Lapack>(a: &mut [T], n: usize, upper: bool) {
    for i in 0..n {
        let row = &mut a[i * n..(i + 1) * n];
        if upper {
            row[..i].fill(T::default());
        } else {
            row[i + 1..].fill(T::default());
        }
    }
}

/// Cholesky in place. The row-major buffer is `Aᵀ = A`, whose upper triangle is `A`'s
/// lower one.
pub(crate) fn cholesky_in_place<T: Lapack>(
    a: &mut [T],
    n: usize,
    upper: bool,
) -> Result<(), String> {
    // Row-major lower ⇔ column-major upper.
    let info = T::potrf(if upper { b'L' } else { b'U' }, n as Int, a);
    if info > 0 {
        return Err("the matrix is not positive definite".into());
    }
    clear_triangle(a, n, upper);
    Ok(())
}

/// The inverse in place: `inv(Aᵀ) = inv(A)ᵀ`, so the row-major buffer needs no transpose.
pub(crate) fn inv_in_place<T: Lapack>(out: &mut [T], n: usize) {
    let mut ipiv = vec![0; n.max(1)];
    T::getrf(n as Int, n as Int, out, &mut ipiv);
    let lwork = query::<T>(|w| T::getri(n as Int, &mut out.to_vec(), &ipiv, w));
    let mut work = vec![T::default(); lwork];
    T::getri(n as Int, out, &ipiv, &mut work);
}

/// The inverse of a triangle in place: row-major lower ⇔ column-major upper.
pub(crate) fn tri_inv_in_place<T: Lapack>(a: &mut [T], n: usize, upper: bool) {
    T::trtri(if upper { b'L' } else { b'U' }, n as Int, a);
    clear_triangle(a, n, upper);
}

/// LU with partial pivoting: the packed factors of `P A` (row-major) and LAPACK's 0-based
/// pivots.
pub(crate) fn lu<T: Lapack>(a: &[T], n: usize, out: &mut [T], pivots: &mut [usize]) {
    let mut cm = transpose(a, n, n);
    let mut ipiv = vec![0; n.max(1)];
    T::getrf(n as Int, n as Int, &mut cm, &mut ipiv);
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = cm[j * n + i];
        }
        pivots[i] = (ipiv[i] - 1) as usize;
    }
}

/// The reduced QR factorisation.
pub(crate) fn qr<T: Lapack>(a: &[T], m: usize, n: usize, q: &mut [T], r: &mut [T]) {
    let k = m.min(n);
    let mut cm = transpose(a, m, n);
    let mut tau = vec![T::default(); k.max(1)];
    let lwork = query::<T>(|w| T::geqrf(m as Int, n as Int, &mut cm.clone(), &mut tau.clone(), w));
    let mut work = vec![T::default(); lwork];
    T::geqrf(m as Int, n as Int, &mut cm, &mut tau, &mut work);
    for i in 0..k {
        for j in 0..n {
            r[i * n + j] = if j >= i { cm[j * m + i] } else { T::default() };
        }
    }
    // Q: the first k columns (column-major m × k, lda = m).
    let mut qm = cm[..m * k].to_vec();
    let lwork = query::<T>(|w| T::orgqr(m as Int, k as Int, k as Int, &mut qm.clone(), &tau, w));
    let mut work = vec![T::default(); lwork];
    T::orgqr(m as Int, k as Int, k as Int, &mut qm, &tau, &mut work);
    for i in 0..m {
        for j in 0..k {
            q[i * k + j] = qm[j * m + i];
        }
    }
}

/// The SVD, on `Aᵀ` (the row-major buffer): `Aᵀ = V S Uᵀ`, so LAPACK's `U` is `V` and its
/// `Vt` is `Uᵀ`, which read row-major are exactly `Vt` and `U`.
pub(crate) fn svd<T: Lapack>(
    a: &[T],
    m: usize,
    n: usize,
    uv: Option<(&mut [T], &mut [T])>,
    s: &mut [T],
) -> Result<(), String> {
    let k = m.min(n);
    let mut buf = a.to_vec();
    let mut iwork = vec![0; 8 * k.max(1)];
    let (mm, nn) = (n as Int, m as Int);
    let info = match uv {
        Some((u, vt)) => {
            let lwork = query::<T>(|w| {
                T::gesdd(
                    b'A',
                    mm,
                    nn,
                    &mut buf.clone(),
                    &mut s.to_vec(),
                    &mut vt.to_vec(),
                    mm.max(1),
                    &mut u.to_vec(),
                    nn.max(1),
                    w,
                    &mut iwork.clone(),
                )
            });
            let mut work = vec![T::default(); lwork];
            T::gesdd(
                b'A',
                mm,
                nn,
                &mut buf,
                s,
                vt,
                mm.max(1),
                u,
                nn.max(1),
                &mut work,
                &mut iwork,
            )
        }
        None => {
            let (mut du, mut dv) = (vec![T::default(); 1], vec![T::default(); 1]);
            let lwork = query::<T>(|w| {
                T::gesdd(
                    b'N',
                    mm,
                    nn,
                    &mut buf.clone(),
                    &mut s.to_vec(),
                    &mut du.clone(),
                    1,
                    &mut dv.clone(),
                    1,
                    w,
                    &mut iwork.clone(),
                )
            });
            let mut work = vec![T::default(); lwork];
            T::gesdd(
                b'N', mm, nn, &mut buf, s, &mut du, 1, &mut dv, 1, &mut work, &mut iwork,
            )
        }
    };
    if info != 0 {
        return Err(format!("the SVD did not converge (info {info})"));
    }
    Ok(())
}

/// The symmetric eigendecomposition: row-major lower ⇔ column-major upper; the vectors
/// come back column-major and are transposed into columns.
pub(crate) fn eigh<T: Lapack>(
    a: &[T],
    n: usize,
    upper: bool,
    w: &mut [T],
    v: Option<&mut [T]>,
) -> Result<(), String> {
    let mut buf = a.to_vec();
    let jobz = if v.is_some() { b'V' } else { b'N' };
    let uplo = if upper { b'L' } else { b'U' };
    // Workspace query.
    let mut wq = [T::default(); 1];
    wq[0] = T::NEG_ONE;
    let mut iq = [0; 1];
    T::syevd(
        jobz,
        uplo,
        n as Int,
        &mut buf.clone(),
        &mut w.to_vec(),
        &mut wq,
        &mut iq,
    );
    let mut work = vec![T::default(); wq[0].to_usize().max(1)];
    let mut iwork = vec![0; iq[0].max(1) as usize];
    let info = T::syevd(jobz, uplo, n as Int, &mut buf, w, &mut work, &mut iwork);
    if info != 0 {
        return Err(format!(
            "the eigendecomposition did not converge (info {info})"
        ));
    }
    if let Some(v) = v {
        // Column-major Z: element (i, j) at buf[j · n + i].
        for i in 0..n {
            for j in 0..n {
                v[i * n + j] = buf[j * n + i];
            }
        }
    }
    Ok(())
}

/// `X = A⁻¹ B`: the row-major `A` is `Aᵀ` column-major, so `getrf` on it and `getrs` with
/// `trans = 'T'` solve `A X = B`; `B` and `X` are transposed to and from column-major.
pub(crate) fn solve<T: Lapack>(a: &[T], n: usize, b: &[T], k: usize, out: &mut [T]) {
    let mut f = a.to_vec();
    let mut ipiv = vec![0; n.max(1)];
    T::getrf(n as Int, n as Int, &mut f, &mut ipiv);
    let mut x = transpose(b, n, k);
    T::getrs(b'T', n as Int, k as Int, &f, &ipiv, &mut x);
    for i in 0..n {
        for j in 0..k {
            out[i * k + j] = x[j * n + i];
        }
    }
}
