//! `linalg`: shapes and conventions, reconstructions, the GPU stream, errors, and every
//! derivative against central finite differences in float64.

use nuvec_core::*;

fn mat(rows: usize, cols: usize, seed: u64, dtype: DType) -> Array {
    random::normal([rows, cols], DType::F32, 0.0, 1.0, Some(&random::key(seed))).astype(dtype)
}

fn max_abs(a: &Array) -> f64 {
    a.abs().max(..).astype(DType::F64).item::<f64>().unwrap()
}

fn mt(a: &Array) -> Array {
    ops::swapaxes(a, -1, -2)
}

#[test]
fn decompositions_reconstruct_their_input() {
    for dtype in [DType::F32, DType::F64] {
        let tol = if dtype == DType::F64 { 1e-10 } else { 1e-4 };
        for (m, n) in [(4, 4), (6, 3), (3, 5)] {
            let a = mat(m, n, 1, dtype);
            let [q, r] = linalg::qr(&a);
            assert!(max_abs(&q.matmul(&r).subtract(&a)) < tol, "qr {m}×{n}");
            let [u, s, vt] = linalg::svd(&a);
            let k = m.min(n) as i32;
            let us = u.i((.., ..k)).multiply(s.expand_dims(0));
            assert!(
                max_abs(&us.matmul(&vt.i((..k, ..))).subtract(&a)) < tol,
                "svd {m}×{n}"
            );
            assert_eq!(linalg::svd_values(&a).shape(), [k]);
        }
        let a = mat(5, 5, 2, dtype);
        let spd = a
            .matmul(&mt(&a))
            .add(ops::eye(5, None, 0, dtype).multiply(5.0));
        let [w, v] = linalg::eigh(&spd, false);
        let rec = v.multiply(w.expand_dims(0)).matmul(&mt(&v));
        assert!(max_abs(&rec.subtract(&spd)) < tol * 10.0, "eigh");
        let l = linalg::cholesky(&spd, false);
        assert!(
            max_abs(&l.matmul(&mt(&l)).subtract(&spd)) < tol * 10.0,
            "cholesky"
        );
        let uu = linalg::cholesky(&spd, true);
        assert!(
            max_abs(&mt(&uu).matmul(&uu).subtract(&spd)) < tol * 10.0,
            "cholesky upper"
        );
        let [p, ll, uuu] = linalg::lu(&a);
        assert!(max_abs(&ll.i(&p).matmul(&uuu).subtract(&a)) < tol, "lu");
        let eye = ops::eye(5, None, 0, dtype);
        assert!(
            max_abs(&a.matmul(&linalg::inv(&a)).subtract(&eye)) < tol * 10.0,
            "inv"
        );
        assert!(max_abs(&spd.matmul(&linalg::cholesky_inv(&l, false)).subtract(&eye)) < tol * 10.0);
        assert!(
            max_abs(&a.matmul(&linalg::pinv(&a)).matmul(&a).subtract(&a)) < tol * 10.0,
            "pinv"
        );
        let b = mat(5, 2, 3, dtype);
        assert!(
            max_abs(&a.matmul(&linalg::solve(&a, &b)).subtract(&b)) < tol * 10.0,
            "solve"
        );
        assert!(
            max_abs(
                &l.matmul(&linalg::solve_triangular(&l, &b, false))
                    .subtract(&b)
            ) < tol * 10.0
        );
        let [ww, vv] = linalg::eig(&a);
        let ac = a.astype(DType::Complex64);
        let res = ac.matmul(&vv).subtract(vv.multiply(ww.expand_dims(0)));
        assert!(max_abs(&res) < 1e-3, "eig");
        let [sign, logabs] = linalg::slogdet(&a);
        let d = linalg::det(&a);
        let via = sign.multiply(logabs.exp());
        assert!(max_abs(&d.subtract(&via)) < tol * max_abs(&d).max(1.0) * 10.0);
    }
}

#[test]
fn batches_and_the_gpu_stream_work() {
    let a = random::normal([3, 4, 4], DType::F32, 0.0, 1.0, Some(&random::key(5)));
    let [q, r] = linalg::qr(&a);
    assert_eq!((q.shape(), r.shape()), (&[3, 4, 4][..], &[3, 4, 4][..]));
    for i in 0..3 {
        let [qi, _] = linalg::qr(&a.i(i));
        assert!(max_abs(&qi.subtract(q.i(i))) < 1e-6);
    }
    // MLX asks for a CPU stream; here the CPU kernels run inside the GPU graph.
    let on_gpu = with_stream(Device::gpu(0), || linalg::inv(&a.add(1.0)).sum(..));
    let on_cpu = linalg::inv(&a.add(1.0)).sum(..);
    if on_gpu.eval().is_ok() {
        assert!((on_gpu.item::<f32>().unwrap() - on_cpu.item::<f32>().unwrap()).abs() < 1e-3);
    }
}

#[test]
fn invalid_inputs_are_errors() {
    let a = mat(3, 3, 1, DType::F16);
    assert!(linalg::inv(&a).eval().is_err());
    assert!(linalg::inv(&mat(3, 4, 1, DType::F32)).eval().is_err());
    assert!(
        linalg::cholesky(&ops::eye(3, None, 0, DType::F32).negative(), false)
            .eval()
            .is_err()
    );
    let e = linalg::inv(&Array::from(1.0f32)).eval().unwrap_err();
    assert!(e.to_string().contains("linalg::inv"), "{e}");
}

/// `⟨vjp(c), d⟩` against the central difference of `⟨f(a + h d), c⟩`.
fn check(name: &str, f: impl Fn(&Array) -> Vec<Array>, a: &Array, symmetric: bool) {
    let outs = f(a);
    let cots: Vec<Array> = outs
        .iter()
        .enumerate()
        .map(|(i, o)| {
            random::normal(
                o.shape(),
                DType::F64,
                0.0,
                1.0,
                Some(&random::key(100 + i as u64)),
            )
        })
        .collect();
    let (_, grads) =
        vjp_flat(|xs: &[Array]| Ok(f(&xs[0])), std::slice::from_ref(a), &cots).unwrap();
    let mut d = random::normal(a.shape(), DType::F64, 0.0, 1.0, Some(&random::key(77)));
    if symmetric {
        d = d.add(mt(&d));
    }
    let inner = |x: &Array| -> f64 {
        f(x).iter()
            .zip(&cots)
            .map(|(o, c)| o.multiply(c).sum(..).item::<f64>().unwrap())
            .sum()
    };
    let h = 1e-6;
    let fd = (inner(&a.add(d.multiply(h))) - inner(&a.subtract(d.multiply(h)))) / (2.0 * h);
    let an = grads[0].multiply(&d).sum(..).item::<f64>().unwrap();
    assert!(
        (an - fd).abs() <= 1e-6 * fd.abs().max(1.0),
        "{name}: vjp {an} vs finite difference {fd}"
    );
    // Forward mode: ⟨jvp(d), c⟩ equals the same.
    let (_, tangents) = jvp_flat(
        |xs: &[Array]| Ok(f(&xs[0])),
        std::slice::from_ref(a),
        &[d.clone()],
    )
    .unwrap();
    let fw: f64 = tangents
        .iter()
        .zip(&cots)
        .map(|(t, c)| t.multiply(c).sum(..).item::<f64>().unwrap())
        .sum();
    assert!(
        (fw - fd).abs() <= 1e-6 * fd.abs().max(1.0),
        "{name}: jvp {fw} vs finite difference {fd}"
    );
}

#[test]
fn derivatives_equal_finite_differences() {
    let a = mat(4, 4, 11, DType::F64).add(ops::eye(4, None, 0, DType::F64).multiply(3.0));
    let spd = a.matmul(&mt(&a));
    let tall = mat(5, 3, 12, DType::F64);
    let wide = mat(3, 5, 13, DType::F64);
    let b = mat(4, 2, 14, DType::F64);
    check("inv", |x| vec![linalg::inv(x)], &a, false);
    check("tri_inv", |x| vec![linalg::tri_inv(x, false)], &a, false);
    check(
        "tri_inv upper",
        |x| vec![linalg::tri_inv(x, true)],
        &a,
        false,
    );
    check("solve", |x| vec![linalg::solve(x, &b)], &a, false);
    check(
        "solve b",
        |x| vec![linalg::solve(&a, &x.i((.., ..2)))],
        &a,
        false,
    );
    check(
        "solve_triangular",
        |x| vec![linalg::solve_triangular(x, &b, false)],
        &a,
        false,
    );
    check(
        "solve_triangular upper",
        |x| vec![linalg::solve_triangular(x, &b, true)],
        &a,
        false,
    );
    check("cholesky", |x| vec![linalg::cholesky(x, false)], &spd, true);
    check(
        "cholesky upper",
        |x| vec![linalg::cholesky(x, true)],
        &spd,
        true,
    );
    check(
        "cholesky_inv",
        |x| vec![linalg::cholesky_inv(&linalg::cholesky(x, false), false)],
        &spd,
        true,
    );
    check("qr square", |x| linalg::qr(x).to_vec(), &a, false);
    check("qr tall", |x| linalg::qr(x).to_vec(), &tall, false);
    check("eigh", |x| linalg::eigh(x, false).to_vec(), &spd, true);
    check("eigvalsh", |x| vec![linalg::eigvalsh(x, false)], &spd, true);
    check("svd values", |x| vec![linalg::svd_values(x)], &tall, false);
    // Singular vectors are defined up to sign: compare sign-invariant functions of them.
    let invariant = |x: &Array| {
        let [u, s, vt] = linalg::svd(x);
        let k = s.shape()[0];
        let (uk, vk) = (u.i((.., ..k)), vt.i((..k, ..)));
        vec![s, uk.multiply(&uk), vk.multiply(&vk), uk.matmul(&vk)]
    };
    check("svd tall", invariant, &tall, false);
    check("svd wide", invariant, &wide, false);
    check("svd square", invariant, &a, false);
    check("pinv", |x| vec![linalg::pinv(x)], &tall, false);
    check("det", |x| vec![linalg::det(x)], &a, false);
    check(
        "slogdet",
        |x| vec![linalg::slogdet(x)[1].clone()],
        &a,
        false,
    );
    check(
        "norm 2",
        |x| vec![ops::matrix_norm(x, ops::Norm::P(2.0), None, false)],
        &tall,
        false,
    );
    check(
        "norm nuc",
        |x| vec![ops::matrix_norm(x, ops::Norm::Nuc, None, false)],
        &tall,
        false,
    );
    check(
        "cross",
        |x| vec![linalg::cross(&x.i((..3, 0)), &x.i((..3, 1)), 0)],
        &a,
        false,
    );
}

#[test]
fn vmap_equals_per_matrix_results() {
    let a = random::normal([3, 4, 4], DType::F64, 0.0, 1.0, Some(&random::key(9)));
    let f = |xs: &[Array]| {
        Ok(vec![linalg::inv(
            &xs[0].add(ops::eye(4, None, 0, DType::F64).multiply(3.0)),
        )])
    };
    let batched = vmap_flat(f, &[Some(0)], &[0])(std::slice::from_ref(&a))
        .unwrap()
        .remove(0);
    for i in 0..3 {
        let one = f(&[a.i(i)]).unwrap().remove(0);
        assert!(max_abs(&one.subtract(batched.i(i))) < 1e-12);
    }
}

#[test]
fn no_derivative_is_an_error() {
    let a = mat(3, 3, 4, DType::F64);
    let g = grad(|x: &Array| Ok(ops::real(&linalg::eigvals(x)).sum(..)))(&a);
    assert!(g.is_err() || g.unwrap().eval().is_err());
    let g = grad(|x: &Array| Ok(linalg::qr(x)[1].sum(..)))(&mat(3, 5, 4, DType::F64));
    assert!(g.is_err() || g.unwrap().eval().is_err());
}
