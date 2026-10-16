//! Reductions and matrix multiplication, checked against an independent reference.

mod common;

use common::{Ref, Rng, assert_matches, matmul_ref, random_layout};
use nuvect_core::*;

/// Random cases per check (fewer under Miri, which is ~1000x slower).
const CASES: usize = if cfg!(miri) { 2 } else { 200 };

/// Asserts f32 results within 2 ulps of the reference (NaN matches NaN).
fn assert_close(array: &Array, expected: &Ref<f32>, context: &str) {
    assert_eq!(array.shape(), expected.shape.as_slice(), "{context}: shape");
    for (i, (g, e)) in array
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(&expected.data)
        .enumerate()
    {
        let ok = g == e
            || (g.is_nan() && e.is_nan())
            || (g - e).abs() <= 2.0 * f32::EPSILON * e.abs().max(1e-30);
        assert!(ok, "{context}: element {i}: got {g}, expected {e}");
    }
}

/// Random axes of an `ndim`-dimensional array, in random (possibly negative) spelling,
/// with the sorted normalised axes the reference uses. Empty means "all".
fn random_axes(rng: &mut Rng, ndim: usize) -> (Vec<isize>, Vec<usize>) {
    let chosen: Vec<usize> = (0..ndim).filter(|_| rng.coin()).collect();
    let spelled = chosen
        .iter()
        .map(|&d| {
            if rng.coin() {
                d as isize - ndim as isize
            } else {
                d as isize
            }
        })
        .collect();
    let normalised = if chosen.is_empty() {
        (0..ndim).collect()
    } else {
        chosen
    };
    (spelled, normalised)
}

// ── User story 1: reductions ────────────────────────────────────────────────

#[test]
fn reduce_random_integers() {
    let mut rng = Rng::new(0x5E);
    for _ in 0..CASES {
        let shape = rng.shape();
        let values: Vec<i8> = (0..common::size(&shape))
            .map(|_| rng.next() as i8)
            .collect();
        let (a, r) = random_layout(&mut rng, &shape, values);
        let (axes, norm) = random_axes(&mut rng, shape.len());
        let keep = rng.coin();
        let s = sum(&a, &axes, keep).unwrap();
        assert_eq!(s.dtype(), DType::I32, "int8 sums widen to int32");
        assert_matches(
            &s,
            &r.reduce(&norm, keep, 0i32, |acc, x, _| acc.wrapping_add(x as i32)),
            "sum",
        );
        let p = prod(&a, &axes, keep).unwrap();
        assert_matches(
            &p,
            &r.reduce(&norm, keep, 1i32, |acc, x, _| acc.wrapping_mul(x as i32)),
            "prod",
        );
        if r.data.is_empty() || norm.iter().any(|&d| shape[d] == 0) {
            continue;
        }
        let first =
            |acc: Option<i8>, x: i8, pick: fn(i8, i8) -> i8| Some(acc.map_or(x, |a| pick(a, x)));
        let mx = r
            .reduce(&norm, keep, None, |acc, x, _| first(acc, x, std::cmp::max))
            .map(Option::unwrap);
        assert_matches(&max(&a, &axes, keep).unwrap(), &mx, "max");
        let mn = r
            .reduce(&norm, keep, None, |acc, x, _| first(acc, x, std::cmp::min))
            .map(Option::unwrap);
        assert_matches(&min(&a, &axes, keep).unwrap(), &mn, "min");
        assert_matches(
            &all(&a, &axes, keep).unwrap(),
            &r.reduce(&norm, keep, true, |acc, x, _| acc && x != 0),
            "all",
        );
        assert_matches(
            &any(&a, &axes, keep).unwrap(),
            &r.reduce(&norm, keep, false, |acc, x, _| acc || x != 0),
            "any",
        );
    }
}

#[test]
fn reduce_random_floats() {
    let mut rng = Rng::new(0xF1);
    for _ in 0..CASES {
        let shape = rng.shape();
        let values: Vec<f32> = (0..common::size(&shape))
            .map(|_| {
                if rng.below(50) == 0 {
                    f32::NAN
                } else {
                    rng.f64() as f32
                }
            })
            .collect();
        let (a, r) = random_layout(&mut rng, &shape, values);
        let (axes, norm) = random_axes(&mut rng, shape.len());
        let keep = rng.coin();
        // Sums accumulate in f64 (lane order may differ from the reference's sequential
        // order) and round once: within 2 ulps of the f64 reference rounded to f32.
        let sum_ref = r.reduce(&norm, keep, 0.0f64, |acc, x, _| acc + f64::from(x));
        assert_close(
            &sum(&a, &axes, keep).unwrap(),
            &sum_ref.map(|v| v as f32),
            "sum f32",
        );
        let count: usize = norm.iter().map(|&d| shape[d]).product();
        let scale = (1.0 / count as f64) as f32;
        assert_close(
            &mean(&a, &axes, keep).unwrap(),
            &sum_ref.map(|v| v as f32 * scale),
            "mean f32",
        );
        if norm.iter().any(|&d| shape[d] == 0) {
            continue;
        }
        let nan_max = |acc: Option<f32>, x: f32| {
            Some(acc.map_or(x, |a| {
                if a.is_nan() || x.is_nan() {
                    f32::NAN
                } else {
                    a.max(x)
                }
            }))
        };
        let mx = r
            .reduce(&norm, keep, None, |acc, x, _| nan_max(acc, x))
            .map(Option::unwrap);
        assert_matches(&max(&a, &axes, keep).unwrap(), &mx, "max f32");
    }
}

#[test]
fn reduce_dtypes() {
    let cases = [
        (DType::Bool, DType::I32),
        (DType::U8, DType::U32),
        (DType::U64, DType::U64),
        (DType::I16, DType::I32),
        (DType::I64, DType::I64),
        (DType::BF16, DType::BF16),
        (DType::Complex64, DType::Complex64),
    ];
    for (input, expected) in cases {
        let a = ones(&[2, 3], input).unwrap();
        assert_eq!(
            sum(&a, &[], false).unwrap().dtype(),
            expected,
            "sum {input}"
        );
        assert_eq!(
            prod(&a, &[0], false).unwrap().dtype(),
            expected,
            "prod {input}"
        );
    }
    assert_eq!(
        mean(&ones(&[4], DType::I32).unwrap(), &[], false)
            .unwrap()
            .dtype(),
        DType::F32
    );
    assert_eq!(
        all(&ones(&[4], DType::F32).unwrap(), &[], false)
            .unwrap()
            .dtype(),
        DType::Bool
    );
    let s = sum(
        &Array::from_slice(&[true, true, false], &[3]).unwrap(),
        &[],
        false,
    )
    .unwrap();
    assert_eq!(s.item::<i32>().unwrap(), 2);
    let z =
        Array::from_slice(&[Complex64::new(1.0, 1.0), Complex64::new(2.0, -3.0)], &[2]).unwrap();
    assert_eq!(
        sum(&z, &[], false).unwrap().item::<Complex64>().unwrap(),
        Complex64::new(3.0, -2.0)
    );
}

#[test]
fn reduce_shapes_and_identities() {
    let a = arange(0.0, 24.0, 1.0, DType::F32)
        .unwrap()
        .reshape(&[2, 3, 4])
        .unwrap();
    assert_eq!(
        sum(&a, &[0, 2], false).unwrap().to_vec::<f32>().unwrap(),
        vec![60.0, 92.0, 124.0]
    );
    assert_eq!(sum(&a, &[0, -1], true).unwrap().shape(), &[1, 3, 1]);
    assert_eq!(sum(&a, &[], true).unwrap().shape(), &[1, 1, 1]);
    assert_eq!(sum(&a, &[], false).unwrap().item::<f32>().unwrap(), 276.0);
    let scalar = Array::scalar(5i8);
    assert_eq!(sum(&scalar, &[], false).unwrap().item::<i32>().unwrap(), 5);
    let empty = zeros(&[3, 0], DType::F32).unwrap();
    assert_eq!(
        sum(&empty, &[1], false).unwrap().to_vec::<f32>().unwrap(),
        vec![0.0; 3]
    );
    assert_eq!(
        prod(&empty, &[1], false).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0; 3]
    );
    assert!(all(&empty, &[], false).unwrap().item::<bool>().unwrap());
    assert!(!any(&empty, &[], false).unwrap().item::<bool>().unwrap());
    assert!(
        mean(&empty, &[], false)
            .unwrap()
            .item::<f32>()
            .unwrap()
            .is_nan()
    );
    assert_eq!(sum(&empty, &[0], false).unwrap().shape(), &[0]);
}

#[test]
fn reductions_and_matmul_are_deterministic() {
    let x = arange(0.0, 90_000.0, 0.37, DType::F32).unwrap();
    let a = sum(&x, &[], false).unwrap().item::<f32>().unwrap();
    let b = sum(&x, &[], false).unwrap().item::<f32>().unwrap();
    assert_eq!(a.to_bits(), b.to_bits());
    if !cfg!(miri) {
        let m = sin(&arange(0.0, 90_000.0, 1.0, DType::F32)
            .unwrap()
            .reshape(&[300, 300])
            .unwrap())
        .unwrap();
        let p = matmul(&m, &m).unwrap().to_vec::<f32>().unwrap();
        let q = matmul(&m, &m).unwrap().to_vec::<f32>().unwrap();
        assert!(
            p.iter().zip(&q).all(|(x, y)| x.to_bits() == y.to_bits()),
            "threaded GEMM is deterministic"
        );
    }
}

#[test]
fn reduce_long_sum_is_accurate() {
    // 1e6 copies of 0.1f32: naive f32 accumulation drifts by ~1%; f64 accumulation does not.
    if cfg!(miri) {
        return;
    }
    let a = Array::full(&[1_000_000], 0.1f32).unwrap();
    let s = sum(&a, &[], false).unwrap().item::<f32>().unwrap();
    assert!((s - 100_000.0).abs() < 0.01, "{s}");
}

#[test]
fn reduce_errors() {
    let a = zeros(&[2, 3], DType::F32).unwrap();
    assert!(matches!(
        sum(&a, &[2], false),
        Err(Error::InvalidArgument(_))
    ));
    assert!(
        matches!(sum(&a, &[0, -2], false), Err(Error::InvalidArgument(_))),
        "duplicate axes"
    );
    let empty = zeros(&[2, 0], DType::F32).unwrap();
    assert!(matches!(
        max(&empty, &[1], false),
        Err(Error::InvalidArgument(_))
    ));
    assert!(
        max(&empty, &[0], false).is_ok(),
        "only the reduced axes must be non-empty"
    );
    let z = zeros(&[2], DType::Complex64).unwrap();
    assert!(matches!(
        min(&z, &[], false),
        Err(Error::UnsupportedDType { op: "min", .. })
    ));
}

// ── User story 2: argmax / argmin ───────────────────────────────────────────

#[test]
fn arg_reduce_random() {
    let mut rng = Rng::new(0xA9);
    for _ in 0..CASES {
        let shape: Vec<usize> = rng.shape().into_iter().collect();
        if shape.is_empty() {
            continue;
        }
        let values: Vec<f32> = (0..common::size(&shape))
            .map(|_| {
                if rng.below(40) == 0 {
                    f32::NAN
                } else {
                    rng.below(5) as f32
                }
            })
            .collect();
        let (a, r) = random_layout(&mut rng, &shape, values);
        let axis = rng.below(shape.len());
        let keep = rng.coin();
        let scan = |better: fn(f32, f32) -> bool| {
            move |acc: (f32, u32, bool), x: f32, i: usize| {
                let take = !acc.2 || (!acc.0.is_nan() && (x.is_nan() || better(acc.0, x)));
                if take { (x, i as u32, true) } else { acc }
            }
        };
        let expect_max = r
            .reduce(&[axis], keep, (0.0, 0, false), scan(|best, x| best < x))
            .map(|t| t.1);
        assert_matches(
            &argmax(&a, Some(axis as isize), keep).unwrap(),
            &expect_max,
            "argmax",
        );
        let expect_min = r
            .reduce(&[axis], keep, (0.0, 0, false), scan(|best, x| x < best))
            .map(|t| t.1);
        assert_matches(
            &argmin(&a, Some(axis as isize - shape.len() as isize), keep).unwrap(),
            &expect_min,
            "argmin",
        );
    }
}

#[test]
fn arg_reduce_rules() {
    let a = Array::from_slice(&[1i32, 9, 3, 7, 2, 7], &[2, 3]).unwrap();
    assert_eq!(
        argmax(&a, Some(1), false).unwrap().to_vec::<u32>().unwrap(),
        vec![1, 0]
    );
    assert_eq!(argmax(&a, None, false).unwrap().item::<u32>().unwrap(), 1);
    assert_eq!(argmin(&a, None, true).unwrap().shape(), &[1, 1]);
    assert_eq!(
        argmax(&a, Some(0), true).unwrap().to_vec::<u32>().unwrap(),
        vec![1, 0, 1]
    );
    let nan = Array::from_slice(&[1.0f32, f32::NAN, 5.0, f32::NAN], &[4]).unwrap();
    assert_eq!(argmax(&nan, None, false).unwrap().item::<u32>().unwrap(), 1);
    assert_eq!(argmin(&nan, None, false).unwrap().item::<u32>().unwrap(), 1);
    assert!(argmax(&zeros(&[0], DType::F32).unwrap(), None, false).is_err());
    assert!(matches!(
        argmax(&zeros(&[2], DType::Complex64).unwrap(), None, false),
        Err(Error::UnsupportedDType { .. })
    ));
    assert!(argmax(&a, Some(2), false).is_err());
}

// ── User story 3: matmul ────────────────────────────────────────────────────

/// Checks `matmul` on random shapes and layouts for one dtype. `eps` is the unit roundoff.
fn check_matmul<T: Element>(
    rng: &mut Rng,
    to: fn(f64) -> T,
    from: fn(T) -> f64,
    eps: f64,
    scale: f64,
) {
    for _ in 0..CASES / 2 {
        let (m, k, n) = (1 + rng.below(6), rng.below(6), 1 + rng.below(6));
        let batch: Vec<usize> = (0..rng.below(3)).map(|_| 1 + rng.below(3)).collect();
        let bshape = rng.broadcastable(&batch);
        let mut sa = if rng.coin() {
            batch.clone()
        } else {
            bshape.clone()
        };
        let mut sb = if sa == batch { bshape } else { batch };
        sa.extend([m, k]);
        sb.extend([k, n]);
        let mut values = |shape: &[usize]| -> Vec<f64> {
            (0..common::size(shape))
                .map(|_| (rng.f64() * scale).round() / 8.0)
                .collect()
        };
        let (va, vb) = (values(&sa), values(&sb));
        let (a, ra) = random_layout(rng, &sa, va.iter().map(|&v| to(v)).collect::<Vec<T>>());
        let (b, rb) = random_layout(rng, &sb, vb.iter().map(|&v| to(v)).collect::<Vec<T>>());
        let expected = matmul_ref(&ra.map(from), &rb.map(from));
        let got = matmul(&a, &b).unwrap();
        assert_eq!(got.shape(), expected.shape.as_slice(), "{sa:?} @ {sb:?}");
        let bound = |x: &Ref<f64>| x.data.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let tol = (k.max(1) as f64 + 2.0) * eps * bound(&ra.map(from)) * bound(&rb.map(from)) * 4.0;
        for (g, e) in got.to_vec::<T>().unwrap().into_iter().zip(&expected.data) {
            assert!(
                (from(g) - e).abs() <= tol,
                "{} {sa:?} @ {sb:?}: {} vs {e} (tol {tol})",
                T::DTYPE,
                from(g)
            );
        }
    }
}

#[test]
fn matmul_random() {
    let mut rng = Rng::new(0x3A);
    check_matmul::<f32>(
        &mut rng,
        |v| v as f32,
        f64::from,
        f64::from(f32::EPSILON),
        40.0,
    );
    check_matmul::<f64>(&mut rng, |v| v, |v| v, f64::EPSILON, 40.0);
    check_matmul::<bf16>(&mut rng, bf16::from_f64, bf16::to_f64, 2f64.powi(-8), 16.0);
    if !cfg!(miri) {
        check_matmul::<f16>(&mut rng, f16::from_f64, f16::to_f64, 2f64.powi(-11), 16.0);
    }
    // Complex: compare with a real-arithmetic expansion.
    let a = Array::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(0.0, 1.0)],
        &[1, 2],
    )
    .unwrap();
    let b =
        Array::from_slice(&[Complex64::new(3.0, 0.0), Complex64::new(0.0, -1.0)], &[2]).unwrap();
    assert_eq!(
        matmul(&a, &b).unwrap().to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(4.0, 6.0)]
    );
}

#[test]
fn matmul_shapes() {
    let m = |shape: &[usize]| ones(shape, DType::F32).unwrap();
    assert_eq!(matmul(&m(&[2, 3]), &m(&[3, 4])).unwrap().shape(), &[2, 4]);
    assert_eq!(matmul(&m(&[3]), &m(&[3, 4])).unwrap().shape(), &[4]);
    assert_eq!(matmul(&m(&[2, 3]), &m(&[3])).unwrap().shape(), &[2]);
    assert_eq!(matmul(&m(&[3]), &m(&[3])).unwrap().shape(), &[] as &[usize]);
    assert_eq!(
        matmul(&m(&[5, 1, 2, 3]), &m(&[4, 3, 2])).unwrap().shape(),
        &[5, 4, 2, 2]
    );
    assert_eq!(
        matmul(&m(&[7, 2, 3]), &m(&[3, 4])).unwrap().shape(),
        &[7, 2, 4]
    );
    let k0 = matmul(&m(&[2, 0]), &m(&[0, 3])).unwrap();
    assert_eq!(k0.to_vec::<f32>().unwrap(), vec![0.0; 6]);
    // Mixed int and float promote to float.
    let mixed = matmul(&ones(&[2, 2], DType::I32).unwrap(), &m(&[2, 2])).unwrap();
    assert_eq!(
        (mixed.dtype(), mixed.to_vec::<f32>().unwrap()),
        (DType::F32, vec![2.0; 4])
    );
    // Transposed operand, read without copying.
    let x = arange(0.0, 6.0, 1.0, DType::F64)
        .unwrap()
        .reshape(&[2, 3])
        .unwrap();
    assert_eq!(
        matmul(&x, &x.t().unwrap())
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![5.0, 14.0, 14.0, 50.0]
    );
}

#[test]
fn matmul_errors() {
    let f = |shape: &[usize]| zeros(shape, DType::F32).unwrap();
    assert!(matches!(
        matmul(&f(&[2, 3]), &f(&[4, 2])),
        Err(Error::ShapeMismatch { op: "matmul", .. })
    ));
    assert!(matches!(
        matmul(&Array::scalar(1.0f32), &f(&[2])),
        Err(Error::InvalidArgument(_))
    ));
    let i = zeros(&[2, 2], DType::I32).unwrap();
    assert!(matches!(
        matmul(&i, &i),
        Err(Error::UnsupportedDType {
            op: "matmul",
            dtype: DType::I32
        })
    ));
    assert!(matches!(
        matmul(&f(&[2, 2, 3]), &f(&[3, 3, 2])),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
#[cfg(not(debug_assertions))]
fn perf_matmul_512() {
    let a = ones(&[512, 512], DType::F32).unwrap();
    let b = ones(&[512, 512], DType::F32).unwrap();
    let start = std::time::Instant::now();
    matmul(&a, &b).unwrap().eval().unwrap();
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 0.05,
        "512^3 matmul took {elapsed:?}"
    );
}
