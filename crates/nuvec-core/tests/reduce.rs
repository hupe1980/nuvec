//! Reductions and matrix multiplication, checked against an independent reference.

#[path = "common/api.rs"]
mod api;
mod common;
#[path = "common/fixture.rs"]
mod fixture;

use common::{Ref, Rng, assert_matches, matmul_ref, random_layout};
use nuvec_core::ops;
use nuvec_core::*;

/// Applies reduction `$f` (or its `_keepdims` twin `$fk`) over `$axes` (`None`: every axis).
macro_rules! red {
    ($a:expr, $axes:expr, $keep:expr, $f:ident, $fk:ident) => {
        match (&$axes, $keep) {
            (Some(ax), false) => $a.$f(ax.clone()),
            (Some(ax), true) => $a.$fk(ax.clone()),
            (None, false) => $a.$f(..),
            (None, true) => $a.$fk(..),
        }
    };
}

/// Random cases per check (fewer under Miri, which is ~1000x slower).
const CASES: usize = if cfg!(miri) { 2 } else { 200 };

/// Asserts f32 results within 2 ulps of the reference (NaN matches NaN).
fn assert_close(array: &Array, expected: &Ref<f32>, context: &str) {
    assert_eq!(
        array
            .shape()
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>(),
        expected.shape,
        "{context}: shape"
    );
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
/// with the sorted normalised axes the reference uses. `None` means "all" (`..`).
fn random_axes(rng: &mut Rng, ndim: usize) -> (Option<Vec<i32>>, Vec<usize>) {
    let chosen: Vec<usize> = (0..ndim).filter(|_| rng.coin()).collect();
    let spelled: Vec<i32> = chosen
        .iter()
        .map(|&d| {
            if rng.coin() {
                d as i32 - ndim as i32
            } else {
                d as i32
            }
        })
        .collect();
    if chosen.is_empty() {
        (None, (0..ndim).collect())
    } else {
        (Some(spelled), chosen)
    }
}

// ── User story 1: reductions ────────────────────────────────────────────────

#[test]
fn reduce_random_integers() {
    common::on_cpu();
    let mut rng = Rng::new(0x5E);
    for _ in 0..CASES {
        let shape = rng.shape();
        let values: Vec<i8> = (0..common::size(&shape))
            .map(|_| rng.next() as i8)
            .collect();
        let (a, r) = random_layout(&mut rng, &shape, values);
        let (axes, norm) = random_axes(&mut rng, shape.len());
        let keep = rng.coin();
        // A reduction over no axes (a 0-D input, `..`) is the identity, as in MLX: no
        // widening there.
        let widened = |x: Array| {
            if shape.is_empty() {
                assert_eq!(x.dtype(), DType::I8, "0-D reductions are the identity");
                x.astype(DType::I32)
            } else {
                assert_eq!(x.dtype(), DType::I32, "int8 sums widen to int32");
                x
            }
        };
        let s = widened(red!(a, axes, keep, sum, sum_keepdims));
        assert_matches(
            &s,
            &r.reduce(&norm, keep, 0i32, |acc, x, _| acc.wrapping_add(x as i32)),
            "sum",
        );
        let p = widened(red!(a, axes, keep, prod, prod_keepdims));
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
        assert_matches(&red!(a, axes, keep, max, max_keepdims), &mx, "max");
        let mn = r
            .reduce(&norm, keep, None, |acc, x, _| first(acc, x, std::cmp::min))
            .map(Option::unwrap);
        assert_matches(&red!(a, axes, keep, min, min_keepdims), &mn, "min");
        assert_matches(
            &red!(a, axes, keep, all, all_keepdims),
            &r.reduce(&norm, keep, true, |acc, x, _| acc && x != 0),
            "all",
        );
        assert_matches(
            &red!(a, axes, keep, any, any_keepdims),
            &r.reduce(&norm, keep, false, |acc, x, _| acc || x != 0),
            "any",
        );
    }
}

#[test]
fn reduce_random_floats() {
    common::on_cpu();
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
            &red!(a, axes, keep, sum, sum_keepdims),
            &sum_ref.map(|v| v as f32),
            "sum f32",
        );
        let count: usize = norm.iter().map(|&d| shape[d]).product();
        let scale = (1.0 / count as f64) as f32;
        assert_close(
            &red!(a, axes, keep, mean, mean_keepdims),
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
        assert_matches(&red!(a, axes, keep, max, max_keepdims), &mx, "max f32");
    }
}

#[test]
fn reduce_dtypes() {
    common::on_cpu();
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
        let a = ops::ones([2, 3], input);
        assert_eq!(a.sum(..).dtype(), expected, "sum {input}");
        assert_eq!(a.prod(0).dtype(), expected, "prod {input}");
    }
    assert_eq!(ops::ones([4], DType::I32).mean(..).dtype(), DType::F32);
    assert_eq!(ops::ones([4], DType::F32).all(..).dtype(), DType::Bool);
    let s = Array::from([true, true, false]).sum(..);
    assert_eq!(s.item::<i32>().unwrap(), 2);
    let z = Array::from([Complex64::new(1.0, 1.0), Complex64::new(2.0, -3.0)]);
    assert_eq!(
        z.sum(..).item::<Complex64>().unwrap(),
        Complex64::new(3.0, -2.0)
    );
}

#[test]
fn reduce_shapes_and_identities() {
    common::on_cpu();
    let a = ops::arange(0.0, 24.0, 1.0).reshape([2, 3, 4]);
    assert_eq!(
        a.sum([0, 2]).to_vec::<f32>().unwrap(),
        vec![60.0, 92.0, 124.0]
    );
    assert_eq!(a.sum_keepdims([0, -1]).shape(), [1, 3, 1]);
    assert_eq!(a.sum_keepdims(..).shape(), [1, 1, 1]);
    assert_eq!(a.sum(..).item::<f32>().unwrap(), 276.0);
    // Over no axes (a 0-D input), a reduction is the identity, as in MLX: no widening.
    let scalar = Array::from(5i8);
    assert_eq!(scalar.sum(..).item::<i8>().unwrap(), 5);
    let empty = ops::zeros([3, 0], DType::F32);
    assert_eq!(empty.sum(1).to_vec::<f32>().unwrap(), vec![0.0; 3]);
    assert_eq!(empty.prod(1).to_vec::<f32>().unwrap(), vec![1.0; 3]);
    assert!(empty.all(..).item::<bool>().unwrap());
    assert!(!empty.any(..).item::<bool>().unwrap());
    assert!(empty.mean(..).item::<f32>().unwrap().is_nan());
    assert_eq!(empty.sum(0).shape(), [0]);
}

#[test]
fn reductions_and_matmul_are_deterministic() {
    common::on_cpu();
    let x = ops::arange(0.0, 90_000.0, 0.37);
    let a = x.sum(..).item::<f32>().unwrap();
    let b = x.sum(..).item::<f32>().unwrap();
    assert_eq!(a.to_bits(), b.to_bits());
    if !cfg!(miri) {
        let m = ops::arange(0.0, 90_000.0, 1.0).reshape([300, 300]).sin();
        let p = m.matmul(&m).to_vec::<f32>().unwrap();
        let q = m.matmul(&m).to_vec::<f32>().unwrap();
        assert!(
            p.iter().zip(&q).all(|(x, y)| x.to_bits() == y.to_bits()),
            "threaded GEMM is deterministic"
        );
    }
}

#[test]
fn reduce_long_sum_is_accurate() {
    common::on_cpu();
    // 1e6 copies of 0.1f32: naive f32 accumulation drifts by ~1%; f64 accumulation does not.
    if cfg!(miri) {
        return;
    }
    let a = ops::full([1_000_000], 0.1f32);
    let s = a.sum(..).item::<f32>().unwrap();
    assert!((s - 100_000.0).abs() < 0.01, "{s}");
}

#[test]
fn reduce_errors() {
    common::on_cpu();
    let a = ops::zeros([2, 3], DType::F32);
    assert!(matches!(
        a.sum(2).check(),
        Err(Error::Op {
            kind: OpErrorKind::InvalidAxis { axis: 2, ndim: 2 },
            ..
        })
    ));
    assert!(
        matches!(
            a.sum([0, -2]).check(),
            Err(Error::Op {
                kind: OpErrorKind::InvalidAxis { .. },
                ..
            })
        ),
        "duplicate axes"
    );
    let empty = ops::zeros([2, 0], DType::F32);
    assert!(matches!(
        empty.max(1).check(),
        Err(Error::Op {
            kind: OpErrorKind::InvalidArgument(_),
            ..
        })
    ));
    assert!(
        empty.max(0).check().is_ok(),
        "only the reduced axes must be non-empty"
    );
    let z = ops::zeros([2], DType::Complex64);
    assert!(matches!(
        z.min(..).check(),
        Err(Error::Op {
            op: "min",
            kind: OpErrorKind::UnsupportedDType(_),
            ..
        })
    ));
}

// ── User story 2: argmax / argmin ───────────────────────────────────────────

#[test]
fn arg_reduce_random() {
    common::on_cpu();
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
        let am = if keep {
            a.argmax_keepdims(axis)
        } else {
            a.argmax(axis)
        };
        assert_matches(&am, &expect_max, "argmax");
        let expect_min = r
            .reduce(&[axis], keep, (0.0, 0, false), scan(|best, x| x < best))
            .map(|t| t.1);
        let neg = axis as i32 - shape.len() as i32;
        let an = if keep {
            a.argmin_keepdims(neg)
        } else {
            a.argmin(neg)
        };
        assert_matches(&an, &expect_min, "argmin");
    }
}

#[test]
fn arg_reduce_rules() {
    common::on_cpu();
    let a = array![[1, 9, 3], [7, 2, 7]];
    assert_eq!(a.argmax(1).to_vec::<u32>().unwrap(), vec![1, 0]);
    assert_eq!(a.argmax(..).item::<u32>().unwrap(), 1);
    assert_eq!(a.argmin_keepdims(..).shape(), [1, 1]);
    assert_eq!(a.argmax_keepdims(0).to_vec::<u32>().unwrap(), vec![1, 0, 1]);
    let nan = Array::from([1.0f32, f32::NAN, 5.0, f32::NAN]);
    assert_eq!(nan.argmax(..).item::<u32>().unwrap(), 1);
    assert_eq!(nan.argmin(..).item::<u32>().unwrap(), 1);
    assert!(ops::zeros([0], DType::F32).argmax(..).check().is_err());
    assert!(matches!(
        ops::zeros([2], DType::Complex64).argmax(..).check(),
        Err(Error::Op {
            kind: OpErrorKind::UnsupportedDType(_),
            ..
        })
    ));
    assert!(a.argmax(2).check().is_err());
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
        let got = a.matmul(&b);
        assert_eq!(
            got.shape().iter().map(|&d| d as usize).collect::<Vec<_>>(),
            expected.shape,
            "{sa:?} @ {sb:?}"
        );
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
    common::on_cpu();
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
        [1, 2],
    );
    let b = Array::from([Complex64::new(3.0, 0.0), Complex64::new(0.0, -1.0)]);
    assert_eq!(
        a.matmul(&b).to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(4.0, 6.0)]
    );
}

#[test]
fn matmul_shapes() {
    common::on_cpu();
    let m = |shape: &[i32]| ops::ones(shape, DType::F32);
    assert_eq!(m(&[2, 3]).matmul(&m(&[3, 4])).shape(), [2, 4]);
    assert_eq!(m(&[3]).matmul(&m(&[3, 4])).shape(), [4]);
    assert_eq!(m(&[2, 3]).matmul(&m(&[3])).shape(), [2]);
    assert_eq!(m(&[3]).matmul(&m(&[3])).shape(), [0i32; 0]);
    assert_eq!(
        m(&[5, 1, 2, 3]).matmul(&m(&[4, 3, 2])).shape(),
        [5, 4, 2, 2]
    );
    assert_eq!(m(&[7, 2, 3]).matmul(&m(&[3, 4])).shape(), [7, 2, 4]);
    let k0 = m(&[2, 0]).matmul(&m(&[0, 3]));
    assert_eq!(k0.to_vec::<f32>().unwrap(), vec![0.0; 6]);
    // Mixed int and float promote to float.
    let mixed = ops::ones([2, 2], DType::I32).matmul(&m(&[2, 2]));
    assert_eq!(
        (mixed.dtype(), mixed.to_vec::<f32>().unwrap()),
        (DType::F32, vec![2.0; 4])
    );
    // Transposed operand, read without copying.
    let x = ops::arange(0.0, 6.0, 1.0)
        .astype(DType::F64)
        .reshape([2, 3]);
    assert_eq!(
        x.matmul(&x.t()).to_vec::<f64>().unwrap(),
        vec![5.0, 14.0, 14.0, 50.0]
    );
}

#[test]
fn matmul_errors() {
    common::on_cpu();
    let f = |shape: &[i32]| ops::zeros(shape, DType::F32);
    assert!(matches!(
        f(&[2, 3]).matmul(&f(&[4, 2])).check(),
        Err(Error::Op {
            op: "matmul",
            kind: OpErrorKind::ShapeMismatch { .. },
            ..
        })
    ));
    assert!(matches!(
        Array::from(1.0f32).matmul(&f(&[2])).check(),
        Err(Error::Op {
            kind: OpErrorKind::InvalidArgument(_),
            ..
        })
    ));
    let i = ops::zeros([2, 2], DType::I32);
    assert!(matches!(
        i.matmul(&i).check(),
        Err(Error::Op {
            op: "matmul",
            kind: OpErrorKind::UnsupportedDType(DType::I32),
            ..
        })
    ));
    assert!(matches!(
        f(&[2, 2, 3]).matmul(&f(&[3, 3, 2])).check(),
        Err(Error::Op {
            kind: OpErrorKind::ShapeMismatch { .. },
            ..
        })
    ));
}

#[test]
#[cfg(not(debug_assertions))]
fn perf_matmul_512() {
    common::on_cpu();
    let a = ops::ones([512, 512], DType::F32);
    let b = ops::ones([512, 512], DType::F32);
    let start = std::time::Instant::now();
    a.matmul(&b).eval().unwrap();
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 0.05,
        "512^3 matmul took {elapsed:?}"
    );
}

// ── Keepdims twins, MLX API fixtures, var/std gradients ─────

/// The twelve reductions, by name: the plain result and its `_keepdims` twin over `axes`
/// (`None`: every axis; argmax/argmin take one axis).
fn reduce_pair(x: &Array, name: &str, axes: &Option<Vec<i32>>) -> (Array, Array) {
    macro_rules! pair {
        ($f:ident, $fk:ident) => {
            (red!(x, axes, false, $f, $fk), red!(x, axes, true, $f, $fk))
        };
    }
    match name {
        "sum" => pair!(sum, sum_keepdims),
        "mean" => pair!(mean, mean_keepdims),
        "max" => pair!(max, max_keepdims),
        "min" => pair!(min, min_keepdims),
        "prod" => pair!(prod, prod_keepdims),
        "all" => pair!(all, all_keepdims),
        "any" => pair!(any, any_keepdims),
        "logsumexp" => pair!(logsumexp, logsumexp_keepdims),
        "var" => pair!(var, var_keepdims),
        "std" => pair!(std, std_keepdims),
        "argmax" | "argmin" => {
            // One axis (or `..`, the flattened array).
            let axis = axes.as_ref().map(|v| {
                assert_eq!(v.len(), 1, "{name} takes one axis");
                v[0]
            });
            macro_rules! arg {
                ($f:ident, $fk:ident) => {
                    match axis {
                        Some(a) => (x.$f(a), x.$fk(a)),
                        None => (x.$f(..), x.$fk(..)),
                    }
                };
            }
            if name == "argmax" {
                arg!(argmax, argmax_keepdims)
            } else {
                arg!(argmin, argmin_keepdims)
            }
        }
        other => panic!("unknown reduction {other}"),
    }
}

/// The f64 reference of reduction `name` over `norm` (sorted axes) of `r`.
fn reduce_ref(r: &Ref<f64>, name: &str, norm: &[usize], int: bool) -> Ref<f64> {
    let fold =
        |init: f64, f: fn(f64, f64) -> f64| r.reduce(norm, true, init, move |a, x, _| f(a, x));
    let count = r.reduce(norm, true, 0.0, |a, _, _| a + 1.0);
    let sum = fold(0.0, |a, x| a + x);
    let mean = sum.zip(&count, |s, n| s / n);
    let var = || {
        let sq = r
            .zip(&mean, |x, m| (x - m) * (x - m))
            .reduce(norm, true, 0.0, |a, x, _| a + x);
        sq.zip(&count, |s, n| s / n)
    };
    let arg = |better: fn(f64, f64) -> bool| {
        r.reduce(norm, true, (0.0, 0.0, false), move |acc, x, i| {
            if !acc.2 || better(acc.0, x) {
                (x, i as f64, true)
            } else {
                acc
            }
        })
        .map(|t| t.1)
    };
    match name {
        "sum" => sum,
        "mean" => mean,
        "max" => fold(f64::NEG_INFINITY, f64::max),
        "min" => fold(f64::INFINITY, f64::min),
        "prod" if int => r
            .reduce(norm, true, 1i32, |a, x, _| a.wrapping_mul(x as i32))
            .map(f64::from),
        "prod" => fold(1.0, |a, x| a * x),
        "all" => fold(1.0, |a, x| f64::from(a != 0.0 && x != 0.0)),
        "any" => fold(0.0, |a, x| f64::from(a != 0.0 || x != 0.0)),
        "logsumexp" => {
            let m = fold(f64::NEG_INFINITY, f64::max);
            r.zip(&m, |x, m| (x - m).exp())
                .reduce(norm, true, 0.0, |a, x, _| a + x)
                .zip(&m, |s, m| s.ln() + m)
        }
        "var" => var(),
        "std" => var().map(f64::sqrt),
        "argmax" => arg(|best, x| best < x),
        "argmin" => arg(|best, x| x < best),
        other => panic!("unknown reduction {other}"),
    }
}

const REDUCTIONS: [&str; 12] = [
    "sum",
    "mean",
    "max",
    "min",
    "prod",
    "all",
    "any",
    "logsumexp",
    "var",
    "std",
    "argmax",
    "argmin",
];

#[test]
fn keepdims_twins_agree_with_each_other_and_the_reference() {
    common::on_cpu();
    let mut rng = Rng::new(0x062);
    for case in 0..CASES / 4 {
        let dtype = [DType::F32, DType::F16, DType::I32][case % 3];
        let mut shape = rng.shape();
        if shape.is_empty() {
            shape.push(1 + rng.below(5));
        }
        let n = common::size(&shape);
        // Values exactly representable in every dtype under test.
        let host: Vec<f64> = (0..n)
            .map(|_| match dtype {
                DType::I32 => rng.range(-3, 3) as f64,
                _ => (rng.f64() * 4.0).round() / 4.0,
            })
            .collect();
        let (a, r) = match dtype {
            DType::I32 => {
                let (a, r) =
                    random_layout(&mut rng, &shape, host.iter().map(|&v| v as i32).collect());
                (a, r.map(f64::from))
            }
            DType::F16 => {
                let (a, r) = random_layout(
                    &mut rng,
                    &shape,
                    host.iter().map(|&v| f16::from_f64(v)).collect(),
                );
                (a, r.map(f16::to_f64))
            }
            _ => {
                let (a, r) =
                    random_layout(&mut rng, &shape, host.iter().map(|&v| v as f32).collect());
                (a, r.map(f64::from))
            }
        };
        let (axes, norm) = random_axes(&mut rng, shape.len());
        for name in REDUCTIONS {
            let arg = name.starts_with("arg");
            let (axes, norm) = if arg {
                let axis = rng.below(shape.len());
                (Some(vec![axis as i32 - shape.len() as i32]), vec![axis])
            } else {
                (axes.clone(), norm.clone())
            };
            let ctx = format!("{name} {dtype} {shape:?} axes {axes:?}");
            let (plain, kept) = reduce_pair(&a, name, &axes);
            let expected = reduce_ref(&r, name, &norm, dtype == DType::I32);
            let kept_shape: Vec<i32> = expected.shape.iter().map(|&d| d as i32).collect();
            assert_eq!(kept.shape(), kept_shape.as_slice(), "{ctx}: keepdims shape");
            assert_eq!(plain.dtype(), kept.dtype(), "{ctx}: dtype");
            let plain_shape: Vec<i32> = (0..shape.len())
                .filter(|d| !norm.contains(d))
                .map(|d| shape[d] as i32)
                .collect();
            assert_eq!(plain.shape(), plain_shape.as_slice(), "{ctx}: shape");
            let as64 = |x: &Array| x.astype(DType::F64).to_vec::<f64>().unwrap();
            let (p, k) = (as64(&plain), as64(&kept.reshape(plain.shape())));
            assert!(
                p.iter().zip(&k).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{ctx}: keepdims twin {k:?} vs {p:?}"
            );
            let eps = match plain.dtype() {
                DType::F16 => 2f64.powi(-10),
                DType::F32 => f64::from(f32::EPSILON),
                _ => 0.0,
            };
            let count: usize = norm.iter().map(|&d| shape[d]).product();
            let tol = |e: f64| 4.0 * (count as f64 + 4.0) * eps * e.abs().max(1.0);
            for (i, (g, e)) in p.iter().zip(&expected.data).enumerate() {
                // Results beyond float16's range overflow to infinity.
                let e = &match plain.dtype() {
                    DType::F16 => f16::from_f64(*e).to_f64(),
                    _ => *e,
                };
                assert!(
                    g == e || (g - e).abs() <= tol(*e),
                    "{ctx}: element {i}: got {g}, expected {e}"
                );
            }
        }
    }
}

/// An MLX call recorded as Python, such as `mx.std(x, axis=0, ddof=1, keepdims=True)`.
#[derive(Debug, Default)]
struct Call {
    name: String,
    /// `None`: no `axis` argument (every axis, or the flattened array for scans).
    axis: Option<Vec<i32>>,
    keepdims: bool,
    ddof: i32,
    reverse: bool,
    inclusive: bool,
}

fn parse_call(expr: &str) -> Call {
    let rest = expr.strip_prefix("mx.").expect("mx. call");
    let (name, args) = rest.split_once('(').expect("call");
    let args = args.strip_suffix(')').expect("closing parenthesis");
    let mut call = Call {
        name: name.to_owned(),
        inclusive: true,
        ..Call::default()
    };
    // Split at top-level commas (not inside `[..]`).
    let mut parts = Vec::new();
    let (mut depth, mut start) = (0, 0);
    for (i, c) in args.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&args[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&args[start..]);
    assert_eq!(parts[0].trim(), "x", "{expr}");
    for part in &parts[1..] {
        let (key, value) = part.trim().split_once('=').expect("keyword argument");
        let flag = || match value {
            "True" => true,
            "False" => false,
            other => panic!("{expr}: not a bool: {other}"),
        };
        match key {
            "axis" => {
                let list = value.trim_start_matches('[').trim_end_matches(']');
                call.axis = Some(
                    list.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse().unwrap())
                        .collect(),
                );
            }
            "keepdims" => call.keepdims = flag(),
            "ddof" => call.ddof = value.parse().unwrap(),
            "reverse" => call.reverse = flag(),
            "inclusive" => call.inclusive = flag(),
            other => panic!("{expr}: unknown argument {other}"),
        }
    }
    call
}

/// Runs a parsed call with the nuvec method it corresponds to.
fn run_call(call: &Call, x: &Array) -> Array {
    let axes = &call.axis;
    let options = VarOptions {
        ddof: call.ddof,
        keepdims: call.keepdims,
    };
    let scan = ScanOptions {
        reverse: call.reverse,
        exclusive: !call.inclusive,
    };
    macro_rules! with_axes {
        ($ax:ident => $e:expr) => {
            match axes {
                Some(v) => {
                    let $ax = v.clone();
                    $e
                }
                None => {
                    let $ax = ..;
                    $e
                }
            }
        };
    }
    // Scans take one axis, or none (the flattened array).
    macro_rules! with_axis {
        ($ax:ident => $e:expr) => {
            match axes.as_deref() {
                Some(&[a]) => {
                    let $ax = a;
                    $e
                }
                None => {
                    let $ax = ..;
                    $e
                }
                Some(other) => panic!("a scan takes one axis, got {other:?}"),
            }
        };
    }
    match call.name.as_str() {
        "var" => with_axes!(ax => x.var_with(ax, options)),
        "std" => with_axes!(ax => x.std_with(ax, options)),
        "cumsum" => with_axis!(ax => x.cumsum_with(ax, scan)),
        "cumprod" => with_axis!(ax => x.cumprod_with(ax, scan)),
        "cummax" => with_axis!(ax => x.cummax_with(ax, scan)),
        "cummin" => with_axis!(ax => x.cummin_with(ax, scan)),
        "logcumsumexp" => with_axis!(ax => x.logcumsumexp_with(ax, scan)),
        name => {
            assert_eq!(call.ddof, 0, "ddof only for var/std");
            let (plain, kept) = reduce_pair(x, name, axes);
            if call.keepdims { kept } else { plain }
        }
    }
}

/// Allowed error, in ulps of the output dtype, per MLX call.
fn fixture_ulps(name: &str) -> f64 {
    match name {
        "var" | "std" | "logsumexp" => VAR_ULPS,
        "cumsum" | "cumprod" | "logcumsumexp" => SCAN_ULPS,
        _ => 0.0,
    }
}

/// Measured worst case: 36.4 ulps, `std` of an int32 array with ddof 1 (nuvec computes
/// `mean(a²) − mean(a)²`, which cancels when the mean is large against the spread; MLX's
/// result differs in the last bits). Next: var float32 8.3, float16 2.7.
const VAR_ULPS: f64 = 16.0;
/// Measured worst case: 2 ulps (GPU parallel prefix scans round in another order).
const SCAN_ULPS: f64 = 4.0;

#[test]
#[cfg_attr(miri, ignore = "hundreds of cases: covered natively")]
fn mlx_api_reduce_fixtures() {
    let cases = api::cases("reduce");
    assert!(cases.len() >= 100, "only {} reduce cases", cases.len());
    let mut devices = vec![Device::cpu()];
    if common::require_gpu().is_some() {
        devices.push(Device::gpu(0));
    }
    let mut failures = Vec::new();
    for device in devices {
        for case in &cases {
            let call = parse_call(&case.params["expr"]);
            let result = with_stream(device, || {
                let xs = api::inputs(case);
                let got = run_call(&call, &xs[0]);
                api::compare(case, &got, fixture_ulps(&call.name))
            });
            if let Err(e) = result {
                failures.push(format!("{device:?}: {e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} reduce fixture failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn var_std_gradients() {
    common::on_cpu();
    let opts = |ddof, keepdims| VarOptions { ddof, keepdims };
    api::check_grad("var all", &|xs| xs[0].var(..), &[&[3, 4]], -1.0, 1.0);
    api::check_grad("var -1", &|xs| xs[0].var(-1), &[&[3, 4]], -1.0, 1.0);
    api::check_grad(
        "var ddof1",
        &|xs| xs[0].var_with(-1, opts(1, false)),
        &[&[3, 4]],
        -1.0,
        1.0,
    );
    api::check_grad(
        "var ddof1 all keepdims",
        &|xs| xs[0].var_with(.., opts(1, true)),
        &[&[3, 4]],
        -1.0,
        1.0,
    );
    api::check_grad("std all", &|xs| xs[0].std(..), &[&[3, 4]], -1.0, 1.0);
    api::check_grad("std 0", &|xs| xs[0].std_keepdims(0), &[&[3, 4]], -1.0, 1.0);
    api::check_grad(
        "std ddof1",
        &|xs| xs[0].std_with(0, opts(1, false)),
        &[&[3, 4]],
        -1.0,
        1.0,
    );
    api::check_grad(
        "std ddof1 [0, 1]",
        &|xs| xs[0].std_with([0, 1], opts(1, true)),
        &[&[3, 4]],
        -1.0,
        1.0,
    );
}
