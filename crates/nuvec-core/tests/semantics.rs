//! NumPy/MLX array semantics, checked against an independent reference.

mod common;

use common::{Ref, Rng, assert_matches, broadcast_shape, random_layout};
use nuvec_core::*;

/// Every dtype; without f16 under Miri, which cannot execute `half`'s hardware conversion.
fn all_dtypes() -> Vec<DType> {
    use DType::*;
    let all = [
        Bool, U8, U16, U32, U64, I8, I16, I32, I64, F16, BF16, F32, F64, Complex64,
    ];
    all.into_iter()
        .filter(|&d| !(cfg!(miri) && d == F16))
        .collect()
}

/// Relative tolerance for transcendental functions: 32 ulps. Results are bit-exact on real
/// hardware; Miri deliberately perturbs libm results by a few ulps.
fn close(got: f64, expected: f64, epsilon: f64) -> bool {
    got.to_bits() == expected.to_bits()
        || (got.is_nan() && expected.is_nan())
        || (got - expected).abs() <= 32.0 * epsilon * expected.abs().max(1.0)
}

/// Random cases per check (fewer under Miri, which is ~1000x slower).
const CASES: usize = if cfg!(miri) { 1 } else { 200 };

// ── User story 1: broadcasting and promotion ────────────────────────────────

#[test]
#[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
fn broadcast_column_plus_row() {
    let col = Array::from_slice(&[1i32, 2, 3], &[3, 1]).unwrap();
    let row = Array::from_slice(&[0.5f32, 1.5, 2.5, 3.5].map(f16::from_f32), &[4]).unwrap();
    let out = &col + &row;
    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(out.dtype(), DType::F16);
    assert_eq!(out.to_vec::<f16>().unwrap()[4], f16::from_f32(2.5));
}

#[test]
fn broadcast_scalar_and_zero_size() {
    let s = Array::scalar(2u8);
    let e = zeros(&[3, 0], DType::U8).unwrap();
    assert_eq!((&e * &s).shape(), &[3, 0]);
    assert_eq!(
        multiply(&s, ones(&[2, 2], DType::U8).unwrap())
            .unwrap()
            .to_vec::<u8>()
            .unwrap(),
        vec![2; 4]
    );
    assert!(
        add(
            zeros(&[2], DType::U8).unwrap(),
            zeros(&[0], DType::U8).unwrap()
        )
        .is_err()
    );
}

#[test]
fn broadcast_mismatch_names_shapes() {
    let err = add(
        zeros(&[2, 3], DType::F32).unwrap(),
        zeros(&[4], DType::F32).unwrap(),
    )
    .unwrap_err();
    assert_eq!(
        err,
        Error::ShapeMismatch {
            op: "add",
            lhs: vec![2, 3],
            rhs: vec![4]
        }
    );
}

#[test]
fn promotion_all_pairs() {
    use DType::*;
    // Miri samples representative types; the full 14x14 table runs natively.
    let all = if cfg!(miri) {
        vec![Bool, U64, I8, BF16, Complex64]
    } else {
        all_dtypes()
    };
    for &a in &all {
        for &b in &all {
            let out = add(ones(&[2], a).unwrap(), ones(&[2], b).unwrap()).unwrap();
            assert_eq!(out.dtype(), promote_types(a, b), "{a} + {b}");
            let expected = if out.dtype() == Bool { 1.0 } else { 2.0 };
            assert_eq!(
                out.astype(F64).unwrap().to_vec::<f64>().unwrap(),
                vec![expected; 2],
                "{a} + {b}"
            );
        }
    }
}

#[test]
fn divide_promotes_integers_to_float() {
    let q = divide(
        Array::from_slice(&[7i32, -7], &[2]).unwrap(),
        Array::scalar(2i32),
    )
    .unwrap();
    assert_eq!(q.dtype(), DType::F32);
    assert_eq!(q.to_vec::<f32>().unwrap(), vec![3.5, -3.5]);
    let half = divide(Array::scalar(1i64), Array::scalar(bf16::from_f32(4.0))).unwrap();
    assert_eq!(half.dtype(), DType::BF16);
}

#[test]
fn compare_returns_bool() {
    let a = Array::from_slice(&[1i32, 5, 3], &[3, 1]).unwrap();
    let b = Array::from_slice(&[2.0f32, 3.0], &[2]).unwrap();
    let lt = less(&a, &b).unwrap();
    assert_eq!((lt.dtype(), lt.shape()), (DType::Bool, &[3usize, 2][..]));
    assert_eq!(
        lt.to_vec::<bool>().unwrap(),
        vec![true, true, false, false, false, false]
    );
    assert_eq!(
        greater_equal(&a, &b).unwrap().to_vec::<bool>().unwrap(),
        vec![false, false, true, true, true, true]
    );
}

macro_rules! binary_random_int {
    ($rng:expr; $($t:ty),*) => {$(
        for _ in 0..CASES {
            let shape = $rng.shape();
            let other = $rng.broadcastable(&shape);
            let (sx, sy) = if $rng.coin() { (shape, other) } else { (other, shape) };
            let xv: Vec<$t> = (0..common::size(&sx)).map(|_| $rng.next() as $t).collect();
            let yv: Vec<$t> = (0..common::size(&sy)).map(|_| $rng.next() as $t).collect();
            let (x, rx) = random_layout(&mut $rng, &sx, xv);
            let (y, ry) = random_layout(&mut $rng, &sy, yv);
            let ctx = stringify!($t);
            assert_matches(&(&x + &y), &rx.zip(&ry, <$t>::wrapping_add), ctx);
            assert_matches(&(&x - &y), &rx.zip(&ry, <$t>::wrapping_sub), ctx);
            assert_matches(&(&x * &y), &rx.zip(&ry, <$t>::wrapping_mul), ctx);
            assert_matches(&maximum(&x, &y).unwrap(), &rx.zip(&ry, |a, b| a.max(b)), ctx);
            assert_matches(&minimum(&x, &y).unwrap(), &rx.zip(&ry, |a, b| a.min(b)), ctx);
            assert_matches(&equal(&x, &y).unwrap(), &rx.zip(&ry, |a, b| a == b), ctx);
            assert_matches(&less_equal(&x, &y).unwrap(), &rx.zip(&ry, |a, b| a <= b), ctx);
        }
    )*};
}

macro_rules! binary_random_float {
    ($rng:expr; $($t:ty => $from:expr),*) => {$(
        for _ in 0..CASES {
            let shape = $rng.shape();
            let other = $rng.broadcastable(&shape);
            let (sx, sy) = if $rng.coin() { (shape, other) } else { (other, shape) };
            let conv: fn(f64) -> $t = $from;
            let xv: Vec<$t> = (0..common::size(&sx)).map(|_| conv($rng.f64())).collect();
            let yv: Vec<$t> = (0..common::size(&sy)).map(|_| conv($rng.f64())).collect();
            let (x, rx) = random_layout(&mut $rng, &sx, xv);
            let (y, ry) = random_layout(&mut $rng, &sy, yv);
            let ctx = stringify!($t);
            assert_matches(&(&x + &y), &rx.zip(&ry, |a, b| a + b), ctx);
            assert_matches(&(&x - &y), &rx.zip(&ry, |a, b| a - b), ctx);
            assert_matches(&(&x * &y), &rx.zip(&ry, |a, b| a * b), ctx);
            assert_matches(&(&x / &y), &rx.zip(&ry, |a, b| a / b), ctx);
            assert_matches(&greater(&x, &y).unwrap(), &rx.zip(&ry, |a, b| a > b), ctx);
            assert_matches(&not_equal(&x, &y).unwrap(), &rx.zip(&ry, |a, b| a != b), ctx);
        }
    )*};
}

#[test]
fn binary_random() {
    let mut rng = Rng::new(0xB1);
    binary_random_int!(rng; u8, u16, u32, u64, i8, i16, i32, i64);
    binary_random_float!(rng; f32 => |v| v as f32, f64 => |v| v);
    // Half precision is not run under Miri: `half`'s hardware conversion is inline asm.
    if !cfg!(miri) {
        binary_random_float!(rng; f16 => f16::from_f64, bf16 => bf16::from_f64);
    }
}

#[test]
fn bool_and_complex_binary() {
    let t = Array::from_slice(&[true, true, false, false], &[4]).unwrap();
    let u = Array::from_slice(&[true, false, true, false], &[4]).unwrap();
    assert_eq!(
        (&t - &u).to_vec::<bool>().unwrap(),
        vec![false, true, true, false],
        "bool subtract is XOR"
    );
    assert_eq!(
        maximum(&t, &u).unwrap().to_vec::<bool>().unwrap(),
        vec![true, true, true, false]
    );
    let z =
        Array::from_slice(&[Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)], &[2]).unwrap();
    let w = Array::scalar(Complex64::new(0.0, 1.0));
    assert_eq!(
        (&z / &w).to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(2.0, -1.0), Complex64::new(-1.0, -3.0)]
    );
    assert_eq!(
        equal(&z, &z).unwrap().to_vec::<bool>().unwrap(),
        vec![true, true]
    );
}

#[test]
fn nan_propagation() {
    let a = Array::from_slice(&[f32::NAN, 1.0, 2.0], &[3]).unwrap();
    let b = Array::from_slice(&[0.0f32, f32::NAN, 1.0], &[3]).unwrap();
    let max = maximum(&a, &b).unwrap().to_vec::<f32>().unwrap();
    assert!(max[0].is_nan() && max[1].is_nan() && max[2] == 2.0);
    let min = minimum(&a, &b).unwrap().to_vec::<f32>().unwrap();
    assert!(min[0].is_nan() && min[1].is_nan() && min[2] == 1.0);
    assert_eq!(
        equal(&a, &a).unwrap().to_vec::<bool>().unwrap(),
        vec![false, true, true]
    );
}

// ── User story 2: views ─────────────────────────────────────────────────────

fn random_slice_args(rng: &mut Rng, shape: &[usize]) -> Vec<SliceArg> {
    shape
        .iter()
        .map(|&n| {
            let n = n as isize;
            let bound = |rng: &mut Rng| {
                if rng.below(4) == 0 {
                    None
                } else {
                    Some(rng.range(-n - 2, n + 2))
                }
            };
            let step = [-3, -2, -1, 1, 1, 2, 3][rng.below(7)];
            SliceArg::new(bound(rng), bound(rng), step)
        })
        .collect()
}

#[test]
fn views_random() {
    let mut rng = Rng::new(0x51);
    for case in 0..CASES {
        let shape = rng.shape();
        let values: Vec<f32> = (0..common::size(&shape)).map(|i| i as f32).collect();
        let (mut a, mut r) = random_layout(&mut rng, &shape, values);
        for _ in 0..1 + rng.below(4) {
            match rng.below(5) {
                0 => {
                    let mut axes: Vec<usize> = (0..r.shape.len()).collect();
                    for i in (1..axes.len()).rev() {
                        axes.swap(i, rng.below(i + 1));
                    }
                    a = transpose(&a, &axes).unwrap();
                    r = r.transpose(&axes);
                }
                1 => {
                    let args = random_slice_args(&mut rng, &r.shape);
                    a = slice(&a, &args).unwrap();
                    r = r.slice(&args);
                }
                2 => {
                    let mut target = vec![1 + rng.below(3)];
                    target.extend(
                        r.shape
                            .iter()
                            .map(|&d| if d == 1 { 1 + rng.below(3) } else { d }),
                    );
                    a = broadcast_to(&a, &target).unwrap();
                    r = r.broadcast(&target);
                }
                3 => {
                    a = a.reshape(&[-1]).unwrap();
                    r = r.reshape(vec![r.data.len()]);
                }
                _ => {
                    let axis = rng.below(r.shape.len() + 1);
                    a = expand_dims(&a, axis as isize).unwrap();
                    let mut s = r.shape.clone();
                    s.insert(axis, 1);
                    r = r.reshape(s);
                }
            }
        }
        assert_matches(&a, &r, &format!("case {case}"));
        // Views must also work as operands.
        assert_matches(
            &(&a + &a),
            &r.map(|x| x + x),
            &format!("case {case} as operand"),
        );
    }
}

#[test]
fn slice_python_semantics() {
    let a = arange(0.0, 10.0, 1.0, DType::I32).unwrap();
    let get = |s: SliceArg| slice(&a, &[s]).unwrap().to_vec::<i32>().unwrap();
    assert_eq!(
        get(SliceArg::new(None, None, -1)),
        (0..10).rev().collect::<Vec<_>>()
    );
    assert_eq!(get(SliceArg::new(Some(-3), None, 1)), vec![7, 8, 9]);
    assert_eq!(get(SliceArg::new(Some(8), Some(2), -3)), vec![8, 5]);
    assert_eq!(get(SliceArg::new(Some(-100), Some(100), 4)), vec![0, 4, 8]);
    assert_eq!(get(SliceArg::new(Some(5), Some(5), 1)), Vec::<i32>::new());
    assert_eq!(get(SliceArg::new(Some(2), None, 50)), vec![2]);
    assert_eq!(get(SliceArg::from(-1)), vec![9]);
    assert_eq!(get((2..4).into()), vec![2, 3]);
    assert!(slice(&a, &[SliceArg::new(None, None, 0)]).is_err());
    assert!(slice(&a, &[SliceArg::full(), SliceArg::full()]).is_err());
}

#[test]
fn reshape_infer_and_errors() {
    let a = arange(0.0, 12.0, 1.0, DType::F32).unwrap();
    assert_eq!(a.reshape(&[3, -1]).unwrap().shape(), &[3, 4]);
    assert!(a.reshape(&[5, -1]).is_err());
    assert!(a.reshape(&[-1, -1]).is_err());
    assert!(a.reshape(&[7]).is_err());
    assert!(transpose(&a.reshape(&[3, 4]).unwrap(), &[0, 0]).is_err());
    assert!(squeeze(&a.reshape(&[3, 4]).unwrap(), &[0]).is_err());
    assert!(expand_dims(&a, 5).is_err());
    assert_eq!(
        squeeze(&a.reshape(&[1, 12, 1]).unwrap(), &[-1])
            .unwrap()
            .shape(),
        &[1, 12]
    );
    assert!(broadcast_to(&a, &[2, 6]).is_err());
}

// ── User story 3: unary, where, astype ──────────────────────────────────────

fn sigmoid_ref(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        x.exp() / (1.0 + x.exp())
    }
}

/// A unary operation under test: the operation, its f64 reference, and its name.
type UnaryCase = (fn(&Array) -> Result<Array>, fn(f64) -> f64, &'static str);

#[test]
fn unary_random() {
    let mut rng = Rng::new(0x07);
    let ops: [UnaryCase; 7] = [
        (exp, f64::exp, "exp"),
        (log, f64::ln, "log"),
        (sqrt, f64::sqrt, "sqrt"),
        (sin, f64::sin, "sin"),
        (cos, f64::cos, "cos"),
        (tanh, f64::tanh, "tanh"),
        (sigmoid, sigmoid_ref, "sigmoid"),
    ];
    for _ in 0..CASES / 4 {
        let shape = rng.shape();
        let v64: Vec<f64> = (0..common::size(&shape)).map(|_| rng.f64()).collect();
        let (x64, r64) = random_layout(&mut rng, &shape, v64.clone());
        let (x32, r32) = random_layout(&mut rng, &shape, v64.iter().map(|&v| v as f32).collect());
        let (xi, ri) = random_layout(&mut rng, &shape, v64.iter().map(|&v| v as i32).collect());
        for (op, f, name) in ops {
            let got = op(&x64).unwrap().to_vec::<f64>().unwrap();
            for (g, &x) in got.iter().zip(&r64.data) {
                assert!(
                    close(*g, f(x), f64::EPSILON),
                    "{name}({x}): {g} vs {}",
                    f(x)
                );
            }
            // f32 computes in f32 with the same std functions; compare via f64 tolerance.
            let got = op(&x32).unwrap().to_vec::<f32>().unwrap();
            for (g, &x) in got.iter().zip(&r32.data) {
                let e = f(x as f64) as f32;
                assert!(
                    close(f64::from(*g), f64::from(e), f64::from(f32::EPSILON)),
                    "{name}({x}): {g} vs {e}"
                );
            }
            let promoted = op(&xi).unwrap();
            assert_eq!(promoted.dtype(), DType::F32, "{name} promotes int32");
            assert_eq!(promoted.shape(), ri.shape.as_slice());
        }
        assert_matches(
            &negative(&xi).unwrap(),
            &ri.map(i32::wrapping_neg),
            "negative",
        );
        assert_matches(&abs(&x64).unwrap(), &r64.map(f64::abs), "abs");
        assert_matches(&(-&x32), &r32.map(|v| -v), "neg operator");
    }
}

#[test]
fn unary_dtype_rules() {
    let z = Array::from_slice(&[Complex64::new(3.0, 4.0)], &[1]).unwrap();
    let m = abs(&z).unwrap();
    assert_eq!(m.dtype(), DType::F32);
    assert!(close(
        f64::from(m.item::<f32>().unwrap()),
        5.0,
        f64::from(f32::EPSILON)
    ));
    assert_eq!(
        negative(&z).unwrap().to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(-3.0, -4.0)]
    );
    let b = Array::scalar(true);
    assert_eq!(
        negative(&b).unwrap_err(),
        Error::UnsupportedDType {
            op: "negative",
            dtype: DType::Bool
        }
    );
    assert!(matches!(
        exp(&z),
        Err(Error::UnsupportedDType { op: "exp", .. })
    ));
    assert!(matches!(
        less(&z, &z),
        Err(Error::UnsupportedDType { op: "less", .. })
    ));
    assert!(matches!(
        maximum(&z, &z),
        Err(Error::UnsupportedDType { .. })
    ));
    assert_eq!(
        abs(&Array::scalar(-128i8)).unwrap().item::<i8>().unwrap(),
        -128,
        "abs wraps"
    );
    if !cfg!(miri) {
        let h = exp(&Array::scalar(f16::from_f32(1.0))).unwrap();
        assert_eq!(h.item::<f16>().unwrap(), f16::from_f32(1.0f32.exp()));
    }
}

#[test]
fn where_random() {
    let mut rng = Rng::new(0x3E);
    for _ in 0..CASES {
        let shape = rng.shape();
        let sc = rng.broadcastable(&shape);
        let sb = rng.broadcastable(&shape);
        let cv: Vec<u8> = (0..common::size(&sc))
            .map(|_| (rng.next() % 3) as u8)
            .collect();
        let av: Vec<f32> = (0..common::size(&shape))
            .map(|_| rng.f64() as f32)
            .collect();
        let bv: Vec<i32> = (0..common::size(&sb)).map(|_| rng.next() as i32).collect();
        let (c, rc) = random_layout(&mut rng, &sc, cv);
        let (a, ra) = random_layout(&mut rng, &shape, av);
        let (b, rb) = random_layout(&mut rng, &sb, bv);
        let out = where_(&c, &a, &b).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        let full = broadcast_shape(&broadcast_shape(&sc, &shape).unwrap(), &sb).unwrap();
        let (rc, ra, rb) = (
            rc.broadcast(&full),
            ra.broadcast(&full),
            rb.broadcast(&full),
        );
        let expected: Vec<f32> = (0..rc.data.len())
            .map(|i| {
                if rc.data[i] != 0 {
                    ra.data[i]
                } else {
                    rb.data[i] as f32
                }
            })
            .collect();
        assert_matches(&out, &Ref::new(full, expected), "where");
    }
}

macro_rules! cast_pairs {
    ($rng:expr; $($s:ty),*) => {$( cast_pairs!(@to $rng, $s; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64); )*};
    (@to $rng:expr, $s:ty; $($d:ty),*) => {$(
        {
            let shape = $rng.shape();
            let values: Vec<$s> = (0..common::size(&shape))
                .map(|i| match i % 4 {
                    0 => ($rng.f64() * 30.0) as $s,
                    1 => $rng.next() as $s,
                    2 => (-1.0f64) as $s,
                    _ => <$s>::MAX,
                })
                .collect();
            let (x, r) = random_layout(&mut $rng, &shape, values);
            let dtype = <$d as Element>::DTYPE;
            assert_matches(&x.astype(dtype).unwrap(), &r.map(|v| v as $d), concat!(stringify!($s), " -> ", stringify!($d)));
        }
    )*};
}

#[test]
fn astype_all_pairs() {
    let mut rng = Rng::new(0xCA);
    if cfg!(miri) {
        cast_pairs!(rng; i64, f32); // a sample; all 100 pairs run natively
    } else {
        cast_pairs!(rng; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
    }
    let nan = Array::from_slice(&[f32::NAN, -0.0, 2.5], &[3]).unwrap();
    assert_eq!(
        nan.astype(DType::I32).unwrap().to_vec::<i32>().unwrap(),
        vec![0, 0, 2]
    );
    assert_eq!(
        nan.astype(DType::Bool).unwrap().to_vec::<bool>().unwrap(),
        vec![true, false, true]
    );
    let z = Array::scalar(Complex64::new(1.5, -2.0));
    assert_eq!(z.astype(DType::F64).unwrap().item::<f64>().unwrap(), 1.5);
    assert_eq!(
        Array::scalar(7u8)
            .astype(DType::Complex64)
            .unwrap()
            .item::<Complex64>()
            .unwrap(),
        Complex64::new(7.0, 0.0)
    );
}

// ── User story 4: creation ──────────────────────────────────────────────────

#[test]
fn arange_semantics() {
    assert_eq!(
        arange(0.0, 10.0, 3.0, DType::I32)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        vec![0, 3, 6, 9]
    );
    assert_eq!(
        arange(1.0, 0.0, -0.25, DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        vec![1.0, 0.75, 0.5, 0.25]
    );
    assert_eq!(arange(5.0, 1.0, 1.0, DType::F32).unwrap().size(), 0);
    assert_eq!(
        arange(0.0, 1.0, f64::INFINITY, DType::F64)
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        vec![0.0]
    );
    assert!(matches!(
        arange(0.0, 1.0, 0.0, DType::F32),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        arange(0.0, f64::INFINITY, 1.0, DType::F32),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        arange(f64::NAN, 1.0, 1.0, DType::F32),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        arange(0.0, 1e12, 1.0, DType::F32),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        arange(0.0, 2.0, 1.0, DType::Bool),
        Err(Error::UnsupportedDType { .. })
    ));
}

#[test]
fn zeros_ones_all_dtypes() {
    use DType::*;
    for d in all_dtypes() {
        let z = zeros(&[2, 3], d).unwrap();
        let o = ones(&[2, 3], d).unwrap();
        assert_eq!((z.dtype(), o.shape()), (d, &[2usize, 3][..]));
        assert_eq!(
            z.astype(F64).unwrap().to_vec::<f64>().unwrap(),
            vec![0.0; 6],
            "{d}"
        );
        assert_eq!(
            o.astype(F64).unwrap().to_vec::<f64>().unwrap(),
            vec![1.0; 6],
            "{d}"
        );
    }
}

// ── Errors and sanity performance ───────────────────────────────────────────

#[test]
fn errors_are_typed_and_deferred() {
    let a = zeros(&[2], DType::F32).unwrap();
    let b = zeros(&[3], DType::F32).unwrap();
    let bad = &a / &b;
    let chained = -(&bad - &a);
    assert!(matches!(
        chained.eval(),
        Err(Error::ShapeMismatch { op: "divide", .. })
    ));
    assert!(matches!(
        where_(Array::scalar(true), &a, &b),
        Err(Error::ShapeMismatch { op: "where", .. })
    ));
    assert!(matches!(reshape(&a, &[-2]), Err(Error::InvalidArgument(_))));
}

#[test]
#[cfg(not(debug_assertions))]
fn perf_add_1m() {
    let a = ones(&[1_000_000], DType::F32).unwrap();
    let b = ones(&[1_000_000], DType::F32).unwrap();
    let start = std::time::Instant::now();
    (&a + &b).eval().unwrap();
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 0.005,
        "1M-element add took {elapsed:?}"
    );
}
