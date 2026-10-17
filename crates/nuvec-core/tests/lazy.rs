//! Behaviour of lazy arrays through the public API.

mod common;

use nuvec_core::{
    Array, Complex64, DType, Device, Element, Error, Stream, add, bf16, eval, f16, multiply,
    set_default_stream,
};

const N: usize = if cfg!(miri) { 16 } else { 1024 };

/// Under Miri, deep-graph tests use a smaller depth.
const DEEP: usize = if cfg!(miri) { 200 } else { 1_000_000 };

// ── User story 1: lazy computation, every dtype ─────────────────────────────

/// Checks `(a + b) * a` against `expected`, computed element by element in the test.
fn check<T: Element + PartialEq + std::fmt::Debug>(a: Vec<T>, b: Vec<T>, f: impl Fn(T, T) -> T) {
    let expected: Vec<T> = a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect();
    let (xa, xb) = (
        Array::from_slice(&a, &[a.len()]).unwrap(),
        Array::from_slice(&b, &[b.len()]).unwrap(),
    );
    let c = (&xa + &xb) * &xa;
    assert_eq!(c.dtype(), T::DTYPE);
    assert!(!c.is_evaluated());
    assert_eq!(c.to_vec::<T>().unwrap(), expected, "{}", T::DTYPE);
}

macro_rules! check_int {
    ($($t:ty),*) => {$(
        check::<$t>(
            (0..N).map(|i| i as $t).collect(),
            (0..N).map(|i| (i * 7 + 3) as $t).collect(),
            |x, y| x.wrapping_add(y).wrapping_mul(x),
        );
    )*};
}

#[test]
fn all_dtypes_add_mul() {
    common::on_cpu();
    check_int!(u8, u16, u32, u64, i8, i16, i32, i64);
    check::<bool>(
        (0..N).map(|i| i % 2 == 0).collect(),
        (0..N).map(|i| i % 3 == 0).collect(),
        |x, y| (x | y) & x,
    );
    check::<f32>(
        (0..N).map(|i| i as f32 * 0.5).collect(),
        (0..N).map(|i| 1.0 - i as f32).collect(),
        |x, y| (x + y) * x,
    );
    check::<f64>(
        (0..N).map(|i| i as f64 * 0.25).collect(),
        (0..N).map(|i| i as f64 - 3.0).collect(),
        |x, y| (x + y) * x,
    );
    // Miri cannot execute `half`'s hardware f16 conversion (an aarch64 FP16 instruction).
    if !cfg!(miri) {
        check::<f16>(
            (0..N).map(|i| f16::from_f32(i as f32 / 64.0)).collect(),
            (0..N)
                .map(|i| f16::from_f32(0.5 - i as f32 / 128.0))
                .collect(),
            |x, y| (x + y) * x,
        );
    }
    check::<bf16>(
        (0..N).map(|i| bf16::from_f32(i as f32 / 8.0)).collect(),
        (0..N).map(|i| bf16::from_f32(2.0 - i as f32)).collect(),
        |x, y| (x + y) * x,
    );
    check::<Complex64>(
        (0..N).map(|i| Complex64::new(i as f32, 1.0)).collect(),
        (0..N)
            .map(|i| Complex64::new(-1.0, i as f32 * 0.5))
            .collect(),
        |x, y| (x + y) * x,
    );
}

#[test]
fn lazy_until_read() {
    common::on_cpu();
    let a = Array::from_slice(&[1i32, 2, 3], &[3]).unwrap();
    let b = &a * &a;
    let c = &b + &a;
    assert!(!b.is_evaluated() && !c.is_evaluated());
    c.eval().unwrap();
    assert!(c.is_evaluated() && b.is_evaluated());
    assert_eq!(c.to_vec::<i32>().unwrap(), vec![2, 6, 12]);
    assert_eq!(b.to_vec::<i32>().unwrap(), vec![1, 4, 9]);
}

#[test]
fn explicit_eval_of_many() {
    common::on_cpu();
    let a = Array::full(&[2, 2], 3u32).unwrap();
    let outs = [&a + &a, &a * &a];
    eval(&outs).unwrap();
    assert!(outs.iter().all(Array::is_evaluated));
    assert_eq!(outs[1].to_vec::<u32>().unwrap(), vec![9; 4]);
}

// ── User story 2: errors instead of crashes ─────────────────────────────────

#[test]
fn misuse_shape_mismatch() {
    common::on_cpu();
    let a = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let b = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    let err = add(&a, &b).unwrap_err();
    assert_eq!(
        err,
        Error::ShapeMismatch {
            op: "add",
            lhs: vec![2],
            rhs: vec![3]
        }
    );
    assert!(err.to_string().contains("[2]") && err.to_string().contains("[3]"));
}

#[test]
fn misuse_dtype_mismatch() {
    common::on_cpu();
    // Since F-003, mixed dtypes promote (float32 * int32 -> float32) instead of failing.
    let c = multiply(Array::scalar(1.5f32), Array::scalar(2i32)).unwrap();
    assert_eq!(c.dtype(), DType::F32);
    assert_eq!(c.item::<f32>().unwrap(), 3.0);
    // Readback with the wrong type is still a dtype mismatch, naming both types.
    let err = c.item::<i32>().unwrap_err();
    assert_eq!(
        err,
        Error::DTypeMismatch {
            op: "item",
            lhs: DType::F32,
            rhs: DType::I32
        }
    );
    assert!(err.to_string().contains("float32") && err.to_string().contains("int32"));
}

#[test]
fn misuse_size_mismatch() {
    common::on_cpu();
    let err = Array::from_slice(&[1u8, 2, 3, 4, 5], &[2, 3]).unwrap_err();
    assert_eq!(
        err,
        Error::SizeMismatch {
            expected: 6,
            actual: 5
        }
    );
}

#[test]
fn misuse_readback_type() {
    common::on_cpu();
    let a = Array::scalar(1.0f32);
    assert!(matches!(
        a.to_vec::<i64>(),
        Err(Error::DTypeMismatch { op: "to_vec", .. })
    ));
}

#[test]
fn misuse_unsupported_device() {
    common::on_cpu();
    // GPU 0 is a real device on a Mac; no machine has an eighth GPU.
    let a = Array::scalar(1.0f32);
    set_default_stream(Stream::default_for(Device::gpu(7)));
    let c = &a + &a;
    set_default_stream(Stream::default_for(Device::cpu()));
    assert_eq!(c.device(), Device::gpu(7));
    assert_eq!(c.eval(), Err(Error::UnsupportedDevice(Device::gpu(7))));
    assert_eq!(
        c.eval(),
        Err(Error::UnsupportedDevice(Device::gpu(7))),
        "sticky"
    );
}

#[test]
fn misuse_operator_is_deferred() {
    common::on_cpu();
    let a = Array::from_slice(&[1i16, 2], &[2]).unwrap();
    let b = Array::from_slice(&[1i16, 2, 3], &[3]).unwrap();
    let bad = &a + &b; // must not panic
    let worse = &bad * &a; // chaining keeps the original error
    let expected = Error::ShapeMismatch {
        op: "add",
        lhs: vec![2],
        rhs: vec![3],
    };
    assert_eq!(bad.to_vec::<i16>(), Err(expected.clone()));
    assert_eq!(worse.eval(), Err(expected.clone()));
    assert_eq!(eval(&[a.clone(), worse]), Err(expected));
}

// ── User story 3: deep graphs, threads ──────────────────────────────────────

#[test]
fn deep_chain_eval() {
    common::on_cpu();
    let one = Array::scalar(1u64);
    let mut x = Array::scalar(0u64);
    for _ in 0..DEEP {
        x = &x + &one;
    }
    assert_eq!(x.item::<u64>().unwrap(), DEEP as u64);
}

#[test]
fn deep_chain_drop_unevaluated() {
    common::on_cpu();
    let one = Array::scalar(1u64);
    let mut x = Array::scalar(0u64);
    for _ in 0..DEEP {
        x = &x + &one;
    }
    drop(x); // must not overflow the stack
}

#[test]
fn concurrent_reads() {
    common::on_cpu();
    let a = Array::from_slice(&(0..N as i32).collect::<Vec<_>>(), &[N]).unwrap();
    let c = (&a * &a) + &a;
    let results: Vec<Vec<i32>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = c.clone();
                s.spawn(move || c.to_vec::<i32>().unwrap())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let expected: Vec<i32> = (0..N as i32).map(|i| i * i + i).collect();
    assert!(results.iter().all(|r| *r == expected));
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn edge_empty() {
    common::on_cpu();
    let a = Array::from_slice::<f32>(&[], &[2, 0]).unwrap();
    let c = &a + &a;
    assert_eq!(c.shape(), &[2, 0]);
    assert_eq!(c.size(), 0);
    assert!(c.to_vec::<f32>().unwrap().is_empty());
}

#[test]
fn edge_scalar() {
    common::on_cpu();
    let c = Array::scalar(3i64) * Array::scalar(-4i64);
    assert_eq!(c.ndim(), 0);
    assert_eq!(c.item::<i64>().unwrap(), -12);
}

#[test]
fn edge_size_overflow() {
    common::on_cpu();
    assert!(matches!(
        Array::full(&[usize::MAX, 2], 0u8),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        Array::full(&[usize::MAX / 4], 0u64),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn edge_int_wrap() {
    common::on_cpu();
    assert_eq!(
        (Array::scalar(127i8) + Array::scalar(1i8))
            .item::<i8>()
            .unwrap(),
        -128
    );
    assert_eq!(
        (Array::scalar(200u8) * Array::scalar(2u8))
            .item::<u8>()
            .unwrap(),
        144
    );
}

#[test]
fn edge_bool() {
    common::on_cpu();
    let a = Array::from_slice(&[true, true, false, false], &[4]).unwrap();
    let b = Array::from_slice(&[true, false, true, false], &[4]).unwrap();
    assert_eq!(
        (&a + &b).to_vec::<bool>().unwrap(),
        vec![true, true, true, false]
    );
    assert_eq!(
        (&a * &b).to_vec::<bool>().unwrap(),
        vec![true, false, false, false]
    );
}

#[test]
fn edge_complex() {
    common::on_cpu();
    let i = Array::scalar(Complex64::new(0.0, 1.0));
    assert_eq!(
        (&i * &i).item::<Complex64>().unwrap(),
        Complex64::new(-1.0, 0.0)
    );
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
fn edge_half() {
    common::on_cpu();
    // 2049 is not representable in f16: the sum rounds to the nearest even, 2048.
    let a = Array::scalar(f16::from_f32(2048.0));
    let b = Array::scalar(f16::from_f32(1.0));
    assert_eq!((a + b).item::<f16>().unwrap(), f16::from_f32(2048.0));
    let x = Array::scalar(bf16::from_f32(1.5));
    assert_eq!((&x * &x).item::<bf16>().unwrap(), bf16::from_f32(2.25));
}

/// Timing bounds are only meaningful in optimised builds (as for the other `perf_*` checks);
/// run with `cargo test --release`.
#[test]
#[cfg(not(debug_assertions))]
fn many_small_ops_sanity() {
    common::on_cpu();
    let start = std::time::Instant::now();
    let one = Array::scalar(1.0f32);
    let mut x = Array::scalar(0.0f32);
    for _ in 0..100_000 {
        x = &x + &one;
    }
    assert_eq!(x.item::<f32>().unwrap(), 100_000.0);
    assert!(
        start.elapsed().as_secs_f64() < 1.0,
        "took {:?}",
        start.elapsed()
    );
}
