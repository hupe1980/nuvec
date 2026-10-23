//! Weak scalar operands: MLX's Python typing rules, operators on both sides, functions.

use nuvec_core::ops;
use nuvec_core::*;

fn arr<T: Element>(data: &[T]) -> Array {
    Array::from(data)
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
fn float_scalar_keeps_float_dtype() {
    let x = arr(&[f16::from_f32(1.5), f16::from_f32(-2.0)]);
    let y = &x * 2.0;
    assert_eq!(y.dtype(), DType::F16);
    assert_eq!(
        y.to_vec::<f16>().unwrap(),
        [f16::from_f32(3.0), f16::from_f32(-4.0)]
    );
    assert_eq!((&arr(&[1.0f32]) + 0.5f64).dtype(), DType::F32);
    assert_eq!((&arr(&[bf16::ONE]) - 1.0f32).dtype(), DType::BF16);
}

#[test]
fn float_scalar_with_integer_or_bool_array_is_float32() {
    let i = arr(&[1i32, 2]);
    let y = &i * 0.5;
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(y.to_vec::<f32>().unwrap(), [0.5, 1.0]);
    assert_eq!((&arr(&[true]) + 1.0).dtype(), DType::F32);
    // Promotion still applies after conversion: int64 with float32 is float32 in MLX.
    assert_eq!((&arr(&[1i64]) * 2.0).dtype(), DType::F32);
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
fn integer_scalar_keeps_integer_dtype() {
    let c = arr(&[1i8, -3]);
    let y = &c + 1;
    assert_eq!(y.dtype(), DType::I8);
    assert_eq!(y.to_vec::<i8>().unwrap(), [2, -2]);
    assert_eq!((&arr(&[1u8]) * 3u64).dtype(), DType::U8);
    assert_eq!((&arr(&[f16::ONE]) + 1).dtype(), DType::F16);
    assert_eq!((&arr(&[1.0f64]) + 1).dtype(), DType::F64);
}

#[test]
fn integer_scalar_out_of_range_is_an_error() {
    let c = arr(&[1i8]);
    assert!(matches!(
        ops::add(&c, 300).check(),
        Err(Error::Op {
            kind: OpErrorKind::InvalidArgument(_),
            ..
        })
    ));
    assert!(matches!(
        ops::subtract(arr(&[1u32]), -1).check(),
        Err(Error::Op {
            kind: OpErrorKind::InvalidArgument(_),
            ..
        })
    ));
    // Operators defer the error to evaluation.
    let y = &c + 300;
    assert!(matches!(
        y.eval(),
        Err(Error::Op {
            op: "add",
            kind: OpErrorKind::InvalidArgument(_),
            ..
        })
    ));
    // Boundaries are exact.
    assert_eq!(
        ops::add(arr(&[0u64]), u64::MAX).item::<u64>().unwrap(),
        u64::MAX
    );
    assert_eq!(
        ops::add(arr(&[0i64]), i64::MIN).item::<i64>().unwrap(),
        i64::MIN
    );
}

#[test]
fn integer_scalar_with_bool_array_is_int32() {
    let m = arr(&[true, false]);
    let y = &m + 1;
    assert_eq!(y.dtype(), DType::I32);
    assert_eq!(y.to_vec::<i32>().unwrap(), [2, 1]);
    assert_eq!((&m + (1i64 << 40)).dtype(), DType::I64);
}

#[test]
fn bool_scalar_takes_array_dtype() {
    assert_eq!((&arr(&[2u16]) * true).dtype(), DType::U16);
    assert_eq!((&arr(&[2.5f32]) + true).to_vec::<f32>().unwrap(), [3.5]);
}

#[test]
fn float64_array_keeps_double_precision() {
    let x = arr(&[0.0f64]);
    let y = &x + 0.1;
    assert_eq!(y.item::<f64>().unwrap(), 0.1);
}

#[test]
fn complex_scalar_is_complex64() {
    let y = &arr(&[1.0f32]) * Complex64::new(0.0, 1.0);
    assert_eq!(y.dtype(), DType::Complex64);
    assert_eq!(y.item::<Complex64>().unwrap(), Complex64::new(0.0, 1.0));
}

#[test]
fn scalar_on_the_left() {
    let x = arr(&[1.0f32, 4.0]);
    assert_eq!((1.0 - &x).to_vec::<f32>().unwrap(), [0.0, -3.0]);
    assert_eq!((2.0 / x.clone()).to_vec::<f32>().unwrap(), [2.0, 0.5]);
    assert_eq!((3 * &arr(&[2i16])).dtype(), DType::I16);
    assert!(matches!(
        (300 + &arr(&[1i8])).eval(),
        Err(Error::Op {
            kind: OpErrorKind::InvalidArgument(_),
            ..
        })
    ));
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
fn functions_accept_scalars() {
    let x = arr(&[f16::from_f32(-1.0), f16::from_f32(2.0)]);
    let relu = x.maximum(0.0);
    assert_eq!(relu.dtype(), DType::F16);
    assert_eq!(
        relu.to_vec::<f16>().unwrap(),
        [f16::ZERO, f16::from_f32(2.0)]
    );
    assert_eq!(ops::minimum(1.0, &x).dtype(), DType::F16);

    let ints = arr(&[-1i32, 3]);
    assert_eq!(ints.greater(0).to_vec::<bool>().unwrap(), [false, true]);

    let m = arr(&[true, false]);
    let y = arr(&[5.0f32, 6.0]);
    let w = m.select(&y, 0.0);
    assert_eq!(w.dtype(), DType::F32);
    assert_eq!(w.to_vec::<f32>().unwrap(), [5.0, 0.0]);
    assert_eq!(
        ops::where_(true, 1, &y).to_vec::<f32>().unwrap(),
        [1.0, 1.0]
    );
}

#[test]
fn scalars_alone_use_default_dtypes() {
    assert_eq!(ops::add(1, 2).dtype(), DType::I32);
    assert_eq!(ops::add(1, 2.0).dtype(), DType::F32);
    assert_eq!(ops::add(true, false).dtype(), DType::Bool);
    assert_eq!(ops::add(1i64 << 40, 1).dtype(), DType::I64);
    assert_eq!(
        ops::multiply(2, Complex64::new(1.0, 0.0)).dtype(),
        DType::Complex64
    );
    assert_eq!(ops::divide(1, 2).item::<f32>().unwrap(), 0.5);
}

#[test]
fn earlier_failure_wins_over_overflow() {
    let failed = &arr(&[1i8, 2]) + &arr(&[1i8, 2, 3]);
    assert!(matches!(
        ops::add(&failed, 300).check(),
        Err(Error::Op {
            kind: OpErrorKind::ShapeMismatch { .. },
            ..
        })
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn scalars_on_gpu() {
    if cfg!(miri) {
        return;
    }
    let y = with_stream(Device::gpu(0), || {
        let x = arr(&[f16::from_f32(1.0), f16::from_f32(3.0)]);
        (2.0 * &x - 1).to_vec::<f16>()
    });
    match y {
        Ok(v) => assert_eq!(v, [f16::from_f32(1.0), f16::from_f32(5.0)]),
        Err(e) => eprintln!("skipped: no Metal device ({e})"),
    }
}
