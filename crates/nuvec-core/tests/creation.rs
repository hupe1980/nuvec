//! Creating arrays: `array!`, `From`, `FromIterator`, `from_slice`, `full`,
//! `arange`, `*_like`, with MLX's default dtypes.

use nuvec_core::*;

#[test]
fn array_macro_shapes_and_dtypes() {
    let m = array![[1.0, 2.0], [3.0, 4.0]];
    assert_eq!((m.shape(), m.dtype()), (&[2, 2][..], DType::F32));
    assert_eq!(m.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(array![1, 2, 3].dtype(), DType::I32);
    assert_eq!(array![1u8, 2, 3].dtype(), DType::U8);
    assert_eq!(array![-1i64, 2].dtype(), DType::I64);
    assert_eq!(array![true, false].dtype(), DType::Bool);
    assert_eq!(array![f16::from_f32(1.0)].dtype(), DType::F16);
    assert_eq!(array![2.5].shape(), [1]);
    // f64 literals follow the float32 default, also through variables.
    assert_eq!(array![1.0f64].dtype(), DType::F32);
    let v: f64 = 0.25;
    assert_eq!(array![v, v].dtype(), DType::F32);
    let t = array![[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    assert_eq!(t.shape(), [2, 2, 2]);
    let q = array![[[[1.5]]], [[[2.5]]]];
    assert_eq!(q.shape(), [2, 1, 1, 1]);
    assert_eq!(array![[1, 2],].shape(), [1, 2], "trailing commas");
}

#[test]
fn from_conversions_keep_the_element_type() {
    let a = Array::from([1.0f32, 2.0]);
    assert_eq!((a.shape(), a.dtype()), (&[2][..], DType::F32));
    assert_eq!(Array::from(vec![1i64; 6]).shape(), [6]);
    assert_eq!(Array::from(vec![1i64; 6]).dtype(), DType::I64);
    assert_eq!(Array::from(&[1u16, 2][..]).dtype(), DType::U16);
    let s = Array::from(3i32);
    assert_eq!((s.ndim(), s.dtype()), (0, DType::I32));
    assert_eq!(
        Array::from(3.0f64).dtype(),
        DType::F64,
        "strong: f64 stays float64"
    );
    assert_eq!(Array::from(vec![0.5f64]).dtype(), DType::F64);
    let c: Array = (0..10).map(|i| i as f32).collect();
    assert_eq!((c.shape(), c.dtype()), (&[10][..], DType::F32));
    // A strong scalar keeps its dtype in an operation; a Rust number adapts.
    let h = Array::from([1.0f32]).astype(DType::F16);
    assert_eq!((&h + Array::from(1.0f32)).dtype(), DType::F32);
    assert_eq!((&h + 1.0f32).dtype(), DType::F16);
}

#[test]
fn from_slice_checks_the_length() {
    let a = Array::from_slice(&[1u8, 2, 3, 4, 5, 6], [2, 3]);
    assert_eq!(a.shape(), [2, 3]);
    let line = line!() + 1;
    let bad = Array::from_slice(&[1.0f32, 2.0], [3, 2]);
    assert_eq!(bad.shape(), [3, 2], "the requested shape is kept");
    let e = bad.check().unwrap_err();
    assert_eq!(
        e.kind(),
        Some(&OpErrorKind::SizeMismatch {
            expected: 6,
            actual: 2
        })
    );
    assert_eq!(e.location().unwrap().line(), line);
}

#[test]
fn constructors_and_likes() {
    assert_eq!(
        ops::zeros([2, 3], DType::F32).to_vec::<f32>().unwrap(),
        [0.0; 6]
    );
    assert_eq!(ops::ones((), DType::I8).item::<i8>().unwrap(), 1);
    let f = ops::full([2, 2], 0.5);
    assert_eq!(
        (f.dtype(), f.to_vec::<f32>().unwrap()),
        (DType::F32, vec![0.5; 4])
    );
    assert_eq!(
        ops::full([3], 7).dtype(),
        DType::I32,
        "a Rust integer is weak"
    );
    let strong = ops::full([3], Array::from(7u8));
    assert_eq!(
        (strong.dtype(), strong.to_vec::<u8>().unwrap()),
        (DType::U8, vec![7; 3])
    );
    let x = Array::from([1u16, 2, 3]).reshape([3, 1]);
    for (like, value) in [(x.zeros_like(), 0), (x.ones_like(), 1), (x.full_like(9), 9)] {
        assert_eq!((like.shape(), like.dtype()), (&[3, 1][..], DType::U16));
        assert_eq!(like.to_vec::<u16>().unwrap(), [value; 3]);
    }
    assert_eq!(ops::zeros_like(&x).shape(), [3, 1]);
    assert_eq!(
        ops::full_like(&x, 2.0).dtype(),
        DType::U16,
        "weak: takes x's dtype"
    );
}

#[test]
fn arange_infers_mlx_dtypes() {
    let i = ops::arange(0, 10, 1);
    assert_eq!((i.dtype(), i.shape()), (DType::I32, &[10][..]));
    let f = ops::arange(0.0, 1.0, 0.25);
    assert_eq!(
        (f.dtype(), f.to_vec::<f32>().unwrap()),
        (DType::F32, vec![0.0, 0.25, 0.5, 0.75])
    );
    assert_eq!(
        ops::arange(0, 3, 0.5).dtype(),
        DType::F32,
        "any float gives float32"
    );
    assert_eq!(
        ops::arange(1i64 << 40, (1i64 << 40) + 2, 1).dtype(),
        DType::I64
    );
    assert_eq!(ops::arange(5, 0, -2).to_vec::<i32>().unwrap(), [5, 3, 1]);
    assert!(ops::arange(0, 1, 0).is_failed());
}

#[test]
fn printing() {
    assert_eq!(array![1, 2].to_string(), "array([1, 2], dtype=int32)");
    let lazy = array![1.0] + 1.0;
    let debug = format!("{lazy:?}");
    assert!(
        debug.contains("unevaluated") && !lazy.is_evaluated(),
        "{debug}"
    );
    lazy.eval().unwrap();
    assert!(format!("{lazy:?}").contains("[2]"), "values once evaluated");
    let mut slot = array![1, 2, 3];
    slot = std::mem::take(&mut slot).at(0).set(9);
    assert_eq!(slot.to_vec::<i32>().unwrap(), [9, 2, 3]);
}
