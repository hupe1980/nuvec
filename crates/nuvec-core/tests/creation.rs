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

/// The devices these tests evaluate on: the CPU, and the GPU when there is one.
fn devices() -> Vec<Device> {
    let gpu = !cfg!(miri)
        && with_stream(Stream::default_for(Device::gpu(0)), || {
            (&Array::from(1.0f32) + 1.0).eval().is_ok()
        });
    [Some(Device::cpu()), gpu.then(|| Device::gpu(0))]
        .into_iter()
        .flatten()
        .collect()
}

/// `zeros`, `ones` and `full` with a number give the same elements in every dtype and
/// shape (0-d, empty, small, and large enough to be filled at evaluation), on each device.
#[test]
fn constant_creations_in_every_dtype_shape_and_device() {
    let shapes: [&[i32]; 6] = [&[], &[0], &[3, 0, 2], &[7], &[2, 3, 5], &[1 << 22]];
    let values = [0.0, 1.0, -2.5, 7.0, f64::NAN, f64::INFINITY, -0.0];
    for device in devices() {
        with_stream(Stream::default_for(device), || {
            for dtype in [
                DType::Bool,
                DType::U8,
                DType::U16,
                DType::U32,
                DType::U64,
                DType::I8,
                DType::I16,
                DType::I32,
                DType::I64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ] {
                for shape in shapes {
                    let n: usize = shape.iter().map(|&d| d as usize).product();
                    let check = |a: Array, want: f64, what: &str| {
                        assert_eq!((a.shape(), a.dtype()), (shape, dtype), "{what}");
                        // The expectation: the value converted once, as a 0-d host array.
                        let one = ops::full((), want).astype(dtype);
                        let got = a.astype(DType::F64).reshape([-1]);
                        let want = ops::broadcast_to(&one.astype(DType::F64), [n as i32]);
                        let (g, w) = (got.to_vec::<f64>().unwrap(), want.to_vec::<f64>().unwrap());
                        assert_eq!(g.len(), n, "{what}");
                        assert!(
                            g.iter()
                                .zip(&w)
                                .all(|(a, b)| a.to_bits() == b.to_bits()
                                    || (a.is_nan() && b.is_nan())),
                            "{what} on {device:?}"
                        );
                    };
                    check(
                        ops::zeros(shape, dtype),
                        0.0,
                        &format!("zeros {dtype} {shape:?}"),
                    );
                    check(
                        ops::ones(shape, dtype),
                        1.0,
                        &format!("ones {dtype} {shape:?}"),
                    );
                    for &v in &values {
                        let float = matches!(dtype.category(), Category::Float);
                        if dtype == DType::Bool
                            || !float && (!v.is_finite() || v.fract() != 0.0 || v < 0.0)
                        {
                            continue;
                        }
                        check(
                            ops::full(shape, v).astype(dtype),
                            v,
                            &format!("full {v} {dtype} {shape:?}"),
                        );
                    }
                }
            }
        });
    }
}

/// Creations under `vmap`, `grad`, `jvp` and `checkpoint` give the eager results, and a
/// gradient through a creation is zero; a created array updated in place keeps the rest.
#[test]
fn creations_under_transforms_and_in_place_updates() {
    for device in devices() {
        with_stream(Stream::default_for(device), || {
            let x = Array::from([1.0f32, 2.0, 3.0, 4.0]);
            let f = |x: &Array| {
                Ok(x * ops::ones([4], DType::F32)
                    + ops::full([4], 0.5)
                    + ops::zeros([4], DType::F32))
            };
            let g = grad(|x: &Array| Ok(f(x)?.sum(..)))(&x).unwrap();
            assert_eq!(g.to_vec::<f32>().unwrap(), [1.0; 4]);
            // d/dx of a creation alone is zero.
            let z = grad(|x: &Array| Ok((x * 0.0 + ops::full([4], 3.0)).sum(..)))(&x).unwrap();
            assert_eq!(z.to_vec::<f32>().unwrap(), [0.0; 4]);
            let (y, t) = jvp(|x: &Array| f(x), &x, &ops::ones([4], DType::F32)).unwrap();
            assert_eq!(y.to_vec::<f32>().unwrap(), [1.5, 2.5, 3.5, 4.5]);
            assert_eq!(t.to_vec::<f32>().unwrap(), [1.0; 4]);
            let rows = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4]);
            let v = vmap(|r: &Array| f(r)).in_axes(0).out_axes(0);
            assert_eq!(
                v.call(&rows).unwrap().to_vec::<f32>().unwrap(),
                [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
            );
            let c = checkpoint(move |x: &Array| f(x));
            assert_eq!(
                c(&x).unwrap().to_vec::<f32>().unwrap(),
                [1.5, 2.5, 3.5, 4.5]
            );
            // A large created pool written in place, as nuvec-lm's caches.
            let pool = ops::zeros([1 << 20], DType::F32);
            let written = pool.slice_update(&Array::from([9.0f32, 9.0]), 10..12);
            let back = written.to_vec::<f32>().unwrap();
            assert_eq!(
                (back[9], back[10], back[11], back[12]),
                (0.0, 9.0, 9.0, 0.0)
            );
        });
    }
}

/// Argument errors are located at the creation's call site.
#[test]
fn creation_errors_are_located_at_the_call() {
    let line = line!() + 1;
    let a = ops::zeros([-2, 3], DType::F32);
    let e = a.eval().unwrap_err();
    assert!(
        format!("{e}").contains(&format!("creation.rs:{line}")),
        "{e}"
    );
    let line = line!() + 1;
    let b = ops::full([usize::MAX as i64 as i32, 2], 1.0);
    let _ = line;
    assert!(b.eval().is_err());
}
