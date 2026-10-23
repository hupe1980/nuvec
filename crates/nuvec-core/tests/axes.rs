//! The `Axes` and `IntoShape` argument forms (compile-pass coverage of every impl) and their
//! semantics: `..` = all, `1..` = trailing, an empty list = none (MLX C++), negative axes,
//! and `InvalidAxis`/`InvalidShape` failures.

use nuvec_core::*;

fn x234() -> Array {
    ops::arange(0.0, 24.0, 1.0).reshape([2, 3, 4])
}

fn takes_axes<A: Axes>(_: A) {}
fn takes_shape<S: IntoShape>(_: S) {}

#[test]
fn every_axes_form_compiles_and_resolves() {
    let x = x234();
    let v_usize: Vec<usize> = vec![0, 2];
    let s_i32: &[i32] = &[0, 1];
    takes_axes(-1);
    takes_axes(x.ndim() - 1);
    takes_axes([0, 2]);
    takes_axes([]);
    takes_axes(&[0i32, 1][..]);
    takes_axes(&v_usize[..]);
    takes_axes(vec![0i32]);
    takes_axes(v_usize.clone());
    takes_axes(..);
    takes_axes(1..);
    assert_eq!(x.sum(-1).shape(), [2, 3]);
    assert_eq!(x.sum(x.ndim() - 1).shape(), [2, 3]);
    assert_eq!(x.sum([0, 2]).shape(), [3]);
    assert_eq!(x.sum(s_i32).shape(), [4]);
    assert_eq!(x.sum(&v_usize[..]).shape(), [3]);
    assert_eq!(x.sum(vec![1i32]).shape(), [2, 4]);
    assert_eq!(x.sum(v_usize).shape(), [3]);
    assert_eq!(x.sum(..).shape(), [0i32; 0]);
    assert_eq!(x.sum(1..).shape(), [2]);
    assert_eq!(x.sum(-2..).shape(), [2]);
    assert_eq!(
        x.sum(3..).shape(),
        [2, 3, 4],
        "3.. on 3 dimensions is empty"
    );
    assert_eq!(x.sum([]).shape(), [2, 3, 4]);
    assert_eq!(x.sum_keepdims(-1).shape(), [2, 3, 1]);
    assert_eq!(x.sum_keepdims(..).shape(), [1, 1, 1]);
    assert_eq!(ops::sum(&x, [0, -1]).shape(), [3]);
}

#[test]
fn every_shape_form_compiles() {
    let (b, l): (i32, i32) = (2, 3);
    let (n, k): (usize, usize) = (3, 4);
    let v: Vec<usize> = vec![2, 12];
    takes_shape([b, l, -1]);
    takes_shape([n, k]);
    takes_shape(&[2i32, 3][..]);
    takes_shape(&v[..]);
    takes_shape(vec![1i32]);
    takes_shape(v.clone());
    takes_shape(());
    let x = x234();
    assert_eq!(x.reshape([b, l, -1]).shape(), [2, 3, 4]);
    assert_eq!(x.reshape([n * 2, k]).shape(), [6, 4]);
    assert_eq!(x.reshape(v).shape(), [2, 12]);
    assert_eq!(x.reshape(&[24i32][..]).shape(), [24]);
    assert_eq!(ops::zeros([3, 4], DType::F32).shape(), [3, 4]);
    assert_eq!(ops::zeros((), DType::F32).ndim(), 0);
    assert_eq!(ops::ones(vec![n], DType::I8).shape(), [3]);
    assert_eq!(x.broadcast_to([5, 2, 3, 4]).shape(), [5, 2, 3, 4]);
    assert_eq!(ops::full([2usize], 1.5f32).shape(), [2]);
}

#[test]
fn empty_axes_reduce_nothing() {
    let x = Array::from_slice(&[1i8, -2, 3, 4, 5, 6], [2, 3]);
    for (got, dtype) in [
        (x.sum([]), DType::I8),
        (x.prod([]), DType::I8),
        (x.max([]), DType::I8),
        (x.min([]), DType::I8),
        (x.all([]), DType::Bool),
        (x.any(Vec::<i32>::new()), DType::Bool),
        (x.mean([]), DType::F32),
        (x.logsumexp([]), DType::F32),
        (x.var([]), DType::F32),
        (x.std_keepdims([]), DType::F32),
        (x.softmax([]), DType::F32),
    ] {
        assert_eq!((got.shape(), got.dtype()), (&[2, 3][..], dtype));
    }
    assert_eq!(x.sum([]).to_vec::<i8>().unwrap(), [1, -2, 3, 4, 5, 6]);
    assert_eq!(
        x.mean([]).to_vec::<f32>().unwrap(),
        [1.0, -2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(x.var([]).to_vec::<f32>().unwrap(), [0.0; 6]);
    // `squeeze` and `transpose` with an empty list change nothing either... except that a
    // transpose needs a full permutation.
    assert_eq!(x.squeeze([]).shape(), [2, 3]);
    assert!(x.transpose([]).is_failed());
    // `..` where an operation has no "flattened" meaning.
    let e = x.split(2, ..).remove(0).check().unwrap_err();
    assert!(
        matches!(e.kind(), Some(OpErrorKind::InvalidAxis { .. })),
        "{e}"
    );
}

#[test]
fn negative_and_special_axes() {
    let x = x234();
    assert_eq!(x.transpose(..).shape(), [4, 3, 2]);
    assert_eq!(x.transpose([1, -1, 0]).shape(), [3, 4, 2]);
    assert_eq!(x.expand_dims(-1).shape(), [2, 3, 4, 1]);
    assert_eq!(x.expand_dims([0, 2]).shape(), [1, 2, 1, 3, 4]);
    let y = x.expand_dims([0, -1]);
    assert_eq!(y.shape(), [1, 2, 3, 4, 1]);
    assert_eq!(y.squeeze(..).shape(), [2, 3, 4]);
    assert_eq!(y.squeeze(-1).shape(), [1, 2, 3, 4]);
    assert_eq!(
        x.argmax(..).shape(),
        [0i32; 0],
        "`..` is the flattened array"
    );
    assert_eq!(x.argmax_keepdims(..).shape(), [1, 1, 1]);
    assert_eq!(x.argmax(-1).shape(), [2, 3]);
    assert_eq!(x.cumsum(..).shape(), [24]);
    assert_eq!(x.sort(0).shape(), [2, 3, 4]);
    assert_eq!(x.take(&Array::from([0i32, 2]), ..).shape(), [2]);
}

#[test]
fn invalid_axes_fail_with_invalid_axis() {
    let x = x234();
    let line = line!() + 2;
    for (k, bad) in [
        x.sum(3),
        x.sum(-4),
        x.sum([0, 0]),
        x.sum([1, -2]),
        x.sum(4..),
        x.argmax(3),
        x.cumsum(-4),
        x.sort(usize::MAX),
        x.softmax(5),
        x.squeeze(7),
    ]
    .into_iter()
    .enumerate()
    {
        let e = bad.check().unwrap_err();
        assert!(
            matches!(e.kind(), Some(OpErrorKind::InvalidAxis { .. })),
            "{k}: {e}"
        );
        let at = e.location().unwrap();
        assert!(
            at.file().ends_with("tests/axes.rs") && at.line() >= line,
            "{e}"
        );
    }
    match x.sum(3).check().unwrap_err().kind() {
        Some(OpErrorKind::InvalidAxis { axis: 3, ndim: 3 }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn invalid_shapes_fail_with_invalid_shape() {
    let x = x234();
    for bad in [
        x.reshape([5, -1]),
        x.reshape([-1, -1]),
        x.reshape([-2, 12]),
        x.reshape([25]),
        ops::zeros([-1, 2], DType::F32),
        ops::zeros(vec![i32::MAX as usize + 1], DType::U8),
        x.broadcast_to([2, -3, 4]),
    ] {
        let e = bad.check().unwrap_err();
        assert!(
            matches!(e.kind(), Some(OpErrorKind::InvalidShape(_))),
            "{e}"
        );
    }
    // A constructor's failed array keeps the requested shape when it is valid.
    let f = Array::from_slice(&[1.0f32, 2.0], [2, 2]);
    assert!(f.is_failed());
    assert_eq!(f.shape(), [2, 2]);
}
