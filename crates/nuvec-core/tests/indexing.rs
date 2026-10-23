//! Indexing (`x.i(..)`) and functional updates (`x.at(..).set/add`) against MLX-recorded
//! fixtures (basic, stepped, `NewAxis`, `Ellipsis`, advanced; transposed and reversed
//! layouts; five dtypes), views, invalid indices and gradients.

#[path = "common/api.rs"]
mod api;
#[path = "common/fixture.rs"]
mod fixture;

use nuvec_core::graph;
use nuvec_core::*;

type Index = fn(&Array, &[Array]) -> Array;

/// MLX's `x[...]` expressions and their nuvec spellings (several where Rust has several
/// range types for one Python slice).
// `x[3:1]` is Python's empty slice; `3..1` is the same empty range on purpose.
#[allow(clippy::reversed_empty_ranges)]
fn index_forms(expr: &str) -> Vec<Index> {
    match expr {
        "x[0]" => vec![|x, _| x.i(0), |x, _| x.i(0usize)],
        "x[-1]" => vec![|x, _| x.i(-1)],
        "x[1:3]" => vec![|x, _| x.i(1..3), |x, _| x.i(1usize..3), |x, _| x.i(1..=2)],
        "x[:, 1:3]" => vec![|x, _| x.i((.., 1..3)), |x, _| x.i((.., 1usize..=2usize))],
        "x[..., -1]" => vec![|x, _| x.i((Ellipsis, -1))],
        "x[:, None, :]" => vec![|x, _| x.i((.., NewAxis, ..)), |x, _| x.i((.., NewAxis))],
        "x[::-1]" => vec![|x, _| x.i((..).step(-1))],
        "x[1:, :3, 0:6:2]" => vec![|x, _| x.i((1.., ..3, (0..6).step(2))), |x, _| {
            x.i((1usize.., ..=2, (0usize..6).step(2)))
        }],
        "x[None]" => vec![|x, _| x.i(NewAxis)],
        "x[..., None]" => vec![|x, _| x.i((Ellipsis, NewAxis))],
        "x[2, ..., 1:4]" => vec![|x, _| x.i((2, Ellipsis, 1..4))],
        "x[-3:-1, ::2]" => vec![|x, _| x.i((-3..-1, (..).step(2)))],
        "x[:, ::-2, 4]" => vec![|x, _| x.i((.., (..).step(-2), 4))],
        "x[1:100]" => vec![|x, _| x.i(1..100)],
        "x[3:1]" => vec![|x, _| x.i(3..1)],
        "x[..., ::-1, None, 0]" => vec![|x, _| x.i((Ellipsis, (..).step(-1), NewAxis, 0))],
        "x[:3]" => vec![|x, _| x.i(..3), |x, _| x.i(..=2), |x, _| x.i(..3usize)],
        "x[1, 2, 3]" => vec![|x, _| x.i((1, 2, 3))],
        "x[...]" => vec![|x, _| x.i(Ellipsis)],
        "x[:, -2:]" => vec![|x, _| x.i((.., -2..))],
        "x[::2, 1::2, ::-3]" => vec![|x, _| x.i(((..).step(2), (1..).step(2), (..).step(-3)))],
        // A negative step walks the selected range backwards (ndarray's rule).
        "x[3:0:-1]" => vec![|x, _| x.i((1..4).step(-1)), |x, _| x.i((1..=3).step(-1))],
        "x[1::-1]" => vec![|x, _| x.i((..2).step(-1)), |x, _| x.i((..=1usize).step(-1))],
        "x[:0:-1]" => vec![|x, _| x.i((1..).step(-1))],
        "x[3::-2]" => vec![|x, _| x.i((0..4).step(-2))],
        // Advanced.
        "x[a1]" => vec![|x, a| x.i(&a[0]), |x, a| x.i(a[0].clone())],
        "x[a1, a2]" => vec![|x, a| x.i((&a[0], &a[1]))],
        "x[a1, :, a2]" => vec![|x, a| x.i((&a[0], .., &a[1]))],
        "x[:, a1, a2]" => vec![|x, a| x.i((.., &a[0], &a[1]))],
        "x[a1, 2]" => vec![|x, a| x.i((&a[0], 2))],
        "x[1, :, a1]" => vec![|x, a| x.i((1, .., &a[0]))],
        "x[a1, 1:4]" => vec![|x, a| x.i((&a[0], 1..4))],
        "x[..., a1]" => vec![|x, a| x.i((Ellipsis, &a[0]))],
        "x[None, a1]" => vec![|x, a| x.i((NewAxis, &a[0]))],
        "x[a1, None]" => vec![|x, a| x.i((&a[0], NewAxis))],
        "x[::-1, a1]" => vec![|x, a| x.i(((..).step(-1), &a[0]))],
        "x[a1, a2, a1]" => vec![|x, a| x.i((&a[0], &a[1], &a[0]))],
        other => panic!("no nuvec spelling for {other}"),
    }
}

type Update = fn(Array, &[Array], &Array) -> Array;

fn update_form(expr: &str) -> Update {
    match expr {
        "x[:, 1:3]=v" => |x, _, v| x.at((.., 1..3)).set(v),
        "x[0]=v" => |x, _, v| x.at(0).set(v),
        "x[..., -1]=v" => |x, _, v| x.at((Ellipsis, -1)).set(v),
        "x[::2, :, 1:4]=v" => |x, _, v| x.at(((..).step(2), .., 1..4)).set(v),
        "x[:, None, 2]=v" => |x, _, v| x.at((.., NewAxis, 2)).set(v),
        "x[a1]=v" => |x, a, v| x.at(&a[0]).set(v),
        "x[a1, :, a2]=v" => |x, a, v| x.at((&a[0], .., &a[1])).set(v),
        "x.at[1:3].add(v)" => |x, _, v| x.at(1..3).add(v),
        "x.at[..., 0].add(v)" => |x, _, v| x.at((Ellipsis, 0)).add(v),
        "x.at[a1].add(v)" => |x, a, v| x.at(&a[0]).add(v),
        "x.at[a1, a2].add(v)" => |x, a, v| x.at((&a[0], &a[1])).add(v),
        "x.at[:, a1].add(v)" => |x, a, v| x.at((.., &a[0])).add(v),
        other => panic!("no nuvec spelling for {other}"),
    }
}

fn run_on(device: Device, f: impl FnOnce()) {
    with_stream(device, f);
}

#[test]
#[cfg_attr(miri, ignore = "hundreds of cases: covered natively")]
fn indexing_matches_mlx() {
    let cases = api::cases("index");
    assert!(cases.len() >= 300, "{} cases", cases.len());
    let mut failures = Vec::new();
    run_on(Device::cpu(), || {
        for case in &cases {
            let xs = api::inputs(case);
            for form in index_forms(&case.params["expr"]) {
                if let Err(e) = api::compare(case, &form(&xs[0], &xs[1..]), 0.0) {
                    failures.push(e);
                }
            }
        }
    });
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[cfg_attr(miri, ignore = "GPU")]
fn indexing_matches_mlx_on_the_gpu() {
    if !gpu_available() {
        return;
    }
    let mut failures = Vec::new();
    run_on(Device::gpu(0), || {
        for case in api::cases("index") {
            let xs = api::inputs(&case);
            let form = index_forms(&case.params["expr"])[0];
            if let Err(e) = api::compare(&case, &form(&xs[0], &xs[1..]), 0.0) {
                failures.push(e);
            }
        }
    });
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[cfg_attr(miri, ignore = "dozens of cases: covered natively")]
fn updates_match_mlx() {
    let mut failures = Vec::new();
    let mut n = 0;
    for device in [Device::cpu(), Device::gpu(0)] {
        if device != Device::cpu() && !gpu_available() {
            continue;
        }
        run_on(device, || {
            for case in api::cases("set").into_iter().chain(api::cases("add")) {
                let xs = api::inputs(&case);
                let (x, rest) = xs.split_first().unwrap();
                let (v, idx) = rest.split_last().unwrap();
                let out = update_form(&case.params["expr"])(x.clone(), idx, v);
                n += 1;
                // Repeated positions accumulate in another order on the GPU (atomics).
                let ulps = if case.op == "add" { 8.0 } else { 0.0 };
                if let Err(e) = api::compare(&case, &out, ulps) {
                    failures.push(format!("{device}: {e}"));
                }
            }
        });
    }
    assert!(n >= 48, "{n} cases");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn gpu_available() -> bool {
    let ok = !cfg!(miri)
        && with_stream(Device::gpu(0), || {
            (Array::from(1.0f32) + 1.0).eval().is_ok()
        });
    if !ok && std::env::var("NUVEC_REQUIRE_GPU").is_ok_and(|v| v == "1") {
        panic!("NUVEC_REQUIRE_GPU=1 but no Metal device is available");
    }
    ok
}

#[test]
fn basic_indices_are_views() {
    with_stream(Device::cpu(), || {
        let x = ops::arange(0, 24, 1).reshape([2, 3, 4]);
        x.eval().unwrap();
        for y in [
            x.i(0),
            x.i((.., 1..3)),
            x.i((Ellipsis, (..).step(-1))),
            x.i((.., NewAxis, 1)),
            x.i(Ellipsis),
        ] {
            // Only slice and reshape nodes: no copy, no gather.
            let mut node = y.clone();
            while let Some(p) = graph::primitive(&node) {
                assert!(
                    matches!(
                        p,
                        graph::Primitive::Slice { .. } | graph::Primitive::Reshape(_)
                    ),
                    "{p:?}"
                );
                node = graph::inputs(&node).remove(0);
            }
            y.eval().unwrap();
        }
    });
}

#[test]
fn invalid_indices_fail_without_panicking() {
    let x = ops::zeros([4, 5], DType::F32);
    let line = line!() + 2;
    for (k, bad) in [
        x.i(4),
        x.i(-5),
        x.i((Ellipsis, 0, Ellipsis)),
        x.i((0, 0, 0)),
        x.i((0..2).step(0)),
        x.clone().at(9).set(1.0),
    ]
    .into_iter()
    .enumerate()
    {
        let e = bad.check().unwrap_err();
        assert!(
            matches!(e.kind(), Some(OpErrorKind::InvalidIndex(_))),
            "{k}: {e}"
        );
        let at = e.location().unwrap();
        assert!(
            at.file().ends_with("tests/indexing.rs") && at.line() >= line,
            "{e}"
        );
    }
    let float_index = ops::zeros([2], DType::F32);
    let e = x.i(&float_index).check().unwrap_err();
    assert!(
        matches!(e.kind(), Some(OpErrorKind::DTypeMismatch { .. })),
        "{e}"
    );
}

#[test]
fn repeated_indices_accumulate_in_add() {
    let h = ops::zeros([3, 2], DType::F32);
    let ids = Array::from([0i32, 2, 0, 0]);
    let g = ops::ones([4, 2], DType::F32);
    let out = h.at(&ids).add(&g);
    assert_eq!(out.to_vec::<f32>().unwrap(), [3.0, 3.0, 0.0, 0.0, 1.0, 1.0]);
    // A weak scalar casts to the base dtype; a broadcast value fills the region.
    let x = ops::zeros([2, 3], DType::F16).at((.., 1)).set(2.5);
    assert_eq!(x.dtype(), DType::F16);
    let y = ops::zeros([2, 3], DType::I32)
        .at(..)
        .set(Array::from([1i32, 2, 3]));
    assert_eq!(y.to_vec::<i32>().unwrap(), [1, 2, 3, 1, 2, 3]);
}

#[test]
#[cfg_attr(miri, ignore = "finite differences: slow under Miri")]
fn indexing_and_updates_differentiate() {
    with_stream(Device::cpu(), || {
        let idx = Array::from([2i32, 0, 2]);
        api::check_grad(
            "i basic",
            &|xs| xs[0].i((1.., (..).step(-1))),
            &[&[3, 4]],
            -1.0,
            1.0,
        );
        api::check_grad(
            "i newaxis",
            &|xs| xs[0].i((Ellipsis, NewAxis, 1)),
            &[&[2, 3]],
            -1.0,
            1.0,
        );
        let i2 = idx.clone();
        api::check_grad(
            "i advanced",
            &move |xs| xs[0].i((&i2, 1..)),
            &[&[3, 4]],
            -1.0,
            1.0,
        );
        let i3 = idx.clone();
        api::check_grad(
            "i two arrays",
            &move |xs| xs[0].i((.., &i3, &i3)),
            &[&[2, 3, 3]],
            -1.0,
            1.0,
        );
        api::check_grad(
            "at set basic",
            &|xs| xs[0].clone().at((.., 1..3)).set(&xs[1]),
            &[&[3, 4], &[3, 2]],
            -1.0,
            1.0,
        );
        api::check_grad(
            "at add basic",
            &|xs| xs[0].clone().at((Ellipsis, 0)).add(&xs[1]),
            &[&[3, 4], &[3]],
            -1.0,
            1.0,
        );
        let i4 = Array::from([1i32, 0]);
        api::check_grad(
            "at set advanced",
            &move |xs| xs[0].clone().at(&i4).set(&xs[1]),
            &[&[3, 2], &[2, 2]],
            -1.0,
            1.0,
        );
        let i5 = idx.clone();
        api::check_grad(
            "at add advanced (repeated)",
            &move |xs| xs[0].clone().at(&i5).add(&xs[1]),
            &[&[3, 2], &[3, 2]],
            -1.0,
            1.0,
        );
    });
}

#[test]
fn indexing_failed_arrays_is_sticky() {
    let bad = ops::zeros([2, 3], DType::F32).matmul(&ops::zeros([2, 3], DType::F32));
    let err = bad.clone().check().unwrap_err();
    assert_eq!(bad.i((.., 0)).check().unwrap_err(), err);
    assert_eq!(bad.clone().at(0).set(1.0).check().unwrap_err(), err);
    let ok = ops::zeros([2, 3], DType::F32);
    assert_eq!(ok.clone().at(0).set(&bad).check().unwrap_err(), err);
    assert_eq!(ok.i(bad.astype(DType::I32)).check().unwrap_err(), err);
}

#[test]
fn huge_usize_indices_fail_instead_of_wrapping() {
    let x = ops::zeros([4], DType::F32);
    let e = x.i(usize::MAX).check().unwrap_err();
    assert!(
        matches!(e.kind(), Some(OpErrorKind::InvalidIndex(_))),
        "{e}"
    );
    assert_eq!(
        x.i(usize::MAX - 1..).shape(),
        [0],
        "a huge start clamps, as Python"
    );
}

/// `h.at(&ids).add(&g)` on a large table scatters along the indexed axis: no temporary of the
/// table's size (positions) is allocated, so the peak stays near the table plus the update.
#[test]
#[cfg_attr(miri, ignore = "GPU memory")]
fn embedding_update_does_not_allocate_positions() {
    if !gpu_available() {
        return;
    }
    with_stream(Device::gpu(0), || {
        let (vocab, dim, n) = (32_768usize, 64usize, 256usize);
        let table = ops::zeros([vocab, dim], DType::F32);
        table.eval().unwrap();
        let ids = Array::from_slice(
            &(0..n as i32).map(|i| (i * 97) % 4096).collect::<Vec<_>>(),
            [n],
        );
        let g = ops::ones([n, dim], DType::F32);
        eval(&[ids.clone(), g.clone()]).unwrap();
        let device = nuvec_gpu::Device::shared().unwrap();
        device.reset_peak();
        let before = device.memory().active;
        let out = table.clone().at(&ids).add(&g);
        out.eval().unwrap();
        let peak = device.memory().peak;
        let table_bytes = vocab * dim * 4;
        // The result (a table) and small temporaries; a positions array would add another
        // table-sized int32 buffer.
        assert!(
            peak < before + table_bytes + table_bytes / 2,
            "peak {peak}, before {before}, table {table_bytes}"
        );
        assert_eq!(out.i((0, 0)).item::<f32>().unwrap(), 1.0);
    });
}
