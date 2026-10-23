//! `vmap`: every primitive against the per-example loop, composition with the other
//! transforms, MLX's edge cases and errors.

mod common;

use common::{Rng, random_layout};
use nuvec_core::ops;
use nuvec_core::*;

type F = fn(&[Array]) -> Array;

/// Example `i` of `a` along `axis`.
fn take(a: &Array, axis: usize, i: usize) -> Array {
    a.split(a.shape()[axis], axis)[i].squeeze(axis)
}

fn assert_close(name: &str, got: &Array, expected: &Array) {
    assert_eq!(got.shape(), expected.shape(), "{name}: shape");
    assert_eq!(got.dtype(), expected.dtype(), "{name}: dtype");
    let g = got
        .astype(DType::F64)
        .to_vec::<f64>()
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let e = expected.astype(DType::F64).to_vec::<f64>().unwrap();
    for (i, (g, e)) in g.iter().zip(&e).enumerate() {
        assert!(
            g == e || (g - e).abs() <= 1e-12 * e.abs().max(1.0) || (g.is_nan() && e.is_nan()),
            "{name}: element {i}: {g} vs per-example {e}"
        );
    }
}

/// Checks `vmap_flat(f)` against the per-example loop for every combination of batch axes:
/// each argument with per-example shape `shapes[k]` is batched (at every possible axis,
/// or unbatched when `shared[k]`), with random layouts and output axes.
fn check(name: &str, f: F, shapes: &[&[usize]], shared: &[bool]) {
    let mut rng = Rng::new(name.bytes().map(u64::from).sum());
    let b = 3;
    for round in 0..4 {
        let axes: Vec<Option<usize>> = shapes
            .iter()
            .enumerate()
            .map(|(k, s)| {
                (!shared.get(k).copied().unwrap_or(false)).then(|| rng.below(s.len() + 1))
            })
            .collect();
        let args: Vec<Array> = shapes
            .iter()
            .zip(&axes)
            .map(|(s, ax)| {
                let mut shape = s.to_vec();
                if let Some(ax) = ax {
                    shape.insert(*ax, b);
                }
                let n = shape.iter().product();
                let values: Vec<f64> = (0..n).map(|_| (rng.f64() * 0.2).round() * 0.5).collect();
                random_layout(&mut rng, &shape, values).0
            })
            .collect();
        let expected: Vec<Array> = (0..b)
            .map(|i| {
                let xs: Vec<Array> = args
                    .iter()
                    .zip(&axes)
                    .map(|(a, ax)| ax.map_or_else(|| a.clone(), |ax| take(a, ax, i)))
                    .collect();
                f(&xs)
            })
            .collect();
        let out_axis = rng.below(expected[0].ndim() + 1);
        let in_axes: Vec<Option<i32>> = axes.iter().map(|a| a.map(|a| a as i32)).collect();
        let got = vmap_flat(|xs| Ok(vec![f(xs)]), &in_axes, &[out_axis as i32])(&args)
            .unwrap_or_else(|e| panic!("{name} round {round} {axes:?}: {e}"))
            .remove(0);
        for (i, e) in expected.iter().enumerate() {
            assert_close(
                &format!("{name} round {round} axes {axes:?} out {out_axis} example {i}"),
                &take(&got, out_axis, i),
                e,
            );
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn elementwise() {
    let unary: [(&str, F); 9] = [
        ("negative", |x| x[0].negative()),
        ("abs", |x| x[0].abs()),
        ("exp", |x| x[0].exp()),
        ("log", |x| x[0].abs().log()),
        ("sqrt", |x| x[0].abs().sqrt()),
        ("sin", |x| x[0].sin()),
        ("cos", |x| x[0].cos()),
        ("tanh", |x| x[0].tanh()),
        ("sigmoid", |x| x[0].sigmoid()),
    ];
    for (name, f) in unary {
        check(name, f, &[&[2, 3]], &[]);
    }
    let binary: [(&str, F); 8] = [
        ("add", |x| x[0].add(&x[1])),
        ("subtract", |x| x[0].subtract(&x[1])),
        ("multiply", |x| x[0].multiply(&x[1])),
        ("divide", |x| x[0].divide(x[1].abs() + 1.0)),
        ("maximum", |x| x[0].maximum(&x[1])),
        ("minimum", |x| x[0].minimum(&x[1])),
        ("less", |x| x[0].less(&x[1])),
        ("equal", |x| x[0].equal(&x[1])),
    ];
    for (name, f) in binary {
        check(name, f, &[&[2, 3], &[2, 3]], &[]);
        check(&format!("{name}-broadcast"), f, &[&[2, 3], &[3]], &[]);
        check(
            &format!("{name}-shared"),
            f,
            &[&[3, 1], &[4]],
            &[false, true],
        );
        check(&format!("{name}-scalar"), f, &[&[], &[2]], &[]);
    }
    check(
        "where",
        |x| x[0].greater(0.0).select(&x[1], &x[2]),
        &[&[2, 3], &[3], &[2, 1]],
        &[],
    );
    check(
        "where-shared-cond",
        |x| x[0].greater(0.0).select(&x[1], 0.0),
        &[&[3], &[3]],
        &[true],
    );
    check("astype", |x| x[0].astype(DType::F32), &[&[2, 3]], &[]);
    check("stop_gradient", |x| x[0].stop_gradient(), &[&[2, 3]], &[]);
    check(
        "scalar-operand",
        |x| ops::subtract(1.0, x[0].multiply(2)),
        &[&[4]],
        &[],
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn views() {
    let cases: [(&str, F, &[usize]); 8] = [
        ("reshape", |x| x[0].reshape([3, 2]), &[2, 3]),
        ("reshape-infer", |x| x[0].reshape([-1]), &[2, 3, 2]),
        ("transpose", |x| x[0].transpose([2, 0, 1]), &[2, 3, 4]),
        ("t", |x| x[0].t(), &[2, 3]),
        ("slice", |x| x[0].i((1.., (..).step(-2))), &[3, 5]),
        ("broadcast_to", |x| x[0].broadcast_to([2, 3, 4]), &[3, 1]),
        ("expand_dims", |x| x[0].expand_dims(-1), &[2, 3]),
        ("squeeze", |x| x[0].squeeze(..), &[1, 3, 1]),
    ];
    for (name, f, shape) in cases {
        check(name, f, &[shape], &[]);
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn reductions_and_matmul() {
    let cases: [(&str, F); 9] = [
        ("sum", |x| x[0].sum(1)),
        ("sum-all", |x| x[0].sum(..)),
        ("prod", |x| x[0].prod_keepdims([0, 2])),
        ("max", |x| x[0].max(-1)),
        ("min", |x| x[0].min_keepdims(..)),
        ("mean", |x| x[0].mean(0)),
        ("all", |x| x[0].greater(0.0).all(1)),
        ("argmax", |x| x[0].argmax(1)),
        ("argmin-flat", |x| x[0].argmin(..)),
    ];
    for (name, f) in cases {
        check(name, f, &[&[2, 3, 4]], &[]);
    }
    let mm: F = |x| x[0].matmul(&x[1]);
    check("matmul", mm, &[&[3, 4], &[4, 2]], &[]);
    check(
        "matmul-shared-left",
        mm,
        &[&[3, 4], &[4, 2]],
        &[true, false],
    );
    check(
        "matmul-shared-right",
        mm,
        &[&[3, 4], &[4, 2]],
        &[false, true],
    );
    check("matmul-batch", mm, &[&[2, 3, 4], &[4, 2]], &[]);
    check("matmul-vector", mm, &[&[4], &[4, 3]], &[]);
}

#[test]
fn composition_with_differentiation() {
    let f = |x: &Array| Ok((x.sin() * x).sum(..));
    let xs = Array::from_slice(&[0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6], [3, 2]);
    // Per-example gradients.
    let per_example =
        vmap_flat(move |xs| Ok(vec![grad(f)(&xs[0])?]), &[], &[])(std::slice::from_ref(&xs))
            .unwrap()
            .remove(0);
    for i in 0..3 {
        assert_close(
            "vmap_flat(grad)",
            &take(&per_example, 0, i),
            &grad(f)(&take(&xs, 0, i)).unwrap(),
        );
    }
    // The gradient of a batched sum is the stack of per-example gradients.
    let batched_sum = grad(move |xs: &Array| {
        let ys = vmap_flat(move |v| Ok(vec![f(&v[0])?]), &[], &[])(std::slice::from_ref(xs))?;
        Ok(ys[0].sum(..))
    });
    assert_close("grad(vmap)", &batched_sum(&xs).unwrap(), &per_example);
    // Forward mode through vmap.
    let t = Array::from_slice(&[1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2]);
    let (_, jt) = jvp_flat(
        |v| vmap_flat(|x| Ok(vec![x[0].sin()]), &[], &[])(v),
        std::slice::from_ref(&xs),
        std::slice::from_ref(&t),
    )
    .unwrap();
    assert_close("jvp_flat(vmap)", &jt[0], &(xs.cos() * &t));
    // Slice gradients (SliceUpdate) batch too.
    let g = vmap_flat(
        |v| {
            Ok(vec![grad(|x: &Array| Ok(x.i((..).step(-2)).sum(..)))(
                &v[0],
            )?])
        },
        &[],
        &[],
    )(&[Array::from_slice(&[1.0f32; 6], [2, 3])])
    .unwrap();
    assert_eq!(
        g[0].to_vec::<f32>().unwrap(),
        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn nested_vmap() {
    let outer = vmap_flat(
        |xs| vmap_flat(|ys| Ok(vec![ys[0].multiply(&ys[1])]), &[Some(0), None], &[])(xs),
        &[Some(1), Some(0)],
        &[],
    );
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]); // inner batch 3, outer 2
    let b = array![10.0, 100.0];
    let out = outer(&[a, b]).unwrap().remove(0);
    assert_eq!(out.shape(), [2, 3]);
    assert_eq!(
        out.to_vec::<f32>().unwrap(),
        [10.0, 30.0, 50.0, 200.0, 400.0, 600.0]
    );
}

#[test]
fn mlx_edge_cases_and_errors() {
    let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    // A constant output is broadcast along the batch axis.
    let c =
        vmap_flat(|_| Ok(vec![Array::from(1.0f32)]), &[], &[])(std::slice::from_ref(&x)).unwrap();
    assert_eq!(c[0].to_vec::<f32>().unwrap(), [1.0, 1.0]);
    // Negative axes and output axes.
    let s = vmap_flat(|v| Ok(vec![v[0].sum(..)]), &[Some(-1)], &[])(std::slice::from_ref(&x));
    assert_eq!(s.unwrap()[0].to_vec::<f32>().unwrap(), [5.0, 7.0, 9.0]);
    let t = vmap_flat(|v| Ok(vec![&v[0] * 2]), &[], &[1])(std::slice::from_ref(&x)).unwrap();
    assert_eq!(t[0].shape(), [3, 2]);
    // An empty batch.
    let empty = Array::from_slice::<f32>(&[], [0, 3]);
    let e = vmap_flat(|v| Ok(vec![&v[0] * 2]), &[], &[])(&[empty]).unwrap();
    assert_eq!(e[0].shape(), [0, 3]);
    // Errors.
    let add2 = |v: &[Array]| Ok(vec![v[0].add(&v[1])]);
    let ones3 = ops::ones([3, 3], DType::F32);
    let invalid = |r: Result<Vec<Array>>| matches!(r, Err(Error::InvalidArgument(_)));
    assert!(invalid(vmap_flat(add2, &[None, None], &[])(&[
        x.clone(),
        x.clone()
    ])));
    assert!(invalid(vmap_flat(add2, &[], &[])(&[x.clone(), ones3])));
    assert!(invalid(vmap_flat(add2, &[Some(2), None], &[])(&[
        x.clone(),
        x.clone()
    ])));
    assert!(invalid(vmap_flat(add2, &[Some(0)], &[])(&[
        x.clone(),
        x.clone()
    ])));
    assert!(invalid(vmap_flat(add2, &[], &[0, 0])(&[
        x.clone(),
        x.clone()
    ])));
    let evaluates = |v: &[Array]| {
        let y = &v[0] * 2;
        y.eval()?;
        Ok(vec![y])
    };
    let err = vmap_flat(evaluates, &[], &[])(std::slice::from_ref(&x)).unwrap_err();
    assert!(err.to_string().contains("inside vmap"), "{err}");
    let failing =
        |_: &[Array]| -> Result<Vec<Array>> { Err(Error::InvalidArgument("user".into())) };
    assert!(
        matches!(vmap_flat(failing, &[], &[])(&[x]), Err(Error::InvalidArgument(m)) if m == "user")
    );
}

#[cfg(target_os = "macos")]
#[test]
fn vmap_on_the_gpu_matches_the_cpu() {
    if cfg!(miri) {
        return;
    }
    let run = |device: Device| {
        with_stream(device, || {
            let xs = ops::arange(0.0, 24.0, 1.0).reshape([2, 3, 4]);
            let w = ops::arange(0.0, 8.0, 0.5).reshape([4, 4]);
            let f = vmap_flat(
                |v| Ok(vec![v[0].matmul(&v[1]).tanh().sum(0)]),
                &[Some(0), None],
                &[],
            );
            f(&[xs, w]).and_then(|o| o[0].to_vec::<f32>())
        })
    };
    let cpu = run(Device::cpu()).unwrap();
    match run(Device::gpu(0)) {
        Ok(gpu) => {
            for (c, g) in cpu.iter().zip(&gpu) {
                assert!(
                    (c - g).abs() <= 1e-5 * c.abs().max(1.0),
                    "cpu {c} vs gpu {g}"
                );
            }
        }
        Err(e) => eprintln!("skipped: no Metal device ({e})"),
    }
}

/// The tree `vmap` builder.
mod builder {
    use super::common;
    use nuvec_core::*;

    fn vals(a: &Array) -> Vec<f32> {
        a.to_vec::<f32>().unwrap()
    }

    #[test]
    fn in_axes_out_axes_and_defaults() {
        common::on_cpu();
        let m = array![[1.0, 0.0], [0.0, 2.0]];
        let vs = array![[1.0, 1.0], [3.0, 4.0], [-1.0, 0.5]];
        let mv = vmap(|(m, v): &(Array, Array)| m.matmul(v))
            .in_axes((None, 0))
            .out_axes(0);
        let got = mv.call(&(m.clone(), vs.clone())).unwrap();
        let rows: Vec<f32> = (0..3).flat_map(|i| vals(&m.matmul(&vs.i(i)))).collect();
        assert_eq!(vals(&got), rows, "equals the per-vector loop");
        // Defaults: axis 0 in and out; a single axis applies to every argument.
        let add = vmap(|(a, b): &(Array, Array)| a.add(b));
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(
            vals(&add.call(&(a.clone(), a.clone())).unwrap()),
            [2.0, 4.0, 6.0, 8.0]
        );
        let cols = vmap(|(a, b): &(Array, Array)| a.add(b))
            .in_axes(1)
            .out_axes(1);
        assert_eq!(
            vals(&cols.call(&(a.clone(), a.t())).unwrap()),
            [2.0, 5.0, 5.0, 8.0]
        );
        // A tuple of outputs with one axis each.
        let two = vmap(|x: &Array| (x.multiply(2.0), x.sum(..))).out_axes((1, 0));
        let (doubled, sums) = two.call(&a).unwrap();
        assert_eq!(doubled.shape(), [2, 2]);
        assert_eq!(vals(&doubled), [2.0, 6.0, 4.0, 8.0]);
        assert_eq!(vals(&sums), [3.0, 7.0]);
    }

    #[test]
    fn invalid_axes_name_the_argument() {
        common::on_cpu();
        let m = array![[1.0, 0.0], [0.0, 2.0]];
        let f = vmap(|(m, v): &(Array, Array)| m.matmul(v)).in_axes((None, 0, 1));
        let err = f.call(&(m.clone(), m.clone())).unwrap_err().to_string();
        assert!(err.contains("3 input axes for 2"), "{err}");
        let f = vmap(|(m, v): &(Array, Array)| m.matmul(v)).in_axes((None, 5));
        let err = f.call(&(m.clone(), m.clone())).unwrap_err().to_string();
        assert!(err.contains("argument \"1\""), "{err}");
    }

    #[test]
    fn into_fn_composes_with_grad() {
        common::on_cpu();
        let batched = vmap(|(m, v): &(Array, Array)| m.matmul(v).tanh())
            .in_axes((None, 0))
            .into_fn();
        let m = array![[1.0, 0.5], [0.0, 2.0]];
        let vs = array![[1.0, 1.0], [3.0, 4.0]];
        let direct = vmap(|(m, v): &(Array, Array)| m.matmul(v).tanh()).in_axes((None, 0));
        assert_eq!(
            vals(&batched(&(m.clone(), vs.clone())).unwrap()),
            vals(&direct.call(&(m.clone(), vs.clone())).unwrap())
        );
        let (gm, _) =
            grad(|p: &(Array, Array)| Ok(batched(p)?.sum(..)))(&(m.clone(), vs.clone())).unwrap();
        let loop_grad = grad(|m: &Array| {
            let mut total = Array::from(0.0f32);
            for i in 0..2 {
                total = total.add(m.matmul(&vs.i(i)).tanh().sum(..));
            }
            total
        })(&m)
        .unwrap();
        let (a, b) = (vals(&gm), vals(&loop_grad));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }
}
