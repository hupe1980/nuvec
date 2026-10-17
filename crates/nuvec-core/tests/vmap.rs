//! `vmap`: every primitive against the per-example loop, composition with the other
//! transforms, MLX's edge cases and errors.

mod common;

use common::{Rng, random_layout};
use nuvec_core::*;

type F = fn(&[Array]) -> Result<Array>;

/// Example `i` of `a` along `axis`.
fn take(a: &Array, axis: usize, i: usize) -> Array {
    let mut args = vec![SliceArg::full(); a.ndim()];
    args[axis] = SliceArg::new(Some(i as isize), Some(i as isize + 1), 1);
    squeeze(&slice(a, &args).unwrap(), &[axis as isize]).unwrap()
}

fn assert_close(name: &str, got: &Array, expected: &Array) {
    assert_eq!(got.shape(), expected.shape(), "{name}: shape");
    assert_eq!(got.dtype(), expected.dtype(), "{name}: dtype");
    let g = got
        .astype(DType::F64)
        .and_then(|g| g.to_vec::<f64>())
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let e = expected
        .astype(DType::F64)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    for (i, (g, e)) in g.iter().zip(&e).enumerate() {
        assert!(
            g == e || (g - e).abs() <= 1e-12 * e.abs().max(1.0) || (g.is_nan() && e.is_nan()),
            "{name}: element {i}: {g} vs per-example {e}"
        );
    }
}

/// Checks `vmap(f)` against the per-example loop for every combination of batch axes:
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
                f(&xs).unwrap()
            })
            .collect();
        let out_axis = rng.below(expected[0].ndim() + 1);
        let in_axes: Vec<Option<isize>> = axes.iter().map(|a| a.map(|a| a as isize)).collect();
        let got = vmap(|xs| Ok(vec![f(xs)?]), &in_axes, &[out_axis as isize])(&args)
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
        ("negative", |x| negative(&x[0])),
        ("abs", |x| abs(&x[0])),
        ("exp", |x| exp(&x[0])),
        ("log", |x| log(&abs(&x[0])?)),
        ("sqrt", |x| sqrt(&abs(&x[0])?)),
        ("sin", |x| sin(&x[0])),
        ("cos", |x| cos(&x[0])),
        ("tanh", |x| tanh(&x[0])),
        ("sigmoid", |x| sigmoid(&x[0])),
    ];
    for (name, f) in unary {
        check(name, f, &[&[2, 3]], &[]);
    }
    let binary: [(&str, F); 8] = [
        ("add", |x| add(&x[0], &x[1])),
        ("subtract", |x| subtract(&x[0], &x[1])),
        ("multiply", |x| multiply(&x[0], &x[1])),
        ("divide", |x| divide(&x[0], &add(&abs(&x[1])?, 1.0)?)),
        ("maximum", |x| maximum(&x[0], &x[1])),
        ("minimum", |x| minimum(&x[0], &x[1])),
        ("less", |x| less(&x[0], &x[1])),
        ("equal", |x| equal(&x[0], &x[1])),
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
        |x| where_(greater(&x[0], 0.0)?, &x[1], &x[2]),
        &[&[2, 3], &[3], &[2, 1]],
        &[],
    );
    check(
        "where-shared-cond",
        |x| where_(greater(&x[0], 0.0)?, &x[1], 0.0),
        &[&[3], &[3]],
        &[true],
    );
    check("astype", |x| x[0].astype(DType::F32), &[&[2, 3]], &[]);
    check("stop_gradient", |x| stop_gradient(&x[0]), &[&[2, 3]], &[]);
    check(
        "scalar-operand",
        |x| subtract(1.0, multiply(&x[0], 2)?),
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
        ("reshape", |x| x[0].reshape(&[3, 2]), &[2, 3]),
        ("reshape-infer", |x| x[0].reshape(&[-1]), &[2, 3, 2]),
        ("transpose", |x| transpose(&x[0], &[2, 0, 1]), &[2, 3, 4]),
        ("t", |x| x[0].t(), &[2, 3]),
        (
            "slice",
            |x| {
                slice(
                    &x[0],
                    &[
                        SliceArg::new(Some(1), None, 1),
                        SliceArg::new(None, None, -2),
                    ],
                )
            },
            &[3, 5],
        ),
        ("broadcast_to", |x| broadcast_to(&x[0], &[2, 3, 4]), &[3, 1]),
        ("expand_dims", |x| expand_dims(&x[0], -1), &[2, 3]),
        ("squeeze", |x| squeeze(&x[0], &[]), &[1, 3, 1]),
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
        ("sum", |x| sum(&x[0], &[1], false)),
        ("sum-all", |x| sum(&x[0], &[], false)),
        ("prod", |x| prod(&x[0], &[0, 2], true)),
        ("max", |x| max(&x[0], &[-1], false)),
        ("min", |x| min(&x[0], &[], true)),
        ("mean", |x| mean(&x[0], &[0], false)),
        ("all", |x| all(&greater(&x[0], 0.0)?, &[1], false)),
        ("argmax", |x| argmax(&x[0], Some(1), false)),
        ("argmin-flat", |x| argmin(&x[0], None, false)),
    ];
    for (name, f) in cases {
        check(name, f, &[&[2, 3, 4]], &[]);
    }
    let mm: F = |x| matmul(&x[0], &x[1]);
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
    let f = |x: &Array| sum(&multiply(&sin(x)?, x)?, &[], false);
    let xs = Array::from_slice(&[0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]).unwrap();
    // Per-example gradients.
    let per_example =
        vmap(move |xs| Ok(vec![grad(f)(&xs[0])?]), &[], &[])(std::slice::from_ref(&xs))
            .unwrap()
            .remove(0);
    for i in 0..3 {
        assert_close(
            "vmap(grad)",
            &take(&per_example, 0, i),
            &grad(f)(&take(&xs, 0, i)).unwrap(),
        );
    }
    // The gradient of a batched sum is the stack of per-example gradients.
    let batched_sum = grad(move |xs: &Array| {
        let ys = vmap(move |v| Ok(vec![f(&v[0])?]), &[], &[])(std::slice::from_ref(xs))?;
        sum(&ys[0], &[], false)
    });
    assert_close("grad(vmap)", &batched_sum(&xs).unwrap(), &per_example);
    // Forward mode through vmap.
    let t = Array::from_slice(&[1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]).unwrap();
    let (_, jt) = jvp(
        |v| vmap(|x| Ok(vec![sin(&x[0])?]), &[], &[])(v),
        std::slice::from_ref(&xs),
        std::slice::from_ref(&t),
    )
    .unwrap();
    assert_close(
        "jvp(vmap)",
        &jt[0],
        &multiply(cos(&xs).unwrap(), &t).unwrap(),
    );
    // Slice gradients (SliceUpdate) batch too.
    let g = vmap(
        |v| {
            Ok(vec![grad(|x: &Array| {
                sum(&slice(x, &[SliceArg::new(None, None, -2)])?, &[], false)
            })(&v[0])?])
        },
        &[],
        &[],
    )(&[Array::from_slice(&[1.0f32; 6], &[2, 3]).unwrap()])
    .unwrap();
    assert_eq!(
        g[0].to_vec::<f32>().unwrap(),
        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn nested_vmap() {
    let outer = vmap(
        |xs| {
            vmap(
                |ys| Ok(vec![multiply(&ys[0], &ys[1])?]),
                &[Some(0), None],
                &[],
            )(xs)
        },
        &[Some(1), Some(0)],
        &[],
    );
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap(); // inner batch 3, outer 2
    let b = Array::from_slice(&[10.0f32, 100.0], &[2]).unwrap();
    let out = outer(&[a, b]).unwrap().remove(0);
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(
        out.to_vec::<f32>().unwrap(),
        [10.0, 30.0, 50.0, 200.0, 400.0, 600.0]
    );
}

#[test]
fn mlx_edge_cases_and_errors() {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    // A constant output is broadcast along the batch axis.
    let c = vmap(|_| Ok(vec![Array::scalar(1.0f32)]), &[], &[])(std::slice::from_ref(&x)).unwrap();
    assert_eq!(c[0].to_vec::<f32>().unwrap(), [1.0, 1.0]);
    // Negative axes and output axes.
    let s = vmap(|v| Ok(vec![sum(&v[0], &[], false)?]), &[Some(-1)], &[])(std::slice::from_ref(&x));
    assert_eq!(s.unwrap()[0].to_vec::<f32>().unwrap(), [5.0, 7.0, 9.0]);
    let t = vmap(|v| Ok(vec![&v[0] * 2]), &[], &[1])(std::slice::from_ref(&x)).unwrap();
    assert_eq!(t[0].shape(), &[3, 2]);
    // An empty batch.
    let empty = Array::from_slice::<f32>(&[], &[0, 3]).unwrap();
    let e = vmap(|v| Ok(vec![&v[0] * 2]), &[], &[])(&[empty]).unwrap();
    assert_eq!(e[0].shape(), &[0, 3]);
    // Errors.
    let add2 = |v: &[Array]| Ok(vec![add(&v[0], &v[1])?]);
    let ones3 = Array::from_slice(&[1.0f32; 9], &[3, 3]).unwrap();
    let invalid = |r: Result<Vec<Array>>| matches!(r, Err(Error::InvalidArgument(_)));
    assert!(invalid(vmap(add2, &[None, None], &[])(&[
        x.clone(),
        x.clone()
    ])));
    assert!(invalid(vmap(add2, &[], &[])(&[x.clone(), ones3])));
    assert!(invalid(vmap(add2, &[Some(2), None], &[])(&[
        x.clone(),
        x.clone()
    ])));
    assert!(invalid(vmap(add2, &[Some(0)], &[])(&[
        x.clone(),
        x.clone()
    ])));
    assert!(invalid(vmap(add2, &[], &[0, 0])(&[x.clone(), x.clone()])));
    let evaluates = |v: &[Array]| {
        let y = &v[0] * 2;
        y.eval()?;
        Ok(vec![y])
    };
    let err = vmap(evaluates, &[], &[])(std::slice::from_ref(&x)).unwrap_err();
    assert!(err.to_string().contains("inside vmap"), "{err}");
    let failing =
        |_: &[Array]| -> Result<Vec<Array>> { Err(Error::InvalidArgument("user".into())) };
    assert!(matches!(vmap(failing, &[], &[])(&[x]), Err(Error::InvalidArgument(m)) if m == "user"));
}

#[cfg(target_os = "macos")]
#[test]
fn vmap_on_the_gpu_matches_the_cpu() {
    if cfg!(miri) {
        return;
    }
    let run = |device: Device| {
        let previous = default_stream();
        set_default_stream(Stream::default_for(device));
        let xs = arange(0.0, 24.0, 1.0, DType::F32)
            .unwrap()
            .reshape(&[2, 3, 4])
            .unwrap();
        let w = arange(0.0, 8.0, 0.5, DType::F32)
            .unwrap()
            .reshape(&[4, 4])
            .unwrap();
        let f = vmap(
            |v| Ok(vec![sum(&tanh(&matmul(&v[0], &v[1])?)?, &[0], false)?]),
            &[Some(0), None],
            &[],
        );
        let r = f(&[xs, w]).and_then(|o| o[0].to_vec::<f32>());
        set_default_stream(previous);
        r
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
