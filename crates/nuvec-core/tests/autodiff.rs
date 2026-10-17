//! Automatic differentiation: finite-difference gradient checks and vjp/jvp consistency for
//! every primitive, composition, `stop_gradient`, MLX edge cases and errors.

mod common;

use common::{Rng, random_layout};
use nuvec_core::*;

type F = fn(&[Array]) -> Result<Array>;

/// Uniform in `[0, 1)`.
fn unit(rng: &mut Rng) -> f64 {
    (rng.f64() + 10.0) / 20.0
}

fn values(rng: &mut Rng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n).map(|_| lo + (hi - lo) * unit(rng)).collect()
}

fn vec64(a: &Array) -> Vec<f64> {
    a.to_vec::<f64>().unwrap()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Checks `f` at random float64 inputs of `shapes` with values in `[lo, hi]`:
/// - `vjp` against central finite differences of `⟨c, f(x)⟩` for a random cotangent `c`;
/// - `⟨c, jvp(t)⟩ = Σᵢ ⟨vjpᵢ(c), tᵢ⟩` for random tangents `t`.
///
/// The primals passed to `vjp`/`jvp` have random (non-contiguous) layouts.
fn check(name: &str, f: F, shapes: &[&[usize]], lo: f64, hi: f64) {
    common::on_cpu();
    let mut rng = Rng::new(name.bytes().map(u64::from).sum());
    let data: Vec<Vec<f64>> = shapes
        .iter()
        .map(|s| values(&mut rng, s.iter().product(), lo, hi))
        .collect();
    let build = |data: &[Vec<f64>]| -> Vec<Array> {
        data.iter()
            .zip(shapes)
            .map(|(d, s)| Array::from_slice(d, s).unwrap())
            .collect()
    };
    let primals: Vec<Array> = data
        .iter()
        .zip(shapes)
        .map(|(d, s)| random_layout(&mut rng, s, d.clone()).0)
        .collect();
    let out = f(&build(&data)).unwrap();
    let c = values(&mut rng, out.size(), -1.0, 1.0);
    let cot = Array::from_slice(&c, out.shape()).unwrap();
    let (_, grads) = vjp(|xs| Ok(vec![f(xs)?]), &primals, &[cot]).unwrap();

    // Finite differences.
    let h = 1e-6;
    for (i, g) in grads.iter().enumerate() {
        assert_eq!(g.shape(), shapes[i], "{name}: gradient {i} shape");
        let g = vec64(g);
        for j in 0..data[i].len() {
            let at = |delta: f64| {
                let mut d = data.clone();
                d[i][j] += delta;
                dot(&c, &vec64(&f(&build(&d)).unwrap()))
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            let tol = 1e-6 * numeric.abs().max(1.0);
            assert!(
                (g[j] - numeric).abs() <= tol,
                "{name}: d/dx{i}[{j}] = {} vs finite difference {numeric}",
                g[j]
            );
        }
    }

    // vjp/jvp consistency.
    let t: Vec<Vec<f64>> = shapes
        .iter()
        .map(|s| values(&mut rng, s.iter().product(), -1.0, 1.0))
        .collect();
    let (_, jt) = jvp(|xs| Ok(vec![f(xs)?]), &primals, &build(&t)).unwrap();
    let lhs = dot(&c, &vec64(&jt[0]));
    let rhs: f64 = grads.iter().zip(&t).map(|(g, t)| dot(&vec64(g), t)).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(1.0),
        "{name}: <c, jvp(t)> = {lhs} vs <vjp(c), t> = {rhs}"
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn unary_gradients() {
    let cases: [(&str, F, f64, f64); 9] = [
        ("negative", |x| negative(&x[0]), -2.0, 2.0),
        ("abs", |x| abs(&x[0]), 0.5, 2.0),
        ("exp", |x| exp(&x[0]), -2.0, 2.0),
        ("log", |x| log(&x[0]), 0.5, 3.0),
        ("sqrt", |x| sqrt(&x[0]), 0.5, 3.0),
        ("sin", |x| sin(&x[0]), -3.0, 3.0),
        ("cos", |x| cos(&x[0]), -3.0, 3.0),
        ("tanh", |x| tanh(&x[0]), -2.0, 2.0),
        ("sigmoid", |x| sigmoid(&x[0]), -3.0, 3.0),
    ];
    for (name, f, lo, hi) in cases {
        check(name, f, &[&[2, 3]], lo, hi);
    }
    check("abs-negative", |x| abs(&x[0]), &[&[4]], -2.0, -0.5);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn binary_gradients_with_broadcasting() {
    let cases: [(&str, F); 6] = [
        ("add", |x| add(&x[0], &x[1])),
        ("subtract", |x| subtract(&x[0], &x[1])),
        ("multiply", |x| multiply(&x[0], &x[1])),
        ("divide", |x| divide(&x[0], &x[1])),
        ("maximum", |x| maximum(&x[0], &x[1])),
        ("minimum", |x| minimum(&x[0], &x[1])),
    ];
    for (name, f) in cases {
        check(name, f, &[&[2, 3], &[3]], 0.5, 2.0);
        check(&format!("{name}-outer"), f, &[&[3, 1], &[1, 4]], 0.5, 2.0);
    }
    check(
        "scalar-operands",
        |x| subtract(multiply(2.0, &x[0])?, divide(&x[0], 3)?),
        &[&[3]],
        -1.0,
        1.0,
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn where_gradients() {
    check(
        "where",
        |x| where_(greater(&x[0], 0.0)?, &x[1], &x[2]),
        &[&[2, 3], &[2, 3], &[3]],
        -1.0,
        1.0,
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn view_gradients() {
    let cases: [(&str, F, &[usize]); 7] = [
        ("reshape", |x| x[0].reshape(&[3, 4]), &[2, 6]),
        ("transpose", |x| transpose(&x[0], &[2, 0, 1]), &[2, 3, 4]),
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
        ("expand_dims", |x| expand_dims(&x[0], 1), &[2, 3]),
        ("squeeze", |x| squeeze(&x[0], &[]), &[1, 3, 1]),
        ("t-then-reshape", |x| x[0].t()?.reshape(&[-1]), &[3, 4]),
    ];
    for (name, f, shape) in cases {
        check(name, f, &[shape], -1.0, 1.0);
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn reduction_gradients() {
    let cases: [(&str, F); 8] = [
        ("sum-all", |x| sum(&x[0], &[], false)),
        ("sum-axis", |x| sum(&x[0], &[1], false)),
        ("sum-keep", |x| sum(&x[0], &[0, 2], true)),
        ("prod", |x| prod(&x[0], &[1], false)),
        ("max", |x| max(&x[0], &[2], false)),
        ("min", |x| min(&x[0], &[0, 1], true)),
        ("mean", |x| mean(&x[0], &[1, 2], false)),
        ("prod-all", |x| prod(&x[0], &[], false)),
    ];
    for (name, f) in cases {
        check(name, f, &[&[2, 3, 4]], 0.5, 1.5);
    }
    // prod is a polynomial: finite differences are valid at zeros too.
    check(
        "prod-with-zeros",
        |x| prod(&multiply(&x[0], &x[1])?, &[1], false),
        &[&[3, 4], &[4]],
        -1.0,
        1.0,
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn matmul_gradients() {
    check(
        "matmul",
        |x| matmul(&x[0], &x[1]),
        &[&[3, 4], &[4, 2]],
        -1.0,
        1.0,
    );
    check(
        "matmul-batch-broadcast",
        |x| matmul(&x[0], &x[1]),
        &[&[2, 1, 3, 4], &[3, 4, 2]],
        -1.0,
        1.0,
    );
    check(
        "matmul-vector",
        |x| matmul(&x[0], &x[1]),
        &[&[4], &[4, 3]],
        -1.0,
        1.0,
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn composite_gradients() {
    // A two-layer MLP with a mean-squared-error loss.
    check(
        "mlp",
        |x| {
            let h = tanh(&add(matmul(&x[0], &x[1])?, &x[2])?)?;
            let y = matmul(&h, &x[3])?;
            let r = subtract(&y, 0.5)?;
            mean(&multiply(&r, &r)?, &[], false)
        },
        &[&[4, 3], &[3, 5], &[5], &[5, 2]],
        -1.0,
        1.0,
    );
    check(
        "astype-roundtrip",
        |x| multiply(&x[0].astype(DType::F64)?, &x[0]),
        &[&[3]],
        -1.0,
        1.0,
    );
}

#[test]
fn value_and_grad_selects_arguments() {
    let loss = |xs: &[Array]| sum(&multiply(&xs[0], &xs[1])?, &[], false);
    let a = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let b = Array::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
    let (value, grads) = value_and_grad(loss, &[1, 0])(&[a, b]).unwrap();
    assert_eq!(value.item::<f32>().unwrap(), 11.0);
    assert_eq!(grads[0].to_vec::<f32>().unwrap(), [1.0, 2.0]); // d/db = a
    assert_eq!(grads[1].to_vec::<f32>().unwrap(), [3.0, 4.0]); // d/da = b
}

#[test]
fn eval_inside_the_function_keeps_the_gradient() {
    let g = grad(|x: &Array| {
        let y = multiply(x, x)?;
        y.eval()?;
        let z = multiply(&y, x)?;
        eval(std::slice::from_ref(&z))?;
        sum(&z, &[], false)
    });
    let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    assert_eq!(g(&x).unwrap().to_vec::<f32>().unwrap(), [3.0, 12.0]);
}

#[test]
fn higher_order_derivatives() {
    let x = Array::from_slice(&[0.3f64, 1.2], &[2]).unwrap();
    let sum_of = |f: fn(&Array) -> Result<Array>| move |x: &Array| sum(&f(x)?, &[], false);
    let d2 = grad(sum_of(|x| {
        let g = grad(|x: &Array| sum(&sin(x)?, &[], false))(x)?;
        Ok(g)
    }));
    let expect: Vec<f64> = [0.3f64, 1.2].iter().map(|v| -v.sin()).collect();
    for (g, e) in vec64(&d2(&x).unwrap()).iter().zip(&expect) {
        assert!((g - e).abs() < 1e-12);
    }
    // Third derivative of x^4 is 24x.
    let d3 = grad(grad(grad(|x: &Array| {
        let x2 = multiply(x, x)?;
        multiply(&x2, &x2)
    })));
    let at = Array::scalar(0.5f64);
    assert!((d3(&at).unwrap().item::<f64>().unwrap() - 12.0).abs() < 1e-12);
    // Hessian-vector product: jvp of grad. f = sum(x^3) → H = diag(6x).
    let g = grad(|x: &Array| sum(&multiply(&multiply(x, x)?, x)?, &[], false));
    let t = Array::from_slice(&[1.0f64, -2.0], &[2]).unwrap();
    let (_, hv) = jvp(|xs| Ok(vec![g(&xs[0])?]), &[x], &[t]).unwrap();
    for (h, e) in vec64(&hv[0]).iter().zip([6.0 * 0.3, -2.0 * 6.0 * 1.2]) {
        assert!((h - e).abs() < 1e-12);
    }
}

#[test]
fn stop_gradient_blocks_the_gradient() {
    let g = grad(|x: &Array| sum(&multiply(x, &stop_gradient(x)?)?, &[], false));
    let x = Array::from_slice(&[2.0f32, -1.0], &[2]).unwrap();
    assert_eq!(g(&x).unwrap().to_vec::<f32>().unwrap(), [2.0, -1.0]);
    assert_eq!(
        stop_gradient(&x).unwrap().to_vec::<f32>().unwrap(),
        [2.0, -1.0]
    );
    let (_, t) = jvp(
        |xs| Ok(vec![stop_gradient(&xs[0])?]),
        std::slice::from_ref(&x),
        std::slice::from_ref(&x),
    )
    .unwrap();
    assert_eq!(t[0].to_vec::<f32>().unwrap(), [0.0, 0.0]);
}

#[test]
fn mlx_edge_cases() {
    let one = Array::scalar(1.0f32);
    let grad_of = |f: fn(&Array) -> Result<Array>, x: &[f32]| {
        let x = Array::from_slice(x, &[x.len()]).unwrap();
        let (_, g) = vjp(|xs| Ok(vec![f(&xs[0])?]), &[x], std::slice::from_ref(&one)).unwrap();
        g[0].to_vec::<f32>().unwrap()
    };
    let p = |x: &Array| prod(x, &[], false);
    assert_eq!(grad_of(p, &[1.0, 0.0, 3.0, 2.0]), [0.0, 6.0, 0.0, 0.0]);
    assert_eq!(grad_of(p, &[1.0, 0.0, 3.0, 0.0, 2.0]), [0.0; 5]);
    let m = |x: &Array| max(x, &[], false);
    assert_eq!(grad_of(m, &[2.0, 2.0, 1.0]), [0.5, 0.5, 0.0]);
    let a = |x: &Array| sum(&abs(x)?, &[], false);
    assert_eq!(grad_of(a, &[0.0, -2.0]), [0.0, -1.0]);
    // Element-wise maximum sends ties to the second operand.
    let ab = [Array::scalar(1.0f32), Array::scalar(1.0f32)];
    let (_, g) = vjp(
        |x| Ok(vec![maximum(&x[0], &x[1])?]),
        &ab,
        std::slice::from_ref(&one),
    )
    .unwrap();
    assert_eq!(
        (g[0].item::<f32>().unwrap(), g[1].item::<f32>().unwrap()),
        (0.0, 1.0)
    );
    // Gradients pass through integer arithmetic and casts, as in MLX.
    let g = grad(|x: &Array| sum(&multiply(x, x)?, &[], false));
    let ints = Array::from_slice(&[1i32, 2], &[2]).unwrap();
    assert_eq!(g(&ints).unwrap().to_vec::<i32>().unwrap(), [2, 4]);
    let cast = grad(|x: &Array| sum(&x.astype(DType::I32)?.astype(DType::F32)?, &[], false));
    assert_eq!(
        cast(&Array::scalar(1.5f32)).unwrap().item::<f32>().unwrap(),
        1.0
    );
}

#[test]
fn non_differentiable_and_independent_inputs_get_zero() {
    let x = Array::from_slice(&[1.0f32, 3.0, 2.0], &[3]).unwrap();
    let y = Array::from_slice(&[5.0f32, 6.0, 7.0], &[3]).unwrap();
    let ones = Array::from_slice(&[1.0f32; 3], &[3]).unwrap();
    let (_, g) = vjp(
        |xs| Ok(vec![greater(&xs[0], 2.0)?.astype(DType::F32)?]),
        &[x.clone(), y.clone()],
        std::slice::from_ref(&ones),
    )
    .unwrap();
    assert_eq!(g[0].to_vec::<f32>().unwrap(), [0.0; 3]);
    assert_eq!(
        g[1].to_vec::<f32>().unwrap(),
        [0.0; 3],
        "independent primal"
    );
    let (_, g) = vjp(
        |xs| Ok(vec![argmax(&xs[0], None, false)?.astype(DType::F32)?]),
        &[x],
        &[Array::scalar(1.0f32)],
    )
    .unwrap();
    assert_eq!(g[0].to_vec::<f32>().unwrap(), [0.0; 3]);
}

#[test]
fn several_outputs() {
    let x = Array::from_slice(&[0.5f32, 1.0], &[2]).unwrap();
    let c = Array::from_slice(&[1.0f32, 1.0], &[2]).unwrap();
    let (outs, g) = vjp(
        |xs| Ok(vec![sin(&xs[0])?, multiply(&xs[0], &xs[0])?]),
        &[x],
        &[c.clone(), c],
    )
    .unwrap();
    assert_eq!(outs.len(), 2);
    let expect: Vec<f32> = [0.5f32, 1.0].iter().map(|v| v.cos() + 2.0 * v).collect();
    for (g, e) in g[0].to_vec::<f32>().unwrap().iter().zip(expect) {
        assert!((g - e).abs() < 1e-6);
    }
}

#[test]
fn errors() {
    let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let not_scalar = grad(|x: &Array| multiply(x, x));
    assert!(matches!(not_scalar(&x), Err(Error::InvalidArgument(m)) if m.contains("scalar")));
    let f = |xs: &[Array]| sum(&xs[0], &[], false);
    for argnums in [&[][..], &[1], &[0, 0]] {
        assert!(matches!(
            value_and_grad(f, argnums)(std::slice::from_ref(&x)),
            Err(Error::InvalidArgument(_))
        ));
    }
    let id = |xs: &[Array]| Ok(vec![xs[0].clone()]);
    assert!(vjp(id, std::slice::from_ref(&x), &[Array::scalar(1.0f32)]).is_err());
    assert!(vjp(id, std::slice::from_ref(&x), &[]).is_err());
    assert!(jvp(id, std::slice::from_ref(&x), &[]).is_err());
    let failing =
        |_: &[Array]| -> Result<Vec<Array>> { Err(Error::InvalidArgument("user".into())) };
    assert!(matches!(vjp(failing, &[], &[]), Err(Error::InvalidArgument(m)) if m == "user"));
}

#[test]
fn deep_chains_do_not_overflow_the_stack() {
    common::on_cpu();
    let n = if cfg!(miri) { 50 } else { 1_000_000 };
    let one = Array::scalar(1.0f32);
    let g = grad(move |x: &Array| {
        let mut y = x.clone();
        for _ in 0..n {
            y = &y * &one;
        }
        sum(&y, &[], false)
    });
    let x = Array::scalar(2.0f32);
    assert_eq!(g(&x).unwrap().item::<f32>().unwrap(), 1.0);
}

#[cfg(target_os = "macos")]
#[test]
fn gradients_on_the_gpu_match_the_cpu() {
    if cfg!(miri) {
        return;
    }
    let loss = |xs: &[Array]| {
        let s = slice(&xs[0], &[SliceArg::new(None, None, 2), SliceArg::full()])?;
        let y = sigmoid(&matmul(&s, &xs[1])?)?;
        sum(&multiply(&y, &y)?, &[], false)
    };
    let mut rng = Rng::new(7);
    let a: Vec<f32> = (0..20).map(|_| unit(&mut rng) as f32 - 0.5).collect();
    let b: Vec<f32> = (0..12).map(|_| unit(&mut rng) as f32 - 0.5).collect();
    let run = |device: Device| {
        let previous = default_stream();
        set_default_stream(Stream::default_for(device));
        let args = [
            Array::from_slice(&a, &[5, 4]).unwrap(),
            Array::from_slice(&b, &[4, 3]).unwrap(),
        ];
        let r = value_and_grad(loss, &[0, 1])(&args).and_then(|(_, g)| {
            g.iter()
                .map(|g| g.to_vec::<f32>())
                .collect::<Result<Vec<_>>>()
        });
        set_default_stream(previous);
        r
    };
    let cpu = run(Device::cpu()).unwrap();
    match run(Device::gpu(0)) {
        Ok(gpu) => {
            for (c, g) in cpu.iter().flatten().zip(gpu.iter().flatten()) {
                assert!(
                    (c - g).abs() <= 1e-5 * c.abs().max(1.0),
                    "cpu {c} vs gpu {g}"
                );
            }
        }
        Err(e) => eprintln!("skipped: no Metal device ({e})"),
    }
}

// ── Complex numbers ─────────────────────────────────────────────────────────

fn complex_values(a: &Array) -> Vec<Complex64> {
    match a.dtype() {
        DType::Complex64 => a.to_vec().unwrap(),
        _ => a
            .astype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect(),
    }
}

/// `Re Σ conj(c)·y`, the pairing under which MLX's complex cotangents are defined.
fn pair(c: &[Complex64], y: &[Complex64]) -> f32 {
    c.iter()
        .zip(y)
        .map(|(c, y)| c.re * y.re + c.im * y.im)
        .sum()
}

/// [`check`] for complex inputs (real where `real[k]`), in complex64 precision:
/// - `vjp(c)` against `∂L/∂x + i·∂L/∂y` (MLX's convention) for `L = Re Σ conj(c)·f(z)`,
///   by central differences in the real and the imaginary direction;
/// - `Re⟨c, jvp(t)⟩ = Re Σ⟨vjpᵢ(c), tᵢ⟩`.
fn check_complex(name: &str, f: F, shapes: &[&[usize]], real: &[bool]) {
    common::on_cpu();
    let mut rng = Rng::new(name.bytes().map(u64::from).sum());
    let is_real = |k: usize| real.get(k).copied().unwrap_or(false);
    let random = |rng: &mut Rng, n: usize, real: bool| -> Vec<Complex64> {
        (0..n)
            .map(|_| {
                let re = (unit(rng) * 2.0 - 1.0) as f32;
                let im = if real {
                    0.0
                } else {
                    (unit(rng) * 2.0 - 1.0) as f32
                };
                Complex64::new(re, im)
            })
            .collect()
    };
    let array = |d: &[Complex64], s: &[usize], real: bool| -> Array {
        if real {
            Array::from_slice(&d.iter().map(|z| z.re).collect::<Vec<f32>>(), s).unwrap()
        } else {
            Array::from_slice(d, s).unwrap()
        }
    };
    let data: Vec<Vec<Complex64>> = shapes
        .iter()
        .enumerate()
        .map(|(k, s)| random(&mut rng, s.iter().product(), is_real(k)))
        .collect();
    let build = |data: &[Vec<Complex64>]| -> Vec<Array> {
        (0..shapes.len())
            .map(|k| array(&data[k], shapes[k], is_real(k)))
            .collect()
    };
    let primals: Vec<Array> = (0..shapes.len())
        .map(|k| {
            if is_real(k) {
                let v: Vec<f32> = data[k].iter().map(|z| z.re).collect();
                random_layout(&mut rng, shapes[k], v).0
            } else {
                random_layout(&mut rng, shapes[k], data[k].clone()).0
            }
        })
        .collect();
    let out = f(&build(&data)).unwrap();
    let out_real = out.dtype() != DType::Complex64;
    let c = random(&mut rng, out.size(), out_real);
    let cot = array(&c, out.shape(), out_real);
    let (_, grads) = vjp(|xs| Ok(vec![f(xs)?]), &primals, &[cot]).unwrap();

    let h = 1e-2f32;
    let grads: Vec<Vec<Complex64>> = grads
        .iter()
        .enumerate()
        .map(|(k, g)| {
            assert_eq!(g.shape(), shapes[k], "{name}: gradient {k} shape");
            complex_values(g)
        })
        .collect();
    for (k, g) in grads.iter().enumerate() {
        for j in 0..data[k].len() {
            let at = |re: f32, im: f32| {
                let mut d = data.clone();
                d[k][j] = Complex64::new(d[k][j].re + re, d[k][j].im + im);
                pair(&c, &complex_values(&f(&build(&d)).unwrap()))
            };
            let d_re = (at(h, 0.0) - at(-h, 0.0)) / (2.0 * h);
            let d_im = if is_real(k) {
                0.0
            } else {
                (at(0.0, h) - at(0.0, -h)) / (2.0 * h)
            };
            let tol = 1e-2 * d_re.hypot(d_im).max(1.0);
            assert!(
                (g[j].re - d_re).abs() <= tol && (g[j].im - d_im).abs() <= tol,
                "{name}: d/dz{k}[{j}] = {:?} vs finite differences ({d_re}, {d_im})",
                g[j]
            );
        }
    }

    let t: Vec<Vec<Complex64>> = shapes
        .iter()
        .enumerate()
        .map(|(k, s)| random(&mut rng, s.iter().product(), is_real(k)))
        .collect();
    let (_, jt) = jvp(|xs| Ok(vec![f(xs)?]), &primals, &build(&t)).unwrap();
    let lhs = pair(&c, &complex_values(&jt[0]));
    let rhs: f32 = grads.iter().zip(&t).map(|(g, t)| pair(g, t)).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
        "{name}: Re<c, jvp(t)> = {lhs} vs Re<vjp(c), t> = {rhs}"
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn complex_gradients() {
    let unary: [(&str, F); 3] = [
        ("c-negative", |x| negative(&x[0])),
        ("c-conjugate", |x| conjugate(&x[0])),
        ("c-abs", |x| abs(&x[0])),
    ];
    for (name, f) in unary {
        check_complex(name, f, &[&[2, 3]], &[]);
    }
    let binary: [(&str, F); 4] = [
        ("c-add", |x| add(&x[0], &x[1])),
        ("c-subtract", |x| subtract(&x[0], &x[1])),
        ("c-multiply", |x| multiply(&x[0], &x[1])),
        ("c-divide", |x| {
            divide(&x[0], &add(&x[1], Complex64::new(2.0, 0.5))?)
        }),
    ];
    for (name, f) in binary {
        check_complex(name, f, &[&[2, 3], &[3]], &[]);
        check_complex(
            &format!("{name}-real-rhs"),
            f,
            &[&[2, 3], &[3]],
            &[false, true],
        );
    }
    check_complex(
        "c-where",
        |x| where_(greater(&x[0], 0.0)?, &x[1], &x[2]),
        &[&[2, 3], &[2, 3], &[3]],
        &[true],
    );
    check_complex("c-astype-real", |x| x[0].astype(DType::F32), &[&[4]], &[]);
    check_complex(
        "c-astype-complex",
        |x| x[0].astype(DType::Complex64),
        &[&[4]],
        &[true],
    );
    let views: [(&str, F, &[usize]); 4] = [
        ("c-reshape", |x| x[0].reshape(&[3, 2]), &[2, 3]),
        ("c-transpose", |x| x[0].t(), &[2, 3]),
        (
            "c-slice",
            |x| slice(&x[0], &[SliceArg::new(None, None, -2)]),
            &[5],
        ),
        ("c-broadcast", |x| broadcast_to(&x[0], &[2, 3]), &[3]),
    ];
    for (name, f, shape) in views {
        check_complex(name, f, &[shape], &[]);
    }
    check_complex("c-sum", |x| sum(&x[0], &[1], false), &[&[2, 3]], &[]);
    check_complex("c-prod", |x| prod(&x[0], &[1], false), &[&[2, 3]], &[]);
    check_complex(
        "c-matmul",
        |x| matmul(&x[0], &x[1]),
        &[&[2, 3], &[3, 2]],
        &[],
    );
    check_complex(
        "c-composite",
        |x| abs(&add(multiply(&x[0], &x[1])?, conjugate(&x[2])?)?),
        &[&[3], &[3], &[3]],
        &[],
    );
    check_complex(
        "real-through-complex",
        |x| {
            let z = multiply(x[0].astype(DType::Complex64)?, &x[1])?;
            sum(&abs(&z)?, &[], false)
        },
        &[&[3], &[3]],
        &[true],
    );
}

#[test]
fn complex_abs_at_zero_has_zero_derivative() {
    let z = Array::from_slice(&[Complex64::new(0.0, 0.0), Complex64::new(3.0, 4.0)], &[2]).unwrap();
    let ones = Array::from_slice(&[1.0f32, 1.0], &[2]).unwrap();
    let (_, g) = vjp(|x| Ok(vec![abs(&x[0])?]), std::slice::from_ref(&z), &[ones]).unwrap();
    let g = g[0].to_vec::<Complex64>().unwrap();
    assert_eq!((g[0].re, g[0].im), (0.0, 0.0));
    assert!((g[1].re - 0.6).abs() < 1e-6 && (g[1].im - 0.8).abs() < 1e-6);
    let t = Array::from_slice(&[Complex64::new(1.0, 1.0); 2], &[2]).unwrap();
    let (_, jt) = jvp(|x| Ok(vec![abs(&x[0])?]), std::slice::from_ref(&z), &[t]).unwrap();
    let jt = jt[0].to_vec::<f32>().unwrap();
    assert_eq!(jt[0], 0.0);
    assert!((jt[1] - 1.4).abs() < 1e-6, "Re((3-4i)(1+i)) / 5 = 7/5");
}

#[test]
fn conjugate_values_batching_and_devices() {
    let z = Array::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, -0.5)],
        &[2],
    )
    .unwrap();
    let expect = [Complex64::new(1.0, -2.0), Complex64::new(-3.0, 0.5)];
    assert_eq!(
        conjugate(&z).unwrap().to_vec::<Complex64>().unwrap(),
        expect
    );
    let real = Array::from_slice(&[1.0f32, -2.0], &[2]).unwrap();
    assert_eq!(
        conjugate(&real).unwrap().to_vec::<f32>().unwrap(),
        [1.0, -2.0]
    );
    let batched =
        vmap(|x| Ok(vec![conjugate(&x[0])?]), &[], &[])(std::slice::from_ref(&z)).unwrap();
    assert_eq!(batched[0].to_vec::<Complex64>().unwrap(), expect);
    #[cfg(target_os = "macos")]
    if !cfg!(miri) {
        let previous = default_stream();
        set_default_stream(Stream::default_for(Device::gpu(0)));
        let gpu = conjugate(&(&z * 1)).and_then(|c| c.to_vec::<Complex64>());
        set_default_stream(previous);
        match gpu {
            Ok(v) => assert_eq!(v, expect),
            Err(e) => eprintln!("skipped: no Metal device ({e})"),
        }
    }
}
