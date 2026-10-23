//! Automatic differentiation: finite-difference gradient checks and vjp/jvp consistency for
//! every primitive, composition, `stop_gradient`, MLX edge cases and errors.

mod common;

use common::{Rng, random_layout};
use nuvec_core::ops;
use nuvec_core::*;

type F = fn(&[Array]) -> Array;

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

/// A host shape as the array's `i32` dimensions.
fn dims(shape: &[usize]) -> Vec<i32> {
    shape.iter().map(|&d| d as i32).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Checks `f` at random float64 inputs of `shapes` with values in `[lo, hi]`:
/// - `vjp` against central finite differences of `⟨c, f(x)⟩` for a random cotangent `c`;
/// - `⟨c, jvp_flat(t)⟩ = Σᵢ ⟨vjpᵢ(c), tᵢ⟩` for random tangents `t`.
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
            .map(|(d, s)| Array::from_slice(d, *s))
            .collect()
    };
    let primals: Vec<Array> = data
        .iter()
        .zip(shapes)
        .map(|(d, s)| random_layout(&mut rng, s, d.clone()).0)
        .collect();
    let out = f(&build(&data));
    let c = values(&mut rng, out.size(), -1.0, 1.0);
    let cot = Array::from_slice(&c, out.shape());
    let (_, grads) = vjp_flat(|xs| Ok(vec![f(xs)]), &primals, &[cot]).unwrap();

    // Finite differences.
    let h = 1e-6;
    for (i, g) in grads.iter().enumerate() {
        assert_eq!(g.shape(), dims(shapes[i]), "{name}: gradient {i} shape");
        let g = vec64(g);
        for j in 0..data[i].len() {
            let at = |delta: f64| {
                let mut d = data.clone();
                d[i][j] += delta;
                dot(&c, &vec64(&f(&build(&d))))
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
    let (_, jt) = jvp_flat(|xs| Ok(vec![f(xs)]), &primals, &build(&t)).unwrap();
    let lhs = dot(&c, &vec64(&jt[0]));
    let rhs: f64 = grads.iter().zip(&t).map(|(g, t)| dot(&vec64(g), t)).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(1.0),
        "{name}: <c, jvp_flat(t)> = {lhs} vs <vjp_flat(c), t> = {rhs}"
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn unary_gradients() {
    let cases: [(&str, F, f64, f64); 9] = [
        ("negative", |x| x[0].negative(), -2.0, 2.0),
        ("abs", |x| x[0].abs(), 0.5, 2.0),
        ("exp", |x| x[0].exp(), -2.0, 2.0),
        ("log", |x| x[0].log(), 0.5, 3.0),
        ("sqrt", |x| x[0].sqrt(), 0.5, 3.0),
        ("sin", |x| x[0].sin(), -3.0, 3.0),
        ("cos", |x| x[0].cos(), -3.0, 3.0),
        ("tanh", |x| x[0].tanh(), -2.0, 2.0),
        ("sigmoid", |x| x[0].sigmoid(), -3.0, 3.0),
    ];
    for (name, f, lo, hi) in cases {
        check(name, f, &[&[2, 3]], lo, hi);
    }
    check("abs-negative", |x| x[0].abs(), &[&[4]], -2.0, -0.5);
    check("logsumexp", |x| x[0].logsumexp(1), &[&[3, 4]], -3.0, 3.0);
    check(
        "logsumexp-all",
        |x| x[0].logsumexp_keepdims(..),
        &[&[2, 3]],
        -3.0,
        3.0,
    );
    check("softmax", |x| x[0].softmax(-1), &[&[3, 4]], -3.0, 3.0);
    check(
        "softmax-axes",
        |x| x[0].softmax([0, 2]),
        &[&[2, 2, 3]],
        -3.0,
        3.0,
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn binary_gradients_with_broadcasting() {
    let cases: [(&str, F); 6] = [
        ("add", |x| x[0].add(&x[1])),
        ("subtract", |x| x[0].subtract(&x[1])),
        ("multiply", |x| x[0].multiply(&x[1])),
        ("divide", |x| x[0].divide(&x[1])),
        ("maximum", |x| x[0].maximum(&x[1])),
        ("minimum", |x| x[0].minimum(&x[1])),
    ];
    for (name, f) in cases {
        check(name, f, &[&[2, 3], &[3]], 0.5, 2.0);
        check(&format!("{name}-outer"), f, &[&[3, 1], &[1, 4]], 0.5, 2.0);
    }
    check(
        "scalar-operands",
        |x| ops::multiply(2.0, &x[0]) - x[0].divide(3),
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
        |x| x[0].greater(0.0).select(&x[1], &x[2]),
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
        ("reshape", |x| x[0].reshape([3, 4]), &[2, 6]),
        ("transpose", |x| x[0].transpose([2, 0, 1]), &[2, 3, 4]),
        ("slice", |x| x[0].i((1.., (..).step(-2))), &[3, 5]),
        ("broadcast_to", |x| x[0].broadcast_to([2, 3, 4]), &[3, 1]),
        ("expand_dims", |x| x[0].expand_dims(1), &[2, 3]),
        ("squeeze", |x| x[0].squeeze(..), &[1, 3, 1]),
        ("t-then-reshape", |x| x[0].t().reshape([-1]), &[3, 4]),
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
        ("sum-all", |x| x[0].sum(..)),
        ("sum-axis", |x| x[0].sum(1)),
        ("sum-keep", |x| x[0].sum_keepdims([0, 2])),
        ("prod", |x| x[0].prod(1)),
        ("max", |x| x[0].max(2)),
        ("min", |x| x[0].min_keepdims([0, 1])),
        ("mean", |x| x[0].mean([1, 2])),
        ("prod-all", |x| x[0].prod(..)),
    ];
    for (name, f) in cases {
        check(name, f, &[&[2, 3, 4]], 0.5, 1.5);
    }
    // prod is a polynomial: finite differences are valid at zeros too.
    check(
        "prod-with-zeros",
        |x| (&x[0] * &x[1]).prod(1),
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
        |x| x[0].matmul(&x[1]),
        &[&[3, 4], &[4, 2]],
        -1.0,
        1.0,
    );
    check(
        "matmul-batch-broadcast",
        |x| x[0].matmul(&x[1]),
        &[&[2, 1, 3, 4], &[3, 4, 2]],
        -1.0,
        1.0,
    );
    check(
        "matmul-vector",
        |x| x[0].matmul(&x[1]),
        &[&[4], &[4, 3]],
        -1.0,
        1.0,
    );
    // beta scales c's derivative in both modes (MLX 0.32.4's jvp drops it, a bug there).
    check(
        "addmm",
        |x| x[2].addmm(&x[0], &x[1], 1.5, -0.5),
        &[&[3, 4], &[4, 2], &[2]],
        -1.0,
        1.0,
    );
    check(
        "addmm-batch",
        |x| x[2].addmm(&x[0], &x[1], 1.0, 2.0),
        &[&[2, 3, 4], &[4, 2], &[3, 2]],
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
            let h = (x[0].matmul(&x[1]) + &x[2]).tanh();
            let y = h.matmul(&x[3]);
            let r = y - 0.5;
            (&r * &r).mean(..)
        },
        &[&[4, 3], &[3, 5], &[5], &[5, 2]],
        -1.0,
        1.0,
    );
    check(
        "astype-roundtrip",
        |x| x[0].astype(DType::F64) * &x[0],
        &[&[3]],
        -1.0,
        1.0,
    );
}

#[test]
fn value_and_grad_selects_arguments() {
    let loss = |xs: &[Array]| Ok((&xs[0] * &xs[1]).sum(..));
    let a = array![1.0, 2.0];
    let b = array![3.0, 4.0];
    let (value, grads) = value_and_grad_flat(loss, &[1, 0])(&[a, b]).unwrap();
    assert_eq!(value.item::<f32>().unwrap(), 11.0);
    assert_eq!(grads[0].to_vec::<f32>().unwrap(), [1.0, 2.0]); // d/db = a
    assert_eq!(grads[1].to_vec::<f32>().unwrap(), [3.0, 4.0]); // d/da = b
}

#[test]
fn eval_inside_the_function_keeps_the_gradient() {
    let g = grad(|x: &Array| {
        let y = x * x;
        y.eval()?;
        let z = &y * x;
        eval(std::slice::from_ref(&z))?;
        Ok(z.sum(..))
    });
    let x = array![1.0, 2.0];
    assert_eq!(g(&x).unwrap().to_vec::<f32>().unwrap(), [3.0, 12.0]);
}

#[test]
fn higher_order_derivatives() {
    let x = Array::from([0.3f64, 1.2]);
    let sum_of = |f: fn(&Array) -> Result<Array>| move |x: &Array| Ok(f(x)?.sum(..));
    let d2 = grad(sum_of(|x| {
        let g = grad(|x: &Array| Ok(x.sin().sum(..)))(x)?;
        Ok(g)
    }));
    let expect: Vec<f64> = [0.3f64, 1.2].iter().map(|v| -v.sin()).collect();
    for (g, e) in vec64(&d2(&x).unwrap()).iter().zip(&expect) {
        assert!((g - e).abs() < 1e-12);
    }
    // Third derivative of x^4 is 24x.
    let d3 = grad(grad(grad(|x: &Array| {
        let x2 = x * x;
        Ok(&x2 * &x2)
    })));
    let at = Array::from(0.5f64);
    assert!((d3(&at).unwrap().item::<f64>().unwrap() - 12.0).abs() < 1e-12);
    // Hessian-vector product: jvp of grad. f = sum(x^3) → H = diag(6x).
    let g = grad(|x: &Array| Ok((x * x * x).sum(..)));
    let t = Array::from([1.0f64, -2.0]);
    let (_, hv) = jvp_flat(|xs| Ok(vec![g(&xs[0])?]), &[x], &[t]).unwrap();
    for (h, e) in vec64(&hv[0]).iter().zip([6.0 * 0.3, -2.0 * 6.0 * 1.2]) {
        assert!((h - e).abs() < 1e-12);
    }
}

#[test]
fn stop_gradient_blocks_the_gradient() {
    let g = grad(|x: &Array| Ok((x * x.stop_gradient()).sum(..)));
    let x = array![2.0, -1.0];
    assert_eq!(g(&x).unwrap().to_vec::<f32>().unwrap(), [2.0, -1.0]);
    assert_eq!(x.stop_gradient().to_vec::<f32>().unwrap(), [2.0, -1.0]);
    let (_, t) = jvp_flat(
        |xs| Ok(vec![xs[0].stop_gradient()]),
        std::slice::from_ref(&x),
        std::slice::from_ref(&x),
    )
    .unwrap();
    assert_eq!(t[0].to_vec::<f32>().unwrap(), [0.0, 0.0]);
}

#[test]
fn mlx_edge_cases() {
    let one = Array::from(1.0f32);
    let grad_of = |f: fn(&Array) -> Array, x: &[f32]| {
        let x = Array::from(x);
        let (_, g) = vjp_flat(|xs| Ok(vec![f(&xs[0])]), &[x], std::slice::from_ref(&one)).unwrap();
        g[0].to_vec::<f32>().unwrap()
    };
    let p = |x: &Array| x.prod(..);
    assert_eq!(grad_of(p, &[1.0, 0.0, 3.0, 2.0]), [0.0, 6.0, 0.0, 0.0]);
    assert_eq!(grad_of(p, &[1.0, 0.0, 3.0, 0.0, 2.0]), [0.0; 5]);
    let m = |x: &Array| x.max(..);
    assert_eq!(grad_of(m, &[2.0, 2.0, 1.0]), [0.5, 0.5, 0.0]);
    let a = |x: &Array| x.abs().sum(..);
    assert_eq!(grad_of(a, &[0.0, -2.0]), [0.0, -1.0]);
    // Element-wise maximum sends ties to the second operand.
    let ab = [Array::from(1.0f32), Array::from(1.0f32)];
    let (_, g) = vjp_flat(
        |x| Ok(vec![x[0].maximum(&x[1])]),
        &ab,
        std::slice::from_ref(&one),
    )
    .unwrap();
    assert_eq!(
        (g[0].item::<f32>().unwrap(), g[1].item::<f32>().unwrap()),
        (0.0, 1.0)
    );
    // Integer primals are rejected; gradients pass through casts.
    let g = grad(|x: &Array| Ok((x * x).sum(..)));
    let ints = Array::from([1i32, 2]);
    assert!(matches!(g(&ints), Err(Error::InvalidArgument(_))));
    let cast = grad(|x: &Array| Ok(x.astype(DType::I32).astype(DType::F32).sum(..)));
    assert_eq!(
        cast(&Array::from(1.5f32)).unwrap().item::<f32>().unwrap(),
        1.0
    );
}

#[test]
fn non_differentiable_and_independent_inputs_get_zero() {
    let x = array![1.0, 3.0, 2.0];
    let y = array![5.0, 6.0, 7.0];
    let ones = Array::from([1.0f32; 3]);
    let (_, g) = vjp_flat(
        |xs| Ok(vec![xs[0].greater(2.0).astype(DType::F32)]),
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
    let (_, g) = vjp_flat(
        |xs| Ok(vec![xs[0].argmax(..).astype(DType::F32)]),
        &[x],
        &[Array::from(1.0f32)],
    )
    .unwrap();
    assert_eq!(g[0].to_vec::<f32>().unwrap(), [0.0; 3]);
}

#[test]
fn several_outputs() {
    let x = array![0.5, 1.0];
    let c = array![1.0, 1.0];
    let (outs, g) = vjp_flat(
        |xs| Ok(vec![xs[0].sin(), &xs[0] * &xs[0]]),
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
    let x = array![1.0, 2.0];
    let not_scalar = grad(|x: &Array| Ok(x * x));
    assert!(matches!(not_scalar(&x), Err(Error::InvalidArgument(m)) if m.contains("scalar")));
    let f = |xs: &[Array]| Ok(xs[0].sum(..));
    for argnums in [&[][..], &[1], &[0, 0]] {
        assert!(matches!(
            value_and_grad_flat(f, argnums)(std::slice::from_ref(&x)),
            Err(Error::InvalidArgument(_))
        ));
    }
    let id = |xs: &[Array]| Ok(vec![xs[0].clone()]);
    assert!(vjp_flat(id, std::slice::from_ref(&x), &[Array::from(1.0f32)]).is_err());
    assert!(vjp_flat(id, std::slice::from_ref(&x), &[]).is_err());
    assert!(jvp_flat(id, std::slice::from_ref(&x), &[]).is_err());
    let failing =
        |_: &[Array]| -> Result<Vec<Array>> { Err(Error::InvalidArgument("user".into())) };
    assert!(matches!(vjp_flat(failing, &[], &[]), Err(Error::InvalidArgument(m)) if m == "user"));
}

#[test]
fn deep_chains_do_not_overflow_the_stack() {
    common::on_cpu();
    let n = if cfg!(miri) { 50 } else { 1_000_000 };
    let one = Array::from(1.0f32);
    let g = grad(move |x: &Array| {
        let mut y = x.clone();
        for _ in 0..n {
            y = &y * &one;
        }
        Ok(y.sum(..))
    });
    let x = Array::from(2.0f32);
    assert_eq!(g(&x).unwrap().item::<f32>().unwrap(), 1.0);
}

#[cfg(target_os = "macos")]
#[test]
fn gradients_on_the_gpu_match_the_cpu() {
    if cfg!(miri) {
        return;
    }
    let loss = |xs: &[Array]| {
        let s = xs[0].i(((..).step(2), ..));
        let y = s.matmul(&xs[1]).sigmoid();
        Ok((&y * &y).sum(..))
    };
    let mut rng = Rng::new(7);
    let a: Vec<f32> = (0..20).map(|_| unit(&mut rng) as f32 - 0.5).collect();
    let b: Vec<f32> = (0..12).map(|_| unit(&mut rng) as f32 - 0.5).collect();
    let run = |device: Device| {
        with_stream(device, || {
            let args = [Array::from_slice(&a, [5, 4]), Array::from_slice(&b, [4, 3])];
            value_and_grad_flat(loss, &[0, 1])(&args).and_then(|(_, g)| {
                g.iter()
                    .map(|g| g.to_vec::<f32>())
                    .collect::<Result<Vec<_>>>()
            })
        })
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
/// - `vjp_flat(c)` against `∂L/∂x + i·∂L/∂y` (MLX's convention) for `L = Re Σ conj(c)·f(z)`,
///   by central differences in the real and the imaginary direction;
/// - `Re⟨c, jvp_flat(t)⟩ = Re Σ⟨vjpᵢ(c), tᵢ⟩`.
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
    let array = |d: &[Complex64], s: &[i32], real: bool| -> Array {
        if real {
            Array::from_slice(&d.iter().map(|z| z.re).collect::<Vec<f32>>(), s)
        } else {
            Array::from_slice(d, s)
        }
    };
    let data: Vec<Vec<Complex64>> = shapes
        .iter()
        .enumerate()
        .map(|(k, s)| random(&mut rng, s.iter().product(), is_real(k)))
        .collect();
    let build = |data: &[Vec<Complex64>]| -> Vec<Array> {
        (0..shapes.len())
            .map(|k| array(&data[k], &dims(shapes[k]), is_real(k)))
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
    let out = f(&build(&data));
    let out_real = out.dtype() != DType::Complex64;
    let c = random(&mut rng, out.size(), out_real);
    let cot = array(&c, out.shape(), out_real);
    let (_, grads) = vjp_flat(|xs| Ok(vec![f(xs)]), &primals, &[cot]).unwrap();

    let h = 1e-2f32;
    let grads: Vec<Vec<Complex64>> = grads
        .iter()
        .enumerate()
        .map(|(k, g)| {
            assert_eq!(g.shape(), dims(shapes[k]), "{name}: gradient {k} shape");
            complex_values(g)
        })
        .collect();
    for (k, g) in grads.iter().enumerate() {
        for j in 0..data[k].len() {
            let at = |re: f32, im: f32| {
                let mut d = data.clone();
                d[k][j] = Complex64::new(d[k][j].re + re, d[k][j].im + im);
                pair(&c, &complex_values(&f(&build(&d))))
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
    let (_, jt) = jvp_flat(|xs| Ok(vec![f(xs)]), &primals, &build(&t)).unwrap();
    let lhs = pair(&c, &complex_values(&jt[0]));
    let rhs: f32 = grads.iter().zip(&t).map(|(g, t)| pair(g, t)).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
        "{name}: Re<c, jvp_flat(t)> = {lhs} vs Re<vjp_flat(c), t> = {rhs}"
    );
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of evaluations: too slow under Miri, and covered natively"
)]
fn complex_gradients() {
    let unary: [(&str, F); 3] = [
        ("c-negative", |x| x[0].negative()),
        ("c-conjugate", |x| x[0].conjugate()),
        ("c-abs", |x| x[0].abs()),
    ];
    for (name, f) in unary {
        check_complex(name, f, &[&[2, 3]], &[]);
    }
    let binary: [(&str, F); 4] = [
        ("c-add", |x| x[0].add(&x[1])),
        ("c-subtract", |x| x[0].subtract(&x[1])),
        ("c-multiply", |x| x[0].multiply(&x[1])),
        ("c-divide", |x| {
            x[0].divide(x[1].add(Complex64::new(2.0, 0.5)))
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
        |x| x[0].greater(0.0).select(&x[1], &x[2]),
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
        ("c-reshape", |x| x[0].reshape([3, 2]), &[2, 3]),
        ("c-transpose", |x| x[0].t(), &[2, 3]),
        ("c-slice", |x| x[0].i((..).step(-2)), &[5]),
        ("c-broadcast", |x| x[0].broadcast_to([2, 3]), &[3]),
    ];
    for (name, f, shape) in views {
        check_complex(name, f, &[shape], &[]);
    }
    check_complex("c-sum", |x| x[0].sum(1), &[&[2, 3]], &[]);
    check_complex("c-prod", |x| x[0].prod(1), &[&[2, 3]], &[]);
    check_complex("c-matmul", |x| x[0].matmul(&x[1]), &[&[2, 3], &[3, 2]], &[]);
    check_complex(
        "c-composite",
        |x| (&x[0] * &x[1] + x[2].conjugate()).abs(),
        &[&[3], &[3], &[3]],
        &[],
    );
    check_complex(
        "real-through-complex",
        |x| {
            let z = x[0].astype(DType::Complex64) * &x[1];
            z.abs().sum(..)
        },
        &[&[3], &[3]],
        &[true],
    );
}

#[test]
fn complex_abs_at_zero_has_zero_derivative() {
    let z = Array::from([Complex64::new(0.0, 0.0), Complex64::new(3.0, 4.0)]);
    let ones = array![1.0, 1.0];
    let (_, g) = vjp_flat(|x| Ok(vec![x[0].abs()]), std::slice::from_ref(&z), &[ones]).unwrap();
    let g = g[0].to_vec::<Complex64>().unwrap();
    assert_eq!((g[0].re, g[0].im), (0.0, 0.0));
    assert!((g[1].re - 0.6).abs() < 1e-6 && (g[1].im - 0.8).abs() < 1e-6);
    let t = Array::from([Complex64::new(1.0, 1.0); 2]);
    let (_, jt) = jvp_flat(|x| Ok(vec![x[0].abs()]), std::slice::from_ref(&z), &[t]).unwrap();
    let jt = jt[0].to_vec::<f32>().unwrap();
    assert_eq!(jt[0], 0.0);
    assert!((jt[1] - 1.4).abs() < 1e-6, "Re((3-4i)(1+i)) / 5 = 7/5");
}

#[test]
fn conjugate_values_batching_and_devices() {
    let z = Array::from([Complex64::new(1.0, 2.0), Complex64::new(-3.0, -0.5)]);
    let expect = [Complex64::new(1.0, -2.0), Complex64::new(-3.0, 0.5)];
    assert_eq!(z.conjugate().to_vec::<Complex64>().unwrap(), expect);
    let real = array![1.0, -2.0];
    assert_eq!(real.conjugate().to_vec::<f32>().unwrap(), [1.0, -2.0]);
    let batched =
        vmap_flat(|x| Ok(vec![x[0].conjugate()]), &[], &[])(std::slice::from_ref(&z)).unwrap();
    assert_eq!(batched[0].to_vec::<Complex64>().unwrap(), expect);
    #[cfg(target_os = "macos")]
    if !cfg!(miri) {
        let gpu = with_stream(Device::gpu(0), || {
            (&z * 1).conjugate().to_vec::<Complex64>()
        });
        match gpu {
            Ok(v) => assert_eq!(v, expect),
            Err(e) => eprintln!("skipped: no Metal device ({e})"),
        }
    }
}

/// erfinv is evaluated in float32 (as in MLX), so its derivative is checked with a float32
/// step size.
#[test]
fn erfinv_gradient() {
    let xs: Vec<f32> = vec![-0.9, -0.5, -0.1, 0.0, 0.3, 0.8];
    let x = Array::from(xs.as_slice());
    let g = grad(|x: &Array| Ok(x.erfinv().sum(..)))(&x)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let h = 1e-3f32;
    let at = |v: f32| Array::from(v).erfinv().item::<f32>().unwrap();
    for (i, &v) in xs.iter().enumerate() {
        let numeric = (at(v + h) - at(v - h)) / (2.0 * h);
        assert!(
            (g[i] - numeric).abs() <= 1e-2 * numeric.abs(),
            "x = {v}: {} vs {numeric}",
            g[i]
        );
    }
}

#[test]
fn transforms_reject_mismatched_dtypes_and_integer_primals() {
    common::on_cpu();
    let x = array![1.0, 2.0];
    let square = |xs: &[Array]| Ok(vec![&xs[0] * &xs[0]]);
    let f64s = Array::from([1.0f64, 1.0]);
    assert!(
        matches!(
            vjp_flat(
                square,
                std::slice::from_ref(&x),
                std::slice::from_ref(&f64s)
            ),
            Err(Error::InvalidArgument(_))
        ),
        "f64 cotangent for an f32 output"
    );
    let f16s = f64s.astype(DType::F16);
    assert!(
        matches!(
            jvp_flat(
                square,
                std::slice::from_ref(&x),
                std::slice::from_ref(&f16s)
            ),
            Err(Error::InvalidArgument(_))
        ),
        "f16 tangent for an f32 primal"
    );
    let ints = Array::from([1i32, 2]);
    let total = |x: &Array| Ok((x * x).sum(..));
    assert!(matches!(grad(total)(&ints), Err(Error::InvalidArgument(_))));
    let bools = Array::from([true]);
    assert!(matches!(
        value_and_grad_flat(|xs: &[Array]| Ok(xs[0].sum(..)), &[0])(&[bools]),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn gradients_over_empty_axes_are_empty() {
    common::on_cpu();
    let empty = Array::from_slice::<f32>(&[], [2, 0]);
    for reverse in [false, true] {
        for max in [true, false] {
            let options = ScanOptions {
                reverse,
                exclusive: true,
            };
            let scan = |x: &Array| {
                let y = match max {
                    true => x.cummax_with(1, options),
                    false => x.cummin_with(1, options),
                };
                Ok(y.sum(..))
            };
            let g = grad(scan)(&empty).unwrap();
            assert_eq!(g.shape(), [2, 0], "reverse {reverse} max {max}");
            assert!(g.to_vec::<f32>().unwrap().is_empty());
        }
    }
    type Loss = Box<dyn Fn(&Array) -> Result<Array>>;
    let cases: [(&str, Loss); 5] = [
        ("prod", Box::new(|x| Ok(x.prod(1).sum(..)))),
        ("max", Box::new(|x| Ok(x.max_keepdims(0).sum(..)))),
        ("min", Box::new(|x| Ok(x.min_keepdims(0).sum(..)))),
        ("sort", Box::new(|x| Ok(x.sort(1).sum(..)))),
        (
            "concatenate",
            Box::new(|x| Ok(ops::concatenate(&[x.clone(), x.clone()], 1).sum(..))),
        ),
    ];
    for (name, f) in cases {
        let g = grad(f)(&empty).unwrap();
        assert_eq!(g.shape(), [2, 0], "{name}");
        assert!(g.to_vec::<f32>().unwrap().is_empty(), "{name}");
    }
}

// ── Transforms over trees ───────────────────────────────────────────────────

mod trees {
    use std::collections::BTreeMap;

    use super::common::{self, Rng, random_layout};
    use nuvec_core::tree::{Leaves, Param, Tree, Visit, VisitMut};
    use nuvec_core::*;

    fn bits(a: &Array) -> Vec<u64> {
        a.astype(DType::F64)
            .to_vec::<f64>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }

    fn same<T: Tree>(a: &T, b: &T) {
        assert_eq!(a.paths(), b.paths());
        for (x, y) in a.leaves().iter().zip(b.leaves()) {
            assert_eq!(x.shape(), y.shape());
            assert_eq!(x.dtype(), y.dtype());
            assert_eq!(bits(x), bits(&y));
        }
    }

    /// A random array of `shape` and `dtype` with a random (non-contiguous) layout.
    fn random(rng: &mut Rng, shape: &[usize], dtype: DType) -> Array {
        let n: usize = shape.iter().product();
        let v: Vec<f64> = (0..n).map(|_| rng.f64() / 10.0).collect();
        random_layout(rng, shape, v).0.astype(dtype)
    }

    /// A hand-written "module": a parameter, a frozen parameter and a nested child.
    #[derive(Clone)]
    struct Block {
        w: Param,
        frozen: Param,
        inner: (Array, Vec<Array>),
    }

    impl Leaves for Block {
        fn visit(&self, v: &mut Visit<'_>) {
            v.child("w", &self.w);
            v.child("frozen", &self.frozen);
            v.child("inner", &self.inner);
        }
    }

    impl Tree for Block {
        fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
            v.child("w", &mut self.w);
            v.child("frozen", &mut self.frozen);
            v.child("inner", &mut self.inner);
        }
    }

    /// `Σ tanh(leaf)²` over every leaf: a smooth loss of any tree.
    fn loss<T: Tree>(t: &T) -> Array {
        let mut total = Array::from(0.0f32);
        t.for_each(|_, a| {
            total = total.add(a.tanh().multiply(a.tanh()).sum(..).astype(DType::F32))
        });
        total
    }

    /// `grad`/`vjp`/`jvp` over `tree` equal the flat transforms on its leaves bit for bit.
    fn check_against_flat<T: Tree>(tree: &T) {
        let leaves = tree.leaves();
        let argnums: Vec<usize> = (0..leaves.len()).collect();
        let flat = |xs: &[Array]| loss(&tree.unflatten(xs.iter().cloned())?).check();
        let (v_flat, g_flat) = value_and_grad_flat(flat, &argnums)(&leaves).unwrap();
        let (v, g) = value_and_grad(|t: &T| loss(t))(tree).unwrap();
        assert_eq!(bits(&v), bits(&v_flat));
        same(&g, &tree.unflatten(g_flat).unwrap());
        same(&grad(|t: &T| loss(t))(tree).unwrap(), &g);

        let out_flat = |xs: &[Array]| Ok(vec![loss(&tree.unflatten(xs.iter().cloned())?)]);
        let cot = Array::from(1.5f32);
        let (_, vj_flat) = vjp_flat(out_flat, &leaves, std::slice::from_ref(&cot)).unwrap();
        let (_, vj) = vjp(|t: &T| loss(t), tree, &cot).unwrap();
        same(&vj, &tree.unflatten(vj_flat).unwrap());

        let tangents = tree.map(|_, a| a.ones_like());
        let (_, jv_flat) = jvp_flat(out_flat, &leaves, &tangents.leaves()).unwrap();
        let (_, jv) = jvp(|t: &T| loss(t), tree, &tangents).unwrap();
        assert_eq!(bits(&jv), bits(&jv_flat[0]));
    }

    #[test]
    fn tree_transforms_equal_the_flat_ones() {
        common::on_cpu();
        let mut rng = Rng::new(63);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for _ in 0..4 {
                let r = |rng: &mut Rng| {
                    let s = rng.shape();
                    random(rng, &s, dtype)
                };
                let tuple = (r(&mut rng), r(&mut rng), random(&mut rng, &[], dtype));
                check_against_flat(&tuple);
                check_against_flat(&vec![r(&mut rng), random(&mut rng, &[0, 3], dtype)]);
                let map: BTreeMap<String, Array> =
                    [("b".to_owned(), r(&mut rng)), ("a".to_owned(), r(&mut rng))].into();
                check_against_flat(&map);
                check_against_flat(&(Some(r(&mut rng)), None::<Array>));
                // A transposed and a sliced leaf (`x.i(..)`).
                let m = random(&mut rng, &[3, 4], dtype);
                check_against_flat(&(m.t(), m.i((.., 1..3))));
            }
        }
    }

    #[test]
    fn gradients_check_against_finite_differences() {
        common::on_cpu();
        let x = array![0.3, -0.7].astype(DType::F64);
        let w = array![[0.5, 0.1], [-0.2, 0.4]].astype(DType::F64);
        let f = |(x, w): &(Array, Array)| w.matmul(x).tanh().sum(..);
        let (gx, gw) = grad(f)(&(x.clone(), w.clone())).unwrap();
        let eps = 1e-6;
        for (i, (leaf, g)) in [(&x, &gx), (&w, &gw)].into_iter().enumerate() {
            let n = leaf.size();
            let base = leaf.to_vec::<f64>().unwrap();
            let got = g.to_vec::<f64>().unwrap();
            for k in 0..n {
                let mut up = base.clone();
                up[k] += eps;
                let mut down = base.clone();
                down[k] -= eps;
                let at = |v: Vec<f64>| {
                    let p = Array::from_slice(&v, leaf.shape());
                    let pair = if i == 0 {
                        (p, w.clone())
                    } else {
                        (x.clone(), p)
                    };
                    f(&pair).item::<f64>().unwrap()
                };
                let fd = (at(up) - at(down)) / (2.0 * eps);
                assert!(
                    (fd - got[k]).abs() < 1e-6,
                    "leaf {i}[{k}]: {fd} vs {}",
                    got[k]
                );
            }
        }
        // vjp/jvp consistency: ⟨c, J t⟩ = ⟨Jᵀ c, t⟩.
        let t = (array![1.0, 2.0].astype(DType::F64), w.ones_like());
        let primals = (x.clone(), w.clone());
        let (_, jt) = jvp(|p: &(Array, Array)| p.1.matmul(&p.0).tanh(), &primals, &t).unwrap();
        let c = array![0.5, -1.5].astype(DType::F64);
        let (_, (cx, cw)) =
            vjp(|p: &(Array, Array)| p.1.matmul(&p.0).tanh(), &primals, &c).unwrap();
        let lhs = (&c * &jt).sum(..).item::<f64>().unwrap();
        let rhs = ((&cx * &t.0).sum(..) + (&cw * &t.1).sum(..))
            .item::<f64>()
            .unwrap();
        assert!((lhs - rhs).abs() < 1e-12, "{lhs} vs {rhs}");
    }

    /// Runtime counterparts of the `trybuild` inference cases.
    #[test]
    fn closure_forms_compile_and_agree() {
        common::on_cpu();
        let x = array![1.0, 2.0, 3.0];
        let g = grad(|x: &Array| x.multiply(x).sum(..))(&x).unwrap();
        assert_eq!(g.to_vec::<f32>().unwrap(), [2.0, 4.0, 6.0]);
        let as_result =
            grad(|x: &Array| -> Result<Array> { Ok(x.multiply(x).sum(..)) })(&x).unwrap();
        assert_eq!(bits(&as_result), bits(&g));
        // `?` inside a closure returning a generic `R: IntoResult`.
        let with_try = grad(|x: &Array| {
            let y = x.multiply(x).check()?;
            Ok(y.sum(..))
        })(&x)
        .unwrap();
        assert_eq!(bits(&with_try), bits(&g));
        // A failed `Result` surfaces as `Err`.
        let failed = grad(|x: &Array| -> Result<Array> {
            Err(Error::InvalidArgument(format!("bad {}", x.ndim())))
        })(&x);
        assert!(failed.is_err());
        // A fn item by name; a captured batch gets no gradient.
        fn sq(p: &(Array, Array)) -> Array {
            p.0.multiply(&p.1).sum(..)
        }
        let batch = array![2.0, 2.0, 2.0];
        let (v, (gw, gb)) = value_and_grad(sq)(&(x.clone(), batch.clone())).unwrap();
        assert_eq!(v.item::<f32>().unwrap(), 12.0);
        assert_eq!(gw.to_vec::<f32>().unwrap(), [2.0; 3]);
        assert_eq!(gb.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
        let (_, gw2) = value_and_grad(|w: &Array| w.multiply(&batch).sum(..))(&x).unwrap();
        assert_eq!(bits(&gw2), bits(&gw));
        // Aux outputs by type: `(loss, aux)`; aux carries no gradient.
        let ((loss, logits), g2) =
            value_and_grad(|x: &Array| (x.multiply(x).sum(..), x.multiply(2.0)))(&x).unwrap();
        assert_eq!(loss.item::<f32>().unwrap(), 14.0);
        assert_eq!(logits.to_vec::<f32>().unwrap(), [2.0, 4.0, 6.0]);
        assert_eq!(bits(&g2), bits(&g));
        let ((_, aux), _) =
            value_and_grad(|x: &Array| Ok((x.sum(..), vec![x.clone(), x.exp()])))(&x).unwrap();
        assert_eq!(aux.len(), 2);
        // A `&mut` tree reborrowed as `&`.
        let mut owned = (x.clone(), batch);
        let r: &mut (Array, Array) = &mut owned;
        let _ = value_and_grad(sq)(r).unwrap();
        // The loss must be 0-D.
        assert!(grad(|x: &Array| x.multiply(2.0))(&x).is_err());
        // Empty trees.
        let empty: Vec<Array> = Vec::new();
        assert!(
            grad(|_: &Vec<Array>| Array::from(1.0f32))(&empty)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn frozen_and_integer_leaves_get_free_zero_gradients() {
        common::on_cpu();
        let block = Block {
            w: Param::new(array![1.0, 2.0]),
            frozen: Param::frozen(array![3.0, 4.0]),
            inner: (array![5.0], vec![]),
        };
        let g =
            grad(|b: &Block| b.w.multiply(&b.frozen).sum(..) + b.inner.0.sum(..))(&block).unwrap();
        assert_eq!(g.w.to_vec::<f32>().unwrap(), [3.0, 4.0]);
        assert!(g.frozen.is_frozen(), "the gradient tree keeps the flag");
        assert_eq!(g.frozen.to_vec::<f32>().unwrap(), [0.0, 0.0]);
        // An integer parameter gets a zero of its dtype; a plain integer array is an error.
        let ints = (array![1.0], Param::new(Array::from_slice(&[1i32, 2], [2])));
        let g2 = grad(|p: &(Array, Param)| p.0.sum(..))(&ints).unwrap();
        assert_eq!(g2.1.dtype(), DType::I32, "zero of the leaf's dtype");
        assert_eq!(g2.1.to_vec::<i32>().unwrap(), [0, 0]);
        let plain = (array![1.0], Array::from_slice(&[1i32, 2], [2]));
        let err = grad(|p: &(Array, Array)| p.0.sum(..))(&plain).unwrap_err();
        assert!(err.to_string().contains("\"1\" is int32"), "{err}");
        // A frozen [4096, 4096] leaf: its zero gradient is a stride-0 view of one element.
        let big = Block {
            w: Param::new(array![1.0]),
            frozen: Param::frozen(ops::zeros([4096, 4096], DType::F32)),
            inner: (array![0.0], vec![]),
        };
        let g = grad(|b: &Block| b.w.sum(..))(&big).unwrap();
        nuvec_core::eval(&g).unwrap();
        assert_eq!(graph::strides(&g.frozen), [0, 0]);
    }

    #[test]
    fn cotangents_must_match_the_output_tree() {
        common::on_cpu();
        let x = array![1.0, 2.0];
        let err = vjp(
            |x: &Array| (x.clone(), x.exp()),
            &x,
            &(x.clone(), array![1.0]),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("\"1\""), "{err}");
        let err = jvp(
            |x: &(Array, Array)| x.0.add(&x.1),
            &(x.clone(), x.clone()),
            &(x.clone(), array![1.0]),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("\"1\""), "{err}");
    }

    /// Tree transforms compose like the flat ones (constitution III).
    #[test]
    fn tree_transforms_compose() {
        common::on_cpu();
        let x = array![0.5, -0.25, 1.0];
        let m = array![[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]];
        // grad of a vmapped function.
        let batched = vmap(|(m, v): &(Array, Array)| m.matmul(v).tanh()).in_axes((None, 0));
        let vs = x
            .reshape([1, 3])
            .broadcast_to([2, 3])
            .multiply(array![[1.0], [2.0]]);
        let g = grad(|p: &(Array, Array)| Ok(batched.call(p)?.sum(..)))(&(m.clone(), vs.clone()))
            .unwrap();
        let flat = vmap_flat(
            |xs| Ok(vec![xs[0].matmul(&xs[1]).tanh()]),
            &[None, Some(0)],
            &[],
        );
        let gf =
            grad_flat(|xs: &[Array]| Ok(flat(xs)?[0].sum(..)), &[0, 1])(&[m.clone(), vs.clone()])
                .unwrap();
        assert_eq!(bits(&g.0), bits(&gf[0]));
        assert_eq!(bits(&g.1), bits(&gf[1]));
        // vmap of grad.
        let per_example = vmap(|v: &Array| grad(|v: &Array| v.tanh().sum(..))(v));
        let got = per_example.call(&vs).unwrap();
        let want = vmap_flat(
            |xs| {
                Ok(vec![
                    grad_flat(|ys: &[Array]| Ok(ys[0].tanh().sum(..)), &[0])(xs)?.remove(0),
                ])
            },
            &[],
            &[],
        )(std::slice::from_ref(&vs))
        .unwrap();
        assert_eq!(bits(&got), bits(&want[0]));
        // jvp of grad.
        let (_, hv) = jvp(
            |v: &Array| grad(|v: &Array| v.tanh().sum(..))(v),
            &x,
            &x.ones_like(),
        )
        .unwrap();
        let (_, hv_flat) = jvp_flat(
            |xs| grad_flat(|ys: &[Array]| Ok(ys[0].tanh().sum(..)), &[0])(xs),
            std::slice::from_ref(&x),
            &[x.ones_like()],
        )
        .unwrap();
        assert_eq!(bits(&hv), bits(&hv_flat[0]));
        // checkpoint inside vmap.
        let block = checkpoint(|v: &Array| v.sin().exp());
        let got = vmap(move |v: &Array| block(v)).call(&vs).unwrap();
        let want = vmap_flat(|xs| Ok(vec![xs[0].sin().exp()]), &[], &[])(std::slice::from_ref(&vs))
            .unwrap();
        assert_eq!(bits(&got), bits(&want[0]));
    }
}
