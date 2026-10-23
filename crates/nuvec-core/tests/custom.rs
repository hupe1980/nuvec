//! Custom transform rules and gradient checkpointing, and the `Copy` and `Depends`
//! primitives they build on.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nuvec_core::graph::Primitive;
use nuvec_core::ops;
use nuvec_core::*;

fn sum_all(a: &Array) -> Result<Array> {
    Ok(a.sum(..))
}

fn values(a: &Array) -> Vec<f32> {
    a.to_vec::<f32>().unwrap()
}

/// Equal up to a few ulps (Miri perturbs transcendental functions slightly).
fn assert_close(got: &Array, expected: &Array) {
    for (g, e) in values(got).iter().zip(values(expected)) {
        assert!((g - e).abs() <= 1e-5 * e.abs().max(1.0), "{g} vs {e}");
    }
}

#[test]
fn custom_rules_are_used() {
    let jvp_calls = Arc::new(AtomicUsize::new(0));
    let vmap_calls = Arc::new(AtomicUsize::new(0));
    let (j, v) = (jvp_calls.clone(), vmap_calls.clone());
    let square = custom_function_flat(|xs| Ok(vec![xs[0].multiply(&xs[0])]))
        .with_vjp(|_, _, cot| Ok(vec![cot[0].multiply(10)]))
        .with_jvp(move |_, t| {
            j.fetch_add(1, Ordering::SeqCst);
            Ok(vec![t[0].multiply(100)])
        })
        .with_vmap(move |xs, axes| {
            v.fetch_add(1, Ordering::SeqCst);
            Ok((vec![xs[0].multiply(&xs[0])], axes[0].unwrap_or(0)))
        });
    let x = array![1.0, 2.0];
    let out = square.call(std::slice::from_ref(&x)).unwrap();
    assert_eq!(values(&out[0]), [1.0, 4.0]);

    let s = square.clone();
    let g = grad(move |x: &Array| sum_all(&s.call(std::slice::from_ref(x))?[0]));
    assert_eq!(values(&g(&x).unwrap()), [10.0, 10.0]);

    let s = square.clone();
    let (_, t) = jvp_flat(
        |xs| s.call(xs),
        std::slice::from_ref(&x),
        std::slice::from_ref(&x),
    )
    .unwrap();
    assert_eq!(values(&t[0]), [100.0, 200.0]);
    assert_eq!(jvp_calls.load(Ordering::SeqCst), 1);

    let s = square.clone();
    let batched = vmap_flat(move |xs| s.call(xs), &[], &[])(std::slice::from_ref(&x)).unwrap();
    assert_eq!(values(&batched[0]), [1.0, 4.0]);
    assert_eq!(vmap_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn default_rules_transform_the_function() {
    let body = |xs: &[Array]| Ok(vec![xs[0].sin() * &xs[1], xs[0].exp()]);
    let f = custom_function_flat(body);
    let x = array![0.5, 1.0, 1.5];
    let y = array![2.0, 3.0, 4.0];
    let cots = [array![1.0, 1.0, 1.0], array![0.5, 0.5, 0.5]];
    let primals = [x.clone(), y.clone()];
    let (_, custom) = vjp_flat(|xs| f.call(xs), &primals, &cots).unwrap();
    let (_, plain) = vjp_flat(body, &primals, &cots).unwrap();
    for (c, p) in custom.iter().zip(&plain) {
        assert_close(c, p);
    }
    let (_, custom) = jvp_flat(
        |xs| f.call(xs),
        &primals,
        &[cots[0].clone(), cots[0].clone()],
    )
    .unwrap();
    let (_, plain) = jvp_flat(body, &primals, &[cots[0].clone(), cots[0].clone()]).unwrap();
    for (c, p) in custom.iter().zip(&plain) {
        assert_close(c, p);
    }
    let batched =
        vmap_flat(|xs| f.call(xs), &[Some(0), None], &[])(&[x.clone(), Array::from(2.0f32)]);
    let plain = vmap_flat(body, &[Some(0), None], &[])(&[x, Array::from(2.0f32)]);
    for (c, p) in batched.unwrap().iter().zip(&plain.unwrap()) {
        assert_close(c, p);
    }
}

#[test]
fn wrong_rule_results_are_errors() {
    let x = array![1.0];
    let f = custom_function_flat(|xs| Ok(vec![xs[0].clone()])).with_vjp(|_, _, _| Ok(vec![]));
    let g = grad(move |x: &Array| sum_all(&f.call(std::slice::from_ref(x))?[0]));
    assert!(matches!(g(&x), Err(Error::InvalidArgument(_))));
    let f = custom_function_flat(|xs| Ok(vec![xs[0].clone()]))
        .with_jvp(|_, t| Ok(vec![t[0].clone(); 2]));
    let r = jvp_flat(
        |xs| f.call(xs),
        std::slice::from_ref(&x),
        std::slice::from_ref(&x),
    );
    assert!(matches!(r, Err(Error::InvalidArgument(_))));
    let empty = custom_function_flat(|_| Ok(vec![]));
    assert!(matches!(empty.call(&[x]), Err(Error::InvalidArgument(_))));
}

#[test]
fn checkpoint_recomputes_with_equal_gradients() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let block = move |xs: &[Array]| {
        c.fetch_add(1, Ordering::SeqCst);
        Ok(vec![xs[0].matmul(&xs[1]).tanh()])
    };
    let checkpointed = checkpoint_flat(block.clone());
    let loss = move |xs: &[Array]| sum_all(&(&checkpointed(xs)?[0] * 2.0));
    let plain = move |xs: &[Array]| sum_all(&(&block(xs)?[0] * 2.0));
    let a = ops::arange(0.0, 6.0, 1.0).reshape([2, 3]);
    let b = ops::arange(-1.0, 0.5, 0.25).reshape([3, 2]);
    let args = [&a * 0.1, b];
    calls.store(0, Ordering::SeqCst);
    let (value, grads) = value_and_grad_flat(loss, &[0, 1])(&args).unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "forward plus recomputation"
    );
    let (expected_value, expected) = value_and_grad_flat(plain, &[0, 1])(&args).unwrap();
    assert_close(&value, &expected_value);
    for (g, e) in grads.iter().zip(&expected) {
        assert_close(g, e);
    }
}

// ── Copy and Depends ─────────────────────────────────────────────────────────

fn copy(x: &Array) -> Result<Array> {
    Ok(graph::from_primitive(Primitive::Copy, vec![x.clone()])?.remove(0))
}

fn depends(x: &Array, deps: &[Array]) -> Result<Array> {
    let mut inputs = vec![x.clone()];
    inputs.extend_from_slice(deps);
    let primitive = Primitive::Depends { n_deps: deps.len() };
    Ok(graph::from_primitive(primitive, inputs)?.remove(0))
}

/// The devices to check: the CPU, and the GPU when there is one.
fn devices() -> Vec<Device> {
    let mut devices = vec![Device::cpu()];
    if common::require_gpu().is_some() {
        devices.push(Device::gpu(0));
    }
    devices
}

/// Random values of `dtype` in a random layout, and the values read back.
fn random_input(rng: &mut common::Rng, dtype: DType) -> (Array, Vec<f64>) {
    let shape = rng.shape();
    let n = common::size(&shape);
    let values: Vec<f32> = (0..n).map(|_| (rng.f64() as f32).round()).collect();
    let (x, _) = common::random_layout(rng, &shape, values);
    let x = x.astype(dtype);
    let expected = as_f64(&x);
    (x, expected)
}

fn as_f64(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

#[test]
fn copy_is_a_contiguous_copy_on_every_device() {
    for device in devices() {
        set_default_stream(Stream::default_for(device));
        let mut rng = common::Rng::new(0xC0);
        for case in 0..60 {
            let dtype = [DType::F32, DType::F16, DType::I32, DType::U8, DType::BF16][case % 5];
            let (x, expected) = random_input(&mut rng, dtype);
            let c = copy(&x).unwrap();
            assert_eq!((c.shape(), c.dtype()), (x.shape(), x.dtype()));
            assert!(graph::flags(&c).row_contiguous, "{device:?} case {case}");
            assert_eq!(as_f64(&c), expected, "{device:?} case {case} {dtype:?}");
        }
    }
    common::on_cpu();
}

#[test]
fn copy_differentiates_and_batches_as_the_identity() {
    common::on_cpu();
    let x = array![1.0, -2.0, 3.0];
    let f = |x: &Array| sum_all(&(copy(x)? * x));
    assert_eq!(values(&grad(f)(&x).unwrap()), [2.0, -4.0, 6.0]);
    let t = array![1.0, 0.5, 0.25];
    let (_, tangents) = jvp_flat(
        |xs| Ok(vec![copy(&xs[0])?]),
        std::slice::from_ref(&x),
        std::slice::from_ref(&t),
    )
    .unwrap();
    assert_eq!(values(&tangents[0]), values(&t));
    // Finite differences of a non-linear use.
    let g = grad(|x: &Array| sum_all(&copy(x)?.exp()))(&x).unwrap();
    assert_close(&g, &x.exp());
    let rows = array![[1.0, 2.0], [3.0, 4.0]];
    let batched =
        vmap_flat(|xs| Ok(vec![copy(&xs[0])?]), &[Some(1)], &[])(std::slice::from_ref(&rows))
            .unwrap();
    assert_eq!(values(&batched[0]), values(&rows.t()));
}

#[test]
fn depends_aliases_its_input_after_its_dependencies() {
    for device in devices() {
        set_default_stream(Stream::default_for(device));
        let mut rng = common::Rng::new(0xD0);
        for case in 0..30 {
            let dtype = [DType::F32, DType::I32, DType::F16][case % 3];
            let (x, expected) = random_input(&mut rng, dtype);
            let dep = &array![1.0, 2.0] * 3.0f32;
            let d = depends(&x, std::slice::from_ref(&dep)).unwrap();
            assert_eq!(
                (d.shape(), graph::strides(&d)),
                (x.shape(), graph::strides(&x))
            );
            d.eval().unwrap();
            assert!(dep.is_evaluated(), "dependencies are evaluated first");
            assert_eq!(as_f64(&d), expected, "{device:?} case {case}");
        }
    }
    common::on_cpu();
    let x = Array::from(1.0f32);
    assert!(graph::from_primitive(Primitive::Depends { n_deps: 2 }, vec![x.clone(), x]).is_err());
}

#[test]
fn depends_differentiates_into_its_first_input() {
    common::on_cpu();
    let x = array![1.0, 2.0];
    let y = array![5.0, 7.0, 9.0];
    let f = |xs: &[Array]| sum_all(&(depends(&xs[0], &[xs[1].clone()])? * &xs[0]));
    let (_, grads) = value_and_grad_flat(f, &[0, 1])(&[x.clone(), y.clone()]).unwrap();
    assert_eq!(values(&grads[0]), [2.0, 4.0]);
    assert_eq!(
        values(&grads[1]),
        [0.0, 0.0, 0.0],
        "no gradient into the dependency"
    );
    let t = array![1.0, -1.0];
    let ty = Array::from([1.0f32; 3]);
    let (_, tangents) = jvp_flat(
        |xs| Ok(vec![depends(&xs[0], &[xs[1].clone()])?]),
        &[x.clone(), y.clone()],
        &[t.clone(), ty],
    )
    .unwrap();
    assert_eq!(values(&tangents[0]), values(&t));
    let rows = array![[1.0, 2.0], [3.0, 4.0]];
    let batched = vmap_flat(
        |xs| Ok(vec![depends(&xs[0], &[xs[1].clone()])?]),
        &[Some(0), None],
        &[],
    )(&[rows.clone(), y])
    .unwrap();
    assert_eq!(values(&batched[0]), values(&rows));
}

/// `checkpoint` and `custom_function` over trees.
mod trees {
    use super::common;
    use nuvec_core::*;

    fn vals(a: &Array) -> Vec<f32> {
        a.to_vec::<f32>().unwrap()
    }

    #[test]
    fn checkpoint_over_tuples_keeps_the_gradients() {
        common::on_cpu();
        let block = |(w, x): &(Array, Array)| (w.matmul(x).tanh(), x.sin());
        let checkpointed = checkpoint(block);
        let p = (array![[0.5, -1.0], [2.0, 0.25]], array![0.3, -0.6]);
        let plain = grad(|p: &(Array, Array)| {
            let (a, b) = block(p);
            a.sum(..).add(b.sum(..))
        })(&p)
        .unwrap();
        let through = grad(|p: &(Array, Array)| {
            let (a, b) = checkpointed(p)?;
            Ok(a.sum(..).add(b.sum(..)))
        })(&p)
        .unwrap();
        assert_eq!(vals(&plain.0), vals(&through.0));
        assert_eq!(vals(&plain.1), vals(&through.1));
    }

    #[test]
    fn custom_rules_over_trees() {
        common::on_cpu();
        // Forward (a·b, a+b); the vjp rule pretends both are the identity in `a` only.
        let f = custom_function(|(a, b): &(Array, Array)| (a.multiply(b), a.add(b)))
            .with_vjp(
                |_p: &(Array, Array), _o: &(Array, Array), c: &(Array, Array)| {
                    Ok((c.0.add(&c.1), c.0.zeros_like()))
                },
            )
            .with_jvp(|_p: &(Array, Array), t: &(Array, Array)| Ok((t.0.clone(), t.1.clone())))
            .with_vmap(|(a, b): &(Array, Array), _axes: &[Option<i32>]| {
                Ok(((a.multiply(b), a.add(b)), 0))
            });
        let p = (array![2.0, 3.0], array![4.0, 5.0]);
        let (y, s) = f.call(&p).unwrap();
        assert_eq!(vals(&y), [8.0, 15.0]);
        assert_eq!(vals(&s), [6.0, 8.0]);
        let g = grad(|p: &(Array, Array)| {
            let (y, s) = f.call(p)?;
            Ok(y.sum(..).add(s.sum(..)))
        })(&p)
        .unwrap();
        assert_eq!(vals(&g.0), [2.0, 2.0], "the vjp rule");
        assert_eq!(vals(&g.1), [0.0, 0.0]);
        let (_, (ty, ts)) = jvp(
            |p: &(Array, Array)| f.call(p),
            &p,
            &(array![1.0, 1.0], array![7.0, 7.0]),
        )
        .unwrap();
        assert_eq!(
            (vals(&ty), vals(&ts)),
            (vec![1.0, 1.0], vec![7.0, 7.0]),
            "the jvp rule"
        );
        let batched =
            vmap(|p: &(Array, Array)| f.call(p)).call(&(p.0.reshape([2, 1]), p.1.reshape([2, 1])));
        assert_eq!(vals(&batched.unwrap().0), [8.0, 15.0]);
        // Without rules, transforms differentiate `f` itself (a checkpoint).
        let plain = custom_function(|(a, b): &(Array, Array)| a.multiply(b));
        let g = grad(|p: &(Array, Array)| Ok(plain.call(p)?.sum(..)))(&p).unwrap();
        assert_eq!((vals(&g.0), vals(&g.1)), (vec![4.0, 5.0], vec![2.0, 3.0]));
    }

    #[test]
    fn checkpoint_redraws_the_same_random_numbers() {
        common::on_cpu();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let noisy = |x: &Array| {
            let mask = random::bernoulli(0.5, x.shape(), None).astype(DType::F32);
            x.multiply(&mask).multiply(x)
        };
        random::seed(7);
        let plain = grad(|x: &Array| noisy(x).sum(..))(&x).unwrap();
        let after_plain = random::next_key().to_vec::<u32>().unwrap();
        random::seed(7);
        let block = checkpoint(noisy);
        let through = grad(move |x: &Array| Ok(block(x)?.sum(..)))(&x).unwrap();
        let after = random::next_key().to_vec::<u32>().unwrap();
        assert_eq!(vals(&plain), vals(&through), "same mask on recomputation");
        assert_eq!(
            after_plain, after,
            "the sequence advances as uncheckpointed"
        );
    }
}
