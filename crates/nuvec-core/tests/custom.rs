//! Custom transform rules and gradient checkpointing.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nuvec_core::*;

fn sum_all(a: &Array) -> Result<Array> {
    sum(a, &[], false)
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
    let square = custom_function(|xs| Ok(vec![multiply(&xs[0], &xs[0])?]))
        .with_vjp(|_, _, cot| Ok(vec![multiply(&cot[0], 10)?]))
        .with_jvp(move |_, t| {
            j.fetch_add(1, Ordering::SeqCst);
            Ok(vec![multiply(&t[0], 100)?])
        })
        .with_vmap(move |xs, axes| {
            v.fetch_add(1, Ordering::SeqCst);
            Ok((vec![multiply(&xs[0], &xs[0])?], axes[0].unwrap_or(0)))
        });
    let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
    let out = square.call(std::slice::from_ref(&x)).unwrap();
    assert_eq!(values(&out[0]), [1.0, 4.0]);

    let s = square.clone();
    let g = grad(move |x: &Array| sum_all(&s.call(std::slice::from_ref(x))?[0]));
    assert_eq!(values(&g(&x).unwrap()), [10.0, 10.0]);

    let s = square.clone();
    let (_, t) = jvp(
        |xs| s.call(xs),
        std::slice::from_ref(&x),
        std::slice::from_ref(&x),
    )
    .unwrap();
    assert_eq!(values(&t[0]), [100.0, 200.0]);
    assert_eq!(jvp_calls.load(Ordering::SeqCst), 1);

    let s = square.clone();
    let batched = vmap(move |xs| s.call(xs), &[], &[])(std::slice::from_ref(&x)).unwrap();
    assert_eq!(values(&batched[0]), [1.0, 4.0]);
    assert_eq!(vmap_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn default_rules_transform_the_function() {
    let body = |xs: &[Array]| Ok(vec![multiply(&sin(&xs[0])?, &xs[1])?, exp(&xs[0])?]);
    let f = custom_function(body);
    let x = Array::from_slice(&[0.5f32, 1.0, 1.5], &[3]).unwrap();
    let y = Array::from_slice(&[2.0f32, 3.0, 4.0], &[3]).unwrap();
    let cots = [
        Array::from_slice(&[1.0f32, 1.0, 1.0], &[3]).unwrap(),
        Array::from_slice(&[0.5f32, 0.5, 0.5], &[3]).unwrap(),
    ];
    let primals = [x.clone(), y.clone()];
    let (_, custom) = vjp(|xs| f.call(xs), &primals, &cots).unwrap();
    let (_, plain) = vjp(body, &primals, &cots).unwrap();
    for (c, p) in custom.iter().zip(&plain) {
        assert_close(c, p);
    }
    let (_, custom) = jvp(
        |xs| f.call(xs),
        &primals,
        &[cots[0].clone(), cots[0].clone()],
    )
    .unwrap();
    let (_, plain) = jvp(body, &primals, &[cots[0].clone(), cots[0].clone()]).unwrap();
    for (c, p) in custom.iter().zip(&plain) {
        assert_close(c, p);
    }
    let batched = vmap(|xs| f.call(xs), &[Some(0), None], &[])(&[x.clone(), Array::scalar(2.0f32)]);
    let plain = vmap(body, &[Some(0), None], &[])(&[x, Array::scalar(2.0f32)]);
    for (c, p) in batched.unwrap().iter().zip(&plain.unwrap()) {
        assert_close(c, p);
    }
}

#[test]
fn wrong_rule_results_are_errors() {
    let x = Array::from_slice(&[1.0f32], &[1]).unwrap();
    let f = custom_function(|xs| Ok(vec![xs[0].clone()])).with_vjp(|_, _, _| Ok(vec![]));
    let g = grad(move |x: &Array| sum_all(&f.call(std::slice::from_ref(x))?[0]));
    assert!(matches!(g(&x), Err(Error::InvalidArgument(_))));
    let f =
        custom_function(|xs| Ok(vec![xs[0].clone()])).with_jvp(|_, t| Ok(vec![t[0].clone(); 2]));
    let r = jvp(
        |xs| f.call(xs),
        std::slice::from_ref(&x),
        std::slice::from_ref(&x),
    );
    assert!(matches!(r, Err(Error::InvalidArgument(_))));
    let empty = custom_function(|_| Ok(vec![]));
    assert!(matches!(empty.call(&[x]), Err(Error::InvalidArgument(_))));
}

#[test]
fn checkpoint_recomputes_with_equal_gradients() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let block = move |xs: &[Array]| {
        c.fetch_add(1, Ordering::SeqCst);
        Ok(vec![tanh(&matmul(&xs[0], &xs[1])?)?])
    };
    let checkpointed = checkpoint(block.clone());
    let loss = move |xs: &[Array]| sum_all(&multiply(&checkpointed(xs)?[0], 2.0)?);
    let plain = move |xs: &[Array]| sum_all(&multiply(&block(xs)?[0], 2.0)?);
    let a = arange(0.0, 6.0, 1.0, DType::F32)
        .unwrap()
        .reshape(&[2, 3])
        .unwrap();
    let b = arange(-1.0, 0.5, 0.25, DType::F32)
        .unwrap()
        .reshape(&[3, 2])
        .unwrap();
    let args = [&a * 0.1, b];
    calls.store(0, Ordering::SeqCst);
    let (value, grads) = value_and_grad(loss, &[0, 1])(&args).unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "forward plus recomputation"
    );
    let (expected_value, expected) = value_and_grad(plain, &[0, 1])(&args).unwrap();
    assert_close(&value, &expected_value);
    for (g, e) in grads.iter().zip(&expected) {
        assert_close(g, e);
    }
}
