//! Tracer status is transient: arrays are tracers only while the transform they depend on
//! runs, so gradients, and parameters updated with them, are plain arrays afterwards.

use nuvec_core::*;

fn loss(w: &Array) -> Result<Array> {
    Ok(w.multiply(w).sum(..))
}

#[test]
fn gradients_are_not_tracers_after_the_transform() {
    let w = array![1.0, 2.0, 3.0];
    let g = grad(loss)(&w).unwrap();
    assert!(!graph::is_tracer(&g), "grad");
    let updated = &w - &(&g * 0.1f32);
    assert!(!graph::is_tracer(&updated), "w - lr * g");
    assert_eq!(updated.to_vec::<f32>().unwrap(), [0.8, 1.6, 2.4]);

    let (values, grads) =
        value_and_grad_flat(|xs: &[Array]| loss(&xs[0]), &[0])(std::slice::from_ref(&w)).unwrap();
    assert!(
        !graph::is_tracer(&values) && !graph::is_tracer(&grads[0]),
        "value_and_grad"
    );

    let (outs, vjps) = vjp_flat(
        |xs| Ok(vec![xs[0].multiply(&xs[0])]),
        std::slice::from_ref(&w),
        &[Array::from([1.0f32; 3])],
    )
    .unwrap();
    assert!(
        !graph::is_tracer(&outs[0]) && !graph::is_tracer(&vjps[0]),
        "vjp"
    );

    let (outs, jvps) = jvp_flat(
        |xs| Ok(vec![xs[0].multiply(&xs[0])]),
        std::slice::from_ref(&w),
        &[Array::from([1.0f32; 3])],
    )
    .unwrap();
    assert!(
        !graph::is_tracer(&outs[0]) && !graph::is_tracer(&jvps[0]),
        "jvp"
    );
}

#[test]
fn arrays_are_tracers_while_their_transform_runs() {
    let w = Array::from(3.0f32);
    let inner_seen = std::cell::Cell::new(false);
    let g = grad(|x: &Array| {
        let y = x * x;
        inner_seen.set(graph::is_tracer(&y));
        Ok(y)
    })(&w)
    .unwrap();
    assert!(inner_seen.get(), "inside the transform");
    assert!(!graph::is_tracer(&g), "after it");
}

#[test]
fn nested_transforms_keep_the_outer_tracers() {
    // d²/dx² x³ = 6x: the inner gradient must stay differentiable for the outer one after
    // the inner transform has exited.
    let x = Array::from(2.0f32);
    let outer_saw_tracer = std::cell::Cell::new(false);
    let d2 = grad(|x: &Array| {
        let inner = grad(|y: &Array| Ok(y.multiply(y).multiply(y)))(x)?;
        outer_saw_tracer.set(graph::is_tracer(&inner));
        Ok(inner)
    })(&x)
    .unwrap();
    assert!(
        outer_saw_tracer.get(),
        "an outer tracer after the inner transform exits"
    );
    assert_eq!(d2.item::<f32>().unwrap(), 12.0);
    assert!(!graph::is_tracer(&d2));
}

#[test]
fn nodes_evaluated_inside_a_transform_are_released_on_exit() {
    let x = Array::from(2.0f32);
    let kept = std::cell::RefCell::new(None);
    let g = grad(|x: &Array| {
        let y = x * x;
        y.eval()?;
        assert!(!graph::inputs(&y).is_empty(), "retained while tracing");
        *kept.borrow_mut() = Some(y.clone());
        Ok(y)
    })(&x)
    .unwrap();
    assert_eq!(g.item::<f32>().unwrap(), 4.0);
    let y = kept.borrow_mut().take().unwrap();
    assert!(!graph::is_tracer(&y));
    assert!(
        graph::inputs(&y).is_empty(),
        "detached once the transform exited"
    );
    assert_eq!(y.item::<f32>().unwrap(), 4.0);
}
