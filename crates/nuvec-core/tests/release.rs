//! An evaluated output releases its data when its last handle goes, whatever its siblings
//! do (MLX's `detach`; mrun's B7): `gated_delta`'s state kept and its output dropped holds
//! the state only, on the GPU and the CPU.

mod common;

use std::sync::{Mutex, MutexGuard, PoisonError};

use nuvec_core::*;

/// The allocator is process-wide: the tests take turns.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Bytes held by live arrays (0 without Metal: the checks then pass trivially).
fn active() -> usize {
    nuvec_gpu::Device::shared().map_or(0, |d| d.memory().active)
}

fn devices() -> Vec<Device> {
    let mut d = vec![Device::cpu()];
    if common::require_gpu().is_some() {
        d.push(Device::gpu(0));
    }
    d
}

/// `gated_delta`'s inputs over `t` steps (4 key heads, 8 value heads, head size 128),
/// evaluated: `y` is `t · 8 · 128` floats and the state `8 · 128 · 128`.
fn inputs(t: usize) -> Vec<Array> {
    let n = |shape: &[usize], scale: f32| {
        random::normal(shape, DType::F32, 0.0, 1.0, None).multiply(scale)
    };
    let xs = vec![
        n(&[1, t, 4, 128], 0.05),
        n(&[1, t, 4, 128], 0.05),
        n(&[1, t, 8, 128], 1.0),
        ops::full([1, t, 8], 0.9f32),
        ops::full([1, t, 8], 0.5f32),
        ops::zeros([1, 8, 128, 128], DType::F32),
    ];
    eval(&xs).unwrap();
    xs
}

fn delta(xs: &[Array]) -> (Array, Array) {
    fast::gated_delta(&xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5])
}

const STATE: usize = 8 * 128 * 128 * 4;

/// mrun's pattern: evaluate both outputs, keep the state, drop the output.
#[test]
fn a_kept_state_does_not_hold_the_dropped_output() {
    let _serial = serial();
    for device in devices() {
        let t = if device == Device::cpu() { 64 } else { 2048 };
        with_stream(device, || {
            let xs = inputs(t);
            let before = active();
            let (y, s) = delta(&xs);
            eval(&[&y, &s]).unwrap();
            let want = s.to_vec::<f32>().unwrap();
            drop(y);
            let held = active() - before;
            assert!(
                held < 2 * STATE,
                "{device:?}: {held} bytes held, state {STATE}"
            );
            assert_eq!(s.to_vec::<f32>().unwrap(), want, "{device:?}");
        });
    }
}

/// The output was never held when the operation was evaluated.
#[test]
fn an_output_nobody_holds_is_released_when_evaluated() {
    let _serial = serial();
    for device in devices() {
        let t = if device == Device::cpu() { 64 } else { 2048 };
        with_stream(device, || {
            let xs = inputs(t);
            let before = active();
            let (y, s) = delta(&xs);
            drop(y);
            s.eval().unwrap();
            let held = active() - before;
            assert!(
                held < 2 * STATE,
                "{device:?}: {held} bytes held, state {STATE}"
            );
        });
    }
}

/// A dropped output still used by a pending computation stays until it ran.
#[test]
fn a_pending_consumer_keeps_the_output_until_it_runs() {
    let _serial = serial();
    for device in devices() {
        let t = if device == Device::cpu() { 64 } else { 2048 };
        with_stream(device, || {
            let xs = inputs(t);
            let before = active();
            let (y, s) = delta(&xs);
            let total = y.sum(..);
            let want = y.to_vec::<f32>().unwrap().iter().sum::<f32>();
            drop(y);
            s.eval().unwrap();
            let got = total.item::<f32>().unwrap();
            assert!(
                (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                "{device:?}"
            );
            drop(total);
            let held = active() - before;
            assert!(
                held < 2 * STATE,
                "{device:?}: {held} bytes held, state {STATE}"
            );
        });
    }
}

/// A view of a dropped output keeps its data; the output's other siblings do not.
#[test]
fn views_keep_their_data_and_released_siblings_fail() {
    let _serial = serial();
    for device in devices() {
        with_stream(device, || {
            let xs = inputs(16);
            let (y, s) = delta(&xs);
            eval(&[&y, &s]).unwrap();
            let row = y.i((.., 3..4));
            let want = row.to_vec::<f32>().unwrap();
            drop(y);
            assert_eq!(row.to_vec::<f32>().unwrap(), want, "{device:?}");
            // One operation on the GPU (the CPU composes it from single-output steps).
            let siblings = graph::siblings(&s);
            if let [again, _] = siblings.as_slice() {
                assert_eq!(again.shape(), [1, 16, 8, 128], "{device:?}");
                let e = again.to_vec::<f32>().unwrap_err();
                assert!(matches!(e, Error::Released), "{device:?}: {e}");
            }
        });
    }
}

/// Several outputs dropped in turn: every kept one equals the all-kept result.
#[test]
fn multi_output_operations_keep_their_values() {
    let _serial = serial();
    for device in devices() {
        with_stream(device, || {
            let x = random::normal([64, 64], DType::F32, 0.0, 1.0, None);
            let w = random::normal([128, 64], DType::F32, 0.0, 1.0, None);
            eval(&[&x, &w]).unwrap();
            // quantize's triple: keep the codes and biases, drop the scales.
            let Quantized {
                weight,
                scales,
                biases,
                ..
            } = ops::quantize(&w, Quant::default());
            let biases = biases.unwrap();
            let want = biases.to_vec::<f32>().unwrap();
            eval(&[&weight, &scales, &biases]).unwrap();
            drop(scales);
            assert_eq!(biases.to_vec::<f32>().unwrap(), want, "{device:?}");
            assert!(weight.check().is_ok());
            // Decompositions: drop one output before and after evaluation.
            let [qm, r] = linalg::qr(&x);
            let want = r.to_vec::<f32>().unwrap();
            let [qm2, r2] = linalg::qr(&x);
            drop(qm);
            assert_eq!(r.to_vec::<f32>().unwrap(), want, "{device:?}");
            eval(&[&qm2, &r2]).unwrap();
            drop(r2);
            assert!(qm2.check().is_ok());
            let sym = x.add(x.t());
            let [vals, vecs] = linalg::eigh(&sym, false);
            let want = vals.to_vec::<f32>().unwrap();
            drop(vecs);
            assert_eq!(vals.to_vec::<f32>().unwrap(), want, "{device:?}");
        });
    }
}

/// Inside a transform, dropped outputs change nothing (the transform keeps the graph).
#[test]
fn transforms_are_unchanged() {
    let _serial = serial();
    let xs = inputs(8);
    let f = |q: &Array| {
        let mut ins = xs.clone();
        ins[0] = q.clone();
        let (y, s) = delta(&ins);
        drop(y);
        s.sum(..)
    };
    let g = grad(f)(&xs[0]).unwrap();
    let all = grad(|q: &Array| {
        let mut ins = xs.clone();
        ins[0] = q.clone();
        let (_y, s) = delta(&ins);
        s.sum(..)
    })(&xs[0])
    .unwrap();
    assert_eq!(g.to_vec::<f32>().unwrap(), all.to_vec::<f32>().unwrap());
}

/// An output dropped from another thread while its node runs, and while its GPU work is
/// in flight: the kept state reads its values and the output is released either way.
#[test]
fn drops_racing_evaluation_release_the_output() {
    let _serial = serial();
    for device in devices() {
        let t = if device == Device::cpu() { 64 } else { 2048 };
        with_stream(device, || {
            let xs = inputs(t);
            let want = {
                let (_, s) = delta(&xs);
                s.to_vec::<f32>().unwrap()
            };
            for _ in 0..8 {
                let before = active();
                let (y, s) = delta(&xs);
                let dropper = std::thread::spawn(move || drop(y));
                async_eval(&[&s]).unwrap();
                dropper.join().unwrap();
                assert_eq!(s.to_vec::<f32>().unwrap(), want, "{device:?}");
                let held = active() - before;
                assert!(held < 2 * STATE, "{device:?}: {held} bytes held");
            }
        });
    }
}
