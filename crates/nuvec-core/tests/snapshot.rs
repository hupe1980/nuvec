//! Snapshots with `copy`: an independent copy that in-place updates of the original never
//! reach, its derivatives, and resuming a chunked computation from a snapshot.

#![allow(clippy::type_complexity)]

mod common;

use nuvec_core::ops;
use nuvec_core::*;

fn bits(a: &Array) -> Vec<u64> {
    match a.dtype().category() {
        Category::Float => a
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        _ => a
            .astype(DType::I64)
            .to_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as u64)
            .collect(),
    }
}

fn devices() -> Vec<Device> {
    let mut d = vec![Device::cpu()];
    if !cfg!(miri) && with_stream(Device::gpu(0), || Array::from(1.0f32).eval()).is_ok() {
        d.push(Device::gpu(0));
    }
    d
}

/// `y = x.copy()`, then `x` updated in place (the update owns `x`'s buffer): `y` keeps the
/// old values, in every dtype and layout (contiguous, transposed, sliced with an offset,
/// broadcast, a scalar).
#[test]
fn a_copy_survives_in_place_updates_of_the_original() {
    for device in devices() {
        with_stream(device, || {
            for dtype in [DType::F32, DType::F16, DType::BF16, DType::U32] {
                let base = || ops::arange(0, 48, 1).astype(dtype).reshape([6, 8]);
                let layouts: Vec<(&str, Box<dyn Fn() -> Array>)> = vec![
                    ("contiguous", Box::new(base)),
                    ("transposed", Box::new(move || base().t())),
                    ("sliced", Box::new(move || base().i((1..5, 2..)))),
                    (
                        "broadcast",
                        Box::new(move || ops::arange(0, 8, 1).astype(dtype).broadcast_to([6, 8])),
                    ),
                    ("scalar", Box::new(move || Array::from(5u32).astype(dtype))),
                ];
                for (name, make) in &layouts {
                    let x = make().contiguous();
                    x.eval().unwrap();
                    let want = bits(&x);
                    let y = make().copy();
                    let y2 = x.copy();
                    eval(&[&y, &y2]).unwrap();
                    // In place: `at` takes `x` by value, nothing else holds it.
                    let x = if x.ndim() == 0 {
                        x.at(Ellipsis).set(Array::from(9u32).astype(dtype))
                    } else {
                        x.at(0).set(Array::from(9u32).astype(dtype))
                    };
                    x.eval().unwrap();
                    assert_ne!(bits(&x), want, "{device:?} {dtype} {name}: updated");
                    assert_eq!(bits(&y2), want, "{device:?} {dtype} {name}: the copy");
                    assert_eq!(bits(&y), bits(&make()), "{device:?} {dtype} {name}");
                    assert_eq!(y.shape(), make().shape());
                    assert_eq!(y.dtype(), dtype);
                }
            }
        });
    }
}

/// `copy` passes derivatives through: reverse and forward mode, batching, and finite
/// differences.
#[test]
fn copy_derivatives_are_the_identity() {
    let x = array![[0.5, -1.0, 2.0], [3.0, 0.25, -0.75]];
    let f = |x: &Array| x.copy().multiply(x).sin().sum(..);
    let g = grad(f)(&x).unwrap();
    let want = grad(|x: &Array| x.multiply(x).sin().sum(..))(&x).unwrap();
    assert_eq!(bits(&g), bits(&want));
    let t = array![[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]];
    let (_, jt) = jvp_flat(
        |xs: &[Array]| Ok(vec![xs[0].copy()]),
        std::slice::from_ref(&x),
        std::slice::from_ref(&t),
    )
    .unwrap();
    assert_eq!(bits(&jt[0]), bits(&t));
    let (_, vt) = vjp_flat(
        |xs: &[Array]| Ok(vec![xs[0].copy()]),
        std::slice::from_ref(&x),
        std::slice::from_ref(&t),
    )
    .unwrap();
    assert_eq!(bits(&vt[0]), bits(&t));
    let batched = vmap_flat(
        |xs: &[Array]| Ok(vec![xs[0].copy() * 2.0]),
        &[Some(1)],
        &[0],
    )(std::slice::from_ref(&x))
    .unwrap();
    assert_eq!(bits(&batched[0]), bits(&(x.t() * 2.0)));
    // Finite differences of the gradient.
    let h = 1e-3;
    let v = x.to_vec::<f32>().unwrap();
    let gv = g.to_vec::<f32>().unwrap();
    for i in 0..v.len() {
        let at = |d: f32| {
            let mut w = v.clone();
            w[i] += d;
            f(&Array::from_slice(&w, [2, 3])).item::<f32>().unwrap()
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        assert!((fd - gv[i]).abs() < 1e-2, "{i}: {fd} vs {}", gv[i]);
    }
}

/// A tiny hybrid layer stack (Gated DeltaNet then attention over a growing KV cache),
/// prefilled in two chunks: snapshotting the KV cache and the recurrent state with `copy`
/// after the first chunk, then resuming from the snapshot, gives the uninterrupted run's
/// outputs bit for bit (the same chunk plan).
#[test]
fn resuming_from_a_snapshot_equals_the_uninterrupted_run() {
    for device in devices() {
        with_stream(device, || {
            let (hk, hv, dk, dv, heads) = (1, 2, 32, 16, 2);
            let input = |t0: i32, t: i32| {
                let n = t * 64;
                let base = ops::arange(t0 * 64, t0 * 64 + n, 1).astype(DType::F32);
                (base * 0.0137).sin().reshape([1, t, 64])
            };
            // One chunk: GDN (state) then causal attention over the cache [0, t0 + t).
            let chunk =
                |x: &Array, state: &Array, cache: &Array, t0: i32| -> (Array, Array, Array) {
                    let t = x.shape()[1];
                    let q = x.i((.., .., ..hk * dk)).reshape([1, t, hk, dk]) * 0.1;
                    let k = x.i((.., .., ..hk * dk)).reshape([1, t, hk, dk]).cos() * 0.1;
                    let v = x.i((.., .., 32..32 + hv * dv)).reshape([1, t, hv, dv]);
                    let g = x.i((.., .., ..hv)).sigmoid();
                    let beta = x.i((.., .., 2..2 + hv)).sigmoid();
                    let (y, state) = fast::gated_delta(&q, &k, &v, &g, &beta, state);
                    let h = y.reshape([1, t, heads, dv]).transpose([0, 2, 1, 3]);
                    let cache = cache.clone().at((.., .., t0..t0 + t)).set(&h);
                    let keys = cache.i((.., .., ..t0 + t));
                    let out = fast::scaled_dot_product_attention(
                        &h,
                        &keys,
                        &keys,
                        0.25,
                        &fast::AttentionOptions::causal(),
                    );
                    (out, state, cache)
                };
            let state0 = ops::zeros([1, hv, dv, dk], DType::F32);
            let cache0 = ops::zeros([1, heads, 40, dv], DType::F32);
            // Uninterrupted: chunks of 13 and 11 tokens.
            let (o1, s1, c1) = chunk(&input(0, 13), &state0, &cache0, 0);
            let (o2, _, _) = chunk(&input(13, 11), &s1, &c1, 13);
            let want = (bits(&o1), bits(&o2));
            // Interrupted after the first chunk: snapshot, drop everything, resume.
            let (o1b, s1b, c1b) = chunk(&input(0, 13), &state0, &cache0, 0);
            let (snap_state, snap_cache) = (s1b.copy(), c1b.copy());
            eval(&[&o1b, &snap_state, &snap_cache]).unwrap();
            let first = bits(&o1b);
            drop((o1b, s1b, c1b));
            let (o2b, _, _) = chunk(&input(13, 11), &snap_state, &snap_cache, 13);
            assert_eq!((first, bits(&o2b)), want, "{device:?}");
        });
    }
}
