//! `fast::linear_cross_entropy` against the unchunked `cross_entropy(head(h), t)`: values
//! and gradients for dense, quantized and soft-capped heads, chunk edges, dtypes and
//! devices; and its peak memory. A separate test binary, so that no other test allocates
//! while peak memory is measured.

mod common;

use common::Rng;
use nuvec_core::*;

fn random(rng: &mut Rng, shape: &[usize], scale: f32, dtype: DType) -> Array {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 * scale).collect();
    Array::from_slice(&v, shape).astype(dtype)
}

fn targets(rng: &mut Rng, n: usize, vocab: usize) -> Array {
    let v: Vec<i32> = (0..n)
        .map(|_| (rng.f64().abs() * 1e6) as i32 % vocab as i32)
        .collect();
    Array::from_slice(&v, [n])
}

fn assert_close(name: &str, got: &Array, want: &Array, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    assert_eq!(got.dtype(), want.dtype(), "{name}: dtype");
    let g = got.astype(DType::F64).to_vec::<f64>().unwrap();
    let w = want.astype(DType::F64).to_vec::<f64>().unwrap();
    let scale = w.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-3);
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert!(
            (a - b).abs() <= tol * scale,
            "{name}: element {i}: {a} vs {b}"
        );
    }
}

fn dense(x: &Array, w: &[Array]) -> Array {
    x.matmul(&w[0].t())
}

/// Per-token losses as `nn.losses.cross_entropy`: the fused loss cast to the logits' dtype.
fn unchunked(h: &Array, t: &Array, w: &[Array], head: fn(&Array, &[Array]) -> Array) -> Array {
    let z = head(h, w);
    fast::cross_entropy(&z, t).astype(z.dtype())
}

/// The losses and the gradients of `Σ c·loss` (a random cotangent) with respect to `h` and
/// the weights, chunked and not.
fn both(
    h: &Array,
    t: &Array,
    w: &[Array],
    head: fn(&Array, &[Array]) -> Array,
    rows: Option<usize>,
    cot: &Array,
) -> [Vec<Array>; 2] {
    let run = |chunked: bool| {
        let t = t.clone();
        let mut primals = vec![h.clone()];
        primals.extend(w.iter().cloned());
        let (out, grads) = vjp_flat(
            move |xs| {
                let l = match chunked {
                    true => fast::linear_cross_entropy(&xs[0], &t, &xs[1..], head, rows),
                    false => unchunked(&xs[0], &t, &xs[1..], head),
                };
                Ok(vec![l])
            },
            &primals,
            std::slice::from_ref(cot),
        )
        .unwrap();
        let mut v = out;
        v.extend(grads);
        v
    };
    [run(true), run(false)]
}

fn check(
    name: &str,
    dtype: DType,
    gpu: bool,
    n: usize,
    d: usize,
    vocab: usize,
    rows: Option<usize>,
) {
    let mut rng = Rng::new(7);
    let h = random(&mut rng, &[n, d], 0.5, dtype);
    let w = random(&mut rng, &[vocab, d], 0.5, dtype);
    let t = targets(&mut rng, n, vocab);
    let cot = random(&mut rng, &[n], 1.0, dtype);
    let device = if gpu { Device::gpu(0) } else { Device::cpu() };
    let [got, want] = with_stream(device, || both(&h, &t, &[w], dense, rows, &cot));
    let tol = if dtype == DType::F32 { 2e-5 } else { 2e-2 };
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_close(&format!("{name} output {i}"), g, w, tol);
    }
}

#[test]
fn dense_head_equals_the_unchunked_loss_on_the_cpu() {
    for rows in [None, Some(1), Some(3), Some(4), Some(7), Some(100)] {
        check(&format!("rows {rows:?}"), DType::F32, false, 7, 5, 11, rows);
    }
    check("no rows", DType::F32, false, 0, 5, 11, Some(4));
}

#[test]
fn dense_head_equals_the_unchunked_loss_on_the_gpu() {
    if common::require_gpu().is_none() {
        return;
    }
    for dtype in [DType::F32, DType::BF16] {
        for rows in [Some(64), Some(100), None] {
            check(
                &format!("{dtype:?} rows {rows:?}"),
                dtype,
                true,
                300,
                64,
                1000,
                rows,
            );
        }
    }
}

#[test]
fn quantized_and_soft_capped_heads_equal_the_unchunked_loss() {
    if common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(3);
    let (n, d, vocab) = (130, 128, 700);
    let h = random(&mut rng, &[n, d], 0.5, DType::BF16);
    let q = ops::quantize(
        &random(&mut rng, &[vocab, d], 0.5, DType::BF16),
        Quant::affine(64, 4),
    );
    let biases = q.biases.clone().unwrap();
    let ws = [q.weight.clone(), q.scales.clone(), biases];
    let t = targets(&mut rng, n, vocab);
    let cot = random(&mut rng, &[n], 1.0, DType::BF16);
    fn tied(x: &Array, w: &[Array]) -> Array {
        let q = Quantized::affine(w[0].clone(), w[1].clone(), w[2].clone(), 64, 4);
        x.quantized_matmul(&q, true)
    }
    fn capped(x: &Array, w: &[Array]) -> Array {
        (x.matmul(&w[0].t()) / 30.0).tanh() * 30.0
    }
    with_stream(Device::gpu(0), || {
        // Only the hidden states are differentiated: a quantized head is frozen.
        let run = |chunked: bool| {
            let (t, ws) = (t.clone(), ws.clone());
            vjp_flat(
                move |xs| {
                    Ok(vec![match chunked {
                        true => fast::linear_cross_entropy(&xs[0], &t, &ws, tied, Some(48)),
                        false => unchunked(&xs[0], &t, &ws, tied),
                    }])
                },
                std::slice::from_ref(&h),
                std::slice::from_ref(&cot),
            )
            .unwrap()
        };
        let ((l1, g1), (l2, g2)) = (run(true), run(false));
        assert_close("quantized losses", &l1[0], &l2[0], 2e-2);
        assert_close("quantized dh", &g1[0], &g2[0], 2e-2);
        let w = random(&mut rng, &[vocab, d], 0.5, DType::BF16);
        let [got, want] = both(&h, &t, &[w], capped, Some(48), &cot);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_close(&format!("soft-capped output {i}"), g, w, 2e-2);
        }
    });
}

#[test]
fn non_contiguous_hidden_states_and_batched_shapes() {
    let mut rng = Rng::new(5);
    let h = random(&mut rng, &[6, 3, 4], 0.5, DType::F32).transpose([1, 0, 2]); // [3, 6, 4]
    let w = random(&mut rng, &[9, 4], 0.5, DType::F32);
    let t = targets(&mut rng, 18, 9).reshape([3, 6]);
    let cot = random(&mut rng, &[3, 6], 1.0, DType::F32);
    with_stream(Device::cpu(), || {
        let [got, want] = both(&h, &t, &[w], dense, Some(5), &cot);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_close(&format!("output {i}"), g, w, 2e-5);
        }
    });
}

#[test]
fn hidden_state_gradient_matches_finite_differences() {
    let mut rng = Rng::new(11);
    let (n, d, vocab) = (5, 3, 7);
    let h = random(&mut rng, &[n, d], 0.5, DType::F32);
    let w = random(&mut rng, &[vocab, d], 0.5, DType::F32);
    let t = targets(&mut rng, n, vocab);
    let loss = |h: &Array| {
        fast::linear_cross_entropy(h, &t, std::slice::from_ref(&w), dense, Some(2))
            .sum(..)
            .item::<f32>()
            .unwrap() as f64
    };
    let g = {
        let (t, w) = (t.clone(), w.clone());
        grad(move |h: &Array| {
            Ok(fast::linear_cross_entropy(h, &t, std::slice::from_ref(&w), dense, Some(2)).sum(..))
        })(&h)
        .unwrap()
        .to_vec::<f32>()
        .unwrap()
    };
    let base = h.to_vec::<f32>().unwrap();
    let eps = 1e-2f32;
    for i in 0..base.len() {
        let at = |delta: f32| {
            let mut v = base.clone();
            v[i] += delta;
            loss(&Array::from_slice(&v, [n, d]))
        };
        let fd = (at(eps) - at(-eps)) / (2.0 * eps as f64);
        assert!(
            (fd - g[i] as f64).abs() < 2e-3,
            "element {i}: {fd} vs {}",
            g[i]
        );
    }
}

#[test]
fn outside_a_gradient_transform_no_gradient_is_built() {
    let mut rng = Rng::new(2);
    let h = random(&mut rng, &[10, 4], 0.5, DType::F32);
    let w = random(&mut rng, &[6, 4], 0.5, DType::F32);
    let t = targets(&mut rng, 10, 6);
    let l = fast::linear_cross_entropy(&h, &t, &[w], dense, Some(3));
    // The graph reaches only forward cross-entropy nodes.
    let mut stack = vec![l];
    let mut seen = std::collections::HashSet::new();
    while let Some(a) = stack.pop() {
        if !seen.insert(graph::id(&a)) {
            continue;
        }
        if let Some(p) = graph::primitive(&a) {
            let name = format!("{p:?}");
            assert!(!name.contains("CrossEntropyVjp"), "a backward node: {name}");
        }
        stack.extend(graph::inputs(&a));
    }
}

#[test]
fn invalid_arguments_are_errors() {
    let h = Array::from_slice(&[0.0f32; 8], [2, 4]);
    let w = Array::from_slice(&[0.0f32; 12], [3, 4]);
    let t = Array::from_slice(&[0i32, 1], [2]);
    let zero = fast::linear_cross_entropy(&h, &t, std::slice::from_ref(&w), dense, Some(0));
    assert!(zero.check().is_err());
    let wrong = Array::from_slice(&[0i32, 1, 2], [3]);
    assert!(
        fast::linear_cross_entropy(&h, &wrong, std::slice::from_ref(&w), dense, None)
            .check()
            .is_err()
    );
    let float = Array::from_slice(&[0.0f32, 1.0], [2]);
    assert!(
        fast::linear_cross_entropy(&h, &float, &[w], dense, None)
            .check()
            .is_err()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn peak_memory_is_a_fraction_of_the_unchunked_loss() {
    if common::require_gpu().is_none() {
        return;
    }
    let peak = || {
        nuvec_gpu::Device::shared().map_or(0, |d| {
            let p = d.memory().peak;
            d.reset_peak();
            p
        })
    };
    let active = || nuvec_gpu::Device::shared().map_or(0, |d| d.memory().active);
    let (n, d, vocab) = (4096, 512, 32768);
    with_stream(Device::gpu(0), || {
        let h = random::normal([n, d], DType::BF16, 0.0, 1.0, None);
        let w = random::normal([vocab, d], DType::BF16, 0.0, 0.1, None);
        let t = ops::arange(0, n, 1).astype(DType::I32);
        let cot = ops::ones([n], DType::BF16);
        eval(&[&h, &w, &t, &cot]).unwrap();
        let above = |chunked: bool| {
            let (t, w) = (t.clone(), w.clone());
            peak();
            let base = active();
            let (_, g) = vjp_flat(
                move |xs| {
                    Ok(vec![match chunked {
                        // 16 chunks: about one chunk's logits and their gradient alive.
                        true => fast::linear_cross_entropy(
                            &xs[0],
                            &t,
                            std::slice::from_ref(&w),
                            dense,
                            Some(256),
                        ),
                        false => unchunked(&xs[0], &t, std::slice::from_ref(&w), dense),
                    }])
                },
                std::slice::from_ref(&h),
                std::slice::from_ref(&cot),
            )
            .unwrap();
            g[0].eval().unwrap();
            peak().saturating_sub(base)
        };
        let (full, chunked) = (above(false), above(true));
        assert!(
            chunked * 100 <= full * 15,
            "chunked {chunked} bytes above the inputs, unchunked {full}"
        );
    });
}

#[test]
fn vmap_over_a_leading_axis_equals_each_item() {
    let mut rng = Rng::new(9);
    let h = random(&mut rng, &[3, 5, 4], 0.5, DType::F32);
    let w = random(&mut rng, &[7, 4], 0.5, DType::F32);
    let t = targets(&mut rng, 15, 7).reshape([3, 5]);
    let wc = w.clone();
    let f = vmap_flat(
        move |xs| {
            Ok(vec![fast::linear_cross_entropy(
                &xs[0],
                &xs[1],
                std::slice::from_ref(&wc),
                dense,
                Some(2),
            )])
        },
        &[Some(0), Some(0)],
        &[0],
    );
    let got = f(&[h.clone(), t.clone()]).unwrap().remove(0);
    let want = unchunked(&h.reshape([15, 4]), &t.reshape([15]), &[w], dense).reshape([3, 5]);
    assert_close("vmap", &got, &want, 2e-5);
}
