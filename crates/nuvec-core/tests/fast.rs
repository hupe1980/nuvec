//! Fused operations (`fast`, and softmax/log-sum-exp over the last axis): the GPU kernels
//! against the fallback graphs on the CPU, including gradients and batching, and the
//! fallbacks against direct formulas.

mod common;

use common::Rng;
use nuvec_core::*;

fn random(rng: &mut Rng, shape: &[usize], dtype: DType) -> Array {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 / 4.0).collect();
    Array::from_slice(&v, shape).unwrap().astype(dtype).unwrap()
}

fn values(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).unwrap().to_vec::<f64>().unwrap()
}

fn assert_close(name: &str, got: &Array, want: &Array, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    assert_eq!(got.dtype(), want.dtype(), "{name}: dtype");
    let (g, w) = (values(got), values(want));
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert!(
            a == b || (a - b).abs() <= tol * b.abs().max(1.0),
            "{name}: element {i}: {a} vs {b}"
        );
    }
}

fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => 2e-5,
        DType::F16 => 4e-3,
        // bf16 gradients round at different points in the kernels and the fallbacks.
        _ => 5e-2,
    }
}

#[test]
fn fallbacks_match_direct_formulas() {
    let x = Array::from_slice(&[1.0f32, -2.0, 3.0, 0.5, 0.0, 4.0], &[2, 3]).unwrap();
    let w = Array::from_slice(&[1.0f32, 2.0, 0.5], &[3]).unwrap();
    let b = Array::from_slice(&[0.0f32, 1.0, -1.0], &[3]).unwrap();
    let rows = [[1.0f64, -2.0, 3.0], [0.5, 0.0, 4.0]];
    let (wv, bv) = ([1.0, 2.0, 0.5], [0.0, 1.0, -1.0]);
    let rms = values(&fast::rms_norm(&x, Some(&w), 1e-5).unwrap());
    let ln = values(&fast::layer_norm(&x, Some(&w), Some(&b), 1e-5).unwrap());
    for (r, row) in rows.iter().enumerate() {
        let ms = row.iter().map(|v| v * v).sum::<f64>() / 3.0;
        let mu = row.iter().sum::<f64>() / 3.0;
        let var = row.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / 3.0;
        for j in 0..3 {
            let want = row[j] / (ms + 1e-5).sqrt() * wv[j];
            assert!((rms[r * 3 + j] - want).abs() < 1e-5, "rms {r},{j}");
            let want = (row[j] - mu) / (var + 1e-5).sqrt() * wv[j] + bv[j];
            assert!((ln[r * 3 + j] - want).abs() < 1e-5, "layer norm {r},{j}");
        }
    }
    let targets = Array::from_slice(&[2i32, -3], &[2]).unwrap();
    let ce = values(&fast::cross_entropy(&x, &targets).unwrap());
    for (r, row) in rows.iter().enumerate() {
        let lse = row.iter().map(|v| v.exp()).sum::<f64>().ln();
        let want = lse - row[[2, 0][r]];
        assert!((ce[r] - want).abs() < 1e-5, "cross entropy {r}");
    }
    // RoPE with offset 0 leaves position 0 unchanged and rotates position 1 by θ = 1.
    let x = Array::from_slice(&[1.0f32, 2.0, 1.0, 2.0], &[1, 2, 2]).unwrap();
    let r =
        values(&fast::rope(&x, 2, false, Some(10000.0), 1.0, &Array::scalar(0i32), None).unwrap());
    let (c, s) = (1f64.cos(), 1f64.sin());
    let want = [1.0, 2.0, c - 2.0 * s, s + 2.0 * c];
    for (g, w) in r.iter().zip(want) {
        assert!((g - w).abs() < 1e-5, "{r:?} vs {want:?}");
    }
}

#[test]
fn invalid_arguments_are_rejected() {
    let x = Array::full(&[2, 4], 1.0f32).unwrap();
    assert!(fast::rms_norm(&Array::scalar(1.0f32), None, 1e-5).is_err());
    assert!(fast::rms_norm(&x, Some(&Array::full(&[3], 1.0f32).unwrap()), 1e-5).is_err());
    assert!(fast::layer_norm(&x.astype(DType::I32).unwrap(), None, None, 1e-5).is_err());
    let x3 = Array::full(&[1, 2, 4], 1.0f32).unwrap();
    let zero = Array::scalar(0i32);
    assert!(
        fast::rope(&x, 4, false, Some(1e4), 1.0, &zero, None).is_err(),
        "2-D"
    );
    assert!(
        fast::rope(&x3, 3, false, Some(1e4), 1.0, &zero, None).is_err(),
        "odd dims"
    );
    assert!(
        fast::rope(&x3, 4, false, None, 1.0, &zero, None).is_err(),
        "no base"
    );
    assert!(fast::rope(&x3, 4, false, Some(1e4), 1.0, &Array::scalar(0.5f32), None).is_err());
    assert!(fast::cross_entropy(&x, &Array::full(&[3], 0i32).unwrap()).is_err());
    assert!(fast::cross_entropy(&x, &Array::full(&[2], 0.0f32).unwrap()).is_err());
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;

    fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
        let previous = default_stream();
        set_default_stream(Stream::default_for(device));
        let r = f();
        set_default_stream(previous);
        r
    }

    fn gpu_available() -> bool {
        let ok = !cfg!(miri) && {
            let a = Array::scalar(1.0f32);
            on(Device::gpu(0), || (&a + &a).eval()).is_ok()
        };
        if !ok {
            eprintln!("skipped: no Metal device");
        }
        ok
    }

    /// `f` on both devices (the kernel on the GPU, the fallback on the CPU), compared.
    fn both(name: &str, tol: f64, f: impl Fn() -> Result<Vec<Array>>) {
        let cpu = on(Device::cpu(), &f).unwrap();
        let gpu = on(Device::gpu(0), &f).unwrap();
        assert_eq!(cpu.len(), gpu.len());
        for (i, (g, c)) in gpu.iter().zip(&cpu).enumerate() {
            assert_close(&format!("{name} #{i}"), g, c, tol);
        }
    }

    /// The forward value and the vjp of `f` for every input.
    fn with_grads(
        f: impl Fn(&[Array]) -> Result<Vec<Array>> + Copy,
        xs: &[Array],
        g: &Array,
    ) -> Result<Vec<Array>> {
        let (mut out, grads) = vjp(f, xs, std::slice::from_ref(g))?;
        out.extend(grads);
        Ok(out)
    }

    #[test]
    fn norms_match_the_fallback_with_gradients() {
        if !gpu_available() {
            return;
        }
        let mut rng = Rng::new(0x61);
        // Single-row kernels, the looped kernels (> 4096 / 6656 features) and RMS row groups.
        for shape in [
            vec![9],
            vec![3, 7],
            vec![2, 4, 64],
            vec![1100, 32],
            vec![2, 5000],
            vec![2, 7000],
        ] {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let d = shape[shape.len() - 1];
                let x = random(&mut rng, &shape, dtype);
                let w = random(&mut rng, &[d], dtype);
                let b = random(&mut rng, &[d], dtype);
                let g = random(&mut rng, &shape, dtype);
                let tol = tolerance(dtype) * if shape[0] > 1000 { 8.0 } else { 1.0 };
                let name = format!("{dtype} {shape:?}");
                both(&format!("rms {name}"), tol, || {
                    with_grads(
                        |xs| Ok(vec![fast::rms_norm(&xs[0], Some(&xs[1]), 1e-5)?]),
                        &[x.clone(), w.clone()],
                        &g,
                    )
                });
                both(&format!("rms no weight {name}"), tol, || {
                    with_grads(
                        |xs| Ok(vec![fast::rms_norm(&xs[0], None, 1e-5)?]),
                        std::slice::from_ref(&x),
                        &g,
                    )
                });
                both(&format!("layer norm {name}"), tol, || {
                    with_grads(
                        |xs| {
                            Ok(vec![fast::layer_norm(
                                &xs[0],
                                Some(&xs[1]),
                                Some(&xs[2]),
                                1e-5,
                            )?])
                        },
                        &[x.clone(), w.clone(), b.clone()],
                        &g,
                    )
                });
                both(&format!("layer norm bare {name}"), tol, || {
                    with_grads(
                        |xs| Ok(vec![fast::layer_norm(&xs[0], None, None, 1e-5)?]),
                        std::slice::from_ref(&x),
                        &g,
                    )
                });
            }
        }
        // A transposed (copied) input.
        let x = random(&mut rng, &[16, 6], DType::F32);
        both("rms strided", 2e-5, || {
            Ok(vec![fast::rms_norm(&x.t()?, None, 1e-6)?])
        });
    }

    #[test]
    fn softmax_logsumexp_and_cross_entropy_match_the_fallback() {
        if !gpu_available() {
            return;
        }
        let mut rng = Rng::new(0x62);
        for shape in [vec![5, 10], vec![2, 3, 257], vec![3, 5000], vec![1, 1, 9]] {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let x = random(&mut rng, &shape, dtype);
                let g = random(&mut rng, &shape, dtype);
                let tol = tolerance(dtype);
                let name = format!("{dtype} {shape:?}");
                both(&format!("softmax {name}"), tol, || {
                    with_grads(
                        |xs| Ok(vec![softmax(&xs[0], &[-1])?]),
                        std::slice::from_ref(&x),
                        &g,
                    )
                });
                let rows: Vec<usize> = shape[..shape.len() - 1].to_vec();
                let gr = random(&mut rng, &rows, DType::F32);
                let gl = gr.astype(dtype).unwrap();
                both(&format!("logsumexp {name}"), tol, || {
                    with_grads(
                        |xs| Ok(vec![logsumexp(&xs[0], &[-1], false)?]),
                        std::slice::from_ref(&x),
                        &gl,
                    )
                });
                let classes = shape[shape.len() - 1];
                let n: usize = rows.iter().product();
                let t: Vec<i32> = (0..n)
                    .map(|i| (i * 7 % classes) as i32 - (i % 2) as i32)
                    .collect();
                let targets = Array::from_slice(&t, &rows).unwrap();
                both(&format!("cross entropy {name}"), tol, || {
                    let f = |xs: &[Array]| Ok(vec![fast::cross_entropy(&xs[0], &targets)?]);
                    with_grads(f, std::slice::from_ref(&x), &gr)
                });
            }
        }
        // Infinite rows and an axis other than the last (the composed path).
        let x = Array::from_slice(&[f32::NEG_INFINITY, 1.0, f32::INFINITY, 0.0], &[2, 2]).unwrap();
        both("logsumexp inf", 0.0, || {
            Ok(vec![logsumexp(&x, &[-1], true)?])
        });
        let x = random(&mut rng, &[4, 3], DType::F32);
        both("softmax axis 0", 2e-5, || Ok(vec![softmax(&x, &[0])?]));
    }

    #[test]
    fn rope_matches_the_fallback() {
        if !gpu_available() {
            return;
        }
        let mut rng = Rng::new(0x63);
        // (shape, dims, traditional, offset per batch, freqs)
        let cases: &[(&[usize], usize, bool, bool, bool)] = &[
            (&[2, 3, 5, 8], 8, false, false, false),
            (&[2, 3, 5, 8], 8, true, false, false),
            (&[2, 3, 5, 16], 8, false, true, false),
            (&[2, 4, 16], 16, true, true, true),
            (&[3, 2, 1, 32], 32, false, false, false),
            (&[3, 2, 1, 32], 16, true, false, true),
            (&[2, 2, 3, 6, 8], 8, false, true, false),
        ];
        for &(shape, dims, traditional, per_batch, with_freqs) in cases {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let x = random(&mut rng, shape, dtype);
                let g = random(&mut rng, shape, dtype);
                let offset = if per_batch {
                    let o: Vec<i32> = (0..shape[0] as i32).map(|i| 3 * i + 1).collect();
                    Array::from_slice(&o, &[shape[0]]).unwrap()
                } else {
                    Array::scalar(5i32)
                };
                let freqs = with_freqs.then(|| {
                    let f: Vec<f32> = (0..dims / 2).map(|i| 1.0 + i as f32 * 3.0).collect();
                    Array::from_slice(&f, &[dims / 2]).unwrap()
                });
                let base = (!with_freqs).then_some(10000.0);
                let name = format!("{dtype} {shape:?} dims {dims} trad {traditional}");
                both(&name, tolerance(dtype) * 4.0, || {
                    let f = |xs: &[Array]| {
                        Ok(vec![fast::rope(
                            &xs[0],
                            dims,
                            traditional,
                            base,
                            0.5,
                            &offset,
                            freqs.as_ref(),
                        )?])
                    };
                    with_grads(f, std::slice::from_ref(&x), &g)
                });
            }
        }
    }

    #[test]
    fn forward_mode_and_vmap_match_the_fallback() {
        if !gpu_available() {
            return;
        }
        let mut rng = Rng::new(0x64);
        let x = random(&mut rng, &[3, 4, 32], DType::F32);
        let w = random(&mut rng, &[32], DType::F32);
        let t = random(&mut rng, &[3, 4, 32], DType::F32);
        both("rms jvp", 2e-5, || {
            let f = |xs: &[Array]| Ok(vec![fast::rms_norm(&xs[0], Some(&w), 1e-5)?]);
            Ok(jvp(f, std::slice::from_ref(&x), std::slice::from_ref(&t))?.1)
        });
        both("softmax jvp", 2e-5, || {
            Ok(jvp(
                |xs: &[Array]| Ok(vec![softmax(&xs[0], &[-1])?]),
                std::slice::from_ref(&x),
                std::slice::from_ref(&t),
            )?
            .1)
        });
        // vmap over the normalised axis and over a leading axis, and with batched weights.
        both("vmap rms last axis", 2e-5, || {
            vmap(
                |xs: &[Array]| Ok(vec![fast::rms_norm(&xs[0], None, 1e-5)?]),
                &[Some(2)],
                &[0],
            )(std::slice::from_ref(&x))
        });
        let ws = random(&mut rng, &[3, 32], DType::F32);
        both("vmap layer norm weights", 2e-5, || {
            let f = |xs: &[Array]| Ok(vec![fast::layer_norm(&xs[0], Some(&xs[1]), None, 1e-5)?]);
            vmap(f, &[Some(0), Some(0)], &[0])(&[x.clone(), ws.clone()])
        });
        let targets =
            Array::from_slice(&(0..12).map(|i| i % 32).collect::<Vec<i32>>(), &[3, 4]).unwrap();
        both("vmap cross entropy", 2e-5, || {
            let f = |xs: &[Array]| Ok(vec![fast::cross_entropy(&xs[0], &xs[1])?]);
            vmap(f, &[Some(1), Some(1)], &[0])(&[x.clone(), targets.clone()])
        });
    }
}

/// Attention by hand for one head: `softmax(scale · q kᵀ) v`.
#[test]
fn attention_fallback_matches_a_direct_formula() {
    let mut rng = Rng::new(0x65);
    let (lq, lk, d) = (3, 5, 4);
    let q = random(&mut rng, &[1, 1, lq, d], DType::F32);
    let k = random(&mut rng, &[1, 1, lk, d], DType::F32);
    let v = random(&mut rng, &[1, 1, lk, d], DType::F32);
    let (qv, kv, vv) = (values(&q), values(&k), values(&v));
    let causal =
        fast::scaled_dot_product_attention(&q, &k, &v, 0.5, fast::AttentionMask::Causal, None)
            .unwrap();
    let got = values(&causal);
    for i in 0..lq {
        let visible = i + lk - lq + 1;
        let s: Vec<f64> = (0..visible)
            .map(|j| 0.5 * (0..d).map(|c| qv[i * d + c] * kv[j * d + c]).sum::<f64>())
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
        for c in 0..d {
            let want: f64 = (0..visible)
                .map(|j| (s[j] - m).exp() / z * vv[j * d + c])
                .sum();
            assert!((got[i * d + c] - want).abs() < 1e-5, "{i},{c}");
        }
    }
    // Errors: shapes, head grouping.
    let bad = random(&mut rng, &[1, 3, lk, d], DType::F32);
    let q2 = random(&mut rng, &[1, 2, lq, d], DType::F32);
    assert!(
        fast::scaled_dot_product_attention(&q2, &bad, &bad, 1.0, fast::AttentionMask::None, None)
            .is_err()
    );
    assert!(
        fast::scaled_dot_product_attention(
            &q.reshape(&[1, lq as isize, d as isize]).unwrap(),
            &k,
            &v,
            1.0,
            fast::AttentionMask::None,
            None
        )
        .is_err()
    );
}

#[cfg(target_os = "macos")]
mod attention_gpu {
    use super::*;

    fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
        let previous = default_stream();
        set_default_stream(Stream::default_for(device));
        let r = f();
        set_default_stream(previous);
        r
    }

    #[test]
    fn attention_kernels_match_the_fallback() {
        if cfg!(miri) || on(Device::gpu(0), || (Array::scalar(1.0f32) * 2.0).eval()).is_err() {
            return;
        }
        let mut rng = Rng::new(0x66);
        // (B, Hq, Hkv, Lq, Lk, D, Dv): vector (one pass, two passes, GQA kernel) and tiled.
        let shapes: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
            (2, 4, 4, 1, 37, 64, 64),
            (1, 8, 2, 4, 50, 128, 128),
            (1, 4, 4, 2, 33, 96, 64),
            (1, 2, 1, 1, 20, 192, 128),
            (1, 8, 1, 1, 1100, 128, 128),
            (1, 16, 2, 1, 9000, 128, 128),
            (2, 4, 2, 33, 45, 64, 64),
            (1, 2, 2, 100, 100, 80, 80),
            (1, 4, 4, 64, 96, 128, 128),
            (1, 2, 2, 40, 40, 96, 64),
        ];
        for &(b, hq, hkv, lq, lk, d, dv) in shapes {
            // Every shape here has a kernel: the recorded node is the fused primitive.
            let fused = on(Device::gpu(0), || {
                let z = |l, n, e| zeros(&[b, n, l, e], DType::F16).unwrap();
                let out = fast::scaled_dot_product_attention(
                    &z(lq, hq, d),
                    &z(lk, hkv, d),
                    &z(lk, hkv, dv),
                    1.0,
                    fast::AttentionMask::None,
                    None,
                )
                .unwrap();
                matches!(
                    out.primitive(),
                    Some(Primitive::Fast(fast::FastOp::Attention { .. }))
                )
            });
            assert!(
                fused,
                "{:?} should use a kernel",
                (b, hq, hkv, lq, lk, d, dv)
            );
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let q = random(&mut rng, &[b, hq, lq, d], dtype);
                // Keys and values as transposed views of [B, L, H, D] (as in a model).
                let k =
                    transpose(&random(&mut rng, &[b, lk, hkv, d], dtype), &[0, 2, 1, 3]).unwrap();
                let v =
                    transpose(&random(&mut rng, &[b, lk, hkv, dv], dtype), &[0, 2, 1, 3]).unwrap();
                let bmask = greater(random(&mut rng, &[lq, lk], DType::F32), -0.5).unwrap();
                let amask = random(&mut rng, &[1, hq, lq, lk], dtype);
                let sinks = random(&mut rng, &[hq], dtype);
                let tol = match dtype {
                    DType::F32 => 1e-4,
                    DType::F16 => 1e-2,
                    _ => 5e-2,
                };
                let masks: [(&str, fast::AttentionMask<'_>); 4] = [
                    ("none", fast::AttentionMask::None),
                    ("causal", fast::AttentionMask::Causal),
                    ("bool", fast::AttentionMask::Array(&bmask)),
                    ("additive", fast::AttentionMask::Array(&amask)),
                ];
                for (mname, mask) in masks {
                    for with_sinks in [false, true] {
                        let s = with_sinks.then_some(&sinks);
                        let run = || {
                            fast::scaled_dot_product_attention(&q, &k, &v, 0.3, mask, s).unwrap()
                        };
                        // The reference: the composition in float32 on the CPU, rounded once.
                        let reference = || {
                            let f = |a: &Array| a.astype(DType::F32).unwrap();
                            let amask32;
                            let mask32 = match mask {
                                fast::AttentionMask::Array(m) if m.dtype() != DType::Bool => {
                                    amask32 = f(m);
                                    fast::AttentionMask::Array(&amask32)
                                }
                                other => other,
                            };
                            let s32 = s.map(f);
                            fast::scaled_dot_product_attention(
                                &f(&q),
                                &f(&k),
                                &f(&v),
                                0.3,
                                mask32,
                                s32.as_ref(),
                            )
                            .unwrap()
                            .astype(dtype)
                            .unwrap()
                        };
                        let (gpu, cpu) = (on(Device::gpu(0), run), on(Device::cpu(), reference));
                        let name = format!(
                            "{dtype} {:?} {mname} sinks={with_sinks}",
                            (b, hq, hkv, lq, lk, d, dv)
                        );
                        assert_close(&name, &gpu, &cpu, tol);
                    }
                }
            }
        }
        // Gradients through the fused forward match the CPU.
        let q = random(&mut rng, &[1, 4, 40, 64], DType::F32);
        let k = random(&mut rng, &[1, 2, 40, 64], DType::F32);
        let v = random(&mut rng, &[1, 2, 40, 64], DType::F32);
        let g = random(&mut rng, &[1, 4, 40, 64], DType::F32);
        let grads = || {
            let f = |xs: &[Array]| {
                Ok(vec![fast::scaled_dot_product_attention(
                    &xs[0],
                    &xs[1],
                    &xs[2],
                    0.125,
                    fast::AttentionMask::Causal,
                    None,
                )?])
            };
            vjp(
                f,
                &[q.clone(), k.clone(), v.clone()],
                std::slice::from_ref(&g),
            )
            .unwrap()
            .1
        };
        let (gpu, cpu) = (on(Device::gpu(0), grads), on(Device::cpu(), grads));
        for (i, (a, c)) in gpu.iter().zip(&cpu).enumerate() {
            assert_close(&format!("attention vjp #{i}"), a, c, 1e-4);
        }
    }
}
