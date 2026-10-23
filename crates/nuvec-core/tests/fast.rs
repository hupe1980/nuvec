//! Fused operations (`fast`, and softmax/log-sum-exp over the last axis): the GPU kernels
//! against the fallback graphs on the CPU, including gradients and batching, and the
//! fallbacks against direct formulas.

mod common;

use common::Rng;
use nuvec_core::ops;
use nuvec_core::*;

fn random(rng: &mut Rng, shape: &[usize], dtype: DType) -> Array {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 / 4.0).collect();
    Array::from_slice(&v, shape).astype(dtype)
}

fn values(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
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

#[cfg(target_os = "macos")]
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
    let x = array![[1.0, -2.0, 3.0], [0.5, 0.0, 4.0]];
    let w = array![1.0, 2.0, 0.5];
    let b = array![0.0, 1.0, -1.0];
    let rows = [[1.0f64, -2.0, 3.0], [0.5, 0.0, 4.0]];
    let (wv, bv) = ([1.0, 2.0, 0.5], [0.0, 1.0, -1.0]);
    let rms = values(&fast::rms_norm(&x, Some(&w), 1e-5));
    let ln = values(&fast::layer_norm(&x, Some(&w), Some(&b), 1e-5));
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
    let targets = array![2, -3];
    let ce = values(&fast::cross_entropy(&x, &targets));
    for (r, row) in rows.iter().enumerate() {
        let lse = row.iter().map(|v| v.exp()).sum::<f64>().ln();
        let want = lse - row[[2, 0][r]];
        assert!((ce[r] - want).abs() < 1e-5, "cross entropy {r}");
    }
    // RoPE with offset 0 leaves position 0 unchanged and rotates position 1 by θ = 1.
    let x = array![[[1.0, 2.0], [1.0, 2.0]]];
    let r = values(&fast::rope(
        &x,
        2,
        false,
        Some(10000.0),
        1.0,
        &Array::from(0i32),
        None,
    ));
    let (c, s) = (1f64.cos(), 1f64.sin());
    let want = [1.0, 2.0, c - 2.0 * s, s + 2.0 * c];
    for (g, w) in r.iter().zip(want) {
        assert!((g - w).abs() < 1e-5, "{r:?} vs {want:?}");
    }
}

#[test]
fn invalid_arguments_are_rejected() {
    let x = ops::full([2, 4], 1.0f32);
    assert!(
        fast::rms_norm(&Array::from(1.0f32), None, 1e-5)
            .check()
            .is_err()
    );
    assert!(
        fast::rms_norm(&x, Some(&ops::full([3], 1.0f32)), 1e-5)
            .check()
            .is_err()
    );
    assert!(
        fast::layer_norm(&x.astype(DType::I32), None, None, 1e-5)
            .check()
            .is_err()
    );
    let x3 = ops::full([1, 2, 4], 1.0f32);
    let zero = Array::from(0i32);
    assert!(
        fast::rope(&x, 4, false, Some(1e4), 1.0, &zero, None)
            .check()
            .is_err(),
        "2-D"
    );
    assert!(
        fast::rope(&x3, 3, false, Some(1e4), 1.0, &zero, None)
            .check()
            .is_err(),
        "odd dims"
    );
    assert!(
        fast::rope(&x3, 4, false, None, 1.0, &zero, None)
            .check()
            .is_err(),
        "no base"
    );
    assert!(
        fast::rope(&x3, 4, false, Some(1e4), 1.0, &Array::from(0.5f32), None)
            .check()
            .is_err()
    );
    assert!(
        fast::cross_entropy(&x, &ops::full([3], 0i32))
            .check()
            .is_err()
    );
    assert!(
        fast::cross_entropy(&x, &ops::full([2], 0.0f32))
            .check()
            .is_err()
    );
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;

    fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
        with_stream(device, f)
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
        let (mut out, grads) = vjp_flat(f, xs, std::slice::from_ref(g))?;
        out.extend(grads);
        Ok(out)
    }

    #[test]
    fn norms_match_the_fallback_with_gradients() {
        if crate::common::require_gpu().is_none() {
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
                        |xs| Ok(vec![fast::rms_norm(&xs[0], Some(&xs[1]), 1e-5)]),
                        &[x.clone(), w.clone()],
                        &g,
                    )
                });
                both(&format!("rms no weight {name}"), tol, || {
                    with_grads(
                        |xs| Ok(vec![fast::rms_norm(&xs[0], None, 1e-5)]),
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
                            )])
                        },
                        &[x.clone(), w.clone(), b.clone()],
                        &g,
                    )
                });
                both(&format!("layer norm bare {name}"), tol, || {
                    with_grads(
                        |xs| Ok(vec![fast::layer_norm(&xs[0], None, None, 1e-5)]),
                        std::slice::from_ref(&x),
                        &g,
                    )
                });
            }
        }
        // A transposed (copied) input.
        let x = random(&mut rng, &[16, 6], DType::F32);
        both("rms strided", 2e-5, || {
            Ok(vec![fast::rms_norm(&x.t(), None, 1e-6)])
        });
    }

    #[test]
    fn add_rms_norm_equals_add_then_rms_norm() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x67);
        let composed = |x: &Array, r: &Array, w: Option<&Array>| -> Result<Vec<Array>> {
            let h = x.add(r);
            let y = fast::rms_norm(&h, w, 1e-5);
            Ok(vec![h, y])
        };
        let fused = |x: &Array, r: &Array, w: Option<&Array>| -> Result<Vec<Array>> {
            let (h, y) = fast::add_rms_norm(x, r, w, 1e-5);
            Ok(vec![h, y])
        };
        // Single-row kernel, looped kernel (> 4096 features), many rows.
        for shape in [
            vec![9],
            vec![3, 7],
            vec![2, 1, 1024],
            vec![700, 64],
            vec![2, 5000],
        ] {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let d = shape[shape.len() - 1];
                let x = random(&mut rng, &shape, dtype);
                let r = random(&mut rng, &shape, dtype);
                let w = random(&mut rng, &[d], dtype);
                let name = format!("{dtype} {shape:?}");
                // The kernel against the unfused kernels: bit for bit.
                on(Device::gpu(0), || {
                    for w in [Some(&w), None] {
                        let (got, want) = (fused(&x, &r, w).unwrap(), composed(&x, &r, w).unwrap());
                        for (i, (g, c)) in got.iter().zip(&want).enumerate() {
                            assert_close(&format!("add_rms {name} #{i}"), g, c, 0.0);
                        }
                    }
                });
                // The kernel against the CPU fallback, with gradients through both outputs.
                let (gh, gy) = (
                    random(&mut rng, &shape, dtype),
                    random(&mut rng, &shape, dtype),
                );
                let tol = tolerance(dtype) * if shape[0] > 500 { 8.0 } else { 1.0 };
                both(&format!("add_rms grads {name}"), tol, || {
                    let f = |xs: &[Array]| fused(&xs[0], &xs[1], Some(&xs[2]));
                    let primals = [x.clone(), r.clone(), w.clone()];
                    let (mut out, grads) = vjp_flat(f, &primals, &[gh.clone(), gy.clone()])?;
                    out.extend(grads);
                    Ok(out)
                });
            }
        }
        // Strided and offset operands (a transpose and a slice) are packed first.
        let x = random(&mut rng, &[16, 6], DType::F32).t();
        let r = random(&mut rng, &[8, 20], DType::F32).i((1..7, 2..18));
        on(Device::gpu(0), || {
            let (got, want) = (
                fused(&x, &r, None).unwrap(),
                composed(&x, &r, None).unwrap(),
            );
            for (g, c) in got.iter().zip(&want) {
                assert_close("add_rms strided", g, c, 0.0);
            }
        });
        // A weight that widens the sum's dtype: the composed operations.
        let (x, r) = (
            random(&mut rng, &[2, 8], DType::BF16),
            random(&mut rng, &[2, 8], DType::BF16),
        );
        let w = random(&mut rng, &[8], DType::F32);
        let (got, want) = (
            fused(&x, &r, Some(&w)).unwrap(),
            composed(&x, &r, Some(&w)).unwrap(),
        );
        assert_eq!((got[0].dtype(), got[1].dtype()), (DType::BF16, DType::F32));
        for (g, c) in got.iter().zip(&want) {
            assert_close("add_rms widened", g, c, 0.0);
        }
        // Forward mode and batching (the fallback defines them).
        let (x, r) = (
            random(&mut rng, &[3, 4, 32], DType::F32),
            random(&mut rng, &[3, 4, 32], DType::F32),
        );
        let w = random(&mut rng, &[32], DType::F32);
        let (tx, tr) = (
            random(&mut rng, &[3, 4, 32], DType::F32),
            random(&mut rng, &[3, 4, 32], DType::F32),
        );
        let (_, got) = jvp_flat(
            |xs| fused(&xs[0], &xs[1], Some(&w)),
            &[x.clone(), r.clone()],
            &[tx.clone(), tr.clone()],
        )
        .unwrap();
        let (_, want) = jvp_flat(
            |xs| composed(&xs[0], &xs[1], Some(&w)),
            &[x.clone(), r.clone()],
            &[tx, tr],
        )
        .unwrap();
        for (g, c) in got.iter().zip(&want) {
            assert_close("add_rms jvp", g, c, 1e-6);
        }
        // `x` batched along its middle axis, a residual shared by every example.
        let r = random(&mut rng, &[3, 32], DType::F32);
        let got = vmap_flat(
            |xs| fused(&xs[0], &xs[1], Some(&w)),
            &[Some(1), None],
            &[0, 0],
        )(&[x.clone(), r.clone()])
        .unwrap();
        let want = vmap_flat(
            |xs| composed(&xs[0], &xs[1], Some(&w)),
            &[Some(1), None],
            &[0, 0],
        )(&[x, r])
        .unwrap();
        for (g, c) in got.iter().zip(&want) {
            assert_close("add_rms vmap", g, c, 1e-6);
        }
    }

    /// `norm_rope` composed of its parts.
    fn norm_rope_composed(
        x: &Array,
        norm: Option<(&Array, f32)>,
        rotation: fast::Rotation<'_>,
        offset: usize,
        cache: Option<&Array>,
    ) -> Array {
        let x = match norm {
            Some((w, eps)) => fast::rms_norm(x, Some(w), eps),
            None => x.clone(),
        };
        let r = fast::rope(
            &x,
            rotation.dims,
            rotation.traditional,
            rotation.base,
            rotation.scale,
            &Array::from(offset as i32),
            rotation.freqs,
        );
        match cache {
            Some(c) => c.slice_update(&r, (.., .., offset..offset + r.shape()[2] as usize)),
            None => r,
        }
    }

    #[test]
    fn norm_rope_equals_rms_norm_rope_and_slice_update() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x68);
        let freqs: Array = (0..32).map(|i| 1.5f32 + i as f32 * 0.75).collect();
        // (batch, heads, length, head size, rotated dims, offset, cache capacity)
        let cases = [
            (1, 4, 1, 128, 128, 0, 256),
            (1, 2, 1, 128, 128, 37, 256),
            (2, 3, 5, 64, 64, 9, 16),
            (1, 2, 3, 80, 64, 4, 8),
            (1, 1, 2, 256, 64, 100, 102),
        ];
        for (b, h, l, d, dims, offset, capacity) in cases {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                // The projection's layout, `[B, L, H, D]`, transposed to `[B, H, L, D]`.
                let x = random(&mut rng, &[b, l, h, d], dtype).transpose([0, 2, 1, 3]);
                let w = random(&mut rng, &[d], dtype);
                let cache = random(&mut rng, &[b, h, capacity, d], dtype);
                for (normed, traditional, with_freqs, cached) in [
                    (true, false, false, false),
                    (true, false, false, true),
                    (false, false, false, true),
                    (true, true, false, true),
                    (false, true, false, false),
                    (true, false, true, true),
                ] {
                    if with_freqs && dims != 64 {
                        continue;
                    }
                    let rotation = fast::Rotation {
                        dims,
                        traditional,
                        base: (!with_freqs).then_some(10000.0),
                        scale: 1.0,
                        freqs: with_freqs.then_some(&freqs),
                    };
                    let norm = normed.then_some((&w, 1e-6));
                    let cache = cached.then_some(&cache);
                    let name = format!(
                        "{dtype} {:?} norm {normed} trad {traditional} freqs {with_freqs} cache {cached}",
                        (b, h, l, d, dims, offset)
                    );
                    on(Device::gpu(0), || {
                        let got = fast::norm_rope(&x, norm, rotation, offset as i32, cache);
                        let want = norm_rope_composed(&x, norm, rotation, offset, cache);
                        assert_close(&name, &got, &want, 0.0);
                    });
                }
                // The kernel against the CPU fallback, with gradients for `x`, the weight and
                // the cache.
                let g = random(&mut rng, &[b, h, capacity, d], dtype);
                let rotation = fast::Rotation {
                    dims,
                    traditional: false,
                    base: Some(10000.0),
                    scale: 0.5,
                    freqs: None,
                };
                // The weight's gradient sums every row, rounded to the dtype after the inverse
                // rotation: looser than a single normalisation's.
                let tol = tolerance(dtype) * 4.0;
                both(
                    &format!("norm_rope grads {dtype} {:?}", (b, h, l, d)),
                    tol,
                    || {
                        let f = |xs: &[Array]| {
                            let n = Some((&xs[1], 1e-6));
                            Ok(vec![fast::norm_rope(
                                &xs[0],
                                n,
                                rotation,
                                offset as i32,
                                Some(&xs[2]),
                            )])
                        };
                        with_grads(f, &[x.clone(), w.clone(), cache.clone()], &g)
                    },
                );
            }
        }
    }

    #[test]
    fn norm_rope_writes_an_owned_cache_in_place_and_copies_a_shared_one() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x69);
        let rotation = fast::Rotation {
            dims: 64,
            traditional: false,
            base: Some(10000.0),
            scale: 1.0,
            freqs: None,
        };
        let w = random(&mut rng, &[64], DType::BF16);
        let steps: Vec<Array> = (0..6)
            .map(|_| random(&mut rng, &[1, 2, 1, 64], DType::BF16))
            .collect();
        let start = random(&mut rng, &[1, 2, 8, 64], DType::BF16);
        // Decoding: each step's cache is the only handle to the previous one (donated).
        let decode = |fused: bool| -> Array {
            let mut cache = start.clone();
            cache.eval().unwrap();
            let mut kept = Vec::new();
            for (i, x) in steps.iter().enumerate() {
                let next = if fused {
                    fast::norm_rope(x, Some((&w, 1e-6)), rotation, i as i32, Some(&cache))
                } else {
                    norm_rope_composed(x, Some((&w, 1e-6)), rotation, i, Some(&cache))
                };
                // Every other step keeps the previous cache alive: it must not change.
                if i % 2 == 1 {
                    kept.push((cache.clone(), values(&cache)));
                }
                cache = next;
                async_eval(std::slice::from_ref(&cache)).unwrap();
            }
            for (old, before) in kept {
                assert_eq!(values(&old), before, "a shared cache was written");
            }
            cache
        };
        on(Device::gpu(0), || {
            assert_close("decode", &decode(true), &decode(false), 0.0);
        });
        // Invalid caches are rejected.
        let x = random(&mut rng, &[1, 2, 3, 64], DType::F32);
        for bad in [
            &[1, 2, 4, 64][..],
            &[1, 3, 8, 64],
            &[1, 2, 8, 32],
            &[2, 8, 64],
        ] {
            let c = ops::zeros(bad, DType::F32);
            assert!(
                fast::norm_rope(&x, None, rotation, 2, Some(&c))
                    .check()
                    .is_err(),
                "{bad:?}"
            );
        }
        // Forward mode and batching through the fallback.
        let (x, t) = (
            random(&mut rng, &[2, 2, 3, 64], DType::F32),
            random(&mut rng, &[2, 2, 3, 64], DType::F32),
        );
        let w = random(&mut rng, &[64], DType::F32);
        let cache = random(&mut rng, &[2, 2, 8, 64], DType::F32);
        let fused = |xs: &[Array]| {
            Ok(vec![fast::norm_rope(
                &xs[0],
                Some((&w, 1e-6)),
                rotation,
                4,
                Some(&cache),
            )])
        };
        let composed = |xs: &[Array]| {
            Ok(vec![norm_rope_composed(
                &xs[0],
                Some((&w, 1e-6)),
                rotation,
                4,
                Some(&cache),
            )])
        };
        let got = jvp_flat(fused, std::slice::from_ref(&x), std::slice::from_ref(&t)).unwrap();
        let want = jvp_flat(composed, std::slice::from_ref(&x), std::slice::from_ref(&t)).unwrap();
        assert_close("norm_rope jvp", &got.1[0], &want.1[0], 1e-6);
        let xs = random(&mut rng, &[3, 2, 2, 3, 64], DType::F32);
        let got = vmap_flat(fused, &[Some(0)], &[0])(std::slice::from_ref(&xs)).unwrap();
        let want = vmap_flat(composed, &[Some(0)], &[0])(std::slice::from_ref(&xs)).unwrap();
        assert_close("norm_rope vmap", &got[0], &want[0], 1e-6);
    }

    #[test]
    fn depthwise_conv1d_matches_the_fallback() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x6a);
        // (batch, length, channels, kernel, stride, padding, dilation)
        let cases = [
            (1, 964, 96, 4, 1, 0, 1),
            (2, 17, 5, 3, 1, 2, 1),
            (1, 30, 8, 3, 2, 1, 2),
            (3, 9, 1, 9, 1, 4, 1),
        ];
        for (n, l, c, k, stride, padding, dilation) in cases {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let x = random(&mut rng, &[n, l, c], dtype);
                let w = random(&mut rng, &[c, k, 1], dtype);
                let name = format!(
                    "conv1d {dtype} {:?}",
                    (n, l, c, k, stride, padding, dilation)
                );
                let f = |xs: &[Array]| {
                    Ok(vec![xs[0].conv1d(
                        &xs[1],
                        stride as i32,
                        padding as i32,
                        dilation as i32,
                        c as i32,
                    )])
                };
                // The kernel against its fallback, bit for bit.
                both(&name, 0.0, || f(&[x.clone(), w.clone()]));
                let l_out = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
                let g = random(&mut rng, &[n, l_out, c], dtype);
                both(&format!("{name} grads"), tolerance(dtype), || {
                    with_grads(f, &[x.clone(), w.clone()], &g)
                });
            }
        }
        // A strided input (a transpose) and mixed dtypes (promoted).
        let x = random(&mut rng, &[1, 6, 20], DType::BF16);
        let x = x.transpose([0, 2, 1]);
        let w = random(&mut rng, &[6, 3, 1], DType::F32);
        both("conv1d strided mixed", 0.0, || {
            Ok(vec![x.conv1d(&w, 1, 1, 1, 6)])
        });
    }

    /// Paged caches holding `keys` and `values` (`[B, Hkv, N, D]`) in blocks of `bs`
    /// positions placed in a shuffled order (with a few unused blocks), and the block table.
    fn paged(keys: &Array, values: &Array, bs: usize, rng: &mut Rng) -> (Array, Array, Array) {
        let [b, hkv, n, d] = keys.shape_n::<4>().unwrap().map(|d| d as usize);
        let per = n.div_ceil(bs);
        let blocks = b * per + 3;
        let mut order: Vec<usize> = (0..blocks).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, (rng.f64().abs() * 1e6) as usize % (i + 1));
        }
        let mut caches = [keys, values].map(|_| ops::zeros([blocks, hkv, bs, d], keys.dtype()));
        for s in 0..b {
            for j in 0..per {
                let (block, len) = (order[s * per + j], bs.min(n - j * bs));
                for (cache, src) in caches.iter_mut().zip([keys, values]) {
                    let part = src.i((s..s + 1, .., j * bs..j * bs + len, ..));
                    *cache = cache.slice_update(&part, (block..block + 1, .., 0..len, ..));
                }
            }
        }
        let table: Vec<i32> = order[..b * per].iter().map(|&x| x as i32).collect();
        let [kc, vc] = caches;
        (kc, vc, Array::from_slice(&table, [b, per]))
    }

    #[test]
    fn paged_attention_equals_contiguous_attention() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x6b);
        // (batch, query heads, kv heads, queries, head size, block size, lengths)
        type Case = (usize, usize, usize, usize, usize, usize, &'static [usize]);
        let cases: &[Case] = &[
            (1, 4, 4, 1, 128, 16, &[37]),
            (3, 8, 2, 1, 128, 32, &[1, 64, 300]),
            (2, 4, 1, 4, 64, 16, &[4, 77]),
            (2, 2, 2, 1, 256, 8, &[9, 513]),
            // Long caches: two passes.
            (2, 8, 2, 1, 128, 32, &[1500, 3000]),
            (1, 8, 2, 4, 128, 64, &[2100]),
            (1, 4, 2, 1, 256, 16, &[1100]),
        ];
        for &(b, hq, hkv, lq, d, bs, lengths) in cases {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let n = *lengths.iter().max().unwrap();
                let keys = random(&mut rng, &[b, hkv, n, d], dtype);
                let values = random(&mut rng, &[b, hkv, n, d], dtype);
                let q = random(&mut rng, &[b, hq, lq, d], dtype);
                let (kc, vc, vtable) = paged(&keys, &values, bs, &mut rng);
                let lens: Array = lengths.iter().map(|&l| l as i32).collect();
                let scale = (d as f32).powf(-0.5);
                let name = format!("{dtype} {:?}", (b, hq, hkv, lq, d, bs, lengths));
                on(Device::gpu(0), || {
                    let got = fast::paged_attention(&q, &kc, &vc, &vtable, &lens, scale);
                    for (s, &len) in lengths.iter().enumerate() {
                        let pick = |a: &Array, end: usize| a.i((s..s + 1, .., 0..end, ..));
                        let mask = if lq > 1 {
                            fast::AttentionMask::Causal
                        } else {
                            fast::AttentionMask::None
                        };
                        let want = fast::scaled_dot_product_attention(
                            &pick(&q, lq),
                            &pick(&keys, len),
                            &pick(&values, len),
                            scale,
                            &fast::AttentionOptions::from(mask),
                        );
                        // Below 1,024 keys the contiguous path is the same single-pass kernel;
                        // with two passes, the key partitions may differ (they follow the
                        // table's capacity here, the keys there).
                        let exact = lengths.iter().all(|&l| l < 1024);
                        let tol = if exact { 0.0 } else { tolerance(dtype) };
                        assert_close(&format!("{name} seq {s}"), &pick(&got, lq), &want, tol);
                    }
                });
                // The kernel against the CPU fallback, with gradients for q and the caches.
                let g = random(&mut rng, &[b, hq, lq, d], dtype);
                both(
                    &format!("paged grads {name}"),
                    tolerance(dtype) * 4.0,
                    || {
                        let f = |xs: &[Array]| {
                            Ok(vec![fast::paged_attention(
                                &xs[0], &xs[1], &xs[2], &vtable, &lens, scale,
                            )])
                        };
                        with_grads(f, &[q.clone(), kc.clone(), vc.clone()], &g)
                    },
                );
            }
        }
        // Invalid arguments.
        let q = random(&mut rng, &[1, 4, 1, 64], DType::F32);
        let c = ops::zeros([4, 2, 8, 64], DType::F32);
        let t = array![[0, 1]];
        let l = array![9];
        assert!(
            fast::paged_attention(&q, &c, &c, &t, &l, 1.0)
                .check()
                .is_ok()
        );
        let c3 = ops::zeros([4, 3, 8, 64], DType::F32);
        assert!(
            fast::paged_attention(&q, &c3, &c3, &t, &l, 1.0)
                .check()
                .is_err(),
            "4 heads over 3"
        );
        assert!(
            fast::paged_attention(&q, &c, &c, &l, &l, 1.0)
                .check()
                .is_err(),
            "1-D table"
        );
        let lf = array![9.0];
        assert!(
            fast::paged_attention(&q, &c, &c, &t, &lf, 1.0)
                .check()
                .is_err(),
            "float lengths"
        );
    }

    #[test]
    fn paged_write_matches_the_fallback_and_writes_in_place() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x6c);
        let cache = random(&mut rng, &[6, 2, 4, 64], DType::BF16);
        let new = random(&mut rng, &[5, 2, 64], DType::BF16);
        let slots = array![23, 0, 5, 17, 6];
        both("paged_write", 0.0, || {
            Ok(vec![fast::paged_write(&cache, &new, &slots)])
        });
        // Slot 5 is block 1, position 1; written in place when owned, never into a cache
        // another array holds.
        on(Device::gpu(0), || {
            let before = values(&cache);
            let out = fast::paged_write(&cache, &new, &slots);
            out.eval().unwrap();
            assert_eq!(values(&cache), before, "a held cache changed");
            let block1 = out.i((1..2, .., 1..2, ..));
            let third = new.i((2..3, .., ..));
            assert_eq!(values(&block1), values(&third));
        });
        assert!(
            fast::paged_write(&cache, &random(&mut rng, &[5, 3, 64], DType::BF16), &slots)
                .check()
                .is_err()
        );
        assert!(
            fast::paged_write(&cache, &new, &array![1, 2])
                .check()
                .is_err()
        );
    }

    #[test]
    fn paged_write_skips_out_of_range_slots() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x6d);
        let (blocks, hkv, bs, d) = (6, 2, 4, 64);
        let end = (blocks * bs) as i32;
        let new = random(&mut rng, &[6, hkv, d], DType::F32);
        let slots = Array::from([-1i32, 3, end, i32::MAX, 7, i32::MIN]);
        // Only the rows with valid slots (3 and 7) are written.
        let valid_rows = new.take(&array![1, 4], 0);
        let valid_slots = array![3, 7];
        let cache = random(&mut rng, &[blocks, hkv, bs, d], DType::F32);
        for device in [Device::cpu(), Device::gpu(0)] {
            on(device, || {
                let got = fast::paged_write(&cache, &new, &slots);
                let want = fast::paged_write(&cache, &valid_rows, &valid_slots);
                assert_eq!(values(&got), values(&want), "{device:?}");
            });
        }
        // In place, with a canary allocated right after the cache: a slot one past the
        // end used to write into the next allocation.
        on(Device::gpu(0), || {
            for _ in 0..8 {
                let cache = ops::zeros([blocks, hkv, bs, d], DType::F32);
                cache.eval().unwrap();
                let canary = &ops::zeros([blocks, hkv, bs, d], DType::F32) + 7.0;
                canary.eval().unwrap();
                let out = fast::paged_write(&cache, &new, &slots);
                drop(cache);
                out.eval().unwrap();
                assert!(
                    values(&canary).iter().all(|&v| v == 7.0),
                    "canary overwritten"
                );
                let written = values(&out).iter().filter(|&&v| v != 0.0).count();
                assert!(written <= 2 * hkv * d, "{written} elements written");
            }
        });
    }

    #[test]
    fn paged_attention_clamps_hostile_tables_and_lengths() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x6e);
        // (batch, query heads, kv heads, queries, head size, block size, max_blocks): the
        // last case is long enough for two passes.
        for (b, hq, hkv, lq, d, bs, max_blocks) in [
            (2, 4, 2, 1, 64, 8, 4),
            (2, 4, 2, 3, 128, 16, 3),
            (2, 8, 2, 1, 128, 32, 64),
        ] {
            let blocks = 5;
            let kc = random(&mut rng, &[blocks, hkv, bs, d], DType::F32);
            let vc = random(&mut rng, &[blocks, hkv, bs, d], DType::F32);
            let q = random(&mut rng, &[b, hq, lq, d], DType::F32);
            // Entries past the block count, negative ones, and lengths past the capacity.
            let table: Vec<i32> = (0..b * max_blocks)
                .map(|i| match i % 4 {
                    0 => (i % blocks) as i32,
                    1 => blocks as i32 + 1000,
                    2 => -5,
                    _ => i32::MAX,
                })
                .collect();
            let table = Array::from_slice(&table, [b, max_blocks]);
            let capacity = (max_blocks * bs) as i32;
            let lens = Array::from_slice(&[capacity + 1, i32::MAX][..b], [b]);
            let name = format!("{:?}", (b, hq, hkv, lq, d, bs, max_blocks));
            both(
                &format!("hostile paged {name}"),
                tolerance(DType::F32) * 4.0,
                || {
                    Ok(vec![fast::paged_attention(
                        &q, &kc, &vc, &table, &lens, 0.125,
                    )])
                },
            );
            let got = on(Device::gpu(0), || {
                fast::paged_attention(&q, &kc, &vc, &table, &lens, 0.125)
            });
            assert!(values(&got).iter().all(|v| v.is_finite()), "{name}");
        }
    }

    #[test]
    fn softmax_logsumexp_and_cross_entropy_match_the_fallback() {
        if crate::common::require_gpu().is_none() {
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
                        |xs| Ok(vec![xs[0].softmax(-1)]),
                        std::slice::from_ref(&x),
                        &g,
                    )
                });
                let rows: Vec<usize> = shape[..shape.len() - 1].to_vec();
                let gr = random(&mut rng, &rows, DType::F32);
                let gl = gr.astype(dtype);
                both(&format!("logsumexp {name}"), tol, || {
                    with_grads(
                        |xs| Ok(vec![xs[0].logsumexp(-1)]),
                        std::slice::from_ref(&x),
                        &gl,
                    )
                });
                let classes = shape[shape.len() - 1];
                let n: usize = rows.iter().product();
                let t: Vec<i32> = (0..n)
                    .map(|i| (i * 7 % classes) as i32 - (i % 2) as i32)
                    .collect();
                let targets = Array::from_slice(&t, &rows[..]);
                both(&format!("cross entropy {name}"), tol, || {
                    let f = |xs: &[Array]| Ok(vec![fast::cross_entropy(&xs[0], &targets)]);
                    with_grads(f, std::slice::from_ref(&x), &gr)
                });
            }
        }
        // Infinite rows and an axis other than the last (the composed path).
        let x = Array::from_slice(&[f32::NEG_INFINITY, 1.0, f32::INFINITY, 0.0], [2, 2]);
        both("logsumexp inf", 0.0, || Ok(vec![x.logsumexp_keepdims(-1)]));
        let x = random(&mut rng, &[4, 3], DType::F32);
        both("softmax axis 0", 2e-5, || Ok(vec![x.softmax(0)]));
    }

    #[test]
    fn rope_matches_the_fallback() {
        if crate::common::require_gpu().is_none() {
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
                    (0..shape[0] as i32).map(|i| 3 * i + 1).collect::<Array>()
                } else {
                    Array::from(5i32)
                };
                let freqs = with_freqs.then(|| {
                    (0..dims / 2)
                        .map(|i| 1.0 + i as f32 * 3.0)
                        .collect::<Array>()
                });
                let base = (!with_freqs).then_some(10000.0);
                let name = format!("{dtype} {shape:?} dims {dims} trad {traditional}");
                both(&name, tolerance(dtype) * 4.0, || {
                    let f = |xs: &[Array]| {
                        Ok(vec![fast::rope(
                            &xs[0],
                            dims as i32,
                            traditional,
                            base,
                            0.5,
                            &offset,
                            freqs.as_ref(),
                        )])
                    };
                    with_grads(f, std::slice::from_ref(&x), &g)
                });
            }
        }
    }

    #[test]
    fn forward_mode_and_vmap_match_the_fallback() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x64);
        let x = random(&mut rng, &[3, 4, 32], DType::F32);
        let w = random(&mut rng, &[32], DType::F32);
        let t = random(&mut rng, &[3, 4, 32], DType::F32);
        both("rms jvp", 2e-5, || {
            let f = |xs: &[Array]| Ok(vec![fast::rms_norm(&xs[0], Some(&w), 1e-5)]);
            Ok(jvp_flat(f, std::slice::from_ref(&x), std::slice::from_ref(&t))?.1)
        });
        both("softmax jvp", 2e-5, || {
            Ok(jvp_flat(
                |xs: &[Array]| Ok(vec![xs[0].softmax(-1)]),
                std::slice::from_ref(&x),
                std::slice::from_ref(&t),
            )?
            .1)
        });
        // vmap over the normalised axis and over a leading axis, and with batched weights.
        both("vmap rms last axis", 2e-5, || {
            vmap_flat(
                |xs: &[Array]| Ok(vec![fast::rms_norm(&xs[0], None, 1e-5)]),
                &[Some(2)],
                &[0],
            )(std::slice::from_ref(&x))
        });
        let ws = random(&mut rng, &[3, 32], DType::F32);
        both("vmap layer norm weights", 2e-5, || {
            let f = |xs: &[Array]| Ok(vec![fast::layer_norm(&xs[0], Some(&xs[1]), None, 1e-5)]);
            vmap_flat(f, &[Some(0), Some(0)], &[0])(&[x.clone(), ws.clone()])
        });
        let targets = (0..12).map(|i| i % 32).collect::<Array>().reshape([3, 4]);
        both("vmap cross entropy", 2e-5, || {
            let f = |xs: &[Array]| Ok(vec![fast::cross_entropy(&xs[0], &xs[1])]);
            vmap_flat(f, &[Some(1), Some(1)], &[0])(&[x.clone(), targets.clone()])
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
        fast::scaled_dot_product_attention(&q, &k, &v, 0.5, &fast::AttentionOptions::causal());
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
        fast::scaled_dot_product_attention(
            &q2,
            &bad,
            &bad,
            1.0,
            &fast::AttentionOptions::default()
        )
        .check()
        .is_err()
    );
    assert!(
        fast::scaled_dot_product_attention(
            &q.reshape([1, lq, d]),
            &k,
            &v,
            1.0,
            &fast::AttentionOptions::default()
        )
        .check()
        .is_err()
    );
}

/// The gated delta rule computed directly in float64: `(y, state)` as flat vectors.
#[allow(clippy::too_many_arguments)]
fn gated_delta_reference(
    (b, t, hk, hv, dk, dv): (usize, usize, usize, usize, usize, usize),
    q: &[f64],
    k: &[f64],
    v: &[f64],
    g: &[f64],
    beta: &[f64],
    state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut s = state.to_vec();
    let mut y = vec![0.0; b * t * hv * dv];
    for bi in 0..b {
        for h in 0..hv {
            let kh = h / (hv / hk);
            let st = &mut s[(bi * hv + h) * dv * dk..(bi * hv + h + 1) * dv * dk];
            for ti in 0..t {
                let qk = |a: &[f64], c: usize| a[((bi * t + ti) * hk + kh) * dk + c];
                let gb = (bi * t + ti) * hv + h;
                for r in 0..dv {
                    let row = &mut st[r * dk..(r + 1) * dk];
                    row.iter_mut().for_each(|x| *x *= g[gb]);
                    let mem: f64 = (0..dk).map(|c| row[c] * qk(k, c)).sum();
                    let delta = (v[((bi * t + ti) * hv + h) * dv + r] - mem) * beta[gb];
                    (0..dk).for_each(|c| row[c] += qk(k, c) * delta);
                    y[((bi * t + ti) * hv + h) * dv + r] = (0..dk).map(|c| row[c] * qk(q, c)).sum();
                }
            }
        }
    }
    (y, s)
}

#[test]
fn gated_delta_fallback_matches_a_direct_formula() {
    let mut rng = Rng::new(0x67);
    let dims @ (b, t, hk, hv, dk, dv) = (2, 4, 1, 2, 3, 2);
    let q = random(&mut rng, &[b, t, hk, dk], DType::F64);
    let k = random(&mut rng, &[b, t, hk, dk], DType::F64);
    let v = random(&mut rng, &[b, t, hv, dv], DType::F64);
    let g = random(&mut rng, &[b, t, hv], DType::F64).add(0.75);
    let beta = random(&mut rng, &[b, t, hv], DType::F64).add(0.5);
    let state = random(&mut rng, &[b, hv, dv, dk], DType::F64);
    let (y, s) = fast::gated_delta(&q, &k, &v, &g, &beta, &state);
    let all = [&q, &k, &v, &g, &beta, &state].map(values);
    let (wy, ws) =
        gated_delta_reference(dims, &all[0], &all[1], &all[2], &all[3], &all[4], &all[5]);
    for (i, (a, w)) in values(&y).iter().zip(&wy).enumerate() {
        assert!((a - w).abs() < 1e-12, "y[{i}]: {a} vs {w}");
    }
    for (i, (a, w)) in values(&s).iter().zip(&ws).enumerate() {
        assert!((a - w).abs() < 1e-12, "state[{i}]: {a} vs {w}");
    }
    // Gradients of every input against central differences (float64).
    let inputs = [q, k, v, g, beta, state];
    let loss = |xs: &[Array]| -> Array {
        let (y, s) = fast::gated_delta(&xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5]);
        y.multiply(&y).sum(..).add(s.sum(..))
    };
    let f = |xs: &[Array]| Ok(vec![loss(xs)]);
    let (_, grads) = vjp_flat(f, &inputs, &[Array::from(1.0f64)]).unwrap();
    let h = 1e-6;
    for (i, grad) in grads.iter().enumerate() {
        let gv = values(grad);
        let base = values(&inputs[i]);
        for j in [0, base.len() / 2, base.len() - 1] {
            let shifted = |d: f64| {
                let mut xs = inputs.to_vec();
                let mut p = base.clone();
                p[j] += d;
                xs[i] = Array::from_slice(&p, inputs[i].shape());
                loss(&xs).item::<f64>().unwrap()
            };
            let numeric = (shifted(h) - shifted(-h)) / (2.0 * h);
            assert!(
                (gv[j] - numeric).abs() < 1e-5 * numeric.abs().max(1.0),
                "input {i}[{j}]: {} vs {numeric}",
                gv[j]
            );
        }
    }
    // Batching over a leading axis matches a loop.
    let qs = random(&mut rng, &[3, b, t, hk, dk], DType::F64);
    let f = |xs: &[Array]| {
        let (y, s) = fast::gated_delta(
            &xs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5],
        );
        Ok(vec![y, s])
    };
    let batched = vmap_flat(f, &[Some(0)], &[0, 0])(std::slice::from_ref(&qs)).unwrap();
    for i in 0..3 {
        let qi = qs.take(&Array::from(i), 0);
        let one = f(&[qi]).unwrap();
        for (o, w) in batched.iter().zip(&one) {
            let oi = o.take(&Array::from(i), 0);
            assert_close(&format!("vmap {i}"), &oi, w, 1e-12);
        }
    }
    // Errors: head grouping, gate shapes, state shape.
    let bad_v = random(&mut rng, &[b, t, 3, dv], DType::F64);
    assert!(
        fast::gated_delta(
            &inputs[0], &inputs[1], &bad_v, &inputs[3], &inputs[4], &inputs[5]
        )
        .0
        .check()
        .is_err()
    );
    let bad_g = random(&mut rng, &[b, t, hv, 1], DType::F64);
    assert!(
        fast::gated_delta(
            &inputs[0], &inputs[1], &inputs[2], &bad_g, &inputs[4], &inputs[5]
        )
        .0
        .check()
        .is_err()
    );
    let bad_s = random(&mut rng, &[b, hv, dk, dv], DType::F64);
    assert!(
        fast::gated_delta(
            &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &bad_s
        )
        .0
        .check()
        .is_err()
    );
}

#[cfg(target_os = "macos")]
mod attention_gpu {
    use super::*;

    fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
        with_stream(device, f)
    }

    #[test]
    fn attention_kernels_match_the_fallback() {
        if cfg!(miri) || on(Device::gpu(0), || (Array::from(1.0f32) * 2.0).eval()).is_err() {
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
                let z = |l, n, e| ops::zeros([b, n, l, e], DType::F16);
                let out = fast::scaled_dot_product_attention(
                    &z(lq, hq, d),
                    &z(lk, hkv, d),
                    &z(lk, hkv, dv),
                    1.0,
                    &fast::AttentionOptions::default(),
                );
                matches!(
                    graph::primitive(&out),
                    Some(graph::Primitive::Fast(fast::FastOp::Attention { .. }))
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
                let k = random(&mut rng, &[b, lk, hkv, d], dtype).transpose([0, 2, 1, 3]);
                let v = random(&mut rng, &[b, lk, hkv, dv], dtype).transpose([0, 2, 1, 3]);
                let bmask = random(&mut rng, &[lq, lk], DType::F32).greater(-0.5);
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
                            fast::scaled_dot_product_attention(
                                &q,
                                &k,
                                &v,
                                0.3,
                                &fast::AttentionOptions::from(mask).sinks(s),
                            )
                        };
                        // The reference: the composition in float32 on the CPU, rounded once.
                        let reference = || {
                            let f = |a: &Array| a.astype(DType::F32);
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
                                &fast::AttentionOptions::from(mask32).sinks(s32.as_ref()),
                            )
                            .astype(dtype)
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
                    &fast::AttentionOptions::causal(),
                )])
            };
            vjp_flat(
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

#[cfg(target_os = "macos")]
mod gated_delta_gpu {
    use super::*;

    fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
        with_stream(device, f)
    }

    /// `x` normalised over its last axis, in `dtype`.
    fn unit(x: &Array, dtype: DType) -> Array {
        let norm = x.multiply(x).sum_keepdims(-1).sqrt();
        x.divide(&norm).astype(dtype)
    }

    #[test]
    fn gated_delta_at_slots_equals_gated_delta_per_sequence() {
        if cfg!(miri) || on(Device::gpu(0), || (Array::from(1.0f32) * 2.0).eval()).is_err() {
            return;
        }
        let mut rng = Rng::new(0x6d);
        // (T, Hk, Hv, Dk, Dv): packed and generic kernels.
        for &(t, hk, hv, dk, dv) in &[(1, 2, 4, 128, 128), (3, 2, 4, 128, 64), (2, 1, 2, 64, 48)] {
            let (b, slots_n) = (3, 5);
            let q = random(&mut rng, &[b, t, hk, dk], DType::BF16);
            let k = unit(&random(&mut rng, &[b, t, hk, dk], DType::F32), DType::BF16);
            let v = random(&mut rng, &[b, t, hv, dv], DType::BF16);
            let g = random(&mut rng, &[b, t, hv], DType::F32).add(0.75);
            let beta = random(&mut rng, &[b, t, hv], DType::BF16).add(0.5);
            let pool = random(&mut rng, &[slots_n, hv, dv, dk], DType::F32);
            let slots = Array::from_slice(&[3i32, 0, 4], [b]);
            let row = |a: &Array, i: usize| a.i(i..i + 1);
            let name = format!("{:?}", (t, hk, hv, dk, dv));
            on(Device::gpu(0), || {
                let before = values(&pool);
                let (y, new_pool) = fast::gated_delta_slots(&q, &k, &v, &g, &beta, &pool, &slots);
                eval(&[y.clone(), new_pool.clone()]).unwrap();
                assert_eq!(values(&pool), before, "{name}: a held pool changed");
                for (i, &slot) in [3usize, 0, 4].iter().enumerate() {
                    let (yi, si) = fast::gated_delta(
                        &row(&q, i),
                        &row(&k, i),
                        &row(&v, i),
                        &row(&g, i),
                        &row(&beta, i),
                        &row(&pool, slot),
                    );
                    assert_close(&format!("{name} y {i}"), &row(&y, i), &yi, 0.0);
                    assert_close(
                        &format!("{name} state {i}"),
                        &row(&new_pool, slot),
                        &si,
                        0.0,
                    );
                }
                for untouched in [1, 2] {
                    assert_close(
                        &format!("{name} slot {untouched}"),
                        &row(&new_pool, untouched),
                        &row(&pool, untouched),
                        0.0,
                    );
                }
            });
            let run = || {
                let (y, p) = fast::gated_delta_slots(&q, &k, &v, &g, &beta, &pool, &slots);
                vec![y, p]
            };
            let (gpu, cpu) = (on(Device::gpu(0), run), on(Device::cpu(), run));
            assert_close(
                &format!("{name} y cpu"),
                &gpu[0],
                &cpu[0],
                tolerance(DType::BF16),
            );
            assert_close(&format!("{name} pool cpu"), &gpu[1], &cpu[1], 2e-4);
        }
    }

    #[test]
    fn gated_delta_kernels_match_the_fallback() {
        if cfg!(miri) || on(Device::gpu(0), || (Array::from(1.0f32) * 2.0).eval()).is_err() {
            return;
        }
        let mut rng = Rng::new(0x68);
        // (B, T, Hk, Hv, Dk, Dv): packed (decode, prefill, Qwen3.5's heads) and generic.
        let shapes: &[(usize, usize, usize, usize, usize, usize)] = &[
            (1, 1, 2, 4, 128, 128),
            (1, 9, 2, 4, 128, 64),
            (1, 3, 16, 32, 128, 128),
            (2, 5, 1, 2, 64, 48),
            (1, 4, 4, 4, 32, 17),
        ];
        for &(b, t, hk, hv, dk, dv) in shapes {
            for (dtype, gate) in [
                (DType::F32, DType::F32),
                (DType::BF16, DType::F32),
                (DType::F16, DType::F32),
                (DType::BF16, DType::BF16),
            ] {
                // q as a transposed view, as projections reshape to it. Unit keys, as models
                // normalise them: the recurrence is stable for β‖k‖² ≤ 2.
                let q = random(&mut rng, &[b, hk, t, dk], dtype).transpose([0, 2, 1, 3]);
                let k = unit(&random(&mut rng, &[b, t, hk, dk], DType::F32), dtype);
                let v = random(&mut rng, &[b, t, hv, dv], dtype);
                let g = random(&mut rng, &[b, t, hv], DType::F32)
                    .add(0.75)
                    .astype(gate);
                let beta = random(&mut rng, &[b, t, hv], dtype).add(0.5);
                let state = random(&mut rng, &[b, hv, dv, dk], gate);
                let run = || fast::gated_delta(&q, &k, &v, &g, &beta, &state);
                let fused = on(Device::gpu(0), || {
                    matches!(
                        graph::primitive(&run().0),
                        Some(graph::Primitive::Fast(fast::FastOp::GatedDelta {
                            steps: false
                        }))
                    )
                });
                assert!(fused, "{:?} should use a kernel", (b, t, hk, hv, dk, dv));
                let ((gy, gs), (cy, cs)) = (on(Device::gpu(0), run), on(Device::cpu(), run));
                let name = format!("{dtype}/{gate} {:?}", (b, t, hk, hv, dk, dv));
                // The kernels sum each row's Dk products as a tree, the fallback in order.
                let tol = |d: DType| if d == DType::F32 { 2e-4 } else { tolerance(d) };
                assert_close(&format!("{name} y"), &gy, &cy, tol(dtype));
                assert_close(&format!("{name} state"), &gs, &cs, tol(gate));
                // Every step's state: the same outputs bit for bit, the last state the final
                // one, and step `i`'s state that of the first `i + 1` steps alone.
                let first = |x: &Array, n: usize| x.i((.., 0..n));
                let step = |states: &Array, i: usize| states.i((.., i..i + 1)).squeeze(1);
                let steps = || fast::gated_delta_steps(&q, &k, &v, &g, &beta, &state);
                on(Device::gpu(0), || {
                    let (sy, ss) = steps();
                    assert_eq!(ss.shape(), [b, t, hv, dv, dk].map(|d| d as i32));
                    assert_close(&format!("{name} steps y"), &sy, &gy, 0.0);
                    assert_close(&format!("{name} steps last"), &step(&ss, t - 1), &gs, 0.0);
                    for i in 0..t {
                        let (_, si) = fast::gated_delta(
                            &first(&q, i + 1),
                            &first(&k, i + 1),
                            &first(&v, i + 1),
                            &first(&g, i + 1),
                            &first(&beta, i + 1),
                            &state,
                        );
                        assert_close(&format!("{name} step {i}"), &step(&ss, i), &si, 0.0);
                    }
                });
                let (_, cpu_steps) = on(Device::cpu(), steps);
                let (_, gpu_steps) = on(Device::gpu(0), steps);
                assert_close(
                    &format!("{name} steps cpu"),
                    &gpu_steps,
                    &cpu_steps,
                    tol(gate),
                );
            }
        }
        // Gradients through the fused forward match the CPU.
        let (b, t, hk, hv, dk, dv) = (1, 3, 1, 2, 32, 8);
        let xs = vec![
            random(&mut rng, &[b, t, hk, dk], DType::F32),
            unit(&random(&mut rng, &[b, t, hk, dk], DType::F32), DType::F32),
            random(&mut rng, &[b, t, hv, dv], DType::F32),
            random(&mut rng, &[b, t, hv], DType::F32).add(0.75),
            random(&mut rng, &[b, t, hv], DType::F32).add(0.5),
            random(&mut rng, &[b, hv, dv, dk], DType::F32),
        ];
        let cot = [
            random(&mut rng, &[b, t, hv, dv], DType::F32),
            random(&mut rng, &[b, hv, dv, dk], DType::F32),
        ];
        let grads = || {
            let f = |xs: &[Array]| {
                let (y, s) = fast::gated_delta(&xs[0], &xs[1], &xs[2], &xs[3], &xs[4], &xs[5]);
                Ok(vec![y, s])
            };
            vjp_flat(f, &xs, &cot).unwrap().1
        };
        let (gpu, cpu) = (on(Device::gpu(0), grads), on(Device::cpu(), grads));
        for (i, (a, c)) in gpu.iter().zip(&cpu).enumerate() {
            assert_close(&format!("gated delta vjp #{i}"), a, c, 1e-4);
        }
    }
}

/// Capped attention (the composition in query blocks, or the tiled kernel) and sliding
/// windows: against the uncapped composition in float32 on the CPU (and a direct window
/// formula), on sliced key caches and transposed queries, on both devices, with gradients.
mod capped_attention {
    use super::*;
    use fast::{AttentionKernel, AttentionOptions, plan_attention};

    struct Inputs {
        q: Array,
        k: Array,
        v: Array,
        sinks: Array,
        bmask: Array,
    }

    /// Queries as a transposed view of `[B, L, H, D]`, keys and values as a slice (an
    /// offset) of a longer cache.
    fn inputs(rng: &mut Rng, s: (usize, usize, usize, usize, usize, usize), dt: DType) -> Inputs {
        let (b, hq, hkv, lq, lk, d) = s;
        let q = random(rng, &[b, lq, hq, d], dt).transpose([0, 2, 1, 3]);
        let cache =
            |rng: &mut Rng| random(rng, &[b, hkv, lk + 5, d], dt).i((.., .., 3..3 + lk as i32));
        let (k, v) = (cache(rng), cache(rng));
        Inputs {
            q,
            k,
            v,
            sinks: random(rng, &[hq], dt),
            bmask: random(rng, &[lq, lk], DType::F32).greater(-0.5),
        }
    }

    /// The options: mask kind 0 causal, 1 boolean, 2 causal with a window of 7.
    fn options<'a>(x: &'a Inputs, kind: usize, sinks: bool) -> AttentionOptions<'a> {
        let o = match kind {
            0 => AttentionOptions::causal(),
            1 => AttentionOptions::default().mask(&x.bmask),
            _ => AttentionOptions::causal().window(7),
        };
        o.sinks(sinks.then_some(&x.sinks))
    }

    /// A window as an explicit boolean mask (mlx-lm's `create_causal_mask(window_size)`).
    fn window_mask(lq: usize, lk: usize, w: usize) -> Array {
        let pos = ops::arange((lk - lq) as i32, lk as i32, 1).reshape([-1, 1]);
        let keys = ops::arange(0, lk as i32, 1).reshape([1, -1]);
        pos.greater_equal(&keys)
            .multiply(pos.less(keys.add(w as i32)))
    }

    const SHAPES: [(usize, usize, usize, usize, usize, usize); 5] = [
        (1, 4, 2, 33, 45, 64),
        (2, 2, 2, 20, 29, 80),
        (1, 8, 1, 18, 60, 256),
        (1, 4, 4, 9, 9, 128),
        (1, 2, 2, 20, 29, 512),
    ];

    #[test]
    fn capped_equals_uncapped_and_the_window_formula() {
        let gpu = cfg!(target_os = "macos")
            && !cfg!(miri)
            && with_stream(Device::gpu(0), || (Array::from(1.0f32) * 2.0).eval()).is_ok();
        let devices: Vec<Device> = if gpu {
            vec![Device::cpu(), Device::gpu(0)]
        } else {
            vec![Device::cpu()]
        };
        let mut rng = Rng::new(0x61);
        let mut blocked = 0;
        for &shape in &SHAPES {
            let (b, hq, _, lq, lk, d) = shape;
            for dt in [DType::F32, DType::F16, DType::BF16] {
                let x = inputs(&mut rng, shape, dt);
                let scale = 1.0 / (d as f32).sqrt();
                let tol = match dt {
                    DType::F32 => 1e-4,
                    DType::F16 => 1e-2,
                    _ => 5e-2,
                };
                for kind in 0..3 {
                    for sinks in [false, true] {
                        let o = options(&x, kind, sinks);
                        // The reference: uncapped, in float32 on the CPU, the window as an
                        // explicit mask.
                        let reference = with_stream(Device::cpu(), || {
                            let f = |a: &Array| a.astype(DType::F32);
                            let wmask = window_mask(lq, lk, 7);
                            let s32 = f(&x.sinks);
                            let r = match kind {
                                0 => AttentionOptions::causal(),
                                1 => AttentionOptions::default().mask(&x.bmask),
                                _ => AttentionOptions::default().mask(&wmask),
                            };
                            let r = r.sinks(sinks.then_some(&s32));
                            fast::scaled_dot_product_attention(
                                &f(&x.q),
                                &f(&x.k),
                                &f(&x.v),
                                scale,
                                &r,
                            )
                            .astype(dt)
                        });
                        let e = if sinks || dt == DType::F32 { 4 } else { 2 };
                        for &device in &devices {
                            for blocks in [1, 2, 3, lq.div_ceil(9)] {
                                let rows = lq.div_ceil(blocks);
                                let keys = if kind == 2 { lk.min(7 + rows - 1) } else { lk };
                                let cap = b * hq * keys * e * rows;
                                let o = o.max_score_bytes(cap);
                                let (got, plan) = with_stream(device, || {
                                    let dims = |a: &Array| a.shape().to_vec();
                                    let plan = plan_attention(
                                        &dims(&x.q),
                                        &dims(&x.k),
                                        &dims(&x.v),
                                        dt,
                                        &o,
                                    )
                                    .unwrap();
                                    let got = fast::scaled_dot_product_attention(
                                        &x.q, &x.k, &x.v, scale, &o,
                                    );
                                    got.eval().unwrap();
                                    (got, plan)
                                });
                                assert!(plan.max_intermediate_bytes <= cap, "{plan:?} cap {cap}");
                                if matches!(plan.kernel, AttentionKernel::ComposedBlocks { .. }) {
                                    blocked += 1;
                                }
                                let name = format!(
                                    "{dt} {shape:?} kind {kind} sinks {sinks} {device:?} cap {cap} {plan:?}"
                                );
                                assert_close(&name, &got, &reference, tol);
                                // Deterministic for a fixed plan.
                                let again = with_stream(device, || {
                                    fast::scaled_dot_product_attention(&x.q, &x.k, &x.v, scale, &o)
                                });
                                assert_eq!(values(&again), values(&got), "{name}: run to run");
                            }
                        }
                    }
                }
            }
        }
        assert!(blocked > 50, "only {blocked} blocked runs");
    }

    /// The blocked composition's derivatives equal the unblocked composition's: reverse
    /// mode, forward mode (through the blocks' concatenation), batching, and both match
    /// finite differences.
    #[test]
    fn blocked_gradients_match() {
        let mut rng = Rng::new(0x62);
        let x = &inputs(&mut rng, (1, 4, 2, 11, 14, 8), DType::F32);
        let g = random(&mut rng, &[1, 4, 11, 8], DType::F32);
        let run = |kind: usize, sinks: bool, cap: Option<usize>| {
            move |xs: &[Array]| {
                let mut o = options(x, kind, sinks);
                if let Some(c) = cap {
                    o = o.max_score_bytes(c);
                }
                Ok(vec![fast::scaled_dot_product_attention(
                    &xs[0], &xs[1], &xs[2], 0.3, &o,
                )])
            }
        };
        let primals = [x.q.clone(), x.k.clone(), x.v.clone()];
        with_stream(Device::cpu(), || {
            for kind in 0..3 {
                for sinks in [false, true] {
                    // Blocks of 4 rows (and 3 under the window, whose rows see ≤ 10 keys).
                    let cap = 4 * 4 * 14 * 4;
                    let plan = plan_attention(
                        &[1, 4, 11, 8],
                        &[1, 2, 14, 8],
                        &[1, 2, 14, 8],
                        DType::F32,
                        &options(x, kind, sinks).max_score_bytes(cap),
                    )
                    .unwrap();
                    assert!(
                        matches!(plan.kernel, AttentionKernel::ComposedBlocks { .. }),
                        "{plan:?}"
                    );
                    let name = format!("kind {kind} sinks {sinks}");
                    let (_, blocked) = vjp_flat(
                        run(kind, sinks, Some(cap)),
                        &primals,
                        std::slice::from_ref(&g),
                    )
                    .unwrap();
                    let (_, whole) =
                        vjp_flat(run(kind, sinks, None), &primals, std::slice::from_ref(&g))
                            .unwrap();
                    for (i, (a, w)) in blocked.iter().zip(&whole).enumerate() {
                        assert_close(&format!("{name} vjp #{i}"), a, w, 1e-5);
                    }
                    let tangents: Vec<Array> = primals
                        .iter()
                        .map(|p| {
                            random(
                                &mut Rng::new(7),
                                &p.shape().iter().map(|&n| n as usize).collect::<Vec<_>>(),
                                DType::F32,
                            )
                        })
                        .collect();
                    let (_, jb) =
                        jvp_flat(run(kind, sinks, Some(cap)), &primals, &tangents).unwrap();
                    let (_, jw) = jvp_flat(run(kind, sinks, None), &primals, &tangents).unwrap();
                    assert_close(&format!("{name} jvp"), &jb[0], &jw[0], 1e-5);
                    // vjp and jvp agree: <g, J t> = <Jᵀ g, t>.
                    let lhs: f64 = values(&g)
                        .iter()
                        .zip(values(&jb[0]))
                        .map(|(a, b)| a * b)
                        .sum();
                    let rhs: f64 = blocked
                        .iter()
                        .zip(&tangents)
                        .map(|(a, t)| {
                            values(a)
                                .iter()
                                .zip(values(t))
                                .map(|(x, y)| x * y)
                                .sum::<f64>()
                        })
                        .sum();
                    assert!(
                        (lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0),
                        "{name}: {lhs} vs {rhs}"
                    );
                    // Central finite differences along the tangent (float32: good to ~1e-3).
                    let h = 2e-3;
                    let at = |sign: f64| {
                        let shifted: Vec<Array> = primals
                            .iter()
                            .zip(&tangents)
                            .map(|(p, t)| p + &(t * (sign * h)))
                            .collect();
                        values(&run(kind, sinks, Some(cap))(&shifted).unwrap()[0])
                    };
                    let (up, down) = (at(1.0), at(-1.0));
                    for (i, j) in values(&jb[0]).iter().enumerate() {
                        let fd = (up[i] - down[i]) / (2.0 * h);
                        assert!(
                            (fd - j).abs() <= 2e-3 * j.abs().max(1.0),
                            "{name} fd {i}: {fd} vs {j}"
                        );
                    }
                }
            }
            // vmap over a leading batch axis of the queries equals per-batch calls.
            let qs = random(&mut rng, &[3, 1, 4, 11, 8], DType::F32);
            let f = |xs: &[Array]| {
                let o = AttentionOptions::causal().max_score_bytes(3 * 4 * 14 * 4);
                Ok(vec![fast::scaled_dot_product_attention(
                    &xs[0], &x.k, &x.v, 0.3, &o,
                )])
            };
            let batched = vmap_flat(f, &[Some(0)], &[0])(std::slice::from_ref(&qs)).unwrap();
            for i in 0..3 {
                let one = f(&[qs.i(i)]).unwrap();
                assert_close(&format!("vmap {i}"), &batched[0].i(i), &one[0], 1e-6);
            }
        });
    }
}
