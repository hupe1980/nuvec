//! Affine quantization: round trips, `quantized_matmul` against the dequantized product,
//! derivatives against an independent composition, `vmap` against a loop, and (on macOS)
//! the GPU kernels against the CPU.

mod common;

use common::Rng;
use nuvec_core::ops;
use nuvec_core::*;

const BITS: [i32; 7] = [1, 2, 3, 4, 5, 6, 8];
const GROUPS: [i32; 3] = [32, 64, 128];

fn random(rng: &mut Rng, shape: &[i32], dtype: DType) -> Array {
    let n: i32 = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 / 4.0).collect();
    Array::from_slice(&v, shape).astype(dtype)
}

/// Affine quantization of `w`.
fn aq(w: &Array, gs: i32, bits: i32) -> Quantized {
    w.quantize(Quant::affine(gs, bits))
}

/// Affine quantized weights from their parts.
fn qw(w: &Array, s: &Array, b: &Array, gs: i32, bits: i32) -> Quantized {
    Quantized::new(
        w.clone(),
        s.clone(),
        Some(b.clone()),
        Quant::affine(gs, bits),
    )
}

/// The parts of affine quantized weights: codes, scales and biases.
fn parts(q: Quantized) -> (Array, Array, Array) {
    (q.weight, q.scales, q.biases.expect("affine"))
}

fn values(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

fn assert_close(name: &str, got: &Array, want: &Array, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    let (g, w) = (values(got), values(want));
    let scale = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert!(
            (a - b).abs() <= tol * scale,
            "{name}: element {i}: {a} vs {b} (tolerance {})",
            tol * scale
        );
    }
}

/// `x · dequantize(w)ᵀ` (or without the transpose) with ordinary operations, in the
/// promoted type.
fn reference(x: &Array, w: &Array, s: &Array, b: &Array, t: bool, gs: i32, bits: i32) -> Array {
    let dtype = promote_types(x.dtype(), s.dtype());
    let (s, b) = (s.astype(dtype), b.astype(dtype));
    let dense = qw(w, &s, &b, gs, bits).dequantize(None);
    let dense = if t {
        let n = dense.ndim();
        let mut axes: Vec<usize> = (0..n).collect();
        axes.swap(n - 2, n - 1);
        dense.transpose(axes)
    } else {
        dense
    };
    x.astype(dtype).matmul(&dense)
}

#[test]
fn round_trip_error_is_about_half_a_step() {
    let mut rng = Rng::new(0x51);
    for bits in BITS {
        for gs in GROUPS {
            let w = random(&mut rng, &[3, 2 * gs], DType::F32);
            let (wq, s, b) = parts(aq(&w, gs, bits));
            assert_eq!(wq.dtype(), DType::U32);
            assert_eq!(wq.shape(), [3, 2 * gs * bits / 32]);
            assert_eq!(s.shape(), [3, 2]);
            let restored = values(&qw(&wq, &s, &b, gs, bits).dequantize(None));
            let (orig, scales) = (values(&w), values(&s));
            let (mut total, mut steps) = (0.0, 0.0);
            for (i, (r, o)) in restored.iter().zip(&orig).enumerate() {
                let step = scales[i / gs as usize].abs();
                // Snapping zero (or an edge) to a code can clip the far end of a group's
                // range, as in MLX, so single errors may exceed half a step.
                assert!(
                    (r - o).abs() <= step * 1.0001,
                    "{bits} bits, group {gs}: {r} vs {o}"
                );
                total += (r - o).abs();
                steps += step;
            }
            assert!(total <= steps / 2.0, "{bits} bits, group {gs}: mean error");
        }
    }
}

#[test]
fn quantize_snaps_zero_and_handles_constant_and_one_signed_groups() {
    // A constant group, an all-negative group and a group containing zero.
    let mut v = vec![0.75f32; 32];
    v.extend((0..32).map(|i| -1.0 - i as f32 / 8.0));
    v.extend((0..32).map(|i| i as f32 / 31.0 - 0.25));
    let w = Array::from_slice(&v, [3, 32]);
    let (wq, s, b) = parts(aq(&w, 32, 4));
    let back = values(&qw(&wq, &s, &b, 32, 4).dequantize(None));
    assert!(
        back[..32].iter().all(|&x| (x - 0.75).abs() < 1e-6),
        "{:?}",
        &back[..4]
    );
    // Zero is representable exactly in a group that spans it.
    let codes = values(
        &qw(
            &wq,
            &ops::ones([3, 1], DType::F32),
            &ops::zeros([3, 1], DType::F32),
            32,
            4,
        )
        .dequantize(None),
    );
    let (scale, bias) = (values(&s)[2], values(&b)[2]);
    assert!(codes[64..].iter().any(|&q| (q * scale + bias).abs() < 1e-7));
}

#[test]
fn invalid_arguments_are_rejected() {
    let fails = |q: Quantized| {
        q.weight.clone().check().is_err()
            && q.scales.clone().check().is_err()
            && q.biases.is_some_and(|b| b.check().is_err())
    };
    let w = ops::full([2, 64], 1.0f32);
    assert!(fails(aq(&w, 48, 4)), "group size");
    assert!(fails(aq(&w, 64, 7)), "bits");
    assert!(fails(aq(&ops::full([64], 1.0f32), 64, 4)), "1-D");
    assert!(
        fails(aq(&ops::full([2, 96], 1.0f32), 64, 4)),
        "divisibility"
    );
    assert!(fails(aq(&w.astype(DType::I32), 64, 4)), "dtype");
    let (wq, s, b) = parts(aq(&w, 64, 4));
    assert!(
        qw(&wq, &s, &b, 32, 4).dequantize(None).check().is_err(),
        "group mismatch"
    );
    assert!(
        qw(&s, &s, &b, 64, 4).dequantize(None).check().is_err(),
        "packed dtype"
    );
    let x = ops::full([3, 63], 1.0f32);
    assert!(
        x.quantized_matmul(&qw(&wq, &s, &b, 64, 4), true)
            .check()
            .is_err(),
        "inner size"
    );
    let x = ops::full([3, 64], 1);
    assert!(
        x.astype(DType::F64)
            .quantized_matmul(&qw(&wq, &s, &b, 64, 4), true)
            .check()
            .is_err()
    );
}

#[test]
#[cfg_attr(miri, ignore = "Miri cannot execute half's hardware f16 conversion")]
fn quantized_matmul_equals_the_dequantized_product() {
    let mut rng = Rng::new(0x52);
    for (bits, gs) in [
        (4, 64),
        (3, 32),
        (8, 128),
        (2, 32),
        (6, 64),
        (5, 32),
        (1, 64),
    ] {
        let w = random(&mut rng, &[48, 128], DType::F32);
        let (wq, s, b) = parts(aq(&w, gs, bits));
        for shape in [vec![128], vec![5, 128], vec![2, 3, 128]] {
            let x = random(&mut rng, &shape, DType::F32);
            let got = x.quantized_matmul(&qw(&wq, &s, &b, gs, bits), true);
            let want = reference(&x, &wq, &s, &b, true, gs, bits);
            assert_close(&format!("t {bits}/{gs} {shape:?}"), &got, &want, 1e-6);
        }
        // Without the transpose, `w` is [K, N]: K = 48 rows.
        let x = random(&mut rng, &[4, 48], DType::F32);
        let got = x.quantized_matmul(&qw(&wq, &s, &b, gs, bits), false);
        assert_close(
            "n",
            &got,
            &reference(&x, &wq, &s, &b, false, gs, bits),
            1e-6,
        );
    }
    // Batched weights broadcast against batched activations; half precision promotes.
    let w = random(&mut rng, &[2, 1, 16, 64], DType::F16);
    let (wq, s, b) = parts(aq(&w, 32, 4));
    let x = random(&mut rng, &[3, 5, 64], DType::F32);
    let got = x.quantized_matmul(&qw(&wq, &s, &b, 32, 4), true);
    assert_eq!(got.shape(), [2, 3, 5, 16]);
    assert_eq!(got.dtype(), DType::F32);
    let wq = wq.broadcast_to([2, 3, 16, 8]);
    let s = s.broadcast_to([2, 3, 16, 2]);
    let b = b.broadcast_to([2, 3, 16, 2]);
    assert_close(
        "batched",
        &got,
        &reference(&x, &wq, &s, &b, true, 32, 4),
        1e-6,
    );
}

/// `x · (codes ⊙ s + b)ᵀ` from ordinary differentiable operations, with fixed codes.
fn composed(xs: &[Array], codes: &Array, gs: i32) -> Result<Vec<Array>> {
    let (x, s, b) = (&xs[0], &xs[1], &xs[2]);
    let [rows, cols] = codes.shape_n::<2>()?;
    let grouped = codes.reshape([rows, cols / gs, gs]);
    let dense = (grouped * s.expand_dims(-1)) + b.expand_dims(-1);
    Ok(vec![x.matmul(&dense.reshape([rows, cols]).t())])
}

#[test]
fn derivatives_match_an_independent_composition() {
    let mut rng = Rng::new(0x53);
    let (gs, bits) = (32, 4);
    let w = random(&mut rng, &[6, 64], DType::F32);
    let (wq, s, b) = parts(aq(&w, gs, bits));
    let codes = qw(
        &wq,
        &ops::ones([6, 2], DType::F32),
        &ops::zeros([6, 2], DType::F32),
        gs,
        bits,
    )
    .dequantize(None);
    let x = random(&mut rng, &[2, 3, 64], DType::F32);
    let g = random(&mut rng, &[2, 3, 6], DType::F32);
    let wq2 = wq.clone();
    let f = move |xs: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![xs[0].quantized_matmul(
            &qw(&wq2, &xs[1], &xs[2], gs, bits),
            true,
        )])
    };
    let primals = [x.clone(), s.clone(), b.clone()];
    let (_, got) = vjp_flat(f.clone(), &primals, std::slice::from_ref(&g)).unwrap();
    let c = codes.clone();
    let (_, want) = vjp_flat(move |xs: &[Array]| composed(xs, &c, gs), &primals, &[g]).unwrap();
    for (name, (a, e)) in ["dx", "ds", "db"].iter().zip(got.iter().zip(&want)) {
        assert_close(name, a, e, 1e-5);
    }
    let tangents = [
        random(&mut rng, &[2, 3, 64], DType::F32),
        random(&mut rng, &[6, 2], DType::F32),
        random(&mut rng, &[6, 2], DType::F32),
    ];
    let (_, got) = jvp_flat(f, &primals, &tangents).unwrap();
    let (_, want) = jvp_flat(
        move |xs: &[Array]| composed(xs, &codes, gs),
        &primals,
        &tangents,
    )
    .unwrap();
    assert_close("jvp", &got[0], &want[0], 1e-5);
    // No gradient flows into the packed weights.
    let err = vjp_flat(
        move |xs: &[Array]| {
            Ok(vec![
                x.quantized_matmul(&qw(&xs[0], &s, &b, gs, bits), true),
            ])
        },
        &[wq],
        &[ops::full([2, 3, 6], 1.0f32)],
    );
    assert!(err.is_err());
}

#[test]
fn dequantize_derivatives_are_linear_in_scales_and_biases() {
    let mut rng = Rng::new(0x54);
    let w = random(&mut rng, &[4, 64], DType::F32);
    let (wq, s, b) = parts(aq(&w, 64, 3));
    let f = |xs: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![qw(&xs[0], &xs[1], &xs[2], 64, 3).dequantize(None)])
    };
    let g = random(&mut rng, &[4, 64], DType::F32);
    let wq2 = wq.clone();
    let (_, grads) = vjp_flat(
        move |xs: &[Array]| f(&[wq2.clone(), xs[0].clone(), xs[1].clone()]),
        &[s.clone(), b.clone()],
        std::slice::from_ref(&g),
    )
    .unwrap();
    let codes = qw(
        &wq,
        &ops::ones([4, 1], DType::F32),
        &ops::zeros([4, 1], DType::F32),
        64,
        3,
    )
    .dequantize(None);
    assert_close("ds", &grads[0], &(&g * &codes).sum_keepdims(-1), 1e-6);
    assert_close("db", &grads[1], &g.sum_keepdims(-1), 1e-6);
}

#[test]
fn vmap_matches_a_loop() {
    let mut rng = Rng::new(0x55);
    let w = random(&mut rng, &[3, 8, 64], DType::F32);
    let x = random(&mut rng, &[3, 2, 64], DType::F32);
    // Batched weights and activations.
    let f = |xs: &[Array]| -> Result<Vec<Array>> {
        let (wq, s, b) = parts(aq(&xs[0], 32, 4));
        Ok(vec![xs[1].quantized_matmul(&qw(&wq, &s, &b, 32, 4), true)])
    };
    let got = vmap_flat(f, &[Some(0), Some(0)], &[0])(&[w.clone(), x.clone()]).unwrap();
    for i in 0..3 {
        let pick = |a: &Array| a.i(i..i + 1);
        let (wi, xi) = (pick(&w), pick(&x));
        let want = f(&[wi.squeeze(0), xi.squeeze(0)]).unwrap();
        let gi = pick(&got[0]).squeeze(0);
        assert_close("vmap", &gi, &want[0], 1e-6);
    }
    // Shared weights, batched activations on the last axis.
    let (wq, s, b) = parts(aq(&w.reshape([24, 64]), 64, 8));
    let xt = random(&mut rng, &[64, 5], DType::F32);
    let g = move |xs: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![xs[0].quantized_matmul(&qw(&wq, &s, &b, 64, 8), true)])
    };
    let got = vmap_flat(g.clone(), &[Some(1)], &[0])(std::slice::from_ref(&xt)).unwrap();
    let want = g(&[xt.t()]).unwrap();
    assert_close("vmap shared", &got[0], &want[0], 1e-6);
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;

    fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
        with_stream(device, f)
    }

    fn bits_of(a: &Array) -> Vec<u32> {
        match a.dtype() {
            DType::U32 => a.to_vec::<u32>().unwrap(),
            DType::F32 => a
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
            DType::F16 => a
                .to_vec::<f16>()
                .unwrap()
                .iter()
                .map(|x| u32::from(x.to_bits()))
                .collect(),
            _ => a
                .to_vec::<bf16>()
                .unwrap()
                .iter()
                .map(|x| u32::from(x.to_bits()))
                .collect(),
        }
    }

    /// Codes, scales and biases bit for bit; dequantized values within a rounding of the
    /// terms (`2·eps·(max|scale|·(2^bits − 1) + max|bias|)`): the
    /// GPU computes `code · scale + bias` as one fused multiply-add (MLX's Metal kernel,
    /// pinned by `mlx_gpu.bin`), the CPU rounds twice (MLX's CPU).
    #[test]
    fn quantize_and_dequantize_match_the_cpu_bit_for_bit() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x56);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for bits in BITS {
                for gs in GROUPS {
                    let w = random(&mut rng, &[3, 2 * gs], dtype);
                    // A transposed view exercises the copy to row-contiguous memory.
                    let w = if gs == 64 { w.t().t() } else { w };
                    let run = || {
                        let (wq, s, b) = parts(aq(&w, gs, bits));
                        let d = qw(&wq, &s, &b, gs, bits).dequantize(None);
                        let value = |a: &Array| a.astype(DType::F64).to_vec::<f64>().unwrap();
                        let max = |a: &Array| value(a).iter().fold(0f64, |m, x| m.max(x.abs()));
                        let bound = max(&s) * f64::from((1u32 << bits) - 1) + max(&b);
                        ([&wq, &s, &b].map(bits_of), value(&d), bound)
                    };
                    let (cpu, gpu) = (on(Device::cpu(), run), on(Device::gpu(0), run));
                    for (i, name) in ["codes", "scales", "biases"].iter().enumerate() {
                        assert_eq!(
                            cpu.0[i], gpu.0[i],
                            "{dtype} {bits} bits, group {gs}: {name}"
                        );
                    }
                    let eps = match dtype {
                        DType::F32 => f64::from(f32::EPSILON),
                        DType::F16 => 2f64.powi(-10),
                        _ => 2f64.powi(-7),
                    };
                    for (k, (c, g)) in cpu.1.iter().zip(&gpu.1).enumerate() {
                        assert!(
                            (c - g).abs() <= 2.0 * eps * cpu.2,
                            "{dtype} {bits} bits, group {gs}: dequantized {k}: {c} vs {g}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn quantized_matmul_matches_the_cpu_across_kernels() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x57);
        // (x shape, w rows, w cols, transpose, bits, group): qmv_fast (aligned K, with and
        // without partial rows), qmv_rows, qmv (unaligned K), qvm, qmm_t (aligned and not), split-K,
        // qmm_n, batched and broadcast weights.
        type Case = (&'static [i32], &'static [i32], i32, bool, i32, i32);
        let cases: &[Case] = &[
            (&[1, 512], &[64], 512, true, 4, 64),
            // qmv_rows (4-bit, 4 or more rows): odd rows, partial rows, every group size.
            (&[5, 512], &[64], 512, true, 4, 64),
            (&[4, 1024], &[36], 1024, true, 4, 32),
            (&[8, 512], &[16], 512, true, 4, 128),
            (&[3, 1024], &[36], 1024, true, 4, 32),
            (&[2, 192], &[40], 192, true, 3, 64),
            (&[1, 96], &[13], 96, true, 8, 32),
            (&[4, 256], &[24], 256, true, 6, 128),
            (&[2, 2048], &[16], 2048, true, 1, 64),
            (&[1, 64], &[64], 256, false, 4, 64),
            (&[3, 48], &[48], 128, false, 2, 32),
            (&[40, 256], &[64], 256, true, 4, 64),
            (&[33, 128], &[50], 128, true, 5, 32),
            (&[64, 4096], &[96], 4096, true, 4, 128),
            (&[20, 64], &[64], 96, false, 4, 32),
            (&[2, 17, 128], &[2, 24], 128, true, 4, 32),
            (&[2, 3, 128], &[2, 24], 128, true, 4, 64),
            (&[2, 6, 32], &[2, 32], 64, false, 8, 32),
            (&[3, 20, 128], &[24], 128, true, 2, 32),
        ];
        for &(xs, wb, cols, t, bits, gs) in cases {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let mut wshape = wb.to_vec();
                wshape.push(cols);
                let w = random(&mut rng, &wshape, dtype);
                let x = random(&mut rng, xs, dtype);
                let run = || {
                    let (wq, s, b) = parts(aq(&w, gs, bits));
                    let y = x.quantized_matmul(&qw(&wq, &s, &b, gs, bits), t);
                    y.astype(DType::F32)
                };
                let (cpu, gpu) = (on(Device::cpu(), run), on(Device::gpu(0), run));
                let k = if t { cols } else { *wb.last().unwrap() };
                let eps = match dtype {
                    DType::F32 => 1e-6,
                    DType::F16 => 1e-3,
                    _ => 8e-3,
                };
                let name = format!("{dtype} x{xs:?} w{wshape:?} t={t} {bits}/{gs}");
                assert_close(&name, &gpu, &cpu, eps * f64::from(k).sqrt() * 4.0);
            }
        }
        // Transposed activations are copied; empty products stay on the CPU.
        let w = random(&mut rng, &[16, 64], DType::F32);
        let x = random(&mut rng, &[64, 5], DType::F32);
        let run = || {
            let (wq, s, b) = parts(aq(&w, 64, 4));
            x.t().quantized_matmul(&qw(&wq, &s, &b, 64, 4), true)
        };
        assert_close(
            "strided x",
            &on(Device::gpu(0), run),
            &on(Device::cpu(), run),
            1e-5,
        );
        let empty = on(Device::gpu(0), || {
            let (wq, s, b) = parts(aq(&w, 64, 4));
            ops::zeros([0, 64], DType::F32).quantized_matmul(&qw(&wq, &s, &b, 64, 4), true)
        });
        assert_eq!(empty.shape(), [0, 16]);
        empty.eval().unwrap();
    }
}

/// `gather_qmm` against per-batch `quantized_matmul` of the selected batches (CPU), with
/// explicit and default indices, broadcasting, and gradients and batching for `x`.
#[test]
fn gather_qmm_selects_batches() {
    let mut rng = Rng::new(0x71);
    let (e, n, k) = (3, 8, 64);
    let w = random(&mut rng, &[e, n, k], DType::F32);
    let (wq, s, b) = parts(aq(&w, 32, 4));
    let x = random(&mut rng, &[4, 2, k], DType::F32);
    let lhs = Array::from([3u32, 0, 2]);
    let rhs = Array::from([1u32, 2, 0]);
    let got = x.gather_qmm(
        &qw(&wq, &s, &b, 32, 4),
        Some(&lhs),
        Some(&rhs),
        GatherQmm::default(),
    );
    assert_eq!(got.shape(), [3, 2, n]);
    let pick = |a: &Array, i: u32| a.take(&Array::from(i), 0);
    for (j, (&l, &r)) in [3u32, 0, 2].iter().zip(&[1u32, 2, 0]).enumerate() {
        let want = pick(&x, l)
            .quantized_matmul(&qw(&pick(&wq, r), &pick(&s, r), &pick(&b, r), 32, 4), true);
        assert_close(&format!("batch {j}"), &pick(&got, j as u32), &want, 1e-6);
    }
    // Default indices: every batch in order (x's batch broadcast against the weights').
    let x1 = random(&mut rng, &[1, 1, k], DType::F32);
    let all = x1.gather_qmm(&qw(&wq, &s, &b, 32, 4), None, None, GatherQmm::default());
    assert_eq!(all.shape(), [3, 1, n]);
    // Gradient with respect to x: a row used by two batches gets both contributions.
    let rows = Array::from([1u32, 1]);
    let experts = Array::from([0u32, 1]);
    let x2 = random(&mut rng, &[2, 1, k], DType::F32);
    let f = |xs: &[Array]| {
        let y = xs[0].gather_qmm(
            &qw(&wq, &s, &b, 32, 4),
            Some(&rows),
            Some(&experts),
            GatherQmm::default(),
        );
        Ok(vec![y.reshape([-1])])
    };
    let ones = ops::full([2 * n], 1.0f32);
    let (_, grads) = vjp_flat(f, std::slice::from_ref(&x2), std::slice::from_ref(&ones)).unwrap();
    let column_sums = |i: u32| {
        qw(&pick(&wq, i), &pick(&s, i), &pick(&b, i), 32, 4)
            .dequantize(None)
            .sum(0)
    };
    let row = |i: u32| pick(&grads[0], i).reshape([-1]);
    assert_close("dx row 0", &row(0), &ops::zeros([k], DType::F32), 1e-6);
    let want = column_sums(0) + column_sums(1);
    assert_close("dx row 1", &row(1), &want, 1e-5);
    // Batching over a leading axis of x matches a loop.
    let xs = random(&mut rng, &[2, 4, 2, k], DType::F32);
    let f = |a: &[Array]| {
        Ok(vec![a[0].gather_qmm(
            &qw(&wq, &s, &b, 32, 4),
            Some(&lhs),
            Some(&rhs),
            GatherQmm::default(),
        )])
    };
    let batched = vmap_flat(f, &[Some(0)], &[0])(std::slice::from_ref(&xs)).unwrap();
    for i in 0..2u32 {
        let one = f(&[pick(&xs, i)]).unwrap();
        assert_close(&format!("vmap {i}"), &pick(&batched[0], i), &one[0], 1e-6);
    }
    // Errors: float indices, mismatched index shapes.
    let bad = Array::from([0.0f32]);
    assert!(
        x.gather_qmm(
            &qw(&wq, &s, &b, 32, 4),
            Some(&bad),
            None,
            GatherQmm::default()
        )
        .check()
        .is_err()
    );
    let two = Array::from([0u32, 1]);
    assert!(
        x.gather_qmm(
            &qw(&wq, &s, &b, 32, 4),
            Some(&lhs),
            Some(&two),
            GatherQmm::default()
        )
        .check()
        .is_err()
    );
}

/// `gather_mm` against `matmul` of the selected batches (CPU), with explicit and default
/// indices, gradients for both operands, batching, and errors.
#[test]
fn gather_mm_selects_batches() {
    let mut rng = Rng::new(0x73);
    let (e, n, k) = (3, 5, 7);
    let w = random(&mut rng, &[e, k, n], DType::F32);
    let x = random(&mut rng, &[4, 2, k], DType::F32);
    let lhs = Array::from([3u32, 0, 2]);
    let rhs = Array::from([1u32, 2, 0]);
    let got = x.gather_mm(&w, Some(&lhs), Some(&rhs), false);
    assert_eq!(got.shape(), [3, 2, n]);
    let pick = |a: &Array, i: u32| a.take(&Array::from(i), 0);
    for (j, (&l, &r)) in [3u32, 0, 2].iter().zip(&[1u32, 2, 0]).enumerate() {
        let want = pick(&x, l).matmul(&pick(&w, r));
        assert_close(&format!("batch {j}"), &pick(&got, j as u32), &want, 1e-6);
    }
    // Default indices: every batch in order (equal to a batched matmul).
    let x3 = random(&mut rng, &[3, 2, k], DType::F32);
    let all = x3.gather_mm(&w, None, None, false);
    assert_close("defaults", &all, &x3.matmul(&w), 1e-6);
    // Gradients: a row used by two batches, a weight batch used by none.
    let rows = Array::from([1u32, 1]);
    let experts = Array::from([0u32, 2]);
    let x2 = random(&mut rng, &[2, 1, k], DType::F32);
    let f = |xs: &[Array]| {
        let y = xs[0].gather_mm(&xs[1], Some(&rows), Some(&experts), false);
        Ok(vec![y.reshape([-1])])
    };
    let ones = ops::full([2 * n], 1.0f32);
    let (_, grads) = vjp_flat(f, &[x2.clone(), w.clone()], std::slice::from_ref(&ones)).unwrap();
    let row_sums = |i: u32| pick(&w, i).sum(1);
    let row = |i: u32| pick(&grads[0], i).reshape([-1]);
    assert_close("dx row 0", &row(0), &ops::zeros([k], DType::F32), 1e-6);
    assert_close("dx row 1", &row(1), &(row_sums(0) + row_sums(2)), 1e-5);
    let xr = pick(&x2, 1).reshape([k, 1]);
    let outer = xr.broadcast_to([k, n]);
    assert_close("dw 0", &pick(&grads[1], 0), &outer, 1e-6);
    assert_close(
        "dw 1",
        &pick(&grads[1], 1),
        &ops::zeros([k, n], DType::F32),
        1e-6,
    );
    // Batching over a leading axis of x matches a loop.
    let xs = random(&mut rng, &[2, 4, 2, k], DType::F32);
    let f = |a: &[Array]| Ok(vec![a[0].gather_mm(&w, Some(&lhs), Some(&rhs), false)]);
    let batched = vmap_flat(f, &[Some(0)], &[0])(std::slice::from_ref(&xs)).unwrap();
    for i in 0..2u32 {
        let one = f(&[pick(&xs, i)]).unwrap();
        assert_close(&format!("vmap {i}"), &pick(&batched[0], i), &one[0], 1e-6);
    }
    // Errors: float indices, mismatched inner dimensions, integer operands.
    let bad = Array::from([0.0f32]);
    assert!(x.gather_mm(&w, Some(&bad), None, false).check().is_err());
    assert!(x.gather_mm(&x, None, None, false).check().is_err());
    let int = Array::from_slice(&[1i32; 4], [1, 2, 2]);
    assert!(int.gather_mm(&int, None, None, false).check().is_err());
}

#[cfg(target_os = "macos")]
mod gather_gpu {
    use super::*;

    fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
        with_stream(device, f)
    }

    /// Every gather kernel (matrix-vector, tiled, grouped over sorted rows) against the CPU.
    #[test]
    fn gather_qmm_kernels_match_the_cpu() {
        if cfg!(miri) || on(Device::gpu(0), || (Array::from(1.0f32) * 2.0).eval()).is_err() {
            return;
        }
        let mut rng = Rng::new(0x72);
        // (experts, N, K, rows M per batch, batches, sorted)
        let cases: &[(i32, i32, i32, i32, i32, bool)] = &[
            (8, 64, 512, 1, 6, false),  // qmv_fast
            (8, 36, 192, 1, 5, false),  // qmv (N not a multiple of 8, K not of 512)
            (4, 64, 256, 40, 3, false), // qmm_t
            (4, 40, 128, 33, 2, false), // qmm_t, unaligned
            (8, 64, 256, 1, 100, true), // grouped, sorted rows
            (8, 64, 256, 1, 20, true),  // sorted, too few rows per expert to group (MLX's rule)
            (16, 48, 192, 1, 70, true), // grouped, unaligned N
        ];
        for &(e, n, k, m, batches, sorted) in cases {
            for (dtype, gs, bits) in [
                (DType::F32, 32, 4),
                (DType::BF16, 64, 4),
                (DType::F16, 32, 8),
            ] {
                let w = random(&mut rng, &[e, n, k], dtype);
                let (wq, s, b) = parts(aq(&w, gs, bits));
                let x = random(&mut rng, &[batches, m, k], dtype);
                let mut ids: Vec<u32> = (0..batches).map(|i| ((i * 7 + 3) % e) as u32).collect();
                if sorted {
                    ids.sort_unstable();
                }
                let rhs = Array::from_slice(&ids, [batches]);
                let run = || {
                    x.gather_qmm(
                        &qw(&wq, &s, &b, gs, bits),
                        None,
                        Some(&rhs),
                        GatherQmm::default().sorted_indices(sorted),
                    )
                };
                let (gpu, cpu) = (on(Device::gpu(0), run), on(Device::cpu(), run));
                let tol = match dtype {
                    DType::F32 => 1e-5,
                    _ => 1e-2,
                };
                let name = format!("{dtype} gs{gs} b{bits} {:?}", (e, n, k, m, batches, sorted));
                assert_close(&name, &gpu, &cpu, tol);
            }
        }
        // Explicit row indices with a broadcast batch shape.
        let w = random(&mut rng, &[4, 32, 128], DType::F32);
        let (wq, s, b) = parts(aq(&w, 32, 4));
        let x = random(&mut rng, &[3, 1, 128], DType::F32);
        let lhs = Array::from_slice(&[2u32, 0, 1, 1], [2, 2]);
        let rhs = Array::from_slice(&[3u32, 1], [1, 2]);
        let run = || {
            x.gather_qmm(
                &qw(&wq, &s, &b, 32, 4),
                Some(&lhs),
                Some(&rhs),
                GatherQmm::default(),
            )
        };
        let (gpu, cpu) = (on(Device::gpu(0), run), on(Device::cpu(), run));
        assert_close("broadcast indices", &gpu, &cpu, 1e-5);
    }

    /// Both gather GEMM kernels (tiled over any indices, grouped over sorted rows), with
    /// transposed and unaligned operands, against the CPU.
    #[test]
    fn gather_mm_kernels_match_the_cpu() {
        if cfg!(miri) || on(Device::gpu(0), || (Array::from(1.0f32) * 2.0).eval()).is_err() {
            return;
        }
        let mut rng = Rng::new(0x74);
        // (experts, N, K, rows M per batch, batches, sorted, b transposed)
        let cases: &[(i32, i32, i32, i32, i32, bool, bool)] = &[
            (4, 64, 128, 64, 3, false, false), // aligned
            (4, 40, 72, 33, 2, false, true),   // unaligned, b transposed
            (8, 24, 40, 1, 5, false, false),   // single rows, unsorted
            (8, 64, 256, 1, 100, true, false), // grouped, sorted rows
            (16, 48, 200, 1, 70, true, true),  // grouped, unaligned, b transposed
        ];
        for &(e, n, k, m, batches, sorted, t) in cases {
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                let w = match t {
                    true => random(&mut rng, &[e, n, k], dtype).transpose([0, 2, 1]),
                    false => random(&mut rng, &[e, k, n], dtype),
                };
                let x = random(&mut rng, &[batches, m, k], dtype);
                let mut ids: Vec<u32> = (0..batches).map(|i| ((i * 7 + 3) % e) as u32).collect();
                if sorted {
                    ids.sort_unstable();
                }
                let rhs = Array::from_slice(&ids, [batches]);
                let run = || x.gather_mm(&w, None, Some(&rhs), sorted);
                let (gpu, cpu) = (on(Device::gpu(0), run), on(Device::cpu(), run));
                let tol = match dtype {
                    DType::F32 => 1e-5,
                    _ => 1e-2,
                };
                let name = format!("{dtype} {:?}", (e, n, k, m, batches, sorted, t));
                assert_close(&name, &gpu, &cpu, tol);
            }
        }
        // Explicit row indices with a broadcast batch shape, and a transposed `a`.
        let w = random(&mut rng, &[4, 96, 32], DType::F32);
        let x = random(&mut rng, &[3, 96, 8], DType::F32).transpose([0, 2, 1]);
        let lhs = Array::from_slice(&[2u32, 0, 1, 1], [2, 2]);
        let rhs = Array::from_slice(&[3u32, 1], [1, 2]);
        let run = || x.gather_mm(&w, Some(&lhs), Some(&rhs), false);
        let (gpu, cpu) = (on(Device::gpu(0), run), on(Device::cpu(), run));
        assert_close("broadcast indices", &gpu, &cpu, 1e-5);
    }
}

/// Few-row 4-bit products (nuvec's `qmv_rows`) equal the one-row product of each row bit for
/// bit, so results do not depend on how many rows share a product (speculative
/// verification and batching give plain decoding's numbers).
#[cfg(target_os = "macos")]
#[test]
fn few_row_products_equal_one_row_products_bit_for_bit() {
    nuvec_core::with_row_exact(|| {
        if cfg!(miri) {
            return;
        }
        with_stream(Device::gpu(0), || {
            if (&Array::from(1.0f32) + 1.0).eval().is_err() {
                return;
            }
            // (rows, K, N, group): odd and even rows, partial output rows (N % 8 != 0).
            for (m, k, n, gs) in [(6, 4096, 512, 64), (5, 1024, 1000, 32), (4, 2048, 256, 128)] {
                for dtype in [DType::BF16, DType::F16, DType::F32] {
                    let w = random::normal([n, k], dtype, 0.0, 1.0, &random::key(1));
                    let (wq, s, b) = parts(aq(&w, gs, 4));
                    let x = random::normal([m, k], dtype, 0.0, 1.0, &random::key(2));
                    let bits = |a: Array| -> Vec<u32> {
                        let v = a.astype(DType::F32).to_vec::<f32>().unwrap();
                        v.into_iter().map(f32::to_bits).collect()
                    };
                    let all = bits(x.quantized_matmul(&qw(&wq, &s, &b, gs, 4), true));
                    let n = n as usize;
                    for r in 0..m {
                        let xr = x.i((r..r + 1, ..));
                        let one = bits(xr.quantized_matmul(&qw(&wq, &s, &b, gs, 4), true));
                        let r = r as usize;
                        assert!(
                            one == all[r * n..(r + 1) * n],
                            "{dtype} {:?}: row {r} differs",
                            (m, k, n, gs)
                        );
                    }
                }
            }
        });
    });
}

/// The floating-point modes with their group sizes.
const FP_MODES: [(QuantMode, i32); 3] = [
    (QuantMode::Mxfp4, 32),
    (QuantMode::Mxfp8, 32),
    (QuantMode::Nvfp4, 16),
];

/// MLX's validation of the floating-point modes: sticky errors with MLX's wording, never a
/// panic.
#[test]
fn fp_invalid_params_error() {
    let message = |a: &Array| a.clone().check().unwrap_err().to_string();
    let w = ops::full([2, 64], 1.0f32);
    let q = w.quantize(Quant::new(QuantMode::Mxfp4).group_size(64));
    assert!(message(&q.weight).contains("mxfp4 quantization requires group size 32 but got 64"));
    assert!(q.biases.is_none());
    let q = w.quantize(Quant::new(QuantMode::Nvfp4).bits(8));
    assert!(message(&q.scales).contains("nvfp4 quantization requires bits to be 4 but got 8"));
    let good = w.quantize(QuantMode::Mxfp8);
    good.check().unwrap();
    let with_biases = Quantized::new(
        good.weight.clone(),
        good.scales.clone(),
        Some(ops::zeros([2, 2], DType::F32)),
        QuantMode::Mxfp8,
    );
    assert!(message(&with_biases.dequantize(None)).contains("biases must be null"));
    let affine = Quantized::new(
        good.weight.clone(),
        good.scales.clone(),
        None,
        Quant::affine(32, 8),
    );
    assert!(message(&affine.dequantize(None)).contains("biases must be provided"));
    let float_scales = Quantized::new(
        good.weight.clone(),
        good.scales.astype(DType::F32),
        None,
        QuantMode::Mxfp8,
    );
    assert!(message(&float_scales.dequantize(None)).contains("scale type must be uint8"));
    assert!(
        good.dequantize(DType::I32).check().is_err(),
        "integer output"
    );
    assert!(
        w.astype(DType::F64)
            .quantize(QuantMode::Mxfp4)
            .check()
            .is_err(),
        "float64"
    );
    let x = ops::full([3, 64], 1.0f32).astype(DType::F64);
    assert!(
        x.quantized_matmul(&good, true).check().is_err(),
        "float64 activations"
    );
    assert!(
        ops::full([3, 63], 1.0f32)
            .quantized_matmul(&good, true)
            .check()
            .is_err(),
        "inner size"
    );
    assert_eq!("nvfp4".parse::<QuantMode>().unwrap(), QuantMode::Nvfp4);
    assert_eq!(QuantMode::Mxfp8.to_string(), "mxfp8");
    assert!("fp4".parse::<QuantMode>().is_err());
}

#[cfg(target_os = "macos")]
mod fp_gpu {
    use super::*;

    /// Floating-point products on the GPU over layouts that are not plain matrices: sliced
    /// and transposed activations with offsets, broadcast batches, and weights sliced along
    /// N (rows of the codes and scales), against the CPU, every mode and dtype.
    #[test]
    fn fp_products_random_layouts() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x5a5);
        for (mode, gs) in FP_MODES {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let k = 4 * gs;
                let w = random(&mut rng, &[48, k], dtype);
                let q = w.quantize(mode);
                // Rows 8..40 of the codes and scales: an N-slice with an offset.
                let sliced = Quantized::new(q.weight.i(8..40), q.scales.i(8..40), None, q.quant);
                let wide = random(&mut rng, &[2, 7, k + 32], dtype);
                let x_sliced = wide.i((.., 1..6, 16..16 + k)); // offset, row stride k + 32
                let x_t = random(&mut rng, &[k, 5], dtype).t(); // transposed
                let x_b = random(&mut rng, &[1, 5, k], dtype).broadcast_to([3, 5, k]);
                for (name, x) in [
                    ("sliced", &x_sliced),
                    ("transposed", &x_t),
                    ("broadcast", &x_b),
                ] {
                    for (wname, wq) in [("w", &q), ("w[8..40]", &sliced)] {
                        let run = || x.quantized_matmul(wq, true).astype(DType::F32);
                        let (gpu, cpu) = (
                            with_stream(Device::gpu(0), run),
                            with_stream(Device::cpu(), run),
                        );
                        let tol = match dtype {
                            DType::F32 => 1e-5,
                            DType::F16 => 2e-3,
                            _ => 1.6e-2,
                        };
                        assert_close(&format!("{mode} {dtype} {name} {wname}"), &gpu, &cpu, tol);
                    }
                }
            }
        }
    }

    /// Empty products: no rows, no output columns, and gathers over no indices give empty
    /// results of the right shape on both devices, every mode.
    #[test]
    fn fp_products_empty() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        for (mode, gs) in FP_MODES {
            let k = 2 * gs;
            let q = ops::full([8, k], 0.5f32).quantize(mode);
            let none = ops::full([0, k], 0.5f32).quantize(mode);
            let experts = ops::full([2, 8, k], 0.5f32).quantize(mode);
            for device in [Device::gpu(0), Device::cpu()] {
                with_stream(device, || {
                    let rows = ops::zeros([0, k], DType::F32).quantized_matmul(&q, true);
                    assert_eq!(rows.shape(), [0, 8], "{mode} no rows");
                    rows.eval().unwrap();
                    let cols = ops::full([3, k], 1.0f32).quantized_matmul(&none, true);
                    assert_eq!(cols.shape(), [3, 0], "{mode} no columns");
                    cols.eval().unwrap();
                    let ids = Array::from_slice::<u32>(&[], [0]);
                    let x = ops::zeros([0, 1, k], DType::F32);
                    let g = x.gather_qmm(&experts, None, Some(&ids), GatherQmm::default());
                    assert_eq!(g.shape(), [0, 1, 8], "{mode} no indices");
                    g.eval().unwrap();
                });
            }
        }
    }

    /// Floating-point gathers over layouts that are not plain: sliced and transposed
    /// activations with offsets, broadcast row indices, and experts sliced along N and along
    /// the expert axis, sorted and not, against the CPU, every mode and dtype.
    #[test]
    fn fp_gather_random_layouts() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x5a7);
        for (mode, gs) in FP_MODES {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let k = 4 * gs;
                let w = random(&mut rng, &[6, 40, k], dtype);
                let q = w.quantize(mode);
                // Experts 1..5 and rows 4..36 of each: offsets on two axes.
                let sliced = Quantized::new(
                    q.weight.i((1..5, 4..36)),
                    q.scales.i((1..5, 4..36)),
                    None,
                    q.quant,
                );
                let wide = random(&mut rng, &[9, 1, k + 32], dtype);
                let x_sliced = wide.i((1..9, .., 16..16 + k)); // offset, stride k + 32
                let x_t = random(&mut rng, &[k, 8], dtype).t().reshape([8, 1, k]);
                let x_b = random(&mut rng, &[1, 1, k], dtype).broadcast_to([8, 1, k]);
                let ids: Vec<u32> = (0..8).map(|i| (i * 3 + 1) % 4).collect();
                let mut sorted_ids = ids.clone();
                sorted_ids.sort_unstable();
                for (name, x) in [
                    ("sliced", &x_sliced),
                    ("transposed", &x_t),
                    ("broadcast", &x_b),
                ] {
                    for (wname, wq) in [("w[..4]", &q), ("w[1..5, 4..36]", &sliced)] {
                        for (sorted, ids) in [(false, &ids), (true, &sorted_ids)] {
                            let rhs = Array::from_slice(ids, [8]);
                            let options = GatherQmm::default().sorted_indices(sorted);
                            let run = || {
                                x.gather_qmm(wq, None, Some(&rhs), options)
                                    .astype(DType::F32)
                            };
                            let (gpu, cpu) = (
                                with_stream(Device::gpu(0), run),
                                with_stream(Device::cpu(), run),
                            );
                            let tol = match dtype {
                                DType::F32 => 1e-5,
                                DType::F16 => 2e-3,
                                _ => 1.6e-2,
                            };
                            let label = format!("{mode} {dtype} {name} {wname} sorted={sorted}");
                            assert_close(&label, &gpu, &cpu, tol);
                        }
                    }
                }
                // Broadcast row indices: one activation row per index pair.
                let lhs = Array::from_slice(&[0u32, 2, 1, 1], [2, 2]);
                let rhs = Array::from_slice(&[3u32, 0], [1, 2]);
                let x = random(&mut rng, &[3, 1, k], dtype);
                let run = || {
                    x.gather_qmm(&q, Some(&lhs), Some(&rhs), GatherQmm::default())
                        .astype(DType::F32)
                };
                let (gpu, cpu) = (
                    with_stream(Device::gpu(0), run),
                    with_stream(Device::cpu(), run),
                );
                assert_close(
                    &format!("{mode} {dtype} broadcast indices"),
                    &gpu,
                    &cpu,
                    2e-2,
                );
            }
        }
    }

    /// Every floating-point kernel family is instantiated for every mode and dtype and for
    /// both values of each 0/1 parameter (`batch`, `al`, `fast`, `pr`; a partial last row
    /// block, `pr1`, only on the fast path), so no combination the dispatch can build is
    /// missing (MLX #2887's failure: a kernel absent for one dtype). The `_hgs` gathers
    /// (global scale) and `qmv_fast`'s `r2` (two rows per simdgroup) are nvfp4's only.
    #[test]
    fn fp_kernels_exist() {
        let Some(device) = nuvec_gpu::Device::shared() else {
            return;
        };
        let names: std::collections::BTreeSet<String> =
            device.kernel_names().unwrap().into_iter().collect();
        let modes = ["mxfp4", "mxfp8", "nvfp4"];
        let dtypes = ["float32", "float16", "bfloat16"];
        let mut families = std::collections::BTreeSet::new();
        for name in &names {
            let Some(mode) = modes
                .iter()
                .find(|m| name.starts_with(&format!("nuvec_{m}_")))
            else {
                continue;
            };
            let Some(dtype) = dtypes.iter().find(|d| name.contains(&format!("_{d}"))) else {
                continue;
            };
            families.insert(name.replacen(mode, "{m}", 1).replacen(
                &format!("_{dtype}"),
                "_{t}",
                1,
            ));
        }
        for kind in [
            "quantize",
            "dequantize",
            "qmv_",
            "qmv_fast",
            "qmv_quad",
            "qmv_wide",
            "qmv_rows",
            "qmm_t",
            "qmm_n",
            "qvm",
            "gather_qmv",
            "gather_qmm_t",
            "gather_qmm_rhs",
        ] {
            assert!(
                families
                    .iter()
                    .any(|f| f.starts_with(&format!("nuvec_{{m}}_{kind}"))),
                "no {kind} kernels"
            );
        }
        let mut variants = std::collections::BTreeSet::new();
        for family in &families {
            variants.insert(family.clone());
            for flag in ["batch", "al", "fast", "pr"] {
                for (on, off) in [("1", "0"), ("0", "1")] {
                    let bit = format!("_{flag}{on}");
                    if let Some(at) = family.find(&bit) {
                        let end = at + bit.len();
                        if !family[end..].starts_with(|c: char| c.is_ascii_digit()) {
                            variants.insert(format!(
                                "{}_{flag}{off}{}",
                                &family[..at],
                                &family[end..]
                            ));
                        }
                    }
                }
            }
        }
        let mut missing = Vec::new();
        for family in &variants {
            if family.contains("_fast0_pr1") {
                continue;
            }
            let nvfp4_only = family.contains("_hgs")
                || (family.contains("_qmv_fast_") && family.ends_with("_r2"));
            for mode in modes.iter().filter(|&&m| !nvfp4_only || m == "nvfp4") {
                for dtype in dtypes {
                    let name = family.replacen("{m}", mode, 1).replacen("{t}", dtype, 1);
                    if !names.contains(&name) {
                        missing.push(name);
                    }
                }
            }
        }
        assert!(missing.is_empty(), "missing kernels: {missing:?}");
    }

    fn bytes(a: &Array) -> Vec<u8> {
        match a.dtype() {
            DType::U8 => a.to_vec::<u8>().unwrap(),
            DType::U32 => a
                .to_vec::<u32>()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            _ => a
                .astype(DType::F32)
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        }
    }

    /// Every route of the floating-point products (plan D3 Table 1 and the per-row
    /// untransposed region) on the GPU against the CPU reference.
    #[test]
    fn fp_products_match_cpu() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x5a1);
        // (x shape, weight batch, K, N, transpose): qmv_fast, qmv (K = 2880), qmv_quad (64,
        // 128), per-row (partial N), qmv_wide (between R and MLX's limit), qmv_quad above R,
        // split-K, qmm_t (batched), batched matrix-vector, untransposed qvm and qmm_n.
        type Case = (&'static [i32], &'static [i32], i32, i32, bool);
        let cases: &[Case] = &[
            (&[1, 4096], &[], 4096, 64, true),
            (&[1, 2880], &[], 2880, 72, true),
            (&[1, 128], &[], 128, 40, true),
            (&[2, 64], &[], 64, 24, true),
            (&[3, 2880], &[], 2880, 36, true),
            (&[5, 4096], &[], 4096, 30, true),
            (&[8, 512], &[], 512, 16, true),
            (&[11, 1024], &[], 1024, 40, true),
            (&[12, 128], &[], 128, 64, true),
            (&[40, 512], &[], 512, 64, true),
            (&[64, 2880], &[], 2880, 96, true),
            (&[2, 20, 512], &[2], 512, 24, true),
            (&[2, 3, 512], &[2], 512, 24, true),
            (&[1, 64], &[], 64, 128, false),
            (&[3, 96], &[], 96, 64, false),
            (&[12, 64], &[], 64, 96, false),
        ];
        for (mode, gs) in FP_MODES {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                for &(xs, wb, k, n, t) in cases {
                    let mut wshape = wb.to_vec();
                    if t {
                        wshape.extend([n, k]);
                    } else {
                        if n % gs != 0 {
                            continue;
                        }
                        wshape.extend([k, n]);
                    }
                    let w = random(&mut rng, &wshape, dtype);
                    let x = random(&mut rng, xs, dtype);
                    let run = || {
                        let q = w.quantize(mode);
                        x.quantized_matmul(&q, t).astype(DType::F32)
                    };
                    let (cpu, gpu) = (
                        with_stream(Device::cpu(), run),
                        with_stream(Device::gpu(0), run),
                    );
                    let tol = match dtype {
                        DType::F32 => 1e-5,
                        DType::F16 => 2e-3,
                        _ => 1.6e-2,
                    };
                    let name = format!("{mode} {dtype} x{xs:?} w{wshape:?} t={t}");
                    assert_close(&name, &gpu, &cpu, tol);
                }
            }
        }
    }

    /// Up to `ROW_INDEPENDENT_ROWS` rows, each row of a floating-point product equals the
    /// one-row product bit for bit (aligned and unaligned `K`, partial `N`).
    #[test]
    fn fp_rows_equal_one_row_products() {
        nuvec_core::with_row_exact(|| {
            if crate::common::require_gpu().is_none() {
                return;
            }
            with_stream(Device::gpu(0), || {
                for (mode, _) in FP_MODES {
                    // Transposed, then untransposed (split-K from K = 1024, 32 partitions above
                    // K = 8192).
                    let shapes = [
                        (true, 4096, 64),
                        (true, 4096, 30),
                        (true, 2880, 36),
                        (true, 2880, 6),
                        (true, 128, 40),
                        (false, 4096, 64),
                        (false, 2880, 32),
                        (false, 9216, 32),
                        (false, 512, 64),
                    ];
                    for (t, k, n) in shapes {
                        for dtype in [DType::BF16, DType::F16] {
                            let shape = if t { [n, k] } else { [k, n] };
                            let w = random::normal(shape, dtype, 0.0, 1.0, &random::key(3));
                            let q = w.quantize(mode);
                            for m in 2..=ROW_INDEPENDENT_ROWS as i32 {
                                let x = random::normal([m, k], dtype, 0.0, 1.0, &random::key(4));
                                let all = bits(&x.quantized_matmul(&q, t));
                                for r in 0..m {
                                    let one = bits(&x.i(r..r + 1).quantized_matmul(&q, t));
                                    let (r, n) = (r as usize, n as usize);
                                    assert!(
                                        one == all[r * n..(r + 1) * n],
                                        "{mode} {dtype} K={k} N={n} transpose {t}: row {r} of {m} differs"
                                    );
                                }
                            }
                        }
                    }
                }
            });
        });
    }

    /// Every gather route of the floating-point modes (grouped over sorted rows, per-row
    /// matrix-vector, tiled, untransposed) on the GPU against the CPU reference, with nvfp4's
    /// per-expert global scale on each.
    #[test]
    fn fp_gather_match_cpu() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x5a2);
        // (experts, N, K, rows per batch, batches, sorted, transpose)
        type Case = (i32, i32, i32, i32, i32, bool, bool);
        let cases: &[Case] = &[
            (8, 64, 512, 1, 1, false, true),
            (8, 64, 512, 1, 2, false, true),
            (8, 36, 2880, 1, 5, false, true),
            (8, 64, 512, 1, 16, false, true),
            (8, 64, 512, 1, 64, true, true),
            (8, 64, 256, 1, 100, true, true),
            (16, 48, 192, 1, 70, true, true),
            (4, 64, 256, 40, 3, false, true),
            (32, 96, 2880, 1, 4, false, true),
            (32, 32, 2880, 1, 128, true, true),
            (8, 64, 128, 1, 3, false, false),
            (8, 64, 128, 12, 2, false, false),
            (8, 64, 128, 1, 100, true, false),
        ];
        for (mode, _) in FP_MODES {
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                for &(e, n, k, m, batches, sorted, t) in cases {
                    let wshape = if t { [e, n, k] } else { [e, k, n] };
                    let w = random(&mut rng, &wshape, dtype);
                    let x = random(&mut rng, &[batches, m, k], dtype);
                    let mut ids: Vec<u32> =
                        (0..batches).map(|i| ((i * 7 + 3) % e) as u32).collect();
                    if sorted {
                        ids.sort_unstable();
                    }
                    let rhs = Array::from_slice(&ids, [batches]);
                    let scale: Vec<f32> = (0..e).map(|i| 0.25 + i as f32 / 8.0).collect();
                    let scale = Array::from_slice(&scale, [e]);
                    for global in [false, mode == QuantMode::Nvfp4] {
                        let run = || {
                            let q = w.quantize(mode);
                            let mut options =
                                GatherQmm::default().transpose(t).sorted_indices(sorted);
                            if global {
                                options = options.global_scale(&scale);
                            }
                            x.gather_qmm(&q, None, Some(&rhs), options)
                                .astype(DType::F32)
                        };
                        let (gpu, cpu) = (
                            with_stream(Device::gpu(0), run),
                            with_stream(Device::cpu(), run),
                        );
                        let tol = match dtype {
                            DType::F32 => 1e-5,
                            DType::F16 => 2e-3,
                            _ => 1.6e-2,
                        };
                        let name = format!(
                            "{mode} {dtype} global={global} {:?}",
                            (e, n, k, m, batches, sorted, t)
                        );
                        assert_close(&name, &gpu, &cpu, tol);
                    }
                }
            }
        }
        // Explicit row indices with a broadcast batch shape.
        let w = random(&mut rng, &[4, 32, 128], DType::F32);
        let x = random(&mut rng, &[3, 1, 128], DType::F32);
        let lhs = Array::from_slice(&[2u32, 0, 1, 1], [2, 2]);
        let rhs = Array::from_slice(&[3u32, 1], [1, 2]);
        let run = || {
            let q = w.quantize(QuantMode::Mxfp4);
            x.gather_qmm(&q, Some(&lhs), Some(&rhs), GatherQmm::default())
        };
        let (gpu, cpu) = (
            with_stream(Device::gpu(0), run),
            with_stream(Device::cpu(), run),
        );
        assert_close("broadcast indices", &gpu, &cpu, 1e-5);
    }

    /// The global scale's validation (MLX's messages): nvfp4 only, float32, one per expert,
    /// with indices.
    #[test]
    fn fp_gather_errors() {
        let w = ops::full([4, 16, 64], 1.0f32);
        let x = ops::full([2, 1, 64], 1.0f32);
        let ids = Array::from([0u32, 3]);
        let nv = w.quantize(QuantMode::Nvfp4);
        let mx = w.quantize(QuantMode::Mxfp4);
        let message = |a: Array| a.check().unwrap_err().to_string();
        let gs = ops::full([4], 1.0f32);
        let opts = GatherQmm::default().global_scale(&gs);
        assert!(
            message(x.gather_qmm(&mx, None, Some(&ids), opts))
                .contains("only supported for 'nvfp4'")
        );
        let wrong = ops::full([3], 1.0f32);
        let opts = GatherQmm::default().global_scale(&wrong);
        assert!(
            message(x.gather_qmm(&nv, None, Some(&ids), opts)).contains("one entry per expert")
        );
        let half = gs.astype(DType::F16);
        let opts = GatherQmm::default().global_scale(&half);
        assert!(message(x.gather_qmm(&nv, None, Some(&ids), opts)).contains("dtype float32"));
        let opts = GatherQmm::default().global_scale(&gs);
        assert!(message(x.gather_qmm(&nv, None, None, opts)).contains("without indices"));
        let _ = x
            .gather_qmm(
                &nv,
                None,
                Some(&ids),
                GatherQmm::default().global_scale(&gs),
            )
            .check()
            .unwrap();
    }

    fn bits(a: &Array) -> Vec<u32> {
        a.astype(DType::F32)
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// Seeded random shapes and layouts (transposed and offset views, batches, empty
    /// arrays): the GPU's codes, scales and dequantized values equal the CPU's bit for bit.
    #[test]
    fn fp_cpu_gpu_bytes() {
        if crate::common::require_gpu().is_none() {
            return;
        }
        let mut rng = Rng::new(0x55f);
        for (mode, gs) in FP_MODES {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                for case in 0..6 {
                    let rows = 1 + rng.below(5) as i32;
                    let k = gs * (1 + rng.below(6) as i32);
                    let w = match case {
                        // A transposed view.
                        0 => random(&mut rng, &[k, rows], dtype).t(),
                        // An offset slice of a wider matrix.
                        1 => random(&mut rng, &[rows + 2, k + gs], dtype).i((1..rows + 1, gs..)),
                        // A batch of matrices.
                        2 => random(&mut rng, &[2, rows, k], dtype),
                        // A matrix with no rows.
                        3 => random(&mut rng, &[0, k], dtype),
                        _ => random(&mut rng, &[rows, k], dtype).multiply(16.0),
                    };
                    let run = || {
                        let q = w.quantize(mode);
                        [
                            &q.weight,
                            &q.scales,
                            &q.dequantize(None),
                            &q.dequantize(DType::F32),
                        ]
                        .map(bytes)
                    };
                    let (cpu, gpu) = (
                        with_stream(Device::cpu(), run),
                        with_stream(Device::gpu(0), run),
                    );
                    for (i, what) in ["codes", "scales", "dequantized", "dequantized f32"]
                        .iter()
                        .enumerate()
                    {
                        assert!(
                            cpu[i] == gpu[i],
                            "{mode} {dtype} case {case}: {what} differ"
                        );
                    }
                }
            }
        }
    }
}

/// Transforms over the floating-point products: vjp and jvp in `x` against a composition
/// through the dequantized weights, errors for the codes, the uint8 scales and the global
/// scale, and vmap against a loop.
/// Derivatives in `x` of the floating-point products and gathers against central finite
/// differences (float32 on the CPU; the loss is linear in `x`, so the bound is rounding):
/// both transposes, every mode, sorted and unsorted gathers, NVFP4 with global scales.
#[test]
fn fp_derivatives_match_finite_differences() {
    let mut rng = Rng::new(0x5a4);
    let check = |name: &str, x: &Array, f: &dyn Fn(&Array) -> Array| {
        let (_, g) = vjp_flat(
            |xs: &[Array]| Ok(vec![f(&xs[0])]),
            std::slice::from_ref(x),
            &[Array::from(1.0f32)],
        )
        .unwrap();
        let grad = g[0].to_vec::<f32>().unwrap();
        let base = x.to_vec::<f32>().unwrap();
        let eps = 1e-2f32;
        for i in (0..base.len()).step_by((base.len() / 7).max(1)) {
            let at = |d: f32| {
                let mut v = base.clone();
                v[i] += d;
                f(&Array::from_slice(&v, x.shape())).item::<f32>().unwrap()
            };
            let fd = (at(eps) - at(-eps)) / (2.0 * eps);
            assert!(
                (fd - grad[i]).abs() <= 2e-3 * (1.0 + fd.abs()),
                "{name}: d/dx[{i}] {} vs finite difference {fd}",
                grad[i]
            );
        }
    };
    with_stream(Device::cpu(), || {
        for (mode, gs) in FP_MODES {
            for t in [true, false] {
                let (rows, cols) = if t { (12, 4 * gs) } else { (4 * gs, 2 * gs) };
                let q = random(&mut rng, &[rows, cols], DType::F32).quantize(mode);
                let (k, n) = if t { (cols, rows) } else { (rows, cols) };
                let x = random(&mut rng, &[3, k], DType::F32);
                let c = random(&mut rng, &[3, n], DType::F32);
                check(&format!("{mode} t={t}"), &x, &|x| {
                    x.quantized_matmul(&q, t).multiply(&c).sum(..)
                });
            }
            // Gathers over 4 experts, unsorted and sorted, NVFP4 with global scales.
            let (e, n, k) = (4, 16, 2 * gs);
            let q = random(&mut rng, &[e, n, k], DType::F32).quantize(mode);
            let scale = Array::from_slice(&[0.5f32, 1.0, 1.5, 2.0], [4]);
            for sorted in [false, true] {
                for global in [false, mode == QuantMode::Nvfp4] {
                    let mut ids: Vec<u32> = vec![2, 0, 3, 1, 2];
                    if sorted {
                        ids.sort_unstable();
                    }
                    let ids = Array::from_slice(&ids, [5]);
                    let x = random(&mut rng, &[5, 1, k], DType::F32);
                    let c = random(&mut rng, &[5, 1, n], DType::F32);
                    check(
                        &format!("{mode} gather sorted={sorted} global={global}"),
                        &x,
                        &|x| {
                            let mut options = GatherQmm::default().sorted_indices(sorted);
                            if global {
                                options = options.global_scale(&scale);
                            }
                            x.gather_qmm(&q, None, Some(&ids), options)
                                .multiply(&c)
                                .sum(..)
                        },
                    );
                }
            }
        }
    });
}

#[test]
fn fp_derivatives_and_vmap() {
    let mut rng = Rng::new(0x5a3);
    let swap = |a: &Array| {
        let n = a.ndim();
        let mut axes: Vec<usize> = (0..n).collect();
        axes.swap(n - 2, n - 1);
        a.transpose(axes)
    };
    for (mode, gs) in FP_MODES {
        for t in [true, false] {
            let (rows, cols) = if t { (12, 4 * gs) } else { (4 * gs, 2 * gs) };
            let w = random(&mut rng, &[rows, cols], DType::F32);
            let q = w.quantize(mode);
            let dense = q.dequantize(DType::F32);
            let k = if t { cols } else { rows };
            let x = random(&mut rng, &[3, k], DType::F32);
            let f = |xs: &[Array]| Ok(vec![xs[0].quantized_matmul(&q, t)]);
            let composed = |xs: &[Array]| {
                Ok(vec![xs[0].matmul(&if t {
                    swap(&dense)
                } else {
                    dense.clone()
                })])
            };
            let n = if t { rows } else { cols };
            let g = random(&mut rng, &[3, n], DType::F32);
            let (_, got) = vjp_flat(f, std::slice::from_ref(&x), std::slice::from_ref(&g)).unwrap();
            let (_, want) =
                vjp_flat(composed, std::slice::from_ref(&x), std::slice::from_ref(&g)).unwrap();
            assert_close(&format!("{mode} t={t} vjp"), &got[0], &want[0], 1e-5);
            let dx = random(&mut rng, &[3, k], DType::F32);
            let (_, got) =
                jvp_flat(f, std::slice::from_ref(&x), std::slice::from_ref(&dx)).unwrap();
            let (_, want) = jvp_flat(
                composed,
                std::slice::from_ref(&x),
                std::slice::from_ref(&dx),
            )
            .unwrap();
            assert_close(&format!("{mode} t={t} jvp"), &got[0], &want[0], 1e-5);
            // vmap over a leading axis of x equals a loop, bit for bit.
            let xs = random(&mut rng, &[2, 3, k], DType::F32);
            let mapped = vmap_flat(f, &[Some(0)], &[0])(std::slice::from_ref(&xs)).unwrap();
            for i in 0..2 {
                let one = f(&[xs.i(i)]).unwrap();
                let a = mapped[0].i(i).to_vec::<f32>().unwrap();
                assert_eq!(a, one[0].to_vec::<f32>().unwrap(), "{mode} t={t} vmap {i}");
            }
        }
        // No gradient for the codes or the uint8 scales, as primals of the product or of
        // dequantize.
        let w = random(&mut rng, &[8, 2 * gs], DType::F32);
        let q = w.quantize(mode);
        let x = random(&mut rng, &[2, 2 * gs], DType::F32);
        let ones = ops::full([2, 8], 1.0f32);
        let product = |xs: &[Array]| {
            Ok(vec![x.quantized_matmul(
                &Quantized::new(xs[0].clone(), xs[1].clone(), None, mode),
                true,
            )])
        };
        let primals = [q.weight.clone(), q.scales.clone()];
        assert!(
            vjp_flat(product, &primals, std::slice::from_ref(&ones)).is_err(),
            "{mode} codes"
        );
        let scales_only = |xs: &[Array]| {
            Ok(vec![x.quantized_matmul(
                &Quantized::new(q.weight.clone(), xs[0].clone(), None, mode),
                true,
            )])
        };
        assert!(
            vjp_flat(
                scales_only,
                std::slice::from_ref(&q.scales),
                std::slice::from_ref(&ones)
            )
            .is_err(),
            "{mode} scales"
        );
        let deq = |xs: &[Array]| {
            Ok(vec![
                Quantized::new(q.weight.clone(), xs[0].clone(), None, mode).dequantize(DType::F32),
            ])
        };
        let back = ops::full([8, 2 * gs], 1.0f32);
        assert!(
            vjp_flat(
                deq,
                std::slice::from_ref(&q.scales),
                std::slice::from_ref(&back)
            )
            .is_err(),
            "{mode} dequantize"
        );
    }
    // The gather's vjp in x, sorted and not, with nvfp4's global scale, against the
    // composition `x[lhs] · (dequantize(w)[rhs] · factor)ᵀ`.
    let (e, n, k) = (4, 16, 64);
    for (mode, _) in FP_MODES {
        let w = random(&mut rng, &[e, n, k], DType::F32);
        let q = w.quantize(mode);
        let factor: Vec<f32> = (0..e).map(|i| 0.5 + i as f32).collect();
        let gscale = Array::from_slice(&factor, [e]);
        for (sorted, ids) in [
            (false, vec![2u32, 0, 3, 0, 1]),
            (true, vec![0u32, 0, 1, 2, 3]),
        ] {
            let rhs = Array::from_slice(&ids, [5]);
            let x = random(&mut rng, &[5, 1, k], DType::F32);
            let g = random(&mut rng, &[5, 1, n], DType::F32);
            for global in [false, mode == QuantMode::Nvfp4] {
                let mut options = GatherQmm::default().sorted_indices(sorted);
                if global {
                    options = options.global_scale(&gscale);
                }
                let f = |xs: &[Array]| Ok(vec![xs[0].gather_qmm(&q, None, Some(&rhs), options)]);
                let mut dense = q.dequantize(DType::F32);
                if global {
                    dense = dense * (gscale.reshape([e, 1, 1]) / (448.0 * 6.0));
                }
                let composed = |xs: &[Array]| Ok(vec![xs[0].matmul(&swap(&dense.take(&rhs, 0)))]);
                let (_, got) =
                    vjp_flat(f, std::slice::from_ref(&x), std::slice::from_ref(&g)).unwrap();
                let (_, want) =
                    vjp_flat(composed, std::slice::from_ref(&x), std::slice::from_ref(&g)).unwrap();
                assert_close(
                    &format!("{mode} gather sorted={sorted} gs={global}"),
                    &got[0],
                    &want[0],
                    1e-5,
                );
            }
            if mode == QuantMode::Nvfp4 {
                let f = |xs: &[Array]| {
                    let options = GatherQmm::default().global_scale(&xs[0]);
                    Ok(vec![x.gather_qmm(&q, None, Some(&rhs), options)])
                };
                assert!(
                    vjp_flat(f, std::slice::from_ref(&gscale), std::slice::from_ref(&g)).is_err(),
                    "global scale"
                );
            }
        }
    }
}

/// Speculative verification over mixture-of-experts layers: a sorted gather of 2 to 8
/// tokens' expert pairs (consecutive tokens sharing experts) takes the few-row kernel and
/// gives every pair exactly the bits of the unsorted per-pair gather, and so of each token
/// decoded alone (the row-independence contract), for every bit width and both half types.
#[test]
fn sorted_few_row_gathers_keep_the_one_row_bits() {
    nuvec_core::with_row_exact(|| {
        if common::require_gpu().is_none() {
            return;
        }
        set_default_stream(Stream::default_for(Device::gpu(0)));
        let mut rng = Rng::new(0x1d5);
        let (experts, n, k, top) = (16usize, 256usize, 512usize, 8usize);
        for bits in [2i32, 3, 4, 8] {
            for dtype in [DType::F16, DType::BF16] {
                let w = random::normal(
                    [experts as i32, n as i32, k as i32],
                    DType::F32,
                    0.0,
                    0.1,
                    &random::key(bits as u64),
                )
                .astype(dtype);
                let q = w.quantize(Quant::affine(64, bits));
                for tokens in [2usize, 5, 8] {
                    // Each token's top-8 of 16 experts: consecutive tokens share some.
                    let mut ids = Vec::new();
                    for _ in 0..tokens {
                        let mut chosen: Vec<u32> = (0..experts as u32).collect();
                        for i in 0..top {
                            let j = i + rng.below(experts - i);
                            chosen.swap(i, j);
                        }
                        ids.extend_from_slice(&chosen[..top]);
                    }
                    let x = random::normal(
                        [1, tokens as i32, 1, 1, k as i32],
                        DType::F32,
                        0.0,
                        1.0,
                        &random::key(tokens as u64),
                    )
                    .astype(dtype);
                    // A layer's layout (`[B, L, 1, 1, K]` rows, `[B, L, k]` experts), B = 1.
                    let idx = Array::from_slice(&ids, [1, tokens as i32, top as i32]);
                    // Unsorted: one row per (token, expert) pair.
                    let unsorted = x.gather_qmm(&q, None, Some(&idx), GatherQmm::default());
                    // Sorted by expert, as a mixture-of-experts layer sorts them.
                    let flat = idx.reshape([-1]);
                    let order = flat.argsort(0);
                    let token_of = Array::from_slice(
                        &(0..(tokens * top) as u32)
                            .map(|p| p / top as u32)
                            .collect::<Vec<_>>(),
                        [(tokens * top) as i32],
                    );
                    let rows = x
                        .reshape([-1, 1, k as i32])
                        .take(&token_of.take(&order, 0), 0);
                    let sorted = rows
                        .gather_qmm(
                            &q,
                            None,
                            Some(&flat.take(&order, 0)),
                            GatherQmm::default().sorted_indices(true),
                        )
                        .take(&order.argsort(0), 0)
                        .reshape([1, tokens as i32, top as i32, 1, n as i32]);
                    let a = unsorted.astype(DType::F32).to_vec::<f32>().unwrap();
                    let b = sorted.astype(DType::F32).to_vec::<f32>().unwrap();
                    let worst = a
                        .iter()
                        .zip(&b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    let differing = a
                        .iter()
                        .zip(&b)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    assert!(
                        differing == 0,
                        "{} b {bits}, {tokens} tokens: sorted few-row gather differs in {differing} of {} values (worst {worst})",
                        dtype.name(),
                        a.len()
                    );
                    // Each token alone.
                    for t in 0..tokens as i32 {
                        let one = x.i(t..t + 1).gather_qmm(
                            &q,
                            None,
                            Some(&idx.i(t..t + 1)),
                            GatherQmm::default(),
                        );
                        let one = one.astype(DType::F32).to_vec::<f32>().unwrap();
                        let row = &b[t as usize * top * n..(t as usize + 1) * top * n];
                        assert!(
                            one.iter().zip(row).all(|(x, y)| x.to_bits() == y.to_bits()),
                            "{} b {bits}, token {t} of {tokens}: differs from the token alone",
                            dtype.name()
                        );
                    }
                }
            }
        }
    });
}
