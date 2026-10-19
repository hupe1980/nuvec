//! Affine quantization: round trips, `quantized_matmul` against the dequantized product,
//! derivatives against an independent composition, `vmap` against a loop, and (on macOS)
//! the GPU kernels against the CPU.

mod common;

use common::Rng;
use nuvec_core::*;

const BITS: [u32; 7] = [1, 2, 3, 4, 5, 6, 8];
const GROUPS: [usize; 3] = [32, 64, 128];

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
fn reference(x: &Array, w: &Array, s: &Array, b: &Array, t: bool, gs: usize, bits: u32) -> Array {
    let dtype = promote_types(x.dtype(), s.dtype());
    let (s, b) = (s.astype(dtype).unwrap(), b.astype(dtype).unwrap());
    let dense = dequantize(w, &s, &b, gs, bits).unwrap();
    let dense = if t {
        let n = dense.ndim();
        let mut axes: Vec<usize> = (0..n).collect();
        axes.swap(n - 2, n - 1);
        transpose(&dense, &axes).unwrap()
    } else {
        dense
    };
    matmul(&x.astype(dtype).unwrap(), &dense).unwrap()
}

#[test]
fn round_trip_error_is_about_half_a_step() {
    let mut rng = Rng::new(0x51);
    for bits in BITS {
        for gs in GROUPS {
            let w = random(&mut rng, &[3, 2 * gs], DType::F32);
            let (wq, s, b) = quantize(&w, gs, bits).unwrap();
            assert_eq!(wq.dtype(), DType::U32);
            assert_eq!(wq.shape(), &[3, 2 * gs * bits as usize / 32]);
            assert_eq!(s.shape(), &[3, 2]);
            let restored = values(&dequantize(&wq, &s, &b, gs, bits).unwrap());
            let (orig, scales) = (values(&w), values(&s));
            let (mut total, mut steps) = (0.0, 0.0);
            for (i, (r, o)) in restored.iter().zip(&orig).enumerate() {
                let step = scales[i / gs].abs();
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
    let w = Array::from_slice(&v, &[3, 32]).unwrap();
    let (wq, s, b) = quantize(&w, 32, 4).unwrap();
    let back = values(&dequantize(&wq, &s, &b, 32, 4).unwrap());
    assert!(
        back[..32].iter().all(|&x| (x - 0.75).abs() < 1e-6),
        "{:?}",
        &back[..4]
    );
    // Zero is representable exactly in a group that spans it.
    let codes = values(
        &dequantize(
            &wq,
            &ones(&[3, 1], DType::F32).unwrap(),
            &zeros(&[3, 1], DType::F32).unwrap(),
            32,
            4,
        )
        .unwrap(),
    );
    let (scale, bias) = (values(&s)[2], values(&b)[2]);
    assert!(codes[64..].iter().any(|&q| (q * scale + bias).abs() < 1e-7));
}

#[test]
fn invalid_arguments_are_rejected() {
    let w = Array::full(&[2, 64], 1.0f32).unwrap();
    assert!(quantize(&w, 48, 4).is_err(), "group size");
    assert!(quantize(&w, 64, 7).is_err(), "bits");
    assert!(
        quantize(&Array::full(&[64], 1.0f32).unwrap(), 64, 4).is_err(),
        "1-D"
    );
    assert!(
        quantize(&Array::full(&[2, 96], 1.0f32).unwrap(), 64, 4).is_err(),
        "divisibility"
    );
    assert!(
        quantize(&w.astype(DType::I32).unwrap(), 64, 4).is_err(),
        "dtype"
    );
    let (wq, s, b) = quantize(&w, 64, 4).unwrap();
    assert!(dequantize(&wq, &s, &b, 32, 4).is_err(), "group mismatch");
    assert!(dequantize(&s, &s, &b, 64, 4).is_err(), "packed dtype");
    let x = Array::full(&[3, 63], 1.0f32).unwrap();
    assert!(
        quantized_matmul(&x, &wq, &s, &b, true, 64, 4).is_err(),
        "inner size"
    );
    let x = Array::full(&[3, 64], 1).unwrap();
    assert!(quantized_matmul(&x.astype(DType::F64).unwrap(), &wq, &s, &b, true, 64, 4).is_err());
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
        let (wq, s, b) = quantize(&w, gs, bits).unwrap();
        for shape in [vec![128], vec![5, 128], vec![2, 3, 128]] {
            let x = random(&mut rng, &shape, DType::F32);
            let got = quantized_matmul(&x, &wq, &s, &b, true, gs, bits).unwrap();
            let want = reference(&x, &wq, &s, &b, true, gs, bits);
            assert_close(&format!("t {bits}/{gs} {shape:?}"), &got, &want, 1e-6);
        }
        // Without the transpose, `w` is [K, N]: K = 48 rows.
        let x = random(&mut rng, &[4, 48], DType::F32);
        let got = quantized_matmul(&x, &wq, &s, &b, false, gs, bits).unwrap();
        assert_close(
            "n",
            &got,
            &reference(&x, &wq, &s, &b, false, gs, bits),
            1e-6,
        );
    }
    // Batched weights broadcast against batched activations; half precision promotes.
    let w = random(&mut rng, &[2, 1, 16, 64], DType::F16);
    let (wq, s, b) = quantize(&w, 32, 4).unwrap();
    let x = random(&mut rng, &[3, 5, 64], DType::F32);
    let got = quantized_matmul(&x, &wq, &s, &b, true, 32, 4).unwrap();
    assert_eq!(got.shape(), &[2, 3, 5, 16]);
    assert_eq!(got.dtype(), DType::F32);
    let wq = broadcast_to(&wq, &[2, 3, 16, 8]).unwrap();
    let s = broadcast_to(&s, &[2, 3, 16, 2]).unwrap();
    let b = broadcast_to(&b, &[2, 3, 16, 2]).unwrap();
    assert_close(
        "batched",
        &got,
        &reference(&x, &wq, &s, &b, true, 32, 4),
        1e-6,
    );
}

/// `x · (codes ⊙ s + b)ᵀ` from ordinary differentiable operations, with fixed codes.
fn composed(xs: &[Array], codes: &Array, gs: usize) -> Result<Vec<Array>> {
    let (x, s, b) = (&xs[0], &xs[1], &xs[2]);
    let (rows, cols) = (codes.shape()[0], codes.shape()[1]);
    let grouped = codes.reshape(&[rows as isize, (cols / gs) as isize, gs as isize])?;
    let dense = add(
        multiply(&grouped, expand_dims(s, -1)?)?,
        expand_dims(b, -1)?,
    )?;
    Ok(vec![matmul(
        x,
        &dense.reshape(&[rows as isize, cols as isize])?.t()?,
    )?])
}

#[test]
fn derivatives_match_an_independent_composition() {
    let mut rng = Rng::new(0x53);
    let (gs, bits) = (32, 4);
    let w = random(&mut rng, &[6, 64], DType::F32);
    let (wq, s, b) = quantize(&w, gs, bits).unwrap();
    let codes = dequantize(
        &wq,
        &ones(&[6, 2], DType::F32).unwrap(),
        &zeros(&[6, 2], DType::F32).unwrap(),
        gs,
        bits,
    )
    .unwrap();
    let x = random(&mut rng, &[2, 3, 64], DType::F32);
    let g = random(&mut rng, &[2, 3, 6], DType::F32);
    let wq2 = wq.clone();
    let f = move |xs: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![quantized_matmul(
            &xs[0], &wq2, &xs[1], &xs[2], true, gs, bits,
        )?])
    };
    let primals = [x.clone(), s.clone(), b.clone()];
    let (_, got) = vjp(f.clone(), &primals, std::slice::from_ref(&g)).unwrap();
    let c = codes.clone();
    let (_, want) = vjp(move |xs: &[Array]| composed(xs, &c, gs), &primals, &[g]).unwrap();
    for (name, (a, e)) in ["dx", "ds", "db"].iter().zip(got.iter().zip(&want)) {
        assert_close(name, a, e, 1e-5);
    }
    let tangents = [
        random(&mut rng, &[2, 3, 64], DType::F32),
        random(&mut rng, &[6, 2], DType::F32),
        random(&mut rng, &[6, 2], DType::F32),
    ];
    let (_, got) = jvp(f, &primals, &tangents).unwrap();
    let (_, want) = jvp(
        move |xs: &[Array]| composed(xs, &codes, gs),
        &primals,
        &tangents,
    )
    .unwrap();
    assert_close("jvp", &got[0], &want[0], 1e-5);
    // No gradient flows into the packed weights.
    let err = vjp(
        move |xs: &[Array]| Ok(vec![quantized_matmul(&x, &xs[0], &s, &b, true, gs, bits)?]),
        &[wq],
        &[Array::full(&[2, 3, 6], 1.0f32).unwrap()],
    );
    assert!(err.is_err());
}

#[test]
fn dequantize_derivatives_are_linear_in_scales_and_biases() {
    let mut rng = Rng::new(0x54);
    let w = random(&mut rng, &[4, 64], DType::F32);
    let (wq, s, b) = quantize(&w, 64, 3).unwrap();
    let f = |xs: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![dequantize(&xs[0], &xs[1], &xs[2], 64, 3)?])
    };
    let g = random(&mut rng, &[4, 64], DType::F32);
    let wq2 = wq.clone();
    let (_, grads) = vjp(
        move |xs: &[Array]| f(&[wq2.clone(), xs[0].clone(), xs[1].clone()]),
        &[s.clone(), b.clone()],
        std::slice::from_ref(&g),
    )
    .unwrap();
    let codes = dequantize(
        &wq,
        &ones(&[4, 1], DType::F32).unwrap(),
        &zeros(&[4, 1], DType::F32).unwrap(),
        64,
        3,
    )
    .unwrap();
    assert_close(
        "ds",
        &grads[0],
        &sum(&multiply(&g, &codes).unwrap(), &[-1], true).unwrap(),
        1e-6,
    );
    assert_close("db", &grads[1], &sum(&g, &[-1], true).unwrap(), 1e-6);
}

#[test]
fn vmap_matches_a_loop() {
    let mut rng = Rng::new(0x55);
    let w = random(&mut rng, &[3, 8, 64], DType::F32);
    let x = random(&mut rng, &[3, 2, 64], DType::F32);
    // Batched weights and activations.
    let f = |xs: &[Array]| -> Result<Vec<Array>> {
        let (wq, s, b) = quantize(&xs[0], 32, 4)?;
        Ok(vec![quantized_matmul(&xs[1], &wq, &s, &b, true, 32, 4)?])
    };
    let got = vmap(f, &[Some(0), Some(0)], &[0])(&[w.clone(), x.clone()]).unwrap();
    for i in 0..3 {
        let pick = |a: &Array| {
            let mut args = vec![SliceArg::new(Some(i as isize), Some(i as isize + 1), 1)];
            args.extend(std::iter::repeat_n(SliceArg::full(), a.ndim() - 1));
            slice(a, &args).unwrap()
        };
        let (wi, xi) = (pick(&w), pick(&x));
        let want = f(&[squeeze(&wi, &[0]).unwrap(), squeeze(&xi, &[0]).unwrap()]).unwrap();
        let gi = squeeze(&pick(&got[0]), &[0]).unwrap();
        assert_close("vmap", &gi, &want[0], 1e-6);
    }
    // Shared weights, batched activations on the last axis.
    let (wq, s, b) = quantize(&w.reshape(&[24, 64]).unwrap(), 64, 8).unwrap();
    let xt = random(&mut rng, &[64, 5], DType::F32);
    let g = move |xs: &[Array]| -> Result<Vec<Array>> {
        Ok(vec![quantized_matmul(&xs[0], &wq, &s, &b, true, 64, 8)?])
    };
    let got = vmap(g.clone(), &[Some(1)], &[0])(std::slice::from_ref(&xt)).unwrap();
    let want = g(&[xt.t().unwrap()]).unwrap();
    assert_close("vmap shared", &got[0], &want[0], 1e-6);
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

    #[test]
    fn quantize_and_dequantize_match_the_cpu_bit_for_bit() {
        if !gpu_available() {
            return;
        }
        let mut rng = Rng::new(0x56);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for bits in BITS {
                for gs in GROUPS {
                    let w = random(&mut rng, &[3, 2 * gs], dtype);
                    // A transposed view exercises the copy to row-contiguous memory.
                    let w = if gs == 64 {
                        w.t().unwrap().t().unwrap()
                    } else {
                        w
                    };
                    let run = || {
                        let (wq, s, b) = quantize(&w, gs, bits).unwrap();
                        let d = dequantize(&wq, &s, &b, gs, bits).unwrap();
                        [wq, s, b, d].map(|a| bits_of(&a))
                    };
                    let (cpu, gpu) = (on(Device::cpu(), run), on(Device::gpu(0), run));
                    for (i, name) in ["codes", "scales", "biases", "dequantized"]
                        .iter()
                        .enumerate()
                    {
                        assert_eq!(cpu[i], gpu[i], "{dtype} {bits} bits, group {gs}: {name}");
                    }
                }
            }
        }
    }

    #[test]
    fn quantized_matmul_matches_the_cpu_across_kernels() {
        if !gpu_available() {
            return;
        }
        let mut rng = Rng::new(0x57);
        // (x shape, w rows, w cols, transpose, bits, group): qmv_fast (aligned K, with and
        // without partial rows), qmv (unaligned K), qvm, qmm_t (aligned and not), split-K,
        // qmm_n, batched and broadcast weights.
        type Case = (&'static [usize], &'static [usize], usize, bool, u32, usize);
        let cases: &[Case] = &[
            (&[1, 512], &[64], 512, true, 4, 64),
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
                    let (wq, s, b) = quantize(&w, gs, bits).unwrap();
                    let y = quantized_matmul(&x, &wq, &s, &b, t, gs, bits).unwrap();
                    y.astype(DType::F32).unwrap()
                };
                let (cpu, gpu) = (on(Device::cpu(), run), on(Device::gpu(0), run));
                let k = if t { cols } else { *wb.last().unwrap() };
                let eps = match dtype {
                    DType::F32 => 1e-6,
                    DType::F16 => 1e-3,
                    _ => 8e-3,
                };
                let name = format!("{dtype} x{xs:?} w{wshape:?} t={t} {bits}/{gs}");
                assert_close(&name, &gpu, &cpu, eps * (k as f64).sqrt() * 4.0);
            }
        }
        // Transposed activations are copied; empty products stay on the CPU.
        let w = random(&mut rng, &[16, 64], DType::F32);
        let x = random(&mut rng, &[64, 5], DType::F32);
        let run = || {
            let (wq, s, b) = quantize(&w, 64, 4).unwrap();
            quantized_matmul(&x.t().unwrap(), &wq, &s, &b, true, 64, 4).unwrap()
        };
        assert_close(
            "strided x",
            &on(Device::gpu(0), run),
            &on(Device::cpu(), run),
            1e-5,
        );
        let empty = on(Device::gpu(0), || {
            let (wq, s, b) = quantize(&w, 64, 4).unwrap();
            quantized_matmul(
                &zeros(&[0, 64], DType::F32).unwrap(),
                &wq,
                &s,
                &b,
                true,
                64,
                4,
            )
            .unwrap()
        });
        assert_eq!(empty.shape(), &[0, 16]);
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
    let (wq, s, b) = quantize(&w, 32, 4).unwrap();
    let x = random(&mut rng, &[4, 2, k], DType::F32);
    let lhs = Array::from_slice(&[3u32, 0, 2], &[3]).unwrap();
    let rhs = Array::from_slice(&[1u32, 2, 0], &[3]).unwrap();
    let got = gather_qmm(&x, &wq, &s, &b, Some(&lhs), Some(&rhs), true, 32, 4, false).unwrap();
    assert_eq!(got.shape(), &[3, 2, n]);
    let pick = |a: &Array, i: u32| take(a, &Array::scalar(i), Some(0)).unwrap();
    for (j, (&l, &r)) in [3u32, 0, 2].iter().zip(&[1u32, 2, 0]).enumerate() {
        let want = quantized_matmul(
            &pick(&x, l),
            &pick(&wq, r),
            &pick(&s, r),
            &pick(&b, r),
            true,
            32,
            4,
        )
        .unwrap();
        assert_close(&format!("batch {j}"), &pick(&got, j as u32), &want, 1e-6);
    }
    // Default indices: every batch in order (x's batch broadcast against the weights').
    let x1 = random(&mut rng, &[1, 1, k], DType::F32);
    let all = gather_qmm(&x1, &wq, &s, &b, None, None, true, 32, 4, false).unwrap();
    assert_eq!(all.shape(), &[3, 1, n]);
    // Gradient with respect to x: a row used by two batches gets both contributions.
    let rows = Array::from_slice(&[1u32, 1], &[2]).unwrap();
    let experts = Array::from_slice(&[0u32, 1], &[2]).unwrap();
    let x2 = random(&mut rng, &[2, 1, k], DType::F32);
    let f = |xs: &[Array]| {
        let y = gather_qmm(
            &xs[0],
            &wq,
            &s,
            &b,
            Some(&rows),
            Some(&experts),
            true,
            32,
            4,
            false,
        )?;
        Ok(vec![y.reshape(&[-1])?])
    };
    let ones = Array::full(&[2 * n], 1.0f32).unwrap();
    let (_, grads) = vjp(f, std::slice::from_ref(&x2), std::slice::from_ref(&ones)).unwrap();
    let column_sums = |i: u32| {
        let w = dequantize(&pick(&wq, i), &pick(&s, i), &pick(&b, i), 32, 4).unwrap();
        sum(&w, &[0], false).unwrap()
    };
    let row = |i: u32| pick(&grads[0], i).reshape(&[-1]).unwrap();
    assert_close("dx row 0", &row(0), &zeros(&[k], DType::F32).unwrap(), 1e-6);
    let want = add(column_sums(0), column_sums(1)).unwrap();
    assert_close("dx row 1", &row(1), &want, 1e-5);
    // Batching over a leading axis of x matches a loop.
    let xs = random(&mut rng, &[2, 4, 2, k], DType::F32);
    let f = |a: &[Array]| {
        Ok(vec![gather_qmm(
            &a[0],
            &wq,
            &s,
            &b,
            Some(&lhs),
            Some(&rhs),
            true,
            32,
            4,
            false,
        )?])
    };
    let batched = vmap(f, &[Some(0)], &[0])(std::slice::from_ref(&xs)).unwrap();
    for i in 0..2u32 {
        let one = f(&[pick(&xs, i)]).unwrap();
        assert_close(&format!("vmap {i}"), &pick(&batched[0], i), &one[0], 1e-6);
    }
    // Errors: float indices, mismatched index shapes.
    let bad = Array::from_slice(&[0.0f32], &[1]).unwrap();
    assert!(gather_qmm(&x, &wq, &s, &b, Some(&bad), None, true, 32, 4, false).is_err());
    let two = Array::from_slice(&[0u32, 1], &[2]).unwrap();
    assert!(gather_qmm(&x, &wq, &s, &b, Some(&lhs), Some(&two), true, 32, 4, false).is_err());
}

#[cfg(target_os = "macos")]
mod gather_gpu {
    use super::*;

    fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
        let previous = default_stream();
        set_default_stream(Stream::default_for(device));
        let r = f();
        set_default_stream(previous);
        r
    }

    /// Every gather kernel (matrix-vector, tiled, grouped over sorted rows) against the CPU.
    #[test]
    fn gather_qmm_kernels_match_the_cpu() {
        if cfg!(miri) || on(Device::gpu(0), || (Array::scalar(1.0f32) * 2.0).eval()).is_err() {
            return;
        }
        let mut rng = Rng::new(0x72);
        // (experts, N, K, rows M per batch, batches, sorted)
        let cases: &[(usize, usize, usize, usize, usize, bool)] = &[
            (8, 64, 512, 1, 6, false),  // qmv_fast
            (8, 36, 192, 1, 5, false),  // qmv (N not a multiple of 8, K not of 512)
            (4, 64, 256, 40, 3, false), // qmm_t
            (4, 40, 128, 33, 2, false), // qmm_t, unaligned
            (8, 64, 256, 1, 100, true), // grouped, sorted rows
            (16, 48, 192, 1, 70, true), // grouped, unaligned N
        ];
        for &(e, n, k, m, batches, sorted) in cases {
            for (dtype, gs, bits) in [
                (DType::F32, 32, 4),
                (DType::BF16, 64, 4),
                (DType::F16, 32, 8),
            ] {
                let w = random(&mut rng, &[e, n, k], dtype);
                let (wq, s, b) = quantize(&w, gs, bits).unwrap();
                let x = random(&mut rng, &[batches, m, k], dtype);
                let mut ids: Vec<u32> = (0..batches).map(|i| ((i * 7 + 3) % e) as u32).collect();
                if sorted {
                    ids.sort_unstable();
                }
                let rhs = Array::from_slice(&ids, &[batches]).unwrap();
                let run = || {
                    gather_qmm(&x, &wq, &s, &b, None, Some(&rhs), true, gs, bits, sorted).unwrap()
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
        let (wq, s, b) = quantize(&w, 32, 4).unwrap();
        let x = random(&mut rng, &[3, 1, 128], DType::F32);
        let lhs = Array::from_slice(&[2u32, 0, 1, 1], &[2, 2]).unwrap();
        let rhs = Array::from_slice(&[3u32, 1], &[1, 2]).unwrap();
        let run =
            || gather_qmm(&x, &wq, &s, &b, Some(&lhs), Some(&rhs), true, 32, 4, false).unwrap();
        let (gpu, cpu) = (on(Device::gpu(0), run), on(Device::cpu(), run));
        assert_close("broadcast indices", &gpu, &cpu, 1e-5);
    }
}
