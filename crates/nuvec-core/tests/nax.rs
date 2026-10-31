//! The Neural Accelerator (NAX) kernels: Metal 4 TensorOps GEMM. They run when
//! `Device::nax()` is active: on M5-class GPUs once measured, or forced with `NUVEC_NAX=on`
//! on any GPU with TensorOps (macOS 26.2), which is how `just nax-check` runs this suite on
//! earlier Macs. Each product is compared with the CPU and with the existing (SIMD) kernels
//! within a tolerance (TensorOps accumulate in their own order; float32 runs at relaxed
//! precision, as MLX's TF32 default).

#![cfg(target_os = "macos")]

mod common;

use common::Rng;
use nuvec_core::*;

/// The NAX status, printed; `true` when the NAX kernels run (else the tests skip).
fn nax_active() -> bool {
    if common::require_gpu().is_none() {
        return false;
    }
    let Some(device) = nuvec_gpu::Device::shared() else {
        return false;
    };
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let status = device.nax();
    eprintln!("NAX on {}: {status:?}", device.info().architecture);
    status.is_active()
}

fn random(rng: &mut Rng, shape: &[usize], dtype: DType) -> Array {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|_| rng.below(2001) as f32 / 1000.0 - 1.0)
        .collect();
    Array::from_slice(&v, shape).astype(dtype)
}

/// The largest error of `got` against `want`, relative to `sqrt(k)` (the accumulation
/// length) and the magnitude.
fn scaled_error(got: &Array, want: &Array, k: usize) -> f64 {
    let g = got.astype(DType::F64).to_vec::<f64>().unwrap();
    let w = want.astype(DType::F64).to_vec::<f64>().unwrap();
    g.iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs() / ((k as f64).sqrt() + b.abs()))
        .fold(0.0, f64::max)
}

fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 4e-3,
        DType::BF16 => 3e-2,
        // TensorOps' relaxed precision (TF32-like inputs).
        _ => 4e-3,
    }
}

#[test]
fn the_nax_status_is_reported() {
    if common::require_gpu().is_none() {
        return;
    }
    let Some(device) = nuvec_gpu::Device::shared() else {
        return;
    };
    let status = device.nax();
    eprintln!("NAX: {status:?}");
    // Without the override an M4 or earlier GPU never runs NAX; an M5 waits for its
    // measurement.
    if std::env::var("NUVEC_NAX").is_err() {
        assert!(!status.is_active(), "{status:?}");
    }
}

#[test]
fn nax_products_equal_the_cpu_and_the_simd_kernels() {
    if !nax_active() {
        return;
    }
    let mut rng = Rng::new(7);
    let shapes = [
        (64, 64, 64),
        (9, 17, 33),
        (100, 37, 70),
        (513, 1000, 333),
        (128, 256, 4096),
        (2048, 512, 1024),
    ];
    for (m, n, k) in shapes {
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
                let a = match ta {
                    true => random(&mut rng, &[k, m], dtype).t(),
                    false => random(&mut rng, &[m, k], dtype),
                };
                let b = match tb {
                    true => random(&mut rng, &[n, k], dtype).t(),
                    false => random(&mut rng, &[k, n], dtype),
                };
                let name = format!("{m}x{n}x{k} {} ta {ta} tb {tb}", dtype.name());
                let got = a.matmul(&b);
                let cpu = with_stream(Device::cpu(), || {
                    a.astype(DType::F32).matmul(&b.astype(DType::F32))
                });
                let e = scaled_error(&got, &cpu, k);
                assert!(e < tolerance(dtype), "{name}: {e}");
                let previous = __nax_route(false);
                let simd = a.matmul(&b);
                simd.eval().unwrap();
                __nax_route(previous);
                let e = scaled_error(&got, &simd, k);
                assert!(e < tolerance(dtype) * 2.0, "{name} vs SIMD: {e}");
            }
        }
    }
}

#[test]
fn nax_products_batch_and_take_the_addmm_epilogue() {
    if !nax_active() {
        return;
    }
    let mut rng = Rng::new(11);
    for dtype in [DType::F16, DType::F32] {
        // Batched, and broadcast over a leading batch axis.
        let a = random(&mut rng, &[3, 2, 70, 96], dtype);
        let b = random(&mut rng, &[2, 96, 50], dtype);
        let got = a.matmul(&b);
        let cpu = with_stream(Device::cpu(), || {
            a.astype(DType::F32).matmul(&b.astype(DType::F32))
        });
        assert!(
            scaled_error(&got, &cpu, 96) < tolerance(dtype),
            "{}",
            dtype.name()
        );
        // addmm: alpha · A·B + beta · C, C broadcast over rows.
        let x = random(&mut rng, &[130, 64], dtype);
        let w = random(&mut rng, &[64, 90], dtype);
        let c = random(&mut rng, &[90], dtype);
        for (alpha, beta) in [(1.0, 1.0), (0.5, 2.0), (1.0, 0.0)] {
            let got = c.addmm(&x, &w, alpha, beta);
            let cpu = with_stream(Device::cpu(), || {
                c.astype(DType::F32).addmm(
                    &x.astype(DType::F32),
                    &w.astype(DType::F32),
                    alpha,
                    beta,
                )
            });
            let e = scaled_error(&got, &cpu, 64);
            assert!(
                e < tolerance(dtype),
                "{} addmm {alpha} {beta}: {e}",
                dtype.name()
            );
        }
    }
}

#[test]
fn nax_quantized_products_equal_the_cpu_and_the_simd_kernels() {
    if !nax_active() {
        return;
    }
    let mut rng = Rng::new(13);
    // (100, 200, 256) splits K (SIMD); the others run on NAX.
    for (m, n, k) in [
        (64, 64, 64),
        (100, 200, 256),
        (512, 640, 1024),
        (1000, 1100, 256),
    ] {
        for bits in [2i32, 3, 4, 8] {
            for gs in [32i32, 64, 128].into_iter().filter(|&g| g as usize <= k) {
                for dtype in [DType::F16, DType::BF16] {
                    let w = random(&mut rng, &[n, k], dtype);
                    let q = w.quantize(Quant::affine(gs, bits));
                    let x = random(&mut rng, &[m, k], dtype);
                    let name = format!("{m}x{n}x{k} {} gs {gs} b {bits}", dtype.name());
                    let got = x.quantized_matmul(&q, true);
                    let deq = q.dequantize(None);
                    let cpu = with_stream(Device::cpu(), || {
                        x.astype(DType::F32).matmul(&deq.astype(DType::F32).t())
                    });
                    let e = scaled_error(&got, &cpu, k);
                    assert!(e < tolerance(dtype) * 2.0, "{name}: {e}");
                    let previous = __nax_route(false);
                    let simd = x.quantized_matmul(&q, true);
                    simd.eval().unwrap();
                    __nax_route(previous);
                    let e = scaled_error(&got, &simd, k);
                    assert!(e < tolerance(dtype) * 2.0, "{name} vs SIMD: {e}");
                }
            }
        }
    }
}

/// The sorted mixture-of-experts gather on NAX: rows sorted by expert, with uneven runs
/// (empty experts, runs shorter and longer than a 64-row tile), equals the SIMD kernel and
/// a float32 CPU product of each row with its expert's dequantized weights.
#[test]
fn nax_sorted_gathers_equal_the_cpu_and_the_simd_kernels() {
    if !nax_active() {
        return;
    }
    let mut rng = Rng::new(23);
    let (experts, n, k) = (8, 192, 256);
    // Rows per expert: 0, 1, 63, 64, 65, 130, 0, 7.
    let runs = [0usize, 1, 63, 64, 65, 130, 0, 7];
    let ids: Vec<u32> = runs
        .iter()
        .enumerate()
        .flat_map(|(e, &r)| std::iter::repeat_n(e as u32, r))
        .collect();
    let rows = ids.len();
    let ids = Array::from(ids);
    for bits in [2i32, 4, 8] {
        for dtype in [DType::F16, DType::BF16] {
            let w = random(&mut rng, &[experts, n, k], dtype);
            let q = w.quantize(Quant::affine(64, bits));
            let x = random(&mut rng, &[rows, 1, k], dtype);
            let gather = |x: &Array| {
                x.gather_qmm(
                    &q,
                    None,
                    Some(&ids),
                    GatherQmm::default().sorted_indices(true),
                )
            };
            let got = gather(&x);
            let name = format!("{} b {bits}", dtype.name());
            let deq = q.dequantize(None).astype(DType::F32);
            let cpu = with_stream(Device::cpu(), || {
                let picked = deq.take(&ids, 0); // [rows, n, k]
                x.astype(DType::F32).matmul(&picked.transpose([0, 2, 1]))
            });
            let e = scaled_error(&got, &cpu, k);
            assert!(e < tolerance(dtype) * 2.0, "{name}: {e}");
            let previous = __nax_route(false);
            let simd = gather(&x);
            simd.eval().unwrap();
            __nax_route(previous);
            let e = scaled_error(&got, &simd, k);
            assert!(e < tolerance(dtype) * 2.0, "{name} vs SIMD: {e}");
        }
    }
}

/// With profiling on (`NUVEC_PROFILE=1`, as `just nax-check` runs it), the products above
/// are the NAX kernels: the GEMM and the quantized product dispatch `nuvec_nax_*`.
#[test]
fn the_nax_kernels_are_the_ones_that_run() {
    if !nax_active() || !nuvec_gpu::profile::enabled() {
        return;
    }
    let mut rng = Rng::new(17);
    nuvec_gpu::profile::reset();
    // Many output tiles over a short K (no split).
    let a = random(&mut rng, &[1024, 256], DType::F16);
    let b = random(&mut rng, &[256, 1024], DType::F16);
    a.matmul(&b).eval().unwrap();
    let q = random(&mut rng, &[1024, 256], DType::F16).quantize(Quant::affine(64, 4));
    a.quantized_matmul(&q, true).eval().unwrap();
    // A thin product (24 rows), and few output tiles over a long K (split K).
    random(&mut rng, &[24, 256], DType::F16)
        .matmul(&b)
        .eval()
        .unwrap();
    let long = random(&mut rng, &[96, 4096], DType::F16);
    long.matmul(&random(&mut rng, &[4096, 96], DType::F16))
        .eval()
        .unwrap();
    // A sorted gather over 4 experts of 64 rows each.
    let experts = random(&mut rng, &[4, 128, 256], DType::F16).quantize(Quant::affine(64, 4));
    let ids = Array::from((0..256u32).map(|i| i / 64).collect::<Vec<_>>());
    random(&mut rng, &[256, 1, 256], DType::F16)
        .gather_qmm(
            &experts,
            None,
            Some(&ids),
            GatherQmm::default().sorted_indices(true),
        )
        .eval()
        .unwrap();
    let names: Vec<String> = nuvec_gpu::profile::report()
        .into_iter()
        .map(|r| r.0)
        .collect();
    for kernel in [
        "nuvec_nax_gemm_nn_float16_bm64_bn64",
        "nuvec_nax_gemm_nn_float16_bm32_bn64",
        "nuvec_nax_gemm_splitk_nn_float16_bm64_bn64",
        "nuvec_gemm_splitk_accum_float16",
        "nuvec_nax_gather_qmm_rhs_float16_gs64_b4",
    ] {
        assert!(
            names.iter().any(|n| n.starts_with(kernel)),
            "{kernel}: {names:?}"
        );
    }
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("nuvec_nax_qmm_t_float16_gs64_b4")),
        "{names:?}"
    );
}

/// A differentiated function of its primals.
type Product = Box<dyn Fn(&[Array]) -> Array>;

/// The gradients of `f` at `primals` (cotangent `cot`), evaluated with the NAX kernels
/// allowed (`nax`) or not.
fn gradients(
    f: impl Fn(&[Array]) -> Array,
    primals: &[Array],
    cot: &Array,
    nax: bool,
) -> Vec<Array> {
    let previous = __nax_route(nax);
    let (_, grads) = vjp_flat(|xs| Ok(vec![f(xs)]), primals, std::slice::from_ref(cot)).unwrap();
    eval(&grads).unwrap();
    __nax_route(previous);
    grads
}

/// Derivatives through the NAX kernels (`matmul`, `addmm` and `quantized_matmul` with
/// respect to every differentiable operand: their backward products are NAX products too)
/// equal the SIMD kernels' within the products' tolerance. Finite differences check the
/// SIMD derivatives in `autodiff.rs` and `quantized.rs`.
#[test]
fn nax_derivatives_equal_the_simd_kernels() {
    if !nax_active() {
        return;
    }
    let mut rng = Rng::new(17);
    let (m, n, k) = (128, 96, 256);
    for dtype in [DType::F16, DType::BF16, DType::F32] {
        let a = random(&mut rng, &[m, k], dtype);
        let b = random(&mut rng, &[k, n], dtype);
        let c = random(&mut rng, &[m, n], dtype);
        let cot = random(&mut rng, &[m, n], dtype);
        let cases: [(&str, Product, Vec<Array>); 2] = [
            (
                "matmul",
                Box::new(|xs| xs[0].matmul(&xs[1])),
                vec![a.clone(), b.clone()],
            ),
            (
                "addmm",
                Box::new(|xs| ops::addmm(&xs[2], &xs[0], &xs[1], 0.5, 2.0)),
                vec![a.clone(), b.clone(), c.clone()],
            ),
        ];
        for (name, f, primals) in cases {
            let nax = gradients(&f, &primals, &cot, true);
            let simd = gradients(&f, &primals, &cot, false);
            for (i, (g, s)) in nax.iter().zip(&simd).enumerate() {
                let e = scaled_error(g, s, k.max(m));
                assert!(
                    e < tolerance(dtype) * 2.0,
                    "{name} {} d{i}: {e}",
                    dtype.name()
                );
            }
        }
    }
    // The quantized product: with respect to x, the scales and the biases.
    for dtype in [DType::F16, DType::BF16] {
        let w = random(&mut rng, &[n, k], dtype);
        let q = w.quantize(Quant::affine(64, 4));
        let x = random(&mut rng, &[m, k], dtype);
        let cot = random(&mut rng, &[m, n], dtype);
        let (wq, quant) = (q.weight.clone(), q.quant);
        let f = |xs: &[Array]| {
            let q = Quantized::new(wq.clone(), xs[1].clone(), Some(xs[2].clone()), quant);
            xs[0].quantized_matmul(&q, true)
        };
        let primals = [x, q.scales.clone(), q.biases.clone().unwrap()];
        let nax = gradients(f, &primals, &cot, true);
        let simd = gradients(f, &primals, &cot, false);
        for (i, (g, s)) in nax.iter().zip(&simd).enumerate() {
            let e = scaled_error(g, s, k.max(m));
            assert!(e < tolerance(dtype) * 2.0, "qmm {} d{i}: {e}", dtype.name());
        }
    }
}
