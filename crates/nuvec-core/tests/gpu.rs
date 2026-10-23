//! Evaluation on the GPU stream: MLX fixtures, GPU-vs-CPU differential tests, fallback,
//! and long graphs. Every test skips when no Metal device is available.

#![cfg(target_os = "macos")]

mod common;
#[path = "common/fixture.rs"]
mod fixture;

use common::{Rng, random_layout, size};
use nuvec_core::ops;
use nuvec_core::*;

/// Runs `f` with `device` as this thread's default, restoring the previous default.
fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
    with_stream(device, f)
}

#[test]
fn mlx_fixtures_on_gpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let (_, cases) = fixture::load();
    let failures: Vec<String> = on(Device::gpu(0), || {
        cases.iter().filter_map(fixture::check).collect()
    });
    assert!(
        failures.is_empty(),
        "{} of {} cases failed on the GPU:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// Element-wise result on both devices, compared bit-for-bit (NaN equals NaN).
fn same_on_both<T: Element + common::Same>(name: &str, f: impl Fn() -> Array) {
    let cpu = on(Device::cpu(), &f).to_vec::<T>().unwrap();
    let gpu = on(Device::gpu(0), &f).to_vec::<T>().unwrap();
    assert_eq!(cpu.len(), gpu.len(), "{name}");
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(c.same(*g), "{name}: element {i}: cpu {c:?}, gpu {g:?}");
    }
}

#[test]
fn elementwise_matches_cpu_on_random_layouts() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(0x6B);
    for case in 0..100 {
        let shape = rng.shape();
        let other = rng.broadcastable(&shape);
        let fv: Vec<f32> = (0..common::size(&shape))
            .map(|_| rng.f64() as f32)
            .collect();
        let gv: Vec<f32> = (0..common::size(&other))
            .map(|_| rng.f64() as f32)
            .collect();
        let iv: Vec<i32> = (0..common::size(&shape))
            .map(|_| rng.next() as i32)
            .collect();
        let hv: Vec<f16> = fv.iter().map(|&x| f16::from_f32(x)).collect();
        let (x, _) = random_layout(&mut rng, &shape, fv);
        let (y, _) = random_layout(&mut rng, &other, gv);
        let (i, _) = random_layout(&mut rng, &shape, iv);
        let (h, _) = random_layout(&mut rng, &shape, hv);
        let ctx = |op: &str| format!("case {case} {op} {shape:?}+{other:?}");
        same_on_both::<f32>(&ctx("add"), || x.add(&y));
        same_on_both::<f32>(&ctx("div"), || x.divide(&y));
        same_on_both::<f32>(&ctx("max"), || x.maximum(&y));
        same_on_both::<bool>(&ctx("le"), || x.less_equal(&y));
        same_on_both::<i32>(&ctx("mul i32"), || i.multiply(&i));
        same_on_both::<i32>(&ctx("neg i32"), || i.negative());
        same_on_both::<f16>(&ctx("sub f16"), || h.subtract(h.t().t()));
        same_on_both::<f32>(&ctx("where"), || x.greater(&y).select(&x, &y));
        same_on_both::<i8>(&ctx("cast"), || x.astype(DType::I8));
        same_on_both::<f32>(&ctx("copy"), || x.t().reshape([-1]));
    }
}

/// A unary operation under test.
type UnaryFn = fn(&Array) -> Array;

#[test]
fn transcendentals_are_close_to_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let x = ops::arange(-6.0, 6.0, 0.01);
    let ops: [(&str, UnaryFn); 7] = [
        ("exp", ops::exp),
        ("log", ops::log),
        ("sqrt", ops::sqrt),
        ("sin", ops::sin),
        ("cos", ops::cos),
        ("tanh", ops::tanh),
        ("sigmoid", ops::sigmoid),
    ];
    for (name, op) in ops {
        let cpu = on(Device::cpu(), || op(&x)).to_vec::<f32>().unwrap();
        let gpu = on(Device::gpu(0), || op(&x)).to_vec::<f32>().unwrap();
        for (c, g) in cpu.iter().zip(&gpu) {
            let ok = c.to_bits() == g.to_bits()
                || (c.is_nan() && g.is_nan())
                || (c - g).abs() <= 4.0 * f32::EPSILON * c.abs().max(f32::MIN_POSITIVE);
            assert!(ok, "{name}: cpu {c}, gpu {g}");
        }
    }
}

#[test]
fn reductions_and_matmul_match_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(0x7C);
    for _ in 0..50 {
        let shape: Vec<usize> = (0..1 + rng.below(3)).map(|_| 1 + rng.below(40)).collect();
        let iv: Vec<i32> = (0..common::size(&shape))
            .map(|_| rng.range(-50, 50) as i32)
            .collect();
        let fv: Vec<f32> = (0..common::size(&shape))
            .map(|_| rng.f64() as f32)
            .collect();
        let (ia, _) = random_layout(&mut rng, &shape, iv);
        let (fa, _) = random_layout(&mut rng, &shape, fv);
        let axis = rng.below(shape.len()) as i32;
        same_on_both::<i32>("sum i32", || ia.sum(axis));
        same_on_both::<i32>("max i32", || ia.max_keepdims(..));
        same_on_both::<f32>("min f32", || fa.min(axis));
        same_on_both::<u32>("argmax", || fa.argmax(axis));
        same_on_both::<bool>("any", || fa.greater(Array::from(0.5f32)).any(axis));
        // Float sums: the GPU accumulates in f32, the CPU in f64.
        let c = on(Device::cpu(), || fa.sum(axis)).to_vec::<f32>().unwrap();
        let g = on(Device::gpu(0), || fa.sum(axis)).to_vec::<f32>().unwrap();
        let n = shape[axis as usize] as f32;
        assert!(
            c.iter()
                .zip(&g)
                .all(|(c, g)| (c - g).abs() <= (n + 1.0) * f32::EPSILON * 10.0 * c.abs().max(1.0))
        );
    }
    for (m, k, n) in [(1, 1, 1), (17, 33, 9), (64, 64, 64), (5, 0, 3)] {
        let a = ops::arange(0.0, (m * k) as f64, 1.0).reshape([m, k]).sin();
        let b = ops::arange(0.0, (k * n) as f64, 1.0).reshape([k, n]).cos();
        let c = on(Device::cpu(), || a.matmul(&b.t().t()))
            .to_vec::<f32>()
            .unwrap();
        let g = on(Device::gpu(0), || a.matmul(&b.t().t()))
            .to_vec::<f32>()
            .unwrap();
        assert!(
            c.iter()
                .zip(&g)
                .all(|(c, g)| (c - g).abs() <= 1e-4 * (k as f32 + 1.0)),
            "{m}x{k}x{n}"
        );
    }
    // Batched with broadcast batch dimensions and a transposed operand.
    let a = ops::arange(0.0, 2.0 * 3.0 * 4.0, 1.0)
        .reshape([2, 1, 3, 4])
        .sin();
    let b = ops::arange(0.0, 5.0 * 2.0 * 4.0, 1.0)
        .reshape([5, 2, 4])
        .cos();
    let bt = b.transpose([0, 2, 1]);
    let c = on(Device::cpu(), || a.matmul(&bt)).to_vec::<f32>().unwrap();
    let g = on(Device::gpu(0), || a.matmul(&bt))
        .to_vec::<f32>()
        .unwrap();
    assert!(c.iter().zip(&g).all(|(c, g)| (c - g).abs() <= 1e-5));
}

#[test]
fn unsupported_work_falls_back_to_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    // float64 has no Metal type, complex sums have no kernel yet: both run on the CPU, in
    // between GPU kernels, over the same memory.
    let x = ops::arange(0.0, 100.0, 1.0);
    let y = on(Device::gpu(0), || {
        let doubled = &x + &x; // GPU
        let wide = doubled.astype(DType::F64); // CPU (f64)
        let sq = &wide * &wide; // CPU (f64)
        let back = sq.astype(DType::F32); // CPU (from f64)
        let z = &back + &x; // GPU, reading CPU output
        let c = z.astype(DType::Complex64); // GPU
        c.sum(..) // CPU (complex sum)
    });
    let expected: f32 = (0..100).map(|i| (2 * i * 2 * i + i) as f32).sum();
    assert_eq!(
        y.item::<Complex64>().unwrap(),
        Complex64::new(expected, 0.0)
    );
}

#[test]
fn long_graphs_span_command_buffers() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let one = ops::full([1024], 1.0f32);
    let x = on(Device::gpu(0), || {
        let mut x = ops::zeros([1024], DType::F32);
        for _ in 0..1000 {
            x = &x + &one;
        }
        x
    });
    assert!(x.to_vec::<f32>().unwrap().iter().all(|&v| v == 1000.0));
}

#[test]
fn scalars_empty_and_errors() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    on(Device::gpu(0), || {
        let s = Array::from(3i64) * Array::from(-4i64);
        assert_eq!(s.item::<i64>().unwrap(), -12);
        let e = ops::zeros([0, 3], DType::F32);
        assert_eq!((&e + &e).shape(), [0, 3]);
        assert_eq!(e.sum(0).to_vec::<f32>().unwrap(), vec![0.0; 3]);
        let bad = &ops::zeros([2], DType::F32) + &ops::zeros([3], DType::F32);
        assert!(matches!(
            bad.eval(),
            Err(Error::Op {
                kind: OpErrorKind::ShapeMismatch { .. },
                ..
            })
        ));
    });
}

#[test]
fn long_dependent_chains_stay_ordered_across_command_buffers() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    on(Device::gpu(0), || {
        for n in [1usize, 1 << 16, 1 << 20] {
            let one = ops::full([n], 1.0f32);
            let mut x = ops::full([n], 0.0f32);
            for _ in 0..3000 {
                x = &x + &one;
            }
            let v = x.to_vec::<f32>().unwrap();
            assert!(v.iter().all(|&e| e == 3000.0), "n = {n}: {}", v[0]);
        }
    });
}

#[test]
fn async_eval_results_feed_later_work_on_both_devices() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    on(Device::gpu(0), || {
        let x = ops::arange(0.0, 4096.0, 1.0);
        let mut y = x.clone();
        for _ in 0..600 {
            y = &y * 1.0 + 1.0;
        }
        async_eval(std::slice::from_ref(&y)).unwrap();
        // A GPU consumer and a CPU-fallback consumer (float64) of in-flight results.
        let gpu = y.sum(..);
        let cpu = y.astype(DType::F64) * 2.0;
        async_eval(&[gpu.clone(), cpu.clone()]).unwrap();
        let expected: f64 = (0..4096).map(|i| f64::from(i) + 600.0).sum();
        assert!((f64::from(gpu.item::<f32>().unwrap()) - expected).abs() < expected * 1e-6);
        let c = cpu.to_vec::<f64>().unwrap();
        assert_eq!((c[0], c[4095]), (1200.0, 2.0 * 4695.0));
        assert!(y.is_evaluated());
    });
}

#[test]
fn concurrent_evaluations_from_several_threads() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let shared = on(Device::gpu(0), || ops::arange(0.0, 1024.0, 1.0) * 2.0);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let shared = shared.clone();
            std::thread::spawn(move || {
                on(Device::gpu(0), || {
                    let mut y = shared.clone();
                    for _ in 0..500 {
                        y = &y + t as f32;
                    }
                    if t % 2 == 0 {
                        async_eval(std::slice::from_ref(&y)).unwrap();
                    }
                    y.to_vec::<f32>().unwrap()
                })
            })
        })
        .collect();
    for (t, h) in handles.into_iter().enumerate() {
        let v = h.join().unwrap();
        assert_eq!(v[3], 6.0 + 500.0 * t as f32, "thread {t}");
    }
}

#[test]
fn async_eval_applies_back_pressure() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let device = nuvec_gpu::Device::shared().unwrap();
    on(Device::gpu(0), || {
        let big = ops::full([1 << 20], 1.0f32);
        let mut outs = Vec::new();
        for _ in 0..60 {
            let y = (&big * 2.0 + 1.0).sum(..);
            async_eval(std::slice::from_ref(&y)).unwrap();
            // At most 10 per device, plus one per concurrently submitting test thread.
            assert!(device.in_flight() <= 16, "{} in flight", device.in_flight());
            outs.push(y);
        }
        for y in outs {
            assert_eq!(y.item::<f32>().unwrap(), 3.0 * (1 << 20) as f32);
        }
    });
}

#[test]
fn async_eval_reports_known_errors() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    on(Device::gpu(0), || {
        let a = array![1.0, 2.0];
        let b = array![1.0, 2.0, 3.0];
        let bad = &a + &b;
        assert!(matches!(
            async_eval(&[bad]),
            Err(Error::Op {
                kind: OpErrorKind::ShapeMismatch { .. },
                ..
            })
        ));
    });
}

#[test]
fn long_reductions_split_into_two_passes_match_the_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(17);
    let n = 300_000;
    let values: Vec<f32> = (0..n).map(|_| rng.f64() as f32 * 0.01).collect();
    let both = |f: &dyn Fn() -> Array| {
        let cpu = on(Device::cpu(), f)
            .astype(DType::F64)
            .to_vec::<f64>()
            .unwrap();
        let gpu = on(Device::gpu(0), f)
            .astype(DType::F64)
            .to_vec::<f64>()
            .unwrap();
        (cpu, gpu)
    };
    let close = |name: &str, (c, g): (Vec<f64>, Vec<f64>), tol: f64| {
        assert_eq!(c.len(), g.len(), "{name}");
        for (c, g) in c.iter().zip(&g) {
            assert!(
                (c.is_nan() && g.is_nan()) || (c - g).abs() <= tol * c.abs().max(1.0),
                "{name}: cpu {c} gpu {g}"
            );
        }
    };
    let flat = || Array::from_slice(&values, [n]);
    let rows = || Array::from_slice(&values, [3, n / 3]);
    close("sum-1d", both(&|| flat().sum(..)), 1e-4);
    close("sum-rows", both(&|| rows().sum(1)), 1e-4);
    close("sum-transposed", both(&|| rows().t().sum_keepdims(0)), 1e-4);
    close("max", both(&|| flat().max(..)), 0.0);
    close("min-rows", both(&|| rows().min(1)), 0.0);
    close("sum-f16", both(&|| flat().astype(DType::F16).sum(..)), 5e-3);
    close(
        "sum-i32",
        both(&|| (flat() * 100.0).astype(DType::I32).sum(..)),
        0.0,
    );
    close("all", both(&|| flat().less(1.0).all(..)), 0.0);
    close("any", both(&|| flat().greater(0.0999).any(..)), 0.0);
    let with_nan = || {
        let mut v = values.clone();
        v[n - 7] = f32::NAN;
        Array::from_slice(&v, [n])
    };
    close("max-nan", both(&|| with_nan().max(..)), 0.0);
}

#[test]
fn later_evaluations_consume_in_flight_results() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    on(Device::gpu(0), || {
        let one = ops::full([1 << 16], 1.0f32);
        let mut a = ops::full([1 << 16], 0.0f32);
        for _ in 0..400 {
            a = &a + &one;
        }
        async_eval(std::slice::from_ref(&a)).unwrap();
        // Encoded while `a` may still be running: a GPU consumer, views of it, and a CPU
        // (float64 fallback) consumer.
        let gpu = &a * 2.0;
        let view = a.reshape([256, 256]).t();
        let cpu = a.astype(DType::F64) + 1.0;
        let step = (&gpu + &view.reshape([-1])).sum(..);
        async_eval(&[gpu.clone(), cpu.clone(), step.clone()]).unwrap();
        assert!(gpu.to_vec::<f32>().unwrap().iter().all(|&v| v == 800.0));
        assert!(view.to_vec::<f32>().unwrap().iter().all(|&v| v == 400.0));
        assert!(cpu.to_vec::<f64>().unwrap().iter().all(|&v| v == 401.0));
        assert_eq!(step.item::<f32>().unwrap(), 1200.0 * (1 << 16) as f32);
    });
}

#[test]
fn steel_gemm_matches_the_cpu_over_shapes_layouts_batches_and_dtypes() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(23);
    let mut random = |shape: &[usize]| {
        let n = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 * 0.1).collect();
        Array::from_slice(&v, shape)
    };
    // (A shape, B shape, how to present A, how to present B)
    let cases: Vec<(Vec<usize>, Vec<usize>, &str, &str)> = vec![
        (vec![64, 64], vec![64, 64], "", ""),
        (vec![129, 67], vec![67, 200], "", ""),
        (vec![1, 300], vec![300, 77], "", ""),
        (vec![77, 300], vec![300, 1], "", ""),
        (vec![300, 77], vec![65, 300], "t", "t"),
        (vec![33, 50], vec![65, 50], "", "t"),
        (vec![50, 33], vec![50, 65], "t", ""),
        (vec![40, 60], vec![60, 70], "strided", "strided"),
        (vec![3, 70, 40], vec![3, 40, 90], "", ""),
        (vec![3, 70, 40], vec![40, 90], "", ""),
        (vec![2, 3, 20, 30], vec![3, 2, 30, 10], "batch-t", ""),
    ];
    let present = |x: &Array, how: &str| -> Array {
        match how {
            "t" => x.t(),
            // Every "strided" operand is 2-D.
            "strided" => x.i(((..).step(2), (..).step(2))),
            "batch-t" => x.transpose([1, 0, 2, 3]),
            _ => x.clone(),
        }
    };
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for (sa, sb, ha, hb) in &cases {
            let (a, b) = (random(sa), random(sb));
            let (pa, pb) = (present(&a, ha), present(&b, hb));
            let k = pa.shape()[pa.ndim() - 1];
            let f = || pa.astype(dtype).matmul(&pb.astype(dtype));
            let cpu = on(Device::cpu(), f)
                .astype(DType::F32)
                .to_vec::<f32>()
                .unwrap();
            let gpu = on(Device::gpu(0), f)
                .astype(DType::F32)
                .to_vec::<f32>()
                .unwrap();
            let eps = match dtype {
                DType::F32 => 1e-6,
                DType::F16 => 2e-3,
                _ => 1.6e-2,
            };
            // Standard GEMM error bound: eps · K · max|a| · max|b| (inputs are within ±1).
            let tol = eps * k as f32;
            for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
                assert!(
                    (c - g).abs() <= tol,
                    "{dtype} {sa:?}x{sb:?} ({ha},{hb}) element {i}: cpu {c} gpu {g}"
                );
            }
        }
    }
    // addmm: broadcast bias, full c, alpha/beta; vector operands.
    let (a, b) = (random(&[37, 50]), random(&[50, 41]));
    let (bias, full) = (random(&[41]), random(&[37, 41]));
    for (c, alpha, beta) in [
        (&bias, 1.0, 1.0),
        (&full, 0.5, -2.0),
        (&bias, 0.0, 3.0),
        (&full, 2.0, 0.0),
    ] {
        let f = || c.addmm(&a, &b, alpha, beta);
        let cpu = on(Device::cpu(), f).to_vec::<f32>().unwrap();
        let gpu = on(Device::gpu(0), f).to_vec::<f32>().unwrap();
        for (c, g) in cpu.iter().zip(&gpu) {
            assert!(
                (c - g).abs() <= 1e-6 * 50.0 * 4.0,
                "addmm {alpha} {beta}: {c} vs {g}"
            );
        }
    }
    // Split-K (long K, few output tiles) with the AddMM epilogue.
    let (a, b, full) = (random(&[20, 640]), random(&[640, 30]), random(&[20, 30]));
    for (c, alpha, beta) in [(&full, 1.0, 1.0), (&full, 0.5, -2.0)] {
        let f = || c.addmm(&a, &b, alpha, beta);
        let cpu = on(Device::cpu(), f).to_vec::<f32>().unwrap();
        let gpu = on(Device::gpu(0), f).to_vec::<f32>().unwrap();
        for (c, g) in cpu.iter().zip(&gpu) {
            assert!(
                (c - g).abs() <= 1e-6 * 640.0 * 4.0,
                "split-K addmm {alpha} {beta}: {c} vs {g}"
            );
        }
    }
    let (b, bias) = (random(&[50, 41]), random(&[41]));
    let v = random(&[50]);
    let f = || bias.addmm(&v, &b, 1.0, 1.0);
    assert_eq!(on(Device::gpu(0), f).shape(), [41]);
}

/// The multi-row matrix-vector kernels (2 to 16 rows, `gemv_rows`): against the CPU, over
/// every block-selection branch, odd tails, `K = 1`, transposed and offset operands,
/// batches and a bias; empty outputs; and gradients through them.
#[test]
fn gemv_rows_match_the_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(31);
    let mut random = |shape: &[usize]| {
        let n = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 * 0.1).collect();
        Array::from_slice(&v, shape)
    };
    let check = |name: &str, f: &dyn Fn() -> Array, k: usize, eps: f32| {
        let cpu = on(Device::cpu(), f)
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap();
        let gpu = on(Device::gpu(0), f)
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(cpu.len(), gpu.len(), "{name}");
        for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            assert!(
                (c - g).abs() <= eps * k as f32,
                "{name} element {i}: cpu {c} gpu {g}"
            );
        }
    };
    for (m, k, n) in [
        (2, 300, 77),
        (5, 8192, 2048),
        (16, 40, 5000),
        (9, 1000, 3),
        (3, 1, 9),
        (16, 4097, 513),
        (7, 64, 4096),
    ] {
        let x = random(&[m + 2, k]).i(1..1 + m as i32); // an offset view
        let w = random(&[k, n]);
        let wt = random(&[n, k]);
        let xt = random(&[k, m]).t(); // transposed rows (copied to rows)
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let eps = match dtype {
                DType::F32 => 1e-6,
                DType::F16 => 2e-3,
                _ => 1.6e-2,
            };
            let cast = |a: &Array| a.astype(dtype);
            let name = format!("{dtype} {m}x{k}x{n}");
            check(&name, &|| cast(&x).matmul(&cast(&w)), k, eps);
            check(
                &format!("{name} wT"),
                &|| cast(&x).matmul(&cast(&wt).t()),
                k,
                eps,
            );
            check(
                &format!("{name} xT"),
                &|| cast(&xt).matmul(&cast(&wt).t()),
                k,
                eps,
            );
        }
    }
    // Batched rows with a broadcast matrix, and a bias.
    let (xb, w, bias) = (random(&[3, 4, 128]), random(&[128, 96]), random(&[96]));
    check("batched", &|| xb.matmul(&w), 128, 1e-6);
    check("addmm", &|| bias.addmm(&xb, &w, 0.5, 2.0), 128, 1e-6);
    // Empty outputs.
    let none = random(&[4, 0]);
    assert_eq!(
        on(Device::gpu(0), || random(&[4, 8]).matmul(&random(&[8, 0]))).shape(),
        [4, 0]
    );
    assert_eq!(
        on(Device::gpu(0), || none.matmul(&random(&[0, 5])))
            .to_vec::<f32>()
            .unwrap(),
        vec![0.0; 20]
    );
    // Gradients through the multi-row path equal the CPU's.
    let (a, b, g) = (random(&[6, 33]), random(&[33, 20]), random(&[6, 20]));
    let grads = || {
        vjp_flat(
            |xs: &[Array]| Ok(vec![xs[0].matmul(&xs[1])]),
            &[a.clone(), b.clone()],
            std::slice::from_ref(&g),
        )
        .unwrap()
        .1
    };
    let (gpu, cpu) = (on(Device::gpu(0), grads), on(Device::cpu(), grads));
    for (x, y) in gpu.iter().zip(&cpu) {
        let (x, y) = (x.to_vec::<f32>().unwrap(), y.to_vec::<f32>().unwrap());
        for (p, q) in x.iter().zip(&y) {
            assert!((p - q).abs() <= 1e-5, "gradient {p} vs {q}");
        }
    }
}

#[test]
fn gemv_matches_the_cpu_for_vector_shapes() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(29);
    let mut random = |shape: &[usize]| {
        let n = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 * 0.1).collect();
        Array::from_slice(&v, shape)
    };
    let check = |name: &str, f: &dyn Fn() -> Array, k: usize, eps: f32| {
        let cpu = on(Device::cpu(), f)
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap();
        let gpu = on(Device::gpu(0), f)
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(cpu.len(), gpu.len(), "{name}");
        for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            assert!(
                (c - g).abs() <= eps * k as f32,
                "{name} element {i}: cpu {c} gpu {g}"
            );
        }
    };
    // (M, K, N): every block-selection branch of gemv and gemv_t.
    for (m, k, n) in [
        (1, 300, 77),
        (1, 8192, 2048),
        (1, 40, 5000),
        (1, 1000, 3),
        (77, 300, 1),
        (5000, 64, 1),
        (3, 4096, 1),
        (1, 1, 9),
    ] {
        let (x, w) = (random(&[m, k]), random(&[k, n]));
        let wt = random(&[n, k]);
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let eps = match dtype {
                DType::F32 => 1e-6,
                DType::F16 => 2e-3,
                _ => 1.6e-2,
            };
            let cast = |a: &Array| a.astype(dtype);
            check(
                &format!("{dtype} {m}x{k}x{n}"),
                &|| cast(&x).matmul(&cast(&w)),
                k,
                eps,
            );
            check(
                &format!("{dtype} {m}x{k}x{n} wT"),
                &|| cast(&x).matmul(&cast(&wt).t()),
                k,
                eps,
            );
        }
    }
    // Vector times transposed matrix and matrix times vector views; a strided vector.
    let (v, mat) = (random(&[2, 300]), random(&[300, 64]));
    let strided = v.i((.., (..).step(2)));
    let half = random(&[150, 64]);
    check(
        "strided vector",
        &|| strided.reshape([-1]).reshape([1, 300]).matmul(&mat),
        300,
        1e-6,
    );
    check(
        "strided k-vector",
        &|| strided.i((0..1, ..)).matmul(&half),
        150,
        1e-6,
    );
    // Batched vectors, one and two batch axes.
    let (bv, bm) = (random(&[3, 1, 200]), random(&[3, 200, 50]));
    check("batched", &|| bv.matmul(&bm), 200, 1e-6);
    let (nv, nm) = (random(&[2, 3, 1, 64]), random(&[2, 3, 64, 33]));
    check(
        "nc batch",
        &|| nv.matmul(&nm.transpose([0, 1, 2, 3])),
        64,
        1e-6,
    );
    // addmm through gemv: bias along N (M = 1) and along M (N = 1).
    let (x, w, bias_n) = (random(&[1, 128]), random(&[128, 40]), random(&[40]));
    check("addmm M=1", &|| bias_n.addmm(&x, &w, 0.5, 2.0), 128, 1e-6);
    let (a, y, bias_m) = (random(&[40, 128]), random(&[128, 1]), random(&[40, 1]));
    check("addmm N=1", &|| bias_m.addmm(&a, &y, 1.0, 1.0), 128, 1e-6);
}

#[test]
fn mlx_reduction_kernels_match_the_cpu_on_every_path() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(0x7E);
    // (shape, axes, layout): all-reduce (one and two passes), row reductions (small rows,
    // simple contiguous rows, looped with other reduced axes), column reductions (small,
    // long columns of short rows, two passes, looped), and general layouts.
    // No axes: every axis.
    let cases: &[(&[usize], &[i32], &str)] = &[
        (&[3000], &[], "contiguous"),
        (&[700, 300], &[], "contiguous"),
        (&[64, 7], &[1], "contiguous"),
        (&[96, 600], &[1], "contiguous"),
        (&[6, 9, 600], &[0, 2], "contiguous"),
        (&[5, 3, 4], &[1, 2], "contiguous"),
        (&[4, 5], &[0], "contiguous"),
        (&[3000, 6], &[0], "contiguous"),
        (&[600, 64], &[0], "contiguous"),
        (&[300, 4096], &[0], "contiguous"),
        (&[7, 300, 40], &[0, 1], "contiguous"),
        (&[40, 30], &[1], "transposed"),
        (&[40, 30], &[0], "transposed"),
        (&[3, 40, 50], &[0, 2], "transposed"),
        (&[8, 128], &[0], "broadcast"),
        (&[9, 80], &[1], "sliced"),
        // Several reduced axes that do not merge: 2- and 5-dimensional kernel variants.
        (&[3, 2, 4, 2, 40], &[0, 2, 4], "contiguous"),
        (&[3, 2, 4, 2, 100], &[0, 2, 4], "contiguous"),
        (&[3, 2, 4, 2, 5, 2, 40], &[0, 2, 4, 6], "contiguous"),
        (&[3, 2, 4, 2, 40], &[0, 2], "contiguous"),
        (&[3, 2, 4, 2, 5, 2, 40], &[0, 2, 4], "contiguous"),
    ];
    for &(shape, axes, layout) in cases {
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I32,
            DType::U8,
            DType::I64,
            DType::Bool,
        ] {
            let n: usize = shape.iter().product();
            let raw: Vec<f32> = (0..n)
                .map(|_| (rng.f64() * 4.0).round() as f32 / 4.0)
                .collect();
            let base = Array::from_slice(&raw, shape).astype(dtype);
            let x = match layout {
                "transposed" => {
                    let rev: Vec<usize> = (0..shape.len()).rev().collect();
                    let t =
                        Array::from_slice(&raw, shape.iter().rev().copied().collect::<Vec<_>>())
                            .astype(dtype);
                    t.transpose(rev)
                }
                "broadcast" => base.i((0..1, ..)).broadcast_to(shape),
                "sliced" => base.i((.., (..).step(2))),
                _ => base,
            };
            let axes: Vec<i32> = if axes.is_empty() {
                (0..x.ndim() as i32).collect()
            } else {
                axes.to_vec()
            };
            type R = fn(&Array, Vec<i32>) -> Array;
            let ops: Vec<(&str, R)> = if dtype == DType::Bool {
                vec![
                    ("all", |x, a| x.all(a)),
                    ("any", |x, a| x.any(a)),
                    ("max", |x, a| x.max(a)),
                    ("min", |x, a| x.min(a)),
                ]
            } else {
                vec![
                    ("sum", |x, a| x.sum(a)),
                    ("max", |x, a| x.max(a)),
                    ("min", |x, a| x.min(a)),
                    ("prod", |x, a| x.prod(a)),
                ]
            };
            for (name, op) in ops {
                if name == "prod" && (n > 64 || dtype == DType::F16 || dtype == DType::BF16) {
                    continue; // products of many values overflow or underflow
                }
                let run = || {
                    op(&x, axes.clone())
                        .astype(DType::F64)
                        .to_vec::<f64>()
                        .unwrap()
                };
                let (cpu, gpu) = (on(Device::cpu(), run), on(Device::gpu(0), run));
                let tol = match dtype {
                    DType::F16 => 2e-2,
                    DType::BF16 => 1e-1,
                    DType::F32 => 1e-4,
                    _ => 0.0,
                };
                for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
                    assert!(
                        (c - g).abs() <= tol * c.abs().max(1.0),
                        "{name} {dtype} {shape:?} axes {axes:?} {layout}: element {i}: cpu {c} gpu {g}"
                    );
                }
            }
        }
    }
    // NaN propagates through max and min.
    let x = Array::from_slice(&[1.0f32, f32::NAN, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let m = on(Device::gpu(0), || x.max(1).to_vec::<f32>().unwrap());
    assert!(m[0].is_nan() && m[1] == 6.0, "{m:?}");
}

/// The array as float64 values, evaluated on the device it was recorded on.
fn f64s(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

#[test]
fn indexing_kernels_match_the_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(0x1D3);
    let index_types = [
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
    ];
    let dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I32,
        DType::U8,
        DType::Bool,
        DType::Complex64,
    ];
    for case in 0..80 {
        let ndim = 1 + rng.below(4);
        let shape: Vec<usize> = (0..ndim).map(|_| 1 + rng.below(5)).collect();
        let axis = rng.below(ndim);
        let dtype = dtypes[case % dtypes.len()];
        let itype = index_types[case % index_types.len()];
        let n = shape[axis] as isize;
        let values: Vec<f32> = (0..size(&shape)).map(|_| rng.f64() as f32).collect();
        let (a, _) = random_layout(&mut rng, &shape, values);
        let a = on(Device::cpu(), || a.astype(dtype));
        let mut ishape = shape.clone();
        ishape[axis] = 1 + rng.below(6);
        let signed = itype.category() == Category::Signed;
        let lo = if signed { -n } else { 0 };
        let raw: Vec<i64> = (0..size(&ishape))
            .map(|_| rng.range(lo, n - 1) as i64)
            .collect();
        let (ind, _) = random_layout(&mut rng, &ishape, raw);
        let ind = on(Device::cpu(), || ind.astype(itype));
        let what = format!("case {case}: {dtype} {shape:?} axis {axis}, {itype} {ishape:?}");
        // Gather.
        let f = || a.take_along_axis(&ind, axis as i32);
        let (cpu, gpu) = (on(Device::cpu(), f), on(Device::gpu(0), f));
        if dtype == DType::Complex64 {
            let pair = |x: &Array| x.to_vec::<Complex64>().unwrap();
            assert_eq!(pair(&cpu), pair(&gpu), "take_along_axis {what}");
        } else {
            assert_eq!(f64s(&cpu), f64s(&gpu), "take_along_axis {what}");
        }
        // Concatenate with a transposed copy along the same axis.
        let b = a.t().t();
        let f = || ops::concatenate(&[a.clone(), b.clone(), a.clone()], axis as i32);
        let (cpu, gpu) = (on(Device::cpu(), f), on(Device::gpu(0), f));
        if dtype != DType::Complex64 {
            assert_eq!(f64s(&cpu), f64s(&gpu), "concatenate {what}");
        }
        // Scatter-add with repeated indices (accumulation order differs: a tolerance for
        // floats), and a put with unique indices (exact).
        if matches!(dtype, DType::F32 | DType::F16 | DType::BF16 | DType::I32) {
            let upd = on(Device::cpu(), || a.astype(dtype));
            let base = upd.take_along_axis(&ind, axis as i32);
            let f = || upd.scatter_add_along_axis(&ind, &base, axis as i32);
            let (cpu, gpu) = (on(Device::cpu(), f), on(Device::gpu(0), f));
            let eps = match dtype {
                DType::F16 => 4e-3,
                DType::BF16 => 3e-2,
                DType::F32 => 1e-6,
                _ => 0.0,
            };
            for (c, g) in f64s(&cpu).iter().zip(f64s(&gpu)) {
                assert!(
                    (c - g).abs() <= eps * 8.0 * c.abs().max(1.0),
                    "scatter_add {what}: cpu {c} gpu {g}"
                );
            }
        }
        let unique: Vec<i64> = (0..n as i64).rev().collect();
        let mut ushape = shape.clone();
        ushape[axis] = n as usize;
        let mut along = vec![1usize; ndim];
        along[axis] = n as usize;
        let uind = Array::from_slice(&unique, &along[..]).broadcast_to(&ushape[..]);
        let f = || a.put_along_axis(&uind, &b, axis as i32);
        let (cpu, gpu) = (on(Device::cpu(), f), on(Device::gpu(0), f));
        if dtype != DType::Complex64 {
            assert_eq!(f64s(&cpu), f64s(&gpu), "put_along_axis {what}");
        }
    }
}

#[test]
fn embedding_lookup_and_gradient_on_the_gpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    // A vocabulary of 1000 rows of 64 values, 256 tokens with repeats.
    let table = on(Device::cpu(), || {
        random::uniform(-1.0, 1.0, [1000, 64], DType::F32, Some(&random::key(3)))
    });
    let tokens: Vec<u32> = (0..256).map(|i| (i * 37 % 300) as u32).collect();
    let tokens = Array::from_slice(&tokens, [8, 32]);
    let run = || {
        let rows = table.take(&tokens, 0);
        let g = grad(|t: &Array| Ok(t.take(&tokens, 0).multiply(2.0).sum(..)))(&table)?;
        Ok::<_, Error>((rows, g))
    };
    let (cpu, cpu_grad) = on(Device::cpu(), run).unwrap();
    let (gpu, gpu_grad) = on(Device::gpu(0), run).unwrap();
    assert_eq!(gpu.shape(), [8, 32, 64]);
    assert_eq!(f64s(&cpu), f64s(&gpu));
    assert_eq!(
        f64s(&cpu_grad),
        f64s(&gpu_grad),
        "counts of 2.0 add exactly"
    );
}

/// A cumulative op: `(a, axis, options)`.
type ScanFn = fn(&Array, i32, ScanOptions) -> Array;
#[test]
fn sort_and_scan_kernels_match_the_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(0x5027);
    // (shape, axis): one block, several blocks (merge passes), a non-innermost axis.
    let cases: [(&[usize], usize); 6] = [
        (&[3, 100], 1),
        (&[2, 3000], 1),
        (&[70_000], 0),
        (&[5000, 3], 0),
        (&[4, 600, 5], 1),
        (&[7], 0),
    ];
    let dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I32,
        DType::U8,
        DType::I64,
    ];
    for (case, &(shape, axis)) in cases.iter().enumerate() {
        for &dtype in &dtypes {
            // Few distinct values: many ties, so stability shows.
            let values: Vec<f32> = (0..size(shape))
                .map(|_| (rng.below(50) as f32 - 20.0) / 4.0)
                .collect();
            let (a, _) = random_layout(&mut rng, shape, values);
            let a = on(Device::cpu(), || a.astype(dtype));
            let what = format!("case {case} {dtype} {shape:?} axis {axis}");
            let ax = axis as i32;
            for (name, f) in [
                ("sort", Box::new(|| a.sort(ax)) as Box<dyn Fn() -> Array>),
                ("argsort", Box::new(|| a.argsort(ax))),
            ] {
                let (cpu, gpu) = (on(Device::cpu(), &f), on(Device::gpu(0), &f));
                assert_eq!(f64s(&cpu), f64s(&gpu), "{name} {what}");
            }
            let exact = !matches!(dtype, DType::F32 | DType::F16 | DType::BF16);
            let ops: [(&str, ScanFn); 4] = [
                ("cumsum", |a, ax, o| a.cumsum_with(ax, o)),
                ("cumprod", |a, ax, o| a.cumprod_with(ax, o)),
                ("cummax", |a, ax, o| a.cummax_with(ax, o)),
                ("cummin", |a, ax, o| a.cummin_with(ax, o)),
            ];
            // Float products of near-one factors on short axes, so neither order of
            // multiplication overflows (where the parallel scan's inf · 0 would differ).
            let near_one = on(Device::cpu(), || {
                (a.astype(DType::F32) * 0.01 + 1.0).astype(dtype)
            });
            for (name, op) in ops {
                let float_prod = name == "cumprod" && !exact;
                // Half-precision running sums over long axes stall (sequentially) or
                // overflow (in parallel): only short axes compare meaningfully.
                let half_sum = name == "cumsum" && matches!(dtype, DType::F16 | DType::BF16);
                if (float_prod || half_sum) && shape[axis] > 600 {
                    continue;
                }
                let input = if float_prod { &near_one } else { &a };
                for (reverse, inclusive) in [(false, true), (true, false)] {
                    let options = ScanOptions {
                        reverse,
                        exclusive: !inclusive,
                    };
                    let f = || op(input, ax, options);
                    let (cpu, gpu) = (on(Device::cpu(), f), on(Device::gpu(0), f));
                    let (c, g) = (f64s(&cpu), f64s(&gpu));
                    if exact || matches!(name, "cummax" | "cummin") {
                        assert_eq!(c, g, "{name} r{reverse} i{inclusive} {what}");
                        continue;
                    }
                    // Products of many factors overflow alike; sums round in another order.
                    let eps = match dtype {
                        DType::F16 => 1e-3,
                        DType::BF16 => 8e-3,
                        _ => 1e-6,
                    } * shape[axis] as f64;
                    for (x, y) in c.iter().zip(&g) {
                        let ok = (x - y).abs() <= eps * x.abs().max(1.0)
                            || (x.is_nan() && y.is_nan())
                            || (x.is_infinite() && x == y);
                        assert!(ok, "{name} r{reverse} i{inclusive} {what}: cpu {x} gpu {y}");
                    }
                }
            }
        }
    }
}

#[test]
fn random_bits_match_the_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    // Every width, odd and even word counts, partial last words, batched keys.
    let keys = random::split(&random::key(7), 3);
    for width in [1, 2, 4] {
        for shape in [vec![1], vec![3], vec![5, 7], vec![151_936], vec![2, 1001]] {
            let name = format!("width {width} shape {shape:?}");
            let bits = |k: &Array| random::bits(&shape[..], width, Some(k)).astype(DType::U32);
            same_on_both::<u32>(&name, || bits(&random::key(42)));
            for i in 0..3u32 {
                let key = keys.take(&Array::from(i), 0);
                same_on_both::<u32>(&format!("{name} key {i}"), || bits(&key));
            }
        }
    }
    same_on_both::<u32>("split", || random::split(&random::key(3), 5));
}

/// Large intermediates (16 MiB and up) return to the allocator as soon as their consumers
/// are encoded, before their command buffer is committed; the next large output reuses the
/// memory, ordered after its readers. The results equal the CPU's.
#[test]
fn large_intermediates_released_early_are_reused_safely() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let n = 6 << 20; // 24 MiB of f32
    let values: Vec<f32> = (0..n).map(|i| (i % 1009) as f32 * 0.01).collect();
    for round in 0..3 {
        same_on_both::<f32>(&format!("round {round}"), || {
            let x = Array::from_slice(&values, [n]);
            // A chain of large intermediates, each read by the next only; few enough
            // dispatches to share one command buffer.
            let mut y = &x * 2.0f32;
            for k in 0..5 {
                y = &(&y + (k as f32)) * &x;
                y = y.minimum(100.0f32); // exact on both devices, unlike `sin`
            }
            &y + &x
        });
    }
}

/// `gather_mm` with one row per batch (the gathered matrix-vector kernels) against the CPU:
/// unsorted and sorted indices, transposed weights, batched rows with lhs indices.
#[test]
fn gather_mv_matches_the_cpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let mut rng = Rng::new(37);
    let mut random = |shape: &[usize]| {
        let n = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.f64() as f32 * 0.1).collect();
        Array::from_slice(&v, shape)
    };
    let (a, b, bt) = (
        random(&[9, 1, 40]),
        random(&[6, 40, 2100]),
        random(&[6, 2100, 40]),
    );
    let ids = Array::from(vec![5u32, 0, 3, 3, 1, 2, 4, 0, 5]);
    let sorted = Array::from(vec![0u32, 0, 1, 2, 2, 3, 4, 5, 5]);
    type Case<'a> = (&'a str, Box<dyn Fn() -> Array + 'a>);
    let cases: [Case<'_>; 4] = [
        (
            "unsorted",
            Box::new(|| a.gather_mm(&b, None, Some(&ids), false)),
        ),
        (
            "sorted",
            Box::new(|| a.gather_mm(&b, None, Some(&sorted), true)),
        ),
        (
            "transposed",
            Box::new(|| a.gather_mm(&bt.transpose([0, 2, 1]), None, Some(&ids), false)),
        ),
        (
            "lhs",
            Box::new(|| {
                let lhs = Array::from(vec![8u32, 1, 1, 0]);
                a.gather_mm(&b, Some(&lhs), Some(&ids.i(..4)), false)
            }),
        ),
    ];
    for (name, f) in &cases {
        let got = crate::common::with_timeout(std::time::Duration::from_secs(30), {
            let f: &dyn Fn() -> Array = f.as_ref();
            let g = on(Device::gpu(0), f).to_vec::<f32>().unwrap();
            move || g
        });
        let want = on(Device::cpu(), f.as_ref()).to_vec::<f32>().unwrap();
        assert_eq!(got.len(), want.len(), "{name}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-5, "{name} {i}: {g} vs {w}");
        }
    }
}
