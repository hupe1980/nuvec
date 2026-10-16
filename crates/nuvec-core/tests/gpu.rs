//! Evaluation on the GPU stream: MLX fixtures, GPU-vs-CPU differential tests, fallback,
//! and long graphs. Every test skips when no Metal device is available.

#![cfg(target_os = "macos")]

mod common;
#[path = "common/fixture.rs"]
mod fixture;

use common::{Rng, random_layout};
use nuvec_core::*;

/// Whether the GPU stream works here.
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

/// Runs `f` with `device` as this thread's default, restoring the previous default.
fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
    let previous = default_stream();
    set_default_stream(Stream::default_for(device));
    let r = f();
    set_default_stream(previous);
    r
}

#[test]
fn mlx_fixtures_on_gpu() {
    if !gpu_available() {
        return;
    }
    let (_, cases) = fixture::load();
    let failures: Vec<String> = on(Device::gpu(0), || {
        cases
            .iter()
            .filter_map(
                |case| match fixture::run(case).and_then(|a| fixture::from_array(&a)) {
                    Ok(t) => fixture::compare(case, &t, &case.outputs[0]).err(),
                    Err(e) => Some(format!("{} ({}): nuvec error: {e}", case.name, case.op)),
                },
            )
            .collect()
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
fn same_on_both<T: Element + common::Same>(name: &str, f: impl Fn() -> Result<Array>) {
    let cpu = on(Device::cpu(), &f).unwrap().to_vec::<T>().unwrap();
    let gpu = on(Device::gpu(0), &f).unwrap().to_vec::<T>().unwrap();
    assert_eq!(cpu.len(), gpu.len(), "{name}");
    for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
        assert!(c.same(*g), "{name}: element {i}: cpu {c:?}, gpu {g:?}");
    }
}

#[test]
fn elementwise_matches_cpu_on_random_layouts() {
    if !gpu_available() {
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
        same_on_both::<f32>(&ctx("add"), || add(&x, &y));
        same_on_both::<f32>(&ctx("div"), || divide(&x, &y));
        same_on_both::<f32>(&ctx("max"), || maximum(&x, &y));
        same_on_both::<bool>(&ctx("le"), || less_equal(&x, &y));
        same_on_both::<i32>(&ctx("mul i32"), || multiply(&i, &i));
        same_on_both::<i32>(&ctx("neg i32"), || negative(&i));
        same_on_both::<f16>(&ctx("sub f16"), || subtract(&h, &h.t()?.t()?));
        same_on_both::<f32>(&ctx("where"), || where_(&greater(&x, &y)?, &x, &y));
        same_on_both::<i8>(&ctx("cast"), || x.astype(DType::I8));
        same_on_both::<f32>(&ctx("copy"), || x.t()?.reshape(&[-1]));
    }
}

/// A unary operation under test.
type UnaryFn = fn(&Array) -> Result<Array>;

#[test]
fn transcendentals_are_close_to_cpu() {
    if !gpu_available() {
        return;
    }
    let x = arange(-6.0, 6.0, 0.01, DType::F32).unwrap();
    let ops: [(&str, UnaryFn); 7] = [
        ("exp", exp),
        ("log", log),
        ("sqrt", sqrt),
        ("sin", sin),
        ("cos", cos),
        ("tanh", tanh),
        ("sigmoid", sigmoid),
    ];
    for (name, op) in ops {
        let cpu = on(Device::cpu(), || op(&x))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let gpu = on(Device::gpu(0), || op(&x))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
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
    if !gpu_available() {
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
        let axis = rng.below(shape.len()) as isize;
        same_on_both::<i32>("sum i32", || sum(&ia, &[axis], false));
        same_on_both::<i32>("max i32", || max(&ia, &[], true));
        same_on_both::<f32>("min f32", || min(&fa, &[axis], false));
        same_on_both::<u32>("argmax", || argmax(&fa, Some(axis), false));
        same_on_both::<bool>("any", || {
            any(&greater(&fa, Array::scalar(0.5f32))?, &[axis], false)
        });
        // Float sums: the GPU accumulates in f32, the CPU in f64.
        let c = on(Device::cpu(), || sum(&fa, &[axis], false))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let g = on(Device::gpu(0), || sum(&fa, &[axis], false))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let n = shape[axis as usize] as f32;
        assert!(
            c.iter()
                .zip(&g)
                .all(|(c, g)| (c - g).abs() <= (n + 1.0) * f32::EPSILON * 10.0 * c.abs().max(1.0))
        );
    }
    for (m, k, n) in [(1, 1, 1), (17, 33, 9), (64, 64, 64), (5, 0, 3)] {
        let a = sin(&arange(0.0, (m * k) as f64, 1.0, DType::F32)
            .unwrap()
            .reshape(&[m as isize, k as isize])
            .unwrap())
        .unwrap();
        let b = cos(&arange(0.0, (k * n) as f64, 1.0, DType::F32)
            .unwrap()
            .reshape(&[k as isize, n as isize])
            .unwrap())
        .unwrap();
        let c = on(Device::cpu(), || matmul(&a, &b.t()?.t()?))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let g = on(Device::gpu(0), || matmul(&a, &b.t()?.t()?))
            .unwrap()
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
    let a = sin(&arange(0.0, 2.0 * 3.0 * 4.0, 1.0, DType::F32)
        .unwrap()
        .reshape(&[2, 1, 3, 4])
        .unwrap())
    .unwrap();
    let b = cos(&arange(0.0, 5.0 * 2.0 * 4.0, 1.0, DType::F32)
        .unwrap()
        .reshape(&[5, 2, 4])
        .unwrap())
    .unwrap();
    let bt = transpose(&b, &[0, 2, 1]).unwrap();
    let c = on(Device::cpu(), || matmul(&a, &bt))
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let g = on(Device::gpu(0), || matmul(&a, &bt))
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert!(c.iter().zip(&g).all(|(c, g)| (c - g).abs() <= 1e-5));
}

#[test]
fn unsupported_work_falls_back_to_cpu() {
    if !gpu_available() {
        return;
    }
    // float64 has no Metal type, complex sums have no kernel yet: both run on the CPU, in
    // between GPU kernels, over the same memory.
    let x = arange(0.0, 100.0, 1.0, DType::F32).unwrap();
    let y = on(Device::gpu(0), || {
        let doubled = &x + &x; // GPU
        let wide = doubled.astype(DType::F64)?; // CPU (f64)
        let sq = &wide * &wide; // CPU (f64)
        let back = sq.astype(DType::F32)?; // CPU (from f64)
        let z = &back + &x; // GPU, reading CPU output
        let c = z.astype(DType::Complex64)?; // GPU
        sum(&c, &[], false) // CPU (complex sum)
    })
    .unwrap();
    let expected: f32 = (0..100).map(|i| (2 * i * 2 * i + i) as f32).sum();
    assert_eq!(
        y.item::<Complex64>().unwrap(),
        Complex64::new(expected, 0.0)
    );
}

#[test]
fn long_graphs_span_command_buffers() {
    if !gpu_available() {
        return;
    }
    let one = Array::full(&[1024], 1.0f32).unwrap();
    let x = on(Device::gpu(0), || {
        let mut x = zeros(&[1024], DType::F32).unwrap();
        for _ in 0..1000 {
            x = &x + &one;
        }
        x
    });
    assert!(x.to_vec::<f32>().unwrap().iter().all(|&v| v == 1000.0));
}

#[test]
fn scalars_empty_and_errors() {
    if !gpu_available() {
        return;
    }
    on(Device::gpu(0), || {
        let s = Array::scalar(3i64) * Array::scalar(-4i64);
        assert_eq!(s.item::<i64>().unwrap(), -12);
        let e = zeros(&[0, 3], DType::F32).unwrap();
        assert_eq!((&e + &e).shape(), &[0, 3]);
        assert_eq!(
            sum(&e, &[0], false).unwrap().to_vec::<f32>().unwrap(),
            vec![0.0; 3]
        );
        let bad = &zeros(&[2], DType::F32).unwrap() + &zeros(&[3], DType::F32).unwrap();
        assert!(matches!(bad.eval(), Err(Error::ShapeMismatch { .. })));
    });
}
