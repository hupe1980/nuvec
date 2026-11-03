//! The CUDA backend on a real GPU: generated element-wise and reduction kernels and the Tile
//! GEMM against CPU references (`scripts/cuda_remote.py`).

#![cfg(all(feature = "cuda", target_os = "linux"))]

use half::{bf16, f16};
use nuvec_gpu::cuda::kernels::{self, View};
use nuvec_gpu::cuda::ptx::{DType, Op};
use nuvec_gpu::cuda::{Buffer, Device, tile};

fn device() -> Option<&'static Device> {
    match Device::shared() {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(
                std::env::var("NUVEC_REQUIRE_GPU").as_deref() != Ok("1"),
                "NUVEC_REQUIRE_GPU=1 but {e}"
            );
            eprintln!("skipped: {e}");
            None
        }
    }
}

/// A deterministic generator of values in [-2, 2).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next().abs() * 1e6) as usize % n
    }
}

const DTYPES: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

/// `x` rounded to `dtype` (as f32).
fn round(dtype: DType, x: f32) -> f32 {
    match dtype {
        DType::F32 => x,
        DType::F16 => f16::from_f32(x).to_f32(),
        DType::BF16 => bf16::from_f32(x).to_f32(),
    }
}

fn upload(d: &'static Device, dtype: DType, v: &[f32]) -> Buffer {
    let mut b = Buffer::new(d, v.len() * dtype.size()).unwrap();
    let bytes = b.as_mut_slice();
    for (i, &x) in v.iter().enumerate() {
        match dtype {
            DType::F32 => bytes[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes()),
            DType::F16 => bytes[2 * i..2 * i + 2].copy_from_slice(&f16::from_f32(x).to_le_bytes()),
            DType::BF16 => {
                bytes[2 * i..2 * i + 2].copy_from_slice(&bf16::from_f32(x).to_le_bytes())
            }
        }
    }
    b
}

fn download(d: &'static Device, dtype: DType, b: &Buffer) -> Vec<f32> {
    d.synchronize().unwrap();
    let bytes = b.as_slice();
    (0..b.len() / dtype.size())
        .map(|i| match dtype {
            DType::F32 => f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()),
            DType::F16 => f16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap()).to_f32(),
            DType::BF16 => {
                bf16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap()).to_f32()
            }
        })
        .collect()
}

fn reference(op: Op, a: f32, b: f32) -> f32 {
    match op {
        Op::Add => a + b,
        Op::Mul => a * b,
        Op::Exp => a.exp(),
    }
}

/// Whether `got` is `want` to within `ulps` units in the last place of `dtype`.
fn close(dtype: DType, got: f32, want: f32, ulps: f32) -> bool {
    if want.is_nan() {
        return got.is_nan();
    }
    if want.is_infinite() || got == want {
        return got == want;
    }
    let eps = match dtype {
        DType::F32 => f32::EPSILON,
        DType::F16 => 2f32.powi(-10),
        DType::BF16 => 2f32.powi(-7),
    };
    (got - want).abs() <= ulps * eps * want.abs().max(f32::MIN_POSITIVE)
}

#[test]
fn elementwise_matches_the_cpu() {
    let Some(d) = device() else { return };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for dtype in DTYPES {
        for op in [Op::Add, Op::Mul, Op::Exp] {
            for n in [1usize, 255, 1000, (1 << 20) + 3] {
                let a: Vec<f32> = (0..n).map(|_| round(dtype, rng.next() * 20.0)).collect();
                let b: Vec<f32> = (0..n).map(|_| round(dtype, rng.next())).collect();
                let (ba, bb) = (upload(d, dtype, &a), upload(d, dtype, &b));
                let mut out = Buffer::new(d, n * dtype.size()).unwrap();
                let views = [
                    View {
                        buffer: &ba,
                        offset: 0,
                        strides: &[],
                    },
                    View {
                        buffer: &bb,
                        offset: 0,
                        strides: &[],
                    },
                ];
                kernels::elementwise(d, op, dtype, &mut out, &views[..op.arity()], n).unwrap();
                let got = download(d, dtype, &out);
                // Add and multiply are exact (computed in f32, rounded once); exp within
                // 2 ulps of f32 (then rounded to the type).
                let ulps = match (op, dtype) {
                    (Op::Exp, DType::F32) => 2.0,
                    // One rounding boundary of the half type apart, at most.
                    (Op::Exp, _) => 1.0,
                    _ => 0.0,
                };
                for i in 0..n {
                    let want = round(dtype, reference(op, a[i], b[i]));
                    assert!(
                        close(dtype, got[i], want, ulps),
                        "{op:?} {dtype:?} n={n} i={i}: {} → {} vs {want}",
                        a[i],
                        got[i]
                    );
                }
            }
        }
    }
    // exp's edges: overflow, underflow to subnormals and zero, NaN.
    let edge = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        88.0,
        88.8,
        89.5,
        -87.0,
        -100.0,
        -103.9,
        -110.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let ba = upload(d, DType::F32, &edge);
    let mut out = Buffer::new(d, edge.len() * 4).unwrap();
    kernels::elementwise(
        d,
        Op::Exp,
        DType::F32,
        &mut out,
        &[View {
            buffer: &ba,
            offset: 0,
            strides: &[],
        }],
        edge.len(),
    )
    .unwrap();
    for (x, y) in edge.iter().zip(download(d, DType::F32, &out)) {
        let want = x.exp();
        let ok =
            close(DType::F32, y, want, 2.0) || (want.abs() < 1e-37 && (y - want).abs() < 1e-44);
        assert!(ok, "exp({x}) = {y}, want {want}");
    }
}

#[test]
fn strided_and_broadcast_operands_match_the_cpu() {
    let Some(d) = device() else { return };
    let mut rng = Rng(7);
    for dtype in DTYPES {
        for case in 0..60 {
            let ndim = 1 + case % 4;
            let shape: Vec<usize> = (0..ndim).map(|_| 1 + rng.below(9)).collect();
            let n: usize = shape.iter().product();
            // Each input: a random permutation of a contiguous layout, some axes broadcast,
            // some reversed, at a random offset into a larger buffer.
            let mut views = Vec::new();
            for _ in 0..2 {
                let mut strides = vec![0i64; ndim];
                let mut s = 1i64;
                let mut order: Vec<usize> = (0..ndim).collect();
                order.rotate_left(rng.below(ndim));
                for &ax in order.iter().rev() {
                    strides[ax] = s;
                    s *= shape[ax] as i64;
                }
                for st in strides.iter_mut() {
                    match rng.below(4) {
                        0 => *st = 0,
                        1 => *st = -*st,
                        _ => {}
                    }
                }
                let span: i64 = strides
                    .iter()
                    .zip(&shape)
                    .map(|(st, &d)| st.abs() * (d as i64 - 1))
                    .sum();
                let neg: i64 = strides
                    .iter()
                    .zip(&shape)
                    .filter(|(st, _)| **st < 0)
                    .map(|(st, &d)| -st * (d as i64 - 1))
                    .sum();
                let pad = rng.below(5) as i64;
                let len = (span + 1 + pad) as usize;
                let data: Vec<f32> = (0..len).map(|_| round(dtype, rng.next())).collect();
                views.push((data, strides, (neg + pad) as usize));
            }
            let buffers: Vec<Buffer> = views.iter().map(|(v, _, _)| upload(d, dtype, v)).collect();
            let strides32: Vec<Vec<i32>> = views
                .iter()
                .map(|(_, s, _)| s.iter().map(|&x| x as i32).collect())
                .collect();
            let kviews: Vec<View<'_>> = (0..2)
                .map(|k| View {
                    buffer: &buffers[k],
                    offset: views[k].2,
                    strides: &strides32[k],
                })
                .collect();
            let mut out = Buffer::new(d, n * dtype.size()).unwrap();
            kernels::elementwise_strided(d, Op::Mul, dtype, &mut out, &kviews, &shape).unwrap();
            let got = download(d, dtype, &out);
            for (i, &value) in got.iter().enumerate() {
                let (mut rest, mut offs) = (i, [views[0].2 as i64, views[1].2 as i64]);
                for ax in (0..ndim).rev() {
                    let r = (rest % shape[ax]) as i64;
                    rest /= shape[ax];
                    for k in 0..2 {
                        offs[k] += r * views[k].1[ax];
                    }
                }
                let want = round(
                    dtype,
                    views[0].0[offs[0] as usize] * views[1].0[offs[1] as usize],
                );
                assert_eq!(value, want, "{dtype:?} shape {shape:?} i={i}");
            }
        }
    }
}

#[test]
fn row_sums_match_the_cpu() {
    let Some(d) = device() else { return };
    let mut rng = Rng(11);
    for dtype in DTYPES {
        for (rows, len) in [
            (1, 1),
            (7, 31),
            (3, 256),
            (5, 1000),
            (2, 100_000),
            (513, 17),
        ] {
            let x: Vec<f32> = (0..rows * len).map(|_| round(dtype, rng.next())).collect();
            let bx = upload(d, dtype, &x);
            let mut out = Buffer::new(d, rows * dtype.size()).unwrap();
            kernels::row_sum(d, dtype, &mut out, &bx, len, rows).unwrap();
            let got = download(d, dtype, &out);
            for r in 0..rows {
                let row = &x[r * len..(r + 1) * len];
                let want: f64 = row.iter().map(|&v| f64::from(v)).sum();
                let scale: f64 = row.iter().map(|&v| f64::from(v.abs())).sum();
                // f32 accumulation in another order, then rounding to the type.
                let tol = scale * 1e-6
                    + want.abs()
                        * match dtype {
                            DType::F32 => 0.0,
                            DType::F16 => 1e-3,
                            DType::BF16 => 8e-3,
                        };
                assert!(
                    (f64::from(got[r]) - want).abs() <= tol + 1e-6,
                    "{dtype:?} {rows}×{len} row {r}: {} vs {want}",
                    got[r]
                );
            }
        }
    }
}

#[test]
fn tile_gemm_matches_the_cpu() {
    let Some(d) = device() else { return };
    let mut rng = Rng(13);
    for dtype in DTYPES {
        for (m, n, k) in [
            (1, 1, 1),
            (17, 33, 65),
            (64, 64, 32),
            (128, 128, 128),
            (255, 129, 77),
            (300, 1000, 513),
            (1024, 1024, 1024),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| round(dtype, rng.next())).collect();
            let b: Vec<f32> = (0..k * n).map(|_| round(dtype, rng.next())).collect();
            let (ba, bb) = (upload(d, dtype, &a), upload(d, dtype, &b));
            let mut out = Buffer::new(d, m * n * dtype.size()).unwrap();
            tile::gemm(d, dtype, &mut out, &ba, &bb, (m, n, k))
                .unwrap_or_else(|e| panic!("{dtype:?} {m}×{n}×{k}: {e}"));
            let got = download(d, dtype, &out);
            // Check a sample of entries on big shapes, all on small ones.
            let picks: Vec<(usize, usize)> = if m * n <= 40_000 {
                (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).collect()
            } else {
                (0..2000).map(|_| (rng.below(m), rng.below(n))).collect()
            };
            for (i, j) in picks {
                let want: f64 = (0..k)
                    .map(|p| f64::from(a[i * k + p]) * f64::from(b[p * n + j]))
                    .sum();
                let scale: f64 = (0..k)
                    .map(|p| f64::from((a[i * k + p] * b[p * n + j]).abs()))
                    .sum();
                let tol = scale * (k as f64).sqrt() * 2e-7
                    + want.abs()
                        * match dtype {
                            DType::F32 => 1e-6,
                            DType::F16 => 1e-3,
                            DType::BF16 => 8e-3,
                        };
                assert!(
                    (f64::from(got[i * n + j]) - want).abs() <= tol + 1e-5,
                    "{dtype:?} {m}×{n}×{k} [{i},{j}]: {} vs {want}",
                    got[i * n + j]
                );
            }
        }
    }
}
