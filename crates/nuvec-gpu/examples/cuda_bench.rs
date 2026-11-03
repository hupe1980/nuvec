//! CUDA measurements as JSON lines: element-wise bandwidth, a row reduction, and Tile GEMM
//! throughput. `scripts/bench_cuda.py` measures MLX on the same shapes.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("cuda_bench: the CUDA runtime exists on Linux only");
}

#[cfg(target_os = "linux")]
fn main() {
    imp::main();
}

#[cfg(target_os = "linux")]
mod imp {
    use std::time::Instant;

    use nuvec_gpu::cuda::kernels::{self, View};
    use nuvec_gpu::cuda::ptx::{DType, Op};
    use nuvec_gpu::cuda::{Buffer, Device, tile};

    fn time(d: &'static Device, iters: usize, mut f: impl FnMut()) -> f64 {
        for _ in 0..3 {
            f();
        }
        d.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        d.synchronize().unwrap();
        t.elapsed().as_secs_f64() / iters as f64
    }

    fn filled(d: &'static Device, bytes: usize) -> Buffer {
        let mut b = Buffer::new(d, bytes).unwrap();
        // Small values in every dtype (0x3c00 = 1.0 in f16 pairs; also a sane f32/bf16).
        for (i, x) in b.as_mut_slice().iter_mut().enumerate() {
            *x = if i % 2 == 0 { 0x00 } else { 0x3c };
        }
        b
    }

    fn name(d: DType) -> &'static str {
        match d {
            DType::F32 => "float32",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
        }
    }

    pub(crate) fn main() {
        // `cuda_bench` (everything), `cuda_bench gemm f32` (the GEMM rows of one dtype), or
        // `cuda_bench sweep f16` (one 4096 GEMM row, for tile sweeps with NUVEC_CUDA_GEMM_TILES).
        let args: Vec<String> = std::env::args().skip(1).collect();
        let only = args.first().map(String::as_str);
        let dtypes: Vec<DType> = match args.get(1).map(String::as_str) {
            Some("f32") => vec![DType::F32],
            Some("f16") => vec![DType::F16],
            Some("bf16") => vec![DType::BF16],
            _ => vec![DType::F16, DType::BF16, DType::F32],
        };
        let sweep_size: Vec<usize> = args
            .get(2)
            .and_then(|s| s.parse().ok())
            .into_iter()
            .collect();
        let sizes: &[usize] = if only == Some("sweep") {
            if sweep_size.is_empty() {
                &[4096]
            } else {
                &sweep_size
            }
        } else {
            &[1024, 2048, 4096, 8192]
        };
        let d = Device::shared().expect("a CUDA device");
        let info = d.info();
        println!(
            "{{\"device\":\"{}\",\"arch\":\"{}\",\"driver\":{},\"tool\":\"nuvec\"}}",
            info.name,
            info.arch(),
            info.driver_version
        );
        let n = 1 << 26;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            if only.is_some() {
                break;
            }
            let (a, b) = (filled(d, n * dtype.size()), filled(d, n * dtype.size()));
            let mut out = Buffer::new(d, n * dtype.size()).unwrap();
            for op in [Op::Add, Op::Exp] {
                let views = [
                    View {
                        buffer: &a,
                        offset: 0,
                        strides: &[],
                    },
                    View {
                        buffer: &b,
                        offset: 0,
                        strides: &[],
                    },
                ];
                let s = time(d, 20, || {
                    kernels::elementwise(d, op, dtype, &mut out, &views[..op.arity()], n).unwrap()
                });
                let bytes = (op.arity() + 1) * n * dtype.size();
                println!(
                    "{{\"op\":\"{op:?}\",\"dtype\":\"{}\",\"n\":{n},\"ms\":{:.4},\"gbps\":{:.1}}}",
                    name(dtype),
                    s * 1e3,
                    bytes as f64 / s / 1e9
                );
            }
            let (rows, len) = (8192, 8192);
            let mut sums = Buffer::new(d, rows * dtype.size()).unwrap();
            let s = time(d, 20, || {
                kernels::row_sum(d, dtype, &mut sums, &a, len, rows).unwrap()
            });
            println!(
                "{{\"op\":\"RowSum\",\"dtype\":\"{}\",\"shape\":[{rows},{len}],\"ms\":{:.4},\"gbps\":{:.1}}}",
                name(dtype),
                s * 1e3,
                (rows * len * dtype.size()) as f64 / s / 1e9
            );
        }
        for &dtype in &dtypes {
            for &size in sizes {
                let (a, b) = (
                    filled(d, size * size * dtype.size()),
                    filled(d, size * size * dtype.size()),
                );
                let mut out = Buffer::new(d, size * size * dtype.size()).unwrap();
                let first = Instant::now();
                tile::gemm(d, dtype, &mut out, &a, &b, (size, size, size)).unwrap();
                let jit = first.elapsed().as_secs_f64();
                let iters = if size >= 8192 { 5 } else { 20 };
                let s = time(d, iters, || {
                    tile::gemm(d, dtype, &mut out, &a, &b, (size, size, size)).unwrap()
                });
                println!(
                    "{{\"op\":\"Gemm\",\"dtype\":\"{}\",\"size\":{size},\"ms\":{:.4},\"tflops\":{:.2},\"first_call_s\":{:.2}}}",
                    name(dtype),
                    s * 1e3,
                    2.0 * (size as f64).powi(3) / s / 1e12,
                    jit
                );
            }
        }
    }
}
