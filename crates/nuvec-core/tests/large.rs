//! Arrays of 2^32 elements and more on the GPU (64-bit indexing, as MLX): element-wise
//! operations (flat ones in chunks, strided ones with 64-bit offsets, in place too), matrix
//! products, row softmax and the attention composition over 8 GiB of scores. Benchmark tier
//! (up to ~14 GB of GPU memory; run in the release checklist).

mod common;

use nuvec_core::ops;
use nuvec_core::*;

const R: usize = (1 << 16) + 1;
const C: usize = 1 << 16;

/// Flat positions around the 2^31-element chunks and the 2^32 boundary, and the last.
fn positions() -> Vec<usize> {
    let mut f = vec![
        0,
        1,
        (1 << 31) - 1,
        1 << 31,
        (1 << 31) + 1,
        (1 << 32) - 1,
        1 << 32,
    ];
    f.extend([(1 << 32) + 1, R * C - 1, 3_000_000_007]);
    f
}

fn u8_at(a: &Array, f: usize) -> u8 {
    a.i((f / C, f % C)).item::<u8>().unwrap()
}

fn elementwise() {
    let rows = ops::reshape(
        &ops::astype(&ops::arange(0, R as i32, 1), DType::U8),
        [R as i32, 1],
    );
    let cols = ops::reshape(
        &ops::astype(&ops::arange(0, C as i32, 1), DType::U8),
        [1, C as i32],
    );
    // Strided (broadcast) operands: 64-bit offsets.
    let x = ops::add(&rows, &cols);
    // Flat and scalar operands: in chunks.
    let y = ops::multiply(&x, Array::from(3u8));
    let z = ops::add(&y, &x);
    eval(&[&x, &y, &z]).unwrap();
    // In place: the sum's buffer, with no other handle, takes the product.
    let w = ops::multiply(ops::add(&x, Array::from(1u8)), Array::from(2u8));
    w.eval().unwrap();
    for f in positions() {
        let v = ((f / C) % 256 + f % C % 256) as u8;
        assert_eq!(u8_at(&x, f), v, "x at {f}");
        assert_eq!(u8_at(&y, f), v.wrapping_mul(3), "y at {f}");
        assert_eq!(u8_at(&z, f), v.wrapping_mul(4), "z at {f}");
        assert_eq!(u8_at(&w, f), v.wrapping_add(1).wrapping_mul(2), "w at {f}");
    }
}

fn matmul() {
    const K: usize = 16;
    let a: Vec<f32> = (0..R * K).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..K * C).map(|i| (i % 5) as f32 - 2.0).collect();
    let ga = ops::astype(&Array::from_slice(&a, [R, K]), DType::BF16);
    let gb = ops::astype(&Array::from_slice(&b, [K, C]), DType::BF16);
    let out = ops::matmul(&ga, &gb);
    out.eval().unwrap();
    for f in positions() {
        let (r, c) = (f / C, f % C);
        let want: f32 = (0..K).map(|k| a[r * K + k] * b[k * C + c]).sum();
        let got = ops::astype(&out.i((r, c)), DType::F32)
            .item::<f32>()
            .unwrap();
        assert_eq!(got, want, "matmul at {f}");
    }
}

fn softmax() {
    let cols = ops::astype(&ops::arange(0, C as i32, 1), DType::F32);
    let row = ops::astype(
        &ops::reshape(
            &cols
                .subtract(cols.floor_divide(13.0f32).multiply(13.0f32))
                .multiply(0.25f32),
            [1, C as i32],
        ),
        DType::BF16,
    );
    // Row-contiguous rows, all equal to `row`.
    let x = ops::add(&row, ops::zeros([R, 1], DType::BF16));
    let s = ops::softmax(&x, -1);
    let one = ops::softmax(&row, -1);
    eval(&[&s, &one]).unwrap();
    let want = one.to_vec::<bf16>().unwrap();
    for r in [0, 32_768, R - 1] {
        let got = s.i(r).to_vec::<bf16>().unwrap();
        assert!(got == want, "softmax row {r}");
    }
}

/// Head size 256 over 131K keys: the composition (MLX's fallback) materialises 8 GiB of
/// scores; each key head's half (4 GiB, below 2^32 elements) computes the same rows.
fn attention() {
    let q = random::uniform(-1.0, 1.0, [1, 16, 2048, 256], DType::F32, None).astype(DType::BF16);
    let k = random::uniform(-1.0, 1.0, [1, 2, 131_072, 256], DType::F32, None).astype(DType::BF16);
    eval(&[&q, &k]).unwrap();
    let options = fast::AttentionOptions::causal();
    let full = fast::scaled_dot_product_attention(&q, &k, &k, 0.0625, &options);
    full.eval().unwrap();
    for h in 0..2i32 {
        let (qh, kh) = (q.i((.., h * 8..h * 8 + 8)), k.i((.., h..h + 1)));
        let half = fast::scaled_dot_product_attention(&qh, &kh, &kh, 0.0625, &options);
        let got = full.i((.., h * 8..h * 8 + 8));
        let diff = got
            .astype(DType::F32)
            .subtract(half.astype(DType::F32))
            .abs();
        let worst = diff.max(..).item::<f32>().unwrap();
        assert!(worst <= 1e-2, "key head {h}: {worst}");
    }
}

/// Each case's kernels, so a fallback to the CPU fails (its results are right, 50× slower).
#[test]
#[ignore = "benchmark tier: arrays of 2^32 elements (~14 GB of GPU memory)"]
fn arrays_of_2_32_elements_run_on_the_gpu() {
    if common::require_gpu().is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let profiled = std::env::var_os("NUVEC_PROFILE").is_some();
    if !profiled {
        // The same test in a child with the profiler on.
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "arrays_of_2_32_elements_run_on_the_gpu",
                "--ignored",
                "--nocapture",
            ])
            .env("NUVEC_PROFILE", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        for (case, kernel) in [
            ("elementwise", "nuvec_binary_add_uint8_g3"),
            ("elementwise", "nuvec_binary_mul_uint8_c"),
            ("elementwise", "nuvec_binary_add_uint8_c"),
            ("matmul", "nuvec_gemm_nn_bfloat16"),
            ("softmax", "nuvec_softmax_looped_bfloat16"),
            ("attention", "nuvec_gemm_nt_bfloat16"),
            ("attention", "nuvec_select_16_g3"),
            ("attention", "nuvec_softmax_looped_bfloat16"),
        ] {
            let ran = stdout.lines().any(|l| {
                l.strip_prefix(&format!("@@{case} "))
                    .is_some_and(|k| k.starts_with(kernel))
            });
            assert!(ran, "{case}: no {kernel} on the GPU in\n{stdout}");
        }
        return;
    }
    for (case, run) in [
        ("elementwise", elementwise as fn()),
        ("matmul", matmul),
        ("softmax", softmax),
        ("attention", attention),
    ] {
        nuvec_gpu::profile::reset();
        run();
        for (kernel, _, _) in nuvec_gpu::profile::report() {
            println!("@@{case} {kernel}");
        }
    }
}
