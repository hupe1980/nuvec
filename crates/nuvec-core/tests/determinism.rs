//! Run-to-run determinism on the GPU: the same inputs give bit-identical results across
//! runs, whatever memory the allocator hands out and whatever is still in flight.
//!
//! The workload's shapes are chosen to reach the kernel variants that combine partial
//! results or run in several passes: split-K GEMM, quantized products, the two-pass decode
//! attention, multi-pass reductions, multi-block sorts and scans, the sorted MoE gather and
//! Gated DeltaNet. Each run is submitted before the previous one is read, and runs are
//! interleaved with allocations of other sizes, so buffers are reused while GPU work that
//! used them is still pending.
//!
//! Not covered, because they are nondeterministic by design (as in MLX): scatter-add with
//! repeated indices (float atomics) and `put_along_axis` with repeated indices (the last
//! write wins in no fixed order).

#![cfg(target_os = "macos")]

use nuvec_core::*;

fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = !cfg!(miri) && (&Array::scalar(1.0f32) + 1.0).eval().is_ok();
    if !ok {
        eprintln!("skipped: no Metal device");
    }
    ok
}

/// Fixed inputs in about [-1, 1], one pattern per seed.
fn normal(shape: &[usize], dtype: DType, seed: u64) -> Array {
    let n: usize = shape.iter().product();
    let t = arange(0.0, n as f64, 1.0, DType::F32).unwrap();
    let x = sin(&(&t * (0.7381 + seed as f64 * 0.0137))).unwrap();
    let x = &x * cos(&(&t * (0.0071 * seed as f64 + 0.013))).unwrap();
    let shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    x.reshape(&shape).unwrap().astype(dtype).unwrap()
}

/// Every operation of the workload, from inputs fixed by their seeds.
fn workload() -> Result<Vec<Array>> {
    let (f32_, bf16_) = (DType::F32, DType::BF16);
    let mut out = Vec::new();
    // GEMM: split-K (few output tiles, long K; also the bf16 shape), tiled (short K), and a
    // matrix-vector product.
    out.push(matmul(
        &normal(&[32, 16384], f32_, 1),
        &normal(&[16384, 32], f32_, 2),
    )?);
    out.push(matmul(
        &normal(&[512, 1024], bf16_, 3),
        &normal(&[1024, 768], bf16_, 4),
    )?);
    out.push(matmul(
        &normal(&[2048, 256], bf16_, 34),
        &normal(&[256, 2048], bf16_, 35),
    )?);
    let w = normal(&[4096, 4096], bf16_, 5);
    out.push(matmul(&normal(&[1, 4096], bf16_, 6), &w.t()?)?);
    // Quantized: matrix-vector, matrix-matrix, and the transposed-weight product.
    let (wq, scales, biases) = quantize(&w, 64, 4)?;
    for (rows, seed) in [(1, 7), (8, 8), (256, 9)] {
        let x = normal(&[rows, 4096], bf16_, seed);
        out.push(quantized_matmul(&x, &wq, &scales, &biases, true, 64, 4)?);
    }
    out.push(quantized_matmul(
        &normal(&[1, 4096], bf16_, 10),
        &wq,
        &scales,
        &biases,
        false,
        64,
        4,
    )?);
    // Mixture of experts: rows sorted by expert through 16 quantized experts.
    let experts = normal(&[16, 512, 1024], bf16_, 11);
    let (eq, es, eb) = quantize(&experts, 64, 4)?;
    let ids: Vec<u32> = (0..512u32).map(|i| i * 16 / 512).collect();
    let ids = Array::from_slice(&ids, &[512])?;
    out.push(gather_qmm(
        &normal(&[512, 1, 1024], bf16_, 12),
        &eq,
        &es,
        &eb,
        None,
        Some(&ids),
        true,
        64,
        4,
        true,
    )?);
    // Attention: decode over many keys (two passes), causal prefill (tiled).
    let (k, v) = (
        normal(&[1, 4, 8192, 128], bf16_, 13),
        normal(&[1, 4, 8192, 128], bf16_, 14),
    );
    out.push(fast::scaled_dot_product_attention(
        &normal(&[1, 16, 1, 128], bf16_, 15),
        &k,
        &v,
        0.088,
        fast::AttentionMask::None,
        None,
    )?);
    let (k, v) = (
        normal(&[1, 4, 300, 128], bf16_, 16),
        normal(&[1, 4, 300, 128], bf16_, 17),
    );
    out.push(fast::scaled_dot_product_attention(
        &normal(&[1, 16, 300, 128], bf16_, 18),
        &k,
        &v,
        0.088,
        fast::AttentionMask::Causal,
        None,
    )?);
    // Reductions: a full sum in several passes, row reductions, softmax.
    out.push(sum(&normal(&[1 << 22], f32_, 19), &[], false)?);
    let rows = normal(&[64, 32768], bf16_, 20);
    out.push(logsumexp(&rows, &[-1], false)?);
    out.push(softmax(&rows, &[-1])?);
    out.push(max(&rows, &[0], false)?);
    out.push(sum(&rows, &[0], false)?);
    // Multi-block sort and scan.
    let long = normal(&[4, 50000], f32_, 21);
    out.push(sort(&long, Some(-1))?);
    out.push(argsort(&long, Some(-1))?);
    out.push(cumsum(
        &normal(&[4, 100000], f32_, 22),
        Some(-1),
        false,
        true,
    )?);
    // Fused norms and rotations.
    let (h, n) = fast::add_rms_norm(
        &normal(&[64, 4096], bf16_, 23),
        &normal(&[64, 4096], bf16_, 24),
        Some(&normal(&[4096], bf16_, 25)),
        1e-6,
    )?;
    out.extend([h, n]);
    let rotation = fast::Rotation {
        dims: 128,
        traditional: false,
        base: Some(1e6),
        scale: 1.0,
        freqs: None,
    };
    out.push(fast::norm_rope(
        &transpose(&normal(&[1, 64, 8, 128], bf16_, 26), &[0, 2, 1, 3])?,
        Some((&normal(&[128], bf16_, 27), 1e-6)),
        rotation,
        40,
        Some(&normal(&[1, 8, 256, 128], bf16_, 28)),
    )?);
    // Gated DeltaNet.
    let (y, state) = fast::gated_delta(
        &(normal(&[1, 64, 4, 128], bf16_, 29) * 0.05),
        &(normal(&[1, 64, 4, 128], bf16_, 30) * 0.05),
        &normal(&[1, 64, 8, 128], bf16_, 31),
        &sigmoid(&normal(&[1, 64, 8], f32_, 32))?,
        &sigmoid(&normal(&[1, 64, 8], bf16_, 33))?,
        &zeros(&[1, 8, 128, 128], f32_)?,
    )?;
    out.extend([y, state]);
    Ok(out)
}

/// The bits of every element (floats widened exactly to float32, integers to int64).
fn bits(a: &Array) -> Vec<u64> {
    match a.dtype().category() {
        Category::Float => a
            .astype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        _ => a
            .astype(DType::I64)
            .unwrap()
            .to_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as u64)
            .collect(),
    }
}

#[test]
fn results_are_bit_identical_across_runs() {
    if !gpu() {
        return;
    }
    let first: Vec<Vec<u64>> = workload().unwrap().iter().map(bits).collect();
    let mut pending: Option<Vec<Array>> = None;
    for run in 0..5 {
        // Other allocations, some kept alive, so later runs get different buffers.
        let noise: Vec<Array> = (0..(run + 1) * 3)
            .map(|i| normal(&[1000 + 997 * i, 64], DType::F32, 100 + i as u64) * 2.0)
            .collect();
        async_eval(&noise).unwrap();
        // Submit this run before reading the previous one: its buffers come from memory
        // that pending work still uses.
        let outputs = workload().unwrap();
        async_eval(&outputs).unwrap();
        drop(noise);
        if let Some(previous) = pending.replace(outputs) {
            check(&first, &previous, run);
        }
    }
    check(&first, &pending.unwrap(), 5);
}

fn check(first: &[Vec<u64>], outputs: &[Array], run: usize) {
    assert_eq!(first.len(), outputs.len());
    for (i, (want, got)) in first.iter().zip(outputs).enumerate() {
        let got = bits(got);
        let differ = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        assert_eq!(
            differ, 0,
            "run {run}: output {i} differs in {differ} elements"
        );
    }
}
