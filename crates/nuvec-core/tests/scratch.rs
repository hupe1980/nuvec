//! Scratch memory of the fused and product kernels: the GPU allocator's peak above inputs
//! and outputs stays within the formula documented in each operation's `# Memory` section,
//! and not below 80% of it where the formula is not zero. One test, so that nothing else
//! allocates while it measures. (Attention's scratch is checked against its plan in
//! `tests/attention_plan.rs`.)

#![cfg(target_os = "macos")]

mod common;

use nuvec_core::ops;
use nuvec_core::*;

fn pattern(shape: &[i32], dtype: DType, seed: u64) -> Array {
    let n: i32 = shape.iter().product();
    let t = ops::arange(0, n, 1).astype(DType::F32);
    (&t * (0.37 + seed as f64 * 0.01))
        .sin()
        .reshape(shape)
        .astype(dtype)
}

/// The peak bytes above the (evaluated) inputs and the output while `out` evaluates.
fn scratch(out: &Array) -> usize {
    let dev = nuvec_gpu::Device::shared().expect("Metal");
    dev.reset_peak();
    out.eval().unwrap();
    let m = dev.memory();
    m.peak - m.active
}

/// `measured` is within `[0.8, 1.0]` of `formula` (plus allocator rounding), or at most
/// the rounding when the formula is zero.
fn within(name: &str, measured: usize, formula: usize) {
    let slack = (64 << 10) + formula / 64;
    assert!(
        measured <= formula + slack,
        "{name}: measured {measured} above the formula's {formula}"
    );
    if formula > 0 {
        assert!(
            measured * 10 >= formula * 8,
            "{name}: measured {measured} below 80% of the formula's {formula}"
        );
    }
}

#[test]
fn scratch_matches_the_documented_formulas() {
    if common::require_gpu().is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let x = pattern(&[2048, 4096], DType::BF16, 1);
    let r = pattern(&[2048, 4096], DType::BF16, 2);
    let w = pattern(&[4096], DType::BF16, 3);
    let q = pattern(&[1, 32, 1024, 128], DType::BF16, 4);
    let targets: Array = (0..2048).map(|i: i32| i * 7 % 4096).collect();
    let offset = Array::from(5i32);
    eval(&[&x, &r, &w, &q, &targets, &offset]).unwrap();
    // The fused operations hold nothing beyond their inputs and outputs.
    within("rms_norm", scratch(&fast::rms_norm(&x, &w, 1e-6)), 0);
    within(
        "layer_norm",
        scratch(&fast::layer_norm(&x, &w, &w, 1e-6)),
        0,
    );
    let (h, n) = fast::add_rms_norm(&x, &r, &w, 1e-6);
    let dev = nuvec_gpu::Device::shared().expect("Metal");
    dev.reset_peak();
    eval(&[&h, &n]).unwrap();
    let m = dev.memory();
    within("add_rms_norm", m.peak - m.active, 0);
    drop((h, n));
    within(
        "rope",
        scratch(&fast::rope(&q, 128, false, Some(1e4), 1.0, &offset, None)),
        0,
    );
    within("softmax", scratch(&x.softmax(-1)), 0);
    within("logsumexp", scratch(&x.logsumexp(-1)), 0);
    within(
        "cross_entropy",
        scratch(&fast::cross_entropy(&x, &targets)),
        0,
    );
    // Split-K GEMM: `partitions × M × N × 4` (float32 partial products), with
    // `partitions = (K / 16 / tiles).next_power_of_two()` clamped to 2..=32 and
    // `tiles = ⌈M / 32⌉ · ⌈N / 32⌉`.
    let (m_, n_, k_) = (512usize, 512usize, 16384usize);
    let a = pattern(&[m_ as i32, k_ as i32], DType::F32, 5);
    let b = pattern(&[k_ as i32, n_ as i32], DType::F32, 6);
    eval(&[&a, &b]).unwrap();
    let tiles = m_.div_ceil(32) * n_.div_ceil(32);
    let partitions = (k_ / 16 / tiles).next_power_of_two().clamp(2, 32);
    within(
        "split-K GEMM",
        scratch(&a.matmul(&b)),
        partitions * m_ * n_ * 4,
    );
    // Few-row products (the row-independence paths) hold nothing.
    within(
        "gemv_rows",
        scratch(&a.i(..ROW_INDEPENDENT_ROWS as i32).matmul(&b)),
        0,
    );
    // Quantized split-K: `split × M × N × 4`, `split = min(512 / tiles, K / max(group, 32))`
    // reduced until it divides `K` (here 16).
    let wq = pattern(&[256, 8192], DType::BF16, 7);
    let (qw, qs, qb) = wq.quantize(64, 4);
    let xq = pattern(&[64, 8192], DType::BF16, 8);
    eval(&[&qw, &qs, &qb, &xq]).unwrap();
    let qtiles = 256usize.div_ceil(32) * 64usize.div_ceil(32);
    let mut split = (512 / qtiles).clamp(1, 8192 / 64);
    while split > 1 && 8192 % (split * 64) != 0 {
        split -= 1;
    }
    within(
        "quantized split-K",
        scratch(&xq.quantized_matmul(&qw, &qs, &qb, true, 64, 4)),
        split * 64 * 256 * 4,
    );
    // Sorted gather_qmm over many rows: the experts' counts and offsets (int32 each) and a
    // ones vector per row; a few bytes.
    let experts = pattern(&[8, 512, 1024], DType::BF16, 9);
    let (eq, es, eb) = experts.quantize(64, 4);
    let ids: Vec<u32> = (0..512u32).map(|i| i * 8 / 512).collect();
    let ids = Array::from(ids);
    let xe = pattern(&[512, 1, 1024], DType::BF16, 10);
    eval(&[&eq, &es, &eb, &xe, &ids]).unwrap();
    let gathered = xe.gather_qmm(&eq, &es, &eb, None, Some(&ids), true, 64, 4, true);
    let small = scratch(&gathered);
    assert!(small <= 64 << 10, "sorted gather_qmm: {small}");
}
