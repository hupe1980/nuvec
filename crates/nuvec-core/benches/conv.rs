//! Convolutions on the default device, compared against MLX by `scripts/bench_conv.py`.
//! Prints `name<TAB>dtype<TAB>ms` per case: milliseconds per convolution, ten independent
//! convolutions per evaluation, the best of ten timed evaluations after a warm-up.

use std::time::Instant;

use nuvec_core::{Array, DType, Result, ops, random};

/// A benchmarked case: its name, input shape, weight shape and the convolution.
type Case = (
    &'static str,
    &'static [usize],
    &'static [usize],
    fn(&Array, &Array) -> Array,
);

/// Vision-model shapes: a ViT patch embedding, ResNet blocks (64 and 128 channels, a stem,
/// unaligned channels), ConvNeXt's depthwise 7×7, a decoder's upsampling and a video stem.
pub const CASES: [Case; 14] = [
    ("patch16", &[1, 224, 224, 3], &[768, 16, 16, 3], |x, w| {
        ops::conv2d(x, w, 16, 0, 1, 1)
    }),
    ("resnet3x3", &[8, 56, 56, 64], &[64, 3, 3, 64], |x, w| {
        ops::conv2d(x, w, 1, 1, 1, 1)
    }),
    (
        "resnet3x3-c128",
        &[8, 28, 28, 128],
        &[128, 3, 3, 128],
        |x, w| ops::conv2d(x, w, 1, 1, 1, 1),
    ),
    (
        "resnet3x3-c256",
        &[8, 14, 14, 256],
        &[256, 3, 3, 256],
        |x, w| ops::conv2d(x, w, 1, 1, 1, 1),
    ),
    (
        "resnet3x3-c512",
        &[8, 7, 7, 512],
        &[512, 3, 3, 512],
        |x, w| ops::conv2d(x, w, 1, 1, 1, 1),
    ),
    (
        "unet3x3-c256",
        &[1, 64, 64, 256],
        &[256, 3, 3, 256],
        |x, w| ops::conv2d(x, w, 1, 1, 1, 1),
    ),
    ("stem7x7-s2", &[8, 224, 224, 3], &[64, 7, 7, 3], |x, w| {
        ops::conv2d(x, w, 2, 3, 1, 1)
    }),
    ("unaligned3x3", &[8, 56, 56, 24], &[32, 3, 3, 24], |x, w| {
        ops::conv2d(x, w, 1, 1, 1, 1)
    }),
    (
        "resnet3x3-s2",
        &[8, 56, 56, 64],
        &[128, 3, 3, 64],
        |x, w| ops::conv2d(x, w, 2, 1, 1, 1),
    ),
    ("conv1x1", &[8, 56, 56, 256], &[64, 1, 1, 256], |x, w| {
        ops::conv2d(x, w, 1, 0, 1, 1)
    }),
    (
        "depthwise3x3",
        &[8, 56, 56, 128],
        &[128, 3, 3, 1],
        |x, w| ops::conv2d(x, w, 1, 1, 1, 128),
    ),
    ("depthwise7x7", &[8, 56, 56, 96], &[96, 7, 7, 1], |x, w| {
        ops::conv2d(x, w, 1, 3, 1, 96)
    }),
    (
        "transpose4x4-s2",
        &[4, 32, 32, 64],
        &[32, 4, 4, 64],
        |x, w| ops::conv_transpose2d(x, w, 2, 1, 1, 0, 1),
    ),
    ("conv3d", &[1, 8, 32, 32, 32], &[32, 3, 3, 3, 32], |x, w| {
        ops::conv3d(x, w, 1, 1, 1, 1)
    }),
];

/// Convolutions per evaluation.
const REPS: usize = 10;

fn main() -> Result<()> {
    if !std::env::args().any(|a| a == "--bench") {
        return Ok(());
    }
    let filter = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .unwrap_or_default();
    for (name, xs, ws, f) in CASES {
        if !name.contains(filter.as_str()) {
            continue;
        }
        for (dtype, label) in [(DType::F16, "float16"), (DType::F32, "float32")] {
            let x = random::normal(xs, dtype, 0.0, 1.0, Some(&random::key(1)));
            let w = random::normal(ws, dtype, 0.0, 0.1, Some(&random::key(2)));
            nuvec_core::eval(&[x.clone(), w.clone()])?;
            let batch = || (0..REPS).map(|_| f(&x, &w)).collect::<Vec<_>>();
            nuvec_core::eval(&batch())?;
            let mut best = f64::INFINITY;
            for _ in 0..10 {
                let t = Instant::now();
                nuvec_core::eval(&batch())?;
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!("{name}\t{label}\t{:.4}", best * 1e3 / REPS as f64);
        }
    }
    Ok(())
}
