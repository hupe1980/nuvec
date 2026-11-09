//! `NUVEC_CONV_WINOGRAD` (read once per process, so in its own test binary): `fast` takes
//! MLX's half-precision tiles and `0` no Winograd, against the default's float32 tiles. One
//! test sets the variable for the process before any convolution runs.

mod common;

use nuvec_core::{DType, Device, ops, random, with_stream};

/// The error of a float16 Winograd-class convolution relative to its scale, and whether it
/// equals the float32-tile default's.
#[test]
fn fast_mode_takes_mlx_tiles() {
    if common::require_gpu().is_none() {
        return;
    }
    // SAFETY: set before any thread reads the environment (the first convolution below).
    unsafe { std::env::set_var("NUVEC_CONV_WINOGRAD", "fast") };
    let x = random::uniform(
        -2.5,
        2.5,
        [1, 64, 64, 128],
        DType::F32,
        Some(&random::key(1)),
    );
    let w = random::uniform(
        -2.5,
        2.5,
        [128, 3, 3, 128],
        DType::F32,
        Some(&random::key(2)),
    );
    let reference = with_stream(Device::cpu(), || ops::conv2d(&x, &w, 1, 1, 1, 1));
    let fast = with_stream(Device::gpu(0), || {
        ops::conv2d(&x.astype(DType::F16), &w.astype(DType::F16), 1, 1, 1, 1)
    })
    .astype(DType::F32);
    let scale = reference.abs().max(..).item::<f32>().unwrap();
    let err = fast
        .subtract(&reference)
        .abs()
        .max(..)
        .item::<f32>()
        .unwrap()
        / scale;
    // MLX's half-precision tiles: about 1e-2 of the scale (the default: 4e-4).
    assert!(err > 2e-3 && err < 5e-2, "fast Winograd error {err}");
}
