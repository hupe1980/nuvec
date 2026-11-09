//! Replays outputs recorded from real MLX (`scripts/mlx_fixtures.py`) without Python.

mod common;
#[path = "common/fixture.rs"]
mod fixture;

use fixture::{Case, check, check_exact, compare, from_array, load, load_gpu, run, values};
use nuvec_core::{Device, with_stream};

#[test]
fn fixture_meta() {
    let (meta, cases) = load();
    assert!(
        meta.contains("mlx=0.32.4") && meta.contains("device=cpu"),
        "{meta}"
    );
    assert!(cases.len() >= 150, "only {} cases", cases.len());
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of cases: too slow under Miri, and covered natively"
)]
fn mlx_fixtures() {
    let (_, cases) = load();
    let failures: Vec<String> = cases.iter().filter_map(check).collect();
    assert!(
        failures.is_empty(),
        "{} of {} MLX fixture cases failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn comparator_detects_corruption() {
    let (_, cases) = load();
    let case = cases
        .iter()
        .find(|c| c.name == "add-float32")
        .expect("case exists");
    let mut corrupted = case.outputs[0].clone();
    // Elements 0-3 are NaN/inf/-0.0 edge values; element 4 is an ordinary number.
    corrupted.data[4 * 4 + 1] ^= 0x40; // flip a mantissa bit of element 4
    let got = from_array(&run(case).unwrap()[0]).unwrap();
    let err = compare(case, &got, &corrupted).unwrap_err();
    assert!(
        err.contains("add-float32") && err.contains("element 4"),
        "{err}"
    );
}

/// The floating-point quantization modes against MLX's Metal output, bit for bit, on the
/// CPU and (when present) the GPU.
#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of cases: too slow under Miri, and covered natively"
)]
fn mlx_gpu_fixtures_bit_for_bit() {
    let (meta, cases) = load_gpu();
    assert!(
        meta.contains("mlx=0.32.4") && meta.contains("device=gpu"),
        "{meta}"
    );
    let mut devices = vec![];
    if common::require_gpu().is_some() {
        devices.push(Device::gpu(0));
    }
    devices.push(Device::cpu());
    // The GPU the fixtures were recorded on: other GPU generations' `pow` and `exp` may
    // round differently (by one unit in the last place, in MLX as in nuvec).
    let architecture = nuvec_gpu::Device::shared().map(|d| d.info().architecture.clone());
    let recorded_gpu = architecture.as_deref() == Some(RECORDED_ON);
    for device in devices {
        // Products, affine dequantize, the blocked head-512 attention and `pow` are
        // bit-exact on the GPU (MLX's kernels and dispatch); the CPU follows MLX's CPU
        // (another accumulation order, two roundings per dequantized element, the C
        // library's `pow`) and meets FR-008's bound instead, as do `pow` and attention on
        // another GPU generation.
        let cpu = device == Device::cpu();
        let failures: Vec<String> = cases
            .iter()
            .filter_map(|c| {
                with_stream(device, || {
                    if (cpu && cpu_differs(c)) || (!cpu && !recorded_gpu && transcendental(c)) {
                        check_within(c)
                    } else {
                        check_exact(c)
                    }
                })
            })
            .collect();
        assert!(
            failures.is_empty(),
            "{device:?} ({architecture:?}): {} of {} MLX GPU fixture cases differ:\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
    }
}

/// The GPU architecture `mlx_gpu.bin` was recorded on (an M4 Pro).
const RECORDED_ON: &str = "applegpu_g16s";

/// Cases whose GPU results depend on the GPU's `pow` or `exp`.
fn transcendental(case: &Case) -> bool {
    case.name.starts_with("sdpa-d512-") || case.name.starts_with("power-gpu-")
}

fn cpu_differs(case: &Case) -> bool {
    matches!(
        case.op.as_str(),
        "quantized_matmul" | "gather_qmm" | "qqmm" | "gather_qqmm"
    ) || case.name.starts_with("affine-dequantize-")
        || case.name.starts_with("sdpa-d512-")
        || case.name.starts_with("power-gpu-")
        || case.name.starts_with("hadamard-gpu-")
}

/// FR-008's bound against MLX's GPU output, `|a - b| <= atol · max|b| + rtol · |b|`: the
/// absolute part scales with the output (sums of `K` products cancel to values near zero).
fn check_within(case: &Case) -> Option<String> {
    let got = match run(case).and_then(|a| {
        a.iter()
            .map(from_array)
            .collect::<nuvec_core::Result<Vec<_>>>()
    }) {
        Ok(got) => got,
        Err(e) => return Some(format!("{}: nuvec error: {e}", case.name)),
    };
    let (atol, rtol) = match case.outputs[0].dtype {
        nuvec_core::DType::F32 => (1e-5, 1e-4),
        nuvec_core::DType::F16 => (1e-3, 5e-3),
        _ => (1e-2, 4e-2),
    };
    let (g, e) = (values(&got[0]), values(&case.outputs[0]));
    let scale = e.iter().fold(1.0f64, |m, v| m.max(v.0.abs()));
    g.iter().zip(&e).enumerate().find_map(|(i, (a, b))| {
        ((a.0 - b.0).abs() > atol * scale + rtol * b.0.abs())
            .then(|| format!("{}: element {i}: {} vs MLX {}", case.name, a.0, b.0))
    })
}
