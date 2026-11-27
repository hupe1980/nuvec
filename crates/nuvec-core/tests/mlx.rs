//! Replays outputs recorded from real MLX (`scripts/mlx_fixtures.py`) without Python.

mod common;
#[path = "common/fixture.rs"]
mod fixture;

use fixture::{Case, check, check_exact, compare, from_array, load, load_gpu, run, values};
use nuvec_core::{Device, with_stream};

#[test]
fn fixture_meta() {
    let (meta, cases) = load();
    assert_eq!(meta, "mlx=0.32.5 device=cpu seed=20261016");
    assert!(cases.len() >= 150, "only {} cases", cases.len());
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of cases: too slow under Miri, and covered natively"
)]
fn mlx_fixtures() {
    let (_, cases) = load();
    let mut devices = vec![Device::cpu()];
    if common::require_gpu().is_some() {
        devices.push(Device::gpu(0));
    }
    for device in devices {
        let failures: Vec<_> = with_stream(device, || cases.iter().filter_map(check).collect());
        assert!(
            failures.is_empty(),
            "{device}: {} of {} MLX fixture cases failed:\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
    }
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
    assert_eq!(meta, "mlx=0.32.5 device=gpu seed=20261016");
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
        "quantized_matmul" | "gather_qmm" | "qqmm" | "gather_qqmm" | "block_masked_mm"
    ) || case.name.starts_with("affine-dequantize-")
        || case.name.starts_with("vjp-qqmm-")
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

/// Values recorded by the release probes with MLX 0.32.5.
#[test]
fn release_quantized_accumulation() {
    use nuvec_core::{DType, Quant, QuantMode, ops};
    with_stream(Device::cpu(), || {
        let x = ops::ones([1, 4096], DType::BF16);
        for quant in [Quant::affine(32, 4), Quant::new(QuantMode::Mxfp4)] {
            for transpose in [false, true] {
                let shape = if transpose { [64, 4096] } else { [4096, 64] };
                let w = ops::ones(shape, DType::BF16).quantize(quant);
                let y = x.quantized_matmul(&w, transpose).astype(DType::F32);
                assert_eq!(y.to_vec::<f32>().unwrap(), [4096.; 64]);
            }
        }
    });
}

#[test]
fn release_pad_fill_derivatives() {
    use nuvec_core::{Array, array, jvp, ops, vjp};
    for device in [Device::cpu(), Device::gpu(0)] {
        if device == Device::gpu(0) && common::require_gpu().is_none() {
            continue;
        }
        with_stream(device, || {
            let pad =
                |(x, fill): &(Array, Array)| ops::pad(x, &[(1, 2)], ops::PadMode::Constant, fill);
            let args = (array![1f32, 2.], Array::from(3f32));
            let (_, tangent) = jvp(pad, &args, &(array![5f32, 7.], Array::from(11f32))).unwrap();
            assert_eq!(tangent.to_vec::<f32>().unwrap(), [11., 5., 7., 11., 11.]);
            let (_, (x, fill)) = vjp(pad, &args, &array![1f32, 2., 3., 4., 5.]).unwrap();
            assert_eq!(x.to_vec::<f32>().unwrap(), [2., 3.]);
            assert_eq!(fill.item::<f32>().unwrap(), 10.);
        });
    }
}

#[test]
fn release_gated_delta_preserves_float32_gate() {
    use nuvec_core::{DType, fast, ops};
    if common::require_gpu().is_none() {
        return;
    }
    with_stream(Device::gpu(0), || {
        let q = ops::ones([1, 1, 1, 32], DType::BF16) * 0.03125;
        let zero = ops::zeros([1, 1, 1, 32], DType::BF16);
        let gate = ops::full([1, 1, 1], 0.998_765_4_f32);
        let beta = ops::ones([1, 1, 1], DType::F32);
        let state = ops::ones([1, 1, 32, 32], DType::F32);
        let (_, next) = fast::gated_delta(&q, &zero, &zero, &gate, &beta, &state);
        assert_eq!(next.to_vec::<f32>().unwrap(), [0.998_765_4; 1024]);
        // The f32 key is 1/8. With an initial state and gate of one,
        // S k = 4 and each updated cell is exactly 1 - 4/8 on both devices.
        let key = ops::full([1, 1, 1, 32], 0.125_f32);
        let (_, next) = fast::gated_delta(&q, &key, &zero, &beta, &beta, &state);
        assert!(
            next.to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|&value| value == 0.5)
        );
    });
}
