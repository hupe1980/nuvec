//! Accuracy of the functions whose approximations nuvec chooses: `erf` and `expm1`
//! within one unit in the last place of the exact value (mpmath, 40 digits, at the float32
//! inputs), on the CPU and the GPU.

// The inputs are float32 values written with every digit of their float64 value, so the
// references (computed at exactly those values) stay exact.
#![allow(clippy::excessive_precision)]

use nuvec_core::*;

const ERF: [(f32, f64); 15] = [
    (0.0, 0.0),
    (1.0000000031710769e-30, 1.1283791706736896e-30),
    (9.99999993922529e-09, 1.1283791602378209e-08),
    (-9.999999747378752e-05, -0.00011283791348289937),
    (0.10000000149011612, 0.11246291768297051),
    (0.5, 0.5204998778130465),
    (0.9269999861717224, 0.8101341118911192),
    (0.9279999732971191, 0.810611470729394),
    (1.0, 0.8427007929497149),
    (1.5, 0.9661051464753108),
    (2.1700000762939453, 0.997851108978161),
    (-2.1700000762939453, -0.997851108978161),
    (3.0, 0.9999779095030014),
    (3.9000000953674316, 0.9999999652077781),
    (4.0, 0.9999999845827421),
];
const EXPM1: [(f32, f64); 15] = [
    (1.0000000116860974e-07, 1.0000000616861003e-07),
    (-0.0010000000474974513, -0.000999500214074986),
    (0.10000000149011612, 0.10517091972248063),
    (0.30000001192092896, 0.34985882366757415),
    (0.5, 0.6487212707001282),
    (0.699999988079071, 1.0137526834646737),
    (1.0, 1.7182818284590453),
    (2.0, 6.38905609893065),
    (-0.5, -0.3934693402873666),
    (-2.0, -0.8646647167633873),
    (-10.0, -0.9999546000702375),
    (10.0, 22025.465794806718),
    (20.0, 485165194.4097903),
    (50.0, 5.184705528587072e+21),
    (88.0, 1.6516362549940018e+38),
];

fn gpu_available() -> bool {
    let ok = !cfg!(miri)
        && with_stream(Device::gpu(0), || {
            (Array::from(1.0f32) + 1.0).eval().is_ok()
        });
    if !ok && std::env::var("NUVEC_REQUIRE_GPU").is_ok_and(|v| v == "1") {
        panic!("NUVEC_REQUIRE_GPU=1 but no Metal device is available");
    }
    ok
}

/// The distance of `got` from `exact`, in units in the last place of `exact` (float32).
fn ulps(got: f32, exact: f64) -> f64 {
    let e = exact as f32;
    let ulp = f64::from((f32::from_bits(e.abs().to_bits() + 1) - e.abs()).abs())
        .max(f64::from(f32::MIN_POSITIVE));
    (f64::from(got) - exact).abs() / ulp
}

fn check(name: &str, f: fn(&Array) -> Array, table: &[(f32, f64)]) {
    let x: Vec<f32> = table.iter().map(|&(x, _)| x).collect();
    for device in [Device::cpu(), Device::gpu(0)] {
        if device != Device::cpu() && !gpu_available() {
            continue;
        }
        let got = with_stream(device, || f(&Array::from_slice(&x, [x.len() as i32])))
            .to_vec::<f32>()
            .unwrap();
        for (&g, &(x, exact)) in got.iter().zip(table) {
            let u = ulps(g, exact);
            assert!(
                u <= 1.0,
                "{name}({x}) on {device:?}: {g} is {u:.2} ulp from {exact}"
            );
        }
    }
}

#[test]
fn erf_is_faithfully_rounded() {
    check("erf", ops::erf, &ERF);
}

#[test]
fn expm1_is_faithfully_rounded() {
    check("expm1", ops::expm1, &EXPM1);
}
