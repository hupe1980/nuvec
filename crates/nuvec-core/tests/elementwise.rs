//! Element-wise operations against MLX: every float16 and bfloat16 bit pattern through each
//! unary op, 2^20 pairs through each binary op, and 2^20 float32 patterns, compared by hashes
//! of MLX's outputs (`scripts/elementwise_fixtures.py`, `fixtures/elementwise.txt`).

mod common;

use nuvec_core::{Array, DType, Device, fast, ops, with_stream};

const FIXTURE: &str = include_str!("fixtures/elementwise.txt");

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        h = (h ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The output's bits with NaNs canonical, hashed in `chunks` parts as numpy's
/// `array_split` divides them.
fn hashes(out: &Array, dtype: DType, chunks: usize) -> Vec<u64> {
    let is_nan = out.not_equal(out).to_vec::<bool>().expect("NaNs");
    let words: Vec<u32> = if dtype == DType::F32 {
        out.view(DType::U32).to_vec::<u32>().expect("bits")
    } else {
        let bits = out.view(DType::U16).to_vec::<u16>().expect("bits");
        bits.into_iter().map(u32::from).collect()
    };
    let nan = match dtype {
        DType::F16 => 0x7e00,
        DType::BF16 => 0x7fc0,
        _ => 0x7fc0_0000,
    };
    let width = if dtype == DType::F32 { 4 } else { 2 };
    let words: Vec<u32> = words
        .into_iter()
        .zip(is_nan)
        .map(|(w, n)| if n { nan } else { w })
        .collect();
    let (size, extra) = (words.len() / chunks, words.len() % chunks);
    let mut at = 0;
    (0..chunks)
        .map(|c| {
            let n = size + usize::from(c < extra);
            let part = &words[at..at + n];
            at += n;
            fnv(part
                .iter()
                .flat_map(|w| w.to_le_bytes().into_iter().take(width)))
        })
        .collect()
}

/// The unary inputs and the binary pairs as arrays of `dtype` (the generator's).
fn inputs(dtype: DType) -> (Array, Array, Array) {
    let n = 1u64 << 20;
    if dtype == DType::F32 {
        let x: Vec<u32> = (0..n)
            .map(|i| ((i * 2_654_435_761) & 0xffff_ffff) as u32)
            .collect();
        let y: Vec<u32> = x
            .iter()
            .map(|&x| ((u64::from(x) * 40503 + 12345) & 0xffff_ffff) as u32)
            .collect();
        let a = Array::from_slice(&x, [x.len() as i32]).view(dtype);
        let b = Array::from_slice(&y, [y.len() as i32]).view(dtype);
        return (a.clone(), a, b);
    }
    let all: Vec<u16> = (0..=u16::MAX).collect();
    let (x, y): (Vec<u16>, Vec<u16>) = (0..n)
        .map(|i| {
            (
                (i & 0xffff) as u16,
                (((i * 40503 + 12345) >> 4) & 0xffff) as u16,
            )
        })
        .unzip();
    (
        Array::from_slice(&all, [all.len() as i32]).view(dtype),
        Array::from_slice(&x, [x.len() as i32]).view(dtype),
        Array::from_slice(&y, [y.len() as i32]).view(dtype),
    )
}

/// The generator's `values(n, start)`: `((i · 2654435761 mod 2^32) >> 8) · 2^-21 − 4`.
fn values(n: usize, start: u64) -> Array {
    let v: Vec<f32> = (start..start + n as u64)
        .map(|i| (((i * 2_654_435_761) & 0xffff_ffff) >> 8) as f32 * 2f32.powi(-21) - 4.0)
        .collect();
    Array::from_slice(&v, [n as i32])
}

/// The generator's fused and reduction cases.
fn fused(op: &str, dt: DType) -> Array {
    let x = values(32 * 4096, 0).reshape([32, 4096]).astype(dt);
    let w = values(4096, 7).multiply(0.25).add(1.0).astype(dt);
    let b = values(4096, 13).multiply(0.1).astype(dt);
    match op {
        "rms_norm" => fast::rms_norm(&x, &w, 1e-6),
        "layer_norm" => fast::layer_norm(&x, &w, &b, 1e-5),
        "rope" => fast::rope(
            &x.reshape([1, 16, 64, 128]),
            64,
            false,
            Some(10000.0),
            1.0,
            &Array::from(7i32),
            None,
        ),
        "rope_traditional" => fast::rope(
            &x.reshape([1, 16, 64, 128]),
            128,
            true,
            Some(1e6),
            1.0,
            &Array::from(300i32),
            None,
        ),
        "softmax" => x.softmax(-1),
        "logsumexp" => x.logsumexp(-1),
        "sum" => x.sum(-1),
        "mean" => x.mean(-1),
        "max" => x.max(-1),
        "cumsum" => x.cumsum(-1),
        "logcumsumexp" => x.logcumsumexp(-1),
        "hadamard" => ops::hadamard_transform(&x.reshape([128, 1024]), None),
        "sum_columns" => x.reshape([4096, 32]).sum(0),
        "sum_long_columns" => values(1100 * 32, 0).reshape([1100, 32]).astype(dt).sum(0),
        "conv_depthwise" => x.reshape([1, 256, 512]).conv1d(
            &values(512 * 4, 3)
                .multiply(0.5)
                .reshape([512, 4, 1])
                .astype(dt),
            1,
            0,
            1,
            512,
        ),
        _ => panic!("unknown fused op {op}"),
    }
}

fn unary(op: &str, x: &Array) -> Array {
    match op {
        "abs" => x.abs(),
        "arccos" => x.arccos(),
        "arccosh" => x.arccosh(),
        "arcsin" => x.arcsin(),
        "arcsinh" => x.arcsinh(),
        "arctan" => x.arctan(),
        "arctanh" => x.arctanh(),
        "ceil" => x.ceil(),
        "cos" => x.cos(),
        "cosh" => x.cosh(),
        "erf" => x.erf(),
        "erfinv" => x.erfinv(),
        "exp" => x.exp(),
        "expm1" => x.expm1(),
        "floor" => x.floor(),
        "log" => x.log(),
        "log2" => x.log2(),
        "log10" => x.log10(),
        "log1p" => x.log1p(),
        "negative" => x.negative(),
        "round" => x.round(0),
        "sigmoid" => x.sigmoid(),
        "sign" => x.sign(),
        "sin" => x.sin(),
        "sinh" => x.sinh(),
        "square" => x.square(),
        "sqrt" => x.sqrt(),
        "rsqrt" => x.rsqrt(),
        "tan" => x.tan(),
        "tanh" => x.tanh(),
        _ => panic!("unknown unary op {op}"),
    }
}

fn binary(op: &str, x: &Array, y: &Array) -> Array {
    match op {
        "add" => x.add(y),
        "subtract" => x.subtract(y),
        "multiply" => x.multiply(y),
        "divide" => x.divide(y),
        "floor_divide" => x.floor_divide(y),
        "remainder" => x.remainder(y),
        "logaddexp" => x.logaddexp(y),
        "maximum" => x.maximum(y),
        "minimum" => x.minimum(y),
        "power" => x.power(y),
        "arctan2" => x.arctan2(y),
        _ => panic!("unknown binary op {op}"),
    }
}

/// The cases of `device` (`"gpu"` or `"cpu"`) that differ from MLX's.
fn differing(device: &str) -> (usize, Vec<String>) {
    let mut lines = FIXTURE.lines();
    let meta = lines.next().expect("meta");
    assert!(meta.contains("mlx=0.32.4"), "{meta}");
    let chunks: usize = meta
        .split("chunks=")
        .nth(1)
        .and_then(|c| c.trim().parse().ok())
        .expect("chunks");
    let on = if device == "gpu" {
        Device::gpu(0)
    } else {
        Device::cpu()
    };
    let (mut checked, mut failures) = (0, vec![]);
    for line in lines {
        let mut parts = line.split_whitespace();
        let (dev, dtype, op) = (
            parts.next().expect("device"),
            parts.next().expect("dtype"),
            parts.next().expect("op"),
        );
        if dev != device {
            continue;
        }
        let want: Vec<u64> = parts.map(|h| h.parse().expect("hash")).collect();
        let dtype = match dtype {
            "float16" => DType::F16,
            "bfloat16" => DType::BF16,
            _ => DType::F32,
        };
        let got = with_stream(on, || {
            let (u, x, y) = inputs(dtype);
            let out = if let Some(f) = op.strip_prefix("fast:") {
                fused(f, dtype)
            } else if op_is_binary(op) {
                binary(op, &x, &y)
            } else {
                unary(op, &u)
            };
            hashes(&out, dtype, chunks)
        });
        checked += 1;
        let bad: Vec<usize> = (0..chunks).filter(|&c| got[c] != want[c]).collect();
        if !bad.is_empty() {
            failures.push(format!("{device} {dtype:?} {op}: chunks {bad:?} differ"));
        }
    }
    (checked, failures)
}

#[test]
#[cfg_attr(miri, ignore = "millions of elements")]
fn gpu_elementwise_ops_equal_mlx_exhaustively() {
    if common::require_gpu().is_none() {
        return;
    }
    let (checked, failures) = differing("gpu");
    assert!(
        failures.is_empty(),
        "{} of {checked} GPU cases differ from MLX:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// MLX's CPU computes half precision with Accelerate's vector math; nuvec's CPU meets
/// MLX's tolerance there, not its bits. Lists the cases that differ.
#[test]
#[ignore = "reports the CPU cases that differ from MLX's bits"]
fn cpu_elementwise_ops_against_mlx() {
    let (checked, failures) = differing("cpu");
    println!(
        "{} of {checked} CPU cases differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn op_is_binary(op: &str) -> bool {
    matches!(
        op,
        "add"
            | "subtract"
            | "multiply"
            | "divide"
            | "floor_divide"
            | "remainder"
            | "logaddexp"
            | "maximum"
            | "minimum"
            | "power"
            | "arctan2"
    )
}
