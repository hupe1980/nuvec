//! `graph::from_primitive` is public (`compile` replays graphs through it), so every
//! primitive validates its parameters and inputs when it is recorded: an invalid primitive
//! returns `Err`, and a valid one evaluates without panicking, on the CPU and the GPU.

mod common;

use nuvec_core::graph::{
    ArgReduceOp, BinaryOp, FastOp, Primitive, ReduceOp, ScanOp, ScatterReduce, SliceArg, UnaryOp,
};
use nuvec_core::*;

fn f32s(shape: &[usize]) -> Array {
    let n: usize = shape.iter().product();
    Array::from_slice(&(0..n).map(|i| i as f32).collect::<Vec<_>>(), shape)
}

fn rejected(primitive: Primitive, inputs: Vec<Array>) {
    let name = format!("{primitive:?}");
    assert!(
        graph::from_primitive(primitive, inputs).is_err(),
        "{name} must be rejected"
    );
}

#[test]
fn reshape_must_keep_the_element_count() {
    rejected(Primitive::Reshape(vec![100]), vec![f32s(&[6])]);
    rejected(Primitive::Reshape(vec![usize::MAX, 2]), vec![f32s(&[6])]);
}

#[test]
fn unit_axes_must_be_sorted_in_range_and_of_size_one() {
    rejected(Primitive::Squeeze(vec![0]), vec![f32s(&[2, 1])]);
    rejected(Primitive::Squeeze(vec![2]), vec![f32s(&[2, 1])]);
    rejected(Primitive::Squeeze(vec![1, 1]), vec![f32s(&[2, 1])]);
    rejected(Primitive::ExpandDims(vec![3]), vec![f32s(&[2])]);
    rejected(Primitive::ExpandDims(vec![1, 0]), vec![f32s(&[2])]);
}

#[test]
fn slices_must_stay_in_bounds() {
    let slice = |start: usize, step: isize, len: usize| Primitive::Slice {
        start: vec![start],
        step: vec![step],
        len: vec![len],
    };
    rejected(slice(5, 1, 4), vec![f32s(&[6])]);
    rejected(slice(0, 0, 3), vec![f32s(&[6])]);
    rejected(slice(6, 1, 1), vec![f32s(&[6])]);
    rejected(slice(0, -1, 2), vec![f32s(&[6])]);
    rejected(slice(1, isize::MAX, 3), vec![f32s(&[6])]);
    rejected(slice(0, 1, 2), vec![f32s(&[2, 3])]);
    // Valid: reversed, and empty.
    let out = graph::from_primitive(slice(5, -2, 3), vec![f32s(&[6])]).unwrap();
    assert_eq!(out[0].to_vec::<f32>().unwrap(), [5.0, 3.0, 1.0]);
    let out = graph::from_primitive(slice(0, 1, 0), vec![f32s(&[6])]).unwrap();
    assert_eq!(out[0].size(), 0);
}

#[test]
fn slice_updates_must_stay_in_bounds() {
    let update = |start: usize, step: isize, len: usize| Primitive::SliceUpdate {
        start: vec![start],
        step: vec![step],
        len: vec![len],
    };
    rejected(update(0, isize::MAX, 3), vec![f32s(&[6]), f32s(&[3])]);
    rejected(update(0, isize::MIN, 2), vec![f32s(&[6]), f32s(&[2])]);
    rejected(update(0, 0, 2), vec![f32s(&[6]), f32s(&[2])]);
    rejected(update(5, 1, 2), vec![f32s(&[6]), f32s(&[2])]);
}

#[test]
fn concatenation_extents_must_not_overflow() {
    // A dimension may not exceed `i32::MAX` (MLX's `Shape`), so the largest extents that
    // are fine alone overflow it when joined.
    let half = i32::MAX as usize / 2 + 1;
    assert!(
        Array::from_slice::<f32>(&[], [0, usize::MAX / 2 + 1])
            .check()
            .is_err(),
        "an oversized dimension is itself rejected"
    );
    let huge = Array::from_slice::<f32>(&[], [0, half]).check().unwrap();
    rejected(
        Primitive::Concatenate { axis: 1 },
        vec![huge.clone(), huge.clone(), huge],
    );
    let a = Array::from_slice::<f32>(&[], [half, 0]).check().unwrap();
    rejected(
        Primitive::Concatenate { axis: 0 },
        vec![a.clone(), a.clone(), a],
    );
}

#[test]
fn quantization_parameters_are_validated() {
    let w = f32s(&[4, 64]);
    for quant in [
        Quant::affine(0, 4),
        Quant::affine(64, 7),
        Quant::affine(64, 0),
        Quant::new(QuantMode::Mxfp4).group_size(64),
        Quant::new(QuantMode::Nvfp4).bits(8),
    ] {
        rejected(Primitive::Quantize { quant }, vec![w.clone()]);
    }
    let q = w.quantize(Quant::affine(64, 4));
    let (packed, scales, biases) = (q.weight, q.scales, q.biases.unwrap());
    let inputs = vec![packed.clone(), scales.clone(), biases.clone()];
    for quant in [Quant::affine(64, 7), Quant::affine(0, 4)] {
        let dtype = DType::F32;
        rejected(Primitive::Dequantize { quant, dtype }, inputs.clone());
    }
    // Floating-point modes take uint8 scales and no biases.
    rejected(
        Primitive::Dequantize {
            quant: Quant::new(QuantMode::Mxfp4),
            dtype: DType::F32,
        },
        inputs.clone(),
    );
    rejected(
        Primitive::Dequantize {
            quant: Quant::new(QuantMode::Mxfp4),
            dtype: DType::F32,
        },
        vec![packed.clone(), scales.clone()],
    );
    rejected(
        Primitive::QuantizedMatmul {
            quant: Quant::affine(64, 7),
            transpose: true,
        },
        vec![
            f32s(&[1, 64]),
            packed.clone(),
            scales.clone(),
            biases.clone(),
        ],
    );
    let indices = Array::from([0u32]);
    rejected(
        Primitive::Fast(FastOp::GatherQmm {
            quant: Quant::affine(64, 7),
            transpose: true,
            global_scale: false,
        }),
        vec![
            f32s(&[1, 1, 64]),
            packed.reshape([1, 4, 8]),
            scales.reshape([1, 4, 1]),
            biases.reshape([1, 4, 1]),
            indices.clone(),
            indices,
        ],
    );
}

#[test]
fn single_index_slices_do_not_overflow() {
    let arg = SliceArg::from(isize::MAX);
    let x = f32s(&[6]);
    let out = graph::slice(&x, &[arg]).unwrap();
    assert_eq!(out.size(), 0);
}

// ── Random primitives ────────────────────────────────────────────────────────

/// SplitMix64: a deterministic generator, so a failing case is reproduced by its seed.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn coin(&mut self) -> bool {
        self.next() & 1 == 1
    }

    /// Mostly small values, sometimes an extreme one.
    fn size(&mut self) -> usize {
        match self.below(16) {
            0 => usize::MAX - self.below(3),
            1 => 1 << (16 + self.below(48)),
            _ => self.below(7),
        }
    }

    fn isize(&mut self) -> isize {
        match self.below(16) {
            0 => isize::MAX - self.below(3) as isize,
            1 => isize::MIN + self.below(3) as isize,
            _ => self.below(9) as isize - 4,
        }
    }

    fn sizes(&mut self, n: usize) -> Vec<usize> {
        (0..n).map(|_| self.size()).collect()
    }

    fn shape(&mut self) -> Vec<usize> {
        (0..self.below(6)).map(|_| self.below(5)).collect()
    }

    fn dtype(&mut self) -> DType {
        const ALL: [DType; 14] = [
            DType::Bool,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
            DType::Complex64,
        ];
        // Mostly the dtypes the kernels and indices use.
        match self.below(4) {
            0 => ALL[self.below(ALL.len())],
            1 => DType::U32,
            2 => DType::I32,
            _ => DType::F32,
        }
    }

    fn bits(&mut self) -> u32 {
        self.next() as u32
    }
}

/// A random array: random small values (also negative, as hostile indices), cast to
/// `dtype`, sometimes in a transposed layout.
fn array(rng: &mut SplitMix64, dtype: DType, shape: &[usize]) -> Array {
    let n: usize = shape.iter().product();
    let values: Vec<f32> = (0..n).map(|_| rng.below(13) as f32 - 4.0).collect();
    let a = Array::from_slice(&values, shape).astype(dtype);
    if shape.len() >= 2 && rng.below(4) == 0 {
        let reversed: Vec<usize> = shape.iter().rev().copied().collect();
        let t = a.reshape(reversed).t();
        // The same shape again (a non-contiguous layout of other values).
        if t.shape()
            .iter()
            .map(|&d| d as usize)
            .eq(shape.iter().copied())
        {
            return t;
        }
    }
    a
}

/// A random primitive and inputs: an input count around the primitive's arity, shapes
/// mostly shared (to reach past the first checks) and random parameters.
fn case(rng: &mut SplitMix64) -> (Primitive, Vec<Array>) {
    let float = |rng: &mut SplitMix64| match rng.below(4) {
        0 => rng.bits(),
        _ => (rng.below(4) as f32 * 0.5).to_bits(),
    };
    let unary = [
        UnaryOp::Negative,
        UnaryOp::Abs,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tanh,
        UnaryOp::Sigmoid,
        UnaryOp::Conjugate,
        UnaryOp::ErfInv,
    ];
    let binary = [
        BinaryOp::Add,
        BinaryOp::Subtract,
        BinaryOp::Multiply,
        BinaryOp::Divide,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
        BinaryOp::LogAddExp,
        BinaryOp::FloorDivide,
        BinaryOp::Power,
        BinaryOp::Equal,
        BinaryOp::Less,
        BinaryOp::GreaterEqual,
    ];
    let scan = [
        ScanOp::Sum,
        ScanOp::Prod,
        ScanOp::Max,
        ScanOp::Min,
        ScanOp::LogAddExp,
    ];
    let reduce = [
        ReduceOp::Sum,
        ReduceOp::Prod,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::And,
        ReduceOp::Or,
    ];
    let ndim = rng.below(5);
    let (primitive, arity) = match rng.below(46) {
        0 => (Primitive::Unary(unary[rng.below(unary.len())]), 1),
        1 => (Primitive::Binary(binary[rng.below(binary.len())]), 2),
        2 => (Primitive::Select, 3),
        3 => (Primitive::AsType(rng.dtype()), 1),
        4 => (Primitive::Reshape(rng.sizes(ndim)), 1),
        5 => (Primitive::Transpose(rng.sizes(ndim)), 1),
        6 | 7 => {
            let p = Primitive::Slice {
                start: rng.sizes(ndim),
                step: (0..ndim).map(|_| rng.isize()).collect(),
                len: rng.sizes(ndim),
            };
            (p, 1)
        }
        8 => (Primitive::Broadcast(rng.sizes(ndim)), 1),
        9 => (Primitive::StopGradient, 1),
        10 | 11 => {
            let p = Primitive::SliceUpdate {
                start: rng.sizes(ndim),
                step: (0..ndim).map(|_| rng.isize()).collect(),
                len: rng.sizes(ndim),
            };
            (p, 2)
        }
        12 => {
            let n = rng.below(4);
            let p = Primitive::Reduce {
                op: reduce[rng.below(reduce.len())],
                axes: rng.sizes(n),
            };
            (p, 1)
        }
        13 => {
            let op = [ArgReduceOp::Max, ArgReduceOp::Min][rng.below(2)];
            (
                Primitive::ArgReduce {
                    op,
                    axis: rng.size(),
                },
                1,
            )
        }
        14 => (Primitive::GatherAxis { axis: rng.size() }, 2),
        15 => {
            let reduce = [ScatterReduce::None, ScatterReduce::Sum][rng.below(2)];
            let p = Primitive::ScatterAxis {
                axis: rng.size(),
                reduce,
            };
            (p, 3)
        }
        16 => (Primitive::Sort { axis: rng.size() }, 1),
        17 => (Primitive::ArgSort { axis: rng.size() }, 1),
        18 => {
            let p = Primitive::Scan {
                op: scan[rng.below(scan.len())],
                axis: rng.size(),
                reverse: rng.coin(),
                inclusive: rng.coin(),
            };
            (p, 1)
        }
        19 => (
            Primitive::Concatenate { axis: rng.size() },
            1 + rng.below(3),
        ),
        20 => (Primitive::Matmul, 2),
        21 => {
            let p = Primitive::AddMM {
                alpha: float(rng),
                beta: float(rng),
            };
            (p, 3)
        }
        22..=24 => {
            let modes = [
                QuantMode::Affine,
                QuantMode::Mxfp4,
                QuantMode::Mxfp8,
                QuantMode::Nvfp4,
            ];
            let mode = modes[rng.below(4)];
            let quant = Quant {
                mode,
                group_size: [0, 1, 16, 32, 64, 128][rng.below(6)],
                bits: rng.below(10) as i32,
            };
            let biases = usize::from(mode == QuantMode::Affine);
            match rng.below(3) {
                0 => (Primitive::Quantize { quant }, 1),
                1 => {
                    let dtype = [DType::F32, DType::BF16, DType::U8][rng.below(3)];
                    (Primitive::Dequantize { quant, dtype }, 2 + biases)
                }
                _ => {
                    let transpose = rng.coin();
                    (Primitive::QuantizedMatmul { quant, transpose }, 3 + biases)
                }
            }
        }
        25 => {
            let p = Primitive::RandomBits {
                shape: (0..rng.below(3)).map(|_| rng.below(5)).collect(),
                width: rng.below(6) as u8,
            };
            (p, 1)
        }
        26 => (Primitive::Copy, 1),
        27 => {
            let n_deps = rng.below(3);
            (Primitive::Depends { n_deps }, 1 + n_deps)
        }
        28 => {
            let n = rng.below(3);
            (Primitive::Squeeze(rng.sizes(n)), 1)
        }
        29 => {
            let n = rng.below(3);
            (Primitive::ExpandDims(rng.sizes(n)), 1)
        }
        _ => {
            let (op, arity) = fast_op(rng, float);
            (Primitive::Fast(op), arity)
        }
    };
    // Inputs: usually the arity, sometimes one more or fewer.
    let count = match rng.below(10) {
        0 => arity.saturating_sub(1),
        1 => arity + 1,
        _ => arity,
    };
    let shared = rng.shape();
    let dtype = rng.dtype();
    let inputs = (0..count)
        .map(|_| {
            let shape = if rng.below(3) == 0 {
                rng.shape()
            } else {
                shared.clone()
            };
            let dtype = if rng.below(3) == 0 {
                rng.dtype()
            } else {
                dtype
            };
            array(rng, dtype, &shape)
        })
        .collect();
    (primitive, inputs)
}

/// A random fast operation and its usual input count.
fn fast_op(rng: &mut SplitMix64, float: impl Fn(&mut SplitMix64) -> u32) -> (FastOp, usize) {
    let eps = float(rng);
    match rng.below(23) {
        0 => (FastOp::RmsNorm { eps }, 2),
        1 => (FastOp::AddRmsNorm { eps }, 3),
        2 => (FastOp::RmsNormVjp { eps }, 3),
        3 => (FastOp::LayerNorm { eps }, 3),
        4 => (FastOp::LayerNormVjp { eps }, 4),
        5 => {
            let op = FastOp::Rope {
                dims: rng.size(),
                traditional: rng.coin(),
                base: float(rng),
                scale: float(rng),
                forward: rng.coin(),
            };
            (op, 2 + rng.below(2))
        }
        6 => {
            // `paged` indexes a pool with its slots' values (see `PagedWrite` below).
            let (cache, positions) = (rng.coin(), rng.coin());
            let op = FastOp::NormRope {
                eps: rng.coin().then_some(eps),
                dims: rng.size(),
                traditional: rng.coin(),
                base: float(rng),
                scale: float(rng),
                offset: rng.size(),
                cache,
                positions,
                paged: false,
            };
            (
                op,
                2 + usize::from(cache) + usize::from(positions) + rng.below(2),
            )
        }
        7 => {
            let op = FastOp::DepthwiseConv1d {
                stride: rng.size(),
                padding: rng.size(),
                dilation: rng.size(),
            };
            (op, 2)
        }
        // `PagedAttention`, `PagedWrite` and `ConvSlots` index caches with their inputs'
        // values; their bounds rules (skip and clamp) are tested on hostile values in
        // `tests/fast.rs`.
        8 => (FastOp::Softmax, 1),
        9 => (FastOp::LogSumExp, 1),
        10 => (FastOp::CrossEntropy, 2),
        11 => (FastOp::CrossEntropyVjp, 4),
        12 | 13 => {
            let (mask, sinks) = (rng.coin(), rng.coin());
            let op = FastOp::Attention {
                scale: float(rng),
                causal: rng.coin(),
                mask,
                sinks,
                window: [0, 0, 1, 7, 128][rng.below(5)],
                passes: rng.below(4) as u8,
                blocks: [0, 1, 32, 128, 2048][rng.below(5)],
                gqa_kernel: rng.coin(),
                tk1: rng.coin(),
                lse: rng.coin(),
            };
            (op, 3 + usize::from(mask) + usize::from(sinks))
        }
        14 => (FastOp::GatedDelta { steps: rng.coin() }, 6),
        15 => (FastOp::GatedDeltaSlots, 7),
        16 => (FastOp::GatherMm, 4 + rng.below(2)),
        17 => {
            let op = FastOp::MoeRoute {
                k: rng.size(),
                normalize: rng.coin(),
            };
            (op, 1)
        }
        18 => (FastOp::WeightedSum, 2),
        19 => {
            let (scaled, next) = (rng.coin(), rng.coin());
            let op = FastOp::ResidualRmsNorm {
                eps,
                scaled,
                next: next.then(|| float(rng)),
            };
            (op, 3 + usize::from(scaled) + usize::from(next))
        }
        _ => {
            let modes = [QuantMode::Affine, QuantMode::Mxfp4, QuantMode::Nvfp4];
            let mode = modes[rng.below(3)];
            let global_scale = mode == QuantMode::Nvfp4 && rng.coin();
            let op = FastOp::GatherQmm {
                quant: Quant {
                    mode,
                    group_size: [0, 16, 32, 64, 128][rng.below(5)],
                    bits: rng.below(10) as i32,
                },
                transpose: rng.coin(),
                global_scale,
            };
            let arity = 5 + usize::from(mode == QuantMode::Affine) + usize::from(global_scale);
            (op, arity + rng.below(2))
        }
    }
}

/// Reads `a` back in its own element type.
fn read(a: &Array) -> Result<()> {
    match a.dtype() {
        DType::Bool => a.to_vec::<bool>().map(drop),
        DType::U8 => a.to_vec::<u8>().map(drop),
        DType::U16 => a.to_vec::<u16>().map(drop),
        DType::U32 => a.to_vec::<u32>().map(drop),
        DType::U64 => a.to_vec::<u64>().map(drop),
        DType::I8 => a.to_vec::<i8>().map(drop),
        DType::I16 => a.to_vec::<i16>().map(drop),
        DType::I32 => a.to_vec::<i32>().map(drop),
        DType::I64 => a.to_vec::<i64>().map(drop),
        DType::F16 => a.to_vec::<f16>().map(drop),
        DType::BF16 => a.to_vec::<bf16>().map(drop),
        DType::F32 => a.to_vec::<f32>().map(drop),
        DType::F64 => a.to_vec::<f64>().map(drop),
        DType::Complex64 => a.to_vec::<Complex64>().map(drop),
        _ => Ok(()),
    }
}

/// Cases per run: `NUVEC_FUZZ_CASES`, default 10 000.
fn cases() -> u64 {
    std::env::var("NUVEC_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000)
}

/// Runs the random cases on `device`; every panicking case is reported with its seed.
fn fuzz(device: Device, salt: u64) {
    let mut failures = Vec::new();
    for seed in 0..cases() {
        let seed = seed ^ salt;
        let outcome = std::panic::catch_unwind(|| {
            set_default_stream(Stream::default_for(device));
            let (primitive, inputs) = case(&mut SplitMix64(seed));
            if let Ok(outputs) = graph::from_primitive(primitive, inputs) {
                let _ = eval(&outputs);
                for o in &outputs {
                    let _ = read(o);
                }
            }
        });
        if outcome.is_err() {
            let (primitive, inputs) = case(&mut SplitMix64(seed));
            let shapes: Vec<_> = inputs
                .iter()
                .map(|a| (a.dtype(), a.shape().to_vec()))
                .collect();
            failures.push(format!("seed {seed:#x}: {primitive:?} on {shapes:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} panicking cases on {device:?}:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn random_primitives_never_panic_on_the_cpu() {
    fuzz(Device::cpu(), 0);
}

#[test]
fn random_primitives_never_panic_on_the_gpu() {
    if common::require_gpu().is_none() {
        return;
    }
    fuzz(Device::gpu(0), 0x5eed_0000_0000);
}
