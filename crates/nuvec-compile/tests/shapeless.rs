//! Shapeless `compile` (MLX's `shapeless=True`): one trace per input structure, dtypes and
//! ranks; replays on other shapes equal the uncompiled function bit for bit, on the CPU and
//! the GPU (where the element-wise chains are fused); operations whose parameters fix a
//! shape fail the trace with an error naming them.

use nuvec_compile::{__trace_count, Options, compile_flat_with};
use nuvec_core::{Array, DType, Device, Result, Stream, array, set_default_stream};

/// The streams to test on: the CPU, and the GPU when there is one.
fn devices() -> Vec<Device> {
    let mut devices = vec![Device::cpu()];
    if cfg!(target_os = "macos") && !cfg!(miri) {
        set_default_stream(Stream::default_for(Device::gpu(0)));
        if (&Array::from(1.0f32) + 1.0).eval().is_ok() {
            devices.push(Device::gpu(0));
        } else {
            assert!(
                std::env::var("NUVEC_REQUIRE_GPU").as_deref() != Ok("1"),
                "NUVEC_REQUIRE_GPU=1 but no Metal device is available"
            );
        }
    }
    devices
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn values(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.next() % 1601) as f32 - 800.0) / 200.0)
            .collect()
    }
    /// Random values of `shape` and `dtype`, contiguous or transposed.
    fn input(&mut self, shape: &[usize], dtype: DType) -> Array {
        let n = shape.iter().product();
        let a = if shape.len() == 2 && self.next().is_multiple_of(2) {
            Array::from_slice(&self.values(n), [shape[1], shape[0]]).t()
        } else {
            Array::from_slice(&self.values(n), shape)
        };
        let a = a.astype(dtype);
        a.eval().unwrap();
        a
    }
}

/// The outputs' bits (as float32: exact for these dtypes), and their shapes.
fn bits(outputs: &[Array]) -> Vec<(Vec<i32>, Vec<u32>)> {
    outputs
        .iter()
        .map(|o| {
            let v = o.astype(DType::F32).to_vec::<f32>().unwrap();
            let v = v
                .iter()
                .map(|x| if x.is_nan() { u32::MAX } else { x.to_bits() })
                .collect();
            (o.shape().to_vec(), v)
        })
        .collect()
}

/// A function over positional arrays, as `compile_flat` takes.
type Outputs = Result<Vec<Array>>;
type Flat<'a> = dyn Fn(&[Array]) -> Outputs + 'a;
type FlatFn = fn(&[Array]) -> Outputs;

fn silu(xs: &[Array]) -> Result<Vec<Array>> {
    Ok(vec![xs[0].multiply(xs[0].sigmoid())])
}

fn swiglu(xs: &[Array]) -> Result<Vec<Array>> {
    let g = &xs[0];
    Ok(vec![g.multiply(g.sigmoid()).multiply(&xs[1])])
}

/// mrun's `compute_g` (Gated DeltaNet's decay): `-exp(A_log) · softplus(a + dt_bias)` with
/// the two per-head parameters captured as constants.
fn gate(a_log: &Array, dt_bias: &Array) -> impl Fn(&[Array]) -> Result<Vec<Array>> {
    move |xs: &[Array]| {
        let a = xs[0].astype(DType::F32).add(dt_bias);
        Ok(vec![
            a_log
                .exp()
                .negative()
                .multiply(a.logaddexp(0.0))
                .astype(xs[0].dtype()),
        ])
    }
}

#[test]
fn activations_trace_once_for_every_length() {
    let mut rng = Rng(0x5eed);
    let heads = 8;
    let a_log = Array::from_slice(&rng.values(heads), [heads]);
    let dt_bias = Array::from_slice(&rng.values(heads), [heads]);
    let gate = gate(&a_log, &dt_bias);
    let functions: [(&str, &Flat<'_>, usize); 3] = [
        ("silu", &silu, 1),
        ("swiglu", &swiglu, 2),
        ("gate", &gate, 1),
    ];
    for device in devices() {
        set_default_stream(Stream::default_for(device));
        for (name, f, arity) in functions {
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let compiled = compile_flat_with(Options::default().shapeless(true), f);
                let before = __trace_count();
                for rows in [1, 7, 512, 2048, 3841, 0] {
                    let width = if name == "gate" { heads } else { 64 };
                    let xs: Vec<Array> = (0..arity)
                        .map(|_| rng.input(&[rows, width], dtype))
                        .collect();
                    let want = bits(&f(&xs).unwrap());
                    let got = bits(&compiled(&xs).unwrap());
                    assert_eq!(got, want, "{name} {dtype:?} {rows} rows on {device:?}");
                }
                assert_eq!(
                    __trace_count() - before,
                    1,
                    "{name} {dtype:?} on {device:?}"
                );
                // Another rank traces again.
                let xs: Vec<Array> = (0..arity)
                    .map(|_| rng.input(&[2, 3, if name == "gate" { heads } else { 4 }], dtype))
                    .collect();
                assert_eq!(bits(&compiled(&xs).unwrap()), bits(&f(&xs).unwrap()));
                assert_eq!(__trace_count() - before, 2);
            }
        }
    }
}

/// The traced broadcasting pattern does not bind replays: a row vector against a matrix,
/// a column against it, sizes of 1 on either side.
#[test]
fn broadcasting_follows_the_shapes_of_each_call() {
    let f = |xs: &[Array]| -> Result<Vec<Array>> {
        let (x, b) = (&xs[0], &xs[1]);
        let y = x.multiply(b).add(x).subtract(0.5);
        Ok(vec![y.greater(b).select(&y, b), y])
    };
    let mut rng = Rng(7);
    for device in devices() {
        set_default_stream(Stream::default_for(device));
        let compiled = compile_flat_with(Options::default().shapeless(true), f);
        let before = __trace_count();
        for (x, b) in [
            ([4, 8], [1, 8]),
            ([1, 8], [1, 8]),
            ([5, 8], [5, 1]),
            ([3, 1], [1, 6]),
            ([1, 1], [9, 2]),
            ([6, 8], [6, 8]),
        ] {
            let xs = [rng.input(&x, DType::F32), rng.input(&b, DType::F32)];
            assert_eq!(
                bits(&compiled(&xs).unwrap()),
                bits(&f(&xs).unwrap()),
                "{x:?} with {b:?} on {device:?}"
            );
        }
        assert_eq!(__trace_count() - before, 1);
        // Shapes that do not broadcast fail as the uncompiled call does.
        let xs = [array![[1.0, 2.0, 3.0]], array![[1.0, 2.0]]];
        assert!(f(&xs).unwrap()[0].eval().is_err());
        // (On the GPU the error names the fused kernel the operations became.)
        let err = compiled(&xs).unwrap_err().to_string();
        assert!(err.contains("shape mismatch"), "{err}");
    }
}

/// A fused member the function also returns keeps its own (smaller) shape.
#[test]
fn intermediate_outputs_keep_their_shapes() {
    let f = |xs: &[Array]| -> Result<Vec<Array>> {
        let scaled = xs[1].multiply(2.0).exp();
        Ok(vec![scaled.clone(), xs[0].multiply(&scaled).add(1.0)])
    };
    let mut rng = Rng(11);
    for device in devices() {
        set_default_stream(Stream::default_for(device));
        let compiled = compile_flat_with(Options::default().shapeless(true), f);
        for rows in [3, 1, 17] {
            let xs = [
                rng.input(&[rows, 5], DType::F16),
                rng.input(&[5], DType::F16),
            ];
            assert_eq!(bits(&compiled(&xs).unwrap()), bits(&f(&xs).unwrap()));
        }
    }
}

#[test]
fn operations_that_fix_a_shape_are_rejected() {
    let cases: [(&str, FlatFn); 3] = [
        ("reshape", |xs| Ok(vec![xs[0].reshape([12]).exp()])),
        ("slice", |xs| Ok(vec![xs[0].i(0..2).exp()])),
        ("broadcast_to", |xs| {
            Ok(vec![xs[0].broadcast_to([2, 3, 4]).exp()])
        }),
    ];
    for device in devices() {
        set_default_stream(Stream::default_for(device));
        for (name, f) in cases {
            let x = Array::from_slice(&[0.5f32; 12], [3, 4]);
            let err =
                compile_flat_with(Options::default().shapeless(true), f)(std::slice::from_ref(&x))
                    .unwrap_err()
                    .to_string();
            assert!(
                err.contains(name) && err.contains("shapeless"),
                "{name}: {err}"
            );
            // Without the option the same function compiles.
            let shaped = compile_flat_with(Options::default(), f)(std::slice::from_ref(&x));
            assert_eq!(bits(&shaped.unwrap()), bits(&f(&[x]).unwrap()));
        }
    }
}

/// `compile` over trees and `compile_mut` honour the option; `compile_mut` keys its state
/// exactly.
#[test]
fn tree_and_stateful_entry_points() {
    set_default_stream(Stream::default_for(Device::cpu()));
    let before = __trace_count();
    let f = Options::default()
        .shapeless(true)
        .compile(|(x, w): &(Array, Array)| x.multiply(w).sigmoid());
    for rows in [2, 9, 33] {
        let x = Array::from_slice(&vec![0.25f32; rows * 4], [rows, 4]);
        let w = array![1.0, -1.0, 2.0, 0.5];
        let want = x.multiply(&w).sigmoid();
        assert_eq!(bits(&[f(&(x, w)).unwrap()]), bits(&[want]));
    }
    assert_eq!(__trace_count() - before, 1);

    // A step over chunks of varying length: the state (a [4] accumulator) is exact, the
    // chunk shapeless.
    let step = Options::default()
        .shapeless(true)
        .compile_mut(|acc: &mut Array, x: &Array| {
            let total = x.multiply(2.0).sum(0);
            *acc = acc.add(&total);
            Ok::<_, nuvec_core::Error>(total)
        });
    let before = __trace_count();
    let mut acc = array![0.0, 0.0, 0.0, 0.0];
    let mut want = array![0.0, 0.0, 0.0, 0.0];
    for rows in [1, 5, 12, 3] {
        let x = Array::from_slice(&vec![0.5f32; rows * 4], [rows, 4]);
        let _total = step(&mut acc, &x).unwrap();
        want = want.add(x.multiply(2.0).sum(0));
    }
    assert_eq!(bits(&[acc]), bits(&[want]));
    assert_eq!(__trace_count() - before, 1);
}
