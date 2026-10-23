//! Methods and their free-function mirrors: for every operation of the
//! `ops` table, `x.op(args)` and `ops::op(&x, args)` record the same primitive and give the
//! same values, on contiguous and non-contiguous inputs of several dtypes.

use std::collections::BTreeSet;

use nuvec_core::*;

/// The operands of one round.
struct Ctx {
    x: Array,
    y: Array,
    idx: Array,
    idx2: Array,
}

type Op = fn(&Ctx) -> Array;

/// `(name, method form, free form)`.
macro_rules! pairs {
    ($c:ident => $($name:ident ($($arg:tt)*);)*) => {
        vec![$(
            (
                stringify!($name),
                (|$c: &Ctx| $c.x.$name($($arg)*)) as Op,
                (|$c: &Ctx| ops::$name(&$c.x, $($arg)*)) as Op,
            ),
        )*]
    };
}

fn table() -> Vec<(&'static str, Op, Op)> {
    let mut t = pairs! { c =>
        add(&c.y); subtract(&c.y); multiply(&c.y); divide(&c.y); maximum(&c.y); minimum(&c.y); logaddexp(&c.y);
        equal(&c.y); not_equal(&c.y); less(&c.y); less_equal(&c.y); greater(&c.y); greater_equal(&c.y);
        negative(); abs(); exp(); log(); sqrt(); sin(); cos(); tanh(); sigmoid(); erfinv();
        sign(); conjugate(); stop_gradient(); contiguous(); copy();
        astype(DType::F16);
        matmul(&c.y.t());
        addmm(&c.y, &c.y.t(), 0.5, 2.0);
        sum(-1); sum_keepdims([0, 1]); prod(0); prod_keepdims(..); max(-1); max_keepdims(-1);
        min(0); min_keepdims(0); all(..); all_keepdims(1); any(-1); any_keepdims(..);
        mean(1); mean_keepdims(..); logsumexp(-1); logsumexp_keepdims(0); var(..);
        var_keepdims(-1); var_with(0, VarOptions { ddof: 1, keepdims: true }); std(1);
        std_keepdims(..); std_with(.., VarOptions { ddof: 1, keepdims: false });
        argmax(-1); argmax_keepdims(..); argmin(0); argmin_keepdims(1); softmax(-1);
        cumsum(-1); cumsum_with(0, ScanOptions { reverse: true, exclusive: true });
        cumprod(1); cumprod_with(.., ScanOptions { reverse: false, exclusive: true });
        cummax(0); cummax_with(-1, ScanOptions { reverse: true, exclusive: false });
        cummin(-1); cummin_with(0, ScanOptions::default());
        logcumsumexp(1); logcumsumexp_with(1, ScanOptions { reverse: true, exclusive: false });
        sort(-1); argsort(0); partition(1, -1); argpartition(-2, 1); topk(2, -1);
        reshape([2, -1]); transpose([1, 0]); broadcast_to([2, 3, 4]); expand_dims([0, -1]);
        squeeze(..); repeat(2, 0);
        take(&c.idx, 1); take_along_axis(&c.idx2, 1); put_along_axis(&c.idx2, 5.0, 1);
        scatter_add_along_axis(&c.idx2, &c.y, 1);
        zeros_like(); ones_like(); full_like(7);
    };
    // Operations with other result types or operands: compared through one array.
    t.push((
        "select",
        |c| c.x.greater(0.0).select(&c.x, &c.y),
        |c| ops::where_(c.x.greater(0.0), &c.x, &c.y),
    ));
    t.push((
        "split",
        |c| c.x.split(2, 1).remove(1),
        |c| ops::split(&c.x, 2, 1).remove(1),
    ));
    t.push((
        "split_at",
        |c| c.x.split_at(&[1], 0).remove(1),
        |c| ops::split_at(&c.x, &[1], 0).remove(1),
    ));
    t.push((
        "conv1d",
        |c| conv_input(&c.x).conv1d(&conv_weight(), 1, 1, 1, 1),
        |c| ops::conv1d(&conv_input(&c.x), &conv_weight(), 1, 1, 1, 1),
    ));
    t.push((
        "quantize",
        |c| weights(&c.x).quantize(64, 4).1,
        |c| ops::quantize(&weights(&c.x), 64, 4).1,
    ));
    t.push((
        "dequantize",
        |c| {
            let (q, s, b) = weights(&c.x).quantize(64, 4);
            q.dequantize(&s, &b, 64, 4)
        },
        |c| {
            let (q, s, b) = ops::quantize(&weights(&c.x), 64, 4);
            ops::dequantize(&q, &s, &b, 64, 4)
        },
    ));
    t.push((
        "quantized_matmul",
        |c| {
            let (q, s, b) = weights(&c.x).quantize(64, 4);
            weights(&c.y).quantized_matmul(&q, &s, &b, true, 64, 4)
        },
        |c| {
            let (q, s, b) = weights(&c.x).quantize(64, 4);
            ops::quantized_matmul(&weights(&c.y), &q, &s, &b, true, 64, 4)
        },
    ));
    t.push((
        "gather_qmm",
        |c| {
            let (q, s, b) = weights(&c.x).quantize(64, 4);
            weights(&c.y).gather_qmm(&q, &s, &b, None, None, true, 64, 4, false)
        },
        |c| {
            let (q, s, b) = weights(&c.x).quantize(64, 4);
            ops::gather_qmm(&weights(&c.y), &q, &s, &b, None, None, true, 64, 4, false)
        },
    ));
    t.push((
        "gather_mm",
        |c| c.x.gather_mm(&c.y.t(), None, None, false),
        |c| ops::gather_mm(&c.x, &c.y.t(), None, None, false),
    ));
    t
}

fn conv_input(x: &Array) -> Array {
    x.reshape([1, 3, 4])
}

fn conv_weight() -> Array {
    ops::arange(0.0, 16.0, 1.0).reshape([2, 2, 4])
}

/// A [2, 64] matrix made from `x`'s 12 values.
fn weights(x: &Array) -> Array {
    x.astype(DType::F32)
        .reshape([-1])
        .take(&ops::arange(0, 128, 1).multiply(7).remainder_12(), 0)
        .reshape([2, 64])
}

trait Rem12 {
    fn remainder_12(&self) -> Array;
}

impl Rem12 for Array {
    fn remainder_12(&self) -> Array {
        // `x % 12` without a remainder op: x − 12·⌊x / 12⌋ in integers.
        self.subtract(self.divide(12).astype(DType::I32).multiply(12))
    }
}

fn data(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// The rounds: contiguous, transposed and sliced operands, float32 and float16.
fn rounds() -> Vec<Ctx> {
    let base = |seed| Array::from_slice(&data(seed, 12), [3, 4]);
    let transposed = |seed| Array::from_slice(&data(seed, 12), [4, 3]).t();
    let sliced = |seed| Array::from_slice(&data(seed, 24), [3, 8]).i((.., (..).step(2)));
    let idx = Array::from([2i32, 0, -1]);
    let idx2 = Array::from_slice(&[1i32, 3, 0], [3, 1]);
    let mut out = Vec::new();
    for (k, make) in [
        &base as &dyn Fn(u64) -> Array,
        &transposed as &dyn Fn(u64) -> Array,
        &sliced as &dyn Fn(u64) -> Array,
    ]
    .into_iter()
    .enumerate()
    {
        for dtype in [DType::F32, DType::F16] {
            out.push(Ctx {
                x: make(10 + k as u64).astype(dtype),
                y: make(20 + k as u64).astype(dtype),
                idx: idx.clone(),
                idx2: idx2.clone(),
            });
        }
    }
    out
}

#[test]
fn every_method_matches_its_free_function() {
    let table = table();
    let names: BTreeSet<&str> = ops::__op_names().iter().copied().collect();
    let covered: BTreeSet<&str> = table.iter().map(|(n, ..)| *n).collect();
    let missing: Vec<_> = names.difference(&covered).collect();
    assert!(
        missing.is_empty(),
        "methods without a mirror test: {missing:?}"
    );
    for device in [Device::cpu(), Device::gpu(0)] {
        if device != Device::cpu() && !gpu_available() {
            continue;
        }
        with_stream(device, || {
            for (r, ctx) in rounds().iter().enumerate() {
                for (name, method, free) in &table {
                    let (a, b) = (method(ctx), free(ctx));
                    assert_eq!(
                        graph::primitive(&a),
                        graph::primitive(&b),
                        "{name}, round {r}: different primitives"
                    );
                    assert_eq!((a.shape(), a.dtype()), (b.shape(), b.dtype()), "{name}");
                    assert_eq!(a.to_string(), b.to_string(), "{name}, round {r}, {device}");
                }
            }
        });
    }
}

#[test]
fn select_is_where_with_weak_scalars() {
    let m = array![true, false, true];
    let a = Array::from([1.5f32, 2.5, 3.5]).astype(DType::F16);
    let s = m.select(&a, 0.0);
    let w = ops::where_(&m, &a, 0.0);
    assert_eq!(s.dtype(), DType::F16, "a weak 0.0 adapts to float16");
    assert_eq!(s.to_string(), w.to_string());
    assert_eq!(m.select(1, 2.5).dtype(), DType::F32);
    assert_eq!(ops::where_(true, 1, 2).dtype(), DType::I32);
}

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
