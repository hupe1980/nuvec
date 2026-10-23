//! The error model of the array API: every failing op family gives a failed
//! array whose `Error::Op` names the op and *this file's* line; failures are sticky and keep
//! best-effort metadata; random arguments never panic.

use std::panic::{AssertUnwindSafe, catch_unwind};

use nuvec_core::*;

/// Asserts that `a` failed at `line` of this file with `op` and a kind matching `kind`.
#[track_caller]
fn failed_here(a: &Array, op: &str, line: u32, kind: fn(&OpErrorKind) -> bool) {
    let e = a
        .error()
        .unwrap_or_else(|| panic!("{op}: not failed: {a:?}"))
        .clone();
    let Error::Op {
        op: name,
        at,
        kind: k,
    } = &e
    else {
        panic!("{op}: not an op error: {e}");
    };
    assert_eq!(*name, op, "{e}");
    assert!(at.file().ends_with("tests/errors.rs"), "{e}");
    // The call starts on the line of `case!` or, when formatted over several lines, just
    // below it.
    assert!(
        (line..=line + 3).contains(&at.line()),
        "{e} (case at line {line})"
    );
    assert!(kind(k), "{op}: unexpected kind {k:?}");
    assert!(a.is_failed());
    assert_eq!(a.eval(), Err(e.clone()));
}

fn f32s(shape: [i32; 2]) -> Array {
    ops::zeros(shape, DType::F32)
}

macro_rules! case {
    ($expr:expr, $op:literal, $kind:pat) => {{
        let line = line!();
        let a: Array = $expr;
        failed_here(&a, $op, line, |k| matches!(k, $kind));
    }};
}

#[test]
fn every_op_family_reports_the_callers_location() {
    let x = f32s([2, 3]);
    let y = f32s([4, 5]);
    let i = ops::zeros([2], DType::I32);
    let c = ops::zeros([2], DType::Complex64);
    // Element-wise binary (methods, free functions and operators), unary.
    case!(x.add(&y), "add", OpErrorKind::ShapeMismatch { .. });
    case!(
        ops::divide(&x, &y),
        "divide",
        OpErrorKind::ShapeMismatch { .. }
    );
    case!(&x * &y, "multiply", OpErrorKind::ShapeMismatch { .. });
    case!(
        i.add(300_000_000_000i64),
        "add",
        OpErrorKind::InvalidArgument(_)
    );
    case!(c.exp(), "exp", OpErrorKind::UnsupportedDType(_));
    case!(c.less(&c), "less", OpErrorKind::UnsupportedDType(_));
    case!(x.select(&x, &y), "where", OpErrorKind::ShapeMismatch { .. });
    // Reductions, keepdims twins, var/std, arg reductions, softmax.
    case!(
        x.sum(2),
        "sum",
        OpErrorKind::InvalidAxis { axis: 2, ndim: 2 }
    );
    case!(
        x.mean_keepdims([0, 0]),
        "mean",
        OpErrorKind::InvalidAxis { .. }
    );
    case!(x.var(-3), "var", OpErrorKind::InvalidAxis { .. });
    case!(x.argmax(5), "argmax", OpErrorKind::InvalidAxis { .. });
    case!(x.softmax(9), "softmax", OpErrorKind::InvalidAxis { .. });
    case!(c.max(..), "max", OpErrorKind::UnsupportedDType(_));
    // Shape ops.
    case!(x.reshape([4, 2]), "reshape", OpErrorKind::InvalidShape(_));
    case!(
        x.broadcast_to([3, 3]),
        "broadcast_to",
        OpErrorKind::ShapeMismatch { .. }
    );
    case!(
        x.transpose([0, 0]),
        "transpose",
        OpErrorKind::InvalidAxis { .. }
    );
    case!(x.squeeze(0), "squeeze", OpErrorKind::InvalidArgument(_));
    case!(
        x.expand_dims(9),
        "expand_dims",
        OpErrorKind::InvalidAxis { .. }
    );
    // Matrix products.
    case!(x.matmul(&y), "matmul", OpErrorKind::ShapeMismatch { .. });
    case!(
        x.addmm(&x, &y, 1.0, 1.0),
        "addmm",
        OpErrorKind::ShapeMismatch { .. }
    );
    case!(i.matmul(&i), "matmul", OpErrorKind::UnsupportedDType(_));
    // Scans and sorting.
    case!(x.cumsum(3), "cumsum", OpErrorKind::InvalidAxis { .. });
    case!(x.topk(9, -1), "topk", OpErrorKind::InvalidArgument(_));
    case!(
        x.partition(9, 0),
        "partition",
        OpErrorKind::InvalidArgument(_)
    );
    // Gathers, scatters, indexing, updates.
    case!(x.take(&i, 7), "take", OpErrorKind::InvalidAxis { .. });
    case!(
        x.take_along_axis(&i, 0),
        "take_along_axis",
        OpErrorKind::InvalidArgument(_)
    );
    case!(
        x.put_along_axis(&i, 1.0, 0),
        "put_along_axis",
        OpErrorKind::InvalidArgument(_)
    );
    case!(x.i(5), "index", OpErrorKind::InvalidIndex(_));
    case!(
        x.clone().at(5).set(1.0),
        "at.set",
        OpErrorKind::InvalidIndex(_)
    );
    // Joining and splitting.
    case!(
        ops::concatenate(&[x.clone(), y.clone()], 0),
        "concatenate",
        OpErrorKind::ShapeMismatch { .. }
    );
    case!(
        ops::stack(&[x.clone(), y.clone()], 0),
        "stack",
        OpErrorKind::ShapeMismatch { .. }
    );
    case!(
        x.split(4, 1).remove(0),
        "split",
        OpErrorKind::InvalidArgument(_)
    );
    // Quantization and convolution.
    case!(
        x.quantize(64, 4).0,
        "quantize",
        OpErrorKind::InvalidArgument(_)
    );
    case!(x.conv1d(&x, 1, 0, 1, 1), "conv1d", _);
    // fast and random.
    case!(fast::rms_norm(&x, &i, 1e-5), "rms_norm", _);
    case!(fast::cross_entropy(&x, &x), "cross_entropy", _);
    case!(
        random::uniform(0.0, 1.0, [2], DType::I32, None),
        "uniform",
        OpErrorKind::UnsupportedDType(_)
    );
    case!(
        random::normal([-1], DType::F32, 0.0, 1.0, None),
        "normal",
        OpErrorKind::InvalidShape(_)
    );
    // Constructors.
    case!(
        Array::from_slice(&[1.0f32, 2.0], [3]),
        "from_slice",
        OpErrorKind::SizeMismatch {
            expected: 3,
            actual: 2
        }
    );
    case!(
        ops::zeros([-2], DType::F32),
        "zeros",
        OpErrorKind::InvalidShape(_)
    );
    case!(
        ops::arange(0, 10, 0),
        "arange",
        OpErrorKind::InvalidArgument(_)
    );
    case!(ops::full([i32::MAX, 4], 1.0), "full", _);
}

#[test]
fn failures_are_sticky_and_never_evaluate() {
    let x = f32s([2, 3]);
    let c = x.matmul(&f32s([4, 5]));
    let err = c.check_ref();
    // The first error wins, through methods, operators, indexing and reductions.
    let d = (c.exp() + 1.0).sum(-1);
    assert_eq!(d.error(), Some(&err));
    assert_eq!(d.eval(), Err(err.clone()));
    assert_eq!(d.item::<f32>(), Err(err.clone()));
    assert_eq!(d.to_vec::<f32>(), Err(err.clone()));
    assert_eq!(
        x.add(&c).error(),
        Some(&err),
        "argument order: a failed rhs"
    );
    assert_eq!(c.i(0).error(), Some(&err));
    assert_eq!(
        ops::concatenate(&[x.clone(), c.clone()], 0).error(),
        Some(&err)
    );
    // `check` on a valid array is O(1) and does not evaluate.
    let ok = x.exp().sum(..);
    let ok = ok.check().unwrap();
    assert!(!ok.is_evaluated());
    assert!(ok.error().is_none());
}

trait CheckRef {
    fn check_ref(&self) -> Error;
}

impl CheckRef for Array {
    fn check_ref(&self) -> Error {
        self.error().cloned().expect("a failed array")
    }
}

#[test]
fn best_effort_metadata() {
    // Rule 1: inference on a failed input's metadata.
    let failed = Array::from_slice(&[1.0f32], [2, 3]);
    assert!(failed.is_failed());
    let r = failed.reshape([3, 2]);
    assert_eq!((r.shape(), r.is_failed()), (&[3, 2][..], true));
    assert_eq!(failed.sum(-1).shape(), [2]);
    // Rule 2: the op itself fails: the first operand's metadata.
    let m = f32s([2, 3]).matmul(&f32s([4, 5]));
    assert_eq!(m.shape(), [2, 3]);
    // Rule 3: constructors keep the requested shape when it is valid.
    assert_eq!(Array::from_slice(&[1u8, 2], [2, 2]).shape(), [2, 2]);
    assert_eq!(ops::zeros([-1, 2], DType::F16).shape(), [0i32; 0]);
    assert_eq!(ops::zeros([-1, 2], DType::F16).dtype(), DType::F16);
    // Rule 4: multi-output ops return the implied number of outputs.
    let parts: [Array; 3] = failed.split_n::<3>(0);
    assert!(parts.iter().all(Array::is_failed));
    assert_eq!(f32s([2, 3]).split(4, 1).len(), 4);
    assert_eq!(f32s([2, 3]).split_at(&[1, 2], 7).len(), 3);
    let (q, s, b) = f32s([2, 3]).quantize(64, 4);
    assert!(q.is_failed() && s.is_failed() && b.is_failed());
    // Host decisions on a failed array's shape go through `shape_n`, a boundary.
    assert_eq!(failed.shape_n::<2>().unwrap_err(), failed.check_ref());
    assert!(f32s([2, 3]).shape_n::<4>().is_err());
}

#[test]
fn display_names_the_op_and_the_location() {
    let line = line!() + 1;
    let e = f32s([2, 3]).matmul(&f32s([4, 5])).check().unwrap_err();
    assert_eq!(
        e.to_string(),
        format!(
            "matmul at {}:{line}:26: shape mismatch, [2, 3] vs [4, 5]",
            file!()
        )
    );
}

/// A small deterministic generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn axis(&mut self) -> i32 {
        self.below(17) as i32 - 8
    }
    fn shape(&mut self) -> Vec<i32> {
        (0..self.below(4)).map(|_| self.below(4) as i32).collect()
    }
}

/// 10,000 seeded random calls with random shapes (0-size and 0-D included), axes in
/// [-8, 8], dtypes and index forms: no panic, and every failure is a located `Error::Op`.
#[test]
#[cfg_attr(miri, ignore = "10,000 calls: covered natively")]
fn random_arguments_never_panic() {
    let mut rng = Rng(0x0062_0062);
    let dtypes = [
        DType::Bool,
        DType::U8,
        DType::I32,
        DType::F16,
        DType::F32,
        DType::Complex64,
    ];
    let n = if cfg!(miri) { 100 } else { 10_000 };
    with_stream(Device::cpu(), || {
        for k in 0..n {
            let dt = dtypes[rng.below(dtypes.len() as u64) as usize];
            let x = ops::zeros(rng.shape(), dt);
            let y = ops::ones(rng.shape(), dtypes[rng.below(6) as usize]);
            let (a1, a2) = (rng.axis(), rng.axis());
            let shape = rng.shape();
            let result = catch_unwind(AssertUnwindSafe(|| -> Vec<Array> {
                match k % 20 {
                    0 => vec![x.add(&y), x.matmul(&y)],
                    1 => vec![x.sum(a1), x.sum([a1, a2]), x.mean_keepdims(a1..)],
                    2 => vec![x.argmax(a1), x.cumsum(a1), x.sort(a1)],
                    3 => vec![x.reshape(&shape[..]), x.broadcast_to(&shape[..])],
                    4 => vec![
                        x.transpose([a1, a2]),
                        x.squeeze(a1),
                        x.expand_dims([a1, a2]),
                    ],
                    5 => vec![x.i(a1), x.i((a1, .., a2)), x.i((Ellipsis, a1..a2))],
                    6 => vec![x.i(((..).step(a1), NewAxis)), x.i((a2.., ..=a1))],
                    7 => vec![x.clone().at(a1).set(1), x.clone().at((.., a2)).add(&y)],
                    8 => x.split(a1, a2),
                    9 => x.split_at(&[a1.max(0)], a2),
                    10 => vec![ops::concatenate(&[x.clone(), y.clone()], a1)],
                    11 => vec![ops::stack(&[x.clone(), y.clone()], a1)],
                    12 => vec![x.take(&y.astype(DType::I32), a1), x.take_along_axis(&y, a1)],
                    13 => vec![x.var_with(
                        a1,
                        VarOptions {
                            ddof: a2,
                            keepdims: true,
                        },
                    )],
                    14 => vec![x.topk(a1, a2), x.partition(a1, a2), x.repeat(a1, a2)],
                    15 => vec![x.select(&y, 0.0), x.logsumexp(a1), x.softmax([a1])],
                    16 => vec![ops::arange(a1, a2, rng.axis()), ops::full(&shape[..], a1)],
                    17 => vec![x.quantize(a1 * 16, a2).0, x.conv1d(&y, a1, a2, 1, 1)],
                    18 => vec![fast::rms_norm(&x, &y, 1e-5), fast::softmax_like(&x)],
                    _ => vec![
                        x.i(y.astype(DType::I32)),
                        x.clone().at(y.astype(DType::I32)).add(1.0),
                    ],
                }
            }));
            let outs = result.unwrap_or_else(|_| panic!("call {k} panicked"));
            for o in outs {
                if let Some(e) = o.error() {
                    assert!(
                        matches!(e, Error::Op { .. }) && e.location().is_some(),
                        "call {k}: {e} has no caller location"
                    );
                    assert!(
                        e.location().unwrap().file().ends_with("tests/errors.rs"),
                        "{e}"
                    );
                }
                // Evaluating never panics either.
                let _ = o.eval();
            }
        }
    });
}

/// `fast` functions in the random test.
mod fast {
    pub(crate) use nuvec_core::fast::*;

    #[track_caller]
    pub(crate) fn softmax_like(x: &nuvec_core::Array) -> nuvec_core::Array {
        x.softmax(-1)
    }
}
