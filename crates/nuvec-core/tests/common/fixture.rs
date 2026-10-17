//! Recorded MLX fixtures: parser, operation dispatcher and tolerance comparator.
//!
//! The file is written by `scripts/mlx_fixtures.py` (format `BSMXFIX1`, little-endian):
//! `magic | meta:str | count:u32 | case*`, where
//! `case := name:str op:str n_in:u8 tensor* n_out:u8 tensor*`,
//! `tensor := dtype:u8 ndim:u8 dims:u64* nbytes:u64 data` and `str := len:u16 utf8`.

#![allow(unreachable_pub, dead_code)]

use std::collections::HashMap;

use nuvec_core::*;

/// dtype codes, in nuvec's `DType` order.
pub const DTYPES: [DType; 14] = [
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

/// A recorded tensor: row-major little-endian bytes.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// One recorded MLX computation.
#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    pub op: String,
    pub params: HashMap<String, String>,
    pub inputs: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        out
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take(2).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn str(&mut self) -> String {
        let n = self.u16() as usize;
        String::from_utf8(self.take(n).to_vec()).unwrap()
    }
    fn tensor(&mut self) -> Tensor {
        let dtype = DTYPES[self.u8() as usize];
        let ndim = self.u8() as usize;
        let shape = (0..ndim).map(|_| self.u64() as usize).collect();
        let n = self.u64() as usize;
        Tensor {
            dtype,
            shape,
            data: self.take(n).to_vec(),
        }
    }
}

/// Parses a fixture file into its metadata line and cases.
pub fn parse(bytes: &[u8]) -> (String, Vec<Case>) {
    assert_eq!(&bytes[..8], b"BSMXFIX1", "not a fixture file");
    let mut r = Reader { bytes, pos: 8 };
    let meta = r.str();
    let count = r.u32();
    let cases = (0..count)
        .map(|_| {
            let name = r.str();
            let spelled = r.str();
            let mut parts = spelled.split('|');
            let op = parts.next().unwrap().to_owned();
            let params = parts
                .map(|kv| {
                    let (k, v) = kv.split_once('=').unwrap();
                    (k.to_owned(), v.to_owned())
                })
                .collect();
            let inputs = (0..r.u8()).map(|_| r.tensor()).collect();
            let outputs = (0..r.u8()).map(|_| r.tensor()).collect();
            Case {
                name,
                op,
                params,
                inputs,
                outputs,
            }
        })
        .collect();
    assert_eq!(r.pos, bytes.len(), "trailing bytes in fixture file");
    (meta, cases)
}

/// The fixtures committed with the tests.
pub fn load() -> (String, Vec<Case>) {
    parse(include_bytes!("../fixtures/mlx.bin"))
}

// ── Tensor <-> Array ────────────────────────────────────────────────────────

macro_rules! decode {
    ($data:expr, $t:ty) => {
        $data
            .chunks_exact(size_of::<$t>())
            .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<$t>>()
    };
}

/// Elements as f64 pairs `(re, im)`; `im` is 0 for real types.
pub fn values(t: &Tensor) -> Vec<(f64, f64)> {
    let real = |v: Vec<f64>| v.into_iter().map(|x| (x, 0.0)).collect();
    match t.dtype {
        DType::Bool | DType::U8 => real(t.data.iter().map(|&b| f64::from(b)).collect()),
        DType::U16 => real(decode!(t.data, u16).into_iter().map(f64::from).collect()),
        DType::U32 => real(decode!(t.data, u32).into_iter().map(f64::from).collect()),
        DType::U64 => real(decode!(t.data, u64).into_iter().map(|x| x as f64).collect()),
        DType::I8 => real(decode!(t.data, i8).into_iter().map(f64::from).collect()),
        DType::I16 => real(decode!(t.data, i16).into_iter().map(f64::from).collect()),
        DType::I32 => real(decode!(t.data, i32).into_iter().map(f64::from).collect()),
        DType::I64 => real(decode!(t.data, i64).into_iter().map(|x| x as f64).collect()),
        DType::F16 => real(
            decode!(t.data, u16)
                .into_iter()
                .map(|b| f16::from_bits(b).to_f64())
                .collect(),
        ),
        DType::BF16 => real(
            decode!(t.data, u16)
                .into_iter()
                .map(|b| bf16::from_bits(b).to_f64())
                .collect(),
        ),
        DType::F32 => real(decode!(t.data, f32).into_iter().map(f64::from).collect()),
        DType::F64 => real(decode!(t.data, f64)),
        _ => decode!(t.data, f32)
            .chunks_exact(2)
            .map(|c| (f64::from(c[0]), f64::from(c[1])))
            .collect(),
    }
}

/// Exact integer values (for integer dtypes, where f64 would lose precision).
fn int_values(t: &Tensor) -> Option<Vec<i128>> {
    Some(match t.dtype {
        DType::U64 => decode!(t.data, u64).into_iter().map(i128::from).collect(),
        DType::I64 => decode!(t.data, i64).into_iter().map(i128::from).collect(),
        DType::U32 => decode!(t.data, u32).into_iter().map(i128::from).collect(),
        DType::I32 => decode!(t.data, i32).into_iter().map(i128::from).collect(),
        _ => return None,
    })
}

/// Builds an array holding the tensor's elements.
pub fn to_array(t: &Tensor) -> Array {
    let s = &t.shape;
    let a = match t.dtype {
        DType::Bool => Array::from_slice(&t.data.iter().map(|&b| b != 0).collect::<Vec<_>>(), s),
        DType::U8 => Array::from_slice(&t.data, s),
        DType::U16 => Array::from_slice(&decode!(t.data, u16), s),
        DType::U32 => Array::from_slice(&decode!(t.data, u32), s),
        DType::U64 => Array::from_slice(&decode!(t.data, u64), s),
        DType::I8 => Array::from_slice(&decode!(t.data, i8), s),
        DType::I16 => Array::from_slice(&decode!(t.data, i16), s),
        DType::I32 => Array::from_slice(&decode!(t.data, i32), s),
        DType::I64 => Array::from_slice(&decode!(t.data, i64), s),
        DType::F16 => Array::from_slice(
            &decode!(t.data, u16)
                .into_iter()
                .map(f16::from_bits)
                .collect::<Vec<_>>(),
            s,
        ),
        DType::BF16 => Array::from_slice(
            &decode!(t.data, u16)
                .into_iter()
                .map(bf16::from_bits)
                .collect::<Vec<_>>(),
            s,
        ),
        DType::F32 => Array::from_slice(&decode!(t.data, f32), s),
        DType::F64 => Array::from_slice(&decode!(t.data, f64), s),
        _ => {
            let parts = decode!(t.data, f32);
            Array::from_slice(
                &parts
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| Complex64::new(c[0], c[1]))
                    .collect::<Vec<_>>(),
                s,
            )
        }
    };
    a.unwrap()
}

/// Reads an array back into a recorded-style tensor.
pub fn from_array(a: &Array) -> Result<Tensor> {
    macro_rules! bytes {
        ($t:ty) => {
            a.to_vec::<$t>()?
                .into_iter()
                .flat_map(|x| x.to_le_bytes())
                .collect()
        };
    }
    let data: Vec<u8> = match a.dtype() {
        DType::Bool => a.to_vec::<bool>()?.into_iter().map(u8::from).collect(),
        DType::U8 => a.to_vec::<u8>()?,
        DType::U16 => bytes!(u16),
        DType::U32 => bytes!(u32),
        DType::U64 => bytes!(u64),
        DType::I8 => bytes!(i8),
        DType::I16 => bytes!(i16),
        DType::I32 => bytes!(i32),
        DType::I64 => bytes!(i64),
        DType::F16 => a
            .to_vec::<f16>()?
            .into_iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        DType::BF16 => a
            .to_vec::<bf16>()?
            .into_iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        DType::F32 => bytes!(f32),
        DType::F64 => bytes!(f64),
        _ => a
            .to_vec::<Complex64>()?
            .into_iter()
            .flat_map(|z| [z.re.to_le_bytes(), z.im.to_le_bytes()])
            .flatten()
            .collect(),
    };
    Ok(Tensor {
        dtype: a.dtype(),
        shape: a.shape().to_vec(),
        data,
    })
}

// ── Operation dispatch (mirrors `apply` in the recorder) ────────────────────

fn ints(s: &str) -> Vec<isize> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(|x| x.parse().unwrap()).collect()
    }
}

fn usizes(s: &str) -> Vec<usize> {
    ints(s).into_iter().map(|x| x as usize).collect()
}

fn slice_args(spec: &str) -> Vec<SliceArg> {
    spec.split(';')
        .map(|part| {
            let f: Vec<&str> = part.split(':').collect();
            let opt = |s: &str| {
                if s.is_empty() {
                    None
                } else {
                    Some(s.parse().unwrap())
                }
            };
            SliceArg::new(opt(f[0]), opt(f[1]), f[2].parse().unwrap())
        })
        .collect()
}

/// A Python scalar recorded as `<kind>:<value>` (`complex` values are `re,im`).
fn parse_scalar(spec: &str) -> Option<Scalar> {
    let (kind, value) = spec.split_once(':')?;
    Some(match kind {
        "bool" => Scalar::Bool(value == "True"),
        "int" => Scalar::Int(value.parse().unwrap()),
        "float" => Scalar::Float(value.parse().unwrap()),
        "complex" => {
            let (re, im) = value.split_once(',').unwrap();
            Scalar::Complex(Complex64::new(re.parse().unwrap(), im.parse().unwrap()))
        }
        other => panic!("unknown scalar kind {other}"),
    })
}

/// A reduction operation: `(array, axes, keepdims)`.
type Reduction = fn(&Array, &[isize], bool) -> Result<Array>;

/// Runs the case in nuvec: one array per recorded output.
pub fn run(case: &Case) -> Result<Vec<Array>> {
    let p = |k: &str| case.params.get(k).map(String::as_str).unwrap_or("");
    let xs: Vec<Array> = case.inputs.iter().map(to_array).collect();
    let f = |xs: &[Array]| Ok(vec![apply(case, p("fn"), xs.to_vec())?]);
    match case.op.as_str() {
        "vjp" => {
            let n = xs.len() - 1;
            Ok(vjp(f, &xs[..n], &xs[n..])?.1)
        }
        "jvp" => {
            let n = xs.len() / 2;
            Ok(jvp(f, &xs[..n], &xs[n..])?.1)
        }
        "vmap" => {
            let in_axes: Vec<Option<isize>> = p("in_axes")
                .split(',')
                .map(|a| (a != "n").then(|| a.parse().unwrap()))
                .collect();
            let out_axes = [p("out_axes").parse().unwrap()];
            vmap(f, &in_axes, &out_axes)(&xs)
        }
        op => Ok(vec![apply(case, op, xs)?]),
    }
}

/// Runs operation `op` with the case's parameters on `xs`.
fn apply(case: &Case, op: &str, mut xs: Vec<Array>) -> Result<Array> {
    let p = |k: &str| case.params.get(k).map(String::as_str).unwrap_or("");
    if let Some(x) = xs.first_mut() {
        *x = match p("pre") {
            "t" => x.t()?,
            "r" => {
                let mut args = vec![SliceArg::full(); x.ndim()];
                *args.last_mut().unwrap() = SliceArg::new(None, None, -1);
                slice(x, &args)?
            }
            _ => x.clone(),
        };
    }
    let dtype = || DTYPES[p("dtype").parse::<usize>().unwrap()];
    let keep = p("keepdims") == "1";
    if op == "matmul" {
        return matmul(&xs[0], &xs[1]);
    }
    type S = ScalarOrArray;
    let binary: Option<fn(S, S) -> Result<Array>> = match op {
        "add" => Some(add),
        "subtract" => Some(subtract),
        "multiply" => Some(multiply),
        "divide" => Some(divide),
        "maximum" => Some(maximum),
        "minimum" => Some(minimum),
        "equal" => Some(equal),
        "not_equal" => Some(not_equal),
        "less" => Some(less),
        "less_equal" => Some(less_equal),
        "greater" => Some(greater),
        "greater_equal" => Some(greater_equal),
        _ => None,
    };
    if let Some(f) = binary {
        let array = S::from(&xs[0]);
        return match (parse_scalar(p("scalar")), p("side")) {
            (None, _) => f(array, S::from(&xs[1])),
            (Some(s), "left") => f(s.into(), array),
            (Some(s), _) => f(array, s.into()),
        };
    }
    let unary: Option<fn(&Array) -> Result<Array>> = match op {
        "negative" => Some(negative),
        "abs" => Some(abs),
        "exp" => Some(exp),
        "log" => Some(log),
        "sqrt" => Some(sqrt),
        "sin" => Some(sin),
        "cos" => Some(cos),
        "tanh" => Some(tanh),
        "sigmoid" => Some(sigmoid),
        _ => None,
    };
    if let Some(f) = unary {
        return f(&xs[0]);
    }
    let reduction: Option<Reduction> = match op {
        "sum" => Some(sum),
        "prod" => Some(prod),
        "max" => Some(max),
        "min" => Some(min),
        "mean" => Some(mean),
        "all" => Some(all),
        "any" => Some(any),
        _ => None,
    };
    if let Some(f) = reduction {
        return f(&xs[0], &ints(p("axes")), keep);
    }
    match op {
        "where" => where_(&xs[0], &xs[1], &xs[2]),
        "astype" => xs[0].astype(dtype()),
        "reshape" => xs[0].reshape(&ints(p("shape"))),
        "transpose" if p("axes").is_empty() => xs[0].t(),
        "transpose" => transpose(&xs[0], &usizes(p("axes"))),
        "slice" => slice(&xs[0], &slice_args(p("args"))),
        "broadcast_to" => broadcast_to(&xs[0], &usizes(p("shape"))),
        "expand_dims" => expand_dims(&xs[0], p("axis").parse().unwrap()),
        "squeeze" => squeeze(&xs[0], &ints(p("axes"))),
        "argmax" | "argmin" => {
            let axis = if p("axis").is_empty() {
                None
            } else {
                Some(p("axis").parse().unwrap())
            };
            if op == "argmax" {
                argmax(&xs[0], axis, keep)
            } else {
                argmin(&xs[0], axis, keep)
            }
        }
        "arange" => arange(
            p("start").parse().unwrap(),
            p("stop").parse().unwrap(),
            p("step").parse().unwrap(),
            dtype(),
        ),
        "zeros" => zeros(&usizes(p("shape")), dtype()),
        "ones" => ones(&usizes(p("shape")), dtype()),
        other => panic!("unknown fixture op {other}"),
    }
}

// ── Comparison ──────────────────────────────────────────────────────────────

/// Machine epsilon (spacing at 1.0) of a float dtype.
fn epsilon(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 2f64.powi(-10),
        DType::BF16 => 2f64.powi(-7),
        DType::F64 => f64::EPSILON,
        _ => f64::from(f32::EPSILON),
    }
}

/// The allowed absolute error per element for `case`, given the expected values.
fn tolerance(case: &Case, expected: &Tensor, e: f64) -> f64 {
    let eps = epsilon(expected.dtype);
    // A vmap case is as exact as the operation it batches.
    let op = match case.op.as_str() {
        "vmap" => case.params["fn"].as_str(),
        op => op,
    };
    match op {
        // Derivatives chain several rounded operations, accumulated over at most every
        // input element.
        "vjp" | "jvp" => {
            let n = case
                .inputs
                .iter()
                .map(|t| t.shape.iter().product::<usize>())
                .max();
            (n.unwrap_or(1) as f64 + 8.0) * eps * e.abs().max(1.0)
        }
        "exp" | "log" | "sqrt" | "sin" | "cos" | "tanh" | "sigmoid" => {
            4.0 * eps * e.abs().max(f64::MIN_POSITIVE)
        }
        // Complex division is not exactly rounded (and MLX's C++ build may contract to
        // FMA), see D-021.
        "divide" if expected.dtype == DType::Complex64 => {
            4.0 * eps * e.abs().max(f64::MIN_POSITIVE)
        }
        "sum" | "prod" | "mean" => {
            let n = case.inputs[0].data.len() / expected.data.len().max(1) * expected.dtype.size()
                / case.inputs[0].dtype.size().max(1);
            (n.max(1) as f64 + 1.0) * eps * e.abs().max(1.0)
        }
        "matmul" => {
            let bound = |t: &Tensor| {
                values(t)
                    .iter()
                    .fold(1.0f64, |m, &(re, im)| m.max(re.abs()).max(im.abs()))
            };
            let k = case.inputs[0].shape.last().copied().unwrap_or(1) as f64;
            4.0 * (k + 1.0) * eps * bound(&case.inputs[0]) * bound(&case.inputs[1])
        }
        _ => 0.0,
    }
}

fn close(got: f64, expected: f64, tol: f64) -> bool {
    got == expected || (got.is_nan() && expected.is_nan()) || (got - expected).abs() <= tol
}

/// Runs `case` and compares every output with MLX's; `Some` describes the first failure.
pub fn check(case: &Case) -> Option<String> {
    let got = match run(case)
        .and_then(|arrays| arrays.iter().map(from_array).collect::<Result<Vec<_>>>())
    {
        Ok(got) => got,
        Err(e) => return Some(format!("{} ({}): nuvec error: {e}", case.name, case.op)),
    };
    if got.len() != case.outputs.len() {
        return Some(format!(
            "{}: {} outputs vs MLX {}",
            case.name,
            got.len(),
            case.outputs.len()
        ));
    }
    got.iter()
        .zip(&case.outputs)
        .find_map(|(g, e)| compare(case, g, e).err())
}

/// Compares nuvec's output with MLX's; `Err` describes the first mismatch.
pub fn compare(case: &Case, got: &Tensor, expected: &Tensor) -> std::result::Result<(), String> {
    let ctx = format!("{} ({}{:?})", case.name, case.op, case.params);
    if got.dtype != expected.dtype {
        return Err(format!(
            "{ctx}: dtype {} vs MLX {}",
            got.dtype, expected.dtype
        ));
    }
    if got.shape != expected.shape {
        return Err(format!(
            "{ctx}: shape {:?} vs MLX {:?}",
            got.shape, expected.shape
        ));
    }
    if let (Some(g), Some(e)) = (int_values(got), int_values(expected)) {
        return match g.iter().zip(&e).position(|(a, b)| a != b) {
            Some(i) => Err(format!("{ctx}: element {i}: {} vs MLX {}", g[i], e[i])),
            None => Ok(()),
        };
    }
    for (i, (g, e)) in values(got).iter().zip(values(expected)).enumerate() {
        let ok = close(g.0, e.0, tolerance(case, expected, e.0))
            && close(g.1, e.1, tolerance(case, expected, e.1));
        if !ok {
            return Err(format!("{ctx}: element {i}: {g:?} vs MLX {e:?}"));
        }
    }
    Ok(())
}
