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

/// The fixtures recorded on MLX's GPU (the floating-point quantization modes, whose
/// reference is MLX's Metal kernels), compared bit for bit by [`check_exact`].
#[allow(dead_code)]
pub fn load_gpu() -> (String, Vec<Case>) {
    parse(include_bytes!("../fixtures/mlx_gpu.bin"))
}

/// Runs `case` and compares every output with MLX's bit for bit (dtype, shape and bytes;
/// any NaN equals any NaN); `Some` describes the first difference.
#[allow(dead_code)]
pub fn check_exact(case: &Case) -> Option<String> {
    let ran =
        run(case).and_then(|arrays| arrays.iter().map(from_array).collect::<Result<Vec<_>>>());
    let got = match ran {
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
    for (g, e) in got.iter().zip(&case.outputs) {
        if g.dtype != e.dtype || g.shape != e.shape {
            return Some(format!(
                "{}: {} {:?} vs MLX {} {:?}",
                case.name, g.dtype, g.shape, e.dtype, e.shape
            ));
        }
        let size = e.dtype.size();
        let nan = |b: &[u8]| match e.dtype {
            DType::F32 => f32::from_le_bytes(b.try_into().unwrap()).is_nan(),
            DType::F16 => f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_nan(),
            DType::BF16 => bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_nan(),
            _ => false,
        };
        let differs = g
            .data
            .chunks_exact(size)
            .zip(e.data.chunks_exact(size))
            .position(|(a, b)| a != b && !(nan(a) && nan(b)));
        if let Some(i) = differs {
            let bytes = |t: &Tensor| t.data[i * size..(i + 1) * size].to_vec();
            return Some(format!(
                "{} ({}{:?}): element {i}: bytes {:02x?} vs MLX {:02x?}",
                case.name,
                case.op,
                case.params,
                bytes(g),
                bytes(e)
            ));
        }
    }
    None
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
    let s = &t.shape[..];
    match t.dtype {
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
    }
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
        shape: a.shape().iter().map(|&d| d as usize).collect(),
        data,
    })
}

// ── Operation dispatch (mirrors `apply` in the recorder) ────────────────────

fn ints(s: &str) -> Vec<i32> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(|x| x.parse().unwrap()).collect()
    }
}

fn usizes(s: &str) -> Vec<usize> {
    ints(s).into_iter().map(|x| x as usize).collect()
}

/// Reduction axes: the recorder writes no axes for "every axis".
fn reduce_axes(s: &str, x: &Array) -> Vec<i32> {
    if s.is_empty() {
        (0..x.ndim() as i32).collect()
    } else {
        ints(s)
    }
}

fn slice_args(spec: &str) -> Vec<graph::SliceArg> {
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
            graph::SliceArg::new(opt(f[0]), opt(f[1]), f[2].parse().unwrap())
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

/// A recorded number: an integer when it parses as one, otherwise a float.
fn number(s: &str) -> Scalar {
    s.parse::<i128>()
        .map_or_else(|_| Scalar::Float(s.parse().unwrap()), Scalar::Int)
}

/// MLX's random functions, from `key(seed)` or the seeded per-thread sequence.
fn random_case(case: &Case) -> Array {
    let p = |k: &str| case.params.get(k).map(String::as_str).unwrap_or("");
    let f = |k: &str| p(k).parse::<f64>().unwrap();
    let dtype = DTYPES[p("dtype").parse::<usize>().unwrap_or(11)];
    let shape = usizes(p("shape"));
    let seed: u64 = p("seed").parse().unwrap();
    let key = if p("seq") == "1" {
        random::seed(seed);
        for _ in 0..p("skip").parse::<usize>().unwrap_or(0) {
            let _ = random::uniform(0.0, 1.0, &shape[..], DType::F32, None);
        }
        None
    } else {
        Some(random::key(seed))
    };
    let opt = |k: &str| (!p(k).is_empty()).then(|| f(k));
    let shape = &shape[..];
    match p("fn") {
        "bits" => random::bits(shape, p("width").parse().unwrap(), key.as_ref()),
        "split" => random::split(key.as_ref().unwrap(), p("num").parse().unwrap()),
        "uniform" => random::uniform(f("low"), f("high"), shape, dtype, key.as_ref()),
        "normal" => random::normal(
            shape,
            dtype,
            opt("loc").unwrap_or(0.0),
            opt("scale").unwrap_or(1.0),
            key.as_ref(),
        ),
        "gumbel" => random::gumbel(shape, dtype, key.as_ref()),
        "categorical" => random::categorical(
            &to_array(&case.inputs[0]),
            p("axis").parse::<i32>().unwrap(),
            key.as_ref(),
        ),
        "randint" => random::randint(
            p("low").parse().unwrap(),
            p("high").parse().unwrap(),
            shape,
            dtype,
            key.as_ref(),
        ),
        "bernoulli" => random::bernoulli(f("p"), shape, key.as_ref()),
        other => panic!("unknown random function {other}"),
    }
}

/// A reduction operation over the given axes.
type Reduction = fn(&Array, Vec<i32>) -> Array;

/// Runs the case in nuvec: one array per recorded output.
pub fn run(case: &Case) -> Result<Vec<Array>> {
    let p = |k: &str| case.params.get(k).map(String::as_str).unwrap_or("");
    let xs: Vec<Array> = case.inputs.iter().map(to_array).collect();
    // Inputs that are passed to `fn` after the primals but not differentiated.
    let consts = p("consts").parse::<usize>().unwrap_or(0);
    let f = |ps: &[Array], cs: &[Array]| {
        let mut all = ps.to_vec();
        all.extend_from_slice(cs);
        Ok(vec![apply(case, p("fn"), all)?])
    };
    match case.op.as_str() {
        "vjp" => {
            let n = xs.len() - 1 - consts;
            let cs = &xs[n..n + consts];
            Ok(vjp_flat(|ps: &[Array]| f(ps, cs), &xs[..n], &xs[n + consts..])?.1)
        }
        "jvp" => {
            let n = (xs.len() - consts) / 2;
            let cs = &xs[n..n + consts];
            Ok(jvp_flat(|ps: &[Array]| f(ps, cs), &xs[..n], &xs[n + consts..])?.1)
        }
        "einsum_path" => {
            let (path, report) = ops::einsum_path(p("subscripts"), &xs)?;
            let width = path.iter().map(Vec::len).max().unwrap_or(0);
            let rows: Vec<i32> = path
                .iter()
                .flat_map(|s| (0..width).map(move |i| s.get(i).map_or(-1, |&v| v as i32)))
                .collect();
            let nums: Vec<i64> = report
                .lines()
                .skip(1)
                .take(4)
                .map(|l| l.split(':').nth(1).unwrap().trim().parse().unwrap())
                .collect();
            Ok(vec![
                Array::from_slice(&rows, [path.len(), width]),
                Array::from_slice(&nums, [4]),
            ])
        }
        "quantize" => {
            let x = match p("pre") {
                "t" => xs[0].t(),
                "r" => xs[0].i((Ellipsis, (..).step(-1))),
                _ => xs[0].clone(),
            };
            let q = x.quantize(quant(case));
            let mut out = vec![q.weight, q.scales];
            out.extend(q.biases);
            Ok(out)
        }
        "vmap" => {
            let in_axes: Vec<Option<i32>> = p("in_axes")
                .split(',')
                .map(|a| (a != "n").then(|| a.parse().unwrap()))
                .collect();
            let out_axes: [i32; 1] = [p("out_axes").parse().unwrap()];
            vmap_flat(|ps: &[Array]| f(ps, &[]), &in_axes, &out_axes)(&xs)
        }
        "split" => {
            let axis = p("axis").parse::<i32>().unwrap();
            if p("at").is_empty() {
                Ok(xs[0].split(p("parts").parse().unwrap(), axis))
            } else {
                Ok(xs[0].split_at(&ints(p("at")), axis))
            }
        }
        op if op.starts_with("linalg_") => linalg_case(&op[7..], case, &xs),
        "divmod" => Ok(ops::divmod(&xs[0], &xs[1]).to_vec()),
        "unstack" => Ok(ops::unstack(&xs[0], p("axis").parse().unwrap())),
        "broadcast_arrays" => Ok(ops::broadcast_arrays(&xs)),
        "meshgrid" => Ok(ops::meshgrid(&xs, p("sparse") == "1", p("indexing"))),
        "unique" => {
            let u = ops::unique(&xs[0], p("size").parse().unwrap(), p("fill").parse().ok());
            Ok(vec![u.values, u.index, u.inverse, u.counts])
        }
        "minmax" => {
            let axes = if p("axes") == "all" {
                vec![0, 1, 2][..xs[0].ndim()].to_vec()
            } else {
                ints(p("axes"))
            };
            Ok(ops::minmax(&xs[0], &axes[..], p("keepdims") == "1").to_vec())
        }
        op => Ok(vec![apply(case, op, xs)?]),
    }
}

/// The case's quantization: `mode` (default affine) with its defaults, overridden by
/// `group_size` and `bits` when given.
fn quant(case: &Case) -> Quant {
    let p = |k: &str| case.params.get(k).map(String::as_str).unwrap_or("");
    let mode = match p("mode") {
        "" => QuantMode::Affine,
        m => m.parse().unwrap(),
    };
    let mut q = Quant::new(mode);
    if let Ok(g) = p("group_size").parse() {
        q = q.group_size(g);
    }
    if let Ok(b) = p("bits").parse() {
        q = q.bits(b);
    }
    q
}

/// Runs operation `op` with the case's parameters on `xs`.
fn apply(case: &Case, op: &str, mut xs: Vec<Array>) -> Result<Array> {
    let p = |k: &str| case.params.get(k).map(String::as_str).unwrap_or("");
    if let Some(x) = xs.first_mut() {
        *x = match p("pre") {
            "t" => x.t(),
            "r" => x.i((Ellipsis, (..).step(-1))),
            _ => x.clone(),
        };
    }
    let dtype = || DTYPES[p("dtype").parse::<usize>().unwrap()];
    let keep = p("keepdims") == "1";
    let t = || p("transpose") == "1";
    let q = |w: &Array, s: &Array, b: Option<&Array>| {
        Quantized::new(w.clone(), s.clone(), b.cloned(), quant(case))
    };
    // An optional single axis: none means the flattened array.
    let axis = || (!p("axis").is_empty()).then(|| p("axis").parse::<i32>().unwrap());
    if op == "matmul" {
        return Ok(xs[0].matmul(&xs[1]));
    }
    type S = ScalarOrArray;
    let binary: Option<fn(S, S) -> Array> = match op {
        "add" => Some(ops::add),
        "subtract" => Some(ops::subtract),
        "multiply" => Some(ops::multiply),
        "divide" => Some(ops::divide),
        "maximum" => Some(ops::maximum),
        "minimum" => Some(ops::minimum),
        "logaddexp" => Some(ops::logaddexp),
        "floor_divide" => Some(ops::floor_divide),
        "power" => Some(ops::power),
        "equal" => Some(ops::equal),
        "not_equal" => Some(ops::not_equal),
        "less" => Some(ops::less),
        "less_equal" => Some(ops::less_equal),
        "greater" => Some(ops::greater),
        "greater_equal" => Some(ops::greater_equal),
        "arctan2" => Some(ops::arctan2),
        "remainder" => Some(ops::remainder),
        "bitwise_and" => Some(ops::bitwise_and),
        "bitwise_or" => Some(ops::bitwise_or),
        "bitwise_xor" => Some(ops::bitwise_xor),
        "left_shift" => Some(ops::left_shift),
        "right_shift" => Some(ops::right_shift),
        _ => None,
    };
    if let Some(f) = binary {
        let array = S::from(&xs[0]);
        return Ok(match (parse_scalar(p("scalar")), p("side")) {
            (None, _) => f(array, S::from(&xs[1])),
            (Some(s), "left") => f(s.into(), array),
            (Some(s), _) => f(array, s.into()),
        });
    }
    let unary: Option<fn(&Array) -> Array> = match op {
        "negative" => Some(ops::negative),
        "abs" => Some(ops::abs),
        "exp" => Some(ops::exp),
        "log" => Some(ops::log),
        "sqrt" => Some(ops::sqrt),
        "sin" => Some(ops::sin),
        "cos" => Some(ops::cos),
        "tanh" => Some(ops::tanh),
        "sigmoid" => Some(ops::sigmoid),
        "conjugate" => Some(ops::conjugate),
        "erfinv" => Some(ops::erfinv),
        "erf" => Some(ops::erf),
        "expm1" => Some(ops::expm1),
        "log1p" => Some(ops::log1p),
        "log2" => Some(ops::log2),
        "log10" => Some(ops::log10),
        "tan" => Some(ops::tan),
        "sinh" => Some(ops::sinh),
        "cosh" => Some(ops::cosh),
        "arccos" => Some(ops::arccos),
        "arccosh" => Some(ops::arccosh),
        "arcsin" => Some(ops::arcsin),
        "arcsinh" => Some(ops::arcsinh),
        "arctan" => Some(ops::arctan),
        "arctanh" => Some(ops::arctanh),
        "rsqrt" => Some(ops::rsqrt),
        "floor" => Some(ops::floor),
        "ceil" => Some(ops::ceil),
        "round" => Some(|x| ops::round(x, 0)),
        "bitwise_invert" => Some(ops::bitwise_invert),
        _ => None,
    };
    if let Some(f) = unary {
        return Ok(f(&xs[0]));
    }
    let reduction: Option<Reduction> = match (op, keep) {
        ("sum", false) => Some(|x, a| x.sum(a)),
        ("sum", true) => Some(|x, a| x.sum_keepdims(a)),
        ("prod", false) => Some(|x, a| x.prod(a)),
        ("prod", true) => Some(|x, a| x.prod_keepdims(a)),
        ("max", false) => Some(|x, a| x.max(a)),
        ("max", true) => Some(|x, a| x.max_keepdims(a)),
        ("min", false) => Some(|x, a| x.min(a)),
        ("min", true) => Some(|x, a| x.min_keepdims(a)),
        ("mean", false) => Some(|x, a| x.mean(a)),
        ("mean", true) => Some(|x, a| x.mean_keepdims(a)),
        ("all", false) => Some(|x, a| x.all(a)),
        ("all", true) => Some(|x, a| x.all_keepdims(a)),
        ("any", false) => Some(|x, a| x.any(a)),
        ("any", true) => Some(|x, a| x.any_keepdims(a)),
        ("logsumexp", false) => Some(|x, a| x.logsumexp(a)),
        ("logsumexp", true) => Some(|x, a| x.logsumexp_keepdims(a)),
        _ => None,
    };
    if let Some(f) = reduction {
        return Ok(f(&xs[0], reduce_axes(p("axes"), &xs[0])));
    }
    let composed: Option<fn(&Array) -> Array> = match op {
        "square" => Some(ops::square),
        "reciprocal" => Some(ops::reciprocal),
        "degrees" => Some(ops::degrees),
        "radians" => Some(ops::radians),
        "trunc" => Some(ops::trunc),
        "real" => Some(ops::real),
        "imag" => Some(ops::imag),
        "logical_not" => Some(ops::logical_not),
        "isnan" => Some(ops::isnan),
        "isinf" => Some(ops::isinf),
        "isfinite" => Some(ops::isfinite),
        "isposinf" => Some(ops::isposinf),
        "isneginf" => Some(ops::isneginf),
        _ => None,
    };
    if let Some(f) = composed {
        return Ok(f(&xs[0]));
    }
    let close = || {
        ops::CloseOptions::new()
            .rtol(p("rtol").parse().unwrap())
            .atol(p("atol").parse().unwrap())
            .equal_nan(p("equal_nan") == "1")
    };
    let named = || dtype_named(p("dtype"));
    let axes = |k: &str| (p(k) != "all").then(|| ints(p(k)));
    Ok(match op {
        "flip" => ops::flip(&xs[0], axes("axes").as_deref()),
        "roll" => ops::roll(&xs[0], &ints(p("shift")), axes("axes").as_deref()),
        "tile" => ops::tile(&xs[0], &ints(p("reps"))),
        "moveaxis" => ops::moveaxis(
            &xs[0],
            p("source").parse().unwrap(),
            p("destination").parse().unwrap(),
        ),
        "swapaxes" => ops::swapaxes(
            &xs[0],
            p("axis1").parse().unwrap(),
            p("axis2").parse().unwrap(),
        ),
        "atleast_1d" => ops::atleast_1d(&xs[0]),
        "atleast_2d" => ops::atleast_2d(&xs[0]),
        "atleast_3d" => ops::atleast_3d(&xs[0]),
        "unflatten" => ops::unflatten(&xs[0], p("axis").parse().unwrap(), &ints(p("shape"))),
        "pad" => {
            let widths: Vec<(i32, i32)> = p("widths")
                .split(';')
                .map(|w| {
                    let v = ints(w);
                    (v[0], v[1])
                })
                .collect();
            let mode = match p("mode") {
                "edge" => ops::PadMode::Edge,
                "reflect" => ops::PadMode::Reflect,
                "symmetric" => ops::PadMode::Symmetric,
                _ => ops::PadMode::Constant,
            };
            ops::pad(
                &xs[0],
                &widths,
                mode,
                p("value").parse::<f64>().unwrap_or(0.0),
            )
        }
        "tri" => ops::tri(
            p("n").parse().unwrap(),
            p("m").parse().ok(),
            p("k").parse().unwrap(),
            named(),
        ),
        "eye" => ops::eye(
            p("n").parse().unwrap(),
            p("m").parse().ok(),
            p("k").parse().unwrap(),
            named(),
        ),
        "identity" => ops::identity(p("n").parse().unwrap(), named()),
        "tril" => ops::tril(&xs[0], p("k").parse().unwrap()),
        "triu" => ops::triu(&xs[0], p("k").parse().unwrap()),
        "diag" => ops::diag(&xs[0], p("k").parse().unwrap()),
        "diagonal" | "trace" => {
            let (o, a1, a2) = (
                p("offset").parse().unwrap(),
                p("axis1").parse().unwrap(),
                p("axis2").parse().unwrap(),
            );
            if op == "diagonal" {
                ops::diagonal(&xs[0], o, a1, a2)
            } else {
                ops::trace(&xs[0], o, a1, a2)
            }
        }
        "linspace" => ops::linspace(
            p("start").parse().unwrap(),
            p("stop").parse().unwrap(),
            p("num").parse().unwrap(),
            named(),
        ),
        "hanning" => ops::hanning(p("m").parse().unwrap()),
        "hamming" => ops::hamming(p("m").parse().unwrap()),
        "blackman" => ops::blackman(p("m").parse().unwrap()),
        "bartlett" => ops::bartlett(p("m").parse().unwrap()),
        "tensordot" => ops::tensordot(&xs[0], &xs[1], &ints(p("axes_a")), &ints(p("axes_b"))),
        "tensordot_n" => ops::tensordot_n(&xs[0], &xs[1], p("n").parse().unwrap()),
        "inner" => ops::inner(&xs[0], &xs[1]),
        "outer" => ops::outer(&xs[0], &xs[1]),
        "kron" => ops::kron(&xs[0], &xs[1]),
        "vecdot" => ops::vecdot(&xs[0], &xs[1], p("axis").parse().unwrap()),
        "matrix_transpose" => ops::matrix_transpose(&xs[0]),
        "einsum" => ops::einsum(p("subscripts"), &xs),
        "segmented_mm" => ops::segmented_mm(&xs[0], &xs[1], &xs[2]),
        "block_masked_mm" => {
            let masks = p("masks");
            let at = |c: char| masks.find(c).map(|i| &xs[2 + i]);
            ops::block_masked_mm(
                &xs[0],
                &xs[1],
                p("block_size").parse().unwrap(),
                at('o'),
                at('l'),
                at('r'),
            )
        }
        "qqmm" | "gather_qqmm" => {
            let mode = match p("mode") {
                "mxfp8" => nuvec_core::QuantMode::Mxfp8,
                "mxfp4" => nuvec_core::QuantMode::Mxfp4,
                _ => nuvec_core::QuantMode::Nvfp4,
            };
            let quant = nuvec_core::Quant::new(mode);
            let q = ops::quantize(&xs[1], quant);
            let w: nuvec_core::QqmmWeight<'_> = if p("q") == "1" {
                (&q).into()
            } else {
                (&xs[1], quant).into()
            };
            if op == "qqmm" {
                ops::qqmm(&xs[0], w)
            } else {
                ops::gather_qqmm(&xs[0], w, None, Some(&xs[2]), false)
            }
        }
        "hadamard_transform" => ops::hadamard_transform(
            &xs[0],
            match p("scale") {
                "" | "none" => None,
                v => Some(v.parse().unwrap()),
            },
        ),
        "diff" => ops::diff(&xs[0], p("n").parse().unwrap(), p("axis").parse().unwrap()),
        "median" => ops::median(&xs[0], axes("axes").as_deref(), p("keepdims") == "1"),
        "searchsorted" => ops::searchsorted(&xs[0], &xs[1], p("side") == "right"),
        "flatten" => ops::flatten(
            &xs[0],
            p("start").parse().unwrap(),
            p("end").parse().unwrap(),
        ),
        "matrix_norm" => {
            let ord = match p("ord") {
                "none" => ops::Norm::Default,
                "fro" => ops::Norm::Fro,
                o => ops::Norm::P(o.parse().unwrap()),
            };
            ops::matrix_norm(&xs[0], ord, axes("axes").as_deref(), p("keepdims") == "1")
        }
        "as_strided" => {
            let shape: Vec<usize> = ints(p("shape")).iter().map(|&d| d as usize).collect();
            let strides: Vec<i64> = ints(p("strides")).iter().map(|&d| i64::from(d)).collect();
            ops::as_strided(&xs[0], &shape, &strides, p("offset").parse().unwrap())
        }
        "logical_and" => ops::logical_and(&xs[0], &xs[1]),
        "logical_or" => ops::logical_or(&xs[0], &xs[1]),
        "logical_xor" => ops::logical_xor(&xs[0], &xs[1]),
        "round_decimals" => ops::round(&xs[0], p("decimals").parse().unwrap()),
        "nan_to_num" => {
            let mut v = ops::NanToNum::default();
            v.nan = p("nan").parse().unwrap_or(0.0);
            v.posinf = p("posinf").parse().ok();
            v.neginf = p("neginf").parse().ok();
            ops::nan_to_num(&xs[0], v)
        }
        "isclose" => ops::isclose(&xs[0], &xs[1], close()),
        "allclose" => ops::allclose(&xs[0], &xs[1], close()),
        "array_equal" => ops::array_equal(&xs[0], &xs[1], p("equal_nan") == "1"),
        "clip" => ops::clip(&xs[0], &xs[1], &xs[2]),
        "count_nonzero" => {
            if p("axes") == "all" {
                ops::count_nonzero(&xs[0], .., p("keepdims") == "1")
            } else {
                ops::count_nonzero(&xs[0], &ints(p("axes"))[..], p("keepdims") == "1")
            }
        }
        "where" => ops::where_(&xs[0], &xs[1], &xs[2]),
        "addmm" => xs[2].addmm(
            &xs[0],
            &xs[1],
            p("alpha").parse().unwrap(),
            p("beta").parse().unwrap(),
        ),
        "rms_norm" => fast::rms_norm(&xs[0], xs.get(1), p("eps").parse().unwrap()),
        "layer_norm" => fast::layer_norm(&xs[0], xs.get(1), xs.get(2), p("eps").parse().unwrap()),
        "rope" => fast::rope(
            &xs[0],
            p("dims").parse().unwrap(),
            p("traditional") == "1",
            p("base").parse().ok(),
            p("scale").parse().unwrap(),
            &xs[1],
            xs.get(2),
        ),
        "cross_entropy" => fast::cross_entropy(&xs[0], &xs[1]),
        "sdpa" => {
            let mut rest = xs[3..].iter();
            let mask = match p("mask") {
                "causal" => fast::AttentionMask::Causal,
                "array" => fast::AttentionMask::Array(rest.next().unwrap()),
                _ => fast::AttentionMask::None,
            };
            let sinks = rest.next();
            let mut options = fast::AttentionOptions::from(mask).sinks(sinks);
            if let Ok(w) = p("window").parse() {
                options = options.window(w);
            }
            if let Ok(cap) = p("cap").parse() {
                options = options.max_score_bytes(cap);
            }
            fast::scaled_dot_product_attention(
                &xs[0],
                &xs[1],
                &xs[2],
                p("scale").parse().unwrap(),
                &options,
            )
        }
        // The floating-point modes have no biases; `dtype` (a code) picks their output.
        "dequantize" => {
            let out = (!p("dtype").is_empty()).then(dtype);
            q(&xs[0], &xs[1], xs.get(2)).dequantize(out)
        }
        "quantized_matmul" => xs[0].quantized_matmul(&q(&xs[1], &xs[2], xs.get(3)), t()),
        "quantized_matmul_xsb" => xs[0].quantized_matmul(&q(&xs[3], &xs[1], Some(&xs[2])), t()),
        "softmax" => {
            let axes = reduce_axes(p("axes"), &xs[0]);
            xs[0].softmax(axes)
        }
        "random" => random_case(case),
        "astype" => xs[0].astype(dtype()),
        "reshape" => xs[0].reshape(ints(p("shape"))),
        "transpose" if p("axes").is_empty() => xs[0].t(),
        "transpose" => xs[0].transpose(usizes(p("axes"))),
        "slice" => graph::slice(&xs[0], &slice_args(p("args")))?,
        "broadcast_to" => xs[0].broadcast_to(usizes(p("shape"))),
        "expand_dims" => xs[0].expand_dims(p("axis").parse::<i32>().unwrap()),
        "conv2d" | "conv3d" => {
            let v = ints(p("stride"));
            let pd = ints(p("padding"));
            let dl = ints(p("dilation"));
            let g = p("groups").parse::<i32>().unwrap();
            if op == "conv2d" {
                ops::conv2d(
                    &xs[0],
                    &xs[1],
                    [v[0], v[1]],
                    [pd[0], pd[1]],
                    [dl[0], dl[1]],
                    g,
                )
            } else {
                ops::conv3d(
                    &xs[0],
                    &xs[1],
                    [v[0], v[1], v[2]],
                    [pd[0], pd[1], pd[2]],
                    [dl[0], dl[1], dl[2]],
                    g,
                )
            }
        }
        "conv_general" => ops::conv_general(
            &xs[0],
            &xs[1],
            &ops::ConvOptions::new()
                .stride(&ints(p("stride")))
                .padding_lo_hi(&ints(p("lo")), &ints(p("hi")))
                .kernel_dilation(&ints(p("kd")))
                .input_dilation(&ints(p("id")))
                .groups(p("groups").parse().unwrap())
                .flip(p("flip") == "1"),
        ),
        "conv_transpose1d" => {
            let n = |k: &str| p(k).parse::<i32>().unwrap();
            ops::conv_transpose1d(
                &xs[0],
                &xs[1],
                n("stride"),
                n("padding"),
                n("dilation"),
                n("output_padding"),
                n("groups"),
            )
        }
        "conv_transpose2d" => {
            let a = |k: &str| {
                let v = ints(p(k));
                [v[0], v[1]]
            };
            ops::conv_transpose2d(
                &xs[0],
                &xs[1],
                a("stride"),
                a("padding"),
                a("dilation"),
                a("output_padding"),
                p("groups").parse().unwrap(),
            )
        }
        "conv_transpose3d" => {
            let a = |k: &str| {
                let v = ints(p(k));
                [v[0], v[1], v[2]]
            };
            ops::conv_transpose3d(
                &xs[0],
                &xs[1],
                a("stride"),
                a("padding"),
                a("dilation"),
                a("output_padding"),
                p("groups").parse().unwrap(),
            )
        }
        "convolve" => ops::convolve(
            &xs[0],
            &xs[1],
            match p("mode") {
                "same" => ops::ConvolveMode::Same,
                "valid" => ops::ConvolveMode::Valid,
                _ => ops::ConvolveMode::Full,
            },
        ),
        "conv1d" => {
            let n = |k: &str| p(k).parse::<i32>().unwrap();
            xs[0].conv1d(
                &xs[1],
                n("stride"),
                n("padding"),
                n("dilation"),
                n("groups"),
            )
        }
        "gather_mm" => {
            // Inputs: a, b, then the indices that `idx` names ("l" lhs, "r" rhs).
            let mut rest = xs[2..].iter();
            let lhs = p("idx").contains('l').then(|| rest.next().unwrap());
            let rhs = p("idx").contains('r').then(|| rest.next().unwrap());
            let b = match p("tb") {
                "1" => xs[1].transpose([0, 2, 1]),
                _ => xs[1].clone(),
            };
            xs[0].gather_mm(&b, lhs, rhs, p("sorted") == "1")
        }
        "to_fp8" => xs[0].to_fp8(),
        "from_fp8" => xs[0].from_fp8(dtype()),
        "view" => xs[0].view(dtype()),
        "gather_qmm" => {
            // Inputs: x, w, scales, a global scale if `gs`, then the indices `idx` names.
            let mut rest = xs[3..].iter();
            let gs = (p("gs") == "1").then(|| rest.next().unwrap());
            let lhs = p("idx").contains('l').then(|| rest.next().unwrap());
            let rhs = p("idx").contains('r').then(|| rest.next().unwrap());
            let mut options = GatherQmm::default()
                .transpose(t())
                .sorted_indices(p("sorted") == "1");
            if let Some(g) = gs {
                options = options.global_scale(g);
            }
            xs[0].gather_qmm(&q(&xs[1], &xs[2], None), lhs, rhs, options)
        }
        "repeat" => {
            let repeats = p("repeats").parse().unwrap();
            match axis() {
                Some(a) => xs[0].repeat(repeats, a),
                None => xs[0].repeat(repeats, ..),
            }
        }
        "contiguous" if p("pre") == "t" => xs[0].t().contiguous(),
        "contiguous" => xs[0].contiguous(),
        "squeeze" if p("axes").is_empty() => xs[0].squeeze(..),
        "squeeze" => xs[0].squeeze(ints(p("axes"))),
        "sort" => xs[0].sort(p("axis").parse::<i32>().unwrap()),
        "argsort" => xs[0].argsort(p("axis").parse::<i32>().unwrap()),
        "topk" => {
            let axis = p("axis").parse::<i32>().unwrap();
            xs[0].topk(p("k").parse().unwrap(), axis).sort(axis)
        }
        "cumsum" | "cumprod" | "cummax" | "cummin" | "logcumsumexp" => {
            let options = ScanOptions {
                reverse: p("reverse") == "1",
                exclusive: p("inclusive") == "0",
            };
            let x = &xs[0];
            macro_rules! scan {
                ($ax:expr) => {
                    match op {
                        "cumsum" => x.cumsum_with($ax, options),
                        "cumprod" => x.cumprod_with($ax, options),
                        "cummax" => x.cummax_with($ax, options),
                        "cummin" => x.cummin_with($ax, options),
                        _ => x.logcumsumexp_with($ax, options),
                    }
                };
            }
            match axis() {
                Some(a) => scan!(a),
                None => scan!(..),
            }
        }
        "take" => match axis() {
            Some(a) => xs[0].take(&xs[1], a),
            None => xs[0].take(&xs[1], ..),
        },
        "take_along_axis" => xs[0].take_along_axis(&xs[1], p("axis").parse::<i32>().unwrap()),
        "put_along_axis" => {
            let (i, v) = if p("order") == "avi" { (2, 1) } else { (1, 2) };
            xs[0].put_along_axis(&xs[i], &xs[v], p("axis").parse::<i32>().unwrap())
        }
        "concatenate" => ops::concatenate(&xs, p("axis").parse::<i32>().unwrap()),
        "stack" => ops::stack(&xs, p("axis").parse::<i32>().unwrap()),
        "argmax" | "argmin" => {
            let x = &xs[0];
            match (op, axis(), keep) {
                ("argmax", Some(a), false) => x.argmax(a),
                ("argmax", Some(a), true) => x.argmax_keepdims(a),
                ("argmax", None, false) => x.argmax(..),
                ("argmax", None, true) => x.argmax_keepdims(..),
                (_, Some(a), false) => x.argmin(a),
                (_, Some(a), true) => x.argmin_keepdims(a),
                (_, None, false) => x.argmin(..),
                (_, None, true) => x.argmin_keepdims(..),
            }
        }
        "arange" => {
            ops::arange(number(p("start")), number(p("stop")), number(p("step"))).astype(dtype())
        }
        "zeros" => ops::zeros(usizes(p("shape")), dtype()),
        "ones" => ops::ones(usizes(p("shape")), dtype()),
        other => panic!("unknown fixture op {other}"),
    })
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
        // Normal samples go through erfinv's logarithm, whose last bits differ between
        // implementations; bits and uniform samples are exact.
        // Logarithms (normal's erfinv, gumbel's −log(−log u)) differ in the last bits.
        "random" if matches!(case.params["fn"].as_str(), "normal" | "gumbel") => {
            8.0 * eps * e.abs().max(1.0)
        }
        "erfinv" => 8.0 * eps * e.abs().max(f64::MIN_POSITIVE),
        // MLX's CPU erf is Abramowitz and Stegun's 7.1.26 (1.5e-7, plus its float32
        // evaluation); nuvec's is faithfully rounded.
        "erf" => 2.5e-7 + 4.0 * eps * e.abs(),
        // Windows: float32 cosines of MLX's (its zeros are 1e-15-sized).
        "hanning" | "hamming" | "blackman" | "bartlett" => 4.0 * eps,
        // Butterflies: each level's rounding is relative to that level's magnitude, at most
        // `n · max|x|` (before the scale); MLX rounds half types after every radix pass.
        "hadamard_transform" => {
            let x = &case.inputs[0];
            let n = *x.shape.last().unwrap_or(&1) as f64;
            let max = values(x).iter().fold(0f64, |m, &(re, _)| m.max(re.abs()));
            let scale = case
                .params
                .get("scale")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(1.0 / n.sqrt());
            (n.log2() + 30.0) * eps * scale * n * max.max(1.0)
        }
        // Chains of three or more: each product's rounding is relative to the chain over
        // absolute values (cancellation hides it from the result).
        "einsum" if case.inputs.len() > 2 => {
            let abs: Vec<Array> = case.inputs.iter().map(|t| to_array(t).abs()).collect();
            let bound = ops::einsum(&case.params["subscripts"], &abs)
                .max(..)
                .astype(nuvec_core::DType::F64)
                .item::<f64>()
                .unwrap_or(1.0);
            let k: usize = case
                .inputs
                .iter()
                .map(|t| t.shape.iter().product::<usize>())
                .max()
                .unwrap_or(1);
            (k as f64 + 4.0) * eps * bound.max(1.0)
        }
        // Contractions in another order than MLX's.
        "einsum" | "tensordot" | "tensordot_n" | "inner" | "vecdot" | "trace" | "segmented_mm"
        | "block_masked_mm" => {
            let k = case
                .inputs
                .iter()
                .map(|t| t.shape.iter().product::<usize>())
                .max()
                .unwrap_or(1);
            (k as f64 + 4.0) * eps * e.abs().max(1.0)
        }
        "linspace" => 2.0 * eps * e.abs().max(1.0),
        // Decompositions in another order than LAPACK's: relative to the input's size
        // and magnitude (well-conditioned inputs).
        op if op.starts_with("linalg_") => {
            let n = case.inputs[0].shape.iter().rev().take(2).product::<usize>() as f64;
            let scale = values(&case.inputs[0])
                .iter()
                .fold(1.0f64, |m, &(re, _)| m.max(re.abs()));
            4.0 * (n + 4.0) * eps * scale.max(e.abs())
        }
        "matrix_norm" => {
            let k = case.inputs[0].shape.iter().product::<usize>() as f64;
            (k + 4.0) * eps * e.abs().max(1.0)
        }
        "logsumexp" | "softmax" => 8.0 * eps * e.abs().max(1.0),
        // Statistics in float32, then rounded; kernels and fallbacks round at different
        // points.
        "rms_norm" | "layer_norm" => 16.0 * eps * e.abs().max(1.0),
        // Sines and cosines of the rotation angles differ in the last bits between GPU
        // generations.
        "rope" => 64.0 * eps * e.abs().max(1.0),
        // Two products and a softmax; kernels accumulate in float32, MLX's CPU fallback in
        // the input type.
        "sdpa" => {
            let k = case.inputs[1].shape[2] as f64;
            4.0 * (k + 8.0) * eps * e.abs().max(1.0)
        }
        // A float32 loss that MLX's fallback computes in the logits' (half) type.
        "cross_entropy" => 16.0 * epsilon(case.inputs[0].dtype) * e.abs().max(1.0),
        // Derivatives chain several rounded operations, accumulated over at most every
        // input element; logcumsumexp's subtracts nearly equal exponentials.
        "vjp" | "jvp" => {
            let n = case
                .inputs
                .iter()
                .map(|t| t.shape.iter().product::<usize>())
                .max();
            // Quantized products carry codes up to 2^bits − 1 into the scales' derivative.
            let cancels = match case.params["fn"].as_str() {
                "logcumsumexp" => 16.0,
                f if f.starts_with("quantized_matmul") => {
                    let bits: i32 = case.params["bits"].parse().unwrap();
                    f64::from((1 << bits) - 1)
                }
                _ => 1.0,
            };
            cancels * (n.unwrap_or(1) as f64 + 8.0) * eps * e.abs().max(1.0)
        }
        // Sums of `K · C_in / groups` products, in another order than MLX's kernels: as a
        // product, bounded by the operands' magnitudes (the sum may cancel).
        "conv1d" | "conv2d" | "conv3d" | "conv_general" | "conv_transpose1d"
        | "conv_transpose2d" | "conv_transpose3d" | "convolve" => {
            let bound = |t: &Tensor| values(t).iter().fold(1.0f64, |m, &(re, _)| m.max(re.abs()));
            let w = &case.inputs[1].shape;
            let k = w[1..].iter().product::<usize>() as f64;
            4.0 * (k + 1.0) * eps * bound(&case.inputs[0]) * bound(&case.inputs[1])
        }
        // Each step rounds in MLX (float32), once in nuvec (f64): the error grows with
        // the scanned length.
        // Parallel prefix scans (GPU) round in another order than MLX's sequential CPU
        // scan: as reductions, the error grows with the scanned length.
        "cumsum" | "cumprod" => {
            let n = case.inputs[0].shape.iter().product::<usize>() as f64;
            (n + 1.0) * eps * e.abs().max(1.0)
        }
        "logcumsumexp" => {
            let n = case.inputs[0].shape.iter().max().copied().unwrap_or(1) as f64;
            4.0 * (n + 1.0) * eps * e.abs().max(1.0)
        }
        "exp" | "log" | "sqrt" | "sin" | "cos" | "tanh" | "sigmoid" | "power" | "expm1"
        | "log1p" | "log2" | "log10" | "tan" | "sinh" | "cosh" | "arccos" | "arccosh"
        | "arcsin" | "arcsinh" | "arctan" | "arctanh" | "rsqrt" | "arctan2" => {
            4.0 * eps * e.abs().max(f64::MIN_POSITIVE)
        }
        // A sum of the larger operand and log1p(exp(·)): absolute error near zero.
        "logaddexp" => 4.0 * eps * e.abs().max(1.0),
        // Complex division is not exactly rounded (and MLX's C++ build may contract it to
        // fused multiply-adds).
        "divide" if expected.dtype == DType::Complex64 => {
            4.0 * eps * e.abs().max(f64::MIN_POSITIVE)
        }
        "sum" | "prod" | "mean" => {
            let n = case.inputs[0].data.len() / expected.data.len().max(1) * expected.dtype.size()
                / case.inputs[0].dtype.size().max(1);
            (n.max(1) as f64 + 1.0) * eps * e.abs().max(1.0)
        }
        // Affine on the GPU: MLX's Metal kernel (one rounding of `scale · code + bias`,
        // pinned bit for bit by mlx_gpu.bin) against MLX's CPU (two): within one rounding of
        // the larger term. Exact on the CPU.
        "dequantize"
            if !case.params.contains_key("mode")
                && nuvec_core::default_stream().device.kind != nuvec_core::DeviceKind::Cpu =>
        {
            let bound = |t: &Tensor| values(t).iter().fold(0.0f64, |m, &(re, _)| m.max(re.abs()));
            let bits: i32 = case.params["bits"].parse().unwrap();
            2.0 * eps
                * (bound(&case.inputs[1]) * f64::from((1 << bits) - 1) + bound(&case.inputs[2]))
        }
        // Bounded by K · max|x| · max|dequantized w|, with codes up to 2^bits - 1.
        "quantized_matmul" => {
            let bound = |t: &Tensor| values(t).iter().fold(1.0f64, |m, &(re, _)| m.max(re.abs()));
            let bits: i32 = case.params["bits"].parse().unwrap();
            let k = case.inputs[0].shape.last().copied().unwrap_or(1) as f64;
            let w = bound(&case.inputs[2]) * f64::from((1 << bits) - 1) + bound(&case.inputs[3]);
            4.0 * (k + 1.0) * eps * bound(&case.inputs[0]) * w
        }
        "matmul" | "addmm" | "gather_mm" => {
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
    let ran =
        run(case).and_then(|arrays| arrays.iter().map(from_array).collect::<Result<Vec<_>>>());
    let got = match ran {
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
    // Products' tolerances bound the inputs, not the element: computed once.
    let op = match case.op.as_str() {
        "vmap" => case.params["fn"].as_str(),
        op => op,
    };
    let fixed = matches!(op, "quantized_matmul" | "matmul" | "addmm" | "gather_mm")
        .then(|| tolerance(case, expected, 0.0));
    let tol = |e: f64| fixed.unwrap_or_else(|| tolerance(case, expected, e));
    for (i, (g, e)) in values(got).iter().zip(values(expected)).enumerate() {
        let ok = close(g.0, e.0, tol(e.0)) && close(g.1, e.1, tol(e.1));
        if !ok {
            return Err(format!("{ctx}: element {i}: {g:?} vs MLX {e:?}"));
        }
    }
    Ok(())
}

/// A dtype by MLX's name (`float32`, `int8`, …).
fn dtype_named(name: &str) -> DType {
    match name {
        "bool" => DType::Bool,
        "uint8" => DType::U8,
        "uint16" => DType::U16,
        "uint32" => DType::U32,
        "uint64" => DType::U64,
        "int8" => DType::I8,
        "int16" => DType::I16,
        "int32" => DType::I32,
        "int64" => DType::I64,
        "float16" => DType::F16,
        "bfloat16" => DType::BF16,
        "float64" => DType::F64,
        "complex64" => DType::Complex64,
        _ => DType::F32,
    }
}

/// Per column, the sign of its largest-magnitude entry (decompositions' free signs), as in
/// `scripts/mlx_fixtures.py`.
fn signs(cols: &Array) -> Array {
    let idx = cols.abs().argmax_keepdims(-2);
    ops::take_along_axis(cols, &idx, -2).sign()
}

/// nuvec's linalg outputs, normalised as the recorder normalises MLX's.
fn linalg_case(f: &str, case: &Case, xs: &[Array]) -> Result<Vec<Array>> {
    let p = |k: &str| case.params.get(k).map(String::as_str).unwrap_or("");
    let up = p("upper") == "1";
    let t = |x: &Array| ops::swapaxes(x, -1, -2);
    Ok(match f {
        "qr" => {
            let [q, r] = linalg::qr(&xs[0]);
            let s = signs(&q);
            vec![q.multiply(&s), r.multiply(t(&s))]
        }
        "svd" => {
            let [u, s, vt] = linalg::svd(&xs[0]);
            let k = s.shape()[s.ndim() - 1];
            let u = u.i((Ellipsis, ..k));
            let vt = vt.i((Ellipsis, ..k, ..));
            let g = signs(&u);
            vec![u.multiply(&g), s, vt.multiply(t(&g))]
        }
        "svd_values" => vec![linalg::svd_values(&xs[0])],
        "eigh" => {
            let [w, v] = linalg::eigh(&xs[0], up);
            let g = signs(&v);
            vec![w, v.multiply(&g)]
        }
        "eigvalsh" => vec![linalg::eigvalsh(&xs[0], up)],
        "eigvals" => {
            let w = linalg::eigvals(&xs[0]);
            vec![ops::real(&w).sort(-1), ops::imag(&w).sort(-1)]
        }
        "lu" => linalg::lu(&xs[0]).to_vec(),
        "lu_factor" => linalg::lu_factor(&xs[0]).to_vec(),
        "slogdet" => linalg::slogdet(&xs[0]).to_vec(),
        "det" => vec![linalg::det(&xs[0])],
        "inv" => vec![linalg::inv(&xs[0])],
        "pinv" => vec![linalg::pinv(&xs[0])],
        "cholesky" => vec![linalg::cholesky(&xs[0], up)],
        "tri_inv" => vec![linalg::tri_inv(&xs[0], up)],
        "cholesky_inv" => vec![linalg::cholesky_inv(&xs[0], up)],
        "solve" => vec![linalg::solve(&xs[0], &xs[1])],
        "solve_triangular" => vec![linalg::solve_triangular(&xs[0], &xs[1], up)],
        "cross" => vec![linalg::cross(&xs[0], &xs[1], p("axis").parse().unwrap())],
        "norm" => {
            let ord = match p("ord") {
                "nuc" => ops::Norm::Nuc,
                o => ops::Norm::P(o.parse().unwrap()),
            };
            vec![linalg::norm(&xs[0], ord, None, false)]
        }
        other => panic!("linalg case {other}"),
    })
}
