//! Fused element-wise computations: one kernel for a graph of element-wise operations
//! (MLX's `Compiled` primitive), as built by `compile`.
//!
//! On the GPU the graph becomes one generated Metal kernel that reads each input once and
//! writes only the result. Everywhere else (and for the transforms) it expands into the
//! operations it fuses.

use std::fmt::Write as _;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use crate::array::Data;
use crate::kernels::Operand;
use crate::primitive::{BinaryOp, Primitive, UnaryOp};
use crate::{Array, Category, DType, Error, Result};

/// An operand of a fused step: an input of the fused computation or an earlier step's
/// result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedRef {
    /// The input at this position.
    Input(usize),
    /// The result of the step at this position.
    Step(usize),
}

/// A graph of element-wise operations evaluated as one kernel, with one or more of its
/// steps as results. Cheap to clone; equal graphs compare and hash equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fused(Arc<Graph>);

#[derive(Debug, PartialEq, Eq, Hash)]
struct Graph {
    inputs: Vec<DType>,
    steps: Vec<(Primitive, Vec<FusedRef>)>,
    /// The steps whose results are outputs.
    outputs: Vec<usize>,
    /// The element type of each step's result.
    dtypes: Vec<DType>,
    /// A hash of the above, naming the generated kernels.
    id: u64,
}

/// How a generated kernel reads one input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Load {
    /// One element, broadcast.
    Scalar,
    /// Laid out like the output.
    Flat,
    /// Through shape and strides (the `k`-th strided input).
    Strided(usize),
}

fn computes(d: DType) -> bool {
    !matches!(d, DType::F64 | DType::Complex64)
}

fn is_float(d: DType) -> bool {
    matches!(d, DType::F16 | DType::BF16 | DType::F32)
}

impl Fused {
    /// Whether `primitive` applied to inputs of these dtypes can be part of a fused
    /// kernel: unary, binary, select and type conversions over the GPU's element types
    /// (all but float64 and complex64).
    pub fn supports(primitive: &Primitive, dtypes: &[DType]) -> bool {
        if !dtypes.iter().copied().all(computes) {
            return false;
        }
        match (primitive, dtypes) {
            (Primitive::Unary(op), &[d]) => match op {
                UnaryOp::Negative => d != DType::Bool,
                UnaryOp::Abs => true,
                UnaryOp::Conjugate => false,
                _ => is_float(d),
            },
            (Primitive::Binary(op), &[a, b]) => {
                a == b && (!matches!(op, BinaryOp::Divide | BinaryOp::LogAddExp) || is_float(a))
            }
            (Primitive::Select, &[c, a, b]) => c == DType::Bool && a == b,
            (Primitive::AsType(to), &[_]) => computes(*to),
            _ => false,
        }
    }

    /// The fused computation of `steps` over inputs of the given dtypes, with the results
    /// of the steps at positions `outputs` as its outputs. Each step is a primitive
    /// [`Fused::supports`] and its operands, which refer to inputs or earlier steps.
    pub fn new(
        inputs: &[DType],
        steps: Vec<(Primitive, Vec<FusedRef>)>,
        outputs: &[usize],
    ) -> Result<Self> {
        let invalid = |what: String| Err(Error::InvalidArgument(format!("fused: {what}")));
        if steps.is_empty() || outputs.is_empty() {
            return invalid("no steps or no outputs".into());
        }
        if let Some(o) = outputs.iter().find(|&&o| o >= steps.len()) {
            return invalid(format!("output step {o} of {}", steps.len()));
        }
        let mut dtypes: Vec<DType> = Vec::with_capacity(steps.len());
        for (k, (primitive, args)) in steps.iter().enumerate() {
            let mut ds = Vec::with_capacity(args.len());
            for arg in args {
                ds.push(match *arg {
                    FusedRef::Input(i) if i < inputs.len() => inputs[i],
                    FusedRef::Step(j) if j < k => dtypes[j],
                    _ => return invalid(format!("step {k} refers to {arg:?}")),
                });
            }
            if !Self::supports(primitive, &ds) {
                return invalid(format!("{} on {ds:?} cannot be fused", primitive.name()));
            }
            dtypes.push(match primitive {
                Primitive::Binary(op) if op.is_comparison() => DType::Bool,
                Primitive::Select => ds[1],
                Primitive::AsType(to) => *to,
                _ => ds[0],
            });
        }
        let mut graph = Graph {
            inputs: inputs.to_vec(),
            steps,
            outputs: outputs.to_vec(),
            dtypes,
            id: 0,
        };
        let mut h = DefaultHasher::new();
        (&graph.inputs, &graph.steps, &graph.outputs).hash(&mut h);
        graph.id = h.finish();
        Ok(Self(Arc::new(graph)))
    }

    /// The input dtypes.
    pub fn inputs(&self) -> &[DType] {
        &self.0.inputs
    }

    /// The number of fused operations.
    pub fn len(&self) -> usize {
        self.0.steps.len()
    }

    /// Always false: a fused computation has at least one step.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The outputs' dtypes.
    pub fn dtypes(&self) -> Vec<DType> {
        self.0.outputs.iter().map(|&o| self.0.dtypes[o]).collect()
    }

    /// The same computation recorded as ordinary operations.
    pub(crate) fn expand(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let mut results: Vec<Array> = Vec::with_capacity(self.0.steps.len());
        for (primitive, args) in &self.0.steps {
            let args = args
                .iter()
                .map(|r| match *r {
                    FusedRef::Input(i) => inputs[i].clone(),
                    FusedRef::Step(j) => results[j].clone(),
                })
                .collect();
            results.push(Array::from_op(primitive.clone(), args)?.swap_remove(0));
        }
        Ok(self.0.outputs.iter().map(|&o| results[o].clone()).collect())
    }

    /// Evaluates the expanded operations on the CPU, for work the GPU does not take.
    pub(crate) fn eval_cpu(&self, operands: &[Operand<'_>]) -> Result<Vec<Data>> {
        crate::fast::imp::run_on_cpu(operands, |leaves| self.expand(leaves))
    }

    /// Reverse mode through the expanded operations, for the wanted float inputs.
    pub(crate) fn vjp(
        &self,
        inputs: &[Array],
        cotangents: &[Array],
        wanted: &[bool],
    ) -> Result<Vec<Option<Array>>> {
        let active: Vec<usize> = (0..inputs.len())
            .filter(|&i| wanted[i] && inputs[i].dtype().category() == Category::Float)
            .collect();
        let primals: Vec<Array> = active.iter().map(|&i| inputs[i].clone()).collect();
        let f = |ps: &[Array]| self.expand(&substitute(inputs, &active, ps));
        let (_, grads) = crate::vjp_flat(f, &primals, cotangents)?;
        let mut out = vec![None; inputs.len()];
        for (g, &i) in grads.into_iter().zip(&active) {
            out[i] = Some(g);
        }
        Ok(out)
    }

    /// Forward mode through the expanded operations (non-float inputs are constants).
    pub(crate) fn jvp(&self, inputs: &[Array], tangents: &[Option<Array>]) -> Result<Vec<Array>> {
        let active: Vec<usize> = (0..inputs.len())
            .filter(|&i| tangents[i].is_some() && inputs[i].dtype().category() == Category::Float)
            .collect();
        let primals: Vec<Array> = active.iter().map(|&i| inputs[i].clone()).collect();
        let ts: Vec<Array> = active
            .iter()
            .map(|&i| tangents[i].clone().expect("active"))
            .collect();
        if active.is_empty() {
            return self
                .expand(inputs)?
                .iter()
                .map(crate::derivatives::zeros_like)
                .collect();
        }
        let f = |ps: &[Array]| self.expand(&substitute(inputs, &active, ps));
        Ok(crate::jvp_flat(f, &primals, &ts)?.1)
    }

    /// Batching rule: the expanded operations, batched.
    pub(crate) fn vmap(
        &self,
        inputs: &[Array],
        axes: &[Option<usize>],
    ) -> Result<(Vec<Array>, usize)> {
        let f = |xs: &[Array]| self.expand(xs);
        Ok((crate::transforms::batch(&f, inputs, axes)?, 0))
    }

    /// The kernel name for inputs read as `loads`, with `per_thread` elements per thread.
    pub(crate) fn kernel_name(&self, loads: &[Load], per_thread: usize) -> String {
        let mut name = format!("nuvec_fused_{:016x}_", self.0.id);
        name.extend(loads.iter().map(|l| match l {
            Load::Scalar => 's',
            Load::Flat => 'f',
            Load::Strided(_) => 'g',
        }));
        let _ = write!(name, "_n{per_thread}");
        name
    }

    /// Metal source of the kernel `name`, reading inputs as `loads`, `per_thread`
    /// consecutive elements per thread. Arguments: the inputs, the outputs, the inputs'
    /// element offsets, the (collapsed) shape, the strided inputs' strides (`ndim` each,
    /// in order), `ndim` and the element count.
    pub(crate) fn metal_source(&self, name: &str, loads: &[Load], per_thread: usize) -> String {
        let g = &*self.0;
        let (n, m) = (g.inputs.len(), g.outputs.len());
        let mut s = String::from(
            "#include \"common.h\"\nusing namespace metal;\nusing namespace nuvec;\n\n",
        );
        let _ = writeln!(s, "[[kernel]] void {name}(");
        for (i, d) in g.inputs.iter().enumerate() {
            let _ = writeln!(s, "    device const {}* in{i} [[buffer({i})]],", msl(*d));
        }
        for (k, d) in self.dtypes().into_iter().enumerate() {
            let _ = writeln!(s, "    device {}* out{k} [[buffer({})]],", msl(d), n + k);
        }
        let _ = writeln!(s, "    constant long* offs [[buffer({})]],", n + m);
        let _ = writeln!(s, "    constant int* shape [[buffer({})]],", n + m + 1);
        let _ = writeln!(s, "    constant long* strides [[buffer({})]],", n + m + 2);
        let _ = writeln!(s, "    constant int& ndim [[buffer({})]],", n + m + 3);
        let _ = writeln!(s, "    constant uint& size [[buffer({})]],", n + m + 4);
        s.push_str("    uint tid [[thread_position_in_grid]]) {\n");
        for (i, d) in g.inputs.iter().enumerate() {
            let _ = writeln!(s, "  device const {}* a{i} = in{i} + offs[{i}];", msl(*d));
        }
        let _ = writeln!(
            s,
            "  for (uint i = tid * {per_thread}; i < min(size, (tid + 1) * {per_thread}); ++i) {{"
        );
        for (i, (d, load)) in g.inputs.iter().zip(loads).enumerate() {
            let index = match load {
                Load::Scalar => "0".into(),
                Load::Flat => "i".into(),
                Load::Strided(k) => format!("loc(i, shape, strides + {k} * ndim, ndim)"),
            };
            let _ = writeln!(s, "    {} x{i} = a{i}[{index}];", msl(*d));
        }
        let arg = |r: &FusedRef| match *r {
            FusedRef::Input(i) => format!("x{i}"),
            FusedRef::Step(j) => format!("t{j}"),
        };
        let dtype = |r: &FusedRef| match *r {
            FusedRef::Input(i) => g.inputs[i],
            FusedRef::Step(j) => g.dtypes[j],
        };
        for (k, (primitive, args)) in g.steps.iter().enumerate() {
            let a: Vec<String> = args.iter().map(arg).collect();
            let expr = match primitive {
                Primitive::Unary(op) => format!("op_{}({})", unary_fn(*op), a[0]),
                Primitive::Binary(op) => binary_expr(*op, &a[0], &a[1]),
                Primitive::Select => format!("({}.v ? {} : {})", a[0], a[1], a[2]),
                Primitive::AsType(to) => {
                    format!("convert<{}, {}>({})", msl(dtype(&args[0])), msl(*to), a[0])
                }
                _ => unreachable!("validated by Fused::new"),
            };
            let _ = writeln!(s, "    {} t{k} = {expr};", msl(g.dtypes[k]));
        }
        for (k, o) in g.outputs.iter().enumerate() {
            let _ = writeln!(s, "    out{k}[i] = t{o};");
        }
        s.push_str("  }\n}\n");
        s
    }
}

/// `inputs` with the positions in `active` replaced by `values`.
fn substitute(inputs: &[Array], active: &[usize], values: &[Array]) -> Vec<Array> {
    let mut all = inputs.to_vec();
    for (v, &i) in values.iter().zip(active) {
        all[i] = v.clone();
    }
    all
}

/// The Metal type of a GPU dtype (see `common.h`).
fn msl(d: DType) -> &'static str {
    match d {
        DType::Bool => "nbool",
        DType::U8 => "uchar",
        DType::U16 => "ushort",
        DType::U32 => "uint",
        DType::U64 => "ulong",
        DType::I8 => "char",
        DType::I16 => "short",
        DType::I32 => "int",
        DType::I64 => "long",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::F32 => "float",
        DType::F64 | DType::Complex64 => unreachable!("not fused"),
    }
}

fn unary_fn(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Negative => "neg",
        UnaryOp::Abs => "abs",
        UnaryOp::Exp => "exp",
        UnaryOp::Log => "log",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Sigmoid => "sigmoid",
        UnaryOp::ErfInv => "erfinv",
        UnaryOp::Conjugate => unreachable!("not fused"),
    }
}

/// The element-wise kernels' expressions (`elementwise.metal`).
fn binary_expr(op: BinaryOp, a: &str, b: &str) -> String {
    match op {
        BinaryOp::Add => format!("op_add({a}, {b})"),
        BinaryOp::Subtract => format!("op_sub({a}, {b})"),
        BinaryOp::Multiply => format!("op_mul({a}, {b})"),
        BinaryOp::Divide => format!("op_div({a}, {b})"),
        BinaryOp::Maximum => format!("op_max({a}, {b})"),
        BinaryOp::Minimum => format!("op_min({a}, {b})"),
        BinaryOp::LogAddExp => format!("op_logaddexp({a}, {b})"),
        BinaryOp::Equal => format!("nbool{{uchar(op_eq({a}, {b}))}}"),
        BinaryOp::NotEqual => format!("nbool{{uchar(!op_eq({a}, {b}))}}"),
        BinaryOp::Less => format!("nbool{{uchar(op_lt({a}, {b}))}}"),
        BinaryOp::LessEqual => format!("nbool{{uchar(op_lt({a}, {b}) || op_eq({a}, {b}))}}"),
        BinaryOp::Greater => format!("nbool{{uchar(op_lt({b}, {a}))}}"),
        BinaryOp::GreaterEqual => format!("nbool{{uchar(op_lt({b}, {a}) || op_eq({a}, {b}))}}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::imp as ops;

    fn sigmoid_mul() -> Fused {
        // x * sigmoid(x) + y
        Fused::new(
            &[DType::F32, DType::F32],
            vec![
                (Primitive::Unary(UnaryOp::Sigmoid), vec![FusedRef::Input(0)]),
                (
                    Primitive::Binary(BinaryOp::Multiply),
                    vec![FusedRef::Input(0), FusedRef::Step(0)],
                ),
                (
                    Primitive::Binary(BinaryOp::Add),
                    vec![FusedRef::Step(1), FusedRef::Input(1)],
                ),
            ],
            &[1, 2],
        )
        .unwrap()
    }

    #[test]
    fn validates_steps() {
        let bad = |inputs: &[DType], steps| Fused::new(inputs, steps, &[0]).is_err();
        assert!(bad(&[DType::F32], vec![]));
        assert!(bad(
            &[DType::F32],
            vec![(Primitive::Unary(UnaryOp::Exp), vec![FusedRef::Step(0)])]
        ));
        assert!(bad(
            &[DType::I32],
            vec![(Primitive::Unary(UnaryOp::Exp), vec![FusedRef::Input(0)])]
        ));
        assert!(bad(
            &[DType::F64],
            vec![(Primitive::Unary(UnaryOp::Exp), vec![FusedRef::Input(0)])]
        ));
        assert!(bad(&[DType::F32], vec![(Primitive::Matmul, vec![])]));
        let f = sigmoid_mul();
        assert_eq!((f.len(), f.dtypes()), (3, vec![DType::F32; 2]));
        let one = vec![(Primitive::Unary(UnaryOp::Exp), vec![FusedRef::Input(0)])];
        assert!(
            Fused::new(&[DType::F32], one, &[1]).is_err(),
            "no such output step"
        );
        assert_eq!(f, sigmoid_mul(), "equal graphs are equal");
    }

    fn values(arrays: &[Array]) -> Vec<Vec<f32>> {
        arrays.iter().map(|a| a.to_vec::<f32>().unwrap()).collect()
    }

    /// Equal up to rounding (Miri perturbs transcendental functions by a few ulps).
    fn assert_close(a: &[Vec<f32>], b: &[Vec<f32>]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
            assert!((x - y).abs() <= 1e-5 * x.abs().max(1.0), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn evaluates_like_its_operations_on_the_cpu() {
        crate::set_default_stream(crate::Stream::default_for(crate::Device::cpu()));
        let x = Array::from_slice(&[-2.0f32, 0.5, 3.0, 1.0, -1.0, 4.0], [2, 3]);
        let y = Array::from_slice(&[1.0f32, 2.0], [2, 1]);
        let y = ops::broadcast_to(&y, &[2, 3]).unwrap();
        let xt = x.t().t();
        let fused = sigmoid_mul();
        let out = Array::from_op(Primitive::Compiled(fused.clone()), vec![xt, y.clone()]).unwrap();
        assert_eq!(out.len(), 2);
        assert_close(&values(&out), &values(&fused.expand(&[x, y]).unwrap()));
    }

    #[test]
    fn differentiates_through_its_operations() {
        crate::set_default_stream(crate::Stream::default_for(crate::Device::cpu()));
        let fused = sigmoid_mul();
        let x = Array::from_slice(&[-1.0f32, 0.0, 2.0], [3]);
        let y = Array::from_slice(&[3.0f32, 1.0, 0.0], [3]);
        let cot = Array::from_slice(&[1.0f32, 2.0, 0.5], [3]);
        let cot2 = Array::from_slice(&[-1.0f32, 0.25, 3.0], [3]);
        let run = |expanded: bool| {
            let f = |xs: &[Array]| match expanded {
                true => fused.expand(xs),
                false => Array::from_op(Primitive::Compiled(fused.clone()), xs.to_vec()),
            };
            let primals = [x.clone(), y.clone()];
            let (_, g) = crate::vjp_flat(f, &primals, &[cot.clone(), cot2.clone()]).unwrap();
            let (_, t) = crate::jvp_flat(f, &primals, &[cot.clone(), cot2.clone()]).unwrap();
            values(&[g, t].concat())
        };
        assert_close(&run(false), &run(true));
        let batched = crate::vmap_flat(
            |xs| Array::from_op(Primitive::Compiled(fused.clone()), xs.to_vec()),
            &[Some(0), None],
            &[],
        );
        let xs = ops::reshape(&x, &[3, 1]).unwrap();
        let ys = Array::from_slice(&[3.0f32], [1]);
        let out = batched(&[xs.clone(), ys.clone()]).unwrap();
        let expected = fused
            .expand(&[xs, ops::broadcast_to(&ys, &[3, 1]).unwrap()])
            .unwrap();
        assert_close(&values(&out), &values(&expected));
    }
}
