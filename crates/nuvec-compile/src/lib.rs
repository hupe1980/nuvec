//! Function compilation: tracing, graph simplification and kernel fusion.
//!
//! [`compile`] works as MLX's `mx.compile`. The first call with a given input signature
//! traces the function on placeholders and simplifies the graph:
//!
//! - identity views and conversions are removed;
//! - equal scalar constants are merged;
//! - equal operations on equal inputs are computed once (common-subexpression
//!   elimination).
//!
//! On the GPU, chains of element-wise operations are then fused into single generated
//! kernels. Later calls replay the simplified graph on the new inputs without running the
//! function again.
//!
//! ```
//! use nuvec_compile::compile;
//! use nuvec_core::{Array, Result, add, multiply};
//!
//! # fn main() -> Result<()> {
//! // x * x + 1, as one kernel on the GPU
//! let f = compile(|xs: &[Array]| Ok(vec![add(&multiply(&xs[0], &xs[0])?, 1.0)?]));
//! let x = Array::from_slice(&[1.0f32, 2.0], &[2])?;
//! assert_eq!(f(&[x])?[0].to_vec::<f32>()?, [2.0, 5.0]);
//! let y = Array::from_slice(&[3.0f32, 4.0], &[2])?;
//! assert_eq!(f(&[y])?[0].to_vec::<f32>()?, [10.0, 17.0]); // replayed, not traced again
//! # Ok(())
//! # }
//! ```
//!
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use nuvec_core::{
    Array, Complex64, DType, DeviceKind, Fused, FusedRef, Primitive, Result, Stream, bf16,
    default_stream, f16,
};

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The most operations fused into one kernel.
const MAX_FUSED_STEPS: usize = 32;
/// The most inputs and outputs of one fused kernel (Metal binds at most 31 buffers per
/// kernel; a fused kernel uses five more), as MLX's `max_compile_arrays`.
const MAX_FUSED_ARRAYS: usize = 24;

/// Returns `f` compiled: a function with the same results that traces `f` once per input
/// signature (dtypes and shapes) and default stream, and replays the simplified, fused
/// graph on later calls.
///
/// As in MLX, `f` must be pure:
///
/// - arrays it captures are constants, frozen at the first trace (so is random state
///   drawn inside it);
/// - arrays cannot be evaluated inside it.
///
/// Called on arrays being traced by a transform (`grad`, `vmap`, another `compile`), it
/// runs `f` directly, so transforms see the original operations.
pub fn compile<F>(f: F) -> impl Fn(&[Array]) -> Result<Vec<Array>>
where
    F: Fn(&[Array]) -> Result<Vec<Array>>,
{
    let cache: Mutex<HashMap<Key, Arc<Tape>>> = Mutex::default();
    move |inputs| {
        if inputs.iter().any(Array::is_tracer) {
            return f(inputs);
        }
        let key = Key::new(inputs);
        let cached = cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            .cloned();
        let tape = match cached {
            Some(tape) => tape,
            None => {
                let tape = Arc::new(Tape::trace(&f, inputs, key.stream)?);
                cache
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .insert(key, tape.clone());
                tape
            }
        };
        tape.replay(inputs)
    }
}

/// What a traced graph depends on: the inputs' dtypes and shapes and the stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    stream: Stream,
    inputs: Vec<(DType, Vec<usize>)>,
}

impl Key {
    fn new(inputs: &[Array]) -> Self {
        Self {
            stream: default_stream(),
            inputs: inputs
                .iter()
                .map(|a| (a.dtype(), a.shape().to_vec()))
                .collect(),
        }
    }
}

/// A value of the traced graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Ref {
    /// The input at this position.
    Input(usize),
    /// A constant (captured or independent of the inputs).
    Const(usize),
    /// Output `.1` of step `.0`.
    Step(usize, usize),
}

#[derive(Debug, Clone)]
struct Step {
    primitive: Primitive,
    inputs: Vec<Ref>,
    /// Dtype and shape of each output.
    outputs: Vec<(DType, Vec<usize>)>,
}

/// A simplified graph, replayed on each call.
#[derive(Debug)]
struct Tape {
    consts: Vec<Array>,
    steps: Vec<Step>,
    outputs: Vec<Ref>,
}

impl Tape {
    /// Traces `f` on placeholders like `inputs`, simplifies the graph and, on the GPU,
    /// fuses element-wise chains.
    fn trace(
        f: &dyn Fn(&[Array]) -> Result<Vec<Array>>,
        inputs: &[Array],
        stream: Stream,
    ) -> Result<Self> {
        let placeholders = inputs
            .iter()
            .map(|a| Array::placeholder(a.dtype(), a.shape()))
            .collect::<Result<Vec<_>>>()?;
        let outputs = f(&placeholders)?;
        let mut tape = Builder::new(&placeholders).build(&outputs);
        if stream.device.kind == DeviceKind::Gpu {
            tape.fuse(inputs);
        }
        Ok(tape)
    }

    fn replay(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let mut values: Vec<Vec<Array>> = Vec::with_capacity(self.steps.len());
        let get = |r: &Ref, values: &[Vec<Array>]| match *r {
            Ref::Input(i) => inputs[i].clone(),
            Ref::Const(c) => self.consts[c].clone(),
            Ref::Step(s, o) => values[s][o].clone(),
        };
        for step in &self.steps {
            let args = step.inputs.iter().map(|r| get(r, &values)).collect();
            values.push(Array::from_primitive(step.primitive.clone(), args)?);
        }
        Ok(self.outputs.iter().map(|r| get(r, &values)).collect())
    }

    fn dtype(&self, r: Ref, inputs: &[Array]) -> DType {
        match r {
            Ref::Input(i) => inputs[i].dtype(),
            Ref::Const(c) => self.consts[c].dtype(),
            Ref::Step(s, o) => self.steps[s].outputs[o].0,
        }
    }

    /// Replaces chains of element-wise steps by fused steps (MLX's `compile_fuse`). Walking
    /// back from a group's last step, an element-wise step joins the group when every step
    /// that consumes it is in the group and the kernel limits hold. A joining step that is
    /// also an output of the function becomes another output of the fused step. So nothing
    /// outside a group consumes its inner values, and the graph stays acyclic.
    fn fuse(&mut self, inputs: &[Array]) {
        let n = self.steps.len();
        let fusable: Vec<bool> = self
            .steps
            .iter()
            .map(|s| {
                let dtypes: Vec<DType> = s.inputs.iter().map(|&r| self.dtype(r, inputs)).collect();
                s.outputs.len() == 1 && Fused::supports(&s.primitive, &dtypes)
            })
            .collect();
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (k, s) in self.steps.iter().enumerate() {
            for r in &s.inputs {
                if let Ref::Step(j, _) = *r {
                    consumers[j].push(k);
                }
            }
        }
        let mut is_output = vec![false; n];
        for r in &self.outputs {
            if let Ref::Step(j, _) = *r {
                is_output[j] = true;
            }
        }
        let step_inputs = |k: usize| {
            self.steps[k].inputs.iter().filter_map(|r| match *r {
                Ref::Step(j, _) => Some(j),
                _ => None,
            })
        };
        // Per step: its group's root (last step), if fused.
        let mut group: Vec<Option<usize>> = vec![None; n];
        // Per root: the members (ascending) and the external inputs.
        let mut groups: HashMap<usize, (Vec<usize>, Vec<Ref>)> = HashMap::new();
        for root in (0..n).rev() {
            if group[root].is_some() || !fusable[root] {
                continue;
            }
            let mut members = HashSet::from([root]);
            let mut external: Vec<Ref> = Vec::new();
            let mut outputs = 1;
            let add_inputs = |k: usize, external: &mut Vec<Ref>| {
                for &r in &self.steps[k].inputs {
                    if !external.contains(&r) {
                        external.push(r);
                    }
                }
            };
            add_inputs(root, &mut external);
            // Highest first: a step is decided after every step that could consume it.
            let mut candidates: BinaryHeap<usize> = step_inputs(root).collect();
            let mut seen = HashSet::new();
            while let Some(j) = candidates.pop() {
                if !seen.insert(j)
                    || group[j].is_some()
                    || !fusable[j]
                    || members.len() >= MAX_FUSED_STEPS
                    || !consumers[j].iter().all(|c| members.contains(c))
                {
                    continue;
                }
                let new = self.steps[j]
                    .inputs
                    .iter()
                    .filter(|r| !external.contains(r))
                    .collect::<HashSet<_>>()
                    .len();
                let arrays = external.len() - 1 + new + outputs + usize::from(is_output[j]);
                if arrays > MAX_FUSED_ARRAYS {
                    continue;
                }
                members.insert(j);
                outputs += usize::from(is_output[j]);
                external.retain(|&r| r != Ref::Step(j, 0));
                add_inputs(j, &mut external);
                candidates.extend(step_inputs(j));
            }
            if members.len() < 2 || external.len() + outputs > MAX_FUSED_ARRAYS {
                continue;
            }
            let mut members: Vec<usize> = members.into_iter().collect();
            members.sort_unstable();
            for &m in &members {
                group[m] = Some(root);
            }
            groups.insert(root, (members, external));
        }
        if groups.is_empty() {
            return;
        }
        // Rebuild: each group becomes one step at its root's position. `index` maps a
        // kept or fused step's outputs to the new step (and the output offset in it).
        let mut index: Vec<Option<(usize, usize)>> = vec![None; n];
        let remap = |r: Ref, index: &[Option<(usize, usize)>]| match r {
            Ref::Step(j, o) => {
                let (k, offset) = index[j].expect("defined before use");
                Ref::Step(k, offset + o)
            }
            r => r,
        };
        let mut steps = Vec::with_capacity(n);
        for k in 0..n {
            let step = match (group[k], groups.get(&k)) {
                (Some(_), None) => continue, // inside a group: defined with its root
                (Some(_), Some((members, external))) => {
                    let position: HashMap<usize, usize> =
                        members.iter().enumerate().map(|(p, &m)| (m, p)).collect();
                    let graph = members
                        .iter()
                        .map(|&m| {
                            let args = self.steps[m]
                                .inputs
                                .iter()
                                .map(|r| match *r {
                                    Ref::Step(j, _) if position.contains_key(&j) => {
                                        FusedRef::Step(position[&j])
                                    }
                                    r => FusedRef::Input(
                                        external.iter().position(|&e| e == r).expect("external"),
                                    ),
                                })
                                .collect();
                            (self.steps[m].primitive.clone(), args)
                        })
                        .collect();
                    // Outputs: the members the function returns, then the root.
                    let outs: Vec<usize> = members
                        .iter()
                        .copied()
                        .filter(|&m| m != k && is_output[m])
                        .chain([k])
                        .collect();
                    for (o, &m) in outs.iter().enumerate() {
                        index[m] = Some((steps.len(), o));
                    }
                    let dtypes: Vec<DType> =
                        external.iter().map(|&r| self.dtype(r, inputs)).collect();
                    let positions: Vec<usize> = outs.iter().map(|m| position[m]).collect();
                    let Ok(f) = Fused::new(&dtypes, graph, &positions) else {
                        unreachable!("every member is fusable")
                    };
                    steps.push(Step {
                        primitive: Primitive::Compiled(f),
                        inputs: external.iter().map(|&r| remap(r, &index)).collect(),
                        outputs: outs
                            .iter()
                            .map(|&m| self.steps[m].outputs[0].clone())
                            .collect(),
                    });
                    continue;
                }
                (None, _) => {
                    let s = &self.steps[k];
                    Step {
                        primitive: s.primitive.clone(),
                        inputs: s.inputs.iter().map(|&r| remap(r, &index)).collect(),
                        outputs: s.outputs.clone(),
                    }
                }
            };
            index[k] = Some((steps.len(), 0));
            steps.push(step);
        }
        self.outputs = self.outputs.iter().map(|&r| remap(r, &index)).collect();
        self.steps = steps;
    }
}

/// Builds a tape from a traced graph, simplifying as it goes.
struct Builder {
    /// Resolved arrays, by operation id and output index.
    refs: HashMap<(usize, usize), Ref>,
    consts: Vec<Array>,
    /// Constant scalars by value, for merging.
    scalars: HashMap<(DType, Vec<usize>, u64), usize>,
    steps: Vec<Step>,
    /// Steps by primitive and inputs, for common-subexpression elimination.
    seen: HashMap<(Primitive, Vec<Ref>), usize>,
}

impl Builder {
    fn new(placeholders: &[Array]) -> Self {
        let refs = placeholders
            .iter()
            .enumerate()
            .map(|(i, p)| ((p.id(), p.output_index()), Ref::Input(i)))
            .collect();
        Self {
            refs,
            consts: Vec::new(),
            scalars: HashMap::new(),
            steps: Vec::new(),
            seen: HashMap::new(),
        }
    }

    fn key(a: &Array) -> (usize, usize) {
        (a.id(), a.output_index())
    }

    /// Visits the graph of `outputs` inputs first (iteratively, so depth is limited only
    /// by memory) and returns the tape.
    fn build(mut self, outputs: &[Array]) -> Tape {
        let mut stack: Vec<(Array, bool)> = outputs.iter().map(|a| (a.clone(), false)).collect();
        while let Some((a, expanded)) = stack.pop() {
            if self.refs.contains_key(&Self::key(&a)) {
                continue;
            }
            let inputs = a.inputs();
            if expanded || inputs.is_empty() {
                self.resolve(&a, &inputs);
                continue;
            }
            stack.push((a, true));
            stack.extend(
                inputs
                    .into_iter()
                    .filter(|i| !self.refs.contains_key(&Self::key(i)))
                    .map(|i| (i, false)),
            );
        }
        let outputs = outputs.iter().map(|a| self.refs[&Self::key(a)]).collect();
        Tape {
            consts: self.consts,
            steps: self.steps,
            outputs,
        }
    }

    /// Records the operation producing `a` (whose inputs are resolved): a constant if it
    /// does not depend on an input, an existing value if it is an identity or computed
    /// already, else a new step.
    fn resolve(&mut self, a: &Array, inputs: &[Array]) {
        let siblings = a.siblings();
        let args: Vec<Ref> = inputs.iter().map(|i| self.refs[&Self::key(i)]).collect();
        let primitive = a.primitive();
        let Some(primitive) =
            primitive.filter(|_| !args.iter().all(|r| matches!(r, Ref::Const(_))))
        else {
            for s in &siblings {
                let c = self.constant(s);
                self.refs.insert(Self::key(s), Ref::Const(c));
            }
            return;
        };
        if siblings.len() == 1 && is_identity(&primitive, &inputs[0], a) {
            self.refs.insert(Self::key(a), args[0]);
            return;
        }
        let k = match self.seen.get(&(primitive.clone(), args.clone())) {
            Some(&k) => k,
            None => {
                self.steps.push(Step {
                    primitive: primitive.clone(),
                    inputs: args.clone(),
                    outputs: siblings
                        .iter()
                        .map(|s| (s.dtype(), s.shape().to_vec()))
                        .collect(),
                });
                let k = self.steps.len() - 1;
                self.seen.insert((primitive, args), k);
                k
            }
        };
        for (o, s) in siblings.iter().enumerate() {
            self.refs.insert(Self::key(s), Ref::Step(k, o));
        }
    }

    /// The constant index of `a`, shared with an equal scalar constant if there is one.
    fn constant(&mut self, a: &Array) -> usize {
        let value = scalar_bits(a).or_else(|| {
            // A broadcast scalar, as the operations make for scalar operands.
            match (a.primitive(), a.inputs().as_slice()) {
                (Some(Primitive::Broadcast(_)), [x]) => scalar_bits(x),
                _ => None,
            }
        });
        let key = value.map(|v| (a.dtype(), a.shape().to_vec(), v));
        if let Some(&c) = key.as_ref().and_then(|k| self.scalars.get(k)) {
            return c;
        }
        self.consts.push(a.clone());
        let c = self.consts.len() - 1;
        if let Some(key) = key {
            self.scalars.insert(key, c);
        }
        c
    }
}

/// Whether `primitive` maps `input` to `output` unchanged.
fn is_identity(primitive: &Primitive, input: &Array, output: &Array) -> bool {
    let same = input.shape() == output.shape() && input.strides() == output.strides();
    match primitive {
        Primitive::Broadcast(_) | Primitive::Reshape(_) | Primitive::StopGradient => same,
        Primitive::Transpose(axes) => axes.iter().enumerate().all(|(i, &a)| i == a),
        Primitive::AsType(d) => *d == input.dtype(),
        _ => false,
    }
}

/// The bits of an evaluated one-element array's value.
fn scalar_bits(a: &Array) -> Option<u64> {
    if a.size() != 1 || !a.is_evaluated() {
        return None;
    }
    Some(match a.dtype() {
        DType::Bool => u64::from(a.item::<bool>().ok()?),
        DType::U8 => u64::from(a.item::<u8>().ok()?),
        DType::U16 => u64::from(a.item::<u16>().ok()?),
        DType::U32 => u64::from(a.item::<u32>().ok()?),
        DType::U64 => a.item::<u64>().ok()?,
        DType::I8 => a.item::<i8>().ok()? as u64,
        DType::I16 => a.item::<i16>().ok()? as u64,
        DType::I32 => a.item::<i32>().ok()? as u64,
        DType::I64 => a.item::<i64>().ok()? as u64,
        DType::F16 => u64::from(a.item::<f16>().ok()?.to_bits()),
        DType::BF16 => u64::from(a.item::<bf16>().ok()?.to_bits()),
        DType::F32 => u64::from(a.item::<f32>().ok()?.to_bits()),
        DType::F64 => a.item::<f64>().ok()?.to_bits(),
        DType::Complex64 => {
            let z = a.item::<Complex64>().ok()?;
            u64::from(z.re.to_bits()) << 32 | u64::from(z.im.to_bits())
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use nuvec_core::{Device, add, exp, multiply, set_default_stream};

    use super::*;

    fn trace(f: &dyn Fn(&[Array]) -> Result<Vec<Array>>, inputs: &[Array], device: Device) -> Tape {
        Tape::trace(f, inputs, Stream::default_for(device)).unwrap()
    }

    fn names(tape: &Tape) -> Vec<&'static str> {
        tape.steps.iter().map(|s| s.primitive.name()).collect()
    }

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn simplifies_identities_constants_and_common_subexpressions() {
        set_default_stream(Stream::default_for(Device::cpu()));
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let f = |xs: &[Array]| {
            let x = xs[0].astype(DType::F32)?.reshape(&[2])?;
            let a = multiply(&x, 2.0)?; // a scalar constant ...
            let b = multiply(&x, 2.0)?; // ... merged with this one, so `b` is `a`
            Ok(vec![add(&a, &b)?, exp(&x)?, exp(&x)?])
        };
        let tape = trace(&f, std::slice::from_ref(&x), Device::cpu());
        let mut ops = names(&tape);
        ops.sort_unstable();
        assert_eq!(ops, ["add", "exp", "multiply"]);
        assert_eq!(tape.outputs[1], tape.outputs[2]);
        let outs = tape.replay(&[x]).unwrap();
        assert_eq!(outs[0].to_vec::<f32>().unwrap(), [4.0, 8.0]);
    }

    #[test]
    fn fuses_element_wise_chains_on_the_gpu_only() {
        set_default_stream(Stream::default_for(Device::cpu()));
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let w = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        // matmul, then a chain whose middle value is also an output of the function: one
        // kernel with two outputs.
        let f = |xs: &[Array]| {
            let h = nuvec_core::matmul(&xs[0].reshape(&[1, 2])?, &w)?;
            let a = exp(&multiply(&h, 2.0)?)?;
            let b = add(&multiply(&a, &a)?, 1.0)?;
            Ok(vec![a, nuvec_core::sigmoid(&b)?])
        };
        assert_eq!(
            names(&trace(&f, std::slice::from_ref(&x), Device::cpu())),
            [
                "reshape", "matmul", "multiply", "exp", "multiply", "add", "sigmoid"
            ]
        );
        let tape = trace(&f, &[x], Device::gpu(0));
        assert_eq!(names(&tape), ["reshape", "matmul", "compiled"]);
        let Primitive::Compiled(fused) = &tape.steps[2].primitive else {
            unreachable!()
        };
        assert_eq!((fused.len(), fused.dtypes().len()), (5, 2));
        assert_eq!(tape.outputs, [Ref::Step(2, 0), Ref::Step(2, 1)]);
    }

    #[test]
    fn a_shared_value_used_outside_a_group_stays_outside() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        // `t` feeds the chain and a reduction: it cannot be fused into the chain. The
        // broadcast sum is a view the fused kernel reads.
        let f = |xs: &[Array]| {
            let t = exp(&xs[0])?;
            let s = nuvec_core::sum(&t, &[], true)?;
            Ok(vec![add(&multiply(&t, 3.0)?, &s)?])
        };
        let tape = trace(&f, &[x], Device::gpu(0));
        assert_eq!(names(&tape), ["exp", "sum", "broadcast_to", "compiled"]);
    }
}
