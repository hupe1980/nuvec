//! Function compilation: tracing, graph simplification and kernel fusion.
//!
//! [`compile`] works as MLX's `mx.compile`. The first call with a given input structure and
//! signature traces the function on placeholders and simplifies the graph:
//!
//! - identity views and same-type conversions are removed (explicit copies and
//!   dependencies, `Primitive::Copy` and `Primitive::Depends`, are kept);
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
//! use nuvec_core::{Array, Result, array};
//!
//! # fn main() -> Result<()> {
//! // x * x + 1, as one kernel on the GPU
//! let f = compile(|x: &Array| x.multiply(x).add(1.0));
//! assert_eq!(f(&array![1.0, 2.0])?.to_vec::<f32>()?, [2.0, 5.0]);
//! assert_eq!(f(&array![3.0, 4.0])?.to_vec::<f32>()?, [10.0, 17.0]); // replayed
//! # Ok(())
//! # }
//! ```
//!
//! [`compile_mut`] compiles a stateful step (MLX's `mx.compile(step, inputs=state,
//! outputs=state)`): a training step over `(model, optimizer)` whose leaves are inputs and
//! outputs of the tape.
//!
//! **Stability:** no promise when this crate is depended on directly; depend on `nuvec`,
//! whose re-exports are semver-tracked.
//!
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

use std::cell::Cell;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use nuvec_core::graph::{self, Fused, FusedRef, Primitive};
use nuvec_core::tree::{Tree, TreeDef};
use nuvec_core::{
    Array, Complex64, DType, DeviceKind, Error, IntoResult, Result, Stream, bf16, default_stream,
    f16, random,
};

/// Dimension sizes as `usize`.
fn dims(a: &Array) -> Vec<usize> {
    a.shape().iter().map(|&d| d as usize).collect()
}

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The most operations fused into one kernel.
const MAX_FUSED_STEPS: usize = 32;
/// The most inputs and outputs of one fused kernel (Metal binds at most 31 buffers per
/// kernel; a fused kernel uses five more), as MLX's `max_compile_arrays`.
const MAX_FUSED_ARRAYS: usize = 24;

/// Options of [`compile`] and [`compile_mut`]: `Options::default().cache_capacity(8)`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Options {
    /// The most traced signatures a compiled function keeps (least recently used first
    /// out); 0 never caches. Default 64.
    pub cache_capacity: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { cache_capacity: 64 }
    }
}

impl Options {
    /// Sets the cache capacity.
    pub fn cache_capacity(mut self, n: usize) -> Self {
        self.cache_capacity = n;
        self
    }

    /// [`compile`] with these options.
    pub fn compile<'f, I, O, F, R>(self, f: F) -> impl Fn(&I) -> Result<O> + 'f
    where
        I: Tree,
        O: Tree,
        F: Fn(&I) -> R + 'f,
        R: IntoResult<Ok = O>,
    {
        let cache: Mutex<Cache<(Tape, O)>> = Mutex::new(Cache::new(self.cache_capacity));
        move |inputs: &I| {
            let leaves = inputs.leaves();
            if leaves.iter().any(|a| graph::is_tracer(a) || a.is_failed()) {
                // Inside another transform, or on failed inputs (whose failure the direct
                // call carries to the outputs): no trace.
                return f(inputs).into_result();
            }
            let key = Key::new(vec![inputs.def()], &leaves);
            let cached = lock(&cache).get(&key);
            let entry = match cached {
                Some(entry) => entry,
                None => {
                    let mut out = None;
                    let mut body = |xs: &[Array]| -> Result<Vec<Array>> {
                        let o = f(&inputs.unflatten(xs.iter().cloned())?).into_result()?;
                        let leaves = o.leaves();
                        out = Some(o.map(|_, a| light(a)));
                        Ok(leaves)
                    };
                    let tape = Tape::trace(&mut body, &leaves, key.stream)?;
                    let out =
                        out.ok_or_else(|| Error::InvalidArgument("compile: no output".into()))?;
                    let entry = Arc::new((tape, out));
                    if !entry.0.captures_tracer() {
                        lock(&cache).insert(key, entry.clone());
                    }
                    entry
                }
            };
            let (tape, template) = &*entry;
            template.unflatten(tape.replay(&leaves)?)
        }
    }

    /// [`compile_mut`] with these options.
    pub fn compile_mut<'f, S, I, O, F, R>(self, f: F) -> impl Fn(&mut S, &I) -> Result<O> + 'f
    where
        S: Tree,
        I: Tree,
        O: Tree,
        F: Fn(&mut S, &I) -> R + 'f,
        R: IntoResult<Ok = O>,
    {
        let cache: Mutex<Cache<MutTape<S, O>>> = Mutex::new(Cache::new(self.cache_capacity));
        let changed_last = Cell::new(false);
        move |state: &mut S, inputs: &I| {
            let state_leaves = state.leaves();
            let input_leaves = inputs.leaves();
            let direct = state_leaves
                .iter()
                .chain(&input_leaves)
                .any(|a| graph::is_tracer(a) || a.is_failed());
            if direct {
                // Inside another transform (or on failed arrays): run `f` itself.
                return f(state, inputs).into_result();
            }
            let key0 = random::sequence_key();
            let pre_def = state.def();
            let mut all = Vec::with_capacity(1 + state_leaves.len() + input_leaves.len());
            all.push(key0.clone());
            all.extend(state_leaves.iter().cloned());
            all.extend(input_leaves.iter().cloned());
            let key = Key::new(vec![pre_def.clone(), inputs.def()], &all);
            let cached = lock(&cache).get(&key);
            let entry = match cached {
                Some(entry) => entry,
                None => {
                    let entry = Arc::new(MutTape::trace(&f, state, inputs, &all, key.stream)?);
                    if !entry.tape.captures_tracer() {
                        lock(&cache).insert(key, entry.clone());
                    }
                    entry
                }
            };
            let changed = entry.post_def != pre_def;
            if changed && changed_last.get() {
                return Err(Error::InvalidArgument(
                    "compile_mut: state structure keeps changing; call opt.init(&model) before \
                     the loop, or make changing host fields arrays (or `static_unchecked`)"
                        .into(),
                ));
            }
            let mut outputs = entry.tape.replay(&all)?;
            let next_key = outputs.pop().expect("the key is the last output");
            let state_out = outputs.split_off(entry.outputs);
            let out = entry.out.unflatten(outputs)?;
            let next = entry.post.unflatten(state_out)?;
            changed_last.set(changed);
            *state = next;
            random::set_sequence_key(next_key);
            Ok(out)
        }
    }
}

/// Returns `f` compiled: a function with the same results that traces `f` once per input
/// structure (the [`TreeDef`]: paths, leaf kinds, trainable bits, enum variants, mode flags
/// and static fields), leaf dtypes and shapes, and default stream, and replays the
/// simplified, fused graph on later calls.
///
/// `f` takes a tree (an array, a tuple, a struct) and returns one, directly or as a
/// `Result`; it may borrow locals (the cache stores tapes, not `f`). Write the closure's
/// parameter type (`|x: &Array|`).
///
/// As in MLX, `f` must be pure:
///
/// - arrays it captures are constants, frozen at the first trace (so is random state
///   drawn inside it; [`compile_mut`] threads the random key instead);
/// - arrays cannot be evaluated inside it.
///
/// Called on arrays being traced by a transform (`grad`, `vmap`, another `compile`), or on
/// failed arrays, it runs `f` directly, so transforms see the original operations and
/// failures stay sticky.
///
/// Each compiled function keeps the tapes of its 64 most recently used signatures
/// ([`Options::cache_capacity`]). A trace whose constants include an array being traced by
/// a running transform (a tracer `f` captures) is used once and not kept.
pub fn compile<'f, I, O, F, R>(f: F) -> impl Fn(&I) -> Result<O> + 'f
where
    I: Tree,
    O: Tree,
    F: Fn(&I) -> R + 'f,
    R: IntoResult<Ok = O>,
{
    Options::default().compile(f)
}

/// Compiles a stateful step, as MLX's `mx.compile(step, inputs=state, outputs=state)`:
/// `f(&mut state, &inputs)` where `state` is a tree (typically `(model, optimizer)`).
///
/// ```
/// use nuvec_compile::compile_mut;
/// use nuvec_core::{Array, Result, array, value_and_grad};
///
/// # fn main() -> Result<()> {
/// // Gradient descent on w for loss(w) = sum((w x - y)^2), compiled.
/// let step = compile_mut(|w: &mut Array, (x, y): &(Array, Array)| {
///     let loss = |w: &Array| {
///         let r = w.multiply(x).subtract(y);
///         r.multiply(&r).sum(..)
///     };
///     let (l, g) = value_and_grad(loss)(w)?;
///     *w = w.subtract(g.multiply(0.1));
///     Ok(l)
/// });
/// let mut w = array![0.0];
/// for _ in 0..50 {
///     step(&mut w, &(array![1.0], array![2.0]))?;
/// }
/// assert!((w.item::<f32>()? - 2.0).abs() < 1e-3);
/// # Ok(())
/// # }
/// ```
///
/// - The state's leaves (and the thread's random key) are traced inputs; after the call the
///   state is rebuilt from the tape's outputs. So the step traces once, an optimizer's
///   learning-rate array set between calls is just a new input, and random draws inside
///   (dropout) advance the thread's key sequence exactly as an uncompiled call would.
/// - The tape is keyed by the state's and the inputs' structure: switching train/eval,
///   freezing, changing a hashed hyperparameter retraces once.
/// - A step that changes the state's structure (an optimizer creating its slots at the
///   first update) retraces once; a structure that changes on two consecutive calls is an
///   error (call `opt.init(&model)` first). Host fields mutated in `f` are hashed statics,
///   so mutating one counts as a structure change; `#[tree(skip, static_unchecked)]` fields
///   are baked into the trace.
/// - On error the state is unchanged. Called inside another transform, it runs `f`
///   directly.
pub fn compile_mut<'f, S, I, O, F, R>(f: F) -> impl Fn(&mut S, &I) -> Result<O> + 'f
where
    S: Tree,
    I: Tree,
    O: Tree,
    F: Fn(&mut S, &I) -> R + 'f,
    R: IntoResult<Ok = O>,
{
    Options::default().compile_mut(f)
}

/// [`compile`] over positional arrays: traces once per input signature (dtypes and
/// shapes) and default stream.
///
/// ```
/// use nuvec_compile::compile_flat;
/// use nuvec_core::{Array, Result, array};
///
/// # fn main() -> Result<()> {
/// let f = compile_flat(|xs: &[Array]| Ok(vec![xs[0].multiply(&xs[0]).add(1.0)]));
/// assert_eq!(f(&[array![1.0, 2.0]])?[0].to_vec::<f32>()?, [2.0, 5.0]);
/// # Ok(())
/// # }
/// ```
pub fn compile_flat<F>(f: F) -> impl Fn(&[Array]) -> Result<Vec<Array>>
where
    F: Fn(&[Array]) -> Result<Vec<Array>>,
{
    let cache: Mutex<Cache<Tape>> = Mutex::new(Cache::new(Options::default().cache_capacity));
    move |inputs| {
        if inputs.iter().any(|a| graph::is_tracer(a) || a.is_failed()) {
            return f(inputs);
        }
        let key = Key::new(Vec::new(), inputs);
        let cached = lock(&cache).get(&key);
        let tape = match cached {
            Some(tape) => tape,
            None => {
                let tape = Arc::new(Tape::trace(&mut |xs: &[Array]| f(xs), inputs, key.stream)?);
                if !tape.captures_tracer() {
                    lock(&cache).insert(key, tape.clone());
                }
                tape
            }
        };
        tape.replay(inputs)
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A placeholder like `a` (a template leaf: only its dtype and shape matter).
fn light(a: &Array) -> Array {
    graph::placeholder(a.dtype(), a.shape()).unwrap_or_else(|_| a.clone())
}

/// A [`compile_mut`] trace: the tape (outputs: `f`'s, then the post-call state's, then the
/// advanced random key), the output and post-state templates, and the post-call structure.
struct MutTape<S, O> {
    tape: Tape,
    outputs: usize,
    out: O,
    post: S,
    post_def: TreeDef,
}

impl<S: Tree, O: Tree> MutTape<S, O> {
    /// Traces `f` with the thread's random key (`all[0]`), the state's leaves and the
    /// inputs' as placeholders; `state` itself is not changed.
    fn trace<I: Tree, R: IntoResult<Ok = O>>(
        f: &dyn Fn(&mut S, &I) -> R,
        state: &S,
        inputs: &I,
        all: &[Array],
        stream: Stream,
    ) -> Result<Self> {
        let ns = state.len();
        let key0 = all[0].clone();
        let mut traced: Option<(O, S)> = None;
        let mut body = |xs: &[Array]| -> Result<Vec<Array>> {
            random::set_sequence_key(xs[0].clone());
            let mut s = state.unflatten(xs[1..1 + ns].iter().cloned())?;
            let i = inputs.unflatten(xs[1 + ns..].iter().cloned())?;
            let out = f(&mut s, &i).into_result();
            let key = random::sequence_key();
            random::set_sequence_key(key0.clone());
            let out = out?;
            let mut leaves = out.leaves();
            leaves.extend(s.leaves());
            leaves.push(key);
            traced = Some((out, s));
            Ok(leaves)
        };
        let result = Tape::trace(&mut body, all, stream);
        random::set_sequence_key(key0.clone());
        let tape = result?;
        let (out, post) =
            traced.ok_or_else(|| Error::InvalidArgument("compile_mut: no output".into()))?;
        Ok(Self {
            tape,
            outputs: out.len(),
            out: out.map(|_, a| light(a)),
            post_def: post.def(),
            post: post.map(|_, a| light(a)),
        })
    }
}

/// A compiled function's tapes by key, least recently used evicted first.
struct Cache<T> {
    tapes: HashMap<Key, (Arc<T>, u64)>,
    /// Incremented on every use; a tape's stamp is its last use.
    clock: u64,
    capacity: usize,
}

impl<T> Cache<T> {
    fn new(capacity: usize) -> Self {
        Self {
            tapes: HashMap::new(),
            clock: 0,
            capacity,
        }
    }

    fn get(&mut self, key: &Key) -> Option<Arc<T>> {
        self.clock += 1;
        let (tape, used) = self.tapes.get_mut(key)?;
        *used = self.clock;
        Some(tape.clone())
    }

    fn insert(&mut self, key: Key, tape: Arc<T>) {
        if self.capacity == 0 {
            return;
        }
        self.clock += 1;
        if self.tapes.len() >= self.capacity
            && !self.tapes.contains_key(&key)
            && let Some(oldest) = self
                .tapes
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(k, _)| k.clone())
        {
            self.tapes.remove(&oldest);
        }
        self.tapes.insert(key, (tape, self.clock));
    }
}

/// What a traced graph depends on: the inputs' structures, dtypes and shapes and the
/// stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    stream: Stream,
    defs: Vec<TreeDef>,
    inputs: Vec<(DType, Vec<usize>)>,
}

impl Key {
    fn new(defs: Vec<TreeDef>, inputs: &[Array]) -> Self {
        Self {
            stream: default_stream(),
            defs,
            inputs: inputs.iter().map(|a| (a.dtype(), dims(a))).collect(),
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
        f: &mut dyn FnMut(&[Array]) -> Result<Vec<Array>>,
        inputs: &[Array],
        stream: Stream,
    ) -> Result<Self> {
        let placeholders = inputs
            .iter()
            .map(|a| graph::placeholder(a.dtype(), a.shape()))
            .collect::<Result<Vec<_>>>()?;
        let outputs = f(&placeholders)?;
        let mut tape = Builder::new(&placeholders).build(&outputs);
        if stream.device.kind == DeviceKind::Gpu {
            tape.fuse(inputs);
        }
        Ok(tape)
    }

    /// Whether a constant is an array being traced by a running transform (a tracer `f`
    /// captured): such a tape is used once and not cached.
    fn captures_tracer(&self) -> bool {
        self.consts.iter().any(graph::is_tracer)
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
            values.push(graph::from_primitive(step.primitive.clone(), args)?);
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
            .map(|(i, p)| ((graph::id(p), graph::output_index(p)), Ref::Input(i)))
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
        (graph::id(a), graph::output_index(a))
    }

    /// Visits the graph of `outputs` inputs first (iteratively, so depth is limited only
    /// by memory) and returns the tape.
    fn build(mut self, outputs: &[Array]) -> Tape {
        let mut stack: Vec<(Array, bool)> = outputs.iter().map(|a| (a.clone(), false)).collect();
        while let Some((a, expanded)) = stack.pop() {
            if self.refs.contains_key(&Self::key(&a)) {
                continue;
            }
            let inputs = graph::inputs(&a);
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
        let siblings = graph::siblings(a);
        let args: Vec<Ref> = inputs.iter().map(|i| self.refs[&Self::key(i)]).collect();
        let primitive = graph::primitive(a);
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
                    outputs: siblings.iter().map(|s| (s.dtype(), dims(s))).collect(),
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

    /// The constant index of `a`, shared with an equal scalar constant if there is one. A
    /// tracer (an array an outer transform differentiates, captured by `f`) is never merged
    /// with an equal value: the outer transform must see it.
    fn constant(&mut self, a: &Array) -> usize {
        if graph::is_tracer(a) {
            self.consts.push(a.clone());
            return self.consts.len() - 1;
        }
        let value = scalar_bits(a).or_else(|| {
            // A broadcast scalar, as the operations make for scalar operands.
            match (graph::primitive(a), graph::inputs(a).as_slice()) {
                (Some(Primitive::Broadcast(_)), [x]) => scalar_bits(x),
                _ => None,
            }
        });
        let key = value.map(|v| (a.dtype(), dims(a), v));
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
    let same = input.shape() == output.shape() && graph::strides(input) == graph::strides(output);
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
    use nuvec_core::{Device, array, set_default_stream};

    use super::*;

    fn trace(f: &dyn Fn(&[Array]) -> Result<Vec<Array>>, inputs: &[Array], device: Device) -> Tape {
        Tape::trace(
            &mut |xs: &[Array]| f(xs),
            inputs,
            Stream::default_for(device),
        )
        .unwrap()
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
        let x = array![1.0, 2.0];
        let f = |xs: &[Array]| {
            let x = xs[0].astype(DType::F32).reshape([2]);
            let a = x.multiply(2.0); // a scalar constant ...
            let b = x.multiply(2.0); // ... merged with this one, so `b` is `a`
            Ok(vec![a.add(&b), x.exp(), x.exp()])
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
        let x = array![1.0, 2.0];
        let w = array![[1.0, 0.0], [0.0, 1.0]];
        // matmul, then a chain whose middle value is also an output of the function: one
        // kernel with two outputs.
        let f = |xs: &[Array]| {
            let h = xs[0].reshape([1, 2]).matmul(&w);
            let a = h.multiply(2.0).exp();
            let b = a.multiply(&a).add(1.0);
            Ok(vec![a, b.sigmoid()])
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
    fn copies_and_dependencies_are_kept() {
        set_default_stream(Stream::default_for(Device::cpu()));
        let x = array![1.0, 2.0];
        let y = array![3.0, 4.0];
        let f = |xs: &[Array]| {
            let copy = graph::from_primitive(Primitive::Copy, vec![xs[0].clone()])?.remove(0);
            let ordered = graph::from_primitive(
                Primitive::Depends { n_deps: 1 },
                vec![xs[0].clone(), xs[1].clone()],
            )?
            .remove(0);
            // A same-dtype conversion is still an identity.
            Ok(vec![copy.add(ordered.astype(DType::F32))])
        };
        let tape = trace(&f, &[x.clone(), y.clone()], Device::cpu());
        let mut ops = names(&tape);
        ops.sort_unstable();
        assert_eq!(ops, ["add", "copy", "depends"]);
        let out = tape.replay(&[x, y]).unwrap().remove(0);
        assert_eq!(out.to_vec::<f32>().unwrap(), [2.0, 4.0]);
    }

    #[test]
    fn a_shared_value_used_outside_a_group_stays_outside() {
        let x = array![1.0, 2.0];
        // `t` feeds the chain and a reduction: it cannot be fused into the chain. The
        // broadcast sum is a view the fused kernel reads.
        let f = |xs: &[Array]| {
            let t = xs[0].exp();
            let s = t.sum_keepdims(..);
            Ok(vec![t.multiply(3.0).add(&s)])
        };
        let tape = trace(&f, &[x], Device::gpu(0));
        assert_eq!(names(&tape), ["exp", "sum", "broadcast_to", "compiled"]);
    }
}
