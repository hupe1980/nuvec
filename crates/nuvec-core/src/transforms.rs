//! Function transforms: reverse-mode ([`vjp`], [`grad`], [`value_and_grad`]) and
//! forward-mode ([`jvp`]) automatic differentiation.
//!
//! As in MLX, the function runs on *tracer* inputs and records a lazy graph. The derivative
//! is another lazy graph, built from each primitive's rule, so transforms compose: the
//! gradient of a gradient is a second derivative. Tracers are zero-copy identity views of
//! the inputs. While a transform runs, evaluating a node that depends on a tracer keeps its
//! graph instead of detaching it, so calling `eval` inside the function is allowed.

use std::cell::RefCell;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use crate::array::{Node, NodeMap, NodeSet, State};
use crate::cast::Canon;
use crate::derivatives::zeros_like;
use crate::ops::imp as ops;
use crate::primitive::Primitive;
use crate::tree::{LeafKind, Leaves, Tree, Visit, Visitor, broadcast_zero, differentiable};
use crate::{Array, Category, Error, Result};

/// Identifies one run of a transform. Ids increase, and transforms nest like a stack, so
/// the smallest live id among a node's inputs is the outermost transform it depends on.
pub(crate) type TraceId = u64;

/// The trace of `compile`'s placeholders: always live (evaluating them fails anyway).
pub(crate) const PLACEHOLDER: TraceId = 0;

static NEXT_TRACE: AtomicU64 = AtomicU64::new(PLACEHOLDER + 1);

/// A running transform: its id and the nodes it retained (evaluated tracers that keep
/// their graph), released when it exits.
struct Frame {
    id: TraceId,
    retained: Vec<Weak<Node>>,
}

thread_local!(static LIVE: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) });

/// Whether transform `id` is running on this thread.
pub(crate) fn is_live(id: TraceId) -> bool {
    id == PLACEHOLDER || LIVE.with_borrow(|live| live.iter().any(|f| f.id == id))
}

/// The trace of a node recorded on `inputs`: the outermost live transform among theirs.
pub(crate) fn inherited(inputs: &[Array]) -> Option<TraceId> {
    inputs
        .iter()
        .filter_map(|a| a.node.trace)
        .filter(|&id| is_live(id))
        .min()
}

/// Registers `node`, just evaluated, to keep its graph until the transform it depends on
/// exits; false if it is not a tracer (of a transform running on this thread).
pub(crate) fn retain(node: &Arc<Node>) -> bool {
    let Some(id) = node.trace.filter(|&id| id != PLACEHOLDER) else {
        return false;
    };
    LIVE.with_borrow_mut(|live| match live.iter_mut().find(|f| f.id == id) {
        Some(frame) => {
            frame.retained.push(Arc::downgrade(node));
            true
        }
        None => false,
    })
}

/// Marks a transform as running on this thread while alive (nests). On exit, the nodes
/// it retained detach (become plain evaluated nodes) and its arrays stop being tracers.
struct Tracing {
    id: TraceId,
}

impl Tracing {
    fn enter() -> Self {
        let id = NEXT_TRACE.fetch_add(1, Ordering::Relaxed);
        LIVE.with_borrow_mut(|live| {
            live.push(Frame {
                id,
                retained: Vec::new(),
            });
        });
        Tracing { id }
    }

    /// A tracer for `primal`: an identity view (a broadcast to its own shape). It is
    /// differentiable, so an enclosing transform differentiates through it.
    fn tracer(&self, primal: &Array) -> Result<Array> {
        let shape = primal.dims().to_vec();
        let inputs = vec![primal.clone()];
        let trace = Some(inherited(&inputs).map_or(self.id, |outer| outer.min(self.id)));
        Ok(Array::record(Primitive::Broadcast(shape), inputs, trace)?.swap_remove(0))
    }
}

impl Drop for Tracing {
    fn drop(&mut self) {
        let frame = LIVE.with_borrow_mut(|live| {
            let at = live.iter().rposition(|f| f.id == self.id)?;
            Some(live.remove(at))
        });
        for node in frame.into_iter().flat_map(|f| f.retained) {
            let Some(node) = node.upgrade() else { continue };
            let inputs = {
                let mut state = node.lock();
                match std::mem::replace(&mut *state, State::Done) {
                    State::Retained { inputs, .. } => inputs,
                    other => {
                        *state = other;
                        continue;
                    }
                }
            };
            drop(inputs); // detach outside the lock
        }
    }
}

type Key = (*const Node, u32);

fn key(a: &Array) -> Key {
    (Arc::as_ptr(&a.node), a.output)
}

/// A recorded node on a path from the tracers to the outputs.
struct Step {
    node: Arc<Node>,
    primitive: Primitive,
    inputs: Vec<Array>,
    /// Which inputs depend on a tracer.
    wanted: Vec<bool>,
}

impl Step {
    fn outputs(&self) -> Vec<Array> {
        (0..self.node.outputs.len() as u32)
            .map(|output| Array {
                node: self.node.clone(),
                output,
            })
            .collect()
    }
}

/// A node's primitive and inputs.
type Graph = (Primitive, Vec<Array>);

/// The primitive and inputs of a node whose graph is still known.
fn graph(node: &Node) -> Option<Graph> {
    match &*node.lock() {
        State::Pending { primitive, inputs } | State::Retained { primitive, inputs } => {
            Some((primitive.clone(), inputs.clone()))
        }
        State::Running { .. } | State::Done | State::Failed { .. } => None,
    }
}

/// The inputs a transform follows: a custom function's own outputs (its trailing inputs)
/// are handled by its rules, so its graph is never walked.
fn followed(primitive: &Primitive, inputs: &[Array]) -> usize {
    match primitive {
        Primitive::Custom(call) => call.primals,
        _ => inputs.len(),
    }
}

/// Nodes between `sources` and `outputs` (those that depend on a source), inputs before
/// consumers. For differentiation (`stop_gradient` true), `StopGradient` cuts the graph.
/// Iterative, so depth is limited only by memory.
fn tape(outputs: &[Array], sources: &NodeSet, stop_gradient: bool) -> Vec<Step> {
    let mut depends: NodeMap<*const Node, bool> = NodeMap::default();
    let mut visited = NodeSet::default();
    let mut order = Vec::new();
    let mut stack: Vec<(Arc<Node>, Option<Graph>)> =
        outputs.iter().map(|a| (a.node.clone(), None)).collect();
    while let Some((node, expanded)) = stack.pop() {
        let ptr = Arc::as_ptr(&node);
        if let Some((primitive, inputs)) = expanded {
            let n = followed(&primitive, &inputs);
            let wanted: Vec<bool> = inputs
                .iter()
                .enumerate()
                .map(|(i, a)| i < n && depends.get(&Arc::as_ptr(&a.node)).copied().unwrap_or(false))
                .collect();
            let cut = stop_gradient && primitive == Primitive::StopGradient;
            let dep = !cut && wanted.contains(&true);
            depends.insert(ptr, dep);
            if dep {
                order.push(Step {
                    node,
                    primitive,
                    inputs,
                    wanted,
                });
            }
            continue;
        }
        if !visited.insert(ptr) {
            continue;
        }
        if sources.contains(&ptr) {
            depends.insert(ptr, true);
            continue;
        }
        let Some((primitive, inputs)) = graph(&node) else {
            continue; // evaluated data: a constant
        };
        let next: Vec<Arc<Node>> = inputs[..followed(&primitive, &inputs)]
            .iter()
            .map(|a| a.node.clone())
            .filter(|n| !visited.contains(&Arc::as_ptr(n)))
            .collect();
        stack.push((node, Some((primitive, inputs))));
        stack.extend(next.into_iter().map(|n| (n, None)));
    }
    order
}

fn accumulate(map: &mut NodeMap<Key, Array>, key: Key, value: Array) -> Result<()> {
    let value = match map.remove(&key) {
        Some(previous) => ops::add(&previous, &value)?,
        None => value,
    };
    map.insert(key, value);
    Ok(())
}

fn check_shapes(op: &str, what: &str, arrays: &[Array], given: &[Array]) -> Result<()> {
    if arrays.len() != given.len() {
        return Err(Error::InvalidArgument(format!(
            "{op}: {} {what}s for {} arrays",
            given.len(),
            arrays.len()
        )));
    }
    for (a, g) in arrays.iter().zip(given) {
        if a.dims() != g.dims() {
            return Err(Error::InvalidArgument(format!(
                "{op}: shape {:?} does not match {what} shape {:?}",
                a.dims(),
                g.dims()
            )));
        }
        if a.dtype() != g.dtype() {
            return Err(Error::InvalidArgument(format!(
                "{op}: dtype {} does not match {what} dtype {}",
                a.dtype(),
                g.dtype()
            )));
        }
    }
    Ok(())
}

/// Reverse pass: the cotangents of `tracers`, given cotangents for `outputs`.
fn reverse(outputs: &[Array], cotangents: &[Array], tracers: &[Array]) -> Result<Vec<Array>> {
    let sources = tracers.iter().map(|t| Arc::as_ptr(&t.node)).collect();
    let mut cot = NodeMap::default();
    for (o, c) in outputs.iter().zip(cotangents) {
        accumulate(&mut cot, key(o), c.clone())?;
    }
    for step in tape(outputs, &sources, true).iter().rev() {
        let outs = step.outputs();
        if !outs.iter().any(|o| cot.contains_key(&key(o))) {
            continue;
        }
        let cots = outs
            .iter()
            .map(|o| cot.remove(&key(o)).map_or_else(|| zeros_like(o), Ok))
            .collect::<Result<Vec<_>>>()?;
        let grads = step
            .primitive
            .vjp(&step.inputs, &outs, &cots, &step.wanted)?;
        for ((input, grad), &wanted) in step.inputs.iter().zip(grads).zip(&step.wanted) {
            if let (Some(grad), true) = (grad, wanted) {
                accumulate(&mut cot, key(input), grad)?;
            }
        }
    }
    tracers
        .iter()
        .map(|t| cot.remove(&key(t)).map_or_else(|| zeros_like(t), Ok))
        .collect()
}

/// `outputs`, or the error of the first failed one: a function that returns a failed
/// array makes the transform return its error.
fn checked(outputs: Vec<Array>) -> Result<Vec<Array>> {
    outputs.into_iter().map(Array::check).collect()
}

/// Reverse mode over positional arrays (the flat form of [`vjp`]): runs `f` on `primals`
/// and returns its outputs together with the vector-Jacobian product, one gradient per
/// primal, for one cotangent per output.
///
/// The results are lazy arrays. A primal that the outputs do not depend on gets a zero
/// gradient.
///
/// ```
/// use nuvec_core::{Result, array, vjp_flat};
///
/// # fn main() -> Result<()> {
/// let x = array![1.0, 2.0, 3.0];
/// let ones = x.ones_like();
/// let (_, grads) = vjp_flat(|xs| Ok(vec![xs[0].multiply(&xs[0])]), &[x], &[ones])?;
/// assert_eq!(grads[0].to_vec::<f32>()?, [2.0, 4.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub fn vjp_flat(
    f: impl FnOnce(&[Array]) -> Result<Vec<Array>>,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let tracing = Tracing::enter();
    let tracers = primals
        .iter()
        .map(|p| tracing.tracer(p))
        .collect::<Result<Vec<_>>>()?;
    let outputs = checked(f(&tracers)?)?;
    check_shapes("vjp", "cotangent", &outputs, cotangents)?;
    let grads = reverse(&outputs, cotangents, &tracers)?;
    Ok((outputs, grads))
}

/// Forward mode over positional arrays (the flat form of [`jvp`]): runs `f` on `primals`
/// and returns its outputs together with the Jacobian-vector product, one tangent per
/// output, for one tangent per primal.
///
/// ```
/// use nuvec_core::{Array, Result, jvp_flat};
///
/// # fn main() -> Result<()> {
/// let x = Array::from(0.0f32);
/// let (_, tangents) = jvp_flat(|xs| Ok(vec![xs[0].sin()]), &[x], &[Array::from(1.0f32)])?;
/// assert_eq!(tangents[0].item::<f32>()?, 1.0); // cos(0)
/// # Ok(())
/// # }
/// ```
pub fn jvp_flat(
    f: impl FnOnce(&[Array]) -> Result<Vec<Array>>,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    check_shapes("jvp", "tangent", primals, tangents)?;
    let tracing = Tracing::enter();
    let tracers = primals
        .iter()
        .map(|p| tracing.tracer(p))
        .collect::<Result<Vec<_>>>()?;
    let outputs = checked(f(&tracers)?)?;
    let tangents = forward(&outputs, &tracers, tangents)?;
    Ok((outputs, tangents))
}

/// Forward pass: the tangents of `outputs`, given tangents for `tracers`.
fn forward(outputs: &[Array], tracers: &[Array], tangents: &[Array]) -> Result<Vec<Array>> {
    let sources = tracers.iter().map(|t| Arc::as_ptr(&t.node)).collect();
    let mut tan: NodeMap<Key, Array> = tracers
        .iter()
        .zip(tangents)
        .map(|(t, v)| (key(t), v.clone()))
        .collect();
    for step in tape(outputs, &sources, true) {
        let inputs: Vec<Option<Array>> = step
            .inputs
            .iter()
            .map(|a| tan.get(&key(a)).cloned())
            .collect();
        if inputs.iter().all(Option::is_none) {
            continue;
        }
        let outs = step.outputs();
        for (o, t) in outs
            .iter()
            .zip(step.primitive.jvp(&step.inputs, &outs, &inputs)?)
        {
            tan.insert(key(o), t);
        }
    }
    outputs
        .iter()
        .map(|o| tan.get(&key(o)).cloned().map_or_else(|| zeros_like(o), Ok))
        .collect()
}

/// Returns a function computing `f`'s value and its gradients with respect to the
/// positional arguments at `argnums` (the flat form of [`value_and_grad`]).
///
/// `f` must return a 0-D array. Only the selected arguments are traced; the others are
/// treated as constants.
///
/// ```
/// use nuvec_core::{Array, Result, array, value_and_grad_flat};
///
/// # fn main() -> Result<()> {
/// // loss(w, x) = sum((w * x - 1)^2)
/// let loss = |xs: &[Array]| {
///     let r = xs[0].multiply(&xs[1]).subtract(1.0);
///     Ok(r.multiply(&r).sum(..))
/// };
/// let (w, x) = (array![2.0], array![3.0]);
/// let (value, grads) = value_and_grad_flat(loss, &[0])(&[w, x])?;
/// assert_eq!(value.item::<f32>()?, 25.0);
/// assert_eq!(grads[0].to_vec::<f32>()?, [30.0]); // 2 (w x - 1) x
/// # Ok(())
/// # }
/// ```
pub fn value_and_grad_flat<F>(
    f: F,
    argnums: &[usize],
) -> impl Fn(&[Array]) -> Result<(Array, Vec<Array>)>
where
    F: Fn(&[Array]) -> Result<Array>,
{
    let argnums = argnums.to_vec();
    move |args| {
        if argnums.is_empty() {
            return Err(Error::InvalidArgument(
                "value_and_grad: no argument to differentiate".into(),
            ));
        }
        let mut seen = HashSet::new();
        for &i in &argnums {
            if i >= args.len() || !seen.insert(i) {
                return Err(Error::InvalidArgument(format!(
                    "value_and_grad: invalid or repeated argument {i} for {} arguments",
                    args.len()
                )));
            }
            if !matches!(
                args[i].dtype().category(),
                Category::Float | Category::Complex
            ) {
                return Err(Error::InvalidArgument(format!(
                    "value_and_grad: argument {i} is {}; only floating-point and complex \
                     arrays can be differentiated",
                    args[i].dtype()
                )));
            }
        }
        let tracing = Tracing::enter();
        let mut inputs = args.to_vec();
        let mut tracers = Vec::with_capacity(argnums.len());
        for &i in &argnums {
            inputs[i] = tracing.tracer(&args[i])?;
            tracers.push(inputs[i].clone());
        }
        let value = f(&inputs)?.check()?;
        if value.ndim() != 0 {
            return Err(Error::InvalidArgument(format!(
                "value_and_grad: the function must return a scalar (0-D) array, got shape {:?}",
                value.dims()
            )));
        }
        let seed = ops::filled(&[], value.dtype(), |_| Canon::Int(1))?;
        let grads = reverse(std::slice::from_ref(&value), &[seed], &tracers)?;
        Ok((value, grads))
    }
}

/// Returns the gradient function of `f` with respect to the positional arguments at
/// `argnums` (the flat form of [`grad`]).
///
/// ```
/// use nuvec_core::{Array, Result, array, grad_flat};
///
/// # fn main() -> Result<()> {
/// let g = grad_flat(|xs: &[Array]| Ok(xs[0].multiply(&xs[1]).sum(..)), &[1]);
/// let grads = g(&[array![2.0], array![3.0]])?;
/// assert_eq!(grads[0].to_vec::<f32>()?, [2.0]);
/// # Ok(())
/// # }
/// ```
pub fn grad_flat<F>(f: F, argnums: &[usize]) -> impl Fn(&[Array]) -> Result<Vec<Array>>
where
    F: Fn(&[Array]) -> Result<Array>,
{
    let value_and_grad = value_and_grad_flat(f, argnums);
    move |args| Ok(value_and_grad(args)?.1)
}

/// A per-example placeholder for a batched `vmap` input: ops can be recorded on it, but
/// evaluating anything that depends on it fails (as in MLX).
fn placeholder(like: &Array, shape: &[usize], tracing: &Tracing) -> Result<Array> {
    let failed = Array::failed(
        Error::InvalidArgument("vmap: arrays cannot be evaluated inside vmap".into()),
        like.dtype(),
        &[],
    );
    let primitive = Primitive::Broadcast(shape.to_vec());
    Ok(Array::record_unchecked(primitive, vec![failed], Some(tracing.id))?.swap_remove(0))
}

/// Runs `f` batched over `args` (batch axes `axes`, `None` for unbatched arguments) and
/// returns its outputs with the batch axis first.
pub(crate) fn batch(
    f: &dyn Fn(&[Array]) -> Result<Vec<Array>>,
    args: &[Array],
    axes: &[Option<usize>],
) -> Result<Vec<Array>> {
    let mut size = None;
    for (a, axis) in args.iter().zip(axes) {
        let Some(axis) = *axis else { continue };
        let n = *a.dims().get(axis).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "vmap: axis {axis} out of range for shape {:?}",
                a.dims()
            ))
        })?;
        if size.is_some_and(|s| s != n) {
            return Err(Error::InvalidArgument(format!(
                "vmap: inconsistent batch sizes {} and {n}",
                size.unwrap_or(0)
            )));
        }
        size = Some(n);
    }
    let size = size
        .ok_or_else(|| Error::InvalidArgument("vmap: at least one input must be batched".into()))?;
    let tracing = Tracing::enter();
    let mut batched: NodeMap<Key, (Array, usize)> = NodeMap::default();
    let mut sources = NodeSet::default();
    let mut inputs = Vec::with_capacity(args.len());
    for (a, axis) in args.iter().zip(axes) {
        inputs.push(match *axis {
            Some(axis) => {
                let mut shape = a.dims().to_vec();
                shape.remove(axis);
                let p = placeholder(a, &shape, &tracing)?;
                sources.insert(Arc::as_ptr(&p.node));
                batched.insert(key(&p), (a.clone(), axis));
                p
            }
            None => a.clone(),
        });
    }
    let outputs = checked(f(&inputs)?)?;
    for step in tape(&outputs, &sources, false) {
        let (arrays, axes): (Vec<Array>, Vec<Option<usize>>) = step
            .inputs
            .iter()
            .map(|a| match batched.get(&key(a)) {
                Some((b, axis)) => (b.clone(), Some(*axis)),
                None => (a.clone(), None),
            })
            .unzip();
        let (results, axis) = step.primitive.vmap(&arrays, &axes)?;
        for (o, r) in step.outputs().iter().zip(results) {
            batched.insert(key(o), (r, axis));
        }
    }
    outputs
        .iter()
        .map(|o| match batched.get(&key(o)) {
            Some((b, axis)) => crate::batching::move_axis(b, *axis, 0),
            // Independent of every batched input: the same for each example.
            None => {
                let mut shape = vec![size];
                shape.extend_from_slice(o.dims());
                ops::broadcast_to(&ops::expand_dims(o, 0)?, &shape)
            }
        })
        .collect()
}

/// `axis` (negative counts from the end) checked against `ndim` dimensions.
fn normalize_axis(axis: isize, ndim: usize) -> Result<usize> {
    let a = if axis < 0 { axis + ndim as isize } else { axis };
    if (0..ndim as isize).contains(&a) {
        Ok(a as usize)
    } else {
        Err(Error::InvalidArgument(format!(
            "vmap: axis {axis} out of range for {ndim} dimensions"
        )))
    }
}

/// Returns `f` vectorized over a batch axis, over positional arrays (the flat form of
/// [`vmap`]).
///
/// `in_axes` gives each argument's batch axis (`None` for arguments shared by every
/// example) and `out_axes` each output's; empty slices mean axis 0 everywhere. Negative
/// axes count from the end. Outputs that do not depend on a batched argument are broadcast
/// along the batch axis. Arrays cannot be evaluated inside `f`.
///
/// ```
/// use nuvec_core::{Result, array, vmap_flat};
///
/// # fn main() -> Result<()> {
/// // A matrix-vector product per example, sharing the matrix.
/// let mv = vmap_flat(|xs| Ok(vec![xs[0].matmul(&xs[1])]), &[None, Some(0)], &[]);
/// let m = array![[1.0, 0.0], [0.0, 2.0]];
/// let vs = array![[1.0, 1.0], [3.0, 4.0]]; // two vectors
/// assert_eq!(mv(&[m, vs])?[0].to_vec::<f32>()?, [1.0, 2.0, 3.0, 8.0]);
/// # Ok(())
/// # }
/// ```
pub fn vmap_flat<F>(
    f: F,
    in_axes: &[Option<i32>],
    out_axes: &[i32],
) -> impl Fn(&[Array]) -> Result<Vec<Array>>
where
    F: Fn(&[Array]) -> Result<Vec<Array>>,
{
    let (in_axes, out_axes) = (in_axes.to_vec(), out_axes.to_vec());
    move |args| {
        let axes = if in_axes.is_empty() {
            vec![Some(0); args.len()]
        } else if in_axes.len() == args.len() {
            in_axes
                .iter()
                .zip(args)
                .map(|(axis, a)| {
                    axis.map(|axis| normalize_axis(axis as isize, a.ndim()))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            return Err(Error::InvalidArgument(format!(
                "vmap: {} input axes for {} arguments",
                in_axes.len(),
                args.len()
            )));
        };
        let outputs = batch(&f, args, &axes)?;
        if !out_axes.is_empty() && out_axes.len() != outputs.len() {
            return Err(Error::InvalidArgument(format!(
                "vmap: {} output axes for {} outputs",
                out_axes.len(),
                outputs.len()
            )));
        }
        outputs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let axis =
                    normalize_axis(out_axes.get(i).copied().unwrap_or(0) as isize, o.ndim())?;
                crate::batching::move_axis(o, 0, axis)
            })
            .collect()
    }
}

// ── Transforms over trees ───────────────────────────────────────────────────

/// A closure's return value: a tree `T`, or a `Result<T>`. So transform closures may return
/// an array directly (`|x: &Array| x.multiply(x).sum(..)`) or a `Result` (with `?` inside).
///
/// The associated type makes inference unambiguous: a closure returning `Result<Array>` has
/// `Ok = Array`, never `Ok = Result<Array>` (`Result` is not a tree).
pub trait IntoResult {
    /// The value.
    type Ok;
    /// The value, or the closure's error.
    fn into_result(self) -> Result<Self::Ok>;
}

impl<T: Tree> IntoResult for T {
    type Ok = T;
    fn into_result(self) -> Result<T> {
        Ok(self)
    }
}

impl<T> IntoResult for Result<T> {
    type Ok = T;
    fn into_result(self) -> Result<T> {
        self
    }
}

/// What a function differentiated by [`value_and_grad`] returns: the loss alone (an
/// [`Array`]) or the loss with auxiliary outputs (`(Array, A)` for any tree `A`, such as
/// logits or metrics), which are returned as values, without gradients.
pub trait GradOutput: Tree {
    /// The 0-D loss.
    fn loss(&self) -> &Array;
}

impl GradOutput for Array {
    fn loss(&self) -> &Array {
        self
    }
}

impl<A: Tree> GradOutput for (Array, A) {
    fn loss(&self) -> &Array {
        &self.0
    }
}

/// A tree's leaves and which of them are differentiated: trainable floating-point leaves.
/// Frozen parameters, buffers and integer parameters are not (they get zero gradients); a
/// plain `Array` leaf is always differentiated, so an integer or boolean one is an error
/// naming it (capture such data, or make it a frozen `Param`).
fn differentiated<T: Leaves + ?Sized>(op: &str, tree: &T) -> Result<(Vec<Array>, Vec<bool>)> {
    struct D(Vec<Array>, Vec<bool>, Option<usize>);
    impl Visitor for D {
        fn leaf(&mut self, _: &str, a: &Array, kind: LeafKind, trainable: bool) {
            let float = differentiable(a.dtype());
            if kind == LeafKind::Array && !float && self.2.is_none() {
                self.2 = Some(self.0.len());
            }
            self.1.push(trainable && float);
            self.0.push(a.clone());
        }
        fn paths(&self) -> bool {
            false
        }
    }
    let mut d = D(Vec::new(), Vec::new(), None);
    tree.visit(&mut Visit::new(&mut d));
    if let Some(i) = d.2 {
        // The error path: visit again for the leaf's path.
        struct Nth(usize, Option<String>);
        impl Visitor for Nth {
            fn leaf(&mut self, path: &str, _: &Array, _: LeafKind, _: bool) {
                if self.0 == 0 && self.1.is_none() {
                    self.1 = Some(path.to_owned());
                }
                self.0 = self.0.wrapping_sub(1);
            }
        }
        let mut nth = Nth(i, None);
        tree.visit(&mut Visit::new(&mut nth));
        let path = nth.1.unwrap_or_default();
        let dtype = d.0[i].dtype();
        return Err(Error::InvalidArgument(format!(
            "{op}: the array {path:?} is {dtype}; only floating-point and complex arrays can be \
             differentiated (capture it in the closure, or make it a frozen `Param`)"
        )));
    }
    Ok((d.0, d.1))
}

/// `primals` with tracers for the differentiated leaves, and those tracers.
fn traced(tracing: &Tracing, leaves: &[Array], mask: &[bool]) -> Result<(Vec<Array>, Vec<Array>)> {
    let mut inputs = Vec::with_capacity(leaves.len());
    let mut tracers = Vec::new();
    for (a, &d) in leaves.iter().zip(mask) {
        if d {
            let t = tracing.tracer(a)?;
            tracers.push(t.clone());
            inputs.push(t);
        } else {
            inputs.push(a.clone());
        }
    }
    Ok((inputs, tracers))
}

/// Gradients for every leaf: `grads` for the differentiated ones (in order), stride-0
/// zeros (one element, broadcast) for the others.
fn assemble(leaves: &[Array], mask: &[bool], grads: Vec<Array>) -> Vec<Array> {
    let mut grads = grads.into_iter();
    leaves
        .iter()
        .zip(mask)
        .map(|(a, &d)| match d {
            true => grads.next().expect("one gradient per differentiated leaf"),
            false => broadcast_zero(a),
        })
        .collect()
}

/// Checks that `given` (cotangents or tangents) has `like`'s paths, shapes and dtypes.
fn check_tree<A: Tree, B: Tree>(op: &str, what: &str, like: &A, given: &B) -> Result<()> {
    let (lp, gp) = (like.flatten(), given.flatten());
    if lp.len() != gp.len() {
        return Err(Error::InvalidArgument(format!(
            "{op}: {} {what} leaves for {} leaves",
            gp.len(),
            lp.len()
        )));
    }
    for ((p, a), (q, g)) in lp.iter().zip(&gp) {
        if p != q {
            return Err(Error::InvalidArgument(format!(
                "{op}: the {what} leaf {q:?} is not at the path {p:?}"
            )));
        }
        if a.shape() != g.shape() || a.dtype() != g.dtype() {
            return Err(Error::InvalidArgument(format!(
                "{op}: the {what} at {p:?} is {} {:?}, expected {} {:?}",
                g.dtype(),
                g.shape(),
                a.dtype(),
                a.shape()
            )));
        }
    }
    Ok(())
}

/// The value of `f` at `params` and its gradient with respect to the trainable leaves.
fn value_and_grad_tree<P: Tree, O: GradOutput>(
    f: impl FnOnce(&P) -> Result<O>,
    params: &P,
) -> Result<(O, P)> {
    let (leaves, mask) = differentiated("value_and_grad", params)?;
    let tracing = Tracing::enter();
    let (inputs, tracers) = traced(&tracing, &leaves, &mask)?;
    let out = f(&params.unflatten(inputs)?)?;
    let loss = out.loss().clone().check()?;
    if loss.ndim() != 0 {
        return Err(Error::InvalidArgument(format!(
            "value_and_grad: the function must return a scalar (0-D) loss, got shape {:?}",
            loss.dims()
        )));
    }
    let seed = ops::filled(&[], loss.dtype(), |_| Canon::Int(1))?;
    let grads = reverse(std::slice::from_ref(&loss), &[seed], &tracers)?;
    let grads = params.unflatten(assemble(&leaves, &mask, grads))?;
    Ok((out, grads))
}

/// Returns a function computing `f`'s value and its gradient with respect to the trainable
/// leaves of its argument, as a tree of the argument's type (`grads.layers[0].weight`).
///
/// `f` takes the differentiated tree (a tuple, a struct, a model) and returns the 0-D loss,
/// or `(loss, aux)` to also return auxiliary outputs (the return type decides); it may
/// return the value directly or a `Result`. Constants, such as a batch, are captured by
/// the closure. Write the closure's parameter type: `|p: &(Array, Array)|`, `|m: &Mlp|`
/// (Rust cannot infer it through the generic transform: E0282 otherwise).
///
/// Frozen parameters, buffers and integer parameters get zero gradients that allocate no
/// full-size memory (stride-0 broadcasts of one element); a plain integer or boolean
/// `Array` leaf is an error (capture such data in the closure instead).
///
/// ```
/// use nuvec_core::{Array, Result, array, value_and_grad};
///
/// # fn main() -> Result<()> {
/// let x = array![1.0, 2.0];
/// // loss(w, b) = sum((w x + b)^2), with `x` captured.
/// let loss = |(w, b): &(Array, Array)| {
///     let r = w.multiply(&x).add(b);
///     r.multiply(&r).sum(..)
/// };
/// let (value, (gw, gb)) = value_and_grad(loss)(&(array![1.0, 1.0], array![0.0, 0.0]))?;
/// assert_eq!(value.item::<f32>()?, 5.0);
/// assert_eq!(gw.to_vec::<f32>()?, [2.0, 8.0]); // 2 (w x + b) x
/// assert_eq!(gb.to_vec::<f32>()?, [2.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub fn value_and_grad<P, F, R>(f: F) -> impl Fn(&P) -> Result<(R::Ok, P)>
where
    P: Tree,
    F: Fn(&P) -> R,
    R: IntoResult,
    R::Ok: GradOutput,
{
    move |params| value_and_grad_tree(|p: &P| f(p).into_result(), params)
}

/// Returns the gradient function of `f` (a scalar function of a tree): the gradient with
/// respect to the trainable leaves, as a tree of the argument's type (see
/// [`value_and_grad`]).
///
/// Gradient functions can be differentiated again:
///
/// ```
/// use nuvec_core::{Array, Result, grad};
///
/// # fn main() -> Result<()> {
/// let d2 = grad(grad(|x: &Array| x.sin()));
/// let x = Array::from(1.0f32);
/// assert!((d2(&x)?.item::<f32>()? + 1f32.sin()).abs() < 1e-6); // −sin(x)
/// # Ok(())
/// # }
/// ```
pub fn grad<P, F, R>(f: F) -> impl Fn(&P) -> Result<P>
where
    P: Tree,
    F: Fn(&P) -> R,
    R: IntoResult<Ok = Array>,
{
    move |params| Ok(value_and_grad_tree(|p: &P| f(p).into_result(), params)?.1)
}

/// Reverse mode over trees: `f`'s output at `primals` and the vector-Jacobian product for
/// `cotangents` (a tree like the output: same paths, shapes and dtypes, else an error
/// naming the path), as a tree like `primals` (zeros for leaves that are not
/// differentiated).
///
/// ```
/// use nuvec_core::{Array, Result, array, vjp};
///
/// # fn main() -> Result<()> {
/// let x = array![1.0, 2.0, 3.0];
/// let (_, g) = vjp(|x: &Array| x.multiply(x), &x, &x.ones_like())?;
/// assert_eq!(g.to_vec::<f32>()?, [2.0, 4.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub fn vjp<P, O, F, R>(f: F, primals: &P, cotangents: &O) -> Result<(O, P)>
where
    P: Tree,
    O: Tree,
    F: FnOnce(&P) -> R,
    R: IntoResult<Ok = O>,
{
    let (leaves, mask) = differentiated("vjp", primals)?;
    let tracing = Tracing::enter();
    let (inputs, tracers) = traced(&tracing, &leaves, &mask)?;
    let out = f(&primals.unflatten(inputs)?).into_result()?;
    let outputs = checked(out.leaves())?;
    check_tree("vjp", "cotangent", &out, cotangents)?;
    let grads = reverse(&outputs, &cotangents.leaves(), &tracers)?;
    let grads = primals.unflatten(assemble(&leaves, &mask, grads))?;
    Ok((out.unflatten(outputs)?, grads))
}

/// Forward mode over trees: `f`'s output at `primals` and the Jacobian-vector product for
/// `tangents` (a tree like `primals`; tangents of leaves that are not differentiated are
/// ignored), as a tree like the output.
///
/// ```
/// use nuvec_core::{Array, Result, jvp};
///
/// # fn main() -> Result<()> {
/// let (y, t) = jvp(|x: &Array| x.sin(), &Array::from(0.0f32), &Array::from(1.0f32))?;
/// assert_eq!((y.item::<f32>()?, t.item::<f32>()?), (0.0, 1.0)); // cos(0)
/// # Ok(())
/// # }
/// ```
pub fn jvp<P, O, F, R>(f: F, primals: &P, tangents: &P) -> Result<(O, O)>
where
    P: Tree,
    O: Tree,
    F: FnOnce(&P) -> R,
    R: IntoResult<Ok = O>,
{
    check_tree("jvp", "tangent", primals, tangents)?;
    let (leaves, mask) = differentiated("jvp", primals)?;
    let tangents: Vec<Array> = tangents
        .leaves()
        .into_iter()
        .zip(&mask)
        .filter_map(|(t, &d)| d.then_some(t))
        .collect();
    let tracing = Tracing::enter();
    let (inputs, tracers) = traced(&tracing, &leaves, &mask)?;
    let out = f(&primals.unflatten(inputs)?).into_result()?;
    let outputs = checked(out.leaves())?;
    let out_tangents = forward(&outputs, &tracers, &tangents)?;
    Ok((out.unflatten(outputs)?, out.unflatten(out_tangents)?))
}

/// One batch axis per top-level argument of a [`Vmap`] (`None`: not batched): an `i32` or
/// `Option<i32>` (one axis for every argument) or a tuple of them (one per argument).
pub trait InAxes {
    /// The axes, one per top-level argument (one entry: every argument).
    fn axes(self) -> Vec<Option<i32>>;
}

/// One entry of an [`InAxes`] tuple: an `i32` or `Option<i32>`.
pub trait InAxis {
    /// The axis.
    fn axis(self) -> Option<i32>;
}

impl InAxis for i32 {
    fn axis(self) -> Option<i32> {
        Some(self)
    }
}

impl InAxis for Option<i32> {
    fn axis(self) -> Option<i32> {
        self
    }
}

impl InAxes for i32 {
    fn axes(self) -> Vec<Option<i32>> {
        vec![Some(self)]
    }
}

impl InAxes for Option<i32> {
    fn axes(self) -> Vec<Option<i32>> {
        vec![self]
    }
}

impl InAxes for Vec<Option<i32>> {
    fn axes(self) -> Vec<Option<i32>> {
        self
    }
}

/// One output axis per top-level output of a [`Vmap`]: an `i32` (every output) or a tuple
/// of them.
pub trait OutAxes {
    /// The axes, one per top-level output (one entry: every output).
    fn axes(self) -> Vec<i32>;
}

impl OutAxes for i32 {
    fn axes(self) -> Vec<i32> {
        vec![self]
    }
}

impl OutAxes for Vec<i32> {
    fn axes(self) -> Vec<i32> {
        self
    }
}

macro_rules! axes_tuples {
    ($(($($t:ident $i:tt),+)),*) => {$(
        impl<$($t: InAxis),+> InAxes for ($($t,)+) {
            fn axes(self) -> Vec<Option<i32>> {
                vec![$(self.$i.axis()),+]
            }
        }
        impl OutAxes for ($(axes_tuples!(@i32 $t),)+) {
            fn axes(self) -> Vec<i32> {
                vec![$(self.$i),+]
            }
        }
    )*};
    (@i32 $t:ident) => { i32 };
}

axes_tuples!(
    (A 0, B 1),
    (A 0, B 1, C 2),
    (A 0, B 1, C 2, D 3),
    (A 0, B 1, C 2, D 3, E 4),
    (A 0, B 1, C 2, D 3, E 4, F 5),
    (A 0, B 1, C 2, D 3, E 4, F 5, G 6),
    (A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7)
);

/// The top-level argument of each leaf: the index of its path's first segment among the
/// distinct first segments, in order; and how many there are.
fn top_level(paths: &[String]) -> (Vec<usize>, usize) {
    let mut firsts: Vec<&str> = Vec::new();
    let groups = paths
        .iter()
        .map(|p| {
            let first = p.split('.').next().unwrap_or("");
            match firsts.iter().position(|f| *f == first) {
                Some(i) => i,
                None => {
                    firsts.push(first);
                    firsts.len() - 1
                }
            }
        })
        .collect();
    (groups, firsts.len())
}

/// A function vectorized over batch axes ([`vmap`]): configure it with
/// [`Vmap::in_axes`]/[`Vmap::out_axes`], then call it with [`Vmap::call`] or turn it into a
/// closure with [`Vmap::into_fn`] (to compose with `compile` and the other transforms).
pub struct Vmap<I, O, F> {
    f: F,
    in_axes: Vec<Option<i32>>,
    out_axes: Vec<i32>,
    types: PhantomData<fn(&I) -> O>,
}

/// Returns `f` vectorized over a batch axis of its inputs (axis 0 of every top-level
/// argument and output by default; see [`Vmap::in_axes`] and [`Vmap::out_axes`]).
///
/// `f` takes a tree (an array, a tuple of arguments, a struct) and returns one. Outputs
/// that do not depend on a batched input are broadcast along the batch axis. Arrays cannot
/// be evaluated inside `f`.
///
/// ```
/// use nuvec_core::{Array, Result, array, vmap};
///
/// # fn main() -> Result<()> {
/// // A matrix-vector product per example, sharing the matrix.
/// let mv = vmap(|(m, v): &(Array, Array)| m.matmul(v)).in_axes((None, 0));
/// let m = array![[1.0, 0.0], [0.0, 2.0]];
/// let vs = array![[1.0, 1.0], [3.0, 4.0]]; // two vectors
/// assert_eq!(mv.call(&(m, vs))?.to_vec::<f32>()?, [1.0, 2.0, 3.0, 8.0]);
/// # Ok(())
/// # }
/// ```
pub fn vmap<I, O, F, R>(f: F) -> Vmap<I, O, F>
where
    I: Tree,
    O: Tree,
    F: Fn(&I) -> R,
    R: IntoResult<Ok = O>,
{
    Vmap {
        f,
        in_axes: vec![Some(0)],
        out_axes: vec![0],
        types: PhantomData,
    }
}

impl<I: Tree, O: Tree, F> Vmap<I, O, F> {
    /// The batch axis of each top-level argument (`None`: shared by every example), or one
    /// axis for all: `in_axes((None, 0))`, `in_axes(1)`. Negative axes count from the end.
    pub fn in_axes(mut self, axes: impl InAxes) -> Self {
        self.in_axes = axes.axes();
        self
    }

    /// The axis where each top-level output gets the batch dimension, or one for all.
    pub fn out_axes(mut self, axes: impl OutAxes) -> Self {
        self.out_axes = axes.axes();
        self
    }

    /// `f` batched over `inputs`.
    pub fn call<R>(&self, inputs: &I) -> Result<O>
    where
        F: Fn(&I) -> R,
        R: IntoResult<Ok = O>,
    {
        let named = inputs.flatten();
        let paths: Vec<String> = named.iter().map(|(p, _)| p.clone()).collect();
        let args: Vec<Array> = named.into_iter().map(|(_, a)| a).collect();
        let (groups, n) = top_level(&paths);
        let per_arg = |k: usize| match self.in_axes.as_slice() {
            [one] => Ok(*one),
            axes if axes.len() == n => Ok(axes[groups[k]]),
            axes => Err(Error::InvalidArgument(format!(
                "vmap: {} input axes for {n} top-level arguments",
                axes.len()
            ))),
        };
        let axes = args
            .iter()
            .enumerate()
            .map(|(k, a)| {
                per_arg(k)?
                    .map(|axis| normalize_axis(axis as isize, a.ndim()))
                    .transpose()
                    .map_err(|e| match e {
                        Error::InvalidArgument(m) => {
                            Error::InvalidArgument(format!("{m} (argument {:?})", paths[k]))
                        }
                        e => e,
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let template: RefCell<Option<O>> = RefCell::new(None);
        let body = |xs: &[Array]| -> Result<Vec<Array>> {
            let out = (self.f)(&inputs.unflatten(xs.iter().cloned())?).into_result()?;
            let leaves = out.leaves();
            *template.borrow_mut() = Some(out);
            Ok(leaves)
        };
        let outputs = batch(&body, &args, &axes)?;
        let template = template
            .into_inner()
            .ok_or_else(|| Error::InvalidArgument("vmap: the function did not run".into()))?;
        let out_paths = template.paths();
        let (out_groups, m) = top_level(&out_paths);
        let moved = outputs
            .iter()
            .enumerate()
            .map(|(k, o)| {
                let axis = match self.out_axes.as_slice() {
                    [one] => *one,
                    axes if axes.len() == m => axes[out_groups[k]],
                    axes => {
                        return Err(Error::InvalidArgument(format!(
                            "vmap: {} output axes for {m} top-level outputs",
                            axes.len()
                        )));
                    }
                };
                let axis = normalize_axis(axis as isize, o.ndim())?;
                crate::batching::move_axis(o, 0, axis)
            })
            .collect::<Result<Vec<_>>>()?;
        template.unflatten(moved)
    }

    /// The configured batched function as a closure, which composes with `compile`, `grad`
    /// and the other transforms.
    pub fn into_fn<R>(self) -> impl Fn(&I) -> Result<O>
    where
        F: Fn(&I) -> R,
        R: IntoResult<Ok = O>,
    {
        move |inputs| self.call(inputs)
    }
}
