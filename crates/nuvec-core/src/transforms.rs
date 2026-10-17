//! Function transforms: reverse-mode ([`vjp`], [`grad`], [`value_and_grad`]) and
//! forward-mode ([`jvp`]) automatic differentiation.
//!
//! As in MLX, the function runs on *tracer* inputs and records a lazy graph. The derivative
//! is another lazy graph, built from each primitive's rule, so transforms compose: the
//! gradient of a gradient is a second derivative. Tracers are zero-copy identity views of
//! the inputs. While a transform runs, evaluating a node that depends on a tracer keeps its
//! graph instead of detaching it, so calling `eval` inside the function is allowed.

use std::cell::Cell;
use std::collections::HashSet;
use std::sync::Arc;

use crate::array::{Node, NodeMap, NodeSet, State};
use crate::cast::Canon;
use crate::derivatives::zeros_like;
use crate::primitive::Primitive;
use crate::{Array, Error, Result, ops};

thread_local!(static DEPTH: Cell<usize> = const { Cell::new(0) });

/// Whether a transform is running on this thread.
pub(crate) fn tracing() -> bool {
    DEPTH.get() > 0
}

/// Marks this thread as tracing while alive (nests).
struct Tracing;

impl Tracing {
    fn enter() -> Self {
        DEPTH.set(DEPTH.get() + 1);
        Tracing
    }
}

impl Drop for Tracing {
    fn drop(&mut self) {
        DEPTH.set(DEPTH.get() - 1);
    }
}

/// A tracer for `primal`: an identity view (a broadcast to its own shape). It is
/// differentiable, so an enclosing transform differentiates through it.
fn tracer(primal: &Array) -> Result<Array> {
    let shape = primal.shape().to_vec();
    Ok(Array::record(Primitive::Broadcast(shape), vec![primal.clone()], true)?.swap_remove(0))
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
        State::Done => None,
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
        if a.shape() != g.shape() {
            return Err(Error::InvalidArgument(format!(
                "{op}: shape {:?} does not match {what} shape {:?}",
                a.shape(),
                g.shape()
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

/// Reverse mode: runs `f` on `primals` and returns its outputs together with the
/// vector-Jacobian product, one gradient per primal, for one cotangent per output.
///
/// The results are lazy arrays. A primal that the outputs do not depend on gets a zero
/// gradient.
///
/// ```
/// use nuvec_core::{Array, Result, multiply, vjp};
///
/// # fn main() -> Result<()> {
/// let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
/// let ones = Array::from_slice(&[1.0f32; 3], &[3])?;
/// let (_, grads) = vjp(|xs| Ok(vec![multiply(&xs[0], &xs[0])?]), &[x], &[ones])?;
/// assert_eq!(grads[0].to_vec::<f32>()?, [2.0, 4.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub fn vjp(
    f: impl FnOnce(&[Array]) -> Result<Vec<Array>>,
    primals: &[Array],
    cotangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    let _tracing = Tracing::enter();
    let tracers = primals.iter().map(tracer).collect::<Result<Vec<_>>>()?;
    let outputs = f(&tracers)?;
    check_shapes("vjp", "cotangent", &outputs, cotangents)?;
    let grads = reverse(&outputs, cotangents, &tracers)?;
    Ok((outputs, grads))
}

/// Forward mode: runs `f` on `primals` and returns its outputs together with the
/// Jacobian-vector product, one tangent per output, for one tangent per primal.
///
/// ```
/// use nuvec_core::{Array, Result, jvp, sin};
///
/// # fn main() -> Result<()> {
/// let x = Array::scalar(0.0f32);
/// let (_, tangents) = jvp(|xs| Ok(vec![sin(&xs[0])?]), &[x], &[Array::scalar(1.0f32)])?;
/// assert_eq!(tangents[0].item::<f32>()?, 1.0); // cos(0)
/// # Ok(())
/// # }
/// ```
pub fn jvp(
    f: impl FnOnce(&[Array]) -> Result<Vec<Array>>,
    primals: &[Array],
    tangents: &[Array],
) -> Result<(Vec<Array>, Vec<Array>)> {
    check_shapes("jvp", "tangent", primals, tangents)?;
    let _tracing = Tracing::enter();
    let tracers = primals.iter().map(tracer).collect::<Result<Vec<_>>>()?;
    let outputs = f(&tracers)?;
    let sources = tracers.iter().map(|t| Arc::as_ptr(&t.node)).collect();
    let mut tan: NodeMap<Key, Array> = tracers
        .iter()
        .zip(tangents)
        .map(|(t, v)| (key(t), v.clone()))
        .collect();
    for step in tape(&outputs, &sources, true) {
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
    let tangents = outputs
        .iter()
        .map(|o| tan.get(&key(o)).cloned().map_or_else(|| zeros_like(o), Ok))
        .collect::<Result<Vec<_>>>()?;
    Ok((outputs, tangents))
}

/// Returns a function computing `f`'s value and its gradients with respect to the
/// arguments at positions `argnums`.
///
/// `f` must return a 0-D array. Only the selected arguments are traced; the others are
/// treated as constants.
///
/// ```
/// use nuvec_core::{Array, Result, multiply, subtract, sum, value_and_grad};
///
/// # fn main() -> Result<()> {
/// // loss(w, x) = sum((w * x - 1)^2)
/// let loss = |xs: &[Array]| {
///     let r = subtract(multiply(&xs[0], &xs[1])?, 1.0)?;
///     sum(&multiply(&r, &r)?, &[], false)
/// };
/// let w = Array::from_slice(&[2.0f32], &[1])?;
/// let x = Array::from_slice(&[3.0f32], &[1])?;
/// let (value, grads) = value_and_grad(loss, &[0])(&[w, x])?;
/// assert_eq!(value.item::<f32>()?, 25.0);
/// assert_eq!(grads[0].to_vec::<f32>()?, [30.0]); // 2 (w x - 1) x
/// # Ok(())
/// # }
/// ```
pub fn value_and_grad<F>(
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
        }
        let _tracing = Tracing::enter();
        let mut inputs = args.to_vec();
        let mut tracers = Vec::with_capacity(argnums.len());
        for &i in &argnums {
            inputs[i] = tracer(&args[i])?;
            tracers.push(inputs[i].clone());
        }
        let value = f(&inputs)?;
        if value.ndim() != 0 {
            return Err(Error::InvalidArgument(format!(
                "value_and_grad: the function must return a scalar (0-D) array, got shape {:?}",
                value.shape()
            )));
        }
        let seed = ops::filled(&[], value.dtype(), |_| Canon::Int(1))?;
        let grads = reverse(std::slice::from_ref(&value), &[seed], &tracers)?;
        Ok((value, grads))
    }
}

/// Returns the gradient function of a scalar function of one array.
///
/// Gradient functions can be differentiated again:
///
/// ```
/// use nuvec_core::{Array, Result, grad, sin};
///
/// # fn main() -> Result<()> {
/// let d2 = grad(grad(|x: &Array| sin(x)));
/// let x = Array::scalar(1.0f32);
/// assert!((d2(&x)?.item::<f32>()? + 1f32.sin()).abs() < 1e-6); // −sin(x)
/// # Ok(())
/// # }
/// ```
pub fn grad<F>(f: F) -> impl Fn(&Array) -> Result<Array>
where
    F: Fn(&Array) -> Result<Array>,
{
    let value_and_grad = value_and_grad(move |xs: &[Array]| f(&xs[0]), &[0]);
    move |x| Ok(value_and_grad(std::slice::from_ref(x))?.1.swap_remove(0))
}

/// A per-example placeholder for a batched `vmap` input: ops can be recorded on it, but
/// evaluating anything that depends on it fails (as in MLX).
fn placeholder(like: &Array, shape: &[usize]) -> Result<Array> {
    let failed = Array::failed(
        Error::InvalidArgument("vmap: arrays cannot be evaluated inside vmap".into()),
        like.dtype(),
        &[],
    );
    let primitive = Primitive::Broadcast(shape.to_vec());
    Ok(Array::record_unchecked(primitive, vec![failed], true)?.swap_remove(0))
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
        let n = *a.shape().get(axis).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "vmap: axis {axis} out of range for shape {:?}",
                a.shape()
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
    let _tracing = Tracing::enter();
    let mut batched: NodeMap<Key, (Array, usize)> = NodeMap::default();
    let mut sources = NodeSet::default();
    let mut inputs = Vec::with_capacity(args.len());
    for (a, axis) in args.iter().zip(axes) {
        inputs.push(match *axis {
            Some(axis) => {
                let mut shape = a.shape().to_vec();
                shape.remove(axis);
                let p = placeholder(a, &shape)?;
                sources.insert(Arc::as_ptr(&p.node));
                batched.insert(key(&p), (a.clone(), axis));
                p
            }
            None => a.clone(),
        });
    }
    let outputs = f(&inputs)?;
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
                shape.extend_from_slice(o.shape());
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

/// Returns `f` vectorized over a batch axis.
///
/// `in_axes` gives each argument's batch axis (`None` for arguments shared by every
/// example) and `out_axes` each output's; empty slices mean axis 0 everywhere. Negative
/// axes count from the end. Outputs that do not depend on a batched argument are broadcast
/// along the batch axis. Arrays cannot be evaluated inside `f`.
///
/// ```
/// use nuvec_core::{Array, Result, matmul, vmap};
///
/// # fn main() -> Result<()> {
/// // A matrix-vector product per example, sharing the matrix.
/// let mv = vmap(|xs| Ok(vec![matmul(&xs[0], &xs[1])?]), &[None, Some(0)], &[]);
/// let m = Array::from_slice(&[1.0f32, 0.0, 0.0, 2.0], &[2, 2])?;
/// let vs = Array::from_slice(&[1.0f32, 1.0, 3.0, 4.0], &[2, 2])?; // two vectors
/// assert_eq!(mv(&[m, vs])?[0].to_vec::<f32>()?, [1.0, 2.0, 3.0, 8.0]);
/// # Ok(())
/// # }
/// ```
pub fn vmap<F>(
    f: F,
    in_axes: &[Option<isize>],
    out_axes: &[isize],
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
                .map(|(axis, a)| axis.map(|axis| normalize_axis(axis, a.ndim())).transpose())
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
                let axis = normalize_axis(out_axes.get(i).copied().unwrap_or(0), o.ndim())?;
                crate::batching::move_axis(o, 0, axis)
            })
            .collect()
    }
}
