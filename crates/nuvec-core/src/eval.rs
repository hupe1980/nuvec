//! Graph evaluation.

use std::sync::Arc;

use crate::array::{Data, Node, NodeMap, NodeSet, State};
use crate::gpu::GpuEval;
use crate::kernels::{self, Operand};
use crate::primitive::Primitive;
use crate::{Array, DeviceKind, Error, Result, view};

/// Evaluates `arrays` and everything they depend on, and waits for them. Each node runs at
/// most once, even across threads.
///
/// Nodes on the GPU stream run as Metal kernels where one exists, and on the CPU (over the
/// same unified memory) otherwise. Returns the first error among `arrays`. Errors are
/// sticky: a failed array, and anything computed from it, fails with the same error on
/// every later evaluation or readback.
pub fn eval(arrays: &[Array]) -> Result<()> {
    schedule(arrays).eval()
}

/// Starts evaluating `arrays` and returns without waiting for GPU work: reading the arrays
/// (or evaluating them) later waits for exactly the work they need. As in MLX, use it to
/// overlap building the next computation with the GPU running this one.
///
/// CPU work runs on the calling thread before this returns. Errors are sticky: an error
/// already known is returned; a GPU error is reported by the first read or evaluation of
/// an affected array. Submission blocks while too many command buffers are in flight.
///
/// ```
/// use nuvec_core::{Array, Result, async_eval};
///
/// # fn main() -> Result<()> {
/// let x = Array::from_slice(&[1.0f32, 2.0], &[2])?;
/// let y = &x * 3.0;
/// async_eval(std::slice::from_ref(&y))?;
/// // ... build the next computation here ...
/// assert_eq!(y.to_vec::<f32>()?, [3.0, 6.0]); // waits for `y` only
/// # Ok(())
/// # }
/// ```
pub fn async_eval(arrays: &[Array]) -> Result<()> {
    schedule(arrays).async_eval()
}

/// `arrays` with the work they need put in order, to be started later by
/// [`Scheduled::async_eval`] or [`Scheduled::eval`] with less delay: ordering a large
/// graph takes a while, which can then overlap other work, such as the GPU running the
/// previous step while the next one waits for its result.
///
/// The arrays are evaluated as [`async_eval`] or [`eval`] would at the time they are
/// started; nodes evaluated in between (by another evaluation) are skipped.
///
/// ```
/// use nuvec_core::{Array, Result, schedule};
///
/// # fn main() -> Result<()> {
/// let x = Array::from_slice(&[1.0f32, 2.0], &[2])?;
/// let y = &(&x * 3.0) + 1.0;
/// let scheduled = schedule(std::slice::from_ref(&y));
/// // ... wait for something else here ...
/// scheduled.async_eval()?;
/// assert_eq!(y.to_vec::<f32>()?, [4.0, 7.0]);
/// # Ok(())
/// # }
/// ```
pub fn schedule(arrays: &[Array]) -> Scheduled {
    Scheduled {
        arrays: arrays.to_vec(),
        order: topological_order(arrays),
    }
}

/// Arrays whose evaluation is put in order but not started ([`schedule`]).
pub struct Scheduled {
    arrays: Vec<Array>,
    order: Vec<Arc<Node>>,
}

impl std::fmt::Debug for Scheduled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduled")
            .field("arrays", &self.arrays)
            .field("nodes", &self.order.len())
            .finish()
    }
}

impl Scheduled {
    /// The scheduled arrays.
    pub fn arrays(&self) -> &[Array] {
        &self.arrays
    }

    /// Starts evaluating the arrays without waiting for GPU work, as [`async_eval`].
    pub fn async_eval(self) -> Result<()> {
        let arrays = self.arrays.clone();
        self.submit().commit();
        match arrays.iter().find_map(Array::error) {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Evaluates the arrays and waits for them, as [`eval`].
    pub fn eval(self) -> Result<()> {
        let arrays = self.arrays.clone();
        // Errors are reported through the arrays' slots below.
        let _ = self.submit().flush();
        wait(&arrays)
    }

    /// Encodes or computes every node in order; the caller commits (or flushes) the GPU
    /// work.
    fn submit(self) -> GpuEval {
        let mut gpu = GpuEval::new();
        for node in self.order {
            evaluate(&node, &mut gpu);
            // `node` is released here, so intermediates are freed as soon as their last
            // consumer has run (or, on the GPU, once their command buffer completes).
        }
        gpu
    }
}

/// Waits for `arrays`; the first error among them.
fn wait(arrays: &[Array]) -> Result<()> {
    for array in arrays {
        if let Err(err) = array.out().slot.wait() {
            return Err(err.clone());
        }
    }
    Ok(())
}

/// The widest breadth-first frontier when ordering a graph (MLX's `MLX_BFS_MAX_WIDTH`).
const BFS_MAX_WIDTH: usize = 20;

/// Unevaluated nodes reachable from `arrays`, inputs before consumers. Iterative, so graph
/// depth is limited only by memory.
///
/// As MLX's `eval_impl`: a tape is built breadth-first back from the outputs, a node
/// joining once all its consumers have, and run in reverse. Independent work (the query,
/// key and value projections of a layer, say) is then encoded side by side, so the GPU
/// runs it concurrently without barriers between. Past [`BFS_MAX_WIDTH`] pending nodes the
/// walk goes depth-first, which bounds how many intermediates are alive at once.
fn topological_order(arrays: &[Array]) -> Vec<Arc<Node>> {
    // Unevaluated nodes with their unevaluated inputs, depth-first.
    let mut inputs: NodeMap<*const Node, Vec<Arc<Node>>> = NodeMap::default();
    let mut done = NodeSet::default();
    let mut stack: Vec<Arc<Node>> = arrays.iter().map(|a| a.node.clone()).collect();
    while let Some(node) = stack.pop() {
        let key = Arc::as_ptr(&node);
        if inputs.contains_key(&key) || done.contains(&key) {
            continue;
        }
        let pending: Vec<Arc<Node>> = match &*node.lock() {
            State::Pending { inputs, .. } => inputs.iter().map(|a| a.node.clone()).collect(),
            State::Retained { .. } | State::Done => {
                done.insert(key);
                continue;
            }
        };
        stack.extend(pending.iter().cloned());
        inputs.insert(key, pending);
    }
    for list in inputs.values_mut() {
        list.retain(|n| !done.contains(&Arc::as_ptr(n)));
    }
    let roots: Vec<Arc<Node>> = arrays
        .iter()
        .map(|a| a.node.clone())
        .filter(|n| inputs.contains_key(&Arc::as_ptr(n)))
        .collect();
    // Consumers of each node (an input used twice counts twice; the outputs count once
    // each for the caller).
    let mut consumers: NodeMap<*const Node, usize> = NodeMap::default();
    for n in inputs.values().flatten().chain(&roots) {
        *consumers.entry(Arc::as_ptr(n)).or_default() += 1;
    }
    // The tape: `None` stands for the caller, whose inputs are `roots`.
    let mut tape: Vec<Option<Arc<Node>>> = vec![None];
    let mut deferred: Vec<(Option<Arc<Node>>, usize)> = Vec::new();
    let mut next = 0;
    while !consumers.is_empty() && (next < tape.len() || !deferred.is_empty()) {
        let (node, mut j) = if next < tape.len() {
            next += 1;
            (tape[next - 1].clone(), 0)
        } else {
            deferred.pop().expect("checked above")
        };
        let list = match &node {
            Some(n) => &inputs[&Arc::as_ptr(n)],
            None => &roots,
        };
        while j < list.len() {
            let key = Arc::as_ptr(&list[j]);
            if consumers.contains_key(&key) {
                if tape.len() - next >= BFS_MAX_WIDTH {
                    deferred.push((node.clone(), j));
                    break;
                }
                let left = consumers.get_mut(&key).expect("checked above");
                *left -= 1;
                if *left == 0 {
                    consumers.remove(&key);
                    tape.push(Some(list[j].clone()));
                }
            }
            j += 1;
        }
    }
    tape.into_iter().rev().flatten().collect()
}

/// Runs one node whose inputs are all evaluated, stores (or, for GPU work, defers) its
/// outputs, and detaches it (unless a running transform needs its graph).
fn evaluate(node: &Arc<Node>, gpu: &mut GpuEval) {
    let mut state = node.lock();
    let (primitive, inputs) = match std::mem::replace(&mut *state, State::Done) {
        State::Pending { primitive, inputs } => (primitive, inputs),
        evaluated => {
            *state = evaluated; // evaluated meanwhile by another thread
            return;
        }
    };
    #[cfg(test)]
    counter::inc();
    let mut inputs = inputs;
    let result = match donation(node, &primitive, &mut inputs, &mut *gpu) {
        Some(base) => in_place(node, &primitive, &inputs, base, gpu),
        None => compute(node, &primitive, &inputs, gpu),
    };
    match result {
        Ok((outputs, true)) => gpu.defer(node.clone(), outputs),
        Ok((outputs, false)) => {
            for (output, data) in node.outputs.iter().zip(outputs) {
                let _ = output.slot.set(Ok(data));
            }
        }
        Err(err) => {
            for output in &node.outputs {
                let _ = output.slot.set(Err(err.clone()));
            }
        }
    }
    if node.tracer && crate::transforms::tracing() {
        *state = State::Retained { primitive, inputs };
        return;
    }
    drop(state);
    drop(inputs); // detach: release the inputs outside the lock
}

/// An input to be written in place (MLX's donation), and its buffer: an update's base
/// (`slice_update`, `norm_rope` or `paged_write` into a cache), or the input of an
/// element-wise operation or softmax whose buffer can take the output (same dtype, shape
/// and contiguous layout), as MLX's element-wise kernels reuse their inputs.
fn donation(
    node: &Node,
    primitive: &Primitive,
    inputs: &mut [Array],
    gpu: &mut GpuEval,
) -> Option<(usize, Data)> {
    let device = node.stream.device;
    if node.tracer || device.kind == DeviceKind::Cpu || !gpu.available(device) {
        return None;
    }
    let writes_base = matches!(
        primitive,
        Primitive::SliceUpdate { .. }
            | Primitive::Fast(crate::fast::FastOp::NormRope { cache: true, .. })
            | Primitive::Fast(crate::fast::FastOp::PagedWrite)
    );
    if writes_base {
        return owned(&mut inputs[0], gpu).map(|data| (0, data));
    }
    let [out] = node.outputs.as_slice() else {
        return None;
    };
    for i in in_place_inputs(primitive, inputs, out) {
        if let Some(data) = owned(&mut inputs[i], gpu) {
            return Some((i, data));
        }
    }
    None
}

/// The inputs of an element-wise operation or softmax whose buffer the output may take:
/// the output's dtype (float, where every such kernel exists on the GPU) and shape.
fn in_place_inputs(
    primitive: &Primitive,
    inputs: &[Array],
    out: &crate::array::Output,
) -> Vec<usize> {
    use crate::{DType, UnaryOp};
    let candidates: &[usize] = match primitive {
        Primitive::Unary(op) if *op != UnaryOp::Conjugate => &[0],
        Primitive::Binary(_) => &[0, 1],
        Primitive::Select => &[1, 2],
        Primitive::Fast(crate::fast::FastOp::Softmax) => &[0],
        _ => &[],
    };
    let float = matches!(out.dtype, DType::F16 | DType::BF16 | DType::F32);
    if !float || out.size == 0 || out.size >= u32::MAX as usize {
        return Vec::new();
    }
    candidates
        .iter()
        .copied()
        .filter(|&i| inputs[i].dtype() == out.dtype && inputs[i].shape() == out.shape.as_slice())
        .collect()
}

/// The buffer of `input`, taken over to be written in place: only when the evaluation owns
/// it outright (its node has no other strong handle, so no other array, view or consumer
/// refers to it) and its data is the whole of a writable buffer with no other handle.
/// Ownership proves the write exclusive, so no reference counting of other users is
/// needed.
///
/// The input may be finished, still in flight in an earlier evaluation (pending work holds
/// nodes and data only weakly, so the in-flight copy is then the data's only handle;
/// taking it over under the lock that publishing takes means it is never published), or
/// computed earlier in this evaluation. The buffer's fences and the encoder's barriers
/// order the write after the work that produces and reads it.
fn owned(input: &mut Array, gpu: &mut GpuEval) -> Option<Data> {
    if input.output != 0 {
        return None;
    }
    let base = &mut input.node;
    let usable = |data: &Data, out: &crate::array::Output, handles: usize| {
        out.flags.row_contiguous
            && crate::gpu::writable(data, out)
            && Arc::strong_count(&data.buffer) == handles
    };
    // Produced earlier in this evaluation: taken over from it if nothing else refers to it.
    if let [out] = base.outputs.as_slice()
        && let Some(data) = gpu.take_pending(base, |data, handles| usable(data, out, handles))
    {
        return Some(data);
    }
    if Arc::strong_count(base) != 1 {
        return None;
    }
    // Finished: no pending work refers to it any more, so it is owned outright.
    if let Some(base) = Arc::get_mut(base) {
        let [out] = base.outputs.as_mut_slice() else {
            return None;
        };
        if let Some(Ok(data)) = out.slot.get() {
            // The slot and (if it was published from in-flight work) the in-flight copy.
            let handles = 1 + usize::from(out.inflight.get().is_some());
            if !usable(data, out, handles) {
                return None;
            }
            out.inflight.clear();
            return match out.slot.take() {
                Some(Ok(data)) => Some(data),
                _ => None,
            };
        }
    }
    // In flight: only weak references from pending work remain besides ours.
    let [out] = base.outputs.as_slice() else {
        return None;
    };
    if out.slot.get().is_some() {
        return None;
    }
    let mut state = out.inflight.lock();
    let crate::array::InFlightState::Set(data) = &*state else {
        return None;
    };
    if !usable(data, out, 1) {
        return None;
    }
    match std::mem::replace(&mut *state, crate::array::InFlightState::Donated) {
        crate::array::InFlightState::Set(data) => Some(data),
        _ => None,
    }
}

/// An update written into its donated base (input 0), or an element-wise operation or
/// softmax written into its donated input `at`.
fn in_place(
    node: &Node,
    primitive: &Primitive,
    inputs: &[Array],
    (at, base): (usize, Data),
    gpu: &mut GpuEval,
) -> Result<(Vec<Data>, bool)> {
    let (mut pending, mut foreign) = (false, Vec::new());
    let mut rest = Vec::with_capacity(inputs.len());
    for (i, a) in inputs.iter().enumerate() {
        rest.push(match i == at {
            true => None,
            false => Some(input_data(a, gpu, &mut pending, &mut foreign)?),
        });
    }
    let out = &node.outputs[0];
    let operands = |skip: usize| -> Vec<Operand<'_>> {
        inputs
            .iter()
            .zip(&rest)
            .skip(skip)
            .filter_map(|(a, d)| d.as_ref().map(|d| (a.out(), d)))
            .collect()
    };
    let data = match primitive {
        Primitive::Fast(crate::fast::FastOp::PagedWrite) => {
            gpu.paged_write_in_place(base, &operands(1), out)?
        }
        Primitive::Fast(crate::fast::FastOp::Softmax) => gpu.softmax_in_place(base, out)?,
        Primitive::Fast(op) => gpu.norm_rope_in_place(base, op, &operands(1), out)?,
        Primitive::SliceUpdate { .. } => {
            let slices = primitive.dim_slices(out.shape.len())?;
            gpu.slice_update_in_place(base, out, operands(1)[0], &slices)?
        }
        _ => {
            let metas: Vec<&crate::array::Output> = inputs.iter().map(Array::out).collect();
            let datas: Vec<Option<&Data>> = rest.iter().map(Option::as_ref).collect();
            gpu.elementwise_in_place(primitive, &metas, &datas, base, out)?
        }
    };
    Ok((vec![data], true))
}

/// An input's data: from this evaluation's GPU work, evaluated, or in flight in another
/// evaluation (then `pending`, and recorded in `foreign`).
fn input_data<'a>(
    input: &'a Array,
    gpu: &mut GpuEval,
    pending: &mut bool,
    foreign: &mut Vec<&'a Array>,
) -> Result<Data> {
    if let Some(d) = gpu.lookup(&input.node, input.output) {
        *pending = true;
        Ok(d)
    } else if let Some(result) = input.out().slot.get() {
        result.clone()
    } else if let Some(d) = input.out().inflight.get() {
        // Committed by another evaluation on the device timeline: GPU work encoded now
        // runs after it, so it may read the data at once.
        *pending = true;
        foreign.push(input);
        Ok(d)
    } else {
        // Produced by another evaluation (another thread, or an earlier `async_eval`):
        // submit our own GPU work first, so two evaluations never wait on each other.
        gpu.commit();
        input.out().slot.wait().clone()
    }
}

/// Computes a node's outputs. The flag is true if they depend on GPU work that has not
/// completed yet (and must be deferred until it has).
fn compute(
    node: &Node,
    primitive: &Primitive,
    inputs: &[Array],
    gpu: &mut GpuEval,
) -> Result<(Vec<Data>, bool)> {
    let mut data = Vec::with_capacity(inputs.len());
    let mut pending = false;
    // Inputs read from another evaluation's in-flight GPU work.
    let mut foreign = Vec::new();
    for input in inputs {
        data.push(input_data(input, gpu, &mut pending, &mut foreign)?);
    }
    let operands: Vec<Operand<'_>> = inputs
        .iter()
        .zip(&data)
        .map(|(a, d)| (a.out(), d))
        .collect();
    // Waits until foreign in-flight inputs are complete (for work that does not run on the
    // device timeline).
    let await_foreign = || -> Result<()> {
        for input in &foreign {
            input.out().slot.wait().clone()?;
        }
        Ok(())
    };
    if let Some(views) = view_outputs(primitive, &operands)? {
        // Views do no work: they are published with this evaluation's GPU work (which runs
        // after any foreign producer) or, without GPU work, once the producers complete.
        gpu.depend_on(&foreign);
        return Ok((views, pending));
    }
    let device = node.stream.device;
    if device.kind != DeviceKind::Cpu {
        if !gpu.available(device) {
            return Err(Error::UnsupportedDevice(device));
        }
        if gpu.supports(primitive, &operands, &node.outputs) {
            return Ok((gpu.run(primitive, &operands, &node.outputs)?, true));
        }
    }
    // The CPU reads inputs in place: wait for the GPU work that produces them.
    await_foreign()?;
    if pending {
        gpu.flush()?;
    }
    Ok((cpu(primitive, &operands)?, false))
}

/// Outputs of view primitives, which share their input's buffer (on every device), or
/// `None` for primitives that compute.
fn view_outputs(primitive: &Primitive, operands: &[Operand<'_>]) -> Result<Option<Vec<Data>>> {
    let share = |offset| {
        Some(vec![Data {
            buffer: operands[0].1.buffer.clone(),
            offset,
        }])
    };
    Ok(match primitive {
        Primitive::Transpose(_) | Primitive::Broadcast(_) | Primitive::StopGradient => {
            share(operands[0].1.offset)
        }
        Primitive::Slice { .. } => {
            let (meta, data) = operands[0];
            let (_, delta) =
                view::slice_geometry(&meta.strides, &primitive.dim_slices(meta.shape.len())?);
            share(data.offset + delta)
        }
        Primitive::Custom(call) => Some(
            operands[call.primals..]
                .iter()
                .map(|(_, data)| (*data).clone())
                .collect(),
        ),
        Primitive::Reshape(shape) => {
            let (meta, data) = operands[0];
            view::nocopy_reshape(&meta.shape, &meta.strides, shape).and_then(|_| share(data.offset))
        }
        _ => None,
    })
}

/// Computing primitives on the CPU.
fn cpu(primitive: &Primitive, operands: &[Operand<'_>]) -> Result<Vec<Data>> {
    let new = |buffer| Ok(vec![Data::new(buffer)]);
    match primitive {
        Primitive::Unary(op) => new(kernels::unary(*op, operands[0])?),
        Primitive::Binary(op) => new(kernels::binary(*op, operands[0], operands[1])?),
        Primitive::Select => new(kernels::select(operands[0], operands[1], operands[2])?),
        Primitive::AsType(dtype) => new(kernels::cast(operands[0], *dtype)?),
        Primitive::Reduce { op, axes } => new(kernels::reduce(*op, operands[0], axes)?),
        Primitive::ArgReduce { op, axis } => new(kernels::arg_reduce(*op, operands[0], *axis)?),
        Primitive::GatherAxis { axis } => {
            new(kernels::gather_axis(operands[0], operands[1], *axis)?)
        }
        Primitive::ScatterAxis { axis, reduce } => new(kernels::scatter_axis(
            operands[0],
            operands[1],
            operands[2],
            *axis,
            *reduce,
        )?),
        Primitive::Concatenate { axis } => {
            let mut shape = operands[0].0.shape.clone();
            shape[*axis] = operands.iter().map(|(m, _)| m.shape[*axis]).sum();
            new(kernels::concatenate(operands, &shape, *axis)?)
        }
        Primitive::Sort { axis } => new(kernels::sort(operands[0], *axis, false)?),
        Primitive::ArgSort { axis } => new(kernels::sort(operands[0], *axis, true)?),
        Primitive::Scan {
            op,
            axis,
            reverse,
            inclusive,
        } => new(kernels::scan(
            *op,
            operands[0],
            *axis,
            *reverse,
            *inclusive,
        )?),
        Primitive::Matmul => new(kernels::matmul(operands[0], operands[1])?),
        Primitive::AddMM { .. } => {
            let (alpha, beta) = primitive.addmm_factors();
            new(kernels::addmm(
                operands[0],
                operands[1],
                operands[2],
                alpha,
                beta,
            )?)
        }
        Primitive::RandomBits { shape, .. } => {
            let n: usize = shape.iter().product();
            new(kernels::random_bits(
                operands[0],
                n * primitive_width(primitive),
            )?)
        }
        Primitive::Fast(op) => op.eval_cpu(operands),
        Primitive::MetalKernel(_) => Err(Error::InvalidArgument(
            "metal_kernel: custom kernels run on the GPU only".into(),
        )),
        Primitive::Compiled(fused) => fused.eval_cpu(operands),
        Primitive::Quantize { group_size, bits } => {
            let (packed, scales, biases) =
                kernels::quantize(operands[0], *group_size, usize::from(*bits))?;
            Ok(vec![
                Data::new(packed),
                Data::new(scales),
                Data::new(biases),
            ])
        }
        Primitive::Dequantize { group_size, bits } => new(kernels::dequantize(
            operands[0],
            operands[1],
            operands[2],
            *group_size,
            usize::from(*bits),
        )?),
        Primitive::QuantizedMatmul {
            group_size,
            bits,
            transpose,
        } => new(kernels::quantized_matmul(
            operands,
            *group_size,
            usize::from(*bits),
            *transpose,
        )?),
        Primitive::SliceUpdate { .. } => new(kernels::slice_update(
            operands[0],
            operands[1],
            &primitive.dim_slices(operands[0].0.shape.len())?,
        )?),
        // A reshape that `view_outputs` could not express as a view copies.
        Primitive::Reshape(_) => new(kernels::contiguous(operands[0].0, operands[0].1)?),
        Primitive::Transpose(_)
        | Primitive::Broadcast(_)
        | Primitive::Slice { .. }
        | Primitive::StopGradient
        | Primitive::Custom(_) => {
            unreachable!("views are handled by view_outputs")
        }
        #[cfg(test)]
        Primitive::AddMultiply => {
            use crate::primitive::BinaryOp;
            Ok(vec![
                Data::new(kernels::binary(BinaryOp::Add, operands[0], operands[1])?),
                Data::new(kernels::binary(
                    BinaryOp::Multiply,
                    operands[0],
                    operands[1],
                )?),
            ])
        }
    }
}

fn primitive_width(primitive: &Primitive) -> usize {
    match primitive {
        Primitive::RandomBits { width, .. } => usize::from(*width),
        _ => unreachable!(),
    }
}

/// Per-thread count of evaluated nodes, for run-once tests.
#[cfg(test)]
pub(crate) mod counter {
    use std::cell::Cell;

    thread_local!(static EVALUATED: Cell<usize> = const { Cell::new(0) });

    pub(crate) fn inc() {
        EVALUATED.set(EVALUATED.get() + 1);
    }
    pub(crate) fn take() -> usize {
        EVALUATED.replace(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::live;

    #[test]
    fn shared_node_runs_once() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let shared = &x + &x;
        let left = &shared * &x;
        let right = &shared + &x;
        let top = &left + &right;
        counter::take();
        eval(&[top.clone(), left.clone()]).unwrap();
        assert_eq!(counter::take(), 4, "shared, left, right, top: once each");
        top.eval().unwrap();
        assert_eq!(counter::take(), 0, "already evaluated");
        assert_eq!(top.to_vec::<f32>().unwrap(), vec![5.0, 14.0]);
    }

    #[test]
    fn intermediates_freed() {
        // The live count is per thread; GPU buffers are released on Metal's completion
        // thread, so this checks the CPU path.
        crate::set_default_stream(crate::Stream::default_for(crate::Device::cpu()));
        let before = live::count();
        let mut x = Array::scalar(0i64);
        let one = Array::scalar(1i64);
        let steps = if cfg!(miri) { 100 } else { 10_000 };
        for _ in 0..steps {
            x = &x + &one;
        }
        x.eval().unwrap();
        drop(one);
        assert_eq!(
            live::count() - before,
            1,
            "only the result's buffer is alive"
        );
        assert_eq!(x.item::<i64>().unwrap(), steps as i64);
    }

    #[test]
    fn multi_output_once() {
        let a = Array::from_slice(&[2i32, 3], &[2]).unwrap();
        let b = Array::from_slice(&[5i32, 7], &[2]).unwrap();
        let mut outs = Array::from_op(Primitive::AddMultiply, vec![a, b]).unwrap();
        let product = outs.pop().unwrap();
        let sum = outs.pop().unwrap();
        counter::take();
        eval(&[sum.clone(), product.clone()]).unwrap();
        assert_eq!(counter::take(), 1);
        drop(sum);
        assert_eq!(product.to_vec::<i32>().unwrap(), vec![10, 21]);
    }

    #[test]
    fn slice_update_writes_a_strided_region() {
        // Base: a transposed view; region: rows 0..3 step 2, columns reversed.
        let base = crate::arange(0.0, 12.0, 1.0, crate::DType::I32)
            .unwrap()
            .reshape(&[4, 3])
            .unwrap()
            .t()
            .unwrap(); // [[0,3,6,9],[1,4,7,10],[2,5,8,11]]
        let update = Array::from_slice(&[-1i32, -2, -3, -4, -5, -6, -7, -8], &[2, 4]).unwrap();
        let primitive = Primitive::SliceUpdate {
            start: vec![0, 3],
            step: vec![2, -1],
            len: vec![2, 4],
        };
        let out = Array::from_op(primitive.clone(), vec![base.clone(), update.clone()])
            .unwrap()
            .remove(0);
        assert_eq!(
            out.to_vec::<i32>().unwrap(),
            [-4, -3, -2, -1, 1, 4, 7, 10, -8, -7, -6, -5]
        );
        let wrong = Array::from_slice(&[0i32; 3], &[3]).unwrap();
        assert!(Array::from_op(primitive, vec![base, wrong]).is_err());
    }

    #[test]
    fn tracer_nodes_keep_their_graph_only_while_tracing() {
        let x = Array::scalar(2.0f32);
        let kept = std::cell::Cell::new(false);
        let g = crate::grad(|x: &Array| {
            let y = x * x;
            y.eval()?;
            kept.set(matches!(&*y.node.lock(), State::Retained { .. }));
            Ok(y)
        });
        assert_eq!(g(&x).unwrap().item::<f32>().unwrap(), 4.0);
        assert!(kept.get(), "retained inside a transform");
        let traced = crate::vjp(
            |xs| Ok(vec![&xs[0] * &xs[0]]),
            &[x],
            &[Array::scalar(1.0f32)],
        )
        .unwrap()
        .0
        .remove(0);
        traced.eval().unwrap();
        assert!(
            matches!(&*traced.node.lock(), State::Done),
            "outside a transform"
        );
    }

    #[test]
    fn evaluation_detaches_inputs() {
        let a = Array::scalar(1u8);
        let c = &a + &a;
        assert!(c.primitive().is_some());
        c.eval().unwrap();
        assert_eq!(c.primitive(), None);
        assert_eq!(Arc::strong_count(&a.node), 1, "c no longer references a");
    }

    /// Positions of nodes in the evaluation order of `arrays`, checked to be topological.
    fn positions(arrays: &[Array]) -> NodeMap<*const Node, usize> {
        let order = topological_order(arrays);
        let at: NodeMap<*const Node, usize> = order
            .iter()
            .enumerate()
            .map(|(i, n)| (Arc::as_ptr(n), i))
            .collect();
        assert_eq!(at.len(), order.len(), "each node once");
        for (i, node) in order.iter().enumerate() {
            if let State::Pending { inputs, .. } = &*node.lock() {
                for input in inputs {
                    if let Some(&j) = at.get(&Arc::as_ptr(&input.node)) {
                        assert!(j < i, "inputs come first");
                    }
                }
            }
        }
        at
    }

    #[test]
    fn independent_work_is_ordered_side_by_side() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let (a1, b1) = (&x + 1.0, &x + 3.0);
        let (a2, b2) = (&a1 * 2.0, &b1 * 4.0);
        let top = &a2 + &b2;
        let at = positions(std::slice::from_ref(&top));
        let p = |a: &Array| at[&Arc::as_ptr(&a.node)];
        assert!(
            p(&a1).max(p(&b1)) < p(&a2).min(p(&b2)),
            "both chains advance together"
        );
        assert_eq!(top.to_vec::<f32>().unwrap(), [20.0, 26.0]);
    }

    #[test]
    fn wide_and_shared_graphs_are_ordered() {
        let x = Array::from_slice(&[1i32], &[1]).unwrap();
        let one = Array::from_slice(&[1i32], &[1]).unwrap();
        let shared = &x + &x;
        // 100 chains of different depths (wider than the breadth-first limit), joined.
        let mut joined = shared.clone();
        for i in 0..100 {
            let mut c = &shared + &Array::from_slice(&[i], &[1]).unwrap();
            for _ in 0..i % 7 {
                c = &c * &one;
            }
            joined = &joined + &c;
        }
        positions(&[joined.clone(), shared.clone(), joined.clone()]);
        counter::take();
        eval(&[joined.clone(), shared.clone()]).unwrap();
        let nodes = 1 + 100 * 2 + (0..100).map(|i| i % 7).sum::<usize>();
        assert_eq!(counter::take(), nodes, "every node once");
        assert_eq!(
            joined.item::<i32>().unwrap(),
            2 + 100 * 2 + (0..100).sum::<i32>()
        );
    }

    #[test]
    fn scheduled_evaluation_skips_work_done_meanwhile() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let shared = &x * &x;
        let top = &shared + &x;
        let scheduled = schedule(std::slice::from_ref(&top));
        assert_eq!(scheduled.arrays().len(), 1);
        counter::take();
        shared.eval().unwrap();
        scheduled.eval().unwrap();
        assert_eq!(counter::take(), 2, "shared once, then top");
        assert_eq!(top.to_vec::<f32>().unwrap(), [2.0, 6.0]);
    }
}
