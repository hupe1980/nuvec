//! Graph evaluation.

use std::sync::Arc;

use crate::array::{Data, Node, NodeSet, State};
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
    // Errors are reported through the arrays' slots below.
    let _ = submit(arrays).flush();
    for array in arrays {
        if let Err(err) = array.out().slot.wait() {
            return Err(err.clone());
        }
    }
    Ok(())
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
    submit(arrays).commit();
    match arrays.iter().find_map(Array::error) {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// Encodes or computes every unevaluated node `arrays` depend on; the caller commits (or
/// flushes) the GPU work.
fn submit(arrays: &[Array]) -> GpuEval {
    let mut gpu = GpuEval::new();
    for node in topological_order(arrays) {
        evaluate(&node, &mut gpu);
        // `node` is released here, so intermediates are freed as soon as their last
        // consumer has run (or, on the GPU, once their command buffer completes).
    }
    gpu
}

/// Unevaluated nodes reachable from `arrays`, inputs before consumers. Iterative, so graph
/// depth is limited only by memory.
fn topological_order(arrays: &[Array]) -> Vec<Arc<Node>> {
    let mut visited = NodeSet::default();
    let mut order = Vec::new();
    let mut stack: Vec<(Arc<Node>, bool)> =
        arrays.iter().map(|a| (a.node.clone(), false)).collect();
    while let Some((node, expanded)) = stack.pop() {
        if expanded {
            order.push(node);
            continue;
        }
        if !visited.insert(Arc::as_ptr(&node)) {
            continue;
        }
        let inputs: Vec<Arc<Node>> = match &*node.lock() {
            State::Pending { inputs, .. } => inputs.iter().map(|a| a.node.clone()).collect(),
            State::Retained { .. } | State::Done => continue,
        };
        stack.push((node, true));
        stack.extend(
            inputs
                .into_iter()
                .filter(|n| !visited.contains(&Arc::as_ptr(n)))
                .map(|n| (n, false)),
        );
    }
    order
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
    match compute(node, &primitive, &inputs, gpu) {
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
        if let Some(d) = gpu.lookup(&input.node, input.output) {
            pending = true;
            data.push(d);
        } else if let Some(result) = input.out().slot.get() {
            data.push(result.clone()?);
        } else if let Some(d) = input.out().inflight.get() {
            // Committed by another evaluation on the device timeline: GPU work encoded now
            // runs after it, so it may read the data at once.
            pending = true;
            foreign.push(input);
            data.push(d.clone());
        } else {
            // Produced by another evaluation (another thread, or an earlier `async_eval`):
            // submit our own GPU work first, so two evaluations never wait on each other.
            gpu.commit();
            data.push(input.out().slot.wait().clone()?);
        }
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
}
