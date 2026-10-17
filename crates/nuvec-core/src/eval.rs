//! Graph evaluation.

use std::sync::Arc;

use crate::array::{Data, Node, NodeSet, State};
use crate::gpu::GpuEval;
use crate::kernels::{self, Operand};
use crate::primitive::Primitive;
use crate::{Array, DeviceKind, Error, Result, view};

/// Evaluates `arrays` and everything they depend on. Each node runs at most once, even
/// across threads.
///
/// Nodes on the GPU stream run as Metal kernels where one exists, and on the CPU (over the
/// same unified memory) otherwise. Returns the first error among `arrays`. Errors are
/// sticky: a failed array, and anything computed from it, fails with the same error on
/// every later evaluation or readback.
pub fn eval(arrays: &[Array]) -> Result<()> {
    let mut gpu = GpuEval::new();
    for node in topological_order(arrays) {
        evaluate(&node, &mut gpu);
        // `node` is released here, so intermediates are freed as soon as their last
        // consumer has run (or, on the GPU, once their command buffer completes).
    }
    // GPU errors are published to the nodes they affect; reported through the slots below.
    let _ = gpu.flush();
    for array in arrays {
        if let Err(err) = array.out().slot.wait() {
            return Err(err.clone());
        }
    }
    Ok(())
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
    for input in inputs {
        if let Some(d) = gpu.lookup(&input.node, input.output) {
            pending = true;
            data.push(d);
        } else if let Some(result) = input.out().slot.get() {
            data.push(result.clone()?);
        } else {
            // Produced by another thread's evaluation: finish our own GPU work first, so two
            // evaluations never wait on each other.
            gpu.flush()?;
            data.push(input.out().slot.wait().clone()?);
        }
    }
    let operands: Vec<Operand<'_>> = inputs
        .iter()
        .zip(&data)
        .map(|(a, d)| (a.out(), d))
        .collect();
    if let Some(views) = view_outputs(primitive, &operands)? {
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
        Primitive::Matmul => new(kernels::matmul(operands[0], operands[1])?),
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
