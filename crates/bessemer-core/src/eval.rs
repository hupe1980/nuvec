//! Graph evaluation.

use std::collections::HashSet;
use std::sync::Arc;

use bessemer_cpu::binary_map;
use bytemuck::Pod;

use crate::array::{Node, State};
use crate::buffer::Buffer;
use crate::primitive::Primitive;
use crate::{Array, Complex64, DType, DeviceKind, Error, Result, bf16, f16};

/// Evaluates `arrays` and everything they depend on. Each node runs at most once, even
/// across threads.
///
/// Returns the first error among `arrays`. Errors are sticky: a failed array, and anything
/// computed from it, fails with the same error on every later evaluation or readback.
pub fn eval(arrays: &[Array]) -> Result<()> {
    for node in topological_order(arrays) {
        evaluate(&node);
        // `node` is released here, so intermediates are freed as soon as their last
        // consumer has run.
    }
    match arrays
        .iter()
        .find_map(|a| a.out().slot.get().and_then(|r| r.as_ref().err()))
    {
        Some(err) => Err(err.clone()),
        None => Ok(()),
    }
}

/// Unevaluated nodes reachable from `arrays`, inputs before consumers. Iterative, so graph
/// depth is limited only by memory.
fn topological_order(arrays: &[Array]) -> Vec<Arc<Node>> {
    let mut visited = HashSet::new();
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
            State::Done => continue,
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

/// Runs one node whose inputs are all evaluated, stores its outputs, and detaches it.
fn evaluate(node: &Node) {
    let mut state = node.lock();
    let State::Pending { primitive, inputs } = std::mem::replace(&mut *state, State::Done) else {
        return; // evaluated meanwhile by another thread
    };
    #[cfg(test)]
    counter::inc();
    match run(node, &primitive, &inputs) {
        Ok(buffers) => {
            for (output, buffer) in node.outputs.iter().zip(buffers) {
                let _ = output.slot.set(Ok(buffer));
            }
        }
        Err(err) => {
            for output in &node.outputs {
                let _ = output.slot.set(Err(err.clone()));
            }
        }
    }
    drop(state);
    drop(inputs); // detach: release the inputs outside the lock
}

fn run(node: &Node, primitive: &Primitive, inputs: &[Array]) -> Result<Vec<Buffer>> {
    let mut data = Vec::with_capacity(inputs.len());
    for input in inputs {
        match input
            .out()
            .slot
            .get()
            .expect("inputs are evaluated before consumers")
        {
            Ok(buffer) => data.push(buffer),
            Err(err) => return Err(err.clone()),
        }
    }
    if node.stream.device.kind != DeviceKind::Cpu {
        return Err(Error::UnsupportedDevice(node.stream.device));
    }
    let out = &node.outputs[0];
    Ok(match primitive {
        Primitive::Add => vec![binary(out.dtype, out.size, data[0], data[1], BinaryOp::Add)],
        Primitive::Multiply => vec![binary(
            out.dtype,
            out.size,
            data[0],
            data[1],
            BinaryOp::Multiply,
        )],
        #[cfg(test)]
        Primitive::AddMultiply => vec![
            binary(out.dtype, out.size, data[0], data[1], BinaryOp::Add),
            binary(out.dtype, out.size, data[0], data[1], BinaryOp::Multiply),
        ],
    })
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Multiply,
}

/// Element-wise binary operation on equal-length contiguous inputs.
///
/// Integers wrap around and booleans use OR/AND, matching NumPy and MLX.
fn binary(dtype: DType, size: usize, a: &Buffer, b: &Buffer, op: BinaryOp) -> Buffer {
    fn apply<T: Pod>(
        size: usize,
        a: &Buffer,
        b: &Buffer,
        op: BinaryOp,
        add: fn(T, T) -> T,
        mul: fn(T, T) -> T,
    ) -> Buffer {
        let mut out = Buffer::zeroed(size * size_of::<T>());
        let f = match op {
            BinaryOp::Add => add,
            BinaryOp::Multiply => mul,
        };
        binary_map(a.typed::<T>(), b.typed::<T>(), out.typed_mut::<T>(), f);
        out
    }
    match dtype {
        DType::Bool => apply::<u8>(size, a, b, op, |x, y| x | y, |x, y| x & y),
        DType::U8 => apply(size, a, b, op, u8::wrapping_add, u8::wrapping_mul),
        DType::U16 => apply(size, a, b, op, u16::wrapping_add, u16::wrapping_mul),
        DType::U32 => apply(size, a, b, op, u32::wrapping_add, u32::wrapping_mul),
        DType::U64 => apply(size, a, b, op, u64::wrapping_add, u64::wrapping_mul),
        DType::I8 => apply(size, a, b, op, i8::wrapping_add, i8::wrapping_mul),
        DType::I16 => apply(size, a, b, op, i16::wrapping_add, i16::wrapping_mul),
        DType::I32 => apply(size, a, b, op, i32::wrapping_add, i32::wrapping_mul),
        DType::I64 => apply(size, a, b, op, i64::wrapping_add, i64::wrapping_mul),
        DType::F16 => apply::<f16>(size, a, b, op, |x, y| x + y, |x, y| x * y),
        DType::BF16 => apply::<bf16>(size, a, b, op, |x, y| x + y, |x, y| x * y),
        DType::F32 => apply::<f32>(size, a, b, op, |x, y| x + y, |x, y| x * y),
        DType::F64 => apply::<f64>(size, a, b, op, |x, y| x + y, |x, y| x * y),
        DType::Complex64 => apply::<Complex64>(size, a, b, op, |x, y| x + y, |x, y| x * y),
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
    fn evaluation_detaches_inputs() {
        let a = Array::scalar(1u8);
        let c = &a + &a;
        assert_eq!(c.primitive(), Some(Primitive::Add));
        c.eval().unwrap();
        assert_eq!(c.primitive(), None);
        assert_eq!(Arc::strong_count(&a.node), 1, "c no longer references a");
    }
}
