//! GPU evaluation: encodes supported primitives as Metal kernels on unified memory.
//!
//! Outputs of encoded kernels stay *pending* (in a side table, not yet in the nodes' slots)
//! until their command buffer has completed. [`GpuEval::flush`] commits, waits, and then
//! publishes every pending output, or the GPU error, to its node. This keeps two invariants:
//! data in a slot is always complete, and a failed command marks exactly its outputs as
//! failed. Primitives without a GPU kernel fall back to the CPU after a flush (unified memory
//! needs no copies).

use std::sync::Arc;

use crate::array::{Data, Node, Output};
use crate::kernels::Operand;
use crate::primitive::Primitive;
use crate::{Device, Result};

#[cfg(all(target_os = "macos", not(miri)))]
pub(crate) use metal::GpuEval;
#[cfg(not(all(target_os = "macos", not(miri))))]
pub(crate) use stub::GpuEval;

/// Shared by both implementations: publish outputs to a node's slots.
fn publish(node: &Node, outputs: Vec<Data>, result: &Result<()>) {
    for (output, data) in node.outputs.iter().zip(outputs) {
        let _ = output.slot.set(match result {
            Ok(()) => Ok(data),
            Err(e) => Err(e.clone()),
        });
    }
}

#[cfg(not(all(target_os = "macos", not(miri))))]
mod stub {
    use super::*;

    /// No Metal on this target (or under Miri): GPU streams are unsupported.
    pub(crate) struct GpuEval;

    impl GpuEval {
        pub(crate) fn new() -> Self {
            GpuEval
        }
        pub(crate) fn available(&self, _: Device) -> bool {
            false
        }
        pub(crate) fn supports(&self, _: &Primitive, _: &[Operand<'_>], _: &[Output]) -> bool {
            false
        }
        pub(crate) fn run(
            &mut self,
            _: &Primitive,
            _: &[Operand<'_>],
            _: &[Output],
        ) -> Result<Vec<Data>> {
            unreachable!("no GPU kernels on this target")
        }
        pub(crate) fn defer(&mut self, node: Arc<Node>, outputs: Vec<Data>) {
            publish(&node, outputs, &Ok(()));
        }
        pub(crate) fn lookup(&self, _: &Arc<Node>, _: u32) -> Option<Data> {
            None
        }
        pub(crate) fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(all(target_os = "macos", not(miri)))]
mod metal {
    use crate::array::NodeMap;

    use nuvec_gpu::{Arg, CommandBuffer, Device as MetalDevice};

    use super::*;
    use crate::buffer::Buffer;
    use crate::primitive::{ArgReduceOp, BinaryOp, ReduceOp, UnaryOp};
    use crate::{Category, DType, DeviceKind, Error};

    /// Dispatches per command buffer before it is committed (bounded latency and memory).
    const MAX_DISPATCHES: usize = 256;

    pub(crate) struct GpuEval {
        device: Option<&'static MetalDevice>,
        cmd: Option<CommandBuffer>,
        dispatches: usize,
        pending: Vec<(Arc<Node>, Vec<Data>)>,
        index: NodeMap<*const Node, usize>,
    }

    fn backend(e: nuvec_gpu::GpuError) -> Error {
        Error::Backend(e.to_string())
    }

    /// How an element-wise operand maps onto the output's row-major index.
    enum Mode {
        /// Same layout as the output.
        Flat,
        /// A single element, broadcast.
        Scalar,
        /// Anything else.
        Strided,
    }

    fn mode(meta: &Output, out_shape: &[usize]) -> Mode {
        if meta.size == 1 {
            Mode::Scalar
        } else if meta.flags.row_contiguous && meta.shape.as_slice() == out_shape {
            Mode::Flat
        } else {
            Mode::Strided
        }
    }

    fn bytes<T: bytemuck::Pod>(x: &T) -> Vec<u8> {
        bytemuck::bytes_of(x).to_vec()
    }

    /// Shape as `i32`s, never empty (Metal rejects zero-length inline data).
    fn shape_bytes(shape: &[usize]) -> Vec<u8> {
        let v: Vec<i32> = if shape.is_empty() {
            vec![1]
        } else {
            shape.iter().map(|&d| d as i32).collect()
        };
        bytemuck::cast_slice(&v).to_vec()
    }

    fn strides_bytes(strides: &[isize]) -> Vec<u8> {
        let v: Vec<i64> = if strides.is_empty() {
            vec![0]
        } else {
            strides.iter().map(|&s| s as i64).collect()
        };
        bytemuck::cast_slice(&v).to_vec()
    }

    fn unary_name(op: UnaryOp) -> &'static str {
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
            UnaryOp::Conjugate => "conj",
        }
    }

    fn binary_name(op: BinaryOp) -> &'static str {
        match op {
            BinaryOp::Add => "add",
            BinaryOp::Subtract => "sub",
            BinaryOp::Multiply => "mul",
            BinaryOp::Divide => "div",
            BinaryOp::Maximum => "max",
            BinaryOp::Minimum => "min",
            BinaryOp::Equal => "eq",
            BinaryOp::NotEqual => "ne",
            BinaryOp::Less => "lt",
            BinaryOp::LessEqual => "le",
            BinaryOp::Greater => "gt",
            BinaryOp::GreaterEqual => "ge",
        }
    }

    fn is_float(d: DType) -> bool {
        matches!(d, DType::F16 | DType::BF16 | DType::F32)
    }

    /// The kernel family suffix for element size `bytes`.
    fn bits(bytes: usize) -> usize {
        bytes * 8
    }

    impl GpuEval {
        pub(crate) fn new() -> Self {
            GpuEval {
                device: None,
                cmd: None,
                dispatches: 0,
                pending: Vec::new(),
                index: NodeMap::default(),
            }
        }

        fn device(&mut self) -> Option<&'static MetalDevice> {
            if self.device.is_none() {
                self.device = MetalDevice::shared();
            }
            self.device
        }

        /// Whether `device` is the GPU this evaluator drives.
        pub(crate) fn available(&self, device: Device) -> bool {
            device.kind == DeviceKind::Gpu && device.index == 0 && MetalDevice::shared().is_some()
        }

        /// Whether a GPU kernel exists for `primitive` on these operands.
        pub(crate) fn supports(
            &self,
            primitive: &Primitive,
            operands: &[Operand<'_>],
            outputs: &[Output],
        ) -> bool {
            let fits = |m: &Output| m.size < u32::MAX as usize;
            if !operands
                .iter()
                .all(|(m, d)| fits(m) && d.buffer.metal().is_some())
                || !outputs.iter().all(fits)
            {
                return false;
            }
            let dtype = operands.first().map(|(m, _)| m.dtype);
            let not64 = |d: DType| d != DType::F64;
            match (primitive, dtype) {
                (Primitive::Unary(op), Some(d)) => {
                    not64(d)
                        && match op {
                            UnaryOp::Negative => d != DType::Bool,
                            UnaryOp::Abs => d != DType::Complex64,
                            UnaryOp::Conjugate => d == DType::Complex64,
                            _ => is_float(d),
                        }
                }
                (Primitive::Binary(op), Some(d)) => {
                    not64(d)
                        && match op {
                            BinaryOp::Divide => is_float(d) || d == DType::Complex64,
                            BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply => true,
                            BinaryOp::Equal | BinaryOp::NotEqual => true,
                            _ => d != DType::Complex64,
                        }
                }
                (Primitive::Select, _) => true,
                (Primitive::AsType(to), Some(d)) => not64(d) && not64(*to),
                (Primitive::Reshape(_), _) => true,
                (Primitive::Reduce { op, .. }, Some(d)) => match op {
                    ReduceOp::And | ReduceOp::Or => d == DType::Bool,
                    ReduceOp::Max | ReduceOp::Min => not64(d) && d != DType::Complex64,
                    ReduceOp::Sum | ReduceOp::Prod => {
                        not64(d) && !matches!(d.category(), Category::Bool | Category::Complex)
                    }
                },
                (Primitive::ArgReduce { .. }, Some(d)) => not64(d) && d != DType::Complex64,
                (Primitive::Matmul, Some(d)) => is_float(d),
                _ => false,
            }
        }

        /// Encodes `primitive` and returns its (pending) output data.
        pub(crate) fn run(
            &mut self,
            primitive: &Primitive,
            operands: &[Operand<'_>],
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            let out = &outputs[0];
            let mut buffer = Buffer::try_alloc(out.size * out.dtype.size())?;
            match primitive {
                Primitive::Unary(op) => {
                    let dt = operands[0].0.dtype;
                    self.elementwise(
                        &format!("nuvec_unary_{}_{}", unary_name(*op), dt.name()),
                        operands,
                        out,
                        &mut buffer,
                    )?;
                }
                Primitive::Binary(op) => {
                    let dt = operands[0].0.dtype;
                    self.elementwise(
                        &format!("nuvec_binary_{}_{}", binary_name(*op), dt.name()),
                        operands,
                        out,
                        &mut buffer,
                    )?;
                }
                Primitive::Select => {
                    let name = format!("nuvec_select_{}", bits(out.dtype.size()));
                    self.elementwise(&name, operands, out, &mut buffer)?;
                }
                Primitive::AsType(to) => {
                    let name = format!("nuvec_cast_{}_{}", operands[0].0.dtype.name(), to.name());
                    self.elementwise(&name, operands, out, &mut buffer)?;
                }
                Primitive::Reshape(_) => self.copy(operands[0], &mut buffer)?,
                Primitive::Reduce { op, axes } => {
                    self.reduce(*op, axes, operands[0], &mut buffer)?
                }
                Primitive::ArgReduce { op, axis } => {
                    self.arg_reduce(*op, *axis, operands[0], out, &mut buffer)?
                }
                Primitive::Matmul => self.matmul(operands, out, &mut buffer)?,
                _ => unreachable!("supports() rejects other primitives"),
            }
            Ok(vec![Data::new(buffer)])
        }

        /// Element-wise kernels: the `_c` variant when every operand is flat or a scalar,
        /// else `_g`.
        fn elementwise(
            &mut self,
            base: &str,
            operands: &[Operand<'_>],
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let modes: Vec<Mode> = operands.iter().map(|(m, _)| mode(m, &out.shape)).collect();
            let general = modes.iter().any(|m| matches!(m, Mode::Strided));
            let mut consts: Vec<Vec<u8>> = operands
                .iter()
                .map(|(_, d)| bytes(&(d.offset as i64)))
                .collect();
            let name = if general {
                let strides: Vec<&[isize]> =
                    operands.iter().map(|(m, _)| m.strides.as_slice()).collect();
                let (shape, collapsed) = collapse(&out.shape, &strides);
                consts.push(shape_bytes(&shape));
                consts.extend(collapsed.iter().map(|s| strides_bytes(s)));
                consts.push(bytes(&(shape.len() as i32)));
                format!("{base}_g")
            } else {
                let flags: u32 = modes
                    .iter()
                    .enumerate()
                    .map(|(i, m)| u32::from(matches!(m, Mode::Scalar)) << i)
                    .sum();
                consts.push(bytes(&flags));
                format!("{base}_c")
            };
            let inputs: Vec<&nuvec_gpu::Buffer> = operands
                .iter()
                .map(|(_, d)| d.buffer.metal().expect("checked"))
                .collect();
            let output = buffer.metal_mut().expect("device storage");
            let mut args: Vec<Arg<'_>> = inputs.into_iter().map(Arg::read).collect();
            args.push(Arg::write(output));
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.dispatch(&name, &args, out.size)
        }

        /// Gathers an operand into contiguous memory, walking its own logical shape (a
        /// reshape keeps the row-major element order of its input).
        fn copy(&mut self, (meta, data): Operand<'_>, buffer: &mut Buffer) -> Result<()> {
            let (shape, [strides]) = nuvec_cpu::collapse_dims(&meta.shape, [&meta.strides]);
            let consts = [
                bytes(&(data.offset as i64)),
                shape_bytes(&shape),
                strides_bytes(&strides),
                bytes(&(shape.len() as i32)),
            ];
            let name = format!("nuvec_copy_{}_g", bits(meta.dtype.size()));
            let input = data.buffer.metal().expect("checked");
            let mut args = vec![
                Arg::read(input),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.dispatch(&name, &args, meta.size)
        }

        fn reduce(
            &mut self,
            op: ReduceOp,
            axes: &[usize],
            (meta, data): Operand<'_>,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let (kept, reduced): (Vec<usize>, Vec<usize>) =
                (0..meta.shape.len()).partition(|d| !axes.contains(d));
            let pick = |dims: &[usize]| -> (Vec<usize>, Vec<isize>) {
                (
                    dims.iter().map(|&d| meta.shape[d]).collect(),
                    dims.iter().map(|&d| meta.strides[d]).collect(),
                )
            };
            let (ks, kt) = pick(&kept);
            let (rs, rt) = pick(&reduced);
            let (ks, [kt]) = nuvec_cpu::collapse_dims(&ks, [&kt]);
            let (rs, [rt]) = nuvec_cpu::collapse_dims(&rs, [&rt]);
            let groups: usize = ks.iter().product();
            let n: usize = rs.iter().product();
            let threads = n.next_power_of_two().clamp(1, 256);
            let name = format!(
                "nuvec_reduce_{}_{}",
                match op {
                    ReduceOp::Sum => "sum",
                    ReduceOp::Prod => "prod",
                    ReduceOp::Max => "max",
                    ReduceOp::Min => "min",
                    ReduceOp::And => "and",
                    ReduceOp::Or => "or",
                },
                meta.dtype.name()
            );
            let consts = [
                bytes(&(data.offset as i64)),
                shape_bytes(&ks),
                strides_bytes(&kt),
                bytes(&(ks.len() as i32)),
                shape_bytes(&rs),
                strides_bytes(&rt),
                bytes(&(rs.len() as i32)),
                bytes(&(n as u32)),
            ];
            let input = data.buffer.metal().expect("checked");
            let mut args = vec![
                Arg::read(input),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.dispatch_groups(&name, &args, [groups, 1, 1], [threads, 1, 1])
        }

        fn arg_reduce(
            &mut self,
            op: ArgReduceOp,
            axis: usize,
            (meta, data): Operand<'_>,
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let kept: Vec<usize> = (0..meta.shape.len()).filter(|&d| d != axis).collect();
            let ks: Vec<usize> = kept.iter().map(|&d| meta.shape[d]).collect();
            let kt: Vec<isize> = kept.iter().map(|&d| meta.strides[d]).collect();
            let (ks, [kt]) = nuvec_cpu::collapse_dims(&ks, [&kt]);
            let name = format!(
                "nuvec_{}_{}",
                if op == ArgReduceOp::Max {
                    "argmax"
                } else {
                    "argmin"
                },
                meta.dtype.name()
            );
            let consts = [
                bytes(&(data.offset as i64)),
                shape_bytes(&ks),
                strides_bytes(&kt),
                bytes(&(ks.len() as i32)),
                bytes(&(meta.shape[axis] as u32)),
                bytes(&(meta.strides[axis] as i64)),
            ];
            let input = data.buffer.metal().expect("checked");
            let mut args = vec![
                Arg::read(input),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.dispatch(&name, &args, out.size)
        }

        fn matmul(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let ((ma, da), (mb, db)) = (operands[0], operands[1]);
            let nd = ma.shape.len();
            let (m, k, n) = (ma.shape[nd - 2], ma.shape[nd - 1], mb.shape[nd - 1]);
            let batch_shape = &ma.shape[..nd - 2];
            let batch: usize = batch_shape.iter().product();
            let mnk: [i32; 3] = [m as i32, n as i32, k as i32];
            let mat = |meta: &Output, d: &Data| -> [i64; 3] {
                [
                    d.offset as i64,
                    meta.strides[nd - 2] as i64,
                    meta.strides[nd - 1] as i64,
                ]
            };
            let consts = [
                bytemuck::cast_slice(&mnk).to_vec(),
                bytemuck::cast_slice(&mat(ma, da)).to_vec(),
                bytemuck::cast_slice(&mat(mb, db)).to_vec(),
                shape_bytes(batch_shape),
                strides_bytes(&ma.strides[..nd - 2]),
                strides_bytes(&mb.strides[..nd - 2]),
                bytes(&(batch_shape.len() as i32)),
            ];
            let name = format!("nuvec_matmul_{}", out.dtype.name());
            let (ia, ib) = (
                da.buffer.metal().expect("checked"),
                db.buffer.metal().expect("checked"),
            );
            let mut args = vec![
                Arg::read(ia),
                Arg::read(ib),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.dispatch_groups(
                &name,
                &args,
                [n.div_ceil(16), m.div_ceil(16), batch],
                [16, 16, 1],
            )
        }

        fn command_buffer(&mut self) -> Result<&mut CommandBuffer> {
            if self.cmd.is_none() {
                let device = self
                    .device()
                    .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
                self.cmd = Some(device.command_buffer().map_err(backend)?);
            }
            Ok(self.cmd.as_mut().expect("just created"))
        }

        fn pipeline(&mut self, name: &str) -> Result<nuvec_gpu::Pipeline> {
            let device = self
                .device()
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            device.pipeline(name).map_err(backend)
        }

        fn dispatch(&mut self, name: &str, args: &[Arg<'_>], threads: usize) -> Result<()> {
            let pipeline = self.pipeline(name)?;
            self.command_buffer()?
                .dispatch(&pipeline, args, threads)
                .map_err(backend)?;
            self.counted()
        }

        fn dispatch_groups(
            &mut self,
            name: &str,
            args: &[Arg<'_>],
            groups: [usize; 3],
            threads: [usize; 3],
        ) -> Result<()> {
            let pipeline = self.pipeline(name)?;
            self.command_buffer()?
                .dispatch_groups(&pipeline, args, groups, threads)
                .map_err(backend)?;
            self.counted()
        }

        /// Commits once a command buffer holds enough work.
        fn counted(&mut self) -> Result<()> {
            self.dispatches += 1;
            if self.dispatches >= MAX_DISPATCHES {
                self.commit_and_wait()
            } else {
                Ok(())
            }
        }

        fn commit_and_wait(&mut self) -> Result<()> {
            self.dispatches = 0;
            match self.cmd.take() {
                Some(cmd) => cmd.commit().wait().map_err(backend),
                None => Ok(()),
            }
        }

        /// Holds `outputs` of `node` until the GPU work producing them has completed.
        pub(crate) fn defer(&mut self, node: Arc<Node>, outputs: Vec<Data>) {
            self.index.insert(Arc::as_ptr(&node), self.pending.len());
            self.pending.push((node, outputs));
        }

        /// The pending output `output` of `node`, if it was encoded in this evaluation.
        pub(crate) fn lookup(&self, node: &Arc<Node>, output: u32) -> Option<Data> {
            let i = *self.index.get(&Arc::as_ptr(node))?;
            self.pending[i].1.get(output as usize).cloned()
        }

        /// Completes all encoded work and publishes pending outputs (or the GPU error).
        pub(crate) fn flush(&mut self) -> Result<()> {
            let result = self.commit_and_wait();
            for (node, outputs) in self.pending.drain(..) {
                publish(&node, outputs, &result);
            }
            self.index.clear();
            result
        }
    }

    /// Merges dimensions contiguous for every operand (fewer index divisions per element).
    fn collapse(shape: &[usize], strides: &[&[isize]]) -> (Vec<usize>, Vec<Vec<isize>>) {
        match strides {
            [a] => {
                let (s, [x]) = nuvec_cpu::collapse_dims(shape, [a]);
                (s, vec![x])
            }
            [a, b] => {
                let (s, [x, y]) = nuvec_cpu::collapse_dims(shape, [a, b]);
                (s, vec![x, y])
            }
            [a, b, c] => {
                let (s, [x, y, z]) = nuvec_cpu::collapse_dims(shape, [a, b, c]);
                (s, vec![x, y, z])
            }
            _ => unreachable!("element-wise kernels take one to three operands"),
        }
    }

    impl Drop for GpuEval {
        /// Never leave encoded work unpublished (for example after a panic).
        fn drop(&mut self) {
            let _ = self.flush();
        }
    }
}
