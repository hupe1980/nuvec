//! GPU evaluation: encodes supported primitives as Metal kernels on unified memory.
//!
//! Work is committed in chunks without waiting; an evaluation's chunks are ordered on the GPU
//! by a timeline (D-031). Outputs of encoded kernels stay *pending* (in a side table, not yet
//! in the nodes' slots) until their chunk has completed; completion handlers then publish
//! them, chunk by chunk in commit order, so data in a slot is always complete and a failure
//! marks the outputs of its chunk and of every later chunk of the evaluation as failed.
//! Primitives without a GPU kernel fall back to the CPU after [`GpuEval::flush`], which waits
//! for this evaluation's outstanding work (unified memory needs no copies).

use std::sync::Arc;

use crate::array::{Data, Node, Output};
use crate::kernels::Operand;
use crate::primitive::Primitive;
use crate::{Device, Result};

/// Whether a Metal GPU is available to evaluate on.
pub(crate) fn available() -> bool {
    #[cfg(all(target_os = "macos", not(miri)))]
    return nuvec_gpu::Device::shared().is_some();
    #[cfg(not(all(target_os = "macos", not(miri))))]
    false
}

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
        pub(crate) fn commit(&mut self) {}
        pub(crate) fn depend_on(&mut self, _: &[&crate::Array]) {}
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

    use std::sync::{Mutex, PoisonError};

    use nuvec_gpu::{Arg, CommandBuffer, Device as MetalDevice, Pending};

    use super::*;
    use crate::buffer::Buffer;
    use crate::primitive::{ArgReduceOp, BinaryOp, ReduceOp, UnaryOp};
    use crate::{Category, DType, DeviceKind, Error};

    /// Dispatches per command buffer before it is committed.
    const MAX_DISPATCHES: usize = 256;
    /// Command buffers in flight per device before submission blocks (MLX: 10).
    const MAX_IN_FLIGHT: usize = 10;

    type Chunk = Vec<(Arc<Node>, Vec<Data>)>;

    /// Completion state shared with an evaluation's completion handlers.
    #[derive(Default)]
    struct Shared {
        state: Mutex<Publishing>,
    }

    #[derive(Default)]
    struct Publishing {
        /// Per committed chunk: its outputs, until published.
        chunks: Vec<Option<Chunk>>,
        /// Per committed chunk: its result, once complete.
        results: Vec<Option<Result<()>>>,
        /// Chunks published so far (in commit order).
        next: usize,
        /// The first failure; it fails every later chunk too.
        error: Option<Error>,
    }

    /// Lets later evaluations' GPU work read these outputs while they are in flight.
    fn expose(chunk: &Chunk) {
        for (node, outputs) in chunk {
            for (output, data) in node.outputs.iter().zip(outputs) {
                let _ = output.inflight.set(data.clone());
            }
        }
    }

    /// The first GPU command failure on this device. GPU work may read other evaluations'
    /// in-flight results, so once a command fails every later GPU result is reported as
    /// failed too (D-032), as in MLX, where a command-buffer error ends the stream.
    static DEVICE_FAILURE: Mutex<Option<Error>> = Mutex::new(None);

    fn device_result(result: Result<()>) -> Result<()> {
        let mut failure = DEVICE_FAILURE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let (Err(e), None) = (&result, &*failure) {
            *failure = Some(Error::Backend(format!("a GPU command failed earlier: {e}")));
        }
        match &*failure {
            Some(e) if result.is_ok() => Err(e.clone()),
            _ => result,
        }
    }

    impl Shared {
        /// Records chunk `k`'s result and publishes every chunk whose predecessors are all
        /// complete, in order.
        fn complete(&self, k: usize, result: Result<()>) {
            let mut p = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            p.results[k] = Some(result);
            while let Some(Some(result)) = p.results.get(p.next).cloned() {
                if let (Err(e), None) = (&result, &p.error) {
                    p.error = Some(e.clone());
                }
                let result = p.error.clone().map_or(Ok(()), Err);
                let next = p.next;
                for (node, outputs) in p.chunks[next].take().unwrap_or_default() {
                    publish(&node, outputs, &result);
                }
                p.next += 1;
            }
        }
    }

    pub(crate) struct GpuEval {
        device: Option<&'static MetalDevice>,
        cmd: Option<CommandBuffer>,
        dispatches: usize,
        /// Outputs encoded into the open command buffer.
        chunk: Chunk,
        /// Every pending output of this evaluation, for consumers encoded later.
        index: NodeMap<*const Node, Vec<Data>>,
        shared: Arc<Shared>,
        /// This evaluation's committed command buffers.
        submitted: Vec<Pending>,
        /// Other evaluations' in-flight outputs that deferred views of the open chunk read.
        foreign: Vec<crate::Array>,
    }

    fn backend(e: nuvec_gpu::GpuError) -> Error {
        Error::Backend(e.to_string())
    }

    /// Elements one threadgroup folds per slice in a split reduction.
    const SLICE: usize = 4096;

    /// Slices per output for a reduction of `n` elements into `groups` outputs: several
    /// when there are too few outputs to occupy the GPU, each slice non-empty.
    fn slices(groups: usize, n: usize) -> usize {
        if groups >= 64 || n <= 2 * SLICE {
            return 1;
        }
        let wanted = n.div_ceil(SLICE).min(1024 / groups.max(1)).max(1);
        n.div_ceil(n.div_ceil(wanted))
    }

    /// Kept and reduced dimensions of a reduction's input.
    struct Layout<'a> {
        offset: isize,
        ks: &'a [usize],
        kt: &'a [isize],
        rs: &'a [usize],
        rt: &'a [isize],
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
            UnaryOp::ErfInv => "erfinv",
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
                chunk: Vec::new(),
                index: NodeMap::default(),
                shared: Arc::default(),
                submitted: Vec::new(),
                foreign: Vec::new(),
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
            let input = data.buffer.metal().expect("checked");
            let slices = slices(groups, n);
            if slices == 1 {
                let out = buffer.metal_mut().expect("device storage");
                let layout = Layout {
                    offset: data.offset,
                    ks: &ks,
                    kt: &kt,
                    rs: &rs,
                    rt: &rt,
                };
                return self.reduce_launch(&name, input, &layout, n, 1, out);
            }
            // Pass 1 folds slices of each output's range into partial results; pass 2 folds
            // the partial results (contiguous, `slices` per output).
            let mut partial = Buffer::try_alloc(groups * slices * meta.dtype.size())?;
            let layout = Layout {
                offset: data.offset,
                ks: &ks,
                kt: &kt,
                rs: &rs,
                rt: &rt,
            };
            let pass1 = partial.metal_mut().expect("device storage");
            self.reduce_launch(&name, input, &layout, n, slices, pass1)?;
            let (ks2, kt2, rs2, rt2) = ([groups], [slices as isize], [slices], [1isize]);
            let layout = Layout {
                offset: 0,
                ks: &ks2,
                kt: &kt2,
                rs: &rs2,
                rt: &rt2,
            };
            let out = buffer.metal_mut().expect("device storage");
            self.reduce_launch(
                &name,
                partial.metal().expect("device storage"),
                &layout,
                slices,
                1,
                out,
            )
        }

        /// One reduction launch over a `[outputs, slices]` grid of threadgroups.
        fn reduce_launch(
            &mut self,
            name: &str,
            input: &nuvec_gpu::Buffer,
            layout: &Layout<'_>,
            n: usize,
            slices: usize,
            out: &mut nuvec_gpu::Buffer,
        ) -> Result<()> {
            let groups: usize = layout.ks.iter().product();
            let chunk = n.div_ceil(slices).max(1);
            let threads = chunk.next_power_of_two().clamp(1, 256);
            let consts = [
                bytes(&(layout.offset as i64)),
                shape_bytes(layout.ks),
                strides_bytes(layout.kt),
                bytes(&(layout.ks.len() as i32)),
                shape_bytes(layout.rs),
                strides_bytes(layout.rt),
                bytes(&(layout.rs.len() as i32)),
                bytes(&(n as u32)),
                bytes(&(chunk as u32)),
            ];
            let mut args = vec![Arg::read(input), Arg::write(out)];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.dispatch_groups(name, &args, [groups, slices, 1], [threads, 1, 1])
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
                self.cmd = Some(device.ordered_command_buffer().map_err(backend)?);
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
                self.commit();
            }
            Ok(())
        }

        /// Submits the open command buffer, if any, without waiting; its outputs are
        /// published when it completes. Blocks while too many command buffers are in flight.
        pub(crate) fn commit(&mut self) {
            self.dispatches = 0;
            let Some(cmd) = self.cmd.take() else {
                // Outputs deferred without GPU work (views of pending data) still need a
                // chunk, published after the work they view.
                if self.chunk.is_empty() {
                    return;
                }
                // No GPU work orders these views after their producers: wait for them.
                let result = std::mem::take(&mut self.foreign)
                    .iter()
                    .try_for_each(|a| a.out().slot.wait().clone().map(|_| ()));
                let committed = self.chunk.clone();
                let k = self.reserve();
                self.shared.complete(k, result);
                expose(&committed);
                return;
            };
            // This command buffer runs after every foreign producer (device timeline).
            self.foreign.clear();
            if let Some(device) = self.device {
                device.wait_for_capacity(MAX_IN_FLIGHT);
            }
            let committed = self.chunk.clone();
            let k = self.reserve();
            let shared = self.shared.clone();
            let pending = cmd.commit_then(move |result| {
                shared.complete(k, device_result(result.map_err(backend)));
            });
            self.submitted.push(pending);
            // Only now, after the commit, may other evaluations encode work that reads it.
            expose(&committed);
        }

        /// Registers the open chunk as committed chunk `k`.
        fn reserve(&mut self) -> usize {
            let mut p = self
                .shared
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            p.chunks.push(Some(std::mem::take(&mut self.chunk)));
            p.results.push(None);
            p.chunks.len() - 1
        }

        /// Records that the open chunk reads other evaluations' in-flight `outputs`.
        pub(crate) fn depend_on(&mut self, outputs: &[&crate::Array]) {
            self.foreign.extend(outputs.iter().map(|&a| a.clone()));
        }

        /// Holds `outputs` of `node` until the GPU work producing them has completed.
        pub(crate) fn defer(&mut self, node: Arc<Node>, outputs: Vec<Data>) {
            self.index.insert(Arc::as_ptr(&node), outputs.clone());
            self.chunk.push((node, outputs));
        }

        /// The pending output `output` of `node`, if it was encoded in this evaluation.
        /// GPU consumers may use it at once (the timeline orders them after its producer);
        /// CPU consumers must [`flush`](Self::flush) first.
        pub(crate) fn lookup(&self, node: &Arc<Node>, output: u32) -> Option<Data> {
            self.index
                .get(&Arc::as_ptr(node))?
                .get(output as usize)
                .cloned()
        }

        /// Commits and waits until every chunk of this evaluation is complete and published,
        /// completing them on this thread (lower latency than waiting for Metal's handlers).
        /// Returns the evaluation's first GPU error.
        pub(crate) fn flush(&mut self) -> Result<()> {
            self.commit();
            for pending in self.submitted.drain(..) {
                pending.wait();
            }
            let p = self
                .shared
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            p.error.clone().map_or(Ok(()), Err)
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
        /// Never leave encoded work unsubmitted (for example after a panic).
        fn drop(&mut self) {
            self.commit();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::Array;

        fn pending() -> (Arc<Node>, Vec<Data>) {
            let x = Array::scalar(1.0f32);
            let y = crate::negative(&x).unwrap();
            let data = x.out().slot.get().unwrap().clone().unwrap();
            (y.node.clone(), vec![data])
        }

        #[test]
        fn chunks_publish_in_order_and_failures_poison_later_chunks() {
            let shared = Shared::default();
            let nodes: Vec<_> = (0..3).map(|_| pending()).collect();
            {
                let mut p = shared.state.lock().unwrap();
                for n in &nodes {
                    p.chunks.push(Some(vec![n.clone()]));
                    p.results.push(None);
                }
            }
            let slot = |i: usize| nodes[i].0.outputs[0].slot.get().map(Result::is_ok);
            // Completion handlers may run out of order: nothing is published before chunk 0.
            shared.complete(1, Ok(()));
            assert_eq!((slot(0), slot(1)), (None, None));
            shared.complete(0, Err(Error::Backend("injected".into())));
            assert_eq!(
                (slot(0), slot(1), slot(2)),
                (Some(false), Some(false), None)
            );
            // A later chunk that succeeded still derives from failed work.
            shared.complete(2, Ok(()));
            assert_eq!(slot(2), Some(false));
            let Some(Err(err)) = nodes[2].0.outputs[0].slot.get() else {
                panic!("chunk 2 must fail");
            };
            assert!(err.to_string().contains("injected"), "{err}");
        }
    }
}
