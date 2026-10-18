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

    mod reduce;

    use crate::fused::Load;

    /// Dispatches per command buffer before it is committed while the GPU is busy.
    const MAX_DISPATCHES: usize = 256;
    /// Dispatches per command buffer before it is committed while the GPU is idle.
    const IDLE_DISPATCHES: usize = 8;
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

    /// `GEMMParams` (MLX steel), with the trailing padding Metal's layout has.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: i64,
        batch_stride_b: i64,
        batch_stride_d: i64,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
        _pad: i32,
    }

    /// `GEMMSpiltKParams` (MLX steel).
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct SplitKParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
        tiles_n: i32,
        tiles_m: i32,
        partitions: i32,
        partition_stride: i32,
        partition_size: i32,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
    }

    /// `AttnParams` (MLX steel attention).
    #[repr(C)]
    #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
    struct AttnParams {
        b: i32,
        h: i32,
        d: i32,
        ql: i32,
        kl: i32,
        gqa_factor: i32,
        scale: f32,
        nq: i32,
        nk: i32,
        nq_aligned: i32,
        nk_aligned: i32,
        ql_rem: i32,
        kl_rem: i32,
        ql_off: i32,
        q_strides: [i64; 3],
        k_strides: [i64; 3],
        v_strides: [i64; 3],
        o_strides: [i64; 3],
    }

    /// `GEMMAddMMParams` (MLX steel).
    #[repr(C)]
    #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
    struct AddMmParams {
        ldc: i32,
        fdc: i32,
        batch_stride_c: i64,
        alpha: f32,
        beta: f32,
    }

    enum GemmBuffer<'a> {
        View(&'a nuvec_gpu::Buffer),
        Copy(Buffer),
    }

    /// A GEMM operand: buffer, byte offset, layout of its last two dimensions, batch strides.
    struct Gemmand<'a> {
        buffer: GemmBuffer<'a>,
        offset: usize,
        transposed: bool,
        ld: i32,
        batch_strides: Vec<isize>,
    }

    impl<'a> Gemmand<'a> {
        /// The operand as it is, if its last two dimensions (`rows × cols`) are row-major
        /// (`transposed = false`) or column-major, with non-negative strides (MLX's
        /// `check_transpose`; size-1 dimensions fit either).
        fn view((meta, data): Operand<'a>, rows: usize, cols: usize) -> Option<Self> {
            let nd = meta.shape.len();
            if meta.strides.iter().any(|&s| s < 0) || data.offset < 0 {
                return None;
            }
            let (sr, sc) = (meta.strides[nd - 2], meta.strides[nd - 1]);
            let (transposed, ld) = if sc == 1 || cols == 1 {
                (false, if rows == 1 { cols as isize } else { sr })
            } else if sr == 1 || rows == 1 {
                (true, if cols == 1 { rows as isize } else { sc })
            } else {
                return None;
            };
            Some(Self {
                buffer: GemmBuffer::View(data.buffer.metal()?),
                offset: data.offset as usize * meta.dtype.size(),
                transposed,
                ld: i32::try_from(ld).ok()?,
                batch_strides: meta.strides[..nd - 2].to_vec(),
            })
        }

        fn arg(&self) -> Arg<'_> {
            match &self.buffer {
                GemmBuffer::View(b) => Arg::Read(b, self.offset),
                GemmBuffer::Copy(b) => Arg::Read(b.metal().expect("device storage"), 0),
            }
        }
    }

    /// MLX's tile choice (`GEMM_TPARAM_MACRO`) by GPU class letter: `(bm, bn, bk, wm, wn)`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_tiles(
        class: char,
        dtype: DType,
        transpose_a: bool,
        transpose_b: bool,
        out_elements: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> (usize, usize, usize, usize, usize) {
        let half = dtype != DType::F32;
        let nt = !transpose_a && transpose_b;
        match class {
            // Small devices.
            'g' | 'p' if nt => (64, 32, 32, 2, 2),
            'g' | 'p' if half => (64, 64, 16, 1, 2),
            'g' | 'p' => (64, 64, 16, 2, 2),
            // Large devices.
            'd' if out_elements >= 1 << 20 => match (half, 2 * m.max(n) > k, nt) {
                (true, true, _) => (64, 64, 16, 1, 2),
                (true, false, true) => (64, 32, 32, 2, 2),
                (true, false, false) => (32, 64, 16, 1, 2),
                (false, ..) => (64, 64, 16, 2, 2),
            },
            'd' => match (half, nt) {
                (true, true) => (64, 32, 32, 2, 2),
                (true, false) => (64, 64, 16, 1, 2),
                (false, true) => (32, 64, 16, 1, 2),
                (false, false) => (64, 32, 32, 2, 2),
            },
            // Medium devices.
            _ => (64, 64, 16, 2, 2),
        }
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
        // A broadcast scalar has the output's size but only zero strides: one element.
        if meta.size == 1 || meta.strides.iter().all(|&s| s == 0) {
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

    /// An operand of the quantized kernels: its buffer and byte offset, in a layout the
    /// kernels read (MLX's `ensure_row_contiguous[_matrix]`).
    struct Packed<'a> {
        buffer: GemmBuffer<'a>,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<isize>,
    }

    impl Packed<'_> {
        fn arg(&self) -> Arg<'_> {
            match &self.buffer {
                GemmBuffer::View(b) => Arg::Read(b, self.offset),
                GemmBuffer::Copy(b) => Arg::Read(b.metal().expect("device storage"), 0),
            }
        }

        fn row_contiguous(&self) -> bool {
            crate::shape::flags_for(&self.shape, &self.strides).row_contiguous
        }
    }

    /// MLX's `get_block_dims`: power-of-two threadgroup sides with at most 1024 threads.
    fn block_dims(dims: [usize; 3]) -> [usize; 3] {
        let mut pows = [0usize; 3];
        let mut sum = 0;
        loop {
            let before = sum;
            for i in 0..3 {
                if dims[i] >= 1 << (pows[i] + 1) {
                    pows[i] += 1;
                    sum += 1;
                }
                if sum == 10 {
                    return pows.map(|p| 1 << p);
                }
            }
            if sum == before {
                return pows.map(|p| 1 << p);
            }
        }
    }

    /// Threads per row for MLX's single-row kernels: whole simdgroups covering the row at
    /// `reads` values per thread.
    fn row_threads(axis: usize, reads: usize) -> usize {
        axis.div_ceil(reads).div_ceil(32) * 32
    }

    /// Values per packing unit of `bits`-bit codes (MLX's `get_pack_factor`, in `word` bits).
    fn pack_factor(bits: usize, word: usize) -> usize {
        match bits {
            3 | 5 => 8,
            6 => 4,
            _ => word / bits,
        }
    }

    /// Rows up to which a transposed quantized product runs as matrix-vector products
    /// (MLX's `get_qmv_batch_limit`), by GPU generation and class letter.
    fn qmv_batch_limit(k: usize, n: usize, generation: u32, class: char) -> usize {
        let (small, medium) = (k <= 2048 && n <= 2048, k <= 4096 && n <= 4096);
        let pick = |a, b, c| {
            if small {
                a
            } else if medium {
                b
            } else {
                c
            }
        };
        if generation >= 17 && class != 'd' {
            pick(33, 25, 13)
        } else if generation >= 15 && class != 'd' {
            pick(13, 15, 13)
        } else if class == 'd' {
            pick(32, 18, 12)
        } else if generation >= 13 {
            pick(
                14,
                10,
                if generation == 13 && class == 's' {
                    7
                } else {
                    6
                },
            )
        } else {
            pick(18, 12, 10)
        }
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
                (Primitive::Matmul | Primitive::AddMM { .. }, Some(d)) => {
                    is_float(d) && operands[0].0.shape.last().is_some_and(|&k| k > 0)
                }
                (Primitive::Quantize { .. }, Some(d)) => is_float(d) && operands[0].0.size > 0,
                (Primitive::Fast(_), Some(d)) => is_float(d) && operands[0].0.size > 0,
                (Primitive::Compiled(_), _) => true,
                (Primitive::GatherAxis { .. } | Primitive::Concatenate { .. }, _) => true,
                (Primitive::ScatterAxis { reduce, .. }, Some(d)) => {
                    *reduce == crate::ScatterReduce::None
                        || matches!(
                            d,
                            DType::F32 | DType::F16 | DType::BF16 | DType::I32 | DType::U32
                        )
                }
                (Primitive::Dequantize { .. }, _) => outputs[0].size > 0,
                (Primitive::QuantizedMatmul { .. }, Some(d)) => is_float(d) && outputs[0].size > 0,
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
            if let Primitive::Fast(op) = primitive {
                return self.fast(op, operands, outputs);
            }
            if let Primitive::Compiled(fused) = primitive {
                return self.fused(fused, operands, outputs);
            }
            if let Primitive::Quantize { group_size, bits } = primitive {
                return self.quantize(operands[0], outputs, *group_size, usize::from(*bits));
            }
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
                Primitive::GatherAxis { axis } => {
                    self.gather(operands[0], operands[1], *axis, &mut buffer)?
                }
                Primitive::ScatterAxis { axis, reduce } => {
                    self.copy(operands[0], &mut buffer)?;
                    self.scatter(operands, *axis, *reduce, &mut buffer)?
                }
                Primitive::Concatenate { axis } => {
                    self.concatenate(operands, out, *axis, &mut buffer)?
                }
                Primitive::Reduce { op, axes } => {
                    self.reduce(*op, axes, operands[0], &mut buffer)?
                }
                Primitive::ArgReduce { op, axis } => {
                    self.arg_reduce(*op, *axis, operands[0], out, &mut buffer)?
                }
                Primitive::Matmul => self.matmul(operands, out, &mut buffer)?,
                Primitive::AddMM { .. } => {
                    let (alpha, beta) = primitive.addmm_factors();
                    let c = Some((operands[2], alpha, beta));
                    self.gemm(operands[0], operands[1], c, out, &mut buffer)?
                }
                Primitive::Dequantize { group_size, bits } => {
                    self.dequantize(operands, out, *group_size, usize::from(*bits), &mut buffer)?
                }
                Primitive::QuantizedMatmul {
                    group_size,
                    bits,
                    transpose,
                } => self.quantized_matmul(
                    operands,
                    out,
                    (*group_size, usize::from(*bits), *transpose),
                    &mut buffer,
                )?,
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

        /// A fused element-wise graph as one generated kernel: each input is read as a
        /// scalar, flat or through its (collapsed) strides, and the kernel is compiled once
        /// per graph and combination of reads. Without strided inputs, a thread computes
        /// `8 / itemsize` elements of arrays of at least 65,536 (MLX's rule).
        fn fused(
            &mut self,
            fused: &crate::Fused,
            operands: &[Operand<'_>],
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            let out = &outputs[0];
            let mut buffers = outputs
                .iter()
                .map(|o| Buffer::try_alloc(o.size * o.dtype.size()))
                .collect::<Result<Vec<_>>>()?;
            let mut strided: Vec<&[isize]> = Vec::new();
            let loads: Vec<Load> = operands
                .iter()
                .map(|(m, _)| match mode(m, &out.shape) {
                    Mode::Scalar => Load::Scalar,
                    Mode::Flat => Load::Flat,
                    Mode::Strided => {
                        strided.push(&m.strides);
                        Load::Strided(strided.len() - 1)
                    }
                })
                .collect();
            let per_thread = if strided.is_empty() && out.size >= 1 << 16 {
                (8 / out.dtype.size()).max(1)
            } else {
                1
            };
            let (shape, strides) = collapse_all(&out.shape, &strided);
            let offsets: Vec<i64> = operands.iter().map(|(_, d)| d.offset as i64).collect();
            let strides: Vec<isize> = strides.concat();
            let consts = [
                bytemuck::cast_slice(&offsets).to_vec(),
                shape_bytes(&shape),
                strides_bytes(&strides),
                bytes(&(shape.len() as i32)),
                bytes(&(out.size as u32)),
            ];
            let name = fused.kernel_name(&loads, per_thread);
            let device = self
                .device()
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            let pipeline = device
                .pipeline_from_source(&name, || fused.metal_source(&name, &loads, per_thread))
                .map_err(backend)?;
            let mut args: Vec<Arg<'_>> = operands
                .iter()
                .map(|(_, d)| Arg::read(d.buffer.metal().expect("checked")))
                .collect();
            args.extend(
                buffers
                    .iter_mut()
                    .map(|b| Arg::write(b.metal_mut().expect("device storage"))),
            );
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.command_buffer()?
                .dispatch(&pipeline, &args, out.size.div_ceil(per_thread))
                .map_err(backend)?;
            self.counted()?;
            Ok(buffers.into_iter().map(Data::new).collect())
        }

        /// `take_along_axis`: one thread per output element.
        fn gather(
            &mut self,
            (src, src_data): Operand<'_>,
            (ind, ind_data): Operand<'_>,
            axis: usize,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let name = format!(
                "nuvec_gather_{}_{}",
                bits(src.dtype.size()),
                ind.dtype.name()
            );
            let consts = [
                bytes(&(src_data.offset as i64)),
                bytes(&(ind_data.offset as i64)),
                shape_bytes(&ind.shape),
                strides_bytes(&src.strides),
                strides_bytes(&ind.strides),
                bytes(&(ind.shape.len() as i32)),
                bytes(&(axis as i32)),
                bytes(&(src.shape[axis] as i64)),
            ];
            let mut args = vec![
                Arg::read(src_data.buffer.metal().expect("checked")),
                Arg::read(ind_data.buffer.metal().expect("checked")),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.dispatch(&name, &args, ind.size)
        }

        /// `put_along_axis` or scatter-add into `buffer`, which already holds the
        /// destination (row-contiguous): one thread per update.
        fn scatter(
            &mut self,
            operands: &[Operand<'_>],
            axis: usize,
            reduce: crate::ScatterReduce,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let ((dst, _), (ind, ind_data), (upd, upd_data)) =
                (operands[0], operands[1], operands[2]);
            let name = match reduce {
                crate::ScatterReduce::None => format!(
                    "nuvec_scatter_{}_{}",
                    bits(dst.dtype.size()),
                    ind.dtype.name()
                ),
                _ => format!(
                    "nuvec_scatter_add_{}_{}",
                    dst.dtype.name(),
                    ind.dtype.name()
                ),
            };
            let consts = [
                bytes(&(upd_data.offset as i64)),
                bytes(&(ind_data.offset as i64)),
                shape_bytes(&upd.shape),
                strides_bytes(&upd.strides),
                strides_bytes(&ind.strides),
                strides_bytes(&crate::shape::row_contiguous_strides(&dst.shape)),
                bytes(&(upd.shape.len() as i32)),
                bytes(&(axis as i32)),
                bytes(&(dst.shape[axis] as i64)),
            ];
            let mut args = vec![
                Arg::read(upd_data.buffer.metal().expect("checked")),
                Arg::read(ind_data.buffer.metal().expect("checked")),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            self.dispatch(&name, &args, upd.size)
        }

        /// `concatenate`: each input copied into its region of the output.
        fn concatenate(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            axis: usize,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let out_strides = crate::shape::row_contiguous_strides(&out.shape);
            let name = format!("nuvec_copy_into_{}", bits(out.dtype.size()));
            let mut start = 0isize;
            for &(meta, data) in operands {
                let offset = start * out_strides[axis];
                start += meta.shape[axis] as isize;
                if meta.size == 0 {
                    continue;
                }
                let consts = [
                    bytes(&(data.offset as i64)),
                    bytes(&(offset as i64)),
                    shape_bytes(&meta.shape),
                    strides_bytes(&meta.strides),
                    strides_bytes(&out_strides),
                    bytes(&(meta.shape.len() as i32)),
                ];
                let mut args = vec![
                    Arg::read(data.buffer.metal().expect("checked")),
                    Arg::write(buffer.metal_mut().expect("device storage")),
                ];
                args.extend(consts.iter().map(|c| Arg::bytes(c)));
                self.dispatch(&name, &args, meta.size)?;
            }
            Ok(())
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
            if self.reduce_mlx(op, axes, (meta, data), buffer)? {
                return Ok(());
            }
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
            self.gemm(operands[0], operands[1], None, out, buffer)
        }

        /// MLX's steel GEMM: `D = A·B`, or `D = alpha·A·B + beta·C` for `addmm` (ported
        /// from `steel_matmul_regular_axpby`: tile table, transpose detection, a copy for
        /// other layouts, batching).
        pub(crate) fn gemm(
            &mut self,
            a: Operand<'_>,
            b: Operand<'_>,
            c: Option<(Operand<'_>, f32, f32)>,
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            if out.size == 0 {
                return Ok(());
            }
            let nd = out.shape.len();
            let (m, n, k) = (out.shape[nd - 2], out.shape[nd - 1], a.0.shape[nd - 1]);
            let batch_shape = &out.shape[..nd - 2];
            let batch: usize = batch_shape.iter().product();
            // A vector operand: matrix-vector kernels (MLX's routing).
            if m.min(n) == 1 {
                return self.gemv(a, b, c, (m, n, k), out, buffer);
            }
            let ga = self.gemmand(a, m, k)?;
            let gb = self.gemmand(b, k, n)?;
            let class = self
                .device()
                .and_then(|d| d.info().architecture.chars().last())
                .unwrap_or('s');
            // Few output tiles over a long K: split K (MLX's case 1, without NAX).
            let threshold = if matches!(class, 's' | 'd') {
                2048
            } else {
                1024
            };
            if batch == 1
                && m.div_ceil(16) * n.div_ceil(16) <= threshold
                && k / 16 >= 8
                && k >= m.max(n)
            {
                return self.gemm_splitk(&ga, &gb, c, (m, n, k), out, buffer);
            }
            let (bm, bn, bk, wm, wn) = gemm_tiles(
                class,
                out.dtype,
                ga.transposed,
                gb.transposed,
                batch * m * n,
                m,
                n,
                k,
            );
            let has_batch = batch_shape.len() > 1;
            let source = c.filter(|&(_, alpha, beta)| alpha != 0.0 || beta != 1.0);
            let axpby = source.is_some_and(|(_, alpha, beta)| alpha != 1.0 || beta != 1.0);
            let name = format!(
                "nuvec_gemm_{}{}_{}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}",
                if ga.transposed { 't' } else { 'n' },
                if gb.transposed { 't' } else { 'n' },
                out.dtype.name()
            );
            let constants = [
                (10, has_batch),
                (100, source.is_some()),
                (110, axpby),
                (200, m % bm == 0),
                (201, n % bn == 0),
                (202, k % bk == 0),
            ];
            let first = |strides: &[isize]| strides.first().map_or(0, |&s| s as i64);
            let params = GemmParams {
                m: m as i32,
                n: n as i32,
                k: k as i32,
                lda: ga.ld,
                ldb: gb.ld,
                ldd: n as i32,
                tiles_n: n.div_ceil(bn) as i32,
                tiles_m: m.div_ceil(bm) as i32,
                batch_stride_a: first(&ga.batch_strides),
                batch_stride_b: first(&gb.batch_strides),
                batch_stride_d: (m * n) as i64,
                swizzle_log: 0,
                gemm_k_iterations_aligned: (k / bk) as i32,
                batch_ndim: batch_shape.len() as i32,
                _pad: 0,
            };
            let mut batch_strides: Vec<i64> = ga
                .batch_strides
                .iter()
                .chain(&gb.batch_strides)
                .map(|&s| s as i64)
                .collect();
            let mut addmm = AddMmParams::default();
            if let Some(((cm, _), alpha, beta)) = source {
                addmm = AddMmParams {
                    ldc: cm.strides[nd - 2] as i32,
                    fdc: cm.strides[nd - 1] as i32,
                    batch_stride_c: first(&cm.strides[..nd - 2]),
                    alpha,
                    beta,
                };
                batch_strides.extend(cm.strides[..nd - 2].iter().map(|&s| s as i64));
            }
            batch_strides.push(0); // never empty (Metal rejects empty inline data)
            let pipeline = self.pipeline_with(&name, &constants)?;
            let (params, addmm) = (bytemuck::bytes_of(&params), bytemuck::bytes_of(&addmm));
            let (shape, strides) = (
                shape_bytes(batch_shape),
                bytemuck::cast_slice(&batch_strides),
            );
            let unused = [0u8; 4];
            let c_arg = match source {
                Some(((cm, cd), _, _)) => Arg::Read(
                    cd.buffer.metal().expect("checked"),
                    cd.offset as usize * cm.dtype.size(),
                ),
                None => Arg::bytes(&unused),
            };
            let args = [
                ga.arg(),
                gb.arg(),
                c_arg,
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(params),
                Arg::bytes(addmm),
                Arg::bytes(&shape),
                Arg::bytes(strides),
            ];
            self.command_buffer()?
                .dispatch_groups(
                    &pipeline,
                    &args,
                    [n.div_ceil(bn), m.div_ceil(bm), batch],
                    [32, wn, wm],
                )
                .map_err(backend)?;
            self.counted()
        }

        /// MLX's split-K GEMM (`steel_gemm_splitk_axpby`): partial products of K partitions
        /// in float32, summed (and, for `addmm`, scaled and added to `c`) by an accumulation
        /// kernel.
        fn gemm_splitk(
            &mut self,
            ga: &Gemmand<'_>,
            gb: &Gemmand<'_>,
            c: Option<(Operand<'_>, f32, f32)>,
            (m, n, k): (usize, usize, usize),
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let (bm, bn, bk, wm, wn) = (
                if m < 40 { 16 } else { 32 },
                if n < 40 { 16 } else { 32 },
                16,
                2,
                2,
            );
            let tiles = m.div_ceil(32) * n.div_ceil(32);
            let partitions = ((k / 16) / tiles).next_power_of_two().clamp(2, 32);
            let iterations = (k / bk) / partitions;
            let mut partial = Buffer::try_alloc(partitions * m * n * 4)?;
            let name = format!(
                "nuvec_gemm_splitk_{}{}_{}_bm{bm}_bn{bn}_MN_{}_K_{}",
                if ga.transposed { 't' } else { 'n' },
                if gb.transposed { 't' } else { 'n' },
                out.dtype.name(),
                if m % bm == 0 && n % bn == 0 { 't' } else { 'n' },
                if k % bk == 0 { 't' } else { 'n' },
            );
            let params = SplitKParams {
                m: m as i32,
                n: n as i32,
                k: k as i32,
                lda: ga.ld,
                ldb: gb.ld,
                ldc: n as i32,
                tiles_n: n.div_ceil(bn) as i32,
                tiles_m: m.div_ceil(bm) as i32,
                partitions: partitions as i32,
                partition_stride: (m * n) as i32,
                partition_size: (iterations * bk) as i32,
                swizzle_log: 0,
                gemm_k_iterations_aligned: iterations as i32,
            };
            let pipeline = self.pipeline_with(&name, &[])?;
            let params = bytemuck::bytes_of(&params);
            let args = [
                ga.arg(),
                gb.arg(),
                Arg::write(partial.metal_mut().expect("device storage")),
                Arg::bytes(params),
            ];
            self.command_buffer()?
                .dispatch_groups(
                    &pipeline,
                    &args,
                    [n.div_ceil(bn), m.div_ceil(bm), partitions],
                    [32, wn, wm],
                )
                .map_err(backend)?;
            self.counted()?;
            let axpby = c.filter(|&(_, alpha, beta)| alpha != 1.0 || beta != 0.0);
            let name = match axpby {
                Some(_) => format!("nuvec_gemm_splitk_accum_axpby_{}", out.dtype.name()),
                None => format!("nuvec_gemm_splitk_accum_{}", out.dtype.name()),
            };
            let pipeline = self.pipeline_with(&name, &[])?;
            let ints = |v: usize| (v as i32).to_le_bytes();
            let (parts, stride, ldd) = (ints(partitions), ints(m * n), ints(n));
            let mut args = vec![
                Arg::read(partial.metal().expect("device storage")),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&parts),
                Arg::bytes(&stride),
                Arg::bytes(&ldd),
            ];
            let epilogue = axpby.map(|((cm, cd), alpha, beta)| {
                let nd = cm.shape.len();
                let scalars = [
                    (cm.strides[nd - 2] as i32).to_le_bytes(),
                    (cm.strides[nd - 1] as i32).to_le_bytes(),
                    alpha.to_le_bytes(),
                    beta.to_le_bytes(),
                ];
                (cm, cd, scalars)
            });
            if let Some((cm, cd, scalars)) = &epilogue {
                args.push(Arg::Read(
                    cd.buffer.metal().expect("checked"),
                    cd.offset as usize * cm.dtype.size(),
                ));
                args.extend(scalars.iter().map(|b| Arg::bytes(b)));
            }
            self.command_buffer()?
                .dispatch_threads(&pipeline, &args, [n, m, 1], [n.min(32), m.min(32), 1])
                .map_err(backend)?;
            self.counted()
        }

        /// MLX's matrix-vector product (`gemv_axbpy`) for `M = 1` or `N = 1`: `gemv` when the
        /// matrix is traversed by rows, `gemv_t` when by columns.
        fn gemv(
            &mut self,
            a: Operand<'_>,
            b: Operand<'_>,
            c: Option<(Operand<'_>, f32, f32)>,
            (m, n, k): (usize, usize, usize),
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let nd = out.shape.len();
            let is_b_matrix = n != 1;
            let (mat, vec) = if is_b_matrix { (b, a) } else { (a, b) };
            let mat = if is_b_matrix {
                self.gemmand(mat, k, n)?
            } else {
                self.gemmand(mat, m, k)?
            };
            // The vector is read contiguously along K.
            let vec_k_stride = if is_b_matrix {
                vec.0.strides[nd - 1]
            } else {
                vec.0.strides[nd - 2]
            };
            let vec = if k == 1 || vec_k_stride == 1 {
                match Gemmand::view(
                    vec,
                    if is_b_matrix { 1 } else { k },
                    if is_b_matrix { k } else { 1 },
                ) {
                    Some(g) => g,
                    None => self.gemmand_copy(vec)?,
                }
            } else {
                self.gemmand_copy(vec)?
            };
            let transpose_mat = if is_b_matrix {
                !mat.transposed
            } else {
                mat.transposed
            };
            let (in_len, out_len) = (k, if is_b_matrix { n } else { m });
            let (mut bm, mut bn, mut sm, mut sn, mut tm, mut tn) = (1, 1, 1, 32, 4, 4);
            let (kind, per_group) = if transpose_mat {
                (sm, sn) = if in_len >= 8192 && out_len >= 2048 {
                    (4, 8)
                } else {
                    (8, 4)
                };
                bn = if out_len >= 2048 {
                    16
                } else if out_len >= 512 {
                    4
                } else {
                    2
                };
                tn = if out_len < tn { 1 } else { tn };
                ("gemv_t", bn * sn * tn)
            } else {
                bm = if out_len >= 4096 { 8 } else { 4 };
                if k <= 64 {
                    (bm, sm, sn) = (1, 8, 4);
                } else if k >= 16 * out_len {
                    (bm, bn) = (1, 8);
                }
                tm = if out_len < tm { 1 } else { tm };
                ("gemv", bm * sm * tm)
            };
            let axpby = c.filter(|&(_, alpha, beta)| alpha != 1.0 || beta != 0.0);
            let batch_shape: Vec<usize> = match &out.shape[..nd - 2] {
                [] => vec![1],
                s => s.to_vec(),
            };
            let batch: usize = batch_shape.iter().product();
            let nc = batch_shape.len() > 1;
            let name = format!(
                "nuvec_{kind}_{}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc{}_axpby{}",
                out.dtype.name(),
                u8::from(nc),
                u8::from(axpby.is_some())
            );
            let strides = |v: &[isize]| -> Vec<u8> {
                let v: Vec<i64> = if v.is_empty() {
                    vec![0]
                } else {
                    v.iter().map(|&s| s as i64).collect()
                };
                bytemuck::cast_slice(&v).to_vec()
            };
            let (vec_strides, mat_strides) =
                (strides(&vec.batch_strides), strides(&mat.batch_strides));
            let (bias_strides, bias_stride, alpha, beta) = match axpby {
                Some(((cm, _), alpha, beta)) => (
                    strides(&cm.strides[..nd - 2]),
                    // Along the output vector: N when the matrix is B, else M.
                    cm.strides[if is_b_matrix { nd - 1 } else { nd - 2 }] as i32,
                    alpha,
                    beta,
                ),
                None => (strides(&[]), 0, 1.0, 0.0),
            };
            let ints = |v: usize| (v as i32).to_le_bytes();
            let scalars = [
                ints(in_len),
                ints(out_len),
                mat.ld.to_le_bytes(),
                alpha.to_le_bytes(),
                beta.to_le_bytes(),
                ints(nd.saturating_sub(2).max(1)),
                bias_stride.to_le_bytes(),
            ];
            let shape = shape_bytes(&batch_shape);
            let unused = [0u8; 4];
            let pipeline = self.pipeline_with(&name, &[])?;
            let bias_arg = match axpby {
                Some(((cm, cd), _, _)) => Arg::Read(
                    cd.buffer.metal().expect("checked"),
                    cd.offset as usize * cm.dtype.size(),
                ),
                None => Arg::bytes(&unused),
            };
            let args = [
                mat.arg(),
                vec.arg(),
                bias_arg,
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&scalars[0]),
                Arg::bytes(&scalars[1]),
                Arg::bytes(&scalars[2]),
                Arg::bytes(&scalars[3]),
                Arg::bytes(&scalars[4]),
                Arg::bytes(&scalars[5]),
                Arg::bytes(&shape),
                Arg::bytes(&vec_strides),
                Arg::bytes(&mat_strides),
                Arg::bytes(&bias_strides),
                Arg::bytes(&scalars[6]),
            ];
            self.command_buffer()?
                .dispatch_groups(
                    &pipeline,
                    &args,
                    [out_len.div_ceil(per_group), 1, batch],
                    [32, bn, bm],
                )
                .map_err(backend)?;
            self.counted()
        }

        /// The operand as it is if its last two dimensions (all of it with `whole`) are
        /// row-contiguous, else a row-contiguous copy (encoded now).
        fn packed<'a>(&mut self, operand: Operand<'a>, whole: bool) -> Result<Packed<'a>> {
            let (meta, data) = operand;
            let nd = meta.shape.len();
            let (sh, st) = (&meta.shape, &meta.strides);
            let fits = if whole || nd < 2 {
                meta.flags.row_contiguous
            } else {
                (sh[nd - 1] == 1 || st[nd - 1] == 1)
                    && (sh[nd - 2] == 1 || st[nd - 2] == sh[nd - 1] as isize)
                    && st.iter().all(|&s| s >= 0)
            };
            if fits
                && data.offset >= 0
                && let Some(b) = data.buffer.metal()
            {
                return Ok(Packed {
                    buffer: GemmBuffer::View(b),
                    offset: data.offset as usize * meta.dtype.size(),
                    shape: sh.to_vec(),
                    strides: st.to_vec(),
                });
            }
            let mut copy = Buffer::try_alloc(meta.size * meta.dtype.size())?;
            self.copy(operand, &mut copy)?;
            Ok(Packed {
                buffer: GemmBuffer::Copy(copy),
                offset: 0,
                shape: sh.to_vec(),
                strides: crate::shape::row_contiguous_strides(sh).to_vec(),
            })
        }

        /// A 1-D launch of `threads` threads in groups of at most the pipeline's maximum.
        fn dispatch_linear(&mut self, name: &str, args: &[Arg<'_>], threads: usize) -> Result<()> {
            let pipeline = self.pipeline(name)?;
            let group = pipeline.max_threads_per_threadgroup().min(threads);
            self.command_buffer()?
                .dispatch_threads(&pipeline, args, [threads, 1, 1], [group, 1, 1])
                .map_err(backend)?;
            self.counted()
        }

        /// One threadgroup of `group` threads per row, `rows` rows.
        fn dispatch_rows(
            &mut self,
            pipeline: &nuvec_gpu::Pipeline,
            args: &[Arg<'_>],
            rows: usize,
            group: usize,
        ) -> Result<()> {
            self.command_buffer()?
                .dispatch_threads(pipeline, args, [rows * group, 1, 1], [group, 1, 1])
                .map_err(backend)?;
            self.counted()
        }

        /// `temp` (`[rows, axis]`, row-contiguous) summed over its rows into `out`.
        fn sum_rows(
            &mut self,
            temp: Buffer,
            dtype: DType,
            (rows, axis): (usize, usize),
            out: &mut Buffer,
        ) -> Result<()> {
            let meta = crate::kernels::strided(dtype, &[rows, axis], &[axis as isize, 1]);
            let data = Data::new(temp);
            self.reduce(ReduceOp::Sum, &[0], (&meta, &data), out)
        }

        /// The fused operations (MLX's `fast` kernels and its softmax and log-sum-exp).
        fn fast(
            &mut self,
            op: &crate::fast::FastOp,
            operands: &[Operand<'_>],
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            use crate::fast::FastOp;
            let alloc = |o: &Output| Buffer::try_alloc(o.size * o.dtype.size());
            if let FastOp::Attention {
                scale,
                causal,
                mask,
                sinks,
            } = op
            {
                let mut out = alloc(&outputs[0])?;
                let flags = (f32::from_bits(*scale), *causal, *mask, *sinks);
                self.attention(operands, &outputs[0], flags, &mut out)?;
                return Ok(vec![Data::new(out)]);
            }
            let mut out = alloc(&outputs[0])?;
            let x = self.packed(operands[0], true)?;
            let dt = operands[0].0.dtype;
            let dtn = dt.name();
            let axis = x.shape[x.shape.len() - 1];
            let rows = x.shape.iter().product::<usize>() / axis;
            let u32b = |v: usize| (v as u32).to_le_bytes();
            let i32b = |v: usize| (v as i32).to_le_bytes();
            // A 1-D parameter's stride (0 for the scalar placeholders).
            let stride = |p: &Packed<'_>| u32b(usize::from(!p.shape.is_empty()));
            let mut extra = Vec::new();
            match op {
                FastOp::Softmax | FastOp::LogSumExp => {
                    let kind = if *op == FastOp::Softmax {
                        "softmax"
                    } else {
                        "logsumexp"
                    };
                    let looped = axis > 4096;
                    let name = format!(
                        "nuvec_{kind}_{}_{dtn}",
                        if looped { "looped" } else { "block" }
                    );
                    let pipeline = self.pipeline(&name)?;
                    let group = if looped {
                        pipeline.max_threads_per_threadgroup()
                    } else {
                        row_threads(axis, 4)
                    };
                    let n = i32b(axis);
                    let args = [
                        x.arg(),
                        Arg::write(out.metal_mut().expect("device storage")),
                        Arg::bytes(&n),
                    ];
                    self.dispatch_rows(&pipeline, &args, rows, group)?;
                }
                FastOp::RmsNorm { eps } | FastOp::LayerNorm { eps } => {
                    let layer = matches!(op, FastOp::LayerNorm { .. });
                    let w = self.packed(operands[1], true)?;
                    let b = match layer {
                        true => Some(self.packed(operands[2], true)?),
                        false => None,
                    };
                    let (limit, reads) = if layer { (6656, 8) } else { (4096, 4) };
                    let looped = axis > limit;
                    let name = format!(
                        "nuvec_{}{}_{dtn}",
                        if layer { "layer_norm" } else { "rms" },
                        if looped { "_looped" } else { "" }
                    );
                    let pipeline = self.pipeline(&name)?;
                    let group = if looped {
                        pipeline.max_threads_per_threadgroup()
                    } else {
                        row_threads(axis, reads)
                    };
                    let (e, n, ws) = (eps.to_le_bytes(), u32b(axis), stride(&w));
                    let bs = b.as_ref().map(stride).unwrap_or_default();
                    let mut args = vec![x.arg(), w.arg()];
                    if let Some(b) = &b {
                        args.push(b.arg());
                    }
                    args.push(Arg::write(out.metal_mut().expect("device storage")));
                    args.extend([Arg::bytes(&e), Arg::bytes(&n), Arg::bytes(&ws)]);
                    if layer {
                        args.push(Arg::bytes(&bs));
                    }
                    self.dispatch_rows(&pipeline, &args, rows, group)?;
                }
                FastOp::RmsNormVjp { eps } | FastOp::LayerNormVjp { eps } => {
                    let layer = matches!(op, FastOp::LayerNormVjp { .. });
                    let w = self.packed(operands[1], true)?;
                    let g = self.packed(operands[if layer { 3 } else { 2 }], true)?;
                    let has_w = !w.shape.is_empty();
                    let (limit, reads) = if layer { (8192, 8) } else { (4096, 4) };
                    let looped = axis > limit;
                    // RMS rows are grouped (about 512 threadgroups); layer norm: one row each.
                    let per_group = if !layer && !looped {
                        rows.div_ceil(512)
                    } else {
                        1
                    };
                    let groups = rows.div_ceil(per_group);
                    let name = format!(
                        "nuvec_vjp_{}{}_{dtn}",
                        if layer { "layer_norm" } else { "rms" },
                        if looped { "_looped" } else { "" }
                    );
                    let pipeline = self.pipeline_with(&name, &[(20, has_w)])?;
                    let group = if looped {
                        pipeline.max_threads_per_threadgroup()
                    } else {
                        row_threads(axis, reads)
                    };
                    let mut temp = Buffer::try_alloc(if has_w {
                        groups * axis * dt.size()
                    } else {
                        dt.size()
                    })?;
                    let (e, n, ws) = (eps.to_le_bytes(), u32b(axis), stride(&w));
                    let (nr, rpg) = (u32b(rows), u32b(per_group));
                    let mut args = vec![
                        x.arg(),
                        w.arg(),
                        g.arg(),
                        Arg::write(out.metal_mut().expect("device storage")),
                        Arg::write(temp.metal_mut().expect("device storage")),
                        Arg::bytes(&e),
                        Arg::bytes(&n),
                        Arg::bytes(&ws),
                    ];
                    if !layer && !looped {
                        args.extend([Arg::bytes(&nr), Arg::bytes(&rpg)]);
                    }
                    let launches = if !layer && !looped { groups } else { rows };
                    self.dispatch_rows(&pipeline, &args, launches, group)?;
                    drop(args);
                    let mut dw = alloc(&outputs[1])?;
                    if has_w {
                        self.sum_rows(temp, dt, (groups, axis), &mut dw)?;
                    } else {
                        dw.bytes_mut().fill(0);
                    }
                    extra.push(dw);
                    if layer {
                        let mut db = alloc(&outputs[2])?;
                        if operands[2].0.shape.is_empty() {
                            db.bytes_mut().fill(0);
                        } else {
                            let g = operands[3];
                            let axes: Vec<usize> = (0..g.0.shape.len() - 1).collect();
                            self.reduce(ReduceOp::Sum, &axes, g, &mut db)?;
                        }
                        extra.push(db);
                    }
                }
                FastOp::Rope {
                    dims,
                    traditional,
                    base,
                    scale,
                    forward,
                } => {
                    let nd = x.shape.len();
                    let (b, t, d) = (x.shape[0], x.shape[nd - 2], x.shape[nd - 1]);
                    let heads = x.shape[1..nd - 2].iter().product::<usize>();
                    if *dims < d {
                        // The features past `dims` pass through unchanged.
                        self.copy(operands[0], &mut out)?;
                    }
                    let offset = self.packed(operands[1], true)?;
                    let freqs = match operands.get(2) {
                        Some(f) => Some(self.packed(*f, true)?),
                        None => None,
                    };
                    let size = x.shape.iter().product::<usize>();
                    let large = size > i32::MAX as usize;
                    let single = t == 1 && operands[1].0.size == 1 && !large;
                    let name = format!(
                        "nuvec_rope_{}{}{}{dtn}",
                        if single { "single_" } else { "" },
                        if freqs.is_some() { "freqs_" } else { "" },
                        if large { "large_" } else { "" }
                    );
                    let constants = [(21, *forward), (22, *traditional), (23, false)];
                    let pipeline = self.pipeline_with(&name, &constants)?;
                    let mat = (t * d) as i64;
                    let strides: [i64; 3] = [mat, d as i64, 1];
                    let strides = bytemuck::bytes_of(&strides).to_vec();
                    let offset_stride = i64::from(operands[1].0.size > 1).to_le_bytes();
                    let (sc, nh) = (scale.to_le_bytes(), i32b(heads));
                    let log_base = f32::from_bits(*base).log2().to_le_bytes();
                    let mat_b = mat.to_le_bytes();
                    let freq_stride = 1i64.to_le_bytes();
                    let mut args = vec![
                        x.arg(),
                        Arg::write(out.metal_mut().expect("device storage")),
                        offset.arg(),
                        Arg::bytes(&sc),
                    ];
                    let (grid, group) = if single {
                        args.push(Arg::bytes(&mat_b));
                        let grid = [dims / 2, b * heads, 1];
                        (grid, block_dims(grid))
                    } else {
                        args.extend([
                            Arg::bytes(&strides),
                            Arg::bytes(&strides),
                            Arg::bytes(&offset_stride),
                            Arg::bytes(&nh),
                        ]);
                        let grid = [dims / 2, t, b * heads.div_ceil(4)];
                        (grid, block_dims(grid))
                    };
                    // Buffers 10 and 11: the frequencies (or the base) sit at fixed indices.
                    let unused = [0u8; 4];
                    while args.len() < 10 {
                        args.push(Arg::bytes(&unused));
                    }
                    match &freqs {
                        Some(f) => args.extend([f.arg(), Arg::bytes(&freq_stride)]),
                        None => args.push(Arg::bytes(&log_base)),
                    }
                    self.command_buffer()?
                        .dispatch_threads(&pipeline, &args, grid, group)
                        .map_err(backend)?;
                    self.counted()?;
                }
                FastOp::Attention { .. } => unreachable!("handled above"),
                FastOp::CrossEntropy | FastOp::CrossEntropyVjp => {
                    let vjp = *op == FastOp::CrossEntropyVjp;
                    let name =
                        format!("nuvec_cross_entropy{}_{dtn}", if vjp { "_vjp" } else { "" });
                    let pipeline = self.pipeline(&name)?;
                    let max = pipeline.max_threads_per_threadgroup() / 32 * 32;
                    let group = row_threads(axis, 4).clamp(32, max);
                    let rest = (1..operands.len())
                        .map(|i| self.packed(operands[i], true))
                        .collect::<Result<Vec<_>>>()?;
                    let n = i32b(axis);
                    let mut args = vec![x.arg()];
                    args.extend(rest.iter().map(Packed::arg));
                    args.extend([
                        Arg::write(out.metal_mut().expect("device storage")),
                        Arg::bytes(&n),
                    ]);
                    self.dispatch_rows(&pipeline, &args, rows, group)?;
                }
            }
            let mut data = vec![Data::new(out)];
            data.extend(extra.into_iter().map(Data::new));
            Ok(data)
        }

        /// The operand as it is if `ok` (and its strides are non-negative), else a
        /// row-contiguous copy.
        fn view_if<'a>(&mut self, operand: Operand<'a>, ok: bool) -> Result<Packed<'a>> {
            let (meta, data) = operand;
            if ok
                && data.offset >= 0
                && meta.strides.iter().all(|&s| s >= 0)
                && let Some(b) = data.buffer.metal()
            {
                return Ok(Packed {
                    buffer: GemmBuffer::View(b),
                    offset: data.offset as usize * meta.dtype.size(),
                    shape: meta.shape.to_vec(),
                    strides: meta.strides.to_vec(),
                });
            }
            self.packed(operand, true)
        }

        /// MLX's `ScaledDotProductAttention::eval_gpu` (without NAX): the vector kernels
        /// (one or two passes) for up to 8 queries, the tiled steel kernel otherwise.
        fn attention(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            (scale, causal, has_mask, has_sinks): (f32, bool, bool, bool),
            buffer: &mut Buffer,
        ) -> Result<()> {
            use nuvec_gpu::Constant;
            let (qm, km) = (operands[0].0, operands[1].0);
            let (b, hq, lq, d) = (qm.shape[0], qm.shape[1], qm.shape[2], qm.shape[3]);
            let (hkv, lk) = (km.shape[1], km.shape[2]);
            let dv = operands[2].0.shape[3];
            let gqa = hq / hkv;
            let dtn = qm.dtype.name();
            let class = self
                .device()
                .and_then(|dev| dev.info().class)
                .unwrap_or('g');
            let unit = |m: &Output| m.shape[3] == 1 || m.strides[3] == 1;
            let mask_op = has_mask.then(|| operands[3]);
            let sinks = match has_sinks {
                true => Some(self.packed(operands[operands.len() - 1], true)?),
                false => None,
            };
            let bool_mask = mask_op.is_some_and(|(m, _)| m.dtype == DType::Bool);
            let i32b = |v: usize| (v as i32).to_le_bytes();
            let u64b = |v: isize| (v as u64).to_le_bytes();
            let unused = [0u8; 4];
            if lq <= 8 {
                let q = self.packed(operands[0], true)?;
                let kv_ok = |m: &Output| {
                    unit(m)
                        && (m.shape[0] == 1
                            || m.shape[1] == 1
                            || m.strides[0] == m.strides[1] * m.shape[1] as isize)
                };
                let k = self.view_if(operands[1], kv_ok(km))?;
                let v = self.view_if(operands[2], kv_ok(operands[2].0))?;
                let mask = match mask_op {
                    Some(m) => {
                        let ok = m.0.flags.row_contiguous
                            || b == 1
                            || hq == 1
                            || m.0.strides[0] == m.0.strides[1] * m.0.shape[1] as isize;
                        Some(self.view_if(m, ok)?)
                    }
                    None => None,
                };
                let head = |p: &Packed<'_>| {
                    if p.shape[1] == 1 {
                        p.strides[0]
                    } else {
                        p.strides[1]
                    }
                };
                let strides = [
                    u64b(head(&k)),
                    u64b(k.strides[2]),
                    u64b(head(&v)),
                    u64b(v.strides[2]),
                ];
                let mask_strides = mask.as_ref().map(|m| {
                    let pick = |i: usize| {
                        if m.shape[i] > 1 {
                            m.strides[i] as i32
                        } else {
                            0
                        }
                    };
                    let head = if m.shape[1] > 1 { pick(1) } else { pick(0) };
                    [
                        pick(3).to_le_bytes(),
                        pick(2).to_le_bytes(),
                        head.to_le_bytes(),
                    ]
                });
                let constants = |extra: Option<i32>| {
                    let mut c = vec![
                        (30, Constant::Bool(has_mask)),
                        (31, Constant::Bool(false)),
                        (32, Constant::Bool(causal && lq > 1)),
                        (33, Constant::Bool(bool_mask)),
                        (34, Constant::Bool(has_mask && !bool_mask)),
                        (35, Constant::Bool(has_sinks)),
                    ];
                    if let Some(n) = extra {
                        c.push((36, Constant::Int(n)));
                    }
                    c
                };
                let (sc, n, nh) = (scale.to_le_bytes(), i32b(lk), i32b(hq));
                // Pads unbound argument slots up to `at`.
                fn pad<'a>(args: &mut Vec<Arg<'a>>, at: usize, filler: &'a [u8]) {
                    while args.len() < at {
                        args.push(Arg::bytes(filler));
                    }
                }
                let push_mask = |args: &mut Vec<Arg<'_>>, at: usize| pad(args, at, &[0u8; 4]);
                let two_pass =
                    (matches!(class, 'd' | 's') && lk >= 1024) || (hkv < hq && lk >= 4096);
                if !two_pass {
                    let name = format!("nuvec_sdpa_vector_{dtn}_{d}_{dv}");
                    let device = self
                        .device()
                        .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
                    let pipeline = device
                        .pipeline_with_constants(&name, &constants(None))
                        .map_err(backend)?;
                    let g = i32b(gqa);
                    let mut args = vec![
                        q.arg(),
                        k.arg(),
                        v.arg(),
                        Arg::write(buffer.metal_mut().expect("device storage")),
                        Arg::bytes(&g),
                        Arg::bytes(&n),
                        Arg::bytes(&strides[0]),
                        Arg::bytes(&strides[1]),
                        Arg::bytes(&strides[2]),
                        Arg::bytes(&strides[3]),
                        Arg::bytes(&sc),
                    ];
                    if let (Some(m), Some(ms)) = (&mask, &mask_strides) {
                        push_mask(&mut args, 11 + usize::from(!bool_mask));
                        args.push(m.arg());
                        push_mask(&mut args, 13);
                        args.extend(ms.iter().map(|s| Arg::bytes(s)));
                    }
                    if let Some(s) = &sinks {
                        push_mask(&mut args, 16);
                        args.extend([s.arg(), Arg::bytes(&nh)]);
                    }
                    self.command_buffer()?
                        .dispatch_groups(&pipeline, &args, [b * hq, lq, 1], [1024, 1, 1])
                        .map_err(backend)?;
                    return self.counted();
                }
                let n_simds = gqa * lq;
                let blocks: usize = match class {
                    's' if lk > 1024 && n_simds > 4 => match lk {
                        ..=8192 => 128,
                        8193..=32768 => 256,
                        32769..=65536 => 512,
                        _ => 1024,
                    },
                    's' => 64,
                    'd' if n_simds <= 2 && lk > 8192 => 256,
                    'd' if n_simds >= 6 && lk >= 65536 => 1024,
                    'd' if n_simds >= 6 && lk >= 16384 => 512,
                    'd' => 128,
                    _ if n_simds >= 4 => 64,
                    _ => 32,
                };
                let gqa_kernel = !has_mask
                    && !has_sinks
                    && lq == 1
                    && d == dv
                    && lk >= 8192
                    && ((gqa == 8 && (d == 64 || d == 128))
                        || ((gqa == 12 || gqa == 16) && d == 128));
                let mut name = format!(
                    "nuvec_sdpa_vector_2pass_1_{}{dtn}_{d}_{dv}",
                    if gqa_kernel {
                        format!("gqa_{gqa}_")
                    } else {
                        String::new()
                    }
                );
                let device = self
                    .device()
                    .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
                let consts = constants(Some(blocks as i32));
                let mut pipeline = device
                    .pipeline_with_constants(&name, &consts)
                    .map_err(backend)?;
                let tk1 = (qm.dtype == DType::BF16 && d == 256 && dv == 256)
                    || (qm.dtype == DType::F32 && d == dv && (d == 192 || d == 256));
                if tk1 && 32 * gqa * lq > pipeline.max_threads_per_threadgroup() {
                    name.push_str("_tk1");
                    pipeline = device
                        .pipeline_with_constants(&name, &consts)
                        .map_err(backend)?;
                }
                let rows = b * hq * lq;
                let mut partial = Buffer::try_alloc(rows * blocks * dv * qm.dtype.size())?;
                let mut sums = Buffer::try_alloc(rows * blocks * 4)?;
                let mut maxs = Buffer::try_alloc(rows * blocks * 4)?;
                {
                    let mut args = vec![
                        q.arg(),
                        k.arg(),
                        v.arg(),
                        Arg::write(partial.metal_mut().expect("device storage")),
                        Arg::write(sums.metal_mut().expect("device storage")),
                        Arg::write(maxs.metal_mut().expect("device storage")),
                        Arg::bytes(&unused),
                        Arg::bytes(&n),
                        Arg::bytes(&strides[0]),
                        Arg::bytes(&strides[1]),
                        Arg::bytes(&strides[2]),
                        Arg::bytes(&strides[3]),
                        Arg::bytes(&sc),
                    ];
                    if let (Some(m), Some(ms)) = (&mask, &mask_strides) {
                        push_mask(&mut args, 13 + usize::from(!bool_mask));
                        args.push(m.arg());
                        push_mask(&mut args, 15);
                        args.extend(ms.iter().map(|s| Arg::bytes(s)));
                    }
                    if let Some(s) = &sinks {
                        push_mask(&mut args, 18);
                        args.push(s.arg());
                    }
                    self.command_buffer()?
                        .dispatch_groups(&pipeline, &args, [hkv, b, blocks], [32, gqa, lq])
                        .map_err(backend)?;
                    self.counted()?;
                }
                let name = format!("nuvec_sdpa_vector_2pass_2_{dtn}_{dv}");
                let bl = i32b(blocks);
                let args = [
                    Arg::read(partial.metal().expect("device storage")),
                    Arg::read(sums.metal().expect("device storage")),
                    Arg::read(maxs.metal().expect("device storage")),
                    Arg::write(buffer.metal_mut().expect("device storage")),
                    Arg::bytes(&bl),
                ];
                return self.dispatch_groups(&name, &args, [b * hq, lq, 1], [1024, 1, 1]);
            }
            // Tiled attention: inputs need a unit stride along the head dimension.
            let q = self.view_if(operands[0], unit(qm))?;
            let k = self.view_if(operands[1], unit(km))?;
            let v = self.view_if(operands[2], unit(operands[2].0))?;
            let mask = match mask_op {
                Some(m) => Some(self.view_if(m, unit(m.0))?),
                None => None,
            };
            let (bq, wm) = (32usize, 4usize);
            let bk = if (d == 256 && qm.dtype != DType::F32 && class == 'd') || d < 128 {
                32
            } else {
                16
            };
            let wn = if d == 256 { 2 } else { 1 };
            let mask_name = match mask_op {
                Some((m, _)) if m.dtype == DType::Bool => "bool",
                _ => dtn,
            };
            let name = format!(
                "nuvec_steel_attention_{dtn}_bq{bq}_bk{bk}_bd{d}_bv{dv}_wm{wm}_wn{wn}_mask{mask_name}"
            );
            let constants = [
                (210, lq % bq == 0),
                (211, lk % bk == 0),
                (310, has_mask),
                (311, causal),
                (312, has_sinks),
                (313, false),
            ];
            let pipeline = self.pipeline_with(&name, &constants)?;
            let (nq, nk) = (lq.div_ceil(bq), lk.div_ceil(bk));
            let s3 = |p: &Packed<'_>| {
                [
                    p.strides[0] as i64,
                    p.strides[1] as i64,
                    p.strides[2] as i64,
                ]
            };
            let params = AttnParams {
                b: b as i32,
                h: hq as i32,
                d: d as i32,
                ql: lq as i32,
                kl: lk as i32,
                gqa_factor: gqa as i32,
                scale,
                nq: nq as i32,
                nk: nk as i32,
                nq_aligned: (lq / bq) as i32,
                nk_aligned: (lk / bk) as i32,
                ql_rem: (lq % bq) as i32,
                kl_rem: (lk % bk) as i32,
                ql_off: lk as i32 - lq as i32,
                q_strides: s3(&q),
                k_strides: s3(&k),
                v_strides: s3(&v),
                o_strides: [(hq * lq * dv) as i64, (lq * dv) as i64, dv as i64],
            };
            let params = bytes(&params);
            let mask_params = mask.as_ref().map(|m| bytes(&s3(m)));
            let mut args = vec![
                q.arg(),
                k.arg(),
                v.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&params),
            ];
            if let (Some(m), Some(mp)) = (&mask, &mask_params) {
                args.extend([Arg::bytes(mp), m.arg()]);
            }
            if let Some(s) = &sinks {
                while args.len() < 7 {
                    args.push(Arg::bytes(&unused));
                }
                args.push(s.arg());
            }
            let _ = out;
            self.command_buffer()?
                .dispatch_groups(&pipeline, &args, [nq, hq, b], [32, wm, wn])
                .map_err(backend)?;
            self.counted()
        }

        /// MLX's `affine_quantize`: one simdgroup per group of values.
        fn quantize(
            &mut self,
            w: Operand<'_>,
            outputs: &[Output],
            group_size: usize,
            bits: usize,
        ) -> Result<Vec<Data>> {
            let w = self.packed(w, true)?;
            let n = w.shape.iter().product::<usize>();
            let alloc = |o: &Output| Buffer::try_alloc(o.size * o.dtype.size());
            let mut packed = Buffer::try_alloc(n * bits / 8)?;
            let (mut scales, mut biases) = (alloc(&outputs[1])?, alloc(&outputs[2])?);
            let name = format!(
                "nuvec_affine_quantize_{}_gs{group_size}_b{bits}",
                outputs[1].dtype.name()
            );
            let args = [
                w.arg(),
                Arg::write(packed.metal_mut().expect("device storage")),
                Arg::write(scales.metal_mut().expect("device storage")),
                Arg::write(biases.metal_mut().expect("device storage")),
            ];
            self.dispatch_linear(&name, &args, n / (group_size / 32))?;
            Ok(vec![
                Data::new(packed),
                Data::new(scales),
                Data::new(biases),
            ])
        }

        /// MLX's `affine_dequantize`: one thread per packing unit.
        fn dequantize(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            group_size: usize,
            bits: usize,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let [w, s, b] = [0, 1, 2].map(|i| operands[i]);
            let (w, s, b) = (
                self.packed(w, true)?,
                self.packed(s, true)?,
                self.packed(b, true)?,
            );
            let name = format!(
                "nuvec_affine_dequantize_{}_gs{group_size}_b{bits}",
                out.dtype.name()
            );
            let args = [
                w.arg(),
                s.arg(),
                b.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            self.dispatch_linear(&name, &args, out.size / pack_factor(bits, 8))
        }

        /// MLX's `QuantizedMatmul::eval_gpu`: tiled products (`qmm_t`, split-K, `qmm_n`) from
        /// the GPU's vector limit on, matrix-vector products (`qmv_fast`, `qmv`, `qvm`) below.
        fn quantized_matmul(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            (group_size, bits, transpose): (usize, usize, bool),
            buffer: &mut Buffer,
        ) -> Result<()> {
            let x = self.packed(operands[0], false)?;
            let w = self.packed(operands[1], false)?;
            let s = self.packed(operands[2], false)?;
            let b = self.packed(operands[3], false)?;
            let nd = x.shape.len();
            let k = x.shape[nd - 1];
            let n = out.shape[out.shape.len() - 1];
            let flat = w.shape.len() == 2 && x.row_contiguous();
            let m = if flat { out.size / n } else { x.shape[nd - 2] };
            let batch = out.size / m / n;
            let info = self
                .device()
                .map(|d| d.info())
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            let (generation, class) = (info.generation.unwrap_or(0), info.class.unwrap_or('g'));
            let limit = if transpose {
                qmv_batch_limit(k, n, generation, class)
            } else {
                4
            };
            let dtype = out.dtype.name();
            let base = format!("{dtype}_gs{group_size}_b{bits}");
            let batched = batch > 1;
            if m >= limit && transpose && !batched {
                return self.qmm_splitk(
                    [&w, &s, &b, &x],
                    (m, n, k),
                    &base,
                    out,
                    buffer,
                    group_size,
                );
            }
            let ints = |v: usize| (v as i32).to_le_bytes();
            let output = buffer.metal_mut().expect("device storage");
            // Batch layout (buffers after the sizes), as MLX's `add_strides_and_shapes`.
            let batch_info = [
                ints(nd - 2).to_vec(),
                shape_bytes(&x.shape),
                strides_bytes(&x.strides),
                ints(w.shape.len() - 2).to_vec(),
                shape_bytes(&w.shape),
                strides_bytes(&w.strides),
                strides_bytes(&s.strides),
                strides_bytes(&b.strides),
            ];
            let mut args = vec![w.arg(), s.arg(), b.arg(), x.arg(), Arg::write(output)];
            let (kb, nb, mb) = (ints(k), ints(n), ints(m));
            if m >= limit {
                args.extend([Arg::bytes(&kb), Arg::bytes(&nb), Arg::bytes(&mb)]);
                if batched {
                    args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
                }
                let name = if transpose {
                    format!(
                        "nuvec_affine_qmm_t_{base}_al{}_batch{}",
                        u8::from(n.is_multiple_of(32)),
                        u8::from(batched)
                    )
                } else {
                    format!("nuvec_affine_qmm_n_{base}_batch{}", u8::from(batched))
                };
                return self.dispatch_groups(
                    &name,
                    &args,
                    [n.div_ceil(32), m.div_ceil(32), batch],
                    [32, 2, 2],
                );
            }
            args.extend([Arg::bytes(&kb), Arg::bytes(&nb)]);
            if batched {
                args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
            }
            let (name, per_group) = if transpose {
                let fast = k % (pack_factor(bits, 32) * if bits == 2 { 1 } else { 2 } * 32) == 0;
                let name = if fast {
                    format!(
                        "nuvec_affine_qmv_fast_{base}_batch{}_pr{}",
                        u8::from(batched),
                        u8::from(!n.is_multiple_of(8))
                    )
                } else {
                    format!("nuvec_affine_qmv_{base}_batch{}", u8::from(batched))
                };
                (name, 8)
            } else {
                (
                    format!("nuvec_affine_qvm_{base}_batch{}", u8::from(batched)),
                    group_size.min(32) * 2,
                )
            };
            self.dispatch_groups(&name, &args, [m, n.div_ceil(per_group), batch], [32, 2, 1])
        }

        /// MLX's `qmm_splitk`: K split into partitions of whole groups, partial products in
        /// float32, summed by the GEMM's split-K accumulation. One partition runs `qmm_t`.
        #[allow(clippy::too_many_arguments)]
        fn qmm_splitk(
            &mut self,
            [w, s, b, x]: [&Packed<'_>; 4],
            (m, n, k): (usize, usize, usize),
            base: &str,
            out: &Output,
            buffer: &mut Buffer,
            group_size: usize,
        ) -> Result<()> {
            let (n_tiles, m_tiles) = (n.div_ceil(32), m.div_ceil(32));
            let k_align = group_size.max(32);
            let mut split = (512 / (n_tiles * m_tiles)).max(1).min(k / k_align);
            while split > 1 && k % (split * k_align) != 0 {
                split -= 1;
            }
            let ints = |v: usize| (v as i32).to_le_bytes();
            let (kb, nb, mb) = (ints(k), ints(n), ints(m));
            let aligned = u8::from(n % 32 == 0);
            if split <= 1 {
                let args = [
                    w.arg(),
                    s.arg(),
                    b.arg(),
                    x.arg(),
                    Arg::write(buffer.metal_mut().expect("device storage")),
                    Arg::bytes(&kb),
                    Arg::bytes(&nb),
                    Arg::bytes(&mb),
                ];
                let name = format!("nuvec_affine_qmm_t_{base}_al{aligned}_batch0");
                return self.dispatch_groups(&name, &args, [n_tiles, m_tiles, 1], [32, 2, 2]);
            }
            let mut partial = Buffer::try_alloc(split * m * n * 4)?;
            let (size, stride) = (ints(k / split), ints(m * n));
            let args = [
                w.arg(),
                s.arg(),
                b.arg(),
                x.arg(),
                Arg::write(partial.metal_mut().expect("device storage")),
                Arg::bytes(&kb),
                Arg::bytes(&nb),
                Arg::bytes(&mb),
                Arg::bytes(&size),
                Arg::bytes(&stride),
            ];
            let name = format!("nuvec_affine_qmm_t_splitk_{base}_al{aligned}");
            self.dispatch_groups(&name, &args, [n_tiles, m_tiles, split], [32, 2, 2])?;
            let parts = ints(split);
            let args = [
                Arg::read(partial.metal().expect("device storage")),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&parts),
                Arg::bytes(&stride),
                Arg::bytes(&nb),
            ];
            let name = format!("nuvec_gemm_splitk_accum_{}", out.dtype.name());
            let pipeline = self.pipeline(&name)?;
            self.command_buffer()?
                .dispatch_threads(&pipeline, &args, [n, m, 1], [n.min(32), m.min(32), 1])
                .map_err(backend)?;
            self.counted()
        }

        /// A contiguous copy of an operand (encoded now), row-major.
        fn gemmand_copy<'a>(&mut self, operand: Operand<'_>) -> Result<Gemmand<'a>> {
            let (meta, _) = operand;
            let mut copy = Buffer::try_alloc(meta.size * meta.dtype.size())?;
            self.copy(operand, &mut copy)?;
            let strides = crate::shape::row_contiguous_strides(&meta.shape);
            let nd = strides.len();
            Ok(Gemmand {
                buffer: GemmBuffer::Copy(copy),
                offset: 0,
                transposed: false,
                ld: meta.shape[nd - 1] as i32,
                batch_strides: strides[..nd - 2].to_vec(),
            })
        }

        /// A GEMM operand: its own buffer if its last two dimensions are row- or
        /// column-major, else a contiguous copy (encoded now).
        fn gemmand<'a>(
            &mut self,
            operand: Operand<'a>,
            rows: usize,
            cols: usize,
        ) -> Result<Gemmand<'a>> {
            if let Some(g) = Gemmand::view(operand, rows, cols) {
                return Ok(g);
            }
            let (meta, _) = operand;
            let mut copy = Buffer::try_alloc(meta.size * meta.dtype.size())?;
            self.copy(operand, &mut copy)?;
            let strides = crate::shape::row_contiguous_strides(&meta.shape);
            let nd = strides.len();
            Ok(Gemmand {
                buffer: GemmBuffer::Copy(copy),
                offset: 0,
                transposed: false,
                ld: cols as i32,
                batch_strides: strides[..nd - 2].to_vec(),
            })
        }

        fn pipeline_with(
            &mut self,
            name: &str,
            constants: &[(usize, bool)],
        ) -> Result<nuvec_gpu::Pipeline> {
            let device = self
                .device()
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            device.pipeline_with(name, constants).map_err(backend)
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

        /// Commits once a command buffer holds enough work: a few dispatches while the GPU
        /// is idle, so it starts while the rest is encoded; many while it is busy, so
        /// command buffers stay few.
        fn counted(&mut self) -> Result<()> {
            self.dispatches += 1;
            let busy = self.device().is_some_and(|d| d.in_flight() > 0);
            let limit = if busy {
                MAX_DISPATCHES
            } else {
                IDLE_DISPATCHES
            };
            if self.dispatches >= limit {
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

    /// [`collapse`] for any number of operands: drops size-1 dimensions and merges
    /// neighbours that are contiguous for every operand.
    fn collapse_all(shape: &[usize], strides: &[&[isize]]) -> (Vec<usize>, Vec<Vec<isize>>) {
        let mut out_shape: Vec<usize> = Vec::with_capacity(shape.len());
        let mut out: Vec<Vec<isize>> = vec![Vec::with_capacity(shape.len()); strides.len()];
        for (d, &n) in shape.iter().enumerate() {
            if n == 1 {
                continue;
            }
            let merge = !out_shape.is_empty()
                && strides
                    .iter()
                    .zip(&out)
                    .all(|(s, o)| o.last() == Some(&(s[d] * n as isize)));
            if merge {
                *out_shape.last_mut().expect("not empty") *= n;
                for (s, o) in strides.iter().zip(&mut out) {
                    *o.last_mut().expect("not empty") = s[d];
                }
            } else {
                out_shape.push(n);
                for (s, o) in strides.iter().zip(&mut out) {
                    o.push(s[d]);
                }
            }
        }
        (out_shape, out)
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
