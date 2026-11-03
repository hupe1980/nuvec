//! GPU evaluation: encodes supported primitives as Metal kernels on unified memory.
//!
//! Work is committed in chunks without waiting; an evaluation's chunks are ordered on the GPU
//! by a timeline. Outputs of encoded kernels stay *pending* (exposed to later GPU work as
//! in-flight data, not yet in the nodes' slots) until their chunk has completed; completion
//! publishes them, chunk by chunk in commit order, so data in a slot is always complete,
//! and a failure marks the outputs of its chunk and of every later chunk of the evaluation
//! as failed.
//! Primitives without a GPU kernel fall back to the CPU after [`GpuEval::flush`], which waits
//! for this evaluation's outstanding work (unified memory needs no copies); collectives run
//! on their group's worker instead, without a flush (`GpuEval::collective`).

use std::sync::Arc;

use crate::array::{Data, Node, Output};
use crate::kernels::Operand;
use crate::primitive::Primitive;
use crate::{Device, Result};

#[cfg(test)]
thread_local! {
    /// Command buffers committed by evaluations on this thread (tests count them).
    pub(crate) static COMMITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// GPU dispatches encoded by this process (for [`crate::__gpu_dispatches`]).
pub(crate) static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a Metal GPU is available to evaluate on.
pub(crate) fn available() -> bool {
    #[cfg(all(target_os = "macos", not(miri)))]
    return nuvec_gpu::Device::shared().is_some();
    #[cfg(not(all(target_os = "macos", not(miri))))]
    false
}

/// The shared GPU's class letter for routing decisions (`'g'` when unknown).
pub(crate) fn class() -> char {
    #[cfg(all(target_os = "macos", not(miri)))]
    if let Some(device) = nuvec_gpu::Device::shared() {
        return device.info().class.unwrap_or('g');
    }
    'g'
}

/// The function constants that specialise the vector attention kernels' pipelines.
// Read only by the Metal encoder.
#[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct VectorFlags {
    pub(crate) has_mask: bool,
    /// Causal masking over several query rows.
    pub(crate) causal: bool,
    pub(crate) bool_mask: bool,
    pub(crate) float_mask: bool,
    pub(crate) sinks: bool,
    /// A sliding window.
    pub(crate) window: bool,
}

/// Threads per threadgroup the first pass of two-pass vector attention (four keys per
/// step) allows for these types and constants on the shared GPU, a fixed property of the
/// compiled pipeline (1024 without a GPU).
pub(crate) fn vector_threads(
    dtype: crate::DType,
    d: usize,
    dv: usize,
    flags: VectorFlags,
    blocks: usize,
) -> usize {
    #[cfg(all(target_os = "macos", not(miri)))]
    if let Some(device) = nuvec_gpu::Device::shared() {
        use nuvec_gpu::Constant;
        let w = if flags.window { "_w" } else { "" };
        let name = format!("nuvec_sdpa_vector_2pass_1_{}_{d}_{dv}{w}", dtype.name());
        let constants = [
            (30, Constant::Bool(flags.has_mask)),
            (31, Constant::Bool(false)),
            (32, Constant::Bool(flags.causal)),
            (33, Constant::Bool(flags.bool_mask)),
            (34, Constant::Bool(flags.float_mask)),
            (35, Constant::Bool(flags.sinks)),
            (36, Constant::Int(blocks as i32)),
            (37, Constant::Bool(flags.window)),
        ];
        if let Ok(p) = device.pipeline_with_constants(&name, &constants) {
            return p.max_threads_per_threadgroup();
        }
    }
    let _ = (dtype, d, dv, flags, blocks);
    1024
}

#[cfg(all(test, target_os = "macos", not(miri)))]
pub(crate) use metal::fault;
#[cfg(all(target_os = "macos", not(miri)))]
pub(crate) use metal::{BufferSignal, GpuEval};
#[cfg(not(all(target_os = "macos", not(miri))))]
pub(crate) use stub::GpuEval;

/// Whether `data` can be written in place as `meta`'s storage: the whole of a writable
/// Metal buffer (not a mapped file), from offset 0.
pub(crate) fn writable(data: &Data, meta: &Output) -> bool {
    #[cfg(all(target_os = "macos", not(miri)))]
    return data.offset == 0
        && data.buffer.len() == meta.size * meta.dtype.size()
        && data.buffer.metal().is_some_and(|b| !b.is_mapped());
    #[cfg(not(all(target_os = "macos", not(miri))))]
    {
        let _ = (data, meta);
        false
    }
}

/// Whether the element-wise kernels address `metas`, operands of an output of `shape` with
/// 2^32 elements or more: every operand flat (the output's layout) or a scalar (dispatched
/// in chunks of 2^31 elements), or all of them strided in at most three merged dimensions
/// (64-bit offsets).
#[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
pub(crate) fn large_elementwise(metas: &[&Output], shape: &[usize]) -> bool {
    let scalar = |m: &Output| m.size == 1 || m.strides.iter().all(|&s| s == 0);
    let flat = |m: &Output| m.flags.row_contiguous && m.shape.as_slice() == shape;
    if metas.iter().all(|m| scalar(m) || flat(m)) {
        return true;
    }
    if metas.is_empty() || metas.len() > 3 {
        return false;
    }
    let strides: Vec<&[isize]> = metas.iter().map(|m| m.strides.as_slice()).collect();
    let (merged, _) = collapse(shape, &strides);
    merged.len() <= 3 && merged.iter().all(|&d| d <= u32::MAX as usize)
}

#[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
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
        pub(crate) fn release_unreferenced(&mut self) {}
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
        pub(crate) fn collective(
            &mut self,
            _: &crate::distributed::CollectiveOp,
            _: usize,
            _: &[crate::Array],
            _: Vec<Data>,
            _: &Output,
            _: Option<Data>,
        ) -> Result<Vec<Data>> {
            unreachable!("no GPU on this target")
        }
        pub(crate) fn collectives_pending(&self) -> bool {
            false
        }
        pub(crate) fn defer(&mut self, node: Arc<Node>, outputs: Vec<Data>) {
            for (output, data) in node.outputs.iter().zip(outputs) {
                let _ = output.slot.set(Ok(data));
            }
        }
        pub(crate) fn slice_update_in_place(
            &mut self,
            _: Data,
            _: &Output,
            _: Operand<'_>,
            _: &[crate::view::DimSlice],
        ) -> Result<Data> {
            unreachable!("nothing is writable without Metal")
        }
        pub(crate) fn norm_rope_in_place(
            &mut self,
            _: Data,
            _: &crate::fast::imp::FastOp,
            _: &[Operand<'_>],
            _: &Output,
        ) -> Result<Data> {
            unreachable!("nothing is writable without Metal")
        }
        pub(crate) fn paged_write_in_place(
            &mut self,
            _: Data,
            _: &[Operand<'_>],
            _: &Output,
        ) -> Result<Data> {
            unreachable!("nothing is writable without Metal")
        }
        pub(crate) fn softmax_in_place(
            &mut self,
            _: Data,
            _: &Output,
            _: Option<usize>,
        ) -> Result<Data> {
            unreachable!("nothing is writable without Metal")
        }
        pub(crate) fn gated_delta_slots_in_place(
            &mut self,
            _: &[Operand<'_>],
            _: &Output,
            _: Data,
            _: &[Output],
        ) -> Result<Vec<Data>> {
            unreachable!("nothing is writable without Metal")
        }
        pub(crate) fn conv_slots_in_place(
            &mut self,
            _: &[Operand<'_>],
            _: &Output,
            _: Data,
            _: &[Output],
            _: bool,
        ) -> Result<Vec<Data>> {
            unreachable!("nothing is writable without Metal")
        }
        pub(crate) fn elementwise_in_place(
            &mut self,
            _: &Primitive,
            _: &[&Output],
            _: &[Option<&Data>],
            _: Data,
            _: &Output,
        ) -> Result<Data> {
            unreachable!("nothing is writable without Metal")
        }
        pub(crate) fn commit(&mut self) {}
        pub(crate) fn depend_on(&mut self, _: &[&crate::Array]) {}
        pub(crate) fn read_in_flight(&mut self, _: &crate::Array) {}
        pub(crate) fn lookup(&self, _: &Arc<Node>, _: u32) -> Option<Data> {
            None
        }
        pub(crate) fn take_pending(
            &mut self,
            _: &Arc<Node>,
            _: usize,
            _: impl Fn(&Data, usize) -> bool,
        ) -> Option<Data> {
            None
        }
        pub(crate) fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(all(target_os = "macos", not(miri)))]
mod metal {
    use crate::array::{ChunkStatus, NodeMap};

    use std::sync::{Mutex, PoisonError, Weak};

    use nuvec_gpu::{Arg, CommandBuffer, Device as MetalDevice, Pending};

    use super::*;
    use crate::buffer::Buffer;
    use crate::primitive::{ArgReduceOp, BinaryOp, ReduceOp, UnaryOp};
    use crate::{Category, DType, DeviceKind, Error};

    mod reduce;

    use crate::fused::Load;

    /// What the attention plan decided for one fused attention dispatch.
    #[derive(Debug, Clone, Copy)]
    struct AttentionFlags {
        scale: f32,
        causal: bool,
        has_mask: bool,
        has_sinks: bool,
        /// A sliding window of this many keys (0: none).
        window: u32,
        /// 1 or 2 for the vector kernels, 0 for the tiled kernel.
        passes: u8,
        blocks: usize,
        gqa_kernel: bool,
        tk1: bool,
    }

    /// The most dispatches per command buffer.
    const MAX_DISPATCHES: usize = 256;

    /// Outputs from this size on are released as soon as nothing refers to them.
    const LARGE: usize = 16 << 20;
    /// Dispatches in an evaluation's first command buffer, and in the next one whenever
    /// the GPU is idle.
    const IDLE_DISPATCHES: usize = 8;
    /// Command buffers in flight per device before submission blocks (MLX: 10).
    const MAX_IN_FLIGHT: usize = 10;
    /// Dispatches at which a command buffer that made new buffers is committed, as MLX
    /// commits every 50 ops (`max_ops_per_buffer`): later work then reuses its
    /// intermediates instead of making its own (the first steps after `clear_cache`).
    /// Command buffers that only reuse cached buffers grow up to [`MAX_DISPATCHES`].
    const FRESH_DISPATCHES: usize = 50;

    type Chunk = Vec<(Arc<Node>, Vec<Data>)>;

    /// A committed chunk's nodes, weakly: their data is in flight in the nodes themselves
    /// (exposed at commit) and is published from there. Nodes nothing refers to any more
    /// are dropped at once (their memory returns to the allocator, for later GPU work to
    /// reuse after these uses), and a consumer may take a pending output over (donation),
    /// so neither nodes nor data are held until completion.
    /// With its outputs' in-flight cells, which publishing locks before it upgrades the node:
    /// a consumer that holds a cell's lock (donation) sees no handle of publishing's.
    type Committed = Vec<(Weak<Node>, Vec<Arc<crate::array::InFlight>>)>;

    fn weak(chunk: &[(Arc<Node>, Vec<Data>)]) -> Committed {
        chunk
            .iter()
            .map(|(node, _)| {
                let cells = node.outputs.iter().map(|o| o.inflight.clone()).collect();
                (Arc::downgrade(node), cells)
            })
            .collect()
    }

    /// Publishes a completed node's in-flight outputs (`states`: their locked cells, in
    /// order); donated outputs are never published.
    fn publish_committed(
        node: &Node,
        states: &[std::sync::MutexGuard<'_, crate::array::InFlightState>],
        result: &Result<()>,
    ) {
        for (output, state) in node.outputs.iter().zip(states) {
            let data = match &**state {
                crate::array::InFlightState::Set(d) => Some(d.clone()),
                _ => None,
            };
            let value = match (result, data) {
                (Err(e), _) => Err(e.clone()),
                (Ok(()), Some(d)) => Ok(d),
                (Ok(()), None) => continue,
            };
            let _ = output.slot.set(value);
        }
    }

    /// Completion state shared with an evaluation's completion handlers.
    #[derive(Default)]
    struct Shared {
        state: Mutex<Publishing>,
    }

    #[derive(Default)]
    struct Publishing {
        /// Per committed chunk: its outputs, until published.
        chunks: Vec<Option<Committed>>,
        /// Per committed chunk: its result, once complete.
        results: Vec<Option<Result<()>>>,
        /// Per committed chunk: whether its outputs are exposed as in-flight data (they are
        /// published only then).
        exposed: Vec<bool>,
        /// Chunks published so far (in commit order).
        next: usize,
        /// Per committed chunk: its failure cell, set when it (or an earlier chunk of the
        /// evaluation) fails, for other evaluations that read its in-flight outputs.
        statuses: Vec<ChunkStatus>,
        /// The first failure; it fails every later chunk too.
        error: Option<Error>,
    }

    /// Lets later evaluations' GPU work read these outputs while they are in flight.
    fn expose(chunk: &Chunk, status: &ChunkStatus) {
        for (node, outputs) in chunk {
            for (output, data) in node.outputs.iter().zip(outputs) {
                output.inflight.set_from(data.clone(), status);
            }
        }
    }

    /// A chunk's result: its command buffer's, failed as well if a chunk of another
    /// evaluation whose in-flight outputs it read has failed. A failure is scoped to the
    /// evaluation that encoded the failed chunk and to readers of its results; independent
    /// evaluations are unaffected.
    ///
    /// The readers' command buffers run after the chunks they read (one queue), whose
    /// completion handlers Metal calls in order, so a producer's failure is recorded by the
    /// time a reader's handler checks it.
    /// A failed collective the chunk waits for fails it with the collective's own error.
    fn chunk_result(
        result: Result<()>,
        reads: &[ChunkStatus],
        collectives: &[ChunkStatus],
    ) -> Result<()> {
        if let Some(e) = collectives.iter().find_map(|s| s.get()) {
            return Err(e.clone());
        }
        result?;
        match reads.iter().find_map(|s| s.get()) {
            Some(e) => Err(Error::Backend(format!(
                "GPU work this result reads from failed: {e}"
            ))),
            None => Ok(()),
        }
    }

    impl Shared {
        /// Records chunk `k`'s result and publishes every chunk whose predecessors are all
        /// complete, in order.
        fn complete(&self, k: usize, result: Result<()>) {
            let mut p = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            p.results[k] = Some(result);
            Self::publish_ready(&mut p);
        }

        /// Records that chunk `k`'s outputs are exposed, and publishes what is ready.
        fn exposed(&self, k: usize) {
            let mut p = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            p.exposed[k] = true;
            Self::publish_ready(&mut p);
        }

        /// Publishes, in order, every chunk that is complete and exposed.
        fn publish_ready(p: &mut Publishing) {
            while let (Some(Some(result)), Some(true)) = (
                p.results.get(p.next).cloned(),
                p.exposed.get(p.next).copied(),
            ) {
                if let (Err(e), None) = (&result, &p.error) {
                    p.error = Some(e.clone());
                }
                let result = p.error.clone().map_or(Ok(()), Err);
                let next = p.next;
                if let (Err(e), Some(status)) = (&result, p.statuses.get(next)) {
                    let _ = status.set(e.clone());
                }
                for (node, cells) in p.chunks[next].take().unwrap_or_default() {
                    // Locked first, released only after every handle of publishing's to the
                    // node (the upgraded one, then the weak one) is dropped: a consumer that
                    // takes the lock sees the node's own handles only.
                    let states: Vec<_> = cells.iter().map(|c| c.lock()).collect();
                    if let Some(node) = node.upgrade() {
                        publish_committed(&node, &states, &result);
                    }
                    drop(node);
                    drop(states);
                }
                p.next += 1;
            }
        }
    }

    /// The input signals of one command buffer's collectives: its event, raised after each
    /// collective's input copy, and whether the command buffer has completed (a completed
    /// buffer that did not reach a value failed before it).
    pub(crate) struct BufferSignal {
        event: nuvec_gpu::Event,
        issued: std::sync::atomic::AtomicU64,
        done: std::sync::atomic::AtomicBool,
    }

    impl BufferSignal {
        fn new(device: &MetalDevice) -> Result<Self> {
            Ok(Self {
                event: nuvec_gpu::Event::new(device).map_err(backend)?,
                issued: Default::default(),
                done: Default::default(),
            })
        }

        fn next(&self) -> u64 {
            self.issued
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1
        }

        /// Waits until the command buffer has signalled `value`; an error if it completed
        /// without (its GPU work failed).
        pub(crate) fn wait(&self, value: u64) -> Result<()> {
            use std::sync::atomic::Ordering;
            loop {
                if self.event.wait(value, std::time::Duration::from_millis(5)) {
                    return Ok(());
                }
                if self.done.load(Ordering::Acquire) {
                    if self.event.value() >= value {
                        return Ok(());
                    }
                    return Err(Error::Backend(
                        "the GPU work producing this collective's input failed".into(),
                    ));
                }
            }
        }
    }

    pub(crate) struct GpuEval {
        device: Option<&'static MetalDevice>,
        cmd: Option<CommandBuffer>,
        dispatches: usize,
        /// Dispatches at which the open command buffer is committed (chosen at the first
        /// dispatch).
        limit: Option<usize>,
        /// Outputs encoded into the open command buffer.
        chunk: Chunk,
        /// Whether the open chunk holds an output of at least [`LARGE`] bytes: unreferenced
        /// outputs are then released after every dispatch, not only at commits.
        large: bool,
        /// Every pending output of this evaluation, for consumers encoded later.
        index: NodeMap<*const Node, (Weak<Node>, Vec<Data>)>,
        shared: Arc<Shared>,
        /// This evaluation's committed command buffers.
        submitted: Vec<Pending>,
        /// Other evaluations' in-flight outputs that deferred views of the open chunk read.
        foreign: Vec<crate::Array>,
        /// Failure cells of other evaluations' chunks whose in-flight outputs the open
        /// chunk reads.
        reads: Vec<ChunkStatus>,
        /// Failure cells of the collectives the open chunk waits for.
        collectives: Vec<ChunkStatus>,
        /// The open command buffer's collective input signals.
        signal: Option<Arc<BufferSignal>>,
        /// Holds back the dispatch policy's commits (between a collective's input copy and
        /// its signal).
        hold: bool,
        /// The device's system allocations when the open command buffer began.
        made: u64,
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

    /// A matrix-vector kernel's block shape (MLX's choice in `gemv_axbpy`/`gemv`).
    struct GemvConfig {
        kind: &'static str,
        bm: usize,
        bn: usize,
        sm: usize,
        sn: usize,
        tm: usize,
        tn: usize,
        /// Outputs per threadgroup.
        per_group: usize,
    }

    /// The block shape of a product of a vector of `in_len` with a matrix giving
    /// `out_len` outputs, read along its rows (`gemv`) or columns (`gemv_t`,
    /// `transpose_mat`). Independent of how many vectors a call multiplies, so one-row and
    /// multi-row products share it.
    fn gemv_config(transpose_mat: bool, in_len: usize, out_len: usize) -> GemvConfig {
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
            if in_len <= 64 {
                (bm, sm, sn) = (1, 8, 4);
            } else if in_len >= 16 * out_len {
                (bm, bn) = (1, 8);
            }
            tm = if out_len < tm { 1 } else { tm };
            ("gemv", bm * sm * tm)
        };
        GemvConfig {
            kind,
            bm,
            bn,
            sm,
            sn,
            tm,
            tn,
            per_group,
        }
    }

    /// The NAX GEMM's output tile for an `m × n` product: 64 × 64, or 16- or 32-row (or
    /// column) tiles when a side is shorter (`nax/gemm.metal`'s instantiations).
    fn nax_tile(m: usize, n: usize) -> (usize, usize) {
        let thin = |x: usize| if x > 16 { 32 } else { 16 };
        match (m >= 64, n >= 64) {
            (true, true) => (64, 64),
            (false, true) => (thin(m), 64),
            (true, false) => (64, thin(n)),
            (false, false) => (32, 32),
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
            BinaryOp::LogAddExp => "logaddexp",
            BinaryOp::FloorDivide => "floor_divide",
            BinaryOp::Power => "power",
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

    /// A 3-D copy's geometry in 16-byte units when the innermost dimension is contiguous
    /// in every layout and every stride and offset is a multiple of the unit (fewer, wider
    /// memory accesses); the element width in bits to copy with.
    fn widen(
        size: usize,
        shape: &mut [usize],
        strides: &mut [&mut Vec<isize>],
        offsets: &mut [&mut isize],
    ) -> usize {
        let n = 16 / size;
        let last = shape.len() - 1;
        let aligned = n > 1
            && shape[last].is_multiple_of(n)
            && strides
                .iter()
                .all(|s| s[last] == 1 && s[..last].iter().all(|&x| x.rem_euclid(n as isize) == 0))
            && offsets.iter().all(|o| o.rem_euclid(n as isize) == 0);
        if !aligned {
            return bits(size);
        }
        shape[last] /= n;
        for s in strides.iter_mut() {
            for x in &mut s[..last] {
                *x /= n as isize;
            }
        }
        for o in offsets.iter_mut() {
            **o /= n as isize;
        }
        128
    }

    /// An operand of the quantized kernels: its buffer and byte offset, in a layout the
    /// kernels read (MLX's `ensure_row_contiguous[_matrix]`).
    struct Packed<'a> {
        buffer: GemmBuffer<'a>,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<isize>,
    }

    /// Metal's name of a kernel dtype.
    fn msl_type(dtype: DType) -> &'static str {
        match dtype {
            DType::F16 => "half",
            DType::BF16 => "bfloat",
            _ => "float",
        }
    }

    /// mlx-lm's generic gated delta kernel (`_make_gated_delta_kernel`, scalar gates, no
    /// mask; MIT, Apple): one SIMD-group per value row, `Dk / 32` state values per lane.
    const GATED_DELTA: &str = r#"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        if (dv_idx >= Dv) {
          return;
        }

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * static_cast<float>(g_[hv_idx]);
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
            out += state[i] * static_cast<float>(q_[s_idx]);
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
"#;

    /// Whether the packed gated delta kernel may run: mlx-lm's `MLX_GDN_PACKED` switch,
    /// read once (`0` keeps every shape on the generic kernel, for A/B checks).
    fn gdn_packed_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("MLX_GDN_PACKED").as_deref() != Ok("0"))
    }

    /// mlx-lm's packed gated delta kernel (`_make_gated_delta_packed_kernel`, `Dk = 128`,
    /// float32 gates and state; MIT, Apple): four lanes per value row, eight rows per
    /// SIMD-group, the same reduction tree as the generic kernel.
    const GATED_DELTA_PACKED: &str = r#"
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
"#;

    /// The final state store of [`GATED_DELTA`] and its output pointer.
    const GENERIC_STORE: &str = "        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
";
    const GENERIC_STATE: (&str, &str) = (
        "auto o_state = state_out + (n * Dv + dv_idx) * Dk;",
        "auto o_state = state_out + ((b_idx * T * Hv + hv_idx) * Dv + dv_idx) * Dk;",
    );
    /// The final state store of [`GATED_DELTA_PACKED`] and its output pointer.
    const PACKED_STORE: &str = "        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
";
    const PACKED_STATE: (&str, &str) = (
        "auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;",
        "auto o_state = state_out + ((b_idx * T * Hv + hv_idx) * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;",
    );

    /// The state handling of a gated delta kernel.
    #[derive(Clone, Copy)]
    enum DeltaMode {
        /// The final state, from its own input.
        Final,
        /// Every step's state (`[B, T, ...]`).
        Steps,
        /// States at `slots[b]` of a pool, read and written in place.
        Slots,
    }

    /// A gated delta kernel body reading and writing sequence `b`'s state at slot
    /// `slots[b]` of a state pool (`[slots, Hv, Dv, Dk]`): each thread reads its state
    /// elements first and writes the same ones last, so the pool is updated in place.
    fn at_slots(body: &str) -> String {
        let from = "(n * Dv + dv_idx) * Dk";
        let to = "((slots[b_idx] * Hv + hv_idx) * Dv + dv_idx) * Dk";
        assert!(
            body.matches(from).count() == 2,
            "gated delta kernel source changed"
        );
        body.replace(from, to)
    }

    /// A gated delta kernel body that stores the state after every step (`[B, T, Hv, Dv,
    /// Dk]`) instead of once at the end: the final store moves into the step loop, before
    /// the pointers advance, and the output pointer starts at step 0 of `[B, T, ...]`. The
    /// arithmetic is untouched, so outputs and states equal the plain kernel's.
    fn every_step(body: &str, store: &str, (state, state_steps): (&str, &str)) -> String {
        let advance = "          q_ += Hk * Dk;";
        assert!(
            body.matches(store).count() == 1
                && body.matches(state).count() == 1
                && body.matches(advance).count() == 1,
            "gated delta kernel source changed"
        );
        let in_loop = format!(
            "  {}        o_state += Hv * Dv * Dk;\n",
            store.replace("\n  ", "\n    ")
        );
        body.replace(store, "")
            .replace(state, state_steps)
            .replace(advance, &format!("{in_loop}{advance}"))
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

    /// Whether a grid of `shape` (also the output's row-contiguous layout) and operands at
    /// `(offset, strides)` (in elements) stay below 2^31 elements, for 32-bit offsets.
    fn fits32(shape: &[usize], operands: &[(isize, &[isize])]) -> bool {
        let limit = i32::MAX as usize;
        let size: usize = shape.iter().product();
        size < limit
            && operands.iter().all(|(offset, strides)| {
                offset.unsigned_abs()
                    + strides
                        .iter()
                        .zip(shape)
                        .map(|(s, &n)| s.unsigned_abs() * n.saturating_sub(1))
                        .sum::<usize>()
                    < limit
            })
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
                limit: None,
                chunk: Vec::new(),
                large: false,
                index: NodeMap::default(),
                shared: Arc::default(),
                submitted: Vec::new(),
                foreign: Vec::new(),
                reads: Vec::new(),
                collectives: Vec::new(),
                signal: None,
                hold: false,
                made: 0,
            }
        }

        fn device(&mut self) -> Option<&'static MetalDevice> {
            if self.device.is_none() {
                self.device = MetalDevice::shared();
                self.made = self.device.map_or(0, |d| d.system_allocations());
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
            if !operands.iter().all(|(_, d)| d.buffer.metal().is_some()) {
                return false;
            }
            if !(operands.iter().all(|(m, _)| fits(m)) && outputs.iter().all(fits))
                && !large_supported(primitive, operands, outputs)
            {
                return false;
            }
            let dtype = operands.first().map(|(m, _)| m.dtype);
            let not64 = |d: DType| d != DType::F64;
            match (primitive, dtype) {
                (Primitive::MetalKernel(_), _) => true,
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
                            BinaryOp::LogAddExp => is_float(d),
                            BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply => true,
                            BinaryOp::Equal | BinaryOp::NotEqual => true,
                            _ => d != DType::Complex64,
                        }
                }
                (Primitive::Select, _) => true,
                (Primitive::AsType(to), Some(d)) => not64(d) && not64(*to),
                (
                    Primitive::Reshape(_)
                    | Primitive::Squeeze(_)
                    | Primitive::ExpandDims(_)
                    | Primitive::Copy
                    | Primitive::View(_),
                    _,
                ) => true,
                (Primitive::ToFp8, Some(d)) => is_float(d),
                (Primitive::SortedOffsets { .. }, _) => true,
                (Primitive::FromFp8(d), _) => is_float(*d),
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
                (
                    Primitive::GatherAxis { .. }
                    | Primitive::Concatenate { .. }
                    | Primitive::SliceUpdate { .. },
                    _,
                ) => true,
                (Primitive::Sort { .. } | Primitive::ArgSort { .. }, Some(d)) => {
                    // MLX's kernels: integers and the GPU's floats (no bool, f64, complex).
                    !matches!(d, DType::Bool | DType::F64 | DType::Complex64)
                        && operands[0].0.size < i32::MAX as usize
                }
                (Primitive::Scan { op, .. }, Some(d)) => match op {
                    crate::graph::ScanOp::LogAddExp => is_float(d),
                    _ => !matches!(d, DType::Bool | DType::F64 | DType::Complex64),
                },
                (Primitive::ScatterAxis { reduce, .. }, Some(d)) => {
                    *reduce == crate::graph::ScatterReduce::None
                        || matches!(
                            d,
                            DType::F32 | DType::F16 | DType::BF16 | DType::I32 | DType::U32
                        )
                }
                (Primitive::Dequantize { .. }, _) => outputs[0].size > 0,
                (Primitive::RandomBits { .. }, _) => outputs[0].size > 0,
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
            if let Primitive::MetalKernel(call) = primitive {
                return self.metal_kernel(call, operands, outputs);
            }
            if let Primitive::Compiled(fused) = primitive {
                return self.fused(fused, operands, outputs);
            }
            if let Primitive::Quantize { quant } = primitive {
                let (group_size, bits) = quant.checked("quantize")?;
                if quant.mode != crate::QuantMode::Affine {
                    return self.fp_quantize(operands[0], outputs, quant.mode);
                }
                return self.quantize(operands[0], outputs, group_size, bits);
            }
            let out = &outputs[0];
            let mut buffer = Buffer::try_alloc_gpu(out.size * out.dtype.size())?;
            match primitive {
                Primitive::Unary(_) | Primitive::Binary(_) | Primitive::Select => {
                    let name = Self::elementwise_name(primitive, operands[0].0.dtype, out);
                    self.elementwise(&name, operands, out, &mut buffer)?;
                }
                Primitive::AsType(to) => {
                    let name = format!("nuvec_cast_{}_{}", operands[0].0.dtype.name(), to.name());
                    self.elementwise(&name, operands, out, &mut buffer)?;
                }
                Primitive::ToFp8 => {
                    let name = format!("nuvec_to_fp8_{}", operands[0].0.dtype.name());
                    self.elementwise(&name, operands, out, &mut buffer)?;
                }
                Primitive::SortedOffsets { groups } => {
                    self.sorted_offsets(operands[0], *groups, &mut buffer)?
                }
                Primitive::FromFp8(to) => {
                    let name = format!("nuvec_from_fp8_{}", to.name());
                    self.elementwise(&name, operands, out, &mut buffer)?;
                }
                Primitive::Reshape(_)
                | Primitive::Squeeze(_)
                | Primitive::ExpandDims(_)
                | Primitive::Copy
                | Primitive::View(_) => self.copy(operands[0], &mut buffer)?,
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
                Primitive::Sort { axis } => self.sort(operands[0], *axis, false, &mut buffer)?,
                Primitive::ArgSort { axis } => self.sort(operands[0], *axis, true, &mut buffer)?,
                Primitive::Scan {
                    op,
                    axis,
                    reverse,
                    inclusive,
                } => self.scan(*op, operands[0], *axis, (*reverse, *inclusive), &mut buffer)?,
                Primitive::RandomBits { shape, width } => {
                    let per_key = shape.iter().product::<usize>() * usize::from(*width);
                    self.random_bits(operands[0], per_key, &mut buffer)?;
                }
                Primitive::SliceUpdate { .. } => {
                    self.copy(operands[0], &mut buffer)?;
                    let slices = primitive.dim_slices(out.shape.len())?;
                    let strides = crate::shape::row_contiguous_strides(&out.shape);
                    let (region, delta) = crate::view::slice_geometry(&strides, &slices);
                    self.copy_into(operands[1], &region, delta, &mut buffer)?;
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
                Primitive::Dequantize { quant, .. } => {
                    let (group_size, bits) = quant.checked("dequantize")?;
                    match quant.mode {
                        crate::QuantMode::Affine => {
                            self.dequantize(operands, out, group_size, bits, &mut buffer)?
                        }
                        mode => self.fp_dequantize(operands, out, mode, bits, &mut buffer)?,
                    }
                }
                Primitive::QuantizedMatmul { quant, transpose } => {
                    let (group_size, bits) = quant.checked("quantized_matmul")?;
                    if quant.mode == crate::QuantMode::Affine {
                        self.quantized_matmul(
                            operands,
                            out,
                            (group_size, bits, *transpose),
                            &mut buffer,
                        )?
                    } else {
                        self.fp_quantized_matmul(operands, out, *quant, *transpose, &mut buffer)?
                    }
                }
                _ => unreachable!("supports() rejects other primitives"),
            }
            Ok(vec![Data::new(buffer)])
        }

        /// Element-wise kernels: the `_c` variant when every operand is flat or a scalar
        /// (`_cn` from 2^16 elements), else `_g`.
        fn elementwise(
            &mut self,
            base: &str,
            operands: &[Operand<'_>],
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let metas: Vec<&Output> = operands.iter().map(|(m, _)| *m).collect();
            let datas: Vec<Option<&Data>> = operands.iter().map(|(_, d)| Some(*d)).collect();
            self.elementwise_with(base, &metas, &datas, out, buffer)
        }

        /// The kernel of element-wise `primitive` over operands of dtype `dtype`.
        fn elementwise_name(primitive: &Primitive, dtype: DType, out: &Output) -> String {
            match primitive {
                Primitive::Unary(op) => format!("nuvec_unary_{}_{}", unary_name(*op), dtype.name()),
                Primitive::Binary(op) => {
                    format!("nuvec_binary_{}_{}", binary_name(*op), dtype.name())
                }
                Primitive::Select => format!("nuvec_select_{}", bits(out.dtype.size())),
                _ => unreachable!("an element-wise primitive"),
            }
        }

        /// Element-wise `primitive` written into the donated buffer of its input whose
        /// data is `None` (bound to the output's slot too: each element is read before it
        /// is written, at the same position).
        pub(crate) fn elementwise_in_place(
            &mut self,
            primitive: &Primitive,
            metas: &[&Output],
            datas: &[Option<&Data>],
            mut base: Data,
            out: &Output,
        ) -> Result<Data> {
            let name = Self::elementwise_name(primitive, metas[0].dtype, out);
            let buffer = Arc::get_mut(&mut base.buffer).expect("donated data has one handle");
            self.elementwise_with(&name, metas, datas, out, buffer)?;
            Ok(base)
        }

        /// Element-wise kernel `base` over operands `metas` with their data (`None`: the
        /// output buffer itself, an in-place operand at offset 0).
        fn elementwise_with(
            &mut self,
            base: &str,
            metas: &[&Output],
            datas: &[Option<&Data>],
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let modes: Vec<Mode> = metas.iter().map(|m| mode(m, &out.shape)).collect();
            let general = modes.iter().any(|m| matches!(m, Mode::Strided));
            let mut consts: Vec<Vec<u8>> = datas
                .iter()
                .map(|d| bytes(&(d.map_or(0, |d| d.offset) as i64)))
                .collect();
            let mut grid = None;
            let name = if general {
                let strides: Vec<&[isize]> = metas.iter().map(|m| m.strides.as_slice()).collect();
                let (mut shape, mut collapsed) = collapse(&out.shape, &strides);
                // Up to three dimensions: a 3-D grid, offsets without divisions.
                let small = shape.len() <= 3 && shape.iter().all(|&d| d <= u32::MAX as usize);
                if small {
                    while shape.len() < 3 {
                        shape.insert(0, 1);
                        collapsed.iter_mut().for_each(|s| s.insert(0, 0));
                    }
                    grid = Some([shape[2], shape[1], shape[0]]);
                }
                // 32-bit offsets when the output and every operand's reachable offsets
                // stay below 2^31 (64-bit integer products are emulated on Apple GPUs).
                let reach = |offset: isize, strides: &[isize]| -> usize {
                    offset.unsigned_abs()
                        + strides
                            .iter()
                            .zip(&shape)
                            .map(|(s, &n)| s.unsigned_abs() * n.saturating_sub(1))
                            .sum::<usize>()
                };
                let limit = i32::MAX as usize;
                let fits32 = out.size < limit
                    && collapsed
                        .iter()
                        .zip(datas)
                        .all(|(st, d)| reach(d.map_or(0, |d| d.offset), st) < limit);
                consts.push(shape_bytes(&shape));
                consts.extend(collapsed.iter().map(|s| strides_bytes(s)));
                consts.push(bytes(&(shape.len() as i32)));
                let suffix = match (small, fits32) {
                    (true, true) => "3s",
                    (true, false) => "3",
                    (false, _) => "",
                };
                format!("{base}_g{suffix}")
            } else {
                let flags: u32 = modes
                    .iter()
                    .enumerate()
                    .map(|(i, m)| u32::from(matches!(m, Mode::Scalar)) << i)
                    .sum();
                consts.push(bytes(&flags));
                format!("{base}_c")
            };
            // From 2^16 elements a flat kernel computes `8 / itemsize` of them per thread
            // (MLX's work per thread), bounded by a size argument. The last operand holds
            // the data type (a select's first is its condition).
            let per_thread = match metas.last() {
                Some(m) if !general && out.size >= 1 << 16 => (8 / m.dtype.size()).max(1),
                _ => 1,
            };
            let name = if per_thread > 1 {
                consts.push(Vec::new());
                format!("{name}n")
            } else {
                name
            };
            let output_slot = datas.len();
            // The flat kernels index with 32 bits: 2^32 elements and more run in chunks of
            // 2^31, the output (and an in-place operand) bound at the chunk's first element
            // and every other flat operand offset to it (MLX's `_v2` kernels, without them).
            let chunk = match (&grid, general) {
                (None, false) => FLAT_CHUNK,
                _ => out.size,
            };
            let mut start = 0;
            while start < out.size {
                let n = chunk.min(out.size - start);
                let at = start * out.dtype.size();
                for (i, (d, m)) in datas.iter().zip(&modes).enumerate() {
                    if let (Some(d), Mode::Flat) = (d, m) {
                        consts[i] = bytes(&(d.offset as i64 + start as i64));
                    }
                }
                if per_thread > 1 {
                    *consts.last_mut().expect("the size slot") = bytes(&(n as u32));
                }
                let mut args: Vec<Arg<'_>> = datas
                    .iter()
                    .map(|d| match d {
                        Some(d) => Arg::read(d.buffer.metal().expect("checked")),
                        None => Arg::Same(output_slot, at),
                    })
                    .collect();
                args.push(Arg::Write(buffer.metal_mut().expect("device storage"), at));
                args.extend(consts.iter().map(|c| Arg::bytes(c)));
                match grid {
                    Some(grid) => self.dispatch_grid(&name, &args, grid)?,
                    None => self.dispatch(&name, &args, n.div_ceil(per_thread))?,
                }
                start += n;
            }
            Ok(())
        }

        /// One thread per position of a 3-D grid (`[x, y, z]`, x innermost).
        fn dispatch_grid(&mut self, name: &str, args: &[Arg<'_>], grid: [usize; 3]) -> Result<()> {
            if grid.contains(&0) {
                return Ok(());
            }
            let pipeline = self.pipeline(name)?;
            let group = block_dims(grid);
            self.encode_threads(&pipeline, args, grid, group)?;
            self.counted()
        }

        /// A fused element-wise graph as one generated kernel: each input is read as a
        /// scalar, flat or through its (collapsed) strides, and the kernel is compiled once
        /// per graph and combination of reads. Without strided inputs, a thread computes
        /// `8 / itemsize` elements of arrays of at least 65,536 (MLX's rule).
        fn fused(
            &mut self,
            fused: &crate::graph::Fused,
            operands: &[Operand<'_>],
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            let out = &outputs[0];
            let mut buffers = outputs
                .iter()
                .map(|o| Buffer::try_alloc_gpu(o.size * o.dtype.size()))
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
            self.encode_dispatch(&pipeline, &args, out.size.div_ceil(per_thread))?;
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
            // Whole contiguous rows with one index each (`take` along a leading axis).
            let nd = ind.shape.len();
            let row: usize = ind.shape[axis + 1..].iter().product();
            let rows_layout = ind.shape[..axis].iter().all(|&d| d == 1)
                && (axis + 1..nd).all(|d| ind.strides[d] == 0 || ind.shape[d] == 1)
                && crate::shape::row_contiguous_strides(&src.shape[axis + 1..])
                    .iter()
                    .zip(&src.strides[axis + 1..])
                    .zip(&src.shape[axis + 1..])
                    .all(|((want, got), &n)| n == 1 || want == got)
                && row <= u32::MAX as usize
                && ind.shape[axis] <= u32::MAX as usize;
            if rows_layout && row > 0 {
                let name = format!(
                    "nuvec_gather_rows_{}_{}",
                    bits(src.dtype.size()),
                    ind.dtype.name()
                );
                let consts = [
                    bytes(&(src_data.offset as i64)),
                    bytes(&(ind_data.offset as i64)),
                    bytes(&(ind.strides[axis] as i64)),
                    bytes(&(src.strides[axis] as i64)),
                    bytes(&(src.shape[axis] as i64)),
                    bytes(&(row as i64)),
                ];
                let mut args = vec![
                    Arg::read(src_data.buffer.metal().expect("checked")),
                    Arg::read(ind_data.buffer.metal().expect("checked")),
                    Arg::write(buffer.metal_mut().expect("device storage")),
                ];
                args.extend(consts.iter().map(|c| Arg::bytes(c)));
                return self.dispatch_grid(&name, &args, [row, ind.shape[axis], 1]);
            }
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
            reduce: crate::graph::ScatterReduce,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let ((dst, _), (ind, ind_data), (upd, upd_data)) =
                (operands[0], operands[1], operands[2]);
            let name = match reduce {
                crate::graph::ScatterReduce::None => format!(
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

        /// `slice_update` written into its base's own buffer (`base`, donated: no other
        /// handle exists): only the update's region is written.
        pub(crate) fn slice_update_in_place(
            &mut self,
            mut base: Data,
            out: &Output,
            update: Operand<'_>,
            slices: &[crate::view::DimSlice],
        ) -> Result<Data> {
            let strides = crate::shape::row_contiguous_strides(&out.shape);
            let (region, delta) = crate::view::slice_geometry(&strides, slices);
            let buffer = Arc::get_mut(&mut base.buffer).expect("donated data has one handle");
            self.copy_into(update, &region, delta, buffer)?;
            Ok(base)
        }

        /// MLX's merge sort (`gpu_merge_sort`): one threadgroup per lane for axes that fit a
        /// block, else block sorts and merge passes in temporary buffers. With `arg`, the
        /// output is the sorting indices (uint32).
        fn sort(
            &mut self,
            (meta, data): Operand<'_>,
            axis: usize,
            arg: bool,
            buffer: &mut Buffer,
        ) -> Result<()> {
            if meta.size == 0 {
                return Ok(());
            }
            let n = meta.shape[axis];
            let rows = meta.size / n;
            let tn = 4;
            let mut bn = match n.div_ceil(tn) {
                p if p > 256 => 512,
                p if p > 128 => 256,
                p if p > 64 => 128,
                p if p > 32 => 64,
                _ => 32,
            };
            if bn == 512 && meta.dtype.size() > 4 {
                bn = 256;
            }
            let blocks = n.div_ceil(bn * tn);
            let dtype = meta.dtype.name();
            let i32b = |v: usize| bytes(&(v as i32));
            let mut nc_shape: Vec<usize> = meta.shape.to_vec();
            let mut nc_strides: Vec<isize> = meta.strides.to_vec();
            nc_shape.remove(axis);
            nc_strides.remove(axis);
            let input = Arg::Read(
                data.buffer.metal().expect("checked"),
                data.offset as usize * meta.dtype.size(),
            );
            if blocks == 1 {
                let out_strides = crate::shape::row_contiguous_strides(&meta.shape);
                let mut out_nc: Vec<isize> = out_strides.to_vec();
                out_nc.remove(axis);
                let name = if arg {
                    format!("nuvec_ncarg_block_sort_{dtype}_uint32_bn{bn}_tn{tn}")
                } else {
                    format!("nuvec_nc_block_sort_{dtype}_{dtype}_bn{bn}_tn{tn}")
                };
                let consts = [
                    i32b(n),
                    i32b(meta.strides[axis] as usize),
                    i32b(out_strides[axis] as usize),
                    i32b(nc_shape.len()),
                    shape_bytes(&nc_shape),
                    strides_bytes(&nc_strides),
                    strides_bytes(&out_nc),
                ];
                let mut args = vec![
                    input,
                    Arg::write(buffer.metal_mut().expect("device storage")),
                ];
                args.extend(consts.iter().map(|c| Arg::bytes(c)));
                return self.dispatch_groups(&name, &args, [1, rows, 1], [bn, 1, 1]);
            }
            // Multi-block: sorted lanes as `[rows, n]` values and indices, ping-ponged.
            let lane = rows * n;
            let mut vals = [
                Buffer::try_alloc_gpu(lane * meta.dtype.size())?,
                Buffer::try_alloc_gpu(lane * meta.dtype.size())?,
            ];
            let mut idxs = [
                Buffer::try_alloc_gpu(lane * 4)?,
                Buffer::try_alloc_gpu(lane * 4)?,
            ];
            let mut partitions = Buffer::try_alloc_gpu(rows * (blocks + 1) * 4)?;
            let suffix = format!("{dtype}_uint32_bn{bn}_tn{tn}");
            {
                let consts = [
                    i32b(n),
                    i32b(meta.strides[axis] as usize),
                    i32b(nc_shape.len().max(1)),
                    shape_bytes(if nc_shape.is_empty() { &[0] } else { &nc_shape }),
                    strides_bytes(if nc_strides.is_empty() {
                        &[1]
                    } else {
                        &nc_strides
                    }),
                ];
                let [v0, _] = &mut vals;
                let [i0, _] = &mut idxs;
                let mut args = vec![
                    input,
                    Arg::write(v0.metal_mut().expect("device storage")),
                    Arg::write(i0.metal_mut().expect("device storage")),
                ];
                args.extend(consts.iter().map(|c| Arg::bytes(c)));
                let name = format!("nuvec_sort_mbsort_{suffix}");
                self.dispatch_groups(&name, &args, [blocks, rows, 1], [bn, 1, 1])?;
            }
            let threads = (blocks + 1).min(1024);
            let mut current = 0;
            let mut merge_tiles = 2;
            while merge_tiles / 2 < blocks {
                let (from, to) = (current, 1 - current);
                let [v0, v1] = &mut vals;
                let [i0, i1] = &mut idxs;
                let (vin, vout) = if from == 0 { (v0, v1) } else { (v1, v0) };
                let (iin, iout) = if from == 0 { (i0, i1) } else { (i1, i0) };
                let consts = [i32b(n), i32b(merge_tiles), i32b(blocks)];
                {
                    let mut args = vec![
                        Arg::write(partitions.metal_mut().expect("device storage")),
                        Arg::read(vin.metal().expect("device storage")),
                        Arg::read(iin.metal().expect("device storage")),
                    ];
                    args.extend(consts.iter().map(|c| Arg::bytes(c)));
                    let name = format!("nuvec_partition_mbsort_{suffix}");
                    self.dispatch_groups(&name, &args, [1, rows, 1], [threads, 1, 1])?;
                }
                let mut args = vec![
                    Arg::read(partitions.metal().expect("device storage")),
                    Arg::read(vin.metal().expect("device storage")),
                    Arg::read(iin.metal().expect("device storage")),
                    Arg::write(vout.metal_mut().expect("device storage")),
                    Arg::write(iout.metal_mut().expect("device storage")),
                ];
                args.extend(consts.iter().map(|c| Arg::bytes(c)));
                let name = format!("nuvec_merge_mbsort_{suffix}");
                self.dispatch_groups(&name, &args, [blocks, rows, 1], [bn, 1, 1])?;
                current = to;
                merge_tiles *= 2;
            }
            // `[rows, n]` (the sorted axis innermost) into the output's layout.
            let [v0, v1] = vals;
            let [i0, i1] = idxs;
            let (result, dtype) = match (arg, current) {
                (true, 0) => (i0, DType::U32),
                (true, _) => (i1, DType::U32),
                (false, 0) => (v0, meta.dtype),
                (false, _) => (v1, meta.dtype),
            };
            let mut src_strides = vec![0isize; meta.shape.len()];
            let mut step = n as isize;
            for d in (0..meta.shape.len()).rev() {
                if d == axis {
                    src_strides[d] = 1;
                } else {
                    src_strides[d] = step;
                    step *= meta.shape[d] as isize;
                }
            }
            let src_meta = crate::kernels::strided(dtype, &meta.shape, &src_strides);
            let src = Data::new(result);
            let out_strides = crate::shape::row_contiguous_strides(&meta.shape);
            self.copy_into((&src_meta, &src), &out_strides, 0, buffer)
        }

        /// MLX's scans (`scan_gpu_inplace`): the contiguous kernel when the axis is
        /// innermost, else the strided one, over a row-contiguous input (copied first if
        /// needed).
        fn scan(
            &mut self,
            op: crate::graph::ScanOp,
            operand: Operand<'_>,
            axis: usize,
            (reverse, inclusive): (bool, bool),
            buffer: &mut Buffer,
        ) -> Result<()> {
            let (meta, data) = operand;
            if meta.size == 0 {
                return Ok(());
            }
            let packed;
            let (in_buffer, offset) = if meta.flags.row_contiguous {
                (data.buffer.metal().expect("checked"), data.offset as usize)
            } else {
                let mut copy = Buffer::try_alloc_gpu(meta.size * meta.dtype.size())?;
                self.copy(operand, &mut copy)?;
                packed = copy;
                (packed.metal().expect("device storage"), 0)
            };
            let n = meta.shape[axis];
            let strides = crate::shape::row_contiguous_strides(&meta.shape);
            let stride = strides[axis] as usize;
            let op_name = match op {
                crate::graph::ScanOp::Sum => "sum",
                crate::graph::ScanOp::Prod => "prod",
                crate::graph::ScanOp::Max => "max",
                crate::graph::ScanOp::Min => "min",
                crate::graph::ScanOp::LogAddExp => "logaddexp",
            };
            let variant = format!(
                "{}{}_{op_name}_{}",
                if reverse { "reverse_" } else { "" },
                if inclusive { "inclusive" } else { "exclusive" },
                meta.dtype.name()
            );
            let n_reads = if meta.dtype.size() <= 4 { 4 } else { 2 };
            let size = bytes(&(n as u64));
            let mut args = vec![
                Arg::Read(in_buffer, offset * meta.dtype.size()),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&size),
            ];
            if stride == 1 {
                let per_simd = n_reads * 32;
                let max = 1024;
                let threads = if n <= n_reads * 1024 {
                    n.div_ceil(per_simd) * 32
                } else if n <= n_reads * 2048 {
                    (n / 2).div_ceil(per_simd) * 32
                } else {
                    max
                }
                .min(max);
                let name = format!("nuvec_contig_scan_{variant}");
                self.dispatch_groups(&name, &args, [1, meta.size / n, 1], [threads, 1, 1])
            } else {
                let blocks = stride.div_ceil(32);
                let (stride_b, blocks_b) = (bytes(&(stride as u64)), bytes(&(blocks as u64)));
                args.extend([Arg::bytes(&stride_b), Arg::bytes(&blocks_b)]);
                let threads = (32 / n_reads) * 32;
                let groups = meta.size / (n * stride) * blocks;
                let name = format!("nuvec_strided_scan_{variant}");
                self.dispatch_groups(&name, &args, [1, groups, 1], [threads, 1, 1])
            }
        }

        /// Copies an operand into the region of `buffer` at element `offset` with `strides`.
        fn copy_into(
            &mut self,
            (meta, data): Operand<'_>,
            strides: &[isize],
            offset: isize,
            buffer: &mut Buffer,
        ) -> Result<()> {
            if meta.size == 0 {
                return Ok(());
            }
            // Dimensions contiguous in both layouts merged: at most three take a 3-D grid
            // without index divisions.
            let (mut shape, [mut from, mut to]) =
                nuvec_cpu::collapse_dims(&meta.shape, [&meta.strides, strides]);
            let small = shape.len() <= 3 && shape.iter().all(|&d| d <= u32::MAX as usize);
            let (mut from_off, mut to_off) = (data.offset, offset);
            let mut width = bits(meta.dtype.size());
            if small {
                while shape.len() < 3 {
                    shape.insert(0, 1);
                    from.insert(0, 0);
                    to.insert(0, 0);
                }
                width = widen(
                    meta.dtype.size(),
                    &mut shape,
                    &mut [&mut from, &mut to],
                    &mut [&mut from_off, &mut to_off],
                );
            }
            let consts = [
                bytes(&(from_off as i64)),
                bytes(&(to_off as i64)),
                shape_bytes(&shape),
                strides_bytes(&from),
                strides_bytes(&to),
                bytes(&(shape.len() as i32)),
            ];
            let mut args = vec![
                Arg::read(data.buffer.metal().expect("checked")),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            let name = format!("nuvec_copy_into_{width}");
            if small {
                let s = if fits32(&shape, &[(from_off, &from), (to_off, &to)]) {
                    "s"
                } else {
                    ""
                };
                return self.dispatch_grid(
                    &format!("{name}_g3{s}"),
                    &args,
                    [shape[2], shape[1], shape[0]],
                );
            }
            self.dispatch(&name, &args, meta.size)
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
            let mut start = 0isize;
            for &operand in operands {
                let offset = start * out_strides[axis];
                start += operand.0.shape[axis] as isize;
                self.copy_into(operand, &out_strides, offset, buffer)?;
            }
            Ok(())
        }

        /// Gathers an operand into contiguous memory, walking its own logical shape (a
        /// reshape keeps the row-major element order of its input).
        fn copy(&mut self, (meta, data): Operand<'_>, buffer: &mut Buffer) -> Result<()> {
            let (mut shape, [mut strides]) = nuvec_cpu::collapse_dims(&meta.shape, [&meta.strides]);
            let small = shape.len() <= 3 && shape.iter().all(|&d| d <= u32::MAX as usize);
            let mut from_off = data.offset;
            let mut width = bits(meta.dtype.size());
            if small {
                while shape.len() < 3 {
                    shape.insert(0, 1);
                    strides.insert(0, 0);
                }
                // The output is row-contiguous: aligned with the input's innermost run.
                width = widen(
                    meta.dtype.size(),
                    &mut shape,
                    &mut [&mut strides],
                    &mut [&mut from_off],
                );
            }
            let consts = [
                bytes(&(from_off as i64)),
                shape_bytes(&shape),
                strides_bytes(&strides),
                bytes(&(shape.len() as i32)),
            ];
            let suffix = match small {
                true if fits32(&shape, &[(from_off, &strides)]) => "3s",
                true => "3",
                false => "",
            };
            let name = format!("nuvec_copy_{width}_g{suffix}");
            let input = data.buffer.metal().expect("checked");
            let mut args = vec![
                Arg::read(input),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            args.extend(consts.iter().map(|c| Arg::bytes(c)));
            match small {
                true => self.dispatch_grid(&name, &args, [shape[2], shape[1], shape[0]]),
                false => self.dispatch(&name, &args, meta.size),
            }
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
            let mut partial = Buffer::try_alloc_gpu(groups * slices * meta.dtype.size())?;
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
            // One threadgroup per output, up to 1024 threads (a power of two) scanning
            // at least 4 elements each.
            let n = meta.shape[axis];
            let threads = (n / 4).clamp(1, 1024).next_power_of_two().min(1024);
            self.dispatch_groups(&name, &args, [out.size, 1, 1], [threads, 1, 1])
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
            // A few rows: the multi-row matrix-vector kernels, each row with the bits of
            // its one-row product (the row-independence contract).
            if m <= crate::ROW_INDEPENDENT_ROWS {
                return self.gemv_rows(a, b, c, (m, n, k), out, buffer);
            }
            let ga = self.gemmand(a, m, k)?;
            let gb = self.gemmand(b, k, n)?;
            let class = self
                .device()
                .and_then(|d| d.info().architecture.chars().last())
                .unwrap_or('s');
            if splits_k(class, batch, m, n, k) {
                if self.nax_gemm(out.dtype) {
                    return self.nax_gemm_splitk(&ga, &gb, c, (m, n, k), out, buffer);
                }
                return self.gemm_splitk(&ga, &gb, c, (m, n, k), out, buffer);
            }
            // On the GPU's matrix units (Metal 4 TensorOps: the Neural Accelerators of
            // M5-class GPUs) when active: 64 × 64 tiles, and 16- or 32-row (or column)
            // tiles for thin products.
            let nax = self.nax_gemm(out.dtype);
            let (bm, bn, bk, wm, wn) = match nax {
                true => {
                    let (bm, bn) = nax_tile(m, n);
                    (bm, bn, 64, 0, 0)
                }
                false => gemm_tiles(
                    class,
                    out.dtype,
                    ga.transposed,
                    gb.transposed,
                    batch * m * n,
                    m,
                    n,
                    k,
                ),
            };
            let has_batch = batch_shape.len() > 1;
            let source = c.filter(|&(_, alpha, beta)| alpha != 0.0 || beta != 1.0);
            let axpby = source.is_some_and(|(_, alpha, beta)| alpha != 1.0 || beta != 1.0);
            let layout = format!(
                "{}{}_{}",
                if ga.transposed { 't' } else { 'n' },
                if gb.transposed { 't' } else { 'n' },
                out.dtype.name()
            );
            let name = match nax {
                true => format!("nuvec_nax_gemm_{layout}_bm{bm}_bn{bn}"),
                false => format!("nuvec_gemm_{layout}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}"),
            };
            let all = [
                (10, has_batch),
                (100, source.is_some()),
                (110, axpby),
                (200, m % bm == 0),
                (201, n % bn == 0),
                (202, k % bk == 0),
            ];
            // The NAX kernel has no alignment constants.
            let constants = &all[..if nax { 3 } else { 6 }];
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
            let pipeline = match self.pipeline_with(&name, constants) {
                Ok(pipeline) => pipeline,
                // A NAX failure turns NAX off: the same product on the SIMD kernels.
                Err(_) if nax => return self.gemm(a, b, c, out, buffer),
                Err(e) => return Err(e),
            };
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
                Arg::bytes(&[0u8; 8]), // the NAX kernel's split-K sizes (unused)
            ];
            let threads = match nax {
                true => [32 * bm * bn / 1024, 1, 1], // `execution_simdgroups<BM·BN/1024>`
                false => [32, wn, wm],
            };
            self.encode_groups(
                &pipeline,
                &args,
                [n.div_ceil(bn), m.div_ceil(bm), batch],
                threads,
            )?;
            self.counted()
        }

        /// Whether a product of `dtype` runs on the NAX GEMM ([`nuvec_gpu::Device::nax`]):
        /// half and float32 types (float32 at TensorOps' relaxed precision, as MLX's TF32
        /// default), unless [`crate::__nax_route`] turned it off on this thread.
        fn nax_gemm(&mut self, dtype: DType) -> bool {
            matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
                && crate::NAX_ROUTE.with(std::cell::Cell::get)
                && self.device().is_some_and(|d| d.nax().is_active())
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
            let mut partial = Buffer::try_alloc_gpu(partitions * m * n * 4)?;
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
            self.encode_groups(
                &pipeline,
                &args,
                [n.div_ceil(bn), m.div_ceil(bm), partitions],
                [32, wn, wm],
            )?;
            self.counted()?;
            self.splitk_accum(&partial, partitions, c, (m, n), out, buffer)
        }

        /// The NAX split-K GEMM: float32 partial products of K partitions (multiples of 64)
        /// on the matrix units, summed by [`Self::splitk_accum`].
        fn nax_gemm_splitk(
            &mut self,
            ga: &Gemmand<'_>,
            gb: &Gemmand<'_>,
            c: Option<(Operand<'_>, f32, f32)>,
            (m, n, k): (usize, usize, usize),
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let (bm, bn) = nax_tile(m, n);
            let tiles = m.div_ceil(bm) * n.div_ceil(bn);
            let partitions = ((k / 64) / tiles).next_power_of_two().clamp(2, 32);
            let size = k.div_ceil(partitions).next_multiple_of(64);
            let partitions = k.div_ceil(size);
            let mut partial = Buffer::try_alloc_gpu(partitions * m * n * 4)?;
            let name = format!(
                "nuvec_nax_gemm_splitk_{}{}_{}_bm{bm}_bn{bn}",
                if ga.transposed { 't' } else { 'n' },
                if gb.transposed { 't' } else { 'n' },
                out.dtype.name(),
            );
            let params = GemmParams {
                m: m as i32,
                n: n as i32,
                k: k as i32,
                lda: ga.ld,
                ldb: gb.ld,
                ldd: n as i32,
                tiles_n: n.div_ceil(bn) as i32,
                tiles_m: m.div_ceil(bm) as i32,
                batch_stride_a: 0,
                batch_stride_b: 0,
                batch_stride_d: 0,
                swizzle_log: 0,
                gemm_k_iterations_aligned: 0,
                batch_ndim: 0,
                _pad: 0,
            };
            let pipeline =
                match self.pipeline_with(&name, &[(10, false), (100, false), (110, false)]) {
                    Ok(pipeline) => pipeline,
                    // A NAX failure turns NAX off: MLX's split-K on the SIMD kernels.
                    Err(_) => return self.gemm_splitk(ga, gb, c, (m, n, k), out, buffer),
                };
            let params = bytemuck::bytes_of(&params);
            let split = [size as i32, (m * n) as i32];
            let unused = [0u8; 4];
            let args = [
                ga.arg(),
                gb.arg(),
                Arg::bytes(&unused),
                Arg::write(partial.metal_mut().expect("device storage")),
                Arg::bytes(params),
                Arg::bytes(&unused),
                Arg::bytes(&unused),
                Arg::bytes(&unused),
                Arg::bytes(bytemuck::cast_slice(&split)),
            ];
            self.encode_groups(
                &pipeline,
                &args,
                [n.div_ceil(bn), m.div_ceil(bm), partitions],
                [32 * bm * bn / 1024, 1, 1],
            )?;
            self.counted()?;
            self.splitk_accum(&partial, partitions, c, (m, n), out, buffer)
        }

        /// Sums `partitions` float32 partial products of `m × n` into `buffer` (and, for
        /// `addmm`, scales them and adds `c`): `nuvec_gemm_splitk_accum[_axpby]`.
        fn splitk_accum(
            &mut self,
            partial: &Buffer,
            partitions: usize,
            c: Option<(Operand<'_>, f32, f32)>,
            (m, n): (usize, usize),
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
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
            self.encode_threads(&pipeline, &args, [n, m, 1], [n.min(32), m.min(32), 1])?;
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
            let GemvConfig {
                kind,
                bm,
                bn,
                sm,
                sn,
                tm,
                tn,
                per_group,
            } = gemv_config(transpose_mat, in_len, out_len);
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
            self.encode_groups(
                &pipeline,
                &args,
                [out_len.div_ceil(per_group), 1, batch],
                [32, bn, bm],
            )?;
            self.counted()
        }

        /// `a · b` for `2 ≤ m ≤ ROW_INDEPENDENT_ROWS` rows of `a`: the matrix-vector
        /// kernels' multi-row variants, which read `b` once and give each row the bits of
        /// [`Self::gemv`] on that row alone (same block shape, same accumulation order).
        fn gemv_rows(
            &mut self,
            a: Operand<'_>,
            b: Operand<'_>,
            c: Option<(Operand<'_>, f32, f32)>,
            (m, n, k): (usize, usize, usize),
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let nd = out.shape.len();
            let mat = self.gemmand(b, k, n)?;
            // The rows are read contiguously along K, each `ld` apart.
            let vec = match Gemmand::view(a, m, k) {
                Some(g) if !g.transposed => g,
                _ => self.gemmand_copy(a)?,
            };
            let transpose_mat = !mat.transposed;
            let GemvConfig {
                kind,
                bm,
                bn,
                sm,
                sn,
                tm,
                tn,
                per_group,
            } = gemv_config(transpose_mat, k, n);
            // Groups of rows of one tile are neighbours in the grid, so the matrix block
            // comes from the cache for all but the first group: four rows per group in
            // half precision (more registers per thread are slower), eight in float32
            // (whose larger matrices fall out of the cache between groups). Measured on
            // an M4 Pro (`just bench-rows`).
            let rows = match m {
                ..=2 => 2,
                _ if m <= 4 || out.dtype != DType::F32 => 4,
                _ => 8,
            };
            let groups = m.div_ceil(rows);
            let axpby = c.filter(|&(_, alpha, beta)| alpha != 1.0 || beta != 0.0);
            let batch_shape: Vec<usize> = match &out.shape[..nd - 2] {
                [] => vec![1],
                s => s.to_vec(),
            };
            let batch: usize = batch_shape.iter().product();
            let nc = batch_shape.len() > 1;
            let name = format!(
                "nuvec_{kind}_rows{rows}_{}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc{}_axpby{}",
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
            let (bias_strides, bias_stride, bias_row_stride, alpha, beta) = match axpby {
                Some(((cm, _), alpha, beta)) => (
                    strides(&cm.strides[..nd - 2]),
                    cm.strides[nd - 1] as i32,
                    cm.strides[nd - 2] as i32,
                    alpha,
                    beta,
                ),
                None => (strides(&[]), 0, 0, 1.0, 0.0),
            };
            let ints = |v: usize| (v as i32).to_le_bytes();
            let scalars = [
                ints(k),
                ints(n),
                mat.ld.to_le_bytes(),
                alpha.to_le_bytes(),
                beta.to_le_bytes(),
                ints(nd.saturating_sub(2).max(1)),
                bias_stride.to_le_bytes(),
                ints(m),
                bias_row_stride.to_le_bytes(),
                ints(groups),
            ];
            let row_stride = i64::from(vec.ld).to_le_bytes();
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
                Arg::bytes(&scalars[7]),
                Arg::bytes(&row_stride),
                Arg::bytes(&scalars[8]),
                Arg::bytes(&scalars[9]),
            ];
            self.encode_groups(
                &pipeline,
                &args,
                [n.div_ceil(per_group) * groups, 1, batch],
                [32, bn, bm],
            )?;
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
            let mut copy = Buffer::try_alloc_gpu(meta.size * meta.dtype.size())?;
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
            self.encode_threads(&pipeline, args, [threads, 1, 1], [group, 1, 1])?;
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
            self.encode_threads(pipeline, args, [rows * group, 1, 1], [group, 1, 1])?;
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
            op: &crate::fast::imp::FastOp,
            operands: &[Operand<'_>],
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            use crate::fast::imp::FastOp;
            let alloc = |o: &Output| Buffer::try_alloc_gpu(o.size * o.dtype.size());
            if let FastOp::Attention {
                scale,
                causal,
                mask,
                sinks,
                window,
                passes,
                blocks,
                gqa_kernel,
                tk1,
            } = *op
            {
                let mut out = alloc(&outputs[0])?;
                let flags = AttentionFlags {
                    scale: f32::from_bits(scale),
                    causal,
                    has_mask: mask,
                    has_sinks: sinks,
                    window,
                    passes,
                    blocks: blocks as usize,
                    gqa_kernel,
                    tk1,
                };
                self.attention(operands, &outputs[0], flags, &mut out)?;
                return Ok(vec![Data::new(out)]);
            }
            if let FastOp::GatedDelta { steps } = *op {
                return self.gated_delta(operands, outputs, steps);
            }
            if *op == FastOp::GatedDeltaSlots {
                return self.gated_delta_slots(operands, outputs);
            }
            if let FastOp::MoeRoute { k, normalize } = *op {
                return self.moe_route(operands[0], (k, normalize), outputs);
            }
            if *op == FastOp::WeightedSum {
                return self.weighted_sum(operands, &outputs[0]);
            }
            if let FastOp::ConvSlots { silu } = *op {
                let mut pool = Buffer::try_alloc_gpu(outputs[1].size * outputs[1].dtype.size())?;
                self.copy(operands[2], &mut pool)?;
                let y = self.conv_slots_into(operands, outputs, &mut pool, silu)?;
                return Ok(vec![y, Data::new(pool)]);
            }
            if let FastOp::GatherQmm {
                quant,
                transpose,
                global_scale,
            } = *op
            {
                let (group_size, bits) = quant.checked("gather_qmm")?;
                let mut out = alloc(&outputs[0])?;
                if quant.mode != crate::QuantMode::Affine {
                    let options = (quant, transpose, global_scale);
                    self.fp_gather_qmm(operands, &outputs[0], options, &mut out)?;
                    return Ok(vec![Data::new(out)]);
                }
                self.gather_qmm(operands, &outputs[0], (group_size, bits as u8), &mut out)?;
                return Ok(vec![Data::new(out)]);
            }
            if *op == FastOp::GatherMm {
                let mut out = alloc(&outputs[0])?;
                self.gather_mm(operands, &outputs[0], &mut out)?;
                return Ok(vec![Data::new(out)]);
            }
            if let FastOp::AddRmsNorm { eps } = *op {
                return self.add_rms_norm(operands, outputs, eps);
            }
            if let FastOp::ResidualRmsNorm { eps, scaled, next } = *op {
                return self.residual_rms_norm(operands, outputs, (eps, scaled, next));
            }
            if let FastOp::DepthwiseConv1d {
                stride,
                padding,
                dilation,
            } = *op
            {
                let mut out = alloc(&outputs[0])?;
                let x = self.packed(operands[0], true)?;
                let w = self.packed(operands[1], true)?;
                let (n, l, c) = (x.shape[0], x.shape[1], x.shape[2]);
                let (k, l_out) = (w.shape[1], outputs[0].shape[1]);
                let as_i32 = |v: usize| {
                    i32::try_from(v).map_err(|_| Error::InvalidArgument("conv1d: too large".into()))
                };
                let sizes = [as_i32(l)?, as_i32(c)?, as_i32(k)?, as_i32(l_out)?];
                let params = [as_i32(stride)?, as_i32(padding)?, as_i32(dilation)?];
                let args = [
                    x.arg(),
                    w.arg(),
                    Arg::write(out.metal_mut().expect("device storage")),
                    Arg::bytes(bytemuck::bytes_of(&sizes)),
                    Arg::bytes(bytemuck::bytes_of(&params)),
                ];
                let grid = [c, l_out, n];
                let name = format!("nuvec_depthwise_conv1d_{}", operands[0].0.dtype.name());
                let pipeline = self.pipeline(&name)?;
                self.encode_threads(&pipeline, &args, grid, block_dims(grid))?;
                self.counted()?;
                return Ok(vec![Data::new(out)]);
            }
            if let FastOp::PagedAttention {
                scale,
                window,
                sinks,
            } = *op
            {
                let mut out = alloc(&outputs[0])?;
                let options = (f32::from_bits(scale), window, sinks);
                self.paged_attention(operands, options, &mut out)?;
                return Ok(vec![Data::new(out)]);
            }
            if *op == FastOp::PagedWrite {
                let mut out = alloc(&outputs[0])?;
                self.copy(operands[0], &mut out)?;
                self.paged_write_into(&operands[1..], &outputs[0].shape, &mut out)?;
                return Ok(vec![Data::new(out)]);
            }
            if let FastOp::NormRope { cache, .. } = *op {
                let mut out = alloc(&outputs[0])?;
                if !cache {
                    self.norm_rope_into(op, operands, &outputs[0].shape, &mut out)?;
                    return Ok(vec![Data::new(out)]);
                }
                self.copy(operands[0], &mut out)?;
                self.norm_rope_into(op, &operands[1..], &outputs[0].shape, &mut out)?;
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
                    self.row_softmax(kind, dt, (rows, axis), Some(x.arg()), None, &mut out)?;
                }
                FastOp::CausalSoftmax { rows: tile } => {
                    let tile = Some(*tile as usize);
                    self.row_softmax("softmax", dt, (rows, axis), Some(x.arg()), tile, &mut out)?;
                }
                FastOp::RmsNorm { eps } | FastOp::LayerNorm { eps } => {
                    let layer = matches!(op, FastOp::LayerNorm { .. });
                    let w = self.packed(operands[1], true)?;
                    let b = match layer {
                        true => Some(self.packed(operands[2], true)?),
                        false => None,
                    };
                    let (limit, reads) = if layer { (6656, 8) } else { (4096, 4) };
                    let name = |looped: bool| {
                        format!(
                            "nuvec_{}{}_{dtn}",
                            if layer { "layer_norm" } else { "rms" },
                            if looped { "_looped" } else { "" }
                        )
                    };
                    let mut looped = axis > limit;
                    let mut pipeline = self.pipeline(&name(looped))?;
                    if !looped && row_threads(axis, reads) > pipeline.max_threads_per_threadgroup()
                    {
                        looped = true;
                        pipeline = self.pipeline(&name(true))?;
                    }
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
                    let name = |looped: bool| {
                        format!(
                            "nuvec_vjp_{}{}_{dtn}",
                            if layer { "layer_norm" } else { "rms" },
                            if looped { "_looped" } else { "" }
                        )
                    };
                    let mut looped = axis > limit;
                    let mut pipeline = self.pipeline_with(&name(looped), &[(20, has_w)])?;
                    if !looped && row_threads(axis, reads) > pipeline.max_threads_per_threadgroup()
                    {
                        looped = true;
                        pipeline = self.pipeline_with(&name(true), &[(20, has_w)])?;
                    }
                    // RMS rows are grouped (about 512 threadgroups); layer norm: one row each.
                    let per_group = if !layer && !looped {
                        rows.div_ceil(512)
                    } else {
                        1
                    };
                    let groups = rows.div_ceil(per_group);
                    let group = if looped {
                        pipeline.max_threads_per_threadgroup()
                    } else {
                        row_threads(axis, reads)
                    };
                    let mut temp = Buffer::try_alloc_gpu(if has_w {
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
                    self.encode_threads(&pipeline, &args, grid, group)?;
                    self.counted()?;
                }
                FastOp::Attention { .. }
                | FastOp::AddRmsNorm { .. }
                | FastOp::ResidualRmsNorm { .. }
                | FastOp::NormRope { .. }
                | FastOp::DepthwiseConv1d { .. }
                | FastOp::PagedAttention { .. }
                | FastOp::PagedWrite
                | FastOp::MoeRoute { .. }
                | FastOp::WeightedSum
                | FastOp::ConvSlots { .. }
                | FastOp::GatedDelta { .. }
                | FastOp::GatedDeltaSlots
                | FastOp::GatherQmm { .. }
                | FastOp::GatherMm => {
                    unreachable!("handled above")
                }
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

        /// A custom kernel ([`crate::fast::MetalKernel`]): its source generated for the inputs'
        /// dtypes, compiled once per source, and dispatched over its grid. Outputs with an
        /// initial value start as copies of it.
        fn metal_kernel(
            &mut self,
            call: &crate::kernel::KernelCall,
            operands: &[Operand<'_>],
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            use crate::kernel::{Binding, InputLayout};
            let n = call.input_count();
            let ins = operands[..n]
                .iter()
                .map(|&o| match call.row_contiguous_inputs() {
                    true => self.packed(o, true),
                    false => self.view_if(o, true),
                })
                .collect::<Result<Vec<_>>>()?;
            let layouts: Vec<InputLayout> = operands[..n]
                .iter()
                .map(|(m, _)| InputLayout {
                    dtype: m.dtype,
                    scalar: m.shape.is_empty(),
                })
                .collect();
            let mut outs = outputs
                .iter()
                .map(|o| Buffer::try_alloc_gpu(o.size * o.dtype.size()))
                .collect::<Result<Vec<_>>>()?;
            for (i, out) in outs.iter_mut().enumerate() {
                if let Some(&init) = operands.get(n + i) {
                    self.copy(init, out)?;
                }
            }
            let (name, source) = call.source(&layouts);
            let device = self
                .device()
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            let pipeline = device
                .pipeline_from_source(&name, || source)
                .map_err(backend)?;
            let bindings = call.bindings(&layouts);
            let consts: Vec<Vec<u8>> = bindings
                .iter()
                .map(|b| match *b {
                    Binding::Shape(i) => shape_bytes(&ins[i].shape),
                    Binding::Strides(i) => strides_bytes(&ins[i].strides),
                    Binding::Ndim(i) => (ins[i].shape.len() as i32).to_le_bytes().to_vec(),
                    _ => Vec::new(),
                })
                .collect();
            let mut writes = outs
                .iter_mut()
                .map(|b| Some(b.metal_mut().expect("device storage")))
                .collect::<Vec<_>>();
            let mut args = Vec::with_capacity(bindings.len());
            for (b, c) in bindings.iter().zip(&consts) {
                args.push(match *b {
                    Binding::Input(i) => ins[i].arg(),
                    Binding::Output(i) => Arg::write(writes[i].take().expect("bound once")),
                    _ => Arg::bytes(c),
                });
            }
            self.encode_threads(&pipeline, &args, call.grid, call.threadgroup)?;
            drop(args);
            self.counted()?;
            Ok(outs.into_iter().map(Data::new).collect())
        }

        /// mlx-lm's gated delta kernels (`gated_delta.py`), generated for the head sizes and
        /// dtypes as its `metal_kernel`s are: `(q, k, v, g, beta, state)` to `(y, state)`.
        /// One SIMD-group per value row, or, for `Dk = 128` with float32 gates and state,
        /// the packed kernel (eight rows per SIMD-group).
        /// [`crate::fast::FastOp::AddRmsNorm`]: `x + r` and its RMS normalisation, one
        /// threadgroup per row (looped over long rows, as the RMS kernels).
        fn add_rms_norm(
            &mut self,
            operands: &[Operand<'_>],
            outputs: &[Output],
            eps: u32,
        ) -> Result<Vec<Data>> {
            let x = self.packed(operands[0], true)?;
            let r = self.packed(operands[1], true)?;
            let w = self.packed(operands[2], true)?;
            let dtn = operands[0].0.dtype.name();
            let axis = x.shape[x.shape.len() - 1];
            let rows = x.shape.iter().product::<usize>() / axis;
            let name = |looped: bool| {
                format!("nuvec_add_rms{}_{dtn}", if looped { "_looped" } else { "" })
            };
            let mut looped = axis > 4096;
            let mut pipeline = self.pipeline(&name(looped))?;
            if !looped && row_threads(axis, 4) > pipeline.max_threads_per_threadgroup() {
                looped = true;
                pipeline = self.pipeline(&name(true))?;
            }
            let group = if looped {
                pipeline.max_threads_per_threadgroup()
            } else {
                row_threads(axis, 4)
            };
            let alloc = |o: &Output| Buffer::try_alloc_gpu(o.size * o.dtype.size());
            let (mut sum, mut out) = (alloc(&outputs[0])?, alloc(&outputs[1])?);
            let e = eps.to_le_bytes();
            let n = (axis as u32).to_le_bytes();
            let ws = u32::from(!w.shape.is_empty()).to_le_bytes();
            let args = [
                x.arg(),
                r.arg(),
                w.arg(),
                Arg::write(sum.metal_mut().expect("device storage")),
                Arg::write(out.metal_mut().expect("device storage")),
                Arg::bytes(&e),
                Arg::bytes(&n),
                Arg::bytes(&ws),
            ];
            self.dispatch_rows(&pipeline, &args, rows, group)?;
            Ok(vec![Data::new(sum), Data::new(out)])
        }

        /// [`crate::fast::FastOp::ResidualRmsNorm`]: `(h + rms(x)) · scale` and its RMS
        /// normalisation, one threadgroup per row. The single-row or looped variant and the
        /// threadgroup size are the RMS kernel's, so both reductions add as it adds.
        fn residual_rms_norm(
            &mut self,
            operands: &[Operand<'_>],
            outputs: &[Output],
            (eps, scaled, next): (u32, bool, Option<u32>),
        ) -> Result<Vec<Data>> {
            let packed = operands
                .iter()
                .map(|&o| self.packed(o, true))
                .collect::<Result<Vec<_>>>()?;
            let (h, x, w) = (&packed[0], &packed[1], &packed[2]);
            let dtn = operands[0].0.dtype.name();
            let axis = h.shape[h.shape.len() - 1];
            let rows = h.shape.iter().product::<usize>() / axis;
            let rms =
                |looped: bool| format!("nuvec_rms{}_{dtn}", if looped { "_looped" } else { "" });
            let mut looped = axis > 4096;
            let mut pipeline = self.pipeline(&rms(looped))?;
            if !looped && row_threads(axis, 4) > pipeline.max_threads_per_threadgroup() {
                looped = true;
                pipeline = self.pipeline(&rms(true))?;
            }
            let group = if looped {
                pipeline.max_threads_per_threadgroup()
            } else {
                row_threads(axis, 4)
            };
            let name = format!(
                "nuvec_residual_rms{}_{dtn}_s{}_n{}",
                if looped { "_looped" } else { "" },
                u8::from(scaled),
                u8::from(next.is_some())
            );
            let pipeline = self.pipeline(&name)?;
            if pipeline.max_threads_per_threadgroup() < group {
                return Err(Error::Backend(format!(
                    "{name}: {group} threads per threadgroup needed, {} available",
                    pipeline.max_threads_per_threadgroup()
                )));
            }
            let alloc = |o: &Output| Buffer::try_alloc_gpu(o.size * o.dtype.size());
            let mut outs = outputs.iter().map(alloc).collect::<Result<Vec<_>>>()?;
            let stride = |p: &Packed<'_>| u32::from(!p.shape.is_empty()).to_le_bytes();
            let (e, e2) = (eps.to_le_bytes(), next.unwrap_or(0).to_le_bytes());
            let n = (axis as u32).to_le_bytes();
            let (ws, ws2) = (stride(w), stride(&packed[packed.len() - 1]));
            let mut args = vec![h.arg(), x.arg(), w.arg()];
            // Absent operands bind the weight again (never read).
            args.push(if scaled { packed[3].arg() } else { w.arg() });
            args.push(if next.is_some() {
                packed[packed.len() - 1].arg()
            } else {
                w.arg()
            });
            let (first, rest) = outs.split_at_mut(1);
            args.push(Arg::write(first[0].metal_mut().expect("device storage")));
            match rest.first_mut() {
                Some(o) => args.push(Arg::write(o.metal_mut().expect("device storage"))),
                None => args.push(Arg::bytes(&e2)),
            }
            args.extend([Arg::bytes(&e), Arg::bytes(&e2), Arg::bytes(&n)]);
            args.extend([Arg::bytes(&ws), Arg::bytes(&ws2)]);
            self.dispatch_rows(&pipeline, &args, rows, group)?;
            Ok(outs.into_iter().map(Data::new).collect())
        }

        /// MLX's softmax or log-sum-exp kernel (`kind`) over `rows` rows of `axis` values,
        /// from `input` (`None`: in place, reading `out`).
        fn row_softmax(
            &mut self,
            kind: &str,
            dt: DType,
            (rows, axis): (usize, usize),
            input: Option<Arg<'_>>,
            causal: Option<usize>,
            out: &mut Buffer,
        ) -> Result<()> {
            // The causal variants (`FastOp::CausalSoftmax`) take the rows per block.
            let prefix = if causal.is_some() { "causal_" } else { "" };
            let name = |looped: bool| {
                format!(
                    "nuvec_{prefix}{kind}_{}_{}",
                    if looped { "looped" } else { "block" },
                    dt.name()
                )
            };
            // One block per row while the row fits the pipeline's threadgroup limit (which
            // depends on the GPU), else the looped kernel.
            let mut pipeline = self.pipeline(&name(axis > 4096))?;
            let mut looped = axis > 4096;
            if !looped && row_threads(axis, 4) > pipeline.max_threads_per_threadgroup() {
                looped = true;
                pipeline = self.pipeline(&name(true))?;
            }
            let group = if looped {
                pipeline.max_threads_per_threadgroup()
            } else {
                row_threads(axis, 4)
            };
            let n = (axis as i32).to_le_bytes();
            let tile = (causal.unwrap_or(1) as i32).to_le_bytes();
            let args = [
                input.unwrap_or(Arg::Same(1, 0)),
                Arg::write(out.metal_mut().expect("device storage")),
                Arg::bytes(&n),
                Arg::bytes(&tile),
            ];
            self.dispatch_rows(&pipeline, &args, rows, group)
        }

        /// Softmax over the last axis written into its donated input's buffer (`base`): each
        /// thread writes only the values it read.
        /// With `causal` ([`crate::fast::FastOp::CausalSoftmax`]'s rows per block), masked.
        pub(crate) fn softmax_in_place(
            &mut self,
            mut base: Data,
            out: &Output,
            causal: Option<usize>,
        ) -> Result<Data> {
            let axis = out.shape.last().copied().unwrap_or(1);
            let rows = out.size / axis.max(1);
            let buffer = Arc::get_mut(&mut base.buffer).expect("donated data has one handle");
            self.row_softmax("softmax", out.dtype, (rows, axis), None, causal, buffer)?;
            Ok(base)
        }

        /// [`crate::fast::FastOp::PagedAttention`]: one threadgroup of 32 SIMD-groups per
        /// (sequence, query head, query), as MLX's single-pass vector attention.
        fn paged_attention(
            &mut self,
            operands: &[Operand<'_>],
            (scale, window, sinks): (f32, u32, bool),
            buffer: &mut Buffer,
        ) -> Result<()> {
            use nuvec_gpu::Constant;
            let ins = operands
                .iter()
                .map(|&o| self.packed(o, true))
                .collect::<Result<Vec<_>>>()?;
            let constants = [
                (40, Constant::Bool(sinks)),
                (41, Constant::Bool(window > 0)),
            ];
            let wb = (window as i32).to_le_bytes();
            // The windowed kernels (`_w`).
            let w = if window > 0 { "_w" } else { "" };
            let (q, cache, tables) = (operands[0].0, operands[1].0, operands[3].0);
            let (b, hq, lq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
            let (num_blocks, hkv, block_size) = (cache.shape[0], cache.shape[1], cache.shape[2]);
            let max_blocks = tables.shape[1];
            let as_i32 = |v: usize| {
                i32::try_from(v)
                    .map_err(|_| Error::InvalidArgument("paged_attention: too large".into()))
            };
            // The kernels clamp positions to the capacity in `int`s, with headroom for the
            // key loops' strides.
            let capacity = max_blocks * block_size;
            as_i32(capacity.saturating_mul(2).saturating_add(1 << 16))?;
            let shift = block_size.trailing_zeros() as usize;
            let sc = scale.to_le_bytes();
            let dtn = q.dtype.name();
            // Two passes as MLX's vector attention would take over the keys read (the
            // lengths are not known here: the table's capacity, or the window's keys), with
            // its key partitions.
            let class = self
                .device()
                .and_then(|dev| dev.info().class)
                .unwrap_or('g');
            let gqa = hq / hkv;
            let span = match window {
                0 => capacity,
                w => capacity.min(w as usize + lq - 1),
            };
            if crate::fast::plan::vector_two_pass(class, span, hkv < hq) && 32 * gqa * lq <= 1024 {
                let blocks = crate::fast::plan::vector_blocks(class, span, gqa * lq);
                let rows = b * hq * lq;
                let mut partial = Buffer::try_alloc_gpu(rows * blocks * d * q.dtype.size())?;
                let mut sums = Buffer::try_alloc_gpu(rows * blocks * 4)?;
                let mut maxs = Buffer::try_alloc_gpu(rows * blocks * 4)?;
                let params = [
                    as_i32(hkv)?,
                    as_i32(lq)?,
                    as_i32(shift)?,
                    as_i32(max_blocks)?,
                    as_i32(blocks)?,
                    as_i32(num_blocks)?,
                ];
                {
                    let mut args: Vec<Arg<'_>> = ins.iter().take(3).map(Packed::arg).collect();
                    args.extend([
                        Arg::write(partial.metal_mut().expect("device storage")),
                        Arg::write(sums.metal_mut().expect("device storage")),
                        Arg::write(maxs.metal_mut().expect("device storage")),
                        ins[3].arg(),
                        ins[4].arg(),
                        Arg::bytes(bytemuck::bytes_of(&params)),
                        Arg::bytes(&sc),
                    ]);
                    if sinks {
                        args.push(ins[5].arg());
                    }
                    if window > 0 {
                        while args.len() < 11 {
                            args.push(Arg::bytes(&wb));
                        }
                        args.push(Arg::bytes(&wb));
                    }
                    let name = format!("nuvec_paged_attention_2pass_1_{dtn}_{d}{w}");
                    let pipeline = self
                        .device()
                        .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?
                        .pipeline_with_constants(&name, &constants)
                        .map_err(backend)?;
                    self.encode_groups(&pipeline, &args, [hkv, b, blocks], [32, gqa, lq])?;
                    self.counted()?;
                }
                let bl = as_i32(blocks)?.to_le_bytes();
                let args = [
                    Arg::read(partial.metal().expect("device storage")),
                    Arg::read(sums.metal().expect("device storage")),
                    Arg::read(maxs.metal().expect("device storage")),
                    Arg::write(buffer.metal_mut().expect("device storage")),
                    Arg::bytes(&bl),
                ];
                let name = format!("nuvec_sdpa_vector_2pass_2_{dtn}_{d}");
                return self.dispatch_groups(&name, &args, [b * hq, lq, 1], [1024, 1, 1]);
            }
            let params = [
                as_i32(hq)?,
                as_i32(hkv)?,
                as_i32(lq)?,
                as_i32(shift)?,
                as_i32(max_blocks)?,
                as_i32(num_blocks)?,
            ];
            let mut args: Vec<Arg<'_>> = ins.iter().take(3).map(Packed::arg).collect();
            args.push(Arg::write(buffer.metal_mut().expect("device storage")));
            args.extend([ins[3].arg(), ins[4].arg()]);
            args.extend([Arg::bytes(bytemuck::bytes_of(&params)), Arg::bytes(&sc)]);
            if sinks {
                args.push(ins[5].arg());
            }
            if window > 0 {
                while args.len() < 9 {
                    args.push(Arg::bytes(&wb));
                }
                args.push(Arg::bytes(&wb));
            }
            let name = format!("nuvec_paged_attention_{dtn}_{d}{w}");
            let pipeline = self
                .device()
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?
                .pipeline_with_constants(&name, &constants)
                .map_err(backend)?;
            self.encode_groups(&pipeline, &args, [b * hq, lq, 1], [1024, 1, 1])?;
            self.counted()
        }

        /// [`crate::fast::FastOp::PagedWrite`]'s kernel over `(new, slots)` into a cache
        /// buffer of `cache_shape`: one thread per element of `new` (`[T, H, D]` or
        /// `[B, T, H, D]`), read through its strides.
        fn paged_write_into(
            &mut self,
            operands: &[Operand<'_>],
            cache_shape: &[usize],
            buffer: &mut Buffer,
        ) -> Result<()> {
            let (new, slots) = (
                self.view_if(operands[0], true)?,
                self.packed(operands[1], true)?,
            );
            let (b, s) = match new.shape.len() {
                3 => (1, [0, new.strides[0], new.strides[1], new.strides[2]]),
                _ => (
                    new.shape[0],
                    [
                        new.strides[0],
                        new.strides[1],
                        new.strides[2],
                        new.strides[3],
                    ],
                ),
            };
            let r = new.shape.len();
            let (t, hkv, d) = (new.shape[r - 3], new.shape[r - 2], new.shape[r - 1]);
            if b * t == 0 {
                return Ok(());
            }
            // Slots past `num_blocks · block_size` are skipped by the kernel.
            let as_i32 = |v: usize| {
                i32::try_from(v)
                    .map_err(|_| Error::InvalidArgument("paged_write: too large".into()))
            };
            let params = [
                as_i32(hkv)?,
                as_i32(cache_shape[2])?,
                as_i32(d)?,
                as_i32(cache_shape[0])?,
                as_i32(t)?,
            ];
            as_i32(b * t)?;
            let strides = s.map(|v| v as i64);
            let args = [
                new.arg(),
                slots.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(bytemuck::bytes_of(&params)),
                Arg::bytes(bytemuck::bytes_of(&strides)),
            ];
            let name = format!("nuvec_paged_write_{}", operands[0].0.dtype.name());
            let pipeline = self.pipeline(&name)?;
            let grid = [d, hkv, b * t];
            self.encode_threads(&pipeline, &args, grid, block_dims(grid))?;
            self.counted()
        }

        /// [`crate::fast::FastOp::PagedWrite`] into its cache's own buffer (donated as for
        /// `slice_update`): only the new positions are written.
        pub(crate) fn paged_write_in_place(
            &mut self,
            mut base: Data,
            operands: &[Operand<'_>],
            out: &Output,
        ) -> Result<Data> {
            let buffer = Arc::get_mut(&mut base.buffer).expect("donated data has one handle");
            self.paged_write_into(operands, &out.shape, buffer)?;
            Ok(base)
        }

        /// [`crate::fast::FastOp::NormRope`] into its cache's own buffer (`base`, donated
        /// as for `slice_update`): only the new positions are written.
        pub(crate) fn norm_rope_in_place(
            &mut self,
            mut base: Data,
            op: &crate::fast::imp::FastOp,
            operands: &[Operand<'_>],
            out: &Output,
        ) -> Result<Data> {
            let buffer = Arc::get_mut(&mut base.buffer).expect("donated data has one handle");
            self.norm_rope_into(op, operands, &out.shape, buffer)?;
            Ok(base)
        }

        /// [`crate::fast::FastOp::NormRope`]'s kernel over `([slots,] x, w[, positions]
        /// [, freqs])`, writing rows of a row-contiguous `[B, H, rows, D]` output
        /// (`out_shape`) from row `offset` (a cache) or 0, or of a pool `[blocks, H,
        /// block_size, D]` at their slots: one threadgroup per row, reading `x` through its
        /// strides.
        fn norm_rope_into(
            &mut self,
            op: &crate::fast::imp::FastOp,
            operands: &[Operand<'_>],
            out_shape: &[usize],
            buffer: &mut Buffer,
        ) -> Result<()> {
            let crate::fast::imp::FastOp::NormRope {
                eps,
                dims,
                traditional,
                base,
                scale,
                offset,
                cache,
                positions,
                paged,
            } = *op
            else {
                unreachable!("a norm_rope")
            };
            let at = usize::from(paged);
            let (meta, _) = operands[at];
            let (b, h, l, d) = (meta.shape[0], meta.shape[1], meta.shape[2], meta.shape[3]);
            if b * h * l == 0 {
                return Ok(());
            }
            let x = self.view_if(operands[at], d == 1 || meta.strides[3] == 1)?;
            let w = self.packed(operands[at + 1], true)?;
            let at_freqs = at + 2 + usize::from(positions);
            let freqs = match operands.get(at_freqs) {
                Some(&f) => Some(self.packed(f, true)?),
                None => None,
            };
            let per_row = match positions {
                true => Some(self.packed(operands[at + 2], true)?),
                false => None,
            };
            let slots = match paged {
                true => Some(self.packed(operands[0], true)?),
                false => None,
            };
            let name = format!(
                "nuvec_{}_rope_{}{}",
                if eps.is_some() { "norm" } else { "plain" },
                if freqs.is_some() { "freqs_" } else { "" },
                meta.dtype.name()
            );
            let pipeline =
                self.pipeline_with(&name, &[(22, traditional), (25, positions), (26, paged)])?;
            let rows = out_shape[2];
            let x_strides: [i64; 3] = [
                x.strides[0] as i64,
                x.strides[1] as i64,
                x.strides[2] as i64,
            ];
            let out_strides: [i64; 3] = [(h * rows * d) as i64, (rows * d) as i64, d as i64];
            let sizes: [i32; 4] = [h as i32, l as i32, d as i32, dims as i32];
            let e = f32::from_bits(eps.unwrap_or(0)).to_le_bytes();
            let at_row = (offset as i32).to_le_bytes();
            let write_at = (if cache && !paged { offset as i32 } else { 0 }).to_le_bytes();
            let sc = scale.to_le_bytes();
            let log_base = f32::from_bits(base).log2().to_le_bytes();
            // The pool's positions per block and blocks.
            let pool: [i32; 2] = [rows as i32, out_shape[0] as i32];
            let unused = [0u8; 4];
            let mut args = vec![
                x.arg(),
                w.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&e),
                Arg::bytes(&at_row),
                Arg::bytes(&write_at),
                Arg::bytes(&sc),
                Arg::bytes(bytemuck::bytes_of(&x_strides)),
                Arg::bytes(bytemuck::bytes_of(&out_strides)),
                Arg::bytes(bytemuck::bytes_of(&sizes)),
                Arg::bytes(&log_base),
            ];
            args.push(match &freqs {
                Some(f) => f.arg(),
                None => Arg::bytes(&unused),
            });
            args.push(match &per_row {
                Some(p) => p.arg(),
                None => Arg::bytes(&unused),
            });
            args.push(match &slots {
                Some(s) => s.arg(),
                None => Arg::bytes(&unused),
            });
            args.push(Arg::bytes(bytemuck::bytes_of(&pool)));
            self.dispatch_rows(&pipeline, &args, b * h * l, row_threads(d, 4))
        }

        /// [`crate::fast::FastOp::MoeRoute`]: one threadgroup per row of experts.
        fn moe_route(
            &mut self,
            probs: Operand<'_>,
            (k, normalize): (usize, bool),
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            let p = self.packed(probs, true)?;
            let e = p.shape.last().copied().unwrap_or(0);
            let rows = p.shape.iter().product::<usize>() / e.max(1);
            let mut indices = Buffer::try_alloc_gpu(outputs[0].size * 4)?;
            let mut scores = Buffer::try_alloc_gpu(outputs[1].size * outputs[1].dtype.size())?;
            if rows > 0 {
                let sizes = [e as i32, k as i32, i32::from(normalize)].map(i32::to_le_bytes);
                let args = [
                    p.arg(),
                    Arg::write(indices.metal_mut().expect("device storage")),
                    Arg::write(scores.metal_mut().expect("device storage")),
                    Arg::bytes(&sizes[0]),
                    Arg::bytes(&sizes[1]),
                    Arg::bytes(&sizes[2]),
                ];
                let name = format!("nuvec_moe_route_{}", probs.0.dtype.name());
                self.dispatch_groups(&name, &args, [rows, 1, 1], [e.next_multiple_of(32), 1, 1])?;
            }
            Ok(vec![Data::new(indices), Data::new(scores)])
        }

        /// [`crate::fast::FastOp::ConvSlots`] into its donated pool (`base`): `operands` are
        /// all inputs but the pool (its metadata is `pool_meta`).
        pub(crate) fn conv_slots_in_place(
            &mut self,
            operands: &[Operand<'_>],
            pool_meta: &Output,
            mut base: Data,
            outputs: &[Output],
            silu: bool,
        ) -> Result<Vec<Data>> {
            let buffer = Arc::get_mut(&mut base.buffer).expect("donated data has one handle");
            let empty = Data::new(Buffer::try_alloc(0)?);
            let all = [operands[0], operands[1], (pool_meta, &empty), operands[2]];
            let y = self.conv_slots_into(&all, outputs, buffer, silu)?;
            Ok(vec![y, base])
        }

        /// The convolution at `slots` of `pool` (read and written in place); returns the
        /// outputs. `operands[2]` gives only the pool's metadata.
        fn conv_slots_into(
            &mut self,
            operands: &[Operand<'_>],
            outputs: &[Output],
            pool: &mut Buffer,
            silu: bool,
        ) -> Result<Data> {
            let mut y = Buffer::try_alloc_gpu(outputs[0].size * outputs[0].dtype.size())?;
            let (x, w, slots) = (
                self.packed(operands[0], true)?,
                self.packed(operands[1], true)?,
                self.packed(operands[3], true)?,
            );
            let (b, l, c) = (x.shape[0], x.shape[1], x.shape[2]);
            let (k, s) = (w.shape[1], operands[2].0.shape[0]);
            if b * c == 0 || (s == 0 && k > 1) {
                return Ok(Data::new(y));
            }
            let sizes = [l as i32, c as i32, k as i32, s.max(1) as i32];
            let flag = i32::from(silu).to_le_bytes();
            let args = [
                x.arg(),
                w.arg(),
                Arg::write(pool.metal_mut().expect("device storage")),
                slots.arg(),
                Arg::write(y.metal_mut().expect("device storage")),
                Arg::bytes(bytemuck::bytes_of(&sizes)),
                Arg::bytes(&flag),
            ];
            let name = format!("nuvec_causal_conv1d_slots_{}", operands[0].0.dtype.name());
            let pipeline = self.pipeline(&name)?;
            let grid = [c, b, 1];
            self.encode_threads(&pipeline, &args, grid, block_dims(grid))?;
            self.counted()?;
            Ok(Data::new(y))
        }

        /// [`crate::fast::FastOp::WeightedSum`]: one thread per output element.
        fn weighted_sum(&mut self, operands: &[Operand<'_>], out: &Output) -> Result<Vec<Data>> {
            let (x, w) = (
                self.packed(operands[0], true)?,
                self.packed(operands[1], true)?,
            );
            let n = x.shape.len();
            let (k, d) = (x.shape[n - 2], x.shape[n - 1]);
            let rows = out.size / d.max(1);
            let mut buffer = Buffer::try_alloc_gpu(out.size * out.dtype.size())?;
            if out.size > 0 {
                let sizes = [k as i32, d as i32].map(i32::to_le_bytes);
                let args = [
                    x.arg(),
                    w.arg(),
                    Arg::write(buffer.metal_mut().expect("device storage")),
                    Arg::bytes(&sizes[0]),
                    Arg::bytes(&sizes[1]),
                ];
                let name = format!("nuvec_weighted_sum_{}", out.dtype.name());
                let pipeline = self.pipeline(&name)?;
                let grid = [d, rows, 1];
                self.encode_threads(&pipeline, &args, grid, block_dims(grid))?;
                self.counted()?;
            }
            Ok(vec![Data::new(buffer)])
        }

        fn gated_delta(
            &mut self,
            operands: &[Operand<'_>],
            outputs: &[Output],
            steps: bool,
        ) -> Result<Vec<Data>> {
            let ins = operands
                .iter()
                .map(|&o| self.packed(o, true))
                .collect::<Result<Vec<_>>>()?;
            let mut y = Buffer::try_alloc_gpu(outputs[0].size * outputs[0].dtype.size())?;
            let mut s = Buffer::try_alloc_gpu(outputs[1].size * outputs[1].dtype.size())?;
            if operands[0].0.shape[1] == 0 {
                if !steps {
                    self.copy(operands[5], &mut s)?;
                }
                return Ok(vec![Data::new(y), Data::new(s)]);
            }
            let mode = if steps {
                DeltaMode::Steps
            } else {
                DeltaMode::Final
            };
            let metas: Vec<&Output> = operands.iter().map(|(m, _)| *m).collect();
            self.gated_delta_into(
                &ins[..5],
                Some(&ins[5]),
                &metas,
                None,
                mode,
                (&mut y, &mut s),
            )?;
            Ok(vec![Data::new(y), Data::new(s)])
        }

        /// [`crate::fast::FastOp::GatedDeltaSlots`] into a copy of the state pool.
        fn gated_delta_slots(
            &mut self,
            operands: &[Operand<'_>],
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            let mut pool = Buffer::try_alloc_gpu(outputs[1].size * outputs[1].dtype.size())?;
            self.copy(operands[5], &mut pool)?;
            let y = self.gated_delta_slots_into(operands, outputs, &mut pool)?;
            Ok(vec![y, Data::new(pool)])
        }

        /// [`crate::fast::FastOp::GatedDeltaSlots`] into its donated state pool (`base`):
        /// `operands` are all inputs but the pool (its metadata is `pool_meta`).
        pub(crate) fn gated_delta_slots_in_place(
            &mut self,
            operands: &[Operand<'_>],
            pool_meta: &Output,
            mut base: Data,
            outputs: &[Output],
        ) -> Result<Vec<Data>> {
            let buffer = Arc::get_mut(&mut base.buffer).expect("donated data has one handle");
            let mut all: Vec<Operand<'_>> = operands[..5].to_vec();
            // The pool's metadata for the kernel's types; its data is the output buffer.
            let empty = Data::new(Buffer::try_alloc(0)?);
            all.push((pool_meta, &empty));
            all.push(operands[5]);
            let y = self.gated_delta_slots_into(&all, outputs, buffer)?;
            Ok(vec![y, base])
        }

        /// The gated delta rule over the states at `slots` of `pool` (read and written in
        /// place); returns the outputs. `operands[5]` gives only the pool's metadata.
        fn gated_delta_slots_into(
            &mut self,
            operands: &[Operand<'_>],
            outputs: &[Output],
            pool: &mut Buffer,
        ) -> Result<Data> {
            let mut y = Buffer::try_alloc_gpu(outputs[0].size * outputs[0].dtype.size())?;
            if operands[0].0.shape[1] == 0 {
                return Ok(Data::new(y));
            }
            let ins = operands[..5]
                .iter()
                .map(|&o| self.packed(o, true))
                .collect::<Result<Vec<_>>>()?;
            let slots = self.packed(operands[6], true)?;
            let metas: Vec<&Output> = operands[..6].iter().map(|(m, _)| *m).collect();
            self.gated_delta_into(
                &ins,
                None,
                &metas,
                Some(&slots),
                DeltaMode::Slots,
                (&mut y, pool),
            )?;
            Ok(Data::new(y))
        }

        /// The gated delta kernel (mlx-lm's, generated per head size; transformed for every
        /// step's state or for states at slots of a pool) over `ins` (q, k, v, g, beta),
        /// with the state read from `state` (or, without, from `out.1` in place).
        #[allow(clippy::too_many_arguments)]
        fn gated_delta_into(
            &mut self,
            ins: &[Packed<'_>],
            state: Option<&Packed<'_>>,
            metas: &[&Output],
            slots: Option<&Packed<'_>>,
            mode: DeltaMode,
            (y, s): (&mut Buffer, &mut Buffer),
        ) -> Result<()> {
            let (q, v, g, st) = (metas[0], metas[2], metas[3], metas[5]);
            let (b, t, hk, dk) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
            let (hv, dv) = (v.shape[2], v.shape[3]);
            let packed = dk == 128
                && dv % 8 == 0
                && g.dtype == DType::F32
                && st.dtype == DType::F32
                && gdn_packed_enabled();
            let types: Vec<&str> = metas.iter().map(|m| msl_type(m.dtype)).collect();
            let name = format!(
                "nuvec_gated_delta{}{}_{}_{}_{}_{}_dk{dk}_dv{dv}_hk{hk}_hv{hv}",
                if packed { "_packed" } else { "" },
                match mode {
                    DeltaMode::Final => "",
                    DeltaMode::Steps => "_steps",
                    DeltaMode::Slots => "_slots",
                },
                types[0],
                types[3],
                types[4],
                types[5],
            );
            let device = self
                .device()
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            let source = || {
                let (base, store, state_ptr) = match packed {
                    true => (GATED_DELTA_PACKED, PACKED_STORE, PACKED_STATE),
                    false => (GATED_DELTA, GENERIC_STORE, GENERIC_STATE),
                };
                let body = match mode {
                    DeltaMode::Final => base.to_owned(),
                    DeltaMode::Steps => every_step(base, store, state_ptr),
                    DeltaMode::Slots => at_slots(base),
                };
                let slots_param = match mode {
                    DeltaMode::Slots => "const device int* slots [[buffer(9)]],\n",
                    _ => "",
                };
                format!(
                    "#include <metal_stdlib>\nusing namespace metal;\n\
                     [[kernel]] void {name}(\n\
                     const device {in_t}* q [[buffer(0)]], const device {in_t}* k [[buffer(1)]],\n\
                     const device {in_t}* v [[buffer(2)]], const device {g_t}* g [[buffer(3)]],\n\
                     const device {b_t}* beta [[buffer(4)]], const device {st_t}* state_in [[buffer(5)]],\n\
                     device {in_t}* y [[buffer(6)]], device {st_t}* state_out [[buffer(7)]],\n\
                     constant int& T [[buffer(8)]],\n{slots_param}\
                     uint3 thread_position_in_grid [[thread_position_in_grid]],\n\
                     uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],\n\
                     uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {{\n\
                     typedef {in_t} InT;\ntypedef {st_t} StT;\n\
                     constexpr int Dk = {dk};\nconstexpr int Dv = {dv};\n\
                     constexpr int Hk = {hk};\nconstexpr int Hv = {hv};\n{body}\n}}\n",
                    in_t = types[0],
                    g_t = types[3],
                    b_t = types[4],
                    st_t = types[5],
                )
            };
            let pipeline = device
                .pipeline_from_source(&name, source)
                .map_err(backend)?;
            let steps = (t as i32).to_le_bytes();
            let mut args: Vec<Arg<'_>> = ins.iter().map(Packed::arg).collect();
            // The state input: its own buffer, or (slots) the pool written in place.
            args.push(match state {
                Some(state) => state.arg(),
                None => Arg::Same(7, 0),
            });
            args.extend([
                Arg::write(y.metal_mut().expect("device storage")),
                Arg::write(s.metal_mut().expect("device storage")),
                Arg::bytes(&steps),
            ]);
            if let Some(slots) = slots {
                args.push(slots.arg());
            }
            let (grid, group) = if packed {
                ([32, dv / 8, b * hv], [32, 2, 1])
            } else {
                ([32, dv, b * hv], [32, 4, 1])
            };
            self.encode_threads(&pipeline, &args, grid, group)?;
            self.counted()
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
        /// (one or two passes) or the tiled steel kernel, as the attention plan decided
        /// (`crate::fast::plan`; the encoder makes no routing decisions).
        fn attention(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            flags: AttentionFlags,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let AttentionFlags {
                scale,
                causal,
                has_mask,
                has_sinks,
                window,
                passes,
                blocks,
                gqa_kernel,
                tk1,
            } = flags;
            use nuvec_gpu::Constant;
            let wb = (window as i32).to_le_bytes();
            // The windowed kernels (`_w`).
            let w = if window > 0 { "_w" } else { "" };
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
            if passes > 0 {
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
                        (37, Constant::Bool(window > 0)),
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
                if passes == 1 {
                    let name = format!("nuvec_sdpa_vector_{dtn}_{d}_{dv}{w}");
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
                    if window > 0 {
                        push_mask(&mut args, 18);
                        args.push(Arg::bytes(&wb));
                    }
                    self.encode_groups(&pipeline, &args, [b * hq, lq, 1], [1024, 1, 1])?;
                    return self.counted();
                }
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
                if tk1 {
                    name.push_str("_tk1");
                }
                name.push_str(w);
                let consts = constants(Some(blocks as i32));
                let pipeline = device
                    .pipeline_with_constants(&name, &consts)
                    .map_err(backend)?;
                let rows = b * hq * lq;
                let mut partial = Buffer::try_alloc_gpu(rows * blocks * dv * qm.dtype.size())?;
                let mut sums = Buffer::try_alloc_gpu(rows * blocks * 4)?;
                let mut maxs = Buffer::try_alloc_gpu(rows * blocks * 4)?;
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
                    if window > 0 {
                        push_mask(&mut args, 19);
                        args.push(Arg::bytes(&wb));
                    }
                    self.encode_groups(&pipeline, &args, [hkv, b, blocks], [32, gqa, lq])?;
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
                "nuvec_steel_attention_{dtn}_bq{bq}_bk{bk}_bd{d}_bv{dv}_wm{wm}_wn{wn}_mask{mask_name}{w}"
            );
            let constants = [
                (210, lq % bq == 0),
                (211, lk % bk == 0),
                (310, has_mask),
                (311, causal),
                (312, has_sinks),
                (313, false),
                (314, window > 0),
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
                // The output's layout: `[B, L, H, D]` in memory (MLX's full attention).
                o_strides: [
                    out.strides[0] as i64,
                    out.strides[1] as i64,
                    out.strides[2] as i64,
                ],
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
            if window > 0 {
                while args.len() < 9 {
                    args.push(Arg::bytes(&unused));
                }
                args.push(Arg::bytes(&wb));
            }
            let _ = out;
            self.encode_groups(&pipeline, &args, [nq, hq, b], [32, wm, wn])?;
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
            let alloc = |o: &Output| Buffer::try_alloc_gpu(o.size * o.dtype.size());
            let mut packed = Buffer::try_alloc_gpu(n * bits / 8)?;
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

        /// MLX's `fp_quantize` (mxfp4, mxfp8, nvfp4): one thread per value, a group in
        /// adjacent lanes of a SIMD group (MLX's `get_quantize_kernel_dims`).
        fn fp_quantize(
            &mut self,
            w: Operand<'_>,
            outputs: &[Output],
            mode: crate::QuantMode,
        ) -> Result<Vec<Data>> {
            let name = format!("nuvec_{mode}_quantize_{}", w.0.dtype.name());
            let w = self.packed(w, true)?;
            let n = w.shape.iter().product::<usize>();
            let alloc = |o: &Output| Buffer::try_alloc_gpu(o.size * o.dtype.size());
            let (mut packed, mut scales) = (alloc(&outputs[0])?, alloc(&outputs[1])?);
            let args = [
                w.arg(),
                Arg::write(packed.metal_mut().expect("device storage")),
                Arg::write(scales.metal_mut().expect("device storage")),
            ];
            self.dispatch_linear(&name, &args, n)?;
            Ok(vec![Data::new(packed), Data::new(scales)])
        }

        /// The floating-point modes' products (MLX's `QuantizedMatmul::eval_gpu` for mxfp4,
        /// mxfp8 and nvfp4), except where the row-independence contract governs:
        ///
        /// - Transposed weights: up to `ROW_INDEPENDENT_ROWS` rows, one matrix-vector product
        ///   per row (`qmv_quad` for `K` of 64 or 128, else `qmv_fast` for aligned `K`, else
        ///   `qmv`), each row with MLX's one-row bits; up to MLX's vector limit, MLX's
        ///   `qmv_quad` or `qmv_wide`; from it on, split-K (one batch) or `qmm_t`.
        /// - Untransposed weights: up to `ROW_INDEPENDENT_ROWS` rows, `qvm` (`K` below 1024)
        ///   or `qvm_split_k`, each row with MLX's one-row bits (MLX: below 4 rows); `qmm_n`
        ///   above.
        fn fp_quantized_matmul(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            quant: crate::Quant,
            transpose: bool,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let (group_size, bits) = quant.checked("quantized_matmul")?;
            let x = self.packed(operands[0], false)?;
            let w = self.packed(operands[1], false)?;
            let s = self.packed(operands[2], false)?;
            let nd = x.shape.len();
            let k = x.shape[nd - 1];
            let n = out.shape[out.shape.len() - 1];
            let flat = w.shape.len() == 2 && x.row_contiguous();
            let m = if flat { out.size / n } else { x.shape[nd - 2] };
            let batch = out.size / m / n;
            let batched = batch > 1;
            // Routed by one item's rows (`[B, L, K]`'s L), as the affine path: prompts
            // prefilled together take the kernels each takes alone.
            let item = if flat && nd >= 3 { x.shape[nd - 2] } else { m };
            let info = self
                .device()
                .map(|d| d.info())
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            let (generation, class) = (info.generation.unwrap_or(0), info.class.unwrap_or('g'));
            let rows = crate::ROW_INDEPENDENT_ROWS;
            let mode = quant.mode;
            let dtype = out.dtype.name();
            let ints = |v: usize| (v as i32).to_le_bytes();
            let (kb, nb, mb) = (ints(k), ints(n), ints(m));
            // Batch layout, as MLX's `add_strides_and_shapes` (no biases).
            let batch_info = [
                ints(nd - 2).to_vec(),
                shape_bytes(&x.shape),
                strides_bytes(&x.strides),
                ints(w.shape.len() - 2).to_vec(),
                shape_bytes(&w.shape),
                strides_bytes(&w.strides),
                strides_bytes(&s.strides),
            ];
            let b = u8::from(batched);
            let output = buffer.metal_mut().expect("device storage");
            let quad = k == 128 || k == 64;
            // Row-exact (`with_row_exact`): matrix-vector products up to `rows` rows; else
            // MLX's routing, with `qmv_wide` from 2 rows.
            let exact = crate::row_exact();
            if transpose {
                let limit = if exact {
                    qmv_batch_limit(k, n, generation, class).max(rows + 1)
                } else {
                    qmv_batch_limit(k, n, generation, class)
                };
                if item >= limit {
                    // As the affine path: MLX splits K only for one batch item.
                    if !batched && (exact || item == m) {
                        return self.fp_qmm_splitk(
                            [&w, &s, &x],
                            (m, n, k, item),
                            quant,
                            out,
                            output,
                        );
                    }
                    // The global-scale slot is not read: the scales again.
                    let mut args = vec![w.arg(), s.arg(), s.arg(), x.arg(), Arg::write(output)];
                    args.extend([Arg::bytes(&kb), Arg::bytes(&nb), Arg::bytes(&mb)]);
                    if batched {
                        args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
                    }
                    let al = u8::from(n.is_multiple_of(32));
                    let name = format!("nuvec_{mode}_qmm_t_{dtype}_al{al}_batch{b}");
                    let grid = [n.div_ceil(32), m.div_ceil(32), batch];
                    return self.dispatch_groups(&name, &args, grid, [32, 2, 2]);
                }
                if quad {
                    let mut args = vec![w.arg(), s.arg(), x.arg(), Arg::write(output)];
                    args.extend([Arg::bytes(&kb), Arg::bytes(&nb)]);
                    if batched {
                        args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
                    }
                    let name = format!("nuvec_{mode}_qmv_quad_{dtype}_d{k}_batch{b}");
                    return self.dispatch_groups(
                        &name,
                        &args,
                        [m, n.div_ceil(64), batch],
                        [32, 1, 1],
                    );
                }
                if item > rows || (!exact && m >= 2) {
                    // MLX's `qmv_wide`: up to five vectors per threadgroup, each weight group
                    // decoded once for them.
                    let tiles = m.div_ceil(5);
                    let vecs = m.div_ceil(tiles);
                    let mut args = vec![w.arg(), s.arg(), x.arg(), Arg::write(output)];
                    args.extend([Arg::bytes(&kb), Arg::bytes(&nb), Arg::bytes(&mb)]);
                    if batched {
                        args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
                    }
                    let name = format!("nuvec_{mode}_qmv_wide_{dtype}_nv{vecs}_batch{b}");
                    let grid = [m.div_ceil(vecs), n.div_ceil(4), batch];
                    return self.dispatch_groups(&name, &args, grid, [32, 2, 1]);
                }
                let fast = k % (32 / bits * 2 * 32) == 0;
                if m >= 2 && !batched {
                    // nuvec's few-row kernel: the weights read once for 2 or 4 rows, each
                    // row with the one-row kernel's bits.
                    let per_group = if m >= 4 { 4 } else { 2 };
                    let pr = u8::from(fast && !n.is_multiple_of(8));
                    let args = [
                        w.arg(),
                        s.arg(),
                        x.arg(),
                        Arg::write(output),
                        Arg::bytes(&kb),
                        Arg::bytes(&nb),
                        Arg::bytes(&mb),
                    ];
                    let name = format!(
                        "nuvec_{mode}_qmv_rows_{dtype}_fast{}_pr{pr}_r{per_group}",
                        u8::from(fast)
                    );
                    let grid = [m.div_ceil(per_group), n.div_ceil(8), 1];
                    return self.dispatch_groups(&name, &args, grid, [32, 2, 1]);
                }
                // One matrix-vector product per row: MLX's one-row kernels.
                let narrow = fast
                    && n >= 4096
                    && generation == 17
                    && class == 's'
                    && mode == crate::QuantMode::Nvfp4;
                let per_simdgroup = if narrow { 2 } else { 4 };
                let bn = 2 * per_simdgroup;
                let mut args = vec![w.arg(), s.arg(), s.arg(), x.arg(), Arg::write(output)];
                args.extend([Arg::bytes(&kb), Arg::bytes(&nb)]);
                if batched {
                    args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
                }
                let name = if fast {
                    let pr = u8::from(!n.is_multiple_of(bn));
                    format!("nuvec_{mode}_qmv_fast_{dtype}_batch{b}_pr{pr}_r{per_simdgroup}")
                } else {
                    format!("nuvec_{mode}_qmv_{dtype}_batch{b}")
                };
                let _ = group_size;
                return self.dispatch_groups(&name, &args, [m, n.div_ceil(bn), batch], [32, 2, 1]);
            }
            if m > rows {
                let mut args = vec![w.arg(), s.arg(), x.arg(), Arg::write(output)];
                args.extend([Arg::bytes(&kb), Arg::bytes(&nb), Arg::bytes(&mb)]);
                if batched {
                    args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
                }
                let name = format!("nuvec_{mode}_qmm_n_{dtype}_batch{b}");
                let grid = [n.div_ceil(32), m.div_ceil(32), batch];
                return self.dispatch_groups(&name, &args, grid, [32, 2, 2]);
            }
            if k >= 1024 {
                // MLX's `qvm_split_k` (MLX: M < 4; here every row up to R, so each row has
                // the bits of its one-row product).
                let name = |split| format!("nuvec_{mode}_qvm_split_k_{dtype}_spk{split}");
                let sizes = (m, n, k, batch, group_size);
                return self.qvm_split_k([&w, &s, &x], None, sizes, name, out, output);
            }
            let mut args = vec![w.arg(), s.arg(), s.arg(), x.arg(), Arg::write(output)];
            args.extend([Arg::bytes(&kb), Arg::bytes(&nb)]);
            if batched {
                args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
            }
            let name = format!("nuvec_{mode}_qvm_{dtype}_batch{b}");
            let bn = group_size.min(32) * 2;
            self.dispatch_groups(&name, &args, [m, n.div_ceil(bn), batch], [32, 2, 1])
        }

        /// MLX's `qvm_split_k` (affine with `biases`, or a floating-point mode): `K` in 8
        /// partitions (32 above 8192), each a `qvm` into `[..., split, M, N]` partials of the
        /// output dtype, then MLX's column sum over the partitions in that dtype. `name` is
        /// the kernel's for a split.
        fn qvm_split_k(
            &mut self,
            [w, s, x]: [&Packed<'_>; 3],
            biases: Option<&Packed<'_>>,
            (m, n, k, batch, group_size): (usize, usize, usize, usize, usize),
            name: impl Fn(usize) -> String,
            out: &Output,
            output: &mut nuvec_gpu::Buffer,
        ) -> Result<()> {
            let split = if k > 8192 { 32 } else { 8 };
            let part = k.div_ceil(split);
            let final_block = k - (split - 1) * part;
            // One item: x as `[M, K]` (MLX's reshape); otherwise its batch dimensions.
            let nd = x.shape.len();
            let (mut x_shape, mut x_strides) = if batch == 1 {
                (vec![m, k], vec![x.strides[nd - 2], x.strides[nd - 1]])
            } else {
                (x.shape.clone(), x.strides.clone())
            };
            // The split as one more batch dimension (MLX's reshapes, kept verbatim).
            let xn = x_shape.len();
            x_shape.insert(xn - 2, split);
            *x_shape.last_mut().expect("matrix") /= split;
            x_strides.insert(xn - 2, part as isize);
            x_strides[xn - 1] = part as isize;
            let wn = w.shape.len();
            let (mut w_shape, mut w_strides, mut s_strides) =
                (w.shape.clone(), w.strides.clone(), s.strides.clone());
            w_shape.insert(wn - 2, split);
            w_shape[wn - 1] /= split;
            w_strides.insert(wn - 2, (part * w.shape[wn - 1]) as isize);
            s_strides.insert(wn - 2, (part * s.shape[s.shape.len() - 1]) as isize);
            let b_strides = biases.map(|b| {
                let mut t = b.strides.clone();
                t.insert(wn - 2, (part * b.shape[b.shape.len() - 1]) as isize);
                strides_bytes(&t)
            });
            let mut temp = out.shape[..out.shape.len().saturating_sub(2)].to_vec();
            if batch == 1 {
                temp.clear();
            }
            temp.extend([split, m, n]);
            let dtype = out.dtype;
            let mut partial = Buffer::try_alloc_gpu(temp.iter().product::<usize>() * dtype.size())?;
            let ints = |v: usize| (v as i32).to_le_bytes();
            let (pb, nb, fb) = (ints(part), ints(n), ints(final_block));
            let mut batch_info = vec![
                ints(xn - 1).to_vec(),
                shape_bytes(&x_shape),
                strides_bytes(&x_strides),
                ints(wn - 1).to_vec(),
                shape_bytes(&w_shape),
                strides_bytes(&w_strides),
                strides_bytes(&s_strides),
            ];
            batch_info.extend(b_strides);
            {
                let mut args = vec![w.arg(), s.arg()];
                args.extend(biases.map(Packed::arg));
                args.extend([
                    x.arg(),
                    Arg::write(partial.metal_mut().expect("device storage")),
                    Arg::bytes(&pb),
                    Arg::bytes(&nb),
                ]);
                args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
                args.push(Arg::bytes(&fb));
                let name = name(split);
                let bn = group_size.min(32) * 2;
                let grid = [m, n.div_ceil(bn), batch * split];
                self.dispatch_groups(&name, &args, grid, [32, 2, 1])?;
            }
            self.split_k_sum(dtype, &temp, &partial, output)
        }

        /// MLX's `qmm_splitk` for the floating-point modes: about 512 threadgroups over `K`
        /// partitions (from one item's rows, as the affine path), each partition in float32,
        /// summed and rounded once; `qmm_t` when `K` does not split.
        fn fp_qmm_splitk(
            &mut self,
            [w, s, x]: [&Packed<'_>; 3],
            (m, n, k, item): (usize, usize, usize, usize),
            quant: crate::Quant,
            out: &Output,
            output: &mut nuvec_gpu::Buffer,
        ) -> Result<()> {
            let (group_size, _) = quant.checked("quantized_matmul")?;
            let mode = quant.mode;
            let dtype = out.dtype.name();
            let (n_tiles, m_tiles) = (n.div_ceil(32), m.div_ceil(32));
            let k_align = group_size.max(32);
            let item_tiles = item.div_ceil(32).max(1);
            let mut split = (512 / (n_tiles * item_tiles)).max(1).min(k / k_align);
            while split > 1 && k % (split * k_align) != 0 {
                split -= 1;
            }
            let ints = |v: usize| (v as i32).to_le_bytes();
            let (kb, nb, mb) = (ints(k), ints(n), ints(m));
            let al = u8::from(n % 32 == 0);
            if split <= 1 {
                let args = [
                    w.arg(),
                    s.arg(),
                    s.arg(),
                    x.arg(),
                    Arg::write(output),
                    Arg::bytes(&kb),
                    Arg::bytes(&nb),
                    Arg::bytes(&mb),
                ];
                let name = format!("nuvec_{mode}_qmm_t_{dtype}_al{al}_batch0");
                return self.dispatch_groups(&name, &args, [n_tiles, m_tiles, 1], [32, 2, 2]);
            }
            let mut partial = Buffer::try_alloc_gpu(split * m * n * 4)?;
            let (size, stride) = (ints(k / split), ints(m * n));
            let args = [
                w.arg(),
                s.arg(),
                x.arg(),
                Arg::write(partial.metal_mut().expect("device storage")),
                Arg::bytes(&kb),
                Arg::bytes(&nb),
                Arg::bytes(&mb),
                Arg::bytes(&size),
                Arg::bytes(&stride),
            ];
            let name = format!("nuvec_{mode}_qmm_t_splitk_{dtype}_al{al}");
            self.dispatch_groups(&name, &args, [n_tiles, m_tiles, split], [32, 2, 2])?;
            let parts = ints(split);
            let args = [
                Arg::read(partial.metal().expect("device storage")),
                Arg::write(output),
                Arg::bytes(&parts),
                Arg::bytes(&stride),
                Arg::bytes(&nb),
            ];
            let name = format!("nuvec_gemm_splitk_accum_{dtype}");
            let pipeline = self.pipeline(&name)?;
            self.encode_threads(&pipeline, &args, [n, m, 1], [n.min(32), m.min(32), 1])?;
            self.counted()
        }

        /// MLX's `GatherQMM::eval_gpu` for the floating-point modes: rows sorted by weight
        /// batch with their `offsets` run the grouped `gather_qmm_rhs`; otherwise, with
        /// transposed weights, `gather_qmv(_fast)` per row below MLX's vector limit and
        /// `gather_qmm_t` from it on; with untransposed weights `gather_qvm` per row up to
        /// `ROW_INDEPENDENT_ROWS` rows (MLX tiles from 4) and `gather_qmm_n` above. An nvfp4
        /// global scale (one per weight batch) takes the `_hgs` kernels.
        /// MLX's `gather_mm_offsets` (one thread per group and row of the last axis).
        fn sorted_offsets(
            &mut self,
            indices: Operand<'_>,
            groups: usize,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let indices = self.packed(indices, true)?;
            let m = *indices.shape.last().expect("checked: at least one axis");
            let rows = indices.shape.iter().product::<usize>() / m.max(1);
            if groups == 0 || rows == 0 {
                return Ok(());
            }
            let (mb, gb) = ((m as i32).to_le_bytes(), (groups as i32).to_le_bytes());
            let output = buffer.metal_mut().expect("device storage");
            let args = [
                indices.arg(),
                Arg::write(output),
                Arg::bytes(&mb),
                Arg::bytes(&gb),
            ];
            let pipeline = self.pipeline("nuvec_sorted_offsets")?;
            let group = pipeline.max_threads_per_threadgroup().min(groups);
            self.encode_threads(&pipeline, &args, [groups, rows, 1], [group, 1, 1])?;
            self.counted()
        }

        fn fp_gather_qmm(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            (quant, transpose, global_scale): (crate::Quant, bool, bool),
            buffer: &mut Buffer,
        ) -> Result<()> {
            let (group_size, bits) = quant.checked("gather_qmm")?;
            let mode = quant.mode;
            let dtype = out.dtype.name();
            let hgs = if global_scale { "_hgs" } else { "" };
            // The sorted kernel indexes rows and experts densely (as `gather_qmm`).
            let sorted = operands.len() > 5 + usize::from(global_scale);
            let x = self.packed(operands[0], sorted)?;
            let w = self.packed(operands[1], sorted)?;
            let s = self.packed(operands[2], sorted)?;
            let gs = match global_scale {
                true => Some(self.packed(operands[3], true)?),
                false => None,
            };
            let at = 3 + usize::from(global_scale);
            let (lhs, rhs) = (operands[at], operands[at + 1]);
            let nd = x.shape.len();
            let (m, k) = (x.shape[nd - 2], x.shape[nd - 1]);
            let n = out.shape[out.shape.len() - 1];
            let batch = out.size / m / n;
            let ints = |v: usize| (v as i32).to_le_bytes();
            let output = buffer.metal_mut().expect("device storage");
            // The global-scale slot is read only by the `_hgs` kernels: the scales again
            // otherwise.
            if let Some(&offsets) = operands.get(at + 2) {
                // `x` is `[rows, 1, K]`, sorted by weight batch.
                let offsets = self.packed(offsets, true)?;
                let groups = offsets.shape.iter().product::<usize>();
                let (rows, nb, kb, gb) = (ints(batch), ints(n), ints(k), ints(groups));
                let args = [
                    x.arg(),
                    w.arg(),
                    s.arg(),
                    gs.as_ref().map_or(s.arg(), Packed::arg),
                    offsets.arg(),
                    Arg::write(output),
                    Arg::bytes(&rows),
                    Arg::bytes(&nb),
                    Arg::bytes(&kb),
                    Arg::bytes(&gb),
                ];
                let t = if transpose { "nt" } else { "nn" };
                let name = format!("nuvec_{mode}_gather_qmm_rhs_{dtype}_{t}{hgs}");
                let constants = [(201, n.is_multiple_of(32)), (202, k.is_multiple_of(32))];
                let pipeline = self.pipeline_with(&name, &constants)?;
                // Every expert adds at most one partial row tile.
                let tiles = batch.div_ceil(16) + groups;
                self.encode_groups(&pipeline, &args, [n.div_ceil(32), tiles, 1], [32, 2, 1])?;
                return self.counted();
            }
            let lhs = self.view_if(lhs, true)?;
            let rhs = self.view_if(rhs, true)?;
            let info = self
                .device()
                .map(|d| d.info())
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            let (generation, class) = (info.generation.unwrap_or(0), info.class.unwrap_or('g'));
            let batch_shape = &out.shape[..out.shape.len() - 2];
            // MLX's `add_strides_and_shapes` (no biases) and `add_gather_strides_and_shapes`.
            let layout = [
                ints(nd - 2).to_vec(),
                shape_bytes(&x.shape),
                strides_bytes(&x.strides),
                ints(w.shape.len() - 2).to_vec(),
                shape_bytes(&w.shape),
                strides_bytes(&w.strides),
                strides_bytes(&s.strides),
                ints(batch_shape.len()).to_vec(),
                shape_bytes(batch_shape),
                strides_bytes(&lhs.strides),
                strides_bytes(&rhs.strides),
            ];
            let mut args = vec![
                w.arg(),
                s.arg(),
                gs.as_ref().map_or(s.arg(), Packed::arg),
                x.arg(),
                lhs.arg(),
                rhs.arg(),
                Arg::write(output),
            ];
            let (kb, nb, mb) = (ints(k), ints(n), ints(m));
            let tiled = match transpose {
                true => m >= qmv_batch_limit(k, n, generation, class),
                false => m > crate::ROW_INDEPENDENT_ROWS,
            };
            if tiled {
                args.extend([Arg::bytes(&kb), Arg::bytes(&nb), Arg::bytes(&mb)]);
                args.extend(layout.iter().map(|v| Arg::bytes(v)));
                let name = match transpose {
                    true => {
                        let al = u8::from(n.is_multiple_of(32));
                        format!("nuvec_{mode}_gather_qmm_t_{dtype}_al{al}{hgs}")
                    }
                    false => format!("nuvec_{mode}_gather_qmm_n_{dtype}{hgs}"),
                };
                let grid = [n.div_ceil(32), m.div_ceil(32), batch];
                return self.dispatch_groups(&name, &args, grid, [32, 2, 2]);
            }
            args.extend([Arg::bytes(&kb), Arg::bytes(&nb)]);
            args.extend(layout.iter().map(|v| Arg::bytes(v)));
            if !transpose {
                let name = format!("nuvec_{mode}_gather_qvm_{dtype}{hgs}");
                let bn = group_size.min(32) * 2;
                return self.dispatch_groups(&name, &args, [m, n.div_ceil(bn), batch], [32, 2, 1]);
            }
            let fast = k % (32 / bits * 2 * 32) == 0 && n.is_multiple_of(8);
            let name = match fast {
                true => format!("nuvec_{mode}_gather_qmv_fast_{dtype}{hgs}"),
                false => format!("nuvec_{mode}_gather_qmv_{dtype}{hgs}"),
            };
            self.dispatch_groups(&name, &args, [m, n.div_ceil(8), batch], [32, 2, 1])
        }

        /// MLX's `fp_dequantize`: one thread per packed byte (two E2M1 values or one E4M3).
        fn fp_dequantize(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            mode: crate::QuantMode,
            bits: usize,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let (w, s) = (
                self.packed(operands[0], true)?,
                self.packed(operands[1], true)?,
            );
            let name = format!("nuvec_{mode}_dequantize_{}", out.dtype.name());
            let args = [
                w.arg(),
                s.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            self.dispatch_linear(&name, &args, out.size / (8 / bits))
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
            // Row-exact (`with_row_exact`): matrix-vector products up to
            // `ROW_INDEPENDENT_ROWS` rows; else MLX's limits (4 untransposed), and
            // `qmv_wide` from 2 rows on GPUs from generation 15.
            let exact = crate::row_exact();
            let rows = crate::ROW_INDEPENDENT_ROWS + 1;
            let limit = match (transpose, exact) {
                (true, true) => qmv_batch_limit(k, n, generation, class).max(rows),
                (true, false) => qmv_batch_limit(k, n, generation, class),
                (false, true) => rows,
                (false, false) => 4,
            };
            let dtype = out.dtype.name();
            let base = format!("{dtype}_gs{group_size}_b{bits}");
            let batched = batch > 1;
            // Routed by one item's rows (`[B, L, K]`'s L), not all of them: prompts prefilled
            // together then take the kernel each takes alone, so each keeps its bits (with
            // all rows, two prompts below the matrix-vector limit could cross it together).
            let item = if flat && nd >= 3 { x.shape[nd - 2] } else { m };
            // MLX splits K only for one batch item (`B == 1`); row-exact, every item's rows
            // are partitioned as alone (`item`), so prompts prefilled together keep their
            // solo bits.
            let split = !batched && (exact || item == m);
            if item >= limit && transpose && split {
                // The K partitions are chosen from one item's rows (`x`'s second-to-last
                // dimension), not from all of them: each item of `[B, L, K]` then gets the
                // bits of its product alone (`B = 1`, as MLX chooses).
                return self.qmm_splitk(
                    [&w, &s, &b, &x],
                    (m, n, k, item),
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
            if !transpose && item < limit && k >= 1024 {
                // MLX's `qvm_split_k` (MLX: M < 4; here every row up to R, so each row has
                // the bits of its one-row product).
                let name = |split| format!("nuvec_affine_qvm_split_k_{base}_spk{split}");
                let sizes = (m, n, k, batch, group_size);
                return self.qvm_split_k([&w, &s, &x], Some(&b), sizes, name, out, output);
            }
            let mut args = vec![w.arg(), s.arg(), b.arg(), x.arg(), Arg::write(output)];
            let (kb, nb, mb) = (ints(k), ints(n), ints(m));
            if item >= limit {
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
            if transpose && !exact && m >= 2 && generation >= 15 {
                // MLX's `qmv_wide`: up to five vectors per threadgroup, each weight group
                // decoded once for them.
                let tiles = m.div_ceil(5);
                let vecs = m.div_ceil(tiles);
                args.push(Arg::bytes(&mb));
                if batched {
                    args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
                }
                let name = format!(
                    "nuvec_affine_qmv_wide_{base}_nv{vecs}_batch{}",
                    u8::from(batched)
                );
                let grid = [m.div_ceil(vecs), n.div_ceil(8), batch];
                return self.dispatch_groups(&name, &args, grid, [32, 2, 1]);
            }
            let fast = k % (pack_factor(bits, 32) * if bits == 2 { 1 } else { 2 } * 32) == 0;
            if transpose && fast && !batched && m >= 4 {
                // nuvec's few-row kernel: weights loaded once for two rows.
                args.push(Arg::bytes(&mb));
                let name = format!(
                    "nuvec_affine_qmv_rows_{base}_pr{}_r2",
                    u8::from(!n.is_multiple_of(8))
                );
                return self.dispatch_groups(
                    &name,
                    &args,
                    [m.div_ceil(2), n.div_ceil(8), 1],
                    [32, 2, 1],
                );
            }
            if batched {
                args.extend(batch_info.iter().map(|v| Arg::bytes(v)));
            }
            let (name, per_group) = if transpose {
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

        /// MLX's gather kernels (`GatherQMM::eval_gpu`, transposed weights): rows sorted by
        /// weight batch with their `offsets` run the grouped `gather_qmm_rhs` (each batch's
        /// weights read once per column tile); otherwise `gather_qmv(_fast)` below the
        /// matrix-vector limit and `gather_qmm_t` above it, one output batch per grid layer.
        fn gather_qmm(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            (group_size, bits): (usize, u8),
            buffer: &mut Buffer,
        ) -> Result<()> {
            // The sorted kernels index rows and experts densely: dense operands for them
            // (copied only when not already row-contiguous).
            let sorted = operands.len() > 6;
            let x = self.packed(operands[0], sorted)?;
            let w = self.packed(operands[1], sorted)?;
            let s = self.packed(operands[2], sorted)?;
            let b = self.packed(operands[3], sorted)?;
            let lhs = self.view_if(operands[4], true)?;
            let rhs = self.view_if(operands[5], true)?;
            let nd = x.shape.len();
            let (m, k) = (x.shape[nd - 2], x.shape[nd - 1]);
            let n = out.shape[out.shape.len() - 1];
            let batch = out.size / m / n;
            let base = format!("{}_gs{group_size}_b{bits}", out.dtype.name());
            let ints = |v: usize| (v as i32).to_le_bytes();
            if let Some(&offsets) = operands.get(6) {
                // `x` is `[rows, 1, K]`, sorted by weight batch.
                let offsets = self.packed(offsets, true)?;
                let groups = operands[6].0.size;
                let (rows, nb, kb, gb) = (ints(batch), ints(n), ints(k), ints(groups));
                let args = [
                    x.arg(),
                    w.arg(),
                    s.arg(),
                    b.arg(),
                    offsets.arg(),
                    Arg::write(buffer.metal_mut().expect("device storage")),
                    Arg::bytes(&rows),
                    Arg::bytes(&nb),
                    Arg::bytes(&kb),
                    Arg::bytes(&gb),
                ];
                // On the matrix units (NAX) when active: 64 × 64 tiles within each
                // expert's rows, for the bit widths whose 32 values fill whole words.
                if batch >= 64
                    && n >= 64
                    && k.is_multiple_of(64)
                    && matches!(bits, 2 | 4 | 8)
                    && self.nax_gemm(out.dtype)
                {
                    // A NAX failure turns NAX off: the SIMD kernel below then.
                    let name = format!("nuvec_nax_gather_qmm_rhs_{base}");
                    if let Ok(pipeline) = self.pipeline_with(&name, &[]) {
                        // At most `batch` experts have rows (most of a large model's none).
                        let tiles = batch.div_ceil(64) + groups.min(batch);
                        let grid = [n.div_ceil(64), tiles, 1];
                        self.encode_groups(&pipeline, &args, grid, [128, 1, 1])?;
                        return self.counted();
                    }
                }
                let name = format!("nuvec_affine_gather_qmm_rhs_{base}_nt");
                let constants = [(201, n.is_multiple_of(32)), (202, k.is_multiple_of(32))];
                let pipeline = self.pipeline_with(&name, &constants)?;
                // Every expert adds at most one partial row tile.
                let tiles = batch.div_ceil(16) + groups;
                self.encode_groups(&pipeline, &args, [n.div_ceil(32), tiles, 1], [32, 2, 1])?;
                return self.counted();
            }
            let info = self
                .device()
                .map(|d| d.info())
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            let (generation, class) = (info.generation.unwrap_or(0), info.class.unwrap_or('g'));
            let limit = qmv_batch_limit(k, n, generation, class);
            let batch_shape = &out.shape[..out.shape.len() - 2];
            let layout = [
                ints(nd - 2).to_vec(),
                shape_bytes(&x.shape),
                strides_bytes(&x.strides),
                ints(w.shape.len() - 2).to_vec(),
                shape_bytes(&w.shape),
                strides_bytes(&w.strides),
                strides_bytes(&s.strides),
                strides_bytes(&b.strides),
                ints(batch_shape.len()).to_vec(),
                shape_bytes(batch_shape),
                strides_bytes(&lhs.strides),
                strides_bytes(&rhs.strides),
            ];
            let mut args = vec![
                w.arg(),
                s.arg(),
                b.arg(),
                x.arg(),
                lhs.arg(),
                rhs.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
            ];
            let (kb, nb, mb) = (ints(k), ints(n), ints(m));
            if m >= limit {
                args.extend([Arg::bytes(&kb), Arg::bytes(&nb), Arg::bytes(&mb)]);
                args.extend(layout.iter().map(|v| Arg::bytes(v)));
                let name = format!(
                    "nuvec_affine_gather_qmm_t_{base}_al{}",
                    u8::from(n.is_multiple_of(32))
                );
                return self.dispatch_groups(
                    &name,
                    &args,
                    [n.div_ceil(32), m.div_ceil(32), batch],
                    [32, 2, 2],
                );
            }
            args.extend([Arg::bytes(&kb), Arg::bytes(&nb)]);
            args.extend(layout.iter().map(|v| Arg::bytes(v)));
            let bits = usize::from(bits);
            let fast = k % (pack_factor(bits, 32) * if bits == 2 { 1 } else { 2 } * 32) == 0
                && n.is_multiple_of(8);
            let name = match fast {
                true => format!("nuvec_affine_gather_qmv_fast_{base}"),
                false => format!("nuvec_affine_gather_qmv_{base}"),
            };
            self.dispatch_groups(&name, &args, [m, n.div_ceil(8), batch], [32, 2, 1])
        }

        /// MLX's `GatherMM::eval_gpu` (without the matrix-vector kernels): rows sorted by
        /// weight batch take `gather_mm_rhs`, which walks each batch of `b` once; anything
        /// else `gather_mm`, one output batch per threadgroup slice.
        fn gather_mm(
            &mut self,
            operands: &[Operand<'_>],
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            if out.size == 0 {
                return Ok(());
            }
            let nd = operands[0].0.shape.len();
            let (m, k) = (operands[0].0.shape[nd - 2], operands[0].0.shape[nd - 1]);
            let n = out.shape[out.shape.len() - 1];
            let batch = out.size / m / n;
            // The kernels read `a` row-major only.
            let mut ga = self.gemmand(operands[0], m, k)?;
            if ga.transposed {
                ga = self.gemmand_copy(operands[0])?;
            }
            let mut gb = self.gemmand(operands[1], k, n)?;
            let name = |kind: &str, gb: &Gemmand<'_>| {
                let tb = if gb.transposed { 't' } else { 'n' };
                format!("nuvec_gather_mm{kind}_{}_n{tb}", out.dtype.name())
            };
            let first = |strides: &[isize]| strides.first().map_or(0, |&s| s as i64);
            let ints = |v: usize| (v as i32).to_le_bytes();
            if m == 1 && batch <= crate::quantized::ROW_TOKENS {
                // Few gathered rows: one matrix-vector product each (MLX's `gather_mv`),
                // whose rows are independent.
                return self.gather_mv(operands, &ga, &gb, (n, k, batch), out, buffer);
            }
            if let Some(&offsets) = operands.get(4) {
                // `a` is `[rows, 1, K]`, sorted by weight batch: one stride walks the batches.
                if gb.batch_strides.len() != 1 {
                    gb = self.gemmand_copy(operands[1])?;
                }
                let groups = offsets.0.size;
                let offsets = self.packed(offsets, true)?;
                let params = GemmParams {
                    m: batch as i32,
                    n: n as i32,
                    k: k as i32,
                    lda: first(&ga.batch_strides) as i32,
                    ldb: gb.ld,
                    ldd: n as i32,
                    tiles_n: n.div_ceil(64) as i32,
                    tiles_m: batch.div_ceil(16) as i32,
                    batch_stride_a: 0,
                    batch_stride_b: gb.batch_strides.last().map_or(0, |&s| s as i64),
                    batch_stride_d: 0,
                    swizzle_log: 0,
                    gemm_k_iterations_aligned: (k / 16) as i32,
                    batch_ndim: 0,
                    _pad: 0,
                };
                let constants = [
                    (200, batch.is_multiple_of(16)),
                    (201, n.is_multiple_of(64)),
                    (202, k.is_multiple_of(16)),
                ];
                let pipeline = self.pipeline_with(&name("_rhs", &gb), &constants)?;
                let groups_b = ints(groups);
                let args = [
                    ga.arg(),
                    gb.arg(),
                    offsets.arg(),
                    Arg::write(buffer.metal_mut().expect("device storage")),
                    Arg::bytes(bytemuck::bytes_of(&params)),
                    Arg::bytes(&groups_b),
                ];
                // Every batch of `b` adds at most one partial row tile.
                let tiles = batch.div_ceil(16) + groups;
                self.encode_groups(&pipeline, &args, [n.div_ceil(64), tiles, 1], [32, 2, 1])?;
                return self.counted();
            }
            let lhs = self.view_if(operands[2], true)?;
            let rhs = self.view_if(operands[3], true)?;
            let batch_shape = &out.shape[..out.shape.len() - 2];
            let has_batch = batch_shape.len() > 1;
            let params = GemmParams {
                m: m as i32,
                n: n as i32,
                k: k as i32,
                lda: ga.ld,
                ldb: gb.ld,
                ldd: n as i32,
                tiles_n: n.div_ceil(64) as i32,
                tiles_m: m.div_ceil(64) as i32,
                batch_stride_a: first(&lhs.strides),
                batch_stride_b: first(&rhs.strides),
                batch_stride_d: (m * n) as i64,
                swizzle_log: 0,
                gemm_k_iterations_aligned: (k / 16) as i32,
                batch_ndim: batch_shape.len() as i32,
                _pad: 0,
            };
            let constants = [
                (10, has_batch),
                (200, m.is_multiple_of(64)),
                (201, n.is_multiple_of(64)),
                (202, k.is_multiple_of(16)),
            ];
            let pipeline = self.pipeline_with(&name("", &gb), &constants)?;
            let (a_shape, b_shape) = (&operands[0].0.shape, &operands[1].0.shape);
            let layout = [
                shape_bytes(batch_shape),
                strides_bytes(&lhs.strides),
                strides_bytes(&rhs.strides),
                ints(nd - 2).to_vec(),
                shape_bytes(&a_shape[..nd - 2]),
                strides_bytes(&ga.batch_strides),
                ints(b_shape.len() - 2).to_vec(),
                shape_bytes(&b_shape[..b_shape.len() - 2]),
                strides_bytes(&gb.batch_strides),
            ];
            let mut args = vec![
                ga.arg(),
                gb.arg(),
                lhs.arg(),
                rhs.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(bytemuck::bytes_of(&params)),
            ];
            args.extend(layout.iter().map(|v| Arg::bytes(v)));
            self.encode_groups(
                &pipeline,
                &args,
                [n.div_ceil(64), m.div_ceil(64), batch],
                [32, 2, 2],
            )?;
            self.counted()
        }

        /// `gather_mm` for one row per output batch: MLX's `gather_mv` (the matrix-vector
        /// kernels with gathered operands), each batch the one-row product.
        fn gather_mv(
            &mut self,
            operands: &[Operand<'_>],
            ga: &Gemmand<'_>,
            gb: &Gemmand<'_>,
            (n, k, batch): (usize, usize, usize),
            out: &Output,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let lhs = self.view_if(operands[2], true)?;
            let rhs = self.view_if(operands[3], true)?;
            let transpose_mat = !gb.transposed;
            let GemvConfig {
                kind,
                bm,
                bn,
                sm,
                sn,
                tm,
                tn,
                per_group,
            } = gemv_config(transpose_mat, k, n);
            let name = format!(
                "nuvec_{kind}_gather_{}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}",
                out.dtype.name()
            );
            let ints = |v: usize| (v as i32).to_le_bytes();
            let (a_shape, b_shape) = (&operands[0].0.shape, &operands[1].0.shape);
            let batch_shape: Vec<usize> = match &out.shape[..out.shape.len() - 2] {
                [] => vec![1],
                s => s.to_vec(),
            };
            let nonempty = |s: &[usize]| if s.is_empty() { vec![1] } else { s.to_vec() };
            let nonempty_strides = |s: &[isize]| if s.is_empty() { vec![0] } else { s.to_vec() };
            let (va, vb) = (&a_shape[..a_shape.len() - 2], &b_shape[..b_shape.len() - 2]);
            let scalars = [ints(k), ints(n), gb.ld.to_le_bytes()];
            let layout = [
                ints(batch_shape.len()).to_vec(),
                shape_bytes(&batch_shape),
                strides_bytes(&nonempty_strides(&lhs.strides)),
                strides_bytes(&nonempty_strides(&rhs.strides)),
                ints(va.len().max(1)).to_vec(),
                shape_bytes(&nonempty(va)),
                strides_bytes(&nonempty_strides(&ga.batch_strides)),
                ints(vb.len().max(1)).to_vec(),
                shape_bytes(&nonempty(vb)),
                strides_bytes(&nonempty_strides(&gb.batch_strides)),
            ];
            let (alpha, beta) = (1.0f32.to_le_bytes(), 0.0f32.to_le_bytes());
            let mut args = vec![
                gb.arg(),
                ga.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&scalars[0]),
                Arg::bytes(&scalars[1]),
                Arg::bytes(&scalars[2]),
            ];
            args.extend(layout.iter().map(|v| Arg::bytes(v)));
            args.extend([lhs.arg(), rhs.arg(), Arg::bytes(&alpha), Arg::bytes(&beta)]);
            let pipeline = self.pipeline_with(&name, &[])?;
            self.encode_groups(
                &pipeline,
                &args,
                [n.div_ceil(per_group), 1, batch],
                [32, bn, bm],
            )?;
            self.counted()
        }

        /// MLX's `qmm_splitk`: K split into partitions of whole groups, partial products in
        /// float32, summed by the GEMM's split-K accumulation. One partition runs `qmm_t`.
        /// The partitions are counted for `item` rows (MLX's rule for `m = item`), so the
        /// bits of a row do not depend on how many items share the product.
        #[allow(clippy::too_many_arguments)]
        fn qmm_splitk(
            &mut self,
            [w, s, b, x]: [&Packed<'_>; 4],
            (m, n, k, item): (usize, usize, usize, usize),
            base: &str,
            out: &Output,
            buffer: &mut Buffer,
            group_size: usize,
        ) -> Result<()> {
            let (n_tiles, m_tiles) = (n.div_ceil(32), m.div_ceil(32));
            let k_align = group_size.max(32);
            let item_tiles = item.div_ceil(32).max(1);
            let mut split = (512 / (n_tiles * item_tiles)).max(1).min(k / k_align);
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
                // On the GPU's matrix units when active (MLX's `qmm` NAX case: K a multiple
                // of 64; split K stays on the SIMD kernels), for bits 2, 4 and 8 and both
                // sides of at least one 64 × 64 tile.
                let bits = base
                    .rsplit("_b")
                    .next()
                    .and_then(|b| b.parse::<usize>().ok());
                let nax_bits = matches!(bits, Some(2 | 4 | 8));
                if nax_bits
                    && k % 64 == 0
                    && m.min(n) >= 64
                    && matches!(group_size, 32 | 64 | 128)
                    && self.nax_gemm(out.dtype)
                {
                    // A NAX failure turns NAX off: the SIMD kernel below then.
                    let name = format!("nuvec_nax_qmm_t_{base}");
                    if let Ok(pipeline) = self.pipeline_with(&name, &[(10, false)]) {
                        let grid = [n.div_ceil(64), m.div_ceil(64), 1];
                        self.encode_groups(&pipeline, &args, grid, [128, 1, 1])?;
                        return self.counted();
                    }
                }
                let name = format!("nuvec_affine_qmm_t_{base}_al{aligned}_batch0");
                return self.dispatch_groups(&name, &args, [n_tiles, m_tiles, 1], [32, 2, 2]);
            }
            let mut partial = Buffer::try_alloc_gpu(split * m * n * 4)?;
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
            self.encode_threads(&pipeline, &args, [n, m, 1], [n.min(32), m.min(32), 1])?;
            self.counted()
        }

        /// A contiguous copy of an operand (encoded now), row-major.
        fn gemmand_copy<'a>(&mut self, operand: Operand<'_>) -> Result<Gemmand<'a>> {
            let (meta, _) = operand;
            let mut copy = Buffer::try_alloc_gpu(meta.size * meta.dtype.size())?;
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
            let mut copy = Buffer::try_alloc_gpu(meta.size * meta.dtype.size())?;
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

        // Every dispatch of core's kernels goes through the three helpers below: the only
        // `unsafe` dispatch calls in core.

        /// Encodes `pipeline` over `threads` threads (a 1-D grid).
        #[allow(unsafe_code)]
        fn encode_dispatch(
            &mut self,
            pipeline: &nuvec_gpu::Pipeline,
            args: &[Arg<'_>],
            threads: usize,
        ) -> Result<()> {
            let cmd = self.command_buffer()?;
            super::DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: core's 1-D kernels (elementwise, copies, gathers and scatters, fused
            // elementwise source) guard every access with the element count bound next to
            // the buffers (`if (i >= n) return`, or a grid of exactly that many threads),
            // and the buffers are allocated for, or are validated views of, the shapes
            // those counts and strides come from.
            unsafe { cmd.dispatch(pipeline, args, threads) }.map_err(backend)
        }

        /// Encodes `pipeline` over `groups` threadgroups of `threads` threads.
        #[allow(unsafe_code)]
        fn encode_groups(
            &mut self,
            pipeline: &nuvec_gpu::Pipeline,
            args: &[Arg<'_>],
            groups: [usize; 3],
            threads: [usize; 3],
        ) -> Result<()> {
            let cmd = self.command_buffer()?;
            super::DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: core's threadgroup kernels (reductions, matrix products, attention,
            // quantized products) index rows and tiles by threadgroup position within the
            // grid sized from the operands' shapes, and guard partial tiles and rows with
            // the sizes bound next to the buffers; the buffers hold those shapes.
            unsafe { cmd.dispatch_groups(pipeline, args, groups, threads) }.map_err(backend)
        }

        /// Encodes `pipeline` over exactly `grid` threads in threadgroups of `group`.
        #[allow(unsafe_code)]
        fn encode_threads(
            &mut self,
            pipeline: &nuvec_gpu::Pipeline,
            args: &[Arg<'_>],
            grid: [usize; 3],
            group: [usize; 3],
        ) -> Result<()> {
            let cmd = self.command_buffer()?;
            super::DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // SAFETY: core's grid kernels (random bits, strided copies, reductions,
            // attention, user `MetalKernel`s, whose source is trusted code by contract)
            // launch exactly `grid` threads, sized from the operands' shapes, and index
            // the buffers by grid position within those shapes (checking partial rows
            // against the sizes bound next to the buffers).
            unsafe { cmd.dispatch_threads(pipeline, args, grid, group) }.map_err(backend)
        }

        fn pipeline(&mut self, name: &str) -> Result<nuvec_gpu::Pipeline> {
            let device = self
                .device()
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            device.pipeline(name).map_err(backend)
        }

        /// `bytes_per_key` random bytes per key (MLX's layout, as `kernels::random_bits`):
        /// one thread per threefry counter.
        fn random_bits(
            &mut self,
            keys: Operand<'_>,
            bytes_per_key: usize,
            buffer: &mut Buffer,
        ) -> Result<()> {
            let keys = self.packed(keys, true)?;
            let num_keys = keys.shape.iter().product::<usize>() / 2;
            let words = bytes_per_key.div_ceil(4);
            if num_keys == 0 || words == 0 {
                return Ok(());
            }
            let n = u32::try_from(bytes_per_key)
                .map_err(|_| Error::InvalidArgument("random bits: too many per key".into()))?
                .to_le_bytes();
            let args = [
                keys.arg(),
                Arg::write(buffer.metal_mut().expect("device storage")),
                Arg::bytes(&n),
            ];
            let grid = [words / 2 + words % 2, num_keys, 1];
            let pipeline = self.pipeline("nuvec_random_bits")?;
            self.encode_threads(&pipeline, &args, grid, block_dims(grid))?;
            self.counted()
        }

        fn dispatch(&mut self, name: &str, args: &[Arg<'_>], threads: usize) -> Result<()> {
            let pipeline = self.pipeline(name)?;
            self.encode_dispatch(&pipeline, args, threads)?;
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
            self.encode_groups(&pipeline, args, groups, threads)?;
            self.counted()
        }

        /// Commits once a command buffer holds enough work. If the GPU is idle when an
        /// evaluation starts, its first command buffer holds a few dispatches, so the GPU
        /// starts while the rest is encoded, and each next one twice as many while the GPU
        /// is still busy (up to [`MAX_DISPATCHES`]), so it is ready before the GPU runs out
        /// of work; a few again whenever the GPU is idle. If earlier work keeps the GPU
        /// busy, buffers are full-size from the start, so they stay few. One that made new
        /// buffers commits at [`FRESH_DISPATCHES`].
        fn counted(&mut self) -> Result<()> {
            self.dispatches += 1;
            if self.large {
                // A large intermediate whose consumers are all encoded returns to the
                // allocator now (as MLX frees it), not when the command buffer is committed.
                self.chunk.retain(|(node, _)| Arc::strong_count(node) > 1);
                self.index.retain(|_, (node, _)| node.strong_count() > 0);
                self.large = self
                    .chunk
                    .iter()
                    .any(|(_, out)| out.iter().any(|d| d.buffer.len() >= LARGE));
            }
            let busy = self.device().is_some_and(|d| d.in_flight() > 0);
            // Profiling times each dispatch alone.
            let profiling = nuvec_gpu::profile::enabled();
            let limit = *self.limit.get_or_insert(if profiling {
                1
            } else if busy {
                MAX_DISPATCHES
            } else {
                IDLE_DISPATCHES
            });
            let fresh = self.dispatches >= FRESH_DISPATCHES
                && self
                    .device()
                    .is_some_and(|d| d.system_allocations() > self.made);
            if (self.dispatches >= limit || fresh) && !self.hold {
                self.commit();
                self.limit = Some(if profiling {
                    1
                } else if busy {
                    (limit * 2).min(MAX_DISPATCHES)
                } else {
                    IDLE_DISPATCHES
                });
            }
            Ok(())
        }

        /// Submits the open command buffer, if any, without waiting; its outputs are
        /// published when it completes. Blocks while too many command buffers are in flight.
        pub(crate) fn commit(&mut self) {
            self.dispatches = 0;
            self.made = self.device().map_or(0, |d| d.system_allocations());
            self.release_unreferenced();
            let Some(cmd) = self.cmd.take() else {
                // Outputs deferred without GPU work (views of pending data) still need a
                // chunk, published after the work they view.
                if self.chunk.is_empty() {
                    return;
                }
                // No GPU work orders these views after their producers: wait for them.
                let reads = std::mem::take(&mut self.reads);
                let collectives = std::mem::take(&mut self.collectives);
                let result = std::mem::take(&mut self.foreign)
                    .iter()
                    .try_for_each(|a| a.out().slot.wait().clone().map(|_| ()));
                let committed = std::mem::take(&mut self.chunk);
                let (k, status) = self.reserve(&committed);
                expose(&committed, &status);
                self.shared.exposed(k);
                self.shared
                    .complete(k, chunk_result(result, &reads, &collectives));
                return;
            };
            // This command buffer runs after every foreign producer (device timeline).
            self.foreign.clear();
            let reads = std::mem::take(&mut self.reads);
            let collectives = std::mem::take(&mut self.collectives);
            let signal = self.signal.take();
            if let Some(device) = self.device {
                device.wait_for_capacity(MAX_IN_FLIGHT);
            }
            // Held until exposed: a completion handler that runs first publishes from it.
            let committed = std::mem::take(&mut self.chunk);
            let (k, status) = self.reserve(&committed);
            let shared = self.shared.clone();
            #[cfg(test)]
            let injected = fault::take();
            #[cfg(test)]
            super::COMMITS.set(super::COMMITS.get() + 1);
            let pending = cmd.commit_then(move |result| {
                #[cfg(test)]
                let result = match injected {
                    true => Err(nuvec_gpu::GpuError::CommandBuffer(
                        "injected failure".into(),
                    )),
                    false => result,
                };
                if let Some(signal) = signal {
                    signal
                        .done
                        .store(true, std::sync::atomic::Ordering::Release);
                }
                shared.complete(
                    k,
                    chunk_result(result.map_err(backend), &reads, &collectives),
                );
            });
            // Profiling times each command buffer alone: nothing may overlap it.
            if nuvec_gpu::profile::enabled() {
                pending.wait();
            }
            self.submitted.push(pending);
            // Only now, after the commit, may other evaluations encode work that reads it.
            expose(&committed, &status);
            self.shared.exposed(k);
        }

        /// Registers `chunk` (the open chunk) as committed chunk `k`, with its failure cell.
        fn reserve(&mut self, chunk: &Chunk) -> (usize, ChunkStatus) {
            let mut p = self
                .shared
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            let status = ChunkStatus::default();
            p.chunks.push(Some(weak(chunk)));
            p.results.push(None);
            p.exposed.push(false);
            p.statuses.push(status.clone());
            (p.chunks.len() - 1, status)
        }

        /// Records that the open chunk reads other evaluations' in-flight `outputs`.
        pub(crate) fn depend_on(&mut self, outputs: &[&crate::Array]) {
            self.foreign.extend(outputs.iter().map(|&a| a.clone()));
        }

        /// Records that the open chunk reads `input`'s in-flight data, produced by another
        /// evaluation: it fails if that work fails.
        pub(crate) fn read_in_flight(&mut self, input: &crate::Array) {
            if let Some(status) = input.out().inflight.chunk()
                && !self.reads.iter().any(|s| Arc::ptr_eq(s, &status))
            {
                self.reads.push(status);
            }
        }

        /// Runs a collective without waiting for the GPU or committing (MLX's
        /// communication stream). In the open command buffer, its input is copied on the GPU
        /// (into the output for `Reduce` and `Send`, which then run in place; into a staging
        /// buffer for `Gather` and `SumScatter`), or, `donated` (a `Reduce` or `Send` input
        /// nothing else holds, as MLX's), becomes the output after a dispatch that orders it
        /// after its producers; either is ordered after every producer by the buffers'
        /// fences, and an event is signalled. The group's worker waits for that
        /// signal, runs the collective and signals back; the work encoded after it waits
        /// for that on the GPU. The output is deferred with the open chunk, so it is
        /// published (or failed with the collective's error) once that work completes.
        pub(crate) fn collective(
            &mut self,
            op: &crate::distributed::CollectiveOp,
            group: usize,
            inputs: &[crate::Array],
            data: Vec<Data>,
            output: &Output,
            donated: Option<Data>,
        ) -> Result<Vec<Data>> {
            use crate::distributed::{CollectiveOp, Input, Ready};
            let device = self
                .device()
                .ok_or(Error::UnsupportedDevice(Device::gpu(0)))?;
            let (dtype, shape) = inputs.first().map_or((output.dtype, Vec::new()), |a| {
                (a.dtype(), a.out().shape.to_vec())
            });
            if let Some(mut out) = donated {
                self.hold = true;
                let ordered = self.order_write(&mut out);
                self.hold = false;
                ordered?;
                let ready = self.input_signal(device)?;
                return self.enqueue_collective(
                    op,
                    group,
                    device,
                    dtype,
                    shape,
                    ready,
                    Input::InOutput,
                    out,
                );
            }
            let mut out = Buffer::try_alloc(output.size * output.dtype.size())?;
            let operand = inputs.first().zip(data.first()).map(|(a, d)| (a.out(), d));
            let on_gpu =
                operand.is_none_or(|o| o.0.size > 0 && self.supports(&Primitive::Copy, &[o], &[]));
            let (ready, input) = if on_gpu {
                let input = match (op, operand) {
                    (_, None) => Input::InOutput,
                    (CollectiveOp::Reduce(_) | CollectiveOp::Send { .. }, Some(o)) => {
                        self.hold = true;
                        let copied = self.copy(o, &mut out);
                        self.hold = false;
                        copied?;
                        Input::InOutput
                    }
                    (_, Some(o)) => {
                        let mut staged = Buffer::try_alloc(o.0.size * o.0.dtype.size())?;
                        self.hold = true;
                        let copied = self.copy(o, &mut staged);
                        self.hold = false;
                        copied?;
                        Input::Staged(Data::new(staged))
                    }
                };
                let ready = match operand {
                    Some(_) => self.input_signal(device)?,
                    None => Ready::Now,
                };
                (ready, input)
            } else {
                // Not copyable on the GPU (host memory, very large): wait for the
                // committed work, then read the operand as it is.
                self.commit();
                let pending = self.submitted.last().cloned();
                let input = match operand {
                    Some((_, d)) => Input::Operand(inputs[0].clone(), d.clone()),
                    None => Input::InOutput,
                };
                (Ready::Pending(pending), input)
            };
            self.enqueue_collective(
                op,
                group,
                device,
                dtype,
                shape,
                ready,
                input,
                Data::new(out),
            )
        }

        /// Signals the open command buffer's input event after the work encoded so far.
        fn input_signal(&mut self, device: &MetalDevice) -> Result<crate::distributed::Ready> {
            if self.signal.is_none() {
                self.signal = Some(Arc::new(BufferSignal::new(device)?));
            }
            let signal = self.signal.clone().expect("just set");
            let value = signal.next();
            self.command_buffer()?.signal_event(&signal.event, value);
            Ok(crate::distributed::Ready::Signal(signal, value))
        }

        /// A dispatch writing `out`, which orders what follows after `out`'s producers and
        /// readers (the buffers' fences and barriers), as the input copy does.
        fn order_write(&mut self, out: &mut Data) -> Result<()> {
            let buffer = Arc::get_mut(&mut out.buffer)
                .and_then(Buffer::metal_mut)
                .expect("donated: the only handle, on the GPU");
            self.dispatch("nuvec_order", &[Arg::write(buffer)], 1)
        }

        /// Queues the collective on its group's worker, which waits for `ready`, and makes
        /// the work encoded after it wait for the worker's signal.
        #[allow(clippy::too_many_arguments)]
        fn enqueue_collective(
            &mut self,
            op: &crate::distributed::CollectiveOp,
            group: usize,
            device: &MetalDevice,
            dtype: DType,
            shape: Vec<usize>,
            ready: crate::distributed::Ready,
            input: crate::distributed::Input,
            out: Data,
        ) -> Result<Vec<Data>> {
            use crate::distributed::GpuJob;
            let status = ChunkStatus::default();
            let job = GpuJob {
                op: op.clone(),
                dtype,
                shape,
                ready,
                input,
                out: out.clone(),
                status: status.clone(),
            };
            let (worker, value) = crate::distributed::submit_gpu(group, job, device)?;
            self.command_buffer()?
                .wait_event(crate::distributed::worker_event(&*worker), value);
            self.collectives.push(status);
            Ok(vec![out])
        }

        /// Whether the open command buffer holds collectives: it must be committed before
        /// this thread blocks on anything, since their worker waits for it.
        pub(crate) fn collectives_pending(&self) -> bool {
            !self.collectives.is_empty()
        }

        /// Holds `outputs` of `node` until the GPU work producing them has completed (or
        /// nothing refers to `node` any more).
        pub(crate) fn defer(&mut self, node: Arc<Node>, outputs: Vec<Data>) {
            // The weak reference also keeps the node's address from being reused while the
            // entry exists.
            self.index
                .insert(Arc::as_ptr(&node), (Arc::downgrade(&node), outputs.clone()));
            self.large |= outputs.iter().any(|d| d.buffer.len() >= LARGE);
            self.chunk.push((node, outputs));
        }

        /// Drops the pending outputs of nodes nothing else refers to (intermediates whose
        /// consumers are all encoded): nobody can read them, and their memory returns to the
        /// allocator at once, for later GPU work to reuse (ordered after its uses, as MLX
        /// reuses memory).
        pub(crate) fn release_unreferenced(&mut self) {
            self.chunk.retain(|(node, _)| Arc::strong_count(node) > 1);
            {
                let mut p = self
                    .shared
                    .state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner);
                let next = p.next;
                for chunk in p.chunks[next..].iter_mut().flatten() {
                    chunk.retain(|(node, _)| node.strong_count() > 0);
                }
            }
            self.index.retain(|_, (node, _)| node.strong_count() > 0);
        }

        /// Takes over output `output` of `node` when this evaluation produced it and
        /// nothing but the caller's input refers to it (donation within an evaluation).
        /// In the open command buffer, the node is held by that input and the open chunk,
        /// and its data by the chunk and the index: both entries are removed and the data
        /// returned (nothing else can read the output, so it is never published).
        /// In a committed command buffer, only the index holds the data besides the node:
        /// its entry is removed, and `None` leaves the caller's in-flight (or, once
        /// published, finished) path to take the data. `usable(data, handles)` checks the
        /// layout and that the buffer has exactly `handles` handles. Later dispatches that
        /// write the buffer are ordered after its producer and readers by the encoder's
        /// barriers and the buffer's fences.
        pub(crate) fn take_pending(
            &mut self,
            node: &Arc<Node>,
            output: usize,
            usable: impl Fn(&Data, usize) -> bool,
        ) -> Option<Data> {
            let key = Arc::as_ptr(node);
            let (_, outputs) = self.index.get(&key)?;
            let data = outputs.get(output)?;
            match self.chunk.iter().position(|(n, _)| Arc::ptr_eq(n, node)) {
                Some(at) => {
                    if Arc::strong_count(node) != 2 || !usable(data, 2) {
                        return None;
                    }
                    self.index.remove(&key);
                    // The node's other outputs go too: nothing else refers to the node.
                    Some(self.chunk.remove(at).1.swap_remove(output))
                }
                None => {
                    // Only the caller's input refers to the node (a later consumer in this
                    // evaluation would hold it too): the entry is of no further use, and its
                    // weak handle and data copy would keep the caller from owning the
                    // output, in flight or (once published) finished.
                    if Arc::strong_count(node) == 1 {
                        self.index.remove(&key);
                    }
                    None
                }
            }
        }

        /// The pending output `output` of `node`, if it was encoded in this evaluation.
        /// GPU consumers may use it at once (the timeline orders them after its producer);
        /// CPU consumers must [`flush`](Self::flush) first.
        pub(crate) fn lookup(&self, node: &Arc<Node>, output: u32) -> Option<Data> {
            self.index
                .get(&Arc::as_ptr(node))?
                .1
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

    /// Whether `primitive`'s kernels address these operands of 2^32 elements or more (MLX
    /// runs them on the GPU with 64-bit indices; anything else falls back to the CPU):
    /// element-wise operations whose operands are flat or scalars (in chunks of 2^31
    /// elements) or strided in at most three dimensions (64-bit offsets), row softmax, and
    /// matrix products on the tiled kernels (64-bit tile offsets and batch strides).
    fn large_supported(
        primitive: &Primitive,
        operands: &[Operand<'_>],
        outputs: &[Output],
    ) -> bool {
        let int = |n: usize| n < i32::MAX as usize;
        match primitive {
            Primitive::Unary(_)
            | Primitive::Binary(_)
            | Primitive::Select
            | Primitive::AsType(_) => {
                let metas: Vec<&Output> = operands.iter().map(|(m, _)| *m).collect();
                large_elementwise(&metas, &outputs[0].shape)
            }
            Primitive::Fast(
                crate::fast::imp::FastOp::Softmax
                | crate::fast::imp::FastOp::LogSumExp
                | crate::fast::imp::FastOp::CausalSoftmax { .. },
            ) => {
                let x = operands[0].0;
                let axis = x.shape.last().copied().unwrap_or(1);
                // One threadgroup (at most 1,024 threads) per row on a 1-D grid.
                x.flags.row_contiguous
                    && int(axis)
                    && x.size / axis.max(1) * 1024 < u32::MAX as usize
            }
            Primitive::Matmul => {
                let ((a, _), (b, _)) = (operands[0], operands[1]);
                let out = &outputs[0];
                let nd = out.shape.len();
                let (m, n, k) = (out.shape[nd - 2], out.shape[nd - 1], a.shape[nd - 1]);
                let batch = out.size / (m * n).max(1);
                let nax = crate::NAX_ROUTE.with(std::cell::Cell::get)
                    && MetalDevice::shared().is_some_and(|d| d.nax().is_active());
                let class = MetalDevice::shared()
                    .and_then(|d| d.info().architecture.chars().last())
                    .unwrap_or('s');
                m > crate::ROW_INDEPENDENT_ROWS
                    && n > 1
                    && int(m)
                    && int(n)
                    && int(k)
                    && !nax
                    && !splits_k(class, batch, m, n, k)
                    && Gemmand::view(operands[0], m, k).is_some()
                    && Gemmand::view(operands[1], k, n).is_some()
                    && a.dtype == b.dtype
            }
            _ => false,
        }
    }

    /// Elements per dispatch of a flat element-wise kernel (32-bit thread positions).
    const FLAT_CHUNK: usize = 1 << 31;

    /// MLX's split-K rule (its case 1): few output tiles over a long K.
    fn splits_k(class: char, batch: usize, m: usize, n: usize, k: usize) -> bool {
        let threshold = if matches!(class, 's' | 'd') {
            2048
        } else {
            1024
        };
        batch == 1 && m.div_ceil(16) * n.div_ceil(16) <= threshold && k / 16 >= 8 && k >= m.max(n)
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

    /// Test-only fault injection: the next command buffer this thread commits completes
    /// with an error.
    #[cfg(test)]
    pub(crate) mod fault {
        use std::cell::Cell;

        thread_local!(static FAIL_NEXT: Cell<bool> = const { Cell::new(false) });

        /// Fails the next command buffer committed on this thread.
        pub(crate) fn fail_next_commit() {
            FAIL_NEXT.set(true);
        }

        pub(super) fn take() -> bool {
            FAIL_NEXT.replace(false)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::Array;

        /// A GPU failure fails the evaluation that encoded into the failed command buffer,
        /// and nothing independent of it.
        #[test]
        fn a_failed_chunk_fails_only_its_evaluation() {
            if MetalDevice::shared().is_none() {
                return;
            }
            crate::set_default_stream(crate::Stream::default_for(crate::Device::gpu(0)));
            let x = Array::from_slice(&[1.0f32, 2.0, 3.0], [3]);
            let failing = crate::ops::imp::exp(&x).unwrap();
            fault::fail_next_commit();
            let err = failing.eval().unwrap_err();
            assert!(err.to_string().contains("injected"), "{err}");
            assert!(failing.to_vec::<f32>().is_err(), "sticky for that array");
            // An independent evaluation afterwards succeeds.
            let y = crate::ops::imp::exp(&(&x * 2.0f32)).unwrap();
            let expected: Vec<f32> = [2.0f32, 4.0, 6.0].iter().map(|v| v.exp()).collect();
            let got = y.to_vec::<f32>().unwrap();
            for (g, e) in got.iter().zip(&expected) {
                assert!((g - e).abs() <= 1e-5 * e, "{got:?}");
            }
            // As does reading the inputs of the failed evaluation.
            assert_eq!(x.to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0]);
        }

        /// A consumer reading a failed evaluation's in-flight result in another evaluation
        /// fails too.
        #[test]
        fn readers_of_a_failed_chunk_fail() {
            if MetalDevice::shared().is_none() {
                return;
            }
            crate::set_default_stream(crate::Stream::default_for(crate::Device::gpu(0)));
            let x = Array::from_slice(&[1.0f32; 1024], [1024]);
            let mut producer = x.clone();
            for _ in 0..64 {
                producer = crate::ops::imp::exp(&(&producer * 0.5f32)).unwrap();
            }
            fault::fail_next_commit();
            // The chain spans several command buffers: when the failed first one completes
            // before the last is committed, `async_eval` already reports it.
            if let Err(e) = crate::async_eval(std::slice::from_ref(&producer)) {
                assert!(e.to_string().contains("injected"), "{e}");
            }
            // Encoded while the producer is (usually) still in flight.
            let consumer = &producer + 1.0f32;
            assert!(consumer.eval().is_err());
            assert!(producer.eval().is_err());
        }

        fn pending() -> (Arc<Node>, Vec<Data>) {
            let x = Array::from(1.0f32);
            let y = crate::ops::imp::negative(&x).unwrap();
            let data = x.out().slot.get().unwrap().clone().unwrap();
            (y.node.clone(), vec![data])
        }

        #[test]
        fn chunks_publish_in_order_and_failures_poison_later_chunks() {
            let shared = Shared::default();
            let nodes: Vec<_> = (0..3).map(|_| pending()).collect();
            for (node, data) in &nodes {
                node.outputs[0].inflight.set(data[0].clone());
            }
            {
                let mut p = shared.state.lock().unwrap();
                for n in &nodes {
                    p.chunks.push(Some(weak(std::slice::from_ref(n))));
                    p.results.push(None);
                    p.exposed.push(true);
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
