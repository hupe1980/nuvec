//! Command encoding with minimal barriers, and submission.

use std::ptr::NonNull;

use std::sync::{Arc, Mutex, PoisonError};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice, MTLDispatchType, MTLFence, MTLSize,
};

use super::buffer::Guard;
use super::{Buffer, Device, Pipeline};
use crate::{GpuError, Result};

/// Metal's limit on buffer argument slots per kernel.
const MAX_ARGS: usize = 31;
/// Metal's limit for inline bytes (`setBytes`).
const MAX_INLINE_BYTES: usize = 4096;

/// A kernel argument, bound to the slot of its position in the argument list.
#[derive(Debug)]
pub enum Arg<'a> {
    /// A buffer the kernel only reads, from byte `offset` on.
    Read(&'a Buffer, usize),
    /// A buffer the kernel writes (and may read), from byte `offset` on.
    Write(&'a mut Buffer, usize),
    /// Small constant data copied into the command stream (at most 4 KiB).
    Bytes(&'a [u8]),
}

impl<'a> Arg<'a> {
    /// The whole buffer, read-only.
    pub fn read(buffer: &'a Buffer) -> Self {
        Self::Read(buffer, 0)
    }

    /// The whole buffer, written.
    pub fn write(buffer: &'a mut Buffer) -> Self {
        Self::Write(buffer, 0)
    }

    /// Inline constant bytes.
    pub fn bytes(bytes: &'a [u8]) -> Self {
        Self::Bytes(bytes)
    }

    fn buffer(&self) -> Option<(&Buffer, usize, bool)> {
        match self {
            Self::Read(b, offset) => Some((b, *offset, false)),
            Self::Write(b, offset) => Some((b, *offset, true)),
            Self::Bytes(_) => None,
        }
    }
}

pub(crate) type Fence = Retained<ProtocolObject<dyn MTLFence>>;

/// Orders command buffers that are committed without waiting: each command buffer created
/// on a timeline starts its work only after the previous one's work has finished. Buffers
/// are untracked, so consecutive command buffers would otherwise overlap on the GPU (as
/// [`Device::ordered_command_buffer`]s do when they are independent).
///
/// Implemented with one fence (as in MLX): a command buffer's compute pass waits for the
/// fence and updates it at the end. Command buffers without work leave it untouched, so
/// empty or abandoned ones need no special handling.
pub struct Timeline {
    fence: Fence,
    started: bool,
}

impl std::fmt::Debug for Timeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timeline").finish_non_exhaustive()
    }
}

/// Work being recorded for the GPU: one concurrent compute encoder, with memory barriers
/// inserted only where a dispatch depends on an earlier one.
pub struct CommandBuffer {
    device: Device,
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// The timeline fence, and whether the compute pass must wait for it.
    fence: Option<(Fence, bool)>,
    /// For ordered command buffers: this command buffer's fence (updated at the end of its
    /// work and recorded on the buffers it writes), and the fences already waited for.
    own: Option<Fence>,
    waited: Vec<usize>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// Buffers read and written since the last barrier.
    reads: Vec<usize>,
    writes: Vec<usize>,
    barriers: usize,
    /// In-flight guards of the buffers used, and the buffers themselves (kept alive).
    guards: Vec<(Arc<Guard>, bool)>,
    retained: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// The kernels dispatched, while profiling.
    kernels: Vec<Arc<str>>,
}

impl Device {
    /// A new command buffer on this device's queue.
    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let raw = self
            .queue()
            .commandBuffer()
            .ok_or_else(|| GpuError::CommandBuffer("cannot create a command buffer".into()))?;
        Ok(CommandBuffer {
            device: self.clone(),
            raw,
            fence: None,
            own: None,
            waited: Vec::new(),
            encoder: None,
            reads: Vec::new(),
            writes: Vec::new(),
            barriers: 0,
            guards: Vec::new(),
            retained: Vec::new(),
            kernels: Vec::new(),
        })
    }
}

impl Device {
    /// A new, empty timeline.
    pub fn timeline(&self) -> Result<Timeline> {
        let fence = self
            .raw()
            .newFence()
            .ok_or_else(|| GpuError::CommandBuffer("cannot create a fence".into()))?;
        Ok(Timeline {
            fence,
            started: false,
        })
    }

    /// A new command buffer whose work reads the outputs of earlier *ordered* command
    /// buffers (committed on this device from any thread) safely while they are in flight:
    /// before a dispatch reads or writes a buffer with a pending GPU write, it waits for the
    /// writing command buffer's fence. Independent command buffers overlap on the GPU, as
    /// in MLX.
    ///
    /// Reads wait for the pending writer; writes wait for every pending user (writing a
    /// buffer that earlier work still reads). Other command buffers do not wait.
    pub fn ordered_command_buffer(&self) -> Result<CommandBuffer> {
        let fence = self
            .raw()
            .newFence()
            .ok_or_else(|| GpuError::CommandBuffer("cannot create a fence".into()))?;
        let mut cmd = self.command_buffer()?;
        cmd.own = Some(fence);
        Ok(cmd)
    }

    /// A new command buffer whose work runs after the work of every command buffer created
    /// earlier on `timeline` (commit them in creation order).
    pub fn command_buffer_on(&self, timeline: &mut Timeline) -> Result<CommandBuffer> {
        let mut cmd = self.command_buffer()?;
        cmd.fence = Some((timeline.fence.clone(), timeline.started));
        timeline.started = true;
        Ok(cmd)
    }
}

/// Reports a finished command buffer's GPU error, if any.
fn status(raw: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    if raw.status() == MTLCommandBufferStatus::Error {
        let message = raw.error().map_or_else(
            || "unknown error".to_owned(),
            |e| e.localizedDescription().to_string(),
        );
        return Err(GpuError::CommandBuffer(message));
    }
    Ok(())
}

impl CommandBuffer {
    /// Encodes `pipeline` over `threads` threads (a 1-D grid) with `args` bound to slots
    /// `0..args.len()`. The threadgroup size is derived from the pipeline's limits.
    pub fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'_>],
        threads: usize,
    ) -> Result<()> {
        if threads == 0 {
            return validate(args);
        }
        let width = pipeline.execution_width().max(1);
        let mut group = pipeline.max_threads_per_threadgroup().min(threads).max(1);
        if group > width {
            group -= group % width;
        }
        self.encode(pipeline, args, [threads, 1, 1], |encoder| {
            encoder
                .dispatchThreads_threadsPerThreadgroup(size([threads, 1, 1]), size([group, 1, 1]));
        })
    }

    /// Encodes `pipeline` over a grid of `groups` threadgroups of `threads` threads each, for
    /// kernels that cooperate within a threadgroup (reductions, tiled matrix products).
    pub fn dispatch_groups(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'_>],
        groups: [usize; 3],
        threads: [usize; 3],
    ) -> Result<()> {
        let per_group: usize = threads.iter().product();
        if per_group > pipeline.max_threads_per_threadgroup() {
            return Err(GpuError::InvalidArgument(format!(
                "{per_group} threads per threadgroup; `{}` allows {}",
                pipeline.name(),
                pipeline.max_threads_per_threadgroup()
            )));
        }
        if groups.contains(&0) || per_group == 0 {
            return validate(args);
        }
        let extent = [0, 1, 2].map(|d| groups[d] * threads[d]);
        self.encode(pipeline, args, extent, |encoder| {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
        })
    }

    /// Encodes `pipeline` over exactly `grid` threads in up to three dimensions, in
    /// threadgroups of `group` threads (the edge threadgroups are partial).
    pub fn dispatch_threads(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'_>],
        grid: [usize; 3],
        group: [usize; 3],
    ) -> Result<()> {
        let per_group: usize = group.iter().product();
        if per_group > pipeline.max_threads_per_threadgroup() {
            return Err(GpuError::InvalidArgument(format!(
                "{per_group} threads per threadgroup; `{}` allows {}",
                pipeline.name(),
                pipeline.max_threads_per_threadgroup()
            )));
        }
        if grid.contains(&0) || per_group == 0 {
            return validate(args);
        }
        self.encode(pipeline, args, grid, |encoder| {
            encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size(group));
        })
    }

    /// Validates and binds `args`, inserts a barrier if needed, then lets `dispatch` encode
    /// over `extent` threads (for profiling).
    fn encode(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'_>],
        extent: [usize; 3],
        dispatch: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<()> {
        if super::profile::enabled() {
            self.kernels.push(match super::profile::shapes() {
                true => format!("{} {extent:?}", pipeline.name()).into(),
                false => pipeline.name.clone(),
            });
        }
        validate(args)?;
        if self.conflicts(args) {
            self.encoder()?
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
            self.barriers += 1;
            self.reads.clear();
            self.writes.clear();
        }
        let encoder = self.encoder()?.clone();
        if let Some(own) = &self.own {
            let own = Retained::as_ptr(own) as usize;
            for (buffer, _, write) in args.iter().filter_map(Arg::buffer) {
                for fence in buffer.pending_fences(write) {
                    let key = Retained::as_ptr(&fence) as usize;
                    if key != own && !self.waited.contains(&key) {
                        encoder.waitForFence(&fence);
                        self.waited.push(key);
                    }
                }
            }
        }
        encoder.setComputePipelineState(pipeline.raw());
        for (index, arg) in args.iter().enumerate() {
            match arg {
                Arg::Bytes(bytes) => {
                    let data = NonNull::new(bytes.as_ptr().cast_mut())
                        .unwrap_or(NonNull::dangling())
                        .cast();
                    // SAFETY: `validate` checked `index < 31` and `bytes.len() <= 4096`;
                    // Metal copies the bytes during this call, so the pointer need only be
                    // valid for `bytes.len()` bytes now, which the slice guarantees.
                    unsafe { encoder.setBytes_length_atIndex(data, bytes.len(), index) };
                }
                Arg::Read(..) | Arg::Write(..) => {
                    let (buffer, offset, _) = arg.buffer().expect("buffer argument");
                    // SAFETY: `validate` checked `index < 31` and `offset <= buffer.len()`, so
                    // the binding stays within the buffer; the buffer is retained by this
                    // command buffer until the GPU has finished.
                    unsafe { encoder.setBuffer_offset_atIndex(buffer.raw(), offset, index) };
                }
            }
        }
        for (buffer, _, write) in args.iter().filter_map(Arg::buffer) {
            (if write {
                &mut self.writes
            } else {
                &mut self.reads
            })
            .push(buffer.id());
            let guard = buffer.guard();
            guard.acquire(write);
            self.guards.push((guard, write));
            self.retained.extend(buffer.retained());
        }
        dispatch(&encoder);
        Ok(())
    }

    /// Memory barriers inserted so far.
    pub fn barriers(&self) -> usize {
        self.barriers
    }

    /// Ends encoding, updating the timeline fence after this command buffer's work.
    fn close(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            if let Some((fence, _)) = &self.fence {
                encoder.updateFence(fence);
            }
            if let Some(own) = &self.own {
                encoder.updateFence(own);
                for (guard, write) in &self.guards {
                    guard.record(own, *write);
                }
            }
            encoder.endEncoding();
        }
        self.device.flights().begin();
    }

    /// Submits the work to the GPU.
    pub fn commit(mut self) -> Submitted {
        self.close();
        self.raw.commit();
        self.mark_committed();
        Submitted {
            device: self.device.clone(),
            raw: self.raw.clone(),
            unfenced: self.own.is_none(),
            guards: std::mem::take(&mut self.guards),
            _retained: std::mem::take(&mut self.retained),
            done: false,
        }
    }

    /// Submits the work and returns at once. `on_complete` runs exactly once with the
    /// command buffer's result: on a Metal thread when the GPU finishes, or earlier on the
    /// thread that calls [`Pending::wait`].
    pub fn commit_then(mut self, on_complete: impl FnOnce(Result<()>) + Send + 'static) -> Pending {
        self.close();
        // Without a timeline fence, nothing orders other work after these uses.
        let guards: Vec<Arc<Guard>> = match self.own {
            Some(_) => self.guards.iter().map(|(g, _)| g.clone()).collect(),
            None => Vec::new(),
        };
        let finish: Finish = Arc::new(Mutex::new(Some(Completion {
            on_complete: Box::new(on_complete),
            kernels: std::mem::take(&mut self.kernels),
            unfenced: self.own.is_none(),
            guards: std::mem::take(&mut self.guards),
            _retained: std::mem::take(&mut self.retained),
            device: self.device.clone(),
        })));
        let shared = finish.clone();
        let handler = RcBlock::new(move |raw: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // SAFETY: Metal passes the completed command buffer, valid for this call.
            complete(&shared, unsafe { raw.as_ref() });
        });
        // SAFETY: the block is valid, and Metal copies it before this call returns. It is
        // called at most once, on any thread: its captures are `Send` (the buffers are
        // only retained or released, and the guards are atomics).
        unsafe { self.raw.addCompletedHandler(RcBlock::as_ptr(&handler)) };
        self.raw.commit();
        for guard in guards {
            guard.committed();
        }
        Pending {
            raw: self.raw.clone(),
            finish,
        }
    }

    /// The buffers' uses in this command buffer are committed: their fences now order work
    /// committed later (only with a timeline fence; otherwise they count once complete).
    fn mark_committed(&self) {
        if self.own.is_some() {
            for (guard, _) in &self.guards {
                guard.committed();
            }
        }
    }

    /// Whether a dispatch with `args` depends on dispatches since the last barrier:
    /// read-after-write, write-after-write or write-after-read.
    fn conflicts(&self, args: &[Arg<'_>]) -> bool {
        args.iter().filter_map(Arg::buffer).any(|(b, _, write)| {
            let id = b.id();
            self.writes.contains(&id) || (write && self.reads.contains(&id))
        })
    }

    fn encoder(&mut self) -> Result<&Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        if self.encoder.is_none() {
            let encoder = self
                .raw
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .ok_or_else(|| GpuError::CommandBuffer("cannot create a compute encoder".into()))?;
            if let Some((fence, true)) = &self.fence {
                encoder.waitForFence(fence);
            }
            self.encoder = Some(encoder);
        }
        Ok(self.encoder.as_ref().expect("just created"))
    }
}

impl Drop for CommandBuffer {
    /// Abandoned work is never submitted; release the buffers' guards.
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        for (guard, write) in self.guards.drain(..) {
            guard.committed();
            guard.release(write);
        }
    }
}

fn size([width, height, depth]: [usize; 3]) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

fn validate(args: &[Arg<'_>]) -> Result<()> {
    if args.len() > MAX_ARGS {
        return Err(GpuError::InvalidArgument(format!(
            "{} arguments; Metal allows {MAX_ARGS}",
            args.len()
        )));
    }
    for (index, arg) in args.iter().enumerate() {
        match arg {
            Arg::Bytes(bytes) if bytes.len() > MAX_INLINE_BYTES => {
                return Err(GpuError::InvalidArgument(format!(
                    "argument {index}: {} inline bytes; Metal allows {MAX_INLINE_BYTES}",
                    bytes.len()
                )));
            }
            Arg::Write(b, _) if b.is_mapped() => return Err(GpuError::ReadOnly),
            Arg::Read(..) | Arg::Write(..)
                if arg.buffer().is_some_and(|(b, offset, _)| offset > b.len()) =>
            {
                let (b, offset, _) = arg.buffer().expect("buffer argument");
                return Err(GpuError::InvalidArgument(format!(
                    "argument {index}: offset {offset} is past the end of a {}-byte buffer",
                    b.len()
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

/// What remains to do when a command buffer committed with
/// [`CommandBuffer::commit_then`] completes.
struct Completion {
    on_complete: Box<dyn FnOnce(Result<()>) + Send>,
    /// The kernels dispatched, while profiling.
    kernels: Vec<Arc<str>>,
    guards: Vec<(Arc<Guard>, bool)>,
    /// The command buffer had no timeline fence: its uses count as committed only now.
    unfenced: bool,
    _retained: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    device: Device,
}

// SAFETY: the retained Metal buffers are never accessed, only released (Metal objects may be
// released from any thread); the guards are atomics, the callback is `Send`, and `Device` is
// shared across threads by design (`Device::shared`).
unsafe impl Send for Completion {}

type Finish = Arc<Mutex<Option<Completion>>>;

/// Runs the completion once, whoever gets there first (the waiter or Metal's handler).
fn complete(finish: &Finish, raw: &ProtocolObject<dyn MTLCommandBuffer>) {
    let Some(c) = finish.lock().unwrap_or_else(PoisonError::into_inner).take() else {
        return;
    };
    let result = status(raw);
    super::profile::record(&c.kernels, raw.GPUEndTime() - raw.GPUStartTime());
    for (guard, write) in c.guards {
        if c.unfenced {
            guard.committed();
        }
        guard.release(write);
    }
    drop(c._retained);
    c.device.flights().end();
    (c.on_complete)(result);
}

/// A command buffer committed with [`CommandBuffer::commit_then`]. Dropping it does not
/// wait; [`wait`](Self::wait) blocks until the GPU has finished and runs the completion on
/// this thread if Metal's handler has not run it yet.
pub struct Pending {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    finish: Finish,
}

impl Pending {
    /// Blocks until the GPU has finished; the completion has run when this returns.
    pub fn wait(&self) {
        self.raw.waitUntilCompleted();
        complete(&self.finish, &self.raw);
    }
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").finish_non_exhaustive()
    }
}

/// Submitted GPU work. [`wait`](Self::wait) for it to finish; dropping it also waits.
pub struct Submitted {
    device: Device,
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// The command buffer had no timeline fence: its uses count as committed only once
    /// complete.
    unfenced: bool,
    guards: Vec<(Arc<Guard>, bool)>,
    _retained: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    done: bool,
}

impl Submitted {
    /// Blocks until the GPU has finished, then reports any GPU error.
    pub fn wait(mut self) -> Result<()> {
        self.finish()
    }

    fn finish(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.raw.waitUntilCompleted();
        self.done = true;
        for (guard, write) in self.guards.drain(..) {
            if self.unfenced {
                guard.committed();
            }
            guard.release(write);
        }
        self.device.flights().end();
        status(&self.raw)
    }
}

impl Drop for Submitted {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
