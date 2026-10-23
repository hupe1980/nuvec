//! Command encoding with minimal barriers, and submission.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
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
    /// The buffer of the argument at this index bound again, from byte `offset` on: an
    /// in-place kernel reading and writing one buffer through two slots. That argument must
    /// be an [`Arg::Write`] (whose hazards cover both slots); anything else is an
    /// [`GpuError::InvalidArgument`].
    Same(usize, usize),
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
            Self::Bytes(_) | Self::Same(..) => None,
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
    /// Per buffer (by id): its uses in this command buffer, all and the writing ones.
    uses: HashMap<usize, (usize, usize)>,
    retained: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// The kernels dispatched, while profiling.
    kernels: Vec<Arc<str>>,
}

impl Device {
    /// A new command buffer on this device's queue. It is *unordered*: its work does not
    /// wait for other command buffers, so binding reused memory whose previous owner's GPU
    /// work is still running ([`Device::new_gpu_buffer`]) is refused with
    /// [`GpuError::BufferInUse`]; use an
    /// [`ordered_command_buffer`](Self::ordered_command_buffer) for that.
    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let raw = autoreleasepool(|_| self.queue().commandBuffer())
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
            uses: HashMap::new(),
            retained: Vec::new(),
            kernels: Vec::new(),
        })
    }
}

impl Device {
    /// A new, empty timeline.
    pub fn timeline(&self) -> Result<Timeline> {
        let fence = autoreleasepool(|_| self.raw().newFence())
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
        let fence = autoreleasepool(|_| self.raw().newFence())
            .ok_or_else(|| GpuError::CommandBuffer("cannot create a fence".into()))?;
        let mut cmd = self.command_buffer()?;
        cmd.own = Some(fence);
        Ok(cmd)
    }

    /// A new command buffer whose work runs after the work of every command buffer created
    /// earlier on `timeline` (commit them in creation order). Like
    /// [`command_buffer`](Self::command_buffer), it refuses reused memory whose previous
    /// owner's GPU work is still running.
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
    ///
    /// Arguments are validated (slot count, inline size, offsets within their buffers,
    /// [`Arg::Same`] targets, ordering with other command buffers), but what the kernel does
    /// with them is not: nuvec cannot see which elements a kernel touches.
    ///
    /// # Safety
    ///
    /// The kernel, run over this grid, must access each bound buffer only within its length
    /// from the bound offset (and write only the buffers bound with [`Arg::Write`] or
    /// [`Arg::Same`]). An out-of-bounds access reads or writes memory that other buffers,
    /// possibly referenced by safe code, own.
    pub unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'_>],
        threads: usize,
    ) -> Result<()> {
        if threads == 0 {
            return self.check(args);
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
    ///
    /// # Safety
    ///
    /// As for [`dispatch`](Self::dispatch): the kernel must access each bound buffer only
    /// within its length from the bound offset.
    pub unsafe fn dispatch_groups(
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
            return self.check(args);
        }
        let extent = [0, 1, 2].map(|d| groups[d] * threads[d]);
        self.encode(pipeline, args, extent, |encoder| {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
        })
    }

    /// Encodes `pipeline` over exactly `grid` threads in up to three dimensions, in
    /// threadgroups of `group` threads (the edge threadgroups are partial).
    ///
    /// # Safety
    ///
    /// As for [`dispatch`](Self::dispatch): the kernel must access each bound buffer only
    /// within its length from the bound offset.
    pub unsafe fn dispatch_threads(
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
            return self.check(args);
        }
        self.encode(pipeline, args, grid, |encoder| {
            encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size(group));
        })
    }

    /// Validates and binds `args`, inserts a barrier if needed, then lets `dispatch` encode
    /// over `extent` threads (for profiling). Runs in an autorelease pool: Metal
    /// autoreleases objects while encoding (the compute encoder's context among them), and
    /// a thread other than the main one drains none of them otherwise.
    fn encode(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'_>],
        extent: [usize; 3],
        dispatch: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<()> {
        autoreleasepool(|_| self.encode_now(pipeline, args, extent, dispatch))
    }

    fn encode_now(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'_>],
        extent: [usize; 3],
        dispatch: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) -> Result<()> {
        self.check(args)?;
        if super::profile::enabled() {
            self.kernels.push(match super::profile::shapes() {
                true => format!("{} {extent:?}", pipeline.name()).into(),
                false => pipeline.name.clone(),
            });
        }
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
                Arg::Same(earlier, offset) => {
                    let (buffer, _, _) = args[*earlier].buffer().expect("validated");
                    // SAFETY: as above: `validate` checked that `earlier` names another
                    // buffer argument and `offset <= buffer.len()`.
                    unsafe { encoder.setBuffer_offset_atIndex(buffer.raw(), *offset, index) };
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
            let uses = self.uses.entry(buffer.id()).or_default();
            uses.0 += 1;
            uses.1 += usize::from(write);
            let guard = buffer.guard();
            guard.acquire(write);
            self.guards.push((guard, write));
            self.retained.extend(buffer.retained());
        }
        dispatch(&encoder);
        Ok(())
    }

    /// Validates `args`, then refuses uses that nothing would order after earlier GPU work:
    /// - reused memory whose previous owner's work is pending, in a command buffer without
    ///   per-buffer fences (only ordered command buffers wait for that work);
    /// - a buffer that another command buffer, not submitted yet, writes (or, for a write
    ///   here, uses at all): its fence orders later work only once it is submitted.
    fn check(&self, args: &[Arg<'_>]) -> Result<()> {
        validate(args)?;
        for (buffer, _, write) in args.iter().filter_map(Arg::buffer) {
            if self.own.is_none() && buffer.previous_busy() {
                return Err(GpuError::BufferInUse);
            }
            let (uses, writes) = buffer.unsubmitted();
            let (own_uses, own_writes) = self.uses.get(&buffer.id()).copied().unwrap_or_default();
            if writes > own_writes || (write && uses > own_uses) {
                return Err(GpuError::BufferInUse);
            }
        }
        Ok(())
    }

    /// Memory barriers inserted so far.
    pub fn barriers(&self) -> usize {
        self.barriers
    }

    /// Ends encoding, updating the timeline fence after this command buffer's work.
    fn close(&mut self) {
        autoreleasepool(|_| self.end_encoding());
        self.device.flights().begin();
    }

    fn end_encoding(&mut self) {
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
    }

    /// Submits the work to the GPU. Dropping the returned [`Submitted`] waits for it and
    /// discards its result; [`Submitted::wait`] reports GPU errors.
    #[must_use = "dropping `Submitted` waits for the GPU and discards its errors; call `wait`"]
    pub fn commit(self) -> Submitted {
        Submitted {
            pending: self.commit_then(|_| {}),
            waited: false,
        }
    }

    /// Submits the work and returns at once. `on_complete` runs exactly once with the
    /// command buffer's result: on a Metal thread when the GPU finishes, or earlier on the
    /// thread that calls [`Pending::wait`].
    ///
    /// `on_complete` must not block, in particular not on other GPU work: Metal runs a
    /// queue's completion handlers one at a time, so waiting there for another command
    /// buffer's completion can deadlock.
    /// A panic in `on_complete` on Metal's thread is caught and reported on standard error,
    /// so it never aborts the process; the result is still published to [`Pending::wait`].
    #[must_use = "dropping `Pending` does not wait; keep it to `wait`, or drop it explicitly"]
    pub fn commit_then(mut self, on_complete: impl FnOnce(Result<()>) + Send + 'static) -> Pending {
        self.close();
        let guards = self.guards.clone();
        let finish = Arc::new(Finish {
            completion: Mutex::new(Some(Completion {
                on_complete: Box::new(on_complete),
                own: self.own.clone(),
                kernels: std::mem::take(&mut self.kernels),
                unfenced: self.own.is_none(),
                guards: std::mem::take(&mut self.guards),
                _retained: std::mem::take(&mut self.retained),
                device: self.device.clone(),
            })),
            result: Mutex::new(None),
            done: Condvar::new(),
        });
        let shared = finish.clone();
        let handler = RcBlock::new(move |raw: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // A panic must not unwind into Metal's Objective-C caller, which aborts the
            // process: report it instead (the result is published either way).
            let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // SAFETY: Metal passes the completed command buffer, valid for this call.
                complete(&shared, unsafe { raw.as_ref() });
            }));
            if completed.is_err() {
                eprintln!("nuvec-gpu: a command buffer's completion handler panicked");
            }
        });
        autoreleasepool(|_| {
            // SAFETY: the block is valid, and Metal copies it before this call returns. It
            // is called at most once, on any thread: its captures are `Send` (the buffers
            // are only retained or released, and the guards are atomics).
            unsafe { self.raw.addCompletedHandler(RcBlock::as_ptr(&handler)) };
            self.raw.commit();
        });
        for (guard, write) in guards {
            guard.submitted(write);
            // With a fence, these uses now order work committed later; without one, they
            // count as committed once complete.
            if self.own.is_some() {
                guard.committed();
            }
        }
        Pending {
            raw: self.raw.clone(),
            finish,
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
    /// Abandoned work is never submitted; release the buffers' guards. (Metal objects are
    /// released in an autorelease pool: their deallocation may autorelease others.)
    fn drop(&mut self) {
        autoreleasepool(|_| {
            if let Some(encoder) = self.encoder.take() {
                encoder.endEncoding();
            }
            for (guard, write) in self.guards.drain(..) {
                guard.committed();
                guard.submitted(write);
                guard.release(write, None); // abandoned before its fences were recorded
            }
            self.retained.clear();
        });
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
            Arg::Same(earlier, offset) => match args.get(*earlier) {
                Some(Arg::Write(b, _)) if *offset <= b.len() => {}
                _ => {
                    return Err(GpuError::InvalidArgument(format!(
                        "argument {index}: `Same({earlier}, {offset})` needs a written \
                         (`Arg::Write`) argument and an offset within it"
                    )));
                }
            },
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
    /// The command buffer's fence, recorded on the buffers it uses.
    own: Option<Fence>,
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

/// The completion of a command buffer committed with [`CommandBuffer::commit_then`] (run
/// once, by whoever gets there first: a waiter or Metal's handler), and its result.
struct Finish {
    completion: Mutex<Option<Completion>>,
    /// The command buffer's result, set once the completion has run.
    result: Mutex<Option<Result<()>>>,
    done: Condvar,
}

impl Finish {
    /// Blocks until the completion has run (on any thread), and returns the result.
    fn wait(&self) -> Result<()> {
        let mut result = self.result.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if let Some(r) = &*result {
                return r.clone();
            }
            result = self
                .done
                .wait(result)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// Publishes the result when dropped, so that waiters wake up even if `on_complete` panics.
struct Publish<'a>(&'a Finish, Option<Result<()>>);

impl Drop for Publish<'_> {
    fn drop(&mut self) {
        let mut result = self.0.result.lock().unwrap_or_else(PoisonError::into_inner);
        *result = self.1.take();
        self.0.done.notify_all();
    }
}

/// Runs the completion once, whoever gets there first (a waiter or Metal's handler):
/// releases the buffers' uses, ends the flight, runs `on_complete`, then publishes the
/// result. This is the only place where submitted uses are released.
fn complete(finish: &Finish, raw: &ProtocolObject<dyn MTLCommandBuffer>) {
    let Some(c) = finish
        .completion
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
    else {
        return;
    };
    let result = status(raw);
    let _publish = Publish(finish, Some(result.clone()));
    super::profile::record(&c.kernels, raw.GPUEndTime() - raw.GPUStartTime());
    for (guard, write) in c.guards {
        if c.unfenced {
            guard.committed();
        }
        guard.release(write, c.own.as_ref());
    }
    drop(c._retained);
    c.device.flights().end();
    (c.on_complete)(result);
}

/// A command buffer committed with [`CommandBuffer::commit_then`]. Dropping it does not
/// wait; [`wait`](Self::wait) blocks until the GPU has finished and the completion has run
/// (on this thread, if Metal's handler has not started it yet).
#[must_use = "dropping `Pending` does not wait; keep it to `wait`, or drop it explicitly"]
pub struct Pending {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    finish: Arc<Finish>,
}

impl Pending {
    /// Blocks until the GPU has finished; the completion has run when this returns (its
    /// buffers are released and [`Device::in_flight`] no longer counts it).
    pub fn wait(&self) {
        let _ = self.result();
    }

    /// Waits as [`wait`](Self::wait) does and returns the command buffer's result.
    fn result(&self) -> Result<()> {
        autoreleasepool(|_| {
            self.raw.waitUntilCompleted();
            complete(&self.finish, &self.raw);
        });
        self.finish.wait()
    }
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").finish_non_exhaustive()
    }
}

/// Submitted GPU work. [`wait`](Self::wait) for it to finish; dropping it also waits (and
/// discards a GPU error).
#[must_use = "dropping `Submitted` waits for the GPU and discards its errors; call `wait`"]
pub struct Submitted {
    pending: Pending,
    waited: bool,
}

impl Submitted {
    /// Blocks until the GPU has finished and the buffers are released, then reports any
    /// GPU error.
    pub fn wait(mut self) -> Result<()> {
        self.waited = true;
        self.pending.result()
    }
}

impl std::fmt::Debug for Submitted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submitted").finish_non_exhaustive()
    }
}

impl Drop for Submitted {
    fn drop(&mut self) {
        if !self.waited {
            self.pending.wait();
        }
    }
}
