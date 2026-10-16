//! Command encoding with minimal barriers, and submission.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLDispatchType, MTLSize,
};

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

/// Work being recorded for the GPU: one concurrent compute encoder, with memory barriers
/// inserted only where a dispatch depends on an earlier one.
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// Buffers read and written since the last barrier.
    reads: Vec<usize>,
    writes: Vec<usize>,
    barriers: usize,
    /// In-flight guards of the buffers used, and the buffers themselves (kept alive).
    guards: Vec<Arc<AtomicUsize>>,
    retained: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Device {
    /// A new command buffer on this device's queue.
    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let raw = self
            .queue()
            .commandBuffer()
            .ok_or_else(|| GpuError::CommandBuffer("cannot create a command buffer".into()))?;
        Ok(CommandBuffer {
            raw,
            encoder: None,
            reads: Vec::new(),
            writes: Vec::new(),
            barriers: 0,
            guards: Vec::new(),
            retained: Vec::new(),
        })
    }
}

impl CommandBuffer {
    /// Encodes `pipeline` over `threads` threads (a 1-D grid) with `args` bound to slots
    /// `0..args.len()`.
    pub fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'_>],
        threads: usize,
    ) -> Result<()> {
        validate(args)?;
        if threads == 0 {
            return Ok(());
        }
        if self.conflicts(args) {
            self.encoder()?
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
            self.barriers += 1;
            self.reads.clear();
            self.writes.clear();
        }
        let encoder = self.encoder()?.clone();
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
            guard.fetch_add(1, Ordering::AcqRel);
            self.guards.push(guard);
            self.retained.extend(buffer.retained());
        }
        let width = pipeline.execution_width().max(1);
        let mut group = pipeline.max_threads_per_threadgroup().min(threads).max(1);
        if group > width {
            group -= group % width;
        }
        let size = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(size(threads), size(group));
        Ok(())
    }

    /// Memory barriers inserted so far.
    pub fn barriers(&self) -> usize {
        self.barriers
    }

    /// Submits the work to the GPU.
    pub fn commit(mut self) -> Submitted {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        self.raw.commit();
        Submitted {
            raw: self.raw.clone(),
            guards: std::mem::take(&mut self.guards),
            _retained: std::mem::take(&mut self.retained),
            done: false,
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
        for guard in self.guards.drain(..) {
            guard.fetch_sub(1, Ordering::AcqRel);
        }
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

/// Submitted GPU work. [`wait`](Self::wait) for it to finish; dropping it also waits.
pub struct Submitted {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    guards: Vec<Arc<AtomicUsize>>,
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
        for guard in self.guards.drain(..) {
            guard.fetch_sub(1, Ordering::AcqRel);
        }
        if self.raw.status() == MTLCommandBufferStatus::Error {
            let message = self.raw.error().map_or_else(
                || "unknown error".to_owned(),
                |e| e.localizedDescription().to_string(),
            );
            return Err(GpuError::CommandBuffer(message));
        }
        Ok(())
    }
}

impl Drop for Submitted {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
