//! Distributed groups and collective operations, as MLX's `mx.distributed`.
//!
//! A [`Group`] is a set of processes with a rank and a size. Collectives (`all_sum`,
//! `all_max`, `all_min`, `all_gather`, `sum_scatter`, `send`, `recv`) are lazy array operations over a
//! group: they run when their result is evaluated, compose with `grad`, `vmap` and
//! `compile` as MLX defines, and run on the CPU over the arrays' shared memory. On a GPU
//! stream they run on the group's worker thread once the GPU work producing their inputs
//! completes, and the GPU work reading their outputs waits for them on the GPU, so the
//! evaluating thread never waits there. Without a configured group,
//! the default group has size 1 and every collective is the identity, so single-process
//! code runs unchanged.
//!
//! The transport is a [`Communicator`]; `nuvec-distributed` provides MLX's TCP ring and
//! JACCL backends and installs one as the default group.

use std::panic::Location;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use crate::array::Data;
use crate::buffer::Buffer;
use crate::kernels::Operand;
use crate::primitive::Primitive;
use crate::{Array, DType, Error, IntoShape, Result};

mod worker;

/// The reduction of [`Communicator::all_reduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AllReduce {
    /// Element-wise sum (booleans: or).
    Sum,
    /// Element-wise maximum (booleans: or).
    Max,
    /// Element-wise minimum (booleans: and).
    Min,
}

/// The transport of a [`Group`]: every rank of the group calls the same operations in the
/// same order. Buffers are row-contiguous elements; a failure names the peer.
pub trait Communicator: Send + Sync {
    /// This process's rank, `0..size`.
    fn rank(&self) -> usize;
    /// The number of processes.
    fn size(&self) -> usize;
    /// Reduces `data` (elements of `dtype`) with every rank's `data`, in place; every rank
    /// gets the same bits.
    fn all_reduce(&self, op: AllReduce, dtype: DType, data: &mut [u8]) -> Result<()>;
    /// Every rank's `input` (equal lengths), concatenated in rank order into `output`.
    fn all_gather(&self, input: &[u8], output: &mut [u8]) -> Result<()>;
    /// Sends `data` to rank `dst`.
    fn send(&self, dst: usize, data: &[u8]) -> Result<()>;
    /// Receives exactly `data.len()` bytes from rank `src`.
    fn recv(&self, src: usize, data: &mut [u8]) -> Result<()>;
    /// Rank `r`'s `output` is the element-wise sum (elements of `dtype`) of every rank's
    /// `r`-th chunk of `input` (`input.len() == size · output.len()`), as MLX's
    /// `sum_scatter`.
    fn sum_scatter(&self, dtype: DType, input: &[u8], output: &mut [u8]) -> Result<()> {
        let _ = (dtype, input, output);
        Err(Error::InvalidArgument(
            "distributed: this backend has no sum_scatter".into(),
        ))
    }
    /// Whether every peer is still reachable, checked without a collective and without
    /// blocking; an error names a lost peer. Backends that cannot tell report `Ok`.
    fn health(&self) -> Result<()> {
        Ok(())
    }
    /// A subgroup of the ranks with the same `color`, ordered by `key` (MLX's `split`).
    fn split(&self, color: i32, key: i32) -> Result<Arc<dyn Communicator>> {
        let _ = (color, key);
        Err(Error::InvalidArgument(
            "distributed: this backend does not split groups (as MLX's ring)".into(),
        ))
    }
}

/// A collective operation's kind, with its parameters ([`Primitive::Collective`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CollectiveOp {
    /// [`AllReduce`] over the group.
    Reduce(AllReduce),
    /// Every rank's array, concatenated along the first axis in rank order.
    Gather,
    /// The sum over the group of every rank's rows of this rank's slice of the first axis.
    SumScatter,
    /// The input sent to rank `dst`; the output is the input.
    Send {
        /// The receiving rank.
        dst: usize,
    },
    /// An array of `shape` and `dtype` received from rank `src`.
    Recv {
        /// The sending rank.
        src: usize,
        /// The received shape.
        shape: Vec<usize>,
        /// The received dtype.
        dtype: DType,
    },
}

impl CollectiveOp {
    /// The operation's name (MLX's).
    pub fn name(&self) -> &'static str {
        match self {
            Self::Reduce(AllReduce::Sum) => "all_sum",
            Self::Reduce(AllReduce::Max) => "all_max",
            Self::Reduce(AllReduce::Min) => "all_min",
            Self::Gather => "all_gather",
            Self::SumScatter => "sum_scatter",
            Self::Send { .. } => "send",
            Self::Recv { .. } => "recv",
        }
    }
}

/// A group's registered transport and the lock that keeps its operations in program
/// order (one collective on the wire at a time).
struct Registered {
    comm: Arc<dyn Communicator>,
    order: Mutex<()>,
    /// Runs the collectives of GPU streams in the background, in program order.
    worker: Arc<worker::Worker>,
}

impl Drop for Registered {
    fn drop(&mut self) {
        self.worker.shut_down();
    }
}

/// Groups by id (`id - 1`; `None` once closed); id 0 is the size-1 group.
static GROUPS: Mutex<Vec<Option<Arc<Registered>>>> = Mutex::new(Vec::new());
static DEFAULT: RwLock<Option<Group>> = RwLock::new(None);

fn registered(id: usize) -> Result<Arc<Registered>> {
    let groups = GROUPS.lock().unwrap_or_else(PoisonError::into_inner);
    match id.checked_sub(1).and_then(|i| groups.get(i)) {
        Some(Some(g)) => Ok(g.clone()),
        Some(None) => Err(Error::Backend("distributed: the group was closed".into())),
        None => Err(Error::InvalidArgument(format!(
            "distributed: no group {id}"
        ))),
    }
}

/// A group of processes (MLX's `Group`): cheap to clone.
#[derive(Clone)]
pub struct Group {
    id: usize,
    rank: usize,
    size: usize,
}

impl std::fmt::Debug for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Group(rank {} of {})", self.rank, self.size)
    }
}

impl Group {
    /// The group of this process alone: every collective is the identity.
    pub fn single() -> Self {
        Self {
            id: 0,
            rank: 0,
            size: 1,
        }
    }

    /// A group over `comm` (a backend's transport).
    pub fn new(comm: Arc<dyn Communicator>) -> Self {
        let (rank, size) = (comm.rank(), comm.size());
        if size <= 1 {
            return Self::single();
        }
        let mut groups = GROUPS.lock().unwrap_or_else(PoisonError::into_inner);
        groups.push(Some(Arc::new(Registered {
            comm,
            order: Mutex::new(()),
            worker: Arc::default(),
        })));
        Self {
            id: groups.len(),
            rank,
            size,
        }
    }

    /// This process's rank in the group.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The number of processes in the group.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Closes the group: its transport (connections, buffers) is released once operations
    /// in progress finish, and every later operation on it fails with an error saying it was
    /// closed. A group of size 1 is unaffected.
    pub fn close(&self) {
        if let Some(i) = self.id.checked_sub(1) {
            let mut groups = GROUPS.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(slot) = groups.get_mut(i) {
                *slot = None;
            }
        }
    }

    /// Whether every peer is still reachable, checked without running a collective: an
    /// error names a lost peer (and the group is then failed), or says the group was closed.
    /// It waits for a collective in progress on another thread, never interleaving with it.
    pub fn health(&self) -> Result<()> {
        if self.id == 0 {
            return Ok(());
        }
        let entry = registered(self.id)?;
        let _order = entry.worker.idle_order(&entry.order);
        entry.comm.health()
    }

    /// The ranks with the same `color` as a new group, ordered by `key` (then by rank), as
    /// MLX's `Group.split`. Backends may not support it (MLX's ring does not).
    pub fn split(&self, color: i32, key: i32) -> Result<Group> {
        if self.size == 1 {
            return Ok(self.clone());
        }
        Ok(Group::new(registered(self.id)?.comm.split(color, key)?))
    }
}

/// Makes `group` the default group of the collectives (what `nuvec-distributed`'s `init`
/// does).
pub fn set_default_group(group: Group) {
    *DEFAULT.write().unwrap_or_else(PoisonError::into_inner) = Some(group);
}

/// The default group: the one set by [`set_default_group`], else this process alone.
pub fn default_group() -> Group {
    DEFAULT
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
        .unwrap_or_else(Group::single)
}

fn group_or_default(group: Option<&Group>) -> Group {
    group.cloned().unwrap_or_else(default_group)
}

/// The size of the group with this id (1 for id 0).
pub(crate) fn group_size(id: usize) -> Result<usize> {
    match id {
        0 => Ok(1),
        id => Ok(registered(id)?.comm.size()),
    }
}

/// The rank of this process in the group with this id.
pub(crate) fn group_rank(id: usize) -> Result<usize> {
    match id {
        0 => Ok(0),
        id => Ok(registered(id)?.comm.rank()),
    }
}

fn record(op: CollectiveOp, group: &Group, inputs: Vec<Array>) -> Result<Array> {
    let mut out = Array::from_op(
        Primitive::Collective {
            op,
            group: group.id,
        },
        inputs,
    )?;
    Ok(out.swap_remove(0))
}

fn reduce(x: &Array, op: AllReduce, group: Option<&Group>) -> Result<Array> {
    let group = group_or_default(group);
    if group.size == 1 {
        return Ok(x.clone());
    }
    record(CollectiveOp::Reduce(op), &group, vec![x.clone()])
}

fn gather(x: &Array, group: Option<&Group>) -> Result<Array> {
    let group = group_or_default(group);
    if group.size == 1 {
        return Ok(x.clone());
    }
    record(CollectiveOp::Gather, &group, vec![x.clone()])
}

fn scatter(x: &Array, group: Option<&Group>) -> Result<Array> {
    let group = group_or_default(group);
    if group.size == 1 {
        return Ok(x.clone());
    }
    if x.dims().first().is_none_or(|d| d % group.size != 0) {
        return Err(Error::InvalidArgument(format!(
            "sum_scatter: shape {:?} is invalid for a group of size {}: the first axis must \
             exist and be divisible by the group size",
            x.dims(),
            group.size
        )));
    }
    record(CollectiveOp::SumScatter, &group, vec![x.clone()])
}

fn peer(op: &str, group: &Group, peer: usize) -> Result<()> {
    if group.size == 1 || peer >= group.size || peer == group.rank {
        return Err(Error::InvalidArgument(format!(
            "{op}: rank {peer} is not another rank of a group of {}",
            group.size
        )));
    }
    Ok(())
}

fn send_op(x: &Array, dst: usize, group: Option<&Group>) -> Result<Array> {
    let group = group_or_default(group);
    peer("send", &group, dst)?;
    record(CollectiveOp::Send { dst }, &group, vec![x.clone()])
}

fn recv_op(shape: Vec<usize>, dtype: DType, src: usize, group: Option<&Group>) -> Result<Array> {
    let group = group_or_default(group);
    peer("recv", &group, src)?;
    record(CollectiveOp::Recv { src, shape, dtype }, &group, Vec::new())
}

/// The fixed size of the header a `send` puts before its data.
const HEADER: usize = 512;

/// The header a `send` puts before its data, so a mismatched `recv` is an error rather
/// than reinterpreted bytes: a tag, the payload's length, the dtype's name length and name,
/// the rank and the dimensions, as little-endian `u64`s, zero-padded to [`HEADER`] bytes. One
/// fixed-size message, so a transfer costs two messages (header, payload).
fn header(dtype: DType, shape: &[usize]) -> Result<Vec<u8>> {
    let name = dtype.name().as_bytes();
    let n: usize = shape.iter().product();
    let mut h = Vec::with_capacity(HEADER);
    h.extend_from_slice(b"nuvecsnd");
    h.extend_from_slice(&((n * dtype.size()) as u64).to_le_bytes());
    h.extend_from_slice(&(name.len() as u64).to_le_bytes());
    h.extend_from_slice(name);
    h.extend_from_slice(&(shape.len() as u64).to_le_bytes());
    for &d in shape {
        h.extend_from_slice(&(d as u64).to_le_bytes());
    }
    if h.len() > HEADER {
        return Err(Error::InvalidArgument(format!(
            "send: {} dimensions do not fit the array header",
            shape.len()
        )));
    }
    h.resize(HEADER, 0);
    Ok(h)
}

/// The bytes of a collective's output (`operands`: its inputs).
fn output_len(op: &CollectiveOp, size: usize, operands: &[Operand<'_>]) -> usize {
    match op {
        CollectiveOp::Reduce(_) | CollectiveOp::Send { .. } => {
            operands[0].0.size * operands[0].0.dtype.size()
        }
        CollectiveOp::Gather => operands[0].0.size * operands[0].0.dtype.size() * size,
        CollectiveOp::SumScatter => operands[0].0.size * operands[0].0.dtype.size() / size,
        CollectiveOp::Recv { shape, dtype, .. } => shape.iter().product::<usize>() * dtype.size(),
    }
}

/// Runs a collective over `comm` on row-contiguous bytes: `input` is the input of
/// `dtype` and `shape` (`None` for `Reduce` and `Send`: `out` holds it already, and is
/// reduced or sent in place), `out` the output ([`output_len`] bytes).
fn eval_bytes(
    op: &CollectiveOp,
    comm: &dyn Communicator,
    dtype: DType,
    shape: &[usize],
    input: Option<&[u8]>,
    out: &mut [u8],
) -> Result<()> {
    let input_or_out = |out: &mut [u8]| {
        if let Some(input) = input {
            out.copy_from_slice(input);
        }
    };
    match op {
        CollectiveOp::Reduce(reduce) => {
            input_or_out(out);
            comm.all_reduce(*reduce, dtype, out)
        }
        CollectiveOp::Gather => comm.all_gather(input.expect("an input"), out),
        CollectiveOp::SumScatter => comm.sum_scatter(dtype, input.expect("an input"), out),
        CollectiveOp::Send { dst } => {
            input_or_out(out);
            comm.send(*dst, &header(dtype, shape)?)?;
            comm.send(*dst, out)
        }
        CollectiveOp::Recv { src, shape, dtype } => {
            let want = header(*dtype, shape)?;
            let mut got = vec![0u8; HEADER];
            comm.recv(*src, &mut got)?;
            if &got[..8] != b"nuvecsnd" {
                return Err(Error::InvalidArgument(format!(
                    "recv: rank {src} sent no array header"
                )));
            }
            if got != want {
                // The payload is read and dropped, so the connection stays aligned.
                let bytes = u64::from_le_bytes(got[8..16].try_into().expect("8 bytes")) as usize;
                let mut sink = vec![0u8; bytes.min(1 << 20)];
                let mut left = bytes;
                while left > 0 {
                    let n = left.min(sink.len());
                    comm.recv(*src, &mut sink[..n])?;
                    left -= n;
                }
                return Err(Error::InvalidArgument(format!(
                    "recv: rank {src} sent another dtype or shape than {} {shape:?}",
                    dtype.name()
                )));
            }
            comm.recv(*src, out)
        }
    }
}

/// [`eval_bytes`] on operands of any layout.
fn eval_into(
    op: &CollectiveOp,
    comm: &dyn Communicator,
    operands: &[Operand<'_>],
    out: &mut [u8],
) -> Result<()> {
    let Some(&(meta, data)) = operands.first() else {
        return eval_bytes(op, comm, DType::U8, &[], None, out);
    };
    let input = crate::kernels::contiguous(meta, data)?;
    eval_bytes(op, comm, meta.dtype, &meta.shape, Some(input.bytes()), out)
}

/// Evaluates a collective on the CPU: `operands` are the inputs' layouts and data. Runs
/// after every collective of the group already queued for its worker.
pub(crate) fn eval_cpu(
    op: &CollectiveOp,
    group: usize,
    operands: &[Operand<'_>],
) -> Result<Vec<Data>> {
    let entry = registered(group)?;
    let mut out = Buffer::try_alloc(output_len(op, entry.comm.size(), operands))?;
    let _order = entry.worker.idle_order(&entry.order);
    eval_into(op, &*entry.comm, operands, out.bytes_mut())?;
    Ok(vec![Data::new(out)])
}

/// What a GPU stream's collective waits for before it reads its input.
#[cfg(all(target_os = "macos", not(miri)))]
pub(crate) enum Ready {
    /// The input is in place already (or there is none).
    Now,
    /// The command buffer that copies (or orders) the input signals this value of its event.
    Signal(Arc<crate::gpu::BufferSignal>, u64),
    /// The committed command buffer producing the input completes.
    Pending(Option<nuvec_gpu::Pending>),
}

/// The input of a GPU stream's collective.
#[cfg(all(target_os = "macos", not(miri)))]
pub(crate) enum Input {
    /// None (`Recv`), or already in the output (`Reduce`, `Send`: copied there on the GPU,
    /// or the donated input itself).
    InOutput,
    /// A row-contiguous copy made on the GPU.
    Staged(Data),
    /// The operand as it is, read once [`Ready`].
    Operand(Array, Data),
}

/// A GPU stream's collective, as queued for its group's worker.
#[cfg(all(target_os = "macos", not(miri)))]
pub(crate) struct GpuJob {
    pub(crate) op: CollectiveOp,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
    pub(crate) ready: Ready,
    pub(crate) input: Input,
    pub(crate) out: Data,
    pub(crate) status: crate::array::ChunkStatus,
}

/// Queues `job` for its group's worker. The worker waits until the input is ready, runs the
/// collective into the output (even if the GPU work failed, so the peers stay paired),
/// records a failure in the job's status, and then, whatever happened, raises the returned
/// event to the returned value: GPU work that reads the output waits for it.
#[cfg(all(target_os = "macos", not(miri)))]
pub(crate) fn submit_gpu(
    group: usize,
    job: GpuJob,
    device: &nuvec_gpu::Device,
) -> Result<(Arc<dyn std::any::Any + Send + Sync>, u64)> {
    let entry = registered(group)?;
    entry.worker.event(device)?;
    let worker = entry.worker.clone();
    let value = worker.submit(move |value| {
        let GpuJob {
            op,
            dtype,
            shape,
            ready,
            input,
            out,
            status,
        } = job;
        let run = || {
            let ready = match ready {
                Ready::Now => Ok(()),
                Ready::Signal(signal, v) => signal.wait(v),
                Ready::Pending(p) => p.map_or(Ok(()), |p| {
                    p.wait_result().map_err(|e| Error::Backend(e.to_string()))
                }),
            };
            let _order = entry.order.lock().unwrap_or_else(PoisonError::into_inner);
            let ran = write_unsynchronized(&out, |bytes| match &input {
                Input::InOutput => eval_bytes(&op, &*entry.comm, dtype, &shape, None, bytes),
                Input::Staged(staged) => read_unsynchronized(staged, |i| {
                    eval_bytes(&op, &*entry.comm, dtype, &shape, Some(i), bytes)
                }),
                Input::Operand(a, d) => eval_into(&op, &*entry.comm, &[(a.out(), d)], bytes),
            });
            ready.and(ran)
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run))
            .unwrap_or_else(|_| Err(Error::Backend("distributed: a collective panicked".into())));
        if let Err(e) = result {
            let _ = status.set(e);
        }
        if let Some(event) = entry.worker.event.get() {
            event.signal(value);
        }
        drop((input, out));
    });
    Ok((worker, value))
}

/// The event of a worker that [`submit_gpu`] returned.
#[cfg(all(target_os = "macos", not(miri)))]
pub(crate) fn worker_event(worker: &(dyn std::any::Any + Send + Sync)) -> &nuvec_gpu::Event {
    worker
        .downcast_ref::<worker::Worker>()
        .and_then(|w| w.event.get())
        .expect("created by submit_gpu")
}

/// Runs `f` on the bytes of `staged`, which GPU work wrote before the input signal and no
/// GPU work writes again.
#[cfg(all(target_os = "macos", not(miri)))]
#[allow(unsafe_code)]
fn read_unsynchronized(staged: &Data, f: impl FnOnce(&[u8]) -> Result<()>) -> Result<()> {
    let len = staged.buffer.len();
    let Some(metal) = staged.buffer.metal() else {
        return f(staged.buffer.bytes());
    };
    let (ptr, available) = metal
        .unsynchronized_ptr()
        .map_err(|e| Error::Backend(e.to_string()))?;
    assert!(len <= available, "the staging buffer holds its bytes");
    // SAFETY: the GPU copy that wrote `staged` completed before the input signal the worker
    // waited for, nothing writes it afterwards (it belongs to this collective), and `ptr`
    // addresses `available >= len` bytes that `staged` keeps alive.
    f(unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) })
}

/// Runs `f` on `out`'s bytes while GPU work that reads them may already be submitted.
#[cfg(all(target_os = "macos", not(miri)))]
#[allow(unsafe_code)]
fn write_unsynchronized(out: &Data, f: impl FnOnce(&mut [u8]) -> Result<()>) -> Result<()> {
    let len = out.buffer.len();
    let Some(metal) = out.buffer.metal() else {
        return Err(Error::Backend(
            "distributed: a GPU collective's output is not on the GPU".into(),
        ));
    };
    let (ptr, available) = metal
        .unsynchronized_ptr()
        .map_err(|e| Error::Backend(e.to_string()))?;
    assert!(len <= available, "the output buffer holds its bytes");
    // SAFETY: `out` was allocated for this collective alone and is written only here; the
    // GPU work that reads it waits for the event `submit_gpu` raises after this write, and
    // the CPU reads it only once that work has completed and published it. `ptr` addresses
    // `available >= len` bytes of shared storage that `out` keeps alive.
    let bytes = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), len) };
    f(bytes)
}

/// The element-wise sum of `x` over the ranks of `group` (the default group if `None`), as
/// MLX's `all_sum`: every rank gets the same bits.
///
/// ```
/// use nuvec_core::{array, distributed};
///
/// // Without a configured group, the group is this process alone.
/// let x = array![1.0, 2.0];
/// assert_eq!(distributed::all_sum(&x, None).to_vec::<f32>().unwrap(), [1.0, 2.0]);
/// ```
#[track_caller]
pub fn all_sum(x: &Array, group: Option<&Group>) -> Array {
    Array::sticky("all_sum", Location::caller(), &[x], || {
        reduce(x, AllReduce::Sum, group)
    })
}

/// The element-wise maximum of `x` over the ranks of `group` (MLX's `all_max`).
#[track_caller]
pub fn all_max(x: &Array, group: Option<&Group>) -> Array {
    Array::sticky("all_max", Location::caller(), &[x], || {
        reduce(x, AllReduce::Max, group)
    })
}

/// The element-wise minimum of `x` over the ranks of `group` (MLX's `all_min`).
#[track_caller]
pub fn all_min(x: &Array, group: Option<&Group>) -> Array {
    Array::sticky("all_min", Location::caller(), &[x], || {
        reduce(x, AllReduce::Min, group)
    })
}

/// Every rank's `x` concatenated along the first axis in rank order (a scalar gathers to a
/// vector), as MLX's `all_gather`.
#[track_caller]
pub fn all_gather(x: &Array, group: Option<&Group>) -> Array {
    Array::sticky("all_gather", Location::caller(), &[x], || gather(x, group))
}

/// Rank `r` gets the sum over `group` of every rank's rows `r·k..(r+1)·k` of `x`, whose
/// first axis is `size·k`, as MLX's `sum_scatter` (the reduce-scatter half of an
/// all-reduce). A group of size 1 returns `x`.
///
/// ```
/// use nuvec_core::{array, distributed};
///
/// // Without a configured group, the group is this process alone.
/// let x = array![1.0, 2.0];
/// assert_eq!(distributed::sum_scatter(&x, None).to_vec::<f32>().unwrap(), [1.0, 2.0]);
/// ```
#[track_caller]
pub fn sum_scatter(x: &Array, group: Option<&Group>) -> Array {
    Array::sticky("sum_scatter", Location::caller(), &[x], || {
        scatter(x, group)
    })
}

/// Sends `x` to rank `dst` of `group` when the result (which is `x`) is evaluated, as
/// MLX's `send`. MLX's ring backend sends to its neighbours only.
#[track_caller]
pub fn send(x: &Array, dst: usize, group: Option<&Group>) -> Array {
    Array::sticky("send", Location::caller(), &[x], || send_op(x, dst, group))
}

/// An array of `shape` and `dtype` received from rank `src` of `group`, as MLX's `recv`. A
/// sender's other dtype or shape is an error when it is evaluated.
#[track_caller]
pub fn recv(shape: impl IntoShape, dtype: DType, src: usize, group: Option<&Group>) -> Array {
    let spec = shape.shape_spec();
    Array::constructor("recv", Location::caller(), dtype, &spec, |shape| {
        recv_op(shape.to_vec(), dtype, src, group)
    })
}

/// [`recv`] of `x`'s shape and dtype (MLX's `recv_like`).
#[track_caller]
pub fn recv_like(x: &Array, src: usize, group: Option<&Group>) -> Array {
    let at = Location::caller();
    Array::sticky("recv_like", at, &[x], || {
        recv_op(x.dims().to_vec(), x.dtype(), src, group)
    })
}

/// `x` unchanged, whose gradient is summed over `group` (MLX's `sum_gradients`): the input
/// of a layer whose output is sharded across the group, so each rank's gradient covers
/// every rank's slice of the output.
#[track_caller]
pub fn sum_gradients(x: &Array, group: Option<&Group>) -> Array {
    let group = group_or_default(group);
    if group.size == 1 {
        return x.clone();
    }
    Array::sticky("sum_gradients", Location::caller(), &[x], || {
        let f = crate::custom_function_flat(|xs| Ok(xs.to_vec())).with_vjp(
            move |_primals, _outputs, cotangents| {
                cotangents
                    .iter()
                    .map(|c| reduce(c, AllReduce::Sum, Some(&group)))
                    .collect()
            },
        );
        Ok(f.call(std::slice::from_ref(x))?.remove(0))
    })
}

impl crate::tree::Static for Group {
    fn hash_static(&self, mut h: &mut dyn std::hash::Hasher) {
        std::hash::Hash::hash(&(self.id, self.rank, self.size), &mut h);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;

    /// Rank 0 of two whose peer holds the same data (so a sum doubles it); logs every
    /// operation, slowly, and fails the reduction tagged with `fail_at` elements.
    struct Fake {
        log: Mutex<Vec<String>>,
        fail_at: Option<usize>,
    }

    impl Communicator for Fake {
        fn rank(&self) -> usize {
            0
        }
        fn size(&self) -> usize {
            2
        }
        fn all_reduce(&self, op: AllReduce, dtype: DType, data: &mut [u8]) -> Result<()> {
            std::thread::sleep(Duration::from_millis(2));
            let n = data.len() / dtype.size();
            self.log.lock().unwrap().push(format!("reduce {n}"));
            if self.fail_at == Some(n) {
                return Err(Error::Backend("distributed: rank 1 left".into()));
            }
            if op == AllReduce::Sum && dtype == DType::F32 {
                for v in bytemuck::cast_slice_mut::<u8, f32>(data) {
                    *v *= 2.0;
                }
            }
            Ok(())
        }
        fn all_gather(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
            output[..input.len()].copy_from_slice(input);
            output[input.len()..].copy_from_slice(input);
            Ok(())
        }
        fn send(&self, _: usize, _: &[u8]) -> Result<()> {
            Ok(())
        }
        fn recv(&self, _: usize, data: &mut [u8]) -> Result<()> {
            data.fill(0);
            Ok(())
        }
    }

    fn fake(fail_at: Option<usize>) -> (Group, Arc<Fake>) {
        let comm = Arc::new(Fake {
            log: Mutex::new(Vec::new()),
            fail_at,
        });
        (Group::new(comm.clone()), comm)
    }

    fn gpu() -> bool {
        if !crate::gpu::available() {
            return false;
        }
        crate::set_default_stream(crate::Stream::default_for(crate::Device::gpu(0)));
        true
    }

    /// Collectives run in the order they were issued, whichever path runs them: GPU
    /// stream (background) and CPU stream (inline, after the queued ones).
    #[test]
    fn collectives_run_in_program_order_across_streams() {
        if !gpu() {
            return;
        }
        let (g, comm) = fake(None);
        let x = |n: usize| crate::ops::ones([n as i32], DType::F32);
        let a = all_sum(&x(1).multiply(3.0), Some(&g));
        let b = all_sum(&x(2).multiply(3.0), Some(&g));
        crate::async_eval(&a).unwrap();
        crate::async_eval(&b).unwrap();
        crate::set_default_stream(crate::Stream::default_for(crate::Device::cpu()));
        let c = all_sum(&x(3), Some(&g));
        assert_eq!(c.to_vec::<f32>().unwrap(), [2.0; 3]);
        assert_eq!(a.to_vec::<f32>().unwrap(), [6.0]);
        assert_eq!(b.to_vec::<f32>().unwrap(), [6.0; 2]);
        assert_eq!(
            *comm.log.lock().unwrap(),
            ["reduce 1", "reduce 2", "reduce 3"]
        );
    }

    /// A result nobody reads still runs (the peers expect it), and later ones follow it.
    #[test]
    fn a_dropped_result_still_runs() {
        if !gpu() {
            return;
        }
        let (g, comm) = fake(None);
        let a = all_sum(&crate::ops::ones([4], DType::F32).multiply(2.0), Some(&g));
        crate::async_eval(&a).unwrap();
        drop(a);
        let b = all_sum(&crate::ops::ones([5], DType::F32).add(1.0), Some(&g));
        assert_eq!(b.to_vec::<f32>().unwrap(), [4.0; 5]);
        assert_eq!(*comm.log.lock().unwrap(), ["reduce 4", "reduce 5"]);
    }

    /// A failed collective fails its output and the GPU work after it with its own error,
    /// and nothing hangs.
    #[test]
    fn a_failed_collective_fails_what_follows() {
        if !gpu() {
            return;
        }
        let (g, _) = fake(Some(7));
        let y = all_sum(&crate::ops::ones([7], DType::F32).multiply(2.0), Some(&g)).add(1.0);
        let err = y.eval().unwrap_err().to_string();
        assert!(err.contains("rank 1 left"), "{err}");
    }

    /// A GPU failure before a collective fails its output (the collective itself still runs).
    #[cfg(all(target_os = "macos", not(miri)))]
    #[test]
    fn a_gpu_failure_before_a_collective_fails_it() {
        if !gpu() {
            return;
        }
        let (g, comm) = fake(None);
        let x = crate::ops::ones([6], DType::F32).multiply(2.0);
        crate::gpu::fault::fail_next_commit();
        let y = all_sum(&x, Some(&g)).add(1.0);
        assert!(y.eval().is_err());
        assert_eq!(*comm.log.lock().unwrap(), ["reduce 6"]);
    }

    /// Decode's shape commits as few command buffers as the same work without collectives
    /// (the dispatch policy's), not one per collective.
    #[cfg(all(target_os = "macos", not(miri)))]
    #[test]
    fn collectives_do_not_commit_command_buffers() {
        if !gpu() {
            return;
        }
        let (g, _) = fake(None);
        let x = crate::ops::ones([1, 1024], DType::F32);
        x.eval().unwrap();
        let chain = |x: &Array, collectives: bool| {
            let mut y = x.clone();
            for _ in 0..56 {
                y = y.multiply(0.5);
                if collectives {
                    y = all_sum(&y, Some(&g));
                }
            }
            y
        };
        // Warm both shapes (pipelines, the worker), then count each.
        chain(&x, true).eval().unwrap();
        chain(&x, false).eval().unwrap();
        let count = |collectives: bool| {
            let before = crate::gpu::COMMITS.get();
            let y = chain(&x, collectives);
            y.eval().unwrap();
            (crate::gpu::COMMITS.get() - before, y)
        };
        let (with, y) = count(true);
        let (without, _) = count(false);
        assert!(
            with <= without + 1,
            "{with} command buffers with collectives, {without} without"
        );
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0; 1024]);
    }

    /// A CPU-stream collective after a GPU-stream one in the same evaluation runs after it
    /// (the open command buffer is committed first, so nothing waits on unsubmitted work).
    #[test]
    fn a_cpu_collective_after_a_gpu_one_in_one_evaluation() {
        if !gpu() {
            return;
        }
        let (g, comm) = fake(None);
        let a = all_sum(&crate::ops::ones([3], DType::F32).multiply(3.0), Some(&g));
        crate::set_default_stream(crate::Stream::default_for(crate::Device::cpu()));
        let c = all_sum(&a.add(1.0), Some(&g));
        assert_eq!(c.to_vec::<f32>().unwrap(), [14.0; 3]);
        assert_eq!(*comm.log.lock().unwrap(), ["reduce 3", "reduce 3"]);
    }

    /// An all-reduce writes its input's buffer only when nothing else holds the input (MLX's
    /// donation): a held input keeps its values; an intermediate, or an evaluated array
    /// whose last handle is gone, is reduced in place.
    #[test]
    fn only_inputs_nothing_else_holds_are_reduced_in_place() {
        if !gpu() {
            return;
        }
        let (g, _comm) = fake(None);
        let held = crate::ops::ones([5], DType::F32).multiply(3.0);
        held.eval().unwrap();
        let sum = all_sum(&held, Some(&g));
        assert_eq!(sum.to_vec::<f32>().unwrap(), [6.0; 5]);
        assert_eq!(held.to_vec::<f32>().unwrap(), [3.0; 5]);

        let intermediate = all_sum(&crate::ops::ones([5], DType::F32).multiply(2.0), Some(&g));
        assert_eq!(intermediate.to_vec::<f32>().unwrap(), [4.0; 5]);

        let done = crate::ops::ones([5], DType::F32).multiply(5.0);
        done.eval().unwrap();
        let sum = all_sum(&done, Some(&g));
        drop(done);
        assert_eq!(sum.to_vec::<f32>().unwrap(), [10.0; 5]);

        // Read by another consumer in the same evaluation: not donated.
        let x = crate::ops::ones([5], DType::F32).multiply(4.0);
        let (sum, other) = (all_sum(&x, Some(&g)), x.add(1.0));
        drop(x);
        crate::eval(&[sum.clone(), other.clone()]).unwrap();
        assert_eq!(sum.to_vec::<f32>().unwrap(), [8.0; 5]);
        assert_eq!(other.to_vec::<f32>().unwrap(), [5.0; 5]);
    }

    /// A failed command buffer holding several collectives fails all of their outputs.
    #[cfg(all(target_os = "macos", not(miri)))]
    #[test]
    fn a_failed_buffer_fails_every_collective_in_it() {
        if !gpu() {
            return;
        }
        let (g, comm) = fake(None);
        let mut y = crate::ops::ones([8], DType::F32);
        for _ in 0..3 {
            y = all_sum(&y.multiply(2.0), Some(&g));
        }
        crate::gpu::fault::fail_next_commit();
        assert!(y.eval().is_err());
        assert_eq!(comm.log.lock().unwrap().len(), 3);
    }
}
