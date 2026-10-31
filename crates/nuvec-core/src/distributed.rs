//! Distributed groups and collective operations, as MLX's `mx.distributed`.
//!
//! A [`Group`] is a set of processes with a rank and a size. Collectives (`all_sum`,
//! `all_max`, `all_min`, `all_gather`, `send`, `recv`) are lazy array operations over a
//! group: they run when their result is evaluated, compose with `grad`, `vmap` and
//! `compile` as MLX defines, and run on the CPU over the arrays' shared memory (on a GPU
//! stream they wait for the GPU work producing their inputs). Without a configured group,
//! the default group has size 1 and every collective is the identity, so single-process
//! code runs unchanged.
//!
//! The transport is a [`Communicator`]; `nuvec-distributed` provides MLX's TCP ring and
//! installs it as the default group.

use std::panic::Location;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use crate::array::Data;
use crate::buffer::Buffer;
use crate::kernels::Operand;
use crate::primitive::Primitive;
use crate::{Array, DType, Error, IntoShape, Result};

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
}

/// Groups by id (`id - 1`); id 0 is the size-1 group.
static GROUPS: Mutex<Vec<Arc<Registered>>> = Mutex::new(Vec::new());
static DEFAULT: RwLock<Option<Group>> = RwLock::new(None);

fn registered(id: usize) -> Result<Arc<Registered>> {
    let groups = GROUPS.lock().unwrap_or_else(PoisonError::into_inner);
    id.checked_sub(1)
        .and_then(|i| groups.get(i).cloned())
        .ok_or_else(|| Error::InvalidArgument(format!("distributed: no group {id}")))
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
        groups.push(Arc::new(Registered {
            comm,
            order: Mutex::new(()),
        }));
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

/// The header a `send` puts before its data, so a mismatched `recv` is an error rather
/// than reinterpreted bytes: a tag, the dtype's name length and name, the rank and the
/// dimensions, as little-endian `u64`s.
fn header(dtype: DType, shape: &[usize]) -> Vec<u8> {
    let name = dtype.name().as_bytes();
    let mut h = Vec::with_capacity(32 + name.len() + 8 * shape.len());
    h.extend_from_slice(b"nuvecsnd");
    h.extend_from_slice(&(name.len() as u64).to_le_bytes());
    h.extend_from_slice(name);
    h.extend_from_slice(&(shape.len() as u64).to_le_bytes());
    for &d in shape {
        h.extend_from_slice(&(d as u64).to_le_bytes());
    }
    h
}

/// Evaluates a collective on the CPU: `operands` are the inputs' layouts and data.
pub(crate) fn eval_cpu(
    op: &CollectiveOp,
    group: usize,
    operands: &[Operand<'_>],
) -> Result<Vec<Data>> {
    let entry = registered(group)?;
    let _order = entry.order.lock().unwrap_or_else(PoisonError::into_inner);
    let comm = &entry.comm;
    let data = match op {
        CollectiveOp::Reduce(reduce) => {
            let (meta, data) = operands[0];
            let mut out = crate::kernels::contiguous(meta, data)?;
            comm.all_reduce(*reduce, meta.dtype, out.bytes_mut())?;
            out
        }
        CollectiveOp::Gather => {
            let (meta, data) = operands[0];
            let input = crate::kernels::contiguous(meta, data)?;
            let mut out = Buffer::try_alloc(input.bytes().len() * comm.size())?;
            comm.all_gather(input.bytes(), out.bytes_mut())?;
            out
        }
        CollectiveOp::Send { dst } => {
            let (meta, data) = operands[0];
            let input = crate::kernels::contiguous(meta, data)?;
            let h = header(meta.dtype, &meta.shape);
            comm.send(*dst, &(h.len() as u64).to_le_bytes())?;
            comm.send(*dst, &h)?;
            comm.send(*dst, &(input.bytes().len() as u64).to_le_bytes())?;
            comm.send(*dst, input.bytes())?;
            input
        }
        CollectiveOp::Recv { src, shape, dtype } => {
            let want = header(*dtype, shape);
            let mut len = [0u8; 8];
            comm.recv(*src, &mut len)?;
            let len = u64::from_le_bytes(len) as usize;
            if len > (1 << 20) {
                return Err(Error::InvalidArgument(format!(
                    "recv: rank {src} sent no array header"
                )));
            }
            let mut got = vec![0u8; len];
            comm.recv(*src, &mut got)?;
            let mut bytes = [0u8; 8];
            comm.recv(*src, &mut bytes)?;
            let bytes = u64::from_le_bytes(bytes) as usize;
            if got != want {
                // The payload is read and dropped, so the connection stays aligned.
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
            let n: usize = shape.iter().product();
            let mut out = Buffer::try_alloc(n * dtype.size())?;
            comm.recv(*src, out.bytes_mut())?;
            out
        }
    };
    Ok(vec![Data::new(data)])
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
