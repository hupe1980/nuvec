//! The lazy array handle and the graph nodes behind it.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::panic::Location;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::ThreadId;

use smallvec::SmallVec;

use crate::buffer::Buffer;
use crate::primitive::Primitive;
use crate::shape::{self, Flags, Shape, Strides};
use crate::transforms::TraceId;
use crate::{DType, Device, Element, Error, Result, Stream, default_stream};

/// A lazily evaluated n-dimensional array.
///
/// Cloning is cheap: clones share the computation and its result. Operations record a
/// graph node and return immediately; values are computed on [`Array::eval`],
/// [`crate::eval`], or when read back.
///
/// An operation that rejects its arguments returns a *failed* array instead of panicking:
/// it remembers the operation, the caller's source location and the reason, every array
/// computed from it fails with the same error, and the error surfaces as `Err` from
/// [`Array::check`], [`Array::eval`], [`Array::item`] or [`Array::to_vec`].
#[derive(Clone)]
#[must_use = "arrays are lazy: an unused result computes nothing (and may hold an error)"]
pub struct Array {
    pub(crate) node: Arc<Node>,
    pub(crate) output: u32,
}

/// One recorded computation, possibly with several outputs.
pub(crate) struct Node {
    pub(crate) stream: Stream,
    /// The outermost transform whose tracers the node depends on (see `transforms`). The
    /// node is a tracer while that transform runs, and then keeps its graph when evaluated.
    pub(crate) trace: Option<TraceId>,
    pub(crate) outputs: SmallVec<[Output; 1]>,
    pub(crate) state: Mutex<State>,
    /// Whether the node is a failed array: an operation rejected its arguments, or one of
    /// its inputs had failed, when it was recorded (see [`Array::is_failed`]).
    pub(crate) failed: bool,
}

/// Metadata and (once evaluated) data of one node output.
pub(crate) struct Output {
    pub(crate) dtype: DType,
    pub(crate) shape: Shape,
    /// `shape` as the public `i32` dimensions (validated to fit when the output is made).
    pub(crate) ishape: SmallVec<[i32; 4]>,
    pub(crate) strides: Strides,
    pub(crate) flags: Flags,
    pub(crate) size: usize,
    /// Set exactly once, when the node is evaluated (or at creation for leaves).
    pub(crate) slot: OnceLock<Result<Data>>,
    /// GPU data whose command buffer is committed on the device timeline but may not have
    /// completed: later GPU work (ordered after it on the device) may read it before `slot` is set.
    pub(crate) inflight: InFlight,
}

/// An output's in-flight data (see [`Output::inflight`]). It is published to the slot from
/// here on completion, unless a consumer took it over first (donation): the lock orders
/// the two.
#[derive(Default)]
pub(crate) struct InFlight {
    state: Mutex<InFlightState>,
    /// The failure cell of the command buffer producing the data: consumers in other
    /// evaluations that read the data while it is in flight fail with it.
    chunk: OnceLock<ChunkStatus>,
}

/// Set when a committed chunk of GPU work fails (see `gpu`).
pub(crate) type ChunkStatus = Arc<OnceLock<Error>>;

#[derive(Default)]
pub(crate) enum InFlightState {
    #[default]
    Unset,
    #[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
    Set(Data),
    /// Taken over by a consumer that writes into it in place; never published.
    Donated,
}

impl InFlight {
    pub(crate) fn lock(&self) -> MutexGuard<'_, InFlightState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Exposes `data` produced by the chunk whose failure cell is `chunk`.
    #[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
    pub(crate) fn set_from(&self, data: Data, chunk: &ChunkStatus) {
        let _ = self.chunk.set(chunk.clone());
        self.set(data);
    }

    /// The failure cell of the chunk producing the in-flight data, if known.
    #[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
    pub(crate) fn chunk(&self) -> Option<ChunkStatus> {
        self.chunk.get().cloned()
    }

    /// Exposes `data` (once; only GPU work has in-flight data).
    #[cfg_attr(not(all(target_os = "macos", not(miri))), allow(dead_code))]
    pub(crate) fn set(&self, data: Data) {
        let mut state = self.lock();
        if matches!(*state, InFlightState::Unset) {
            *state = InFlightState::Set(data);
        }
    }

    /// The data, if exposed and not donated.
    pub(crate) fn get(&self) -> Option<Data> {
        match &*self.lock() {
            InFlightState::Set(data) => Some(data.clone()),
            _ => None,
        }
    }

    /// Drops the data (the slot holds it now), unless donated.
    pub(crate) fn clear(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        if matches!(state, InFlightState::Set(_)) {
            *state = InFlightState::Unset;
        }
    }
}

/// Evaluated data: elements of a (possibly shared) buffer, starting at `offset` and laid
/// out by the output's strides. Views share their input's buffer.
#[derive(Clone)]
pub(crate) struct Data {
    pub(crate) buffer: Arc<Buffer>,
    /// Position of the first element, in elements.
    pub(crate) offset: isize,
}

impl Data {
    pub(crate) fn new(buffer: Buffer) -> Self {
        Self {
            buffer: Arc::new(buffer),
            offset: 0,
        }
    }
}

/// Whether a node still has to run. `Done` nodes have dropped their inputs (detach);
/// `Retained` nodes are evaluated but keep their graph for a running transform.
///
/// Transitions: `Pending → Running → Done | Retained | Failed`, and `Retained → Done` when
/// the transform that retained the node exits. The node's lock is held only for a
/// transition, never while the node computes or waits for another node.
pub(crate) enum State {
    Pending {
        primitive: Primitive,
        inputs: Vec<Array>,
    },
    /// One thread (`owner`) is computing the node; `done` is signalled once it has moved
    /// on to `Done`, `Retained` or `Failed` (its outputs set, or deferred in the owner's
    /// GPU work).
    Running {
        owner: ThreadId,
        done: Arc<Completion>,
    },
    Retained {
        primitive: Primitive,
        inputs: Vec<Array>,
    },
    Done,
    /// The evaluation failed; the outputs' slots hold `error`.
    Failed {
        error: Error,
    },
}

/// Signalled once when a running node's evaluation has finished.
#[derive(Default)]
pub(crate) struct Completion {
    done: Mutex<bool>,
    cv: Condvar,
}

impl Completion {
    /// Marks the evaluation finished and wakes every waiter.
    pub(crate) fn finish(&self) {
        *self.done.lock().unwrap_or_else(PoisonError::into_inner) = true;
        self.cv.notify_all();
    }

    /// Blocks until [`Completion::finish`] has been called.
    pub(crate) fn wait(&self) {
        let mut done = self.done.lock().unwrap_or_else(PoisonError::into_inner);
        while !*done {
            done = self.cv.wait(done).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// Hashes node addresses, the keys of the graph walks' sets and maps. Addresses are
/// unique and not attacker-controlled, so a multiplicative hash replaces SipHash.
#[derive(Default, Clone, Copy)]
pub(crate) struct NodeHasher(u64);

impl Hasher for NodeHasher {
    fn finish(&self) -> u64 {
        // Move the well-mixed high bits of the product to the low bits the table indexes
        // with.
        self.0.rotate_left(26)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// A set of nodes by address.
pub(crate) type NodeSet = HashSet<*const Node, BuildHasherDefault<NodeHasher>>;
/// A map keyed by node address (optionally with an output index).
pub(crate) type NodeMap<K, V> = HashMap<K, V, BuildHasherDefault<NodeHasher>>;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Array>();
};

impl Node {
    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether the transform this node depends on is running (see [`Array::is_tracer`]).
    pub(crate) fn is_tracer(&self) -> bool {
        self.trace.is_some_and(crate::transforms::is_live)
    }
}

impl Output {
    /// Why `shape` cannot describe an output of `dtype`.
    #[cold]
    fn invalid(shape: &[usize], dtype: DType) -> Error {
        if shape.iter().any(|&d| i32::try_from(d).is_err()) {
            Error::invalid_shape("shape", shape.iter().map(|&d| d as i64))
        } else {
            shape::checked_nbytes(shape, dtype.size())
                .err()
                .unwrap_or_else(|| Error::invalid_shape("shape", shape.iter().map(|&d| d as i64)))
        }
    }

    pub(crate) fn new(
        dtype: DType,
        shape: Shape,
        strides: Option<Strides>,
        slot: OnceLock<Result<Data>>,
    ) -> Result<Self> {
        // One pass: the element count (checked), the byte size (checked) and the public
        // `i32` dimensions (each must fit, as in MLX's `Shape`).
        let mut size = 1usize;
        let mut ishape = SmallVec::<[i32; 4]>::new();
        for &d in &shape {
            let (Some(n), Ok(di)) = (size.checked_mul(d), i32::try_from(d)) else {
                return Err(Self::invalid(&shape, dtype));
            };
            size = n;
            ishape.push(di);
        }
        if size
            .checked_mul(dtype.size())
            .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(Self::invalid(&shape, dtype));
        }
        let strides = strides.unwrap_or_else(|| shape::row_contiguous_strides(&shape));
        let flags = shape::flags_for(&shape, &strides);
        Ok(Self {
            dtype,
            shape,
            ishape,
            strides,
            flags,
            size,
            slot,
            inflight: InFlight::default(),
        })
    }
}

impl Array {
    // ── Construction ─────────────────────────────────────────────────────────

    /// An evaluated array over a row-contiguous `buffer`.
    pub(crate) fn from_buffer(buffer: Buffer, dtype: DType, shape: &[usize]) -> Result<Self> {
        Self::leaf(Ok(Data::new(buffer)), dtype, shape.into())
    }

    /// A node that is already evaluated, holding `data` (or a sticky error).
    pub(crate) fn leaf(data: Result<Data>, dtype: DType, shape: Shape) -> Result<Self> {
        let output = Output::new(dtype, shape, None, OnceLock::from(data))?;
        Ok(Self::from_node(Node {
            stream: default_stream(),
            trace: None,
            outputs: SmallVec::from_buf([output]),
            state: Mutex::new(State::Done),
            failed: false,
        }))
    }

    /// A failed array: a leaf whose evaluation returns `error`, with the given metadata
    /// (a 0-D shape if `shape` is not a valid one). It takes no buffer.
    #[cold]
    pub(crate) fn failed(error: Error, dtype: DType, shape: &[usize]) -> Self {
        let output = Output::new(
            dtype,
            shape.into(),
            None,
            OnceLock::from(Err(error.clone())),
        )
        .unwrap_or_else(|_| {
            Output::new(dtype, Shape::new(), None, OnceLock::from(Err(error)))
                .unwrap_or_else(|_| unreachable!("a 0-D output is always valid"))
        });
        Self::from_node(Node {
            stream: default_stream(),
            trace: None,
            outputs: SmallVec::from_buf([output]),
            state: Mutex::new(State::Done),
            failed: true,
        })
    }

    /// Runs the body `f` of the public operation `op`, called at `at` with the array
    /// operands `inputs`, and returns its array; on error, a failed array instead (the
    /// sticky error model). The error is the first failed input's (argument order), else
    /// `f`'s error located at `at` under the name `op`; the failed array takes the first
    /// operand's shape and dtype.
    #[inline]
    pub(crate) fn sticky(
        op: &'static str,
        at: &'static Location<'static>,
        inputs: &[&Array],
        f: impl FnOnce() -> Result<Array>,
    ) -> Array {
        match f() {
            Ok(out) => out,
            Err(error) => Self::failed_like(Self::sticky_error(op, at, inputs, error), inputs),
        }
    }

    /// [`Array::sticky`] for an operation with `N` outputs: on error, `N` failed arrays.
    #[inline]
    pub(crate) fn sticky_n<const N: usize>(
        op: &'static str,
        at: &'static Location<'static>,
        inputs: &[&Array],
        f: impl FnOnce() -> Result<[Array; N]>,
    ) -> [Array; N] {
        match f() {
            Ok(out) => out,
            Err(error) => {
                let error = Self::sticky_error(op, at, inputs, error);
                std::array::from_fn(|_| Self::failed_like(error.clone(), inputs))
            }
        }
    }

    /// [`Array::sticky`] for an operation with a variable number of outputs: on error,
    /// `count` failed arrays.
    #[inline]
    pub(crate) fn sticky_vec(
        op: &'static str,
        at: &'static Location<'static>,
        inputs: &[&Array],
        count: usize,
        f: impl FnOnce() -> Result<Vec<Array>>,
    ) -> Vec<Array> {
        match f() {
            Ok(out) => out,
            Err(error) => {
                let error = Self::sticky_error(op, at, inputs, error);
                (0..count)
                    .map(|_| Self::failed_like(error.clone(), inputs))
                    .collect()
            }
        }
    }

    /// The error a failed operation reports: the first failed input's, else `error` with
    /// the caller's location.
    #[cold]
    fn sticky_error(
        op: &'static str,
        at: &'static Location<'static>,
        inputs: &[&Array],
        error: Error,
    ) -> Error {
        inputs
            .iter()
            .find_map(|a| a.slot_error())
            .unwrap_or_else(|| error.locate(op, at))
    }

    /// A failed array with the first operand's shape and dtype (0-D float32 without one).
    #[cold]
    pub(crate) fn failed_like(error: Error, inputs: &[&Array]) -> Self {
        match inputs.first() {
            Some(a) => Self::failed(error, a.dtype(), a.dims()),
            None => Self::failed(error, DType::F32, &[]),
        }
    }

    /// Records `primitive` applied to `inputs` and returns one array per output.
    ///
    /// Inputs that already failed short-circuit with their error.
    pub(crate) fn from_op(primitive: Primitive, inputs: Vec<Array>) -> Result<Vec<Array>> {
        let trace = crate::transforms::inherited(&inputs);
        Self::record(primitive, inputs, trace)
    }

    /// [`Array::from_op`] with an explicit trace (`None`: not a tracer, whatever the
    /// inputs).
    ///
    /// Failure is sticky: if an input has failed, nothing is recorded and every output is
    /// a failed array carrying the first failed input's error, with the shape and dtype the
    /// operation would have produced (an error if those cannot be inferred).
    pub(crate) fn record(
        primitive: Primitive,
        inputs: Vec<Array>,
        trace: Option<TraceId>,
    ) -> Result<Vec<Array>> {
        if let Some(err) = inputs.iter().find_map(Array::slot_error) {
            return Self::record_failed(&primitive, &inputs, err);
        }
        Self::record_unchecked(primitive, inputs, trace)
    }

    #[cold]
    fn record_failed(primitive: &Primitive, inputs: &[Array], err: Error) -> Result<Vec<Array>> {
        let specs = primitive.output_specs(inputs).map_err(|_| err.clone())?;
        Ok(specs
            .into_iter()
            .map(|spec| Self::failed(err.clone(), spec.dtype, &spec.shape))
            .collect())
    }

    /// [`Array::record`] without the failed-input check: the node fails only when
    /// evaluated. Used for `vmap` placeholders.
    pub(crate) fn record_unchecked(
        primitive: Primitive,
        inputs: Vec<Array>,
        trace: Option<TraceId>,
    ) -> Result<Vec<Array>> {
        let outputs = primitive
            .output_specs(&inputs)?
            .into_iter()
            .map(|spec| Output::new(spec.dtype, spec.shape, spec.strides, OnceLock::new()))
            .collect::<Result<SmallVec<_>>>()?;
        let count = outputs.len() as u32;
        let node = Arc::new(Node {
            stream: default_stream(),
            trace,
            outputs,
            state: Mutex::new(State::Pending { primitive, inputs }),
            failed: false,
        });
        Ok((0..count)
            .map(|output| Self {
                node: node.clone(),
                output,
            })
            .collect())
    }

    fn from_node(node: Node) -> Self {
        Self {
            node: Arc::new(node),
            output: 0,
        }
    }

    // ── Graph (for graph transformations such as `compile`) ──────────────────

    /// A placeholder with this dtype and shape, standing for an input while a function is
    /// traced (as in MLX): operations can be recorded on it, but evaluating anything that
    /// depends on it fails. Placeholders are tracers.
    pub(crate) fn placeholder(dtype: DType, shape: &[usize]) -> Result<Array> {
        let failed = Self::failed(
            Error::InvalidArgument(
                "arrays cannot be evaluated while a function is traced (compile)".into(),
            ),
            dtype,
            &[],
        );
        let primitive = Primitive::Broadcast(shape.to_vec());
        let trace = Some(crate::transforms::PLACEHOLDER);
        Ok(Self::record_unchecked(primitive, vec![failed], trace)?.swap_remove(0))
    }

    pub(crate) fn out(&self) -> &Output {
        &self.node.outputs[self.output as usize]
    }

    /// The error this array evaluated to, if it has already failed.
    pub(crate) fn slot_error(&self) -> Option<Error> {
        match self.out().slot.get() {
            Some(Err(e)) => Some(e.clone()),
            _ => None,
        }
    }

    // ── Failure (record time, never evaluates) ───────────────────────────────

    /// Whether this is a failed array: the operation that produced it rejected its
    /// arguments, or one of its inputs had failed, when it was recorded. O(1); errors that
    /// happen during evaluation are not visible here (they surface from [`Array::eval`]).
    pub fn is_failed(&self) -> bool {
        self.node.failed
    }

    /// A failed array (0-D, float32) carrying the op error `kind`, located at the caller,
    /// under the name `op`: for functions built from operations that validate their own
    /// arguments and report failures the same way the operations do.
    ///
    /// ```
    /// use nuvec_core::{Array, OpErrorKind};
    ///
    /// let a = Array::failure("my_layer", OpErrorKind::InvalidArgument("wrong cache".into()));
    /// assert_eq!(a.check().unwrap_err().op_name(), Some("my_layer"));
    /// ```
    #[track_caller]
    pub fn failure(op: &'static str, kind: crate::OpErrorKind) -> Array {
        Self::failed(
            Error::op(op, kind).locate(op, Location::caller()),
            DType::F32,
            &[],
        )
    }

    /// The error of a failed array (see [`Array::is_failed`]), without evaluating.
    pub fn error(&self) -> Option<&Error> {
        match (self.node.failed, self.out().slot.get()) {
            (true, Some(Err(e))) => Some(e),
            _ => None,
        }
    }

    /// `Ok(self)`, or the error of a failed array (see [`Array::is_failed`]). Never
    /// evaluates: use it before host decisions that depend on shapes, and at the boundary
    /// of functions that return `Result`.
    pub fn check(self) -> Result<Array> {
        if self.node.failed
            && let Some(error) = self.slot_error()
        {
            return Err(error);
        }
        Ok(self)
    }

    // ── Metadata (never evaluates) ───────────────────────────────────────────

    /// Dimension sizes (MLX's `Shape`: `i32`).
    ///
    /// A failed array keeps a best-effort shape: the one the operation would have
    /// produced if that can be inferred, else its first operand's. Call
    /// [`Array::check`] first when host logic depends on the shape.
    pub fn shape(&self) -> &[i32] {
        &self.out().ishape
    }

    /// The shape as an array of `N` sizes, for destructuring and host decisions
    /// (`let [b, l, d] = x.shape_n::<3>()?;`): the array's error if it is a failed array
    /// (whose shape is only a best guess), or an error if it does not have `N` dimensions.
    ///
    /// ```
    /// use nuvec_core::array;
    ///
    /// let x = array![[1, 2, 3], [4, 5, 6]];
    /// let [rows, cols] = x.shape_n::<2>().unwrap();
    /// assert_eq!((rows, cols), (2, 3));
    /// assert!(x.shape_n::<3>().is_err());
    /// ```
    #[track_caller]
    pub fn shape_n<const N: usize>(&self) -> Result<[i32; N]> {
        if let Some(e) = self.error() {
            return Err(e.clone());
        }
        <[i32; N]>::try_from(self.shape()).map_err(|_| {
            Error::InvalidArgument(format!(
                "shape_n: expected {N} dimensions, the array has shape {:?}",
                self.shape()
            ))
            .locate("shape_n", Location::caller())
        })
    }

    /// Dimension sizes as `usize`, for nuvec's internals.
    pub(crate) fn dims(&self) -> &[usize] {
        &self.out().shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.out().shape.len()
    }

    /// Number of elements.
    pub fn size(&self) -> usize {
        self.out().size
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.out().dtype
    }

    /// Size of one element in bytes.
    pub fn itemsize(&self) -> usize {
        self.dtype().size()
    }

    /// Size of all elements in bytes.
    pub fn nbytes(&self) -> usize {
        self.size() * self.itemsize()
    }

    /// Per-dimension steps, in elements.
    pub(crate) fn flags(&self) -> Flags {
        self.out().flags
    }

    /// Per-dimension steps, in elements.
    pub(crate) fn strides(&self) -> &[isize] {
        &self.out().strides
    }

    /// The stream the producing operation was recorded on.
    pub fn stream(&self) -> Stream {
        self.node.stream
    }

    /// The device the producing operation runs on.
    pub fn device(&self) -> Device {
        self.node.stream.device
    }

    /// Whether the array has been evaluated (successfully or not).
    pub fn is_evaluated(&self) -> bool {
        self.out().slot.get().is_some()
    }

    // ── Evaluation and readback ──────────────────────────────────────────────

    /// Evaluates this array, if it has not been evaluated yet.
    pub fn eval(&self) -> Result<()> {
        crate::eval(std::slice::from_ref(self))
    }

    /// The elements in row-major order, evaluating first if needed.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        if let Some(e) = self.error() {
            return Err(e.clone());
        }
        if T::DTYPE != self.dtype() {
            return Err(Error::dtype_mismatch("to_vec", self.dtype(), T::DTYPE));
        }
        self.eval()?;
        let data = self.data().expect("eval succeeded, so data is set");
        let out = self.out();
        if out.size == 0 {
            Ok(Vec::new())
        } else if out.flags.row_contiguous || out.size == 1 {
            let start = data.offset.max(0) as usize * out.dtype.size();
            Ok(T::read_bytes(
                &data.buffer.bytes()[start..start + self.nbytes()],
            ))
        } else {
            let gathered = crate::kernels::contiguous(out, data)?;
            Ok(T::read_bytes(gathered.bytes()))
        }
    }

    /// The single element of a one-element array, evaluating first if needed.
    pub fn item<T: Element>(&self) -> Result<T> {
        if let Some(e) = self.error() {
            return Err(e.clone());
        }
        if self.size() != 1 {
            return Err(Error::InvalidArgument(format!(
                "item() needs exactly one element, the array has shape {:?}",
                self.dims()
            )));
        }
        if T::DTYPE != self.dtype() {
            return Err(Error::dtype_mismatch("item", self.dtype(), T::DTYPE));
        }
        Ok(self.to_vec::<T>()?[0])
    }

    /// The evaluated data, if evaluation succeeded.
    pub(crate) fn data(&self) -> Option<&Data> {
        self.out().slot.get().and_then(|r| r.as_ref().ok())
    }
}

/// Never evaluates: shape, dtype, device and status; the values only when the array is
/// already evaluated, the error of a failed array.
impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Array");
        d.field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("device", &self.device());
        match self.out().slot.get() {
            None => d.field("status", &"unevaluated"),
            Some(Ok(_)) => match crate::display::values(self) {
                Ok(values) => d.field("values", &format_args!("{values}")),
                Err(_) => d.field("status", &"evaluated"),
            },
            Some(Err(e)) if self.node.failed => d.field("failed", &format_args!("{e}")),
            Some(Err(e)) => d.field("error", &format_args!("{e}")),
        };
        d.finish()
    }
}

/// Evaluates and prints the values as MLX and NumPy do: `array([[1, 2], [3, 4]], dtype=int32)`
/// (more than 1,000 elements are summarised with the first and last 3 along each axis), or
/// `<failed: …>` with the error.
impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match crate::display::values(self) {
            Ok(values) => write!(f, "array({values}, dtype={})", self.dtype()),
            Err(e) => write!(f, "<failed: {e}>"),
        }
    }
}

/// A cheap placeholder: a 0-D float32 zero. Handy with [`std::mem::take`] to update an array
/// held in a struct by value, so the update can write in place:
/// `self.k = std::mem::take(&mut self.k).at(..).set(&v);`.
impl Default for Array {
    fn default() -> Self {
        crate::creation::from_data(&[0.0f32], &[])
            .unwrap_or_else(|e| Self::failed(e, DType::F32, &[]))
    }
}

/// Drops unevaluated graphs iteratively. Recursive destruction would overflow the stack
/// on long chains: each node's last reference would drop its inputs from inside its own
/// destructor.
impl Drop for Node {
    fn drop(&mut self) {
        let mut stack = take_inputs(self);
        while let Some(Array { node, .. }) = stack.pop() {
            if let Some(mut node) = Arc::into_inner(node) {
                stack.append(&mut take_inputs(&mut node));
            }
        }
    }
}

fn take_inputs(node: &mut Node) -> Vec<Array> {
    let state = node.state.get_mut().unwrap_or_else(PoisonError::into_inner);
    match std::mem::replace(state, State::Done) {
        State::Pending { inputs, .. } | State::Retained { inputs, .. } => inputs,
        State::Running { .. } | State::Done | State::Failed { .. } => Vec::new(),
    }
}

/// Graph accessors as methods, for the unit tests.
#[cfg(test)]
impl Array {
    pub(crate) fn primitive_t(&self) -> Option<Primitive> {
        crate::graph::primitive(self)
    }
    pub(crate) fn inputs_t(&self) -> Vec<Array> {
        crate::graph::inputs(self)
    }
    pub(crate) fn id_t(&self) -> usize {
        crate::graph::id(self)
    }
    pub(crate) fn siblings_t(&self) -> Vec<Array> {
        crate::graph::siblings(self)
    }
    pub(crate) fn output_index_t(&self) -> usize {
        crate::graph::output_index(self)
    }
    pub(crate) fn is_tracer_t(&self) -> bool {
        crate::graph::is_tracer(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OpErrorKind;

    #[test]
    fn graph_accessors_describe_pending_nodes() {
        let a = Array::from_slice(&[1i32, 2], [2]);
        let b = Array::from_slice(&[5i32, 7], [2]);
        let outs =
            crate::graph::from_primitive(Primitive::AddMultiply, vec![a.clone(), b]).unwrap();
        let (sum, product) = (&outs[0], &outs[1]);
        assert_eq!(sum.id_t(), product.id_t(), "siblings share their operation");
        assert_ne!(sum.id_t(), a.id_t());
        assert_eq!((sum.output_index_t(), product.output_index_t()), (0, 1));
        assert_eq!(product.siblings_t().len(), 2);
        assert_eq!(sum.inputs_t()[0].id_t(), a.id_t());
        assert!(a.inputs_t().is_empty(), "host data has no inputs");
        assert!(!sum.is_tracer_t());
        product.eval().unwrap();
        assert!(sum.inputs_t().is_empty(), "evaluated: detached");

        let p = Array::placeholder(DType::F32, &[3]).unwrap();
        let q = &p + 1.0f32;
        assert!(p.is_tracer_t() && q.is_tracer_t());
        assert_eq!(q.dims(), [3]);
        assert!(q.eval().is_err(), "placeholders cannot be evaluated");
    }

    #[test]
    fn failed_leaves_are_sticky_and_typed() {
        let error = Error::size_mismatch("from_slice", 6, 5);
        let f = Array::failed(error.clone(), DType::F16, &[2, 3]);
        assert!(f.is_failed());
        assert_eq!(f.shape(), [2, 3]);
        assert_eq!(f.dtype(), DType::F16);
        assert_eq!(f.eval(), Err(error.clone()));
        assert_eq!(f.clone().check().unwrap_err(), error);
        // Recording on a failed input records nothing and keeps the inferred metadata.
        let g = crate::ops::imp::reshape(&f, &[3, 2]).unwrap();
        assert!(g.is_failed() && g.primitive_t().is_none());
        assert_eq!(g.shape(), [3, 2]);
        assert_eq!(g.eval(), Err(error));
        // A valid array is not failed, and `check` does not evaluate it.
        let ok = &Array::from(1.0f32) + 1.0;
        assert!(!ok.is_failed());
        let ok = ok.check().unwrap();
        assert!(!ok.is_evaluated());
    }

    #[test]
    fn shapes_are_i32_and_bounded() {
        let a = crate::ops::imp::zeros(&[2, 3, 4], DType::U8).unwrap();
        let shape: &[i32] = a.shape();
        assert_eq!(shape, [2, 3, 4]);
        assert_eq!(a.shape_n::<3>().unwrap(), [2, 3, 4]);
        assert!(a.shape_n::<2>().is_err(), "rank mismatch");
        // A single dimension above i32::MAX is an invalid shape, never a wrap.
        let big = crate::ops::imp::broadcast_to(&Array::from(0u8), &[1usize << 31]).unwrap_err();
        assert!(
            matches!(big.kind(), Some(OpErrorKind::InvalidShape(_))),
            "{big}"
        );
        let fits = crate::ops::imp::broadcast_to(&Array::from(0u8), &[i32::MAX as usize]).unwrap();
        assert_eq!(fits.shape(), [i32::MAX]);
    }

    #[test]
    fn node_hasher_spreads_aligned_addresses() {
        use std::hash::BuildHasher;
        let build = BuildHasherDefault::<NodeHasher>::default();
        // 336-byte nodes, as allocated: every low table index should be reachable.
        let buckets: HashSet<u64> = (0..1024u64)
            .map(|i| {
                build.hash_one(std::ptr::without_provenance::<Node>(
                    0x6000_0000_0000 + i as usize * 336,
                )) & 1023
            })
            .collect();
        // A random hash fills about 1 - 1/e ≈ 63% of them.
        assert!(
            buckets.len() > 600,
            "only {} of 1024 buckets used",
            buckets.len()
        );
    }

    #[test]
    fn metadata_without_eval() {
        let a = Array::from_slice(&[1i32, 2, 3, 4, 5, 6], [2, 3]);
        let c = crate::ops::imp::add(&a, &a).unwrap();
        assert!(!c.is_evaluated());
        assert_eq!(c.dims(), &[2, 3]);
        assert_eq!(c.ndim(), 2);
        assert_eq!(c.size(), 6);
        assert_eq!(c.dtype(), DType::I32);
        assert_eq!(c.itemsize(), 4);
        assert_eq!(c.nbytes(), 24);
        assert_eq!(c.strides(), &[3, 1]);
        assert!(c.flags().row_contiguous);
        assert_eq!(c.device(), crate::default_device());
        assert_eq!(
            c.primitive_t(),
            Some(Primitive::Binary(crate::graph::BinaryOp::Add))
        );
        assert!(!c.is_evaluated(), "metadata queries must not evaluate");
    }

    #[test]
    fn item_on_scalar_and_non_scalar() {
        assert_eq!(Array::from(7u16).item::<u16>().unwrap(), 7);
        let a = Array::from_slice(&[1.0f32, 2.0], [2]);
        assert!(matches!(a.item::<f32>(), Err(Error::InvalidArgument(_))));
        assert!(matches!(
            Array::from(1.0f32).item::<f64>(),
            Err(Error::Op {
                kind: OpErrorKind::DTypeMismatch { .. },
                ..
            })
        ));
    }

    #[test]
    fn debug_does_not_evaluate() {
        let a = Array::from(1i8);
        let c = crate::ops::imp::add(&a, &a).unwrap();
        assert!(format!("{c:?}").contains("unevaluated"));
        assert!(!c.is_evaluated());
    }
}
