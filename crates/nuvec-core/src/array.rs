//! The lazy array handle and the graph nodes behind it.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use smallvec::SmallVec;

use crate::buffer::Buffer;
use crate::primitive::Primitive;
use crate::shape::{self, Flags, Shape, Strides};
use crate::{DType, Device, Element, Error, Result, Stream, default_stream};

/// A lazily evaluated n-dimensional array.
///
/// Cloning is cheap: clones share the computation and its result. Operations record a
/// graph node and return immediately; values are computed on [`Array::eval`],
/// [`crate::eval`], or when read back.
#[derive(Clone)]
pub struct Array {
    pub(crate) node: Arc<Node>,
    pub(crate) output: u32,
}

/// One recorded computation, possibly with several outputs.
pub(crate) struct Node {
    pub(crate) stream: Stream,
    /// Whether the node depends on a transform's tracer (see `transforms`). Tracer nodes
    /// keep their graph when evaluated inside a transform.
    pub(crate) tracer: bool,
    pub(crate) outputs: SmallVec<[Output; 1]>,
    pub(crate) state: Mutex<State>,
}

/// Metadata and (once evaluated) data of one node output.
pub(crate) struct Output {
    pub(crate) dtype: DType,
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) flags: Flags,
    pub(crate) size: usize,
    /// Set exactly once, when the node is evaluated (or at creation for leaves).
    pub(crate) slot: OnceLock<Result<Data>>,
    /// GPU data whose command buffer is committed on the device timeline but may not have
    /// completed: later GPU work may read it before `slot` is set (D-032).
    pub(crate) inflight: OnceLock<Data>,
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
pub(crate) enum State {
    Pending {
        primitive: Primitive,
        inputs: Vec<Array>,
    },
    Retained {
        primitive: Primitive,
        inputs: Vec<Array>,
    },
    Done,
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
}

impl Output {
    pub(crate) fn new(
        dtype: DType,
        shape: Shape,
        strides: Option<Strides>,
        slot: OnceLock<Result<Data>>,
    ) -> Result<Self> {
        shape::checked_nbytes(&shape, dtype.size())?;
        let size = shape::checked_size(&shape)?;
        let strides = strides.unwrap_or_else(|| shape::row_contiguous_strides(&shape));
        let flags = shape::flags_for(&shape, &strides);
        Ok(Self {
            dtype,
            shape,
            strides,
            flags,
            size,
            slot,
            inflight: OnceLock::new(),
        })
    }
}

impl Array {
    // ── Construction ─────────────────────────────────────────────────────────

    /// An array holding `data` with the given `shape`.
    ///
    /// Fails with [`Error::SizeMismatch`] if `data.len()` is not the shape's element count.
    pub fn from_slice<T: Element>(data: &[T], shape: &[usize]) -> Result<Self> {
        let expected = shape::checked_size(shape)?;
        if data.len() != expected {
            return Err(Error::SizeMismatch {
                expected,
                actual: data.len(),
            });
        }
        let mut buffer = Buffer::try_alloc(shape::checked_nbytes(shape, T::DTYPE.size())?)?;
        T::write_bytes(data, buffer.bytes_mut());
        Self::leaf(Ok(Data::new(buffer)), T::DTYPE, shape.into())
    }

    /// An array of the given `shape` with every element set to `value`.
    pub fn full<T: Element>(shape: &[usize], value: T) -> Result<Self> {
        let itemsize = T::DTYPE.size();
        let mut buffer = Buffer::try_alloc(shape::checked_nbytes(shape, itemsize)?)?;
        let mut element = vec![0u8; itemsize];
        T::write_bytes(&[value], &mut element);
        for chunk in buffer.bytes_mut().chunks_exact_mut(itemsize) {
            chunk.copy_from_slice(&element);
        }
        Self::leaf(Ok(Data::new(buffer)), T::DTYPE, shape.into())
    }

    /// A zero-dimensional array holding `value`.
    pub fn scalar<T: Element>(value: T) -> Self {
        Self::from_slice(&[value], &[]).expect("a scalar always matches its shape")
    }

    /// An evaluated array over a row-contiguous `buffer`.
    pub(crate) fn from_buffer(buffer: Buffer, dtype: DType, shape: &[usize]) -> Result<Self> {
        Self::leaf(Ok(Data::new(buffer)), dtype, shape.into())
    }

    /// A node that is already evaluated, holding `data` (or a sticky error).
    pub(crate) fn leaf(data: Result<Data>, dtype: DType, shape: Shape) -> Result<Self> {
        let output = Output::new(dtype, shape, None, OnceLock::from(data))?;
        Ok(Self::from_node(Node {
            stream: default_stream(),
            tracer: false,
            outputs: SmallVec::from_buf([output]),
            state: Mutex::new(State::Done),
        }))
    }

    /// An array whose evaluation returns `error`. Used where an error cannot be returned
    /// directly, such as from operators.
    pub(crate) fn failed(error: Error, dtype: DType, shape: &[usize]) -> Self {
        Self::leaf(Err(error.clone()), dtype, shape.into())
            .unwrap_or_else(|_| Self::leaf(Err(error), dtype, Shape::new()).expect("scalar shape"))
    }

    /// Records `primitive` applied to `inputs` and returns one array per output.
    ///
    /// Inputs that already failed short-circuit with their error.
    pub(crate) fn from_op(primitive: Primitive, inputs: Vec<Array>) -> Result<Vec<Array>> {
        let tracer = inputs.iter().any(|a| a.node.tracer);
        Self::record(primitive, inputs, tracer)
    }

    /// [`Array::from_op`] with an explicit tracer flag.
    pub(crate) fn record(
        primitive: Primitive,
        inputs: Vec<Array>,
        tracer: bool,
    ) -> Result<Vec<Array>> {
        if let Some(err) = inputs.iter().find_map(Array::error) {
            return Err(err);
        }
        Self::record_unchecked(primitive, inputs, tracer)
    }

    /// [`Array::record`] without the failed-input check: the node fails only when
    /// evaluated. Used for `vmap` placeholders.
    pub(crate) fn record_unchecked(
        primitive: Primitive,
        inputs: Vec<Array>,
        tracer: bool,
    ) -> Result<Vec<Array>> {
        let outputs = primitive
            .output_specs(&inputs)?
            .into_iter()
            .map(|spec| Output::new(spec.dtype, spec.shape, spec.strides, OnceLock::new()))
            .collect::<Result<SmallVec<_>>>()?;
        let count = outputs.len() as u32;
        let node = Arc::new(Node {
            stream: default_stream(),
            tracer,
            outputs,
            state: Mutex::new(State::Pending { primitive, inputs }),
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

    /// Records `primitive` applied to `inputs`, with the same validation as the operations,
    /// and returns one array per output. Inputs that already failed short-circuit with their
    /// error.
    pub fn from_primitive(primitive: Primitive, inputs: Vec<Array>) -> Result<Vec<Array>> {
        Self::from_op(primitive, inputs)
    }

    /// A placeholder with this dtype and shape, standing for an input while a function is
    /// traced (as in MLX): operations can be recorded on it, but evaluating anything that
    /// depends on it fails. Placeholders are tracers.
    pub fn placeholder(dtype: DType, shape: &[usize]) -> Result<Array> {
        let failed = Self::failed(
            Error::InvalidArgument(
                "arrays cannot be evaluated while a function is traced (compile)".into(),
            ),
            dtype,
            &[],
        );
        let primitive = Primitive::Broadcast(shape.to_vec());
        Ok(Self::record_unchecked(primitive, vec![failed], true)?.swap_remove(0))
    }

    /// The arrays this one is computed from, until it has been evaluated. Empty for host
    /// data and evaluated arrays.
    pub fn inputs(&self) -> Vec<Array> {
        match &*self.node.lock() {
            State::Pending { inputs, .. } | State::Retained { inputs, .. } => inputs.clone(),
            State::Done => Vec::new(),
        }
    }

    /// Every output of the operation that produces this array, in order (this array is
    /// the one at [`Array::output_index`]).
    pub fn siblings(&self) -> Vec<Array> {
        (0..self.node.outputs.len() as u32)
            .map(|output| Self {
                node: self.node.clone(),
                output,
            })
            .collect()
    }

    /// Which output of its operation this array is.
    pub fn output_index(&self) -> usize {
        self.output as usize
    }

    /// Identifies the operation that produces this array (shared by its siblings) for as
    /// long as the array is alive.
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.node) as usize
    }

    /// Whether the array depends on a transform's tracer or a placeholder, that is, it is
    /// being traced by `grad`, `vjp`, `jvp`, `vmap` or `compile`.
    pub fn is_tracer(&self) -> bool {
        self.node.tracer
    }

    pub(crate) fn out(&self) -> &Output {
        &self.node.outputs[self.output as usize]
    }

    /// The error this array evaluated to, if it has already failed.
    pub(crate) fn error(&self) -> Option<Error> {
        match self.out().slot.get() {
            Some(Err(e)) => Some(e.clone()),
            _ => None,
        }
    }

    // ── Metadata (never evaluates) ───────────────────────────────────────────

    /// Dimension sizes.
    pub fn shape(&self) -> &[usize] {
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
    pub fn strides(&self) -> &[isize] {
        &self.out().strides
    }

    /// Memory-layout flags.
    pub fn flags(&self) -> Flags {
        self.out().flags
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

    /// The operation that produces this array, until it has been evaluated.
    /// `None` for host data and evaluated arrays.
    pub fn primitive(&self) -> Option<Primitive> {
        match &*self.node.lock() {
            State::Pending { primitive, .. } => Some(primitive.clone()),
            State::Retained { .. } | State::Done => None,
        }
    }

    // ── Evaluation and readback ──────────────────────────────────────────────

    /// Evaluates this array, if it has not been evaluated yet.
    pub fn eval(&self) -> Result<()> {
        crate::eval(std::slice::from_ref(self))
    }

    /// The elements in row-major order, evaluating first if needed.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                op: "to_vec",
                lhs: self.dtype(),
                rhs: T::DTYPE,
            });
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
        if self.size() != 1 {
            return Err(Error::InvalidArgument(format!(
                "item() needs exactly one element, the array has shape {:?}",
                self.shape()
            )));
        }
        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                op: "item",
                lhs: self.dtype(),
                rhs: T::DTYPE,
            });
        }
        Ok(self.to_vec::<T>()?[0])
    }

    /// The evaluated data, if evaluation succeeded.
    pub(crate) fn data(&self) -> Option<&Data> {
        self.out().slot.get().and_then(|r| r.as_ref().ok())
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = match self.out().slot.get() {
            None => "unevaluated",
            Some(Ok(_)) => "evaluated",
            Some(Err(_)) => "failed",
        };
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .field("status", &status)
            .finish()
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
        State::Done => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_accessors_describe_pending_nodes() {
        let a = Array::from_slice(&[1i32, 2], &[2]).unwrap();
        let b = Array::from_slice(&[5i32, 7], &[2]).unwrap();
        let outs = Array::from_primitive(Primitive::AddMultiply, vec![a.clone(), b]).unwrap();
        let (sum, product) = (&outs[0], &outs[1]);
        assert_eq!(sum.id(), product.id(), "siblings share their operation");
        assert_ne!(sum.id(), a.id());
        assert_eq!((sum.output_index(), product.output_index()), (0, 1));
        assert_eq!(product.siblings().len(), 2);
        assert_eq!(sum.inputs()[0].id(), a.id());
        assert!(a.inputs().is_empty(), "host data has no inputs");
        assert!(!sum.is_tracer());
        product.eval().unwrap();
        assert!(sum.inputs().is_empty(), "evaluated: detached");

        let p = Array::placeholder(DType::F32, &[3]).unwrap();
        let q = &p + 1.0f32;
        assert!(p.is_tracer() && q.is_tracer());
        assert_eq!(q.shape(), [3]);
        assert!(q.eval().is_err(), "placeholders cannot be evaluated");
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
        let a = Array::from_slice(&[1i32, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let c = crate::add(&a, &a).unwrap();
        assert!(!c.is_evaluated());
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.ndim(), 2);
        assert_eq!(c.size(), 6);
        assert_eq!(c.dtype(), DType::I32);
        assert_eq!(c.itemsize(), 4);
        assert_eq!(c.nbytes(), 24);
        assert_eq!(c.strides(), &[3, 1]);
        assert!(c.flags().row_contiguous);
        assert_eq!(c.device(), crate::default_device());
        assert_eq!(c.primitive(), Some(Primitive::Binary(crate::BinaryOp::Add)));
        assert!(!c.is_evaluated(), "metadata queries must not evaluate");
    }

    #[test]
    fn item_on_scalar_and_non_scalar() {
        assert_eq!(Array::scalar(7u16).item::<u16>().unwrap(), 7);
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(matches!(a.item::<f32>(), Err(Error::InvalidArgument(_))));
        assert!(matches!(
            Array::scalar(1.0f32).item::<f64>(),
            Err(Error::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn debug_does_not_evaluate() {
        let a = Array::scalar(1i8);
        let c = crate::add(&a, &a).unwrap();
        assert!(format!("{c:?}").contains("unevaluated"));
        assert!(!c.is_evaluated());
    }
}
