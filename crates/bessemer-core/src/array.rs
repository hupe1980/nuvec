//! The lazy array handle and the graph nodes behind it.

use std::fmt;
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
    pub(crate) slot: OnceLock<Result<Buffer>>,
}

/// Whether a node still has to run. `Done` nodes have dropped their inputs (detach).
pub(crate) enum State {
    Pending {
        primitive: Primitive,
        inputs: Vec<Array>,
    },
    Done,
}

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
    fn new(dtype: DType, shape: Shape, slot: OnceLock<Result<Buffer>>) -> Result<Self> {
        shape::checked_nbytes(&shape, dtype.size())?;
        let size = shape::checked_size(&shape)?;
        let strides = shape::row_contiguous_strides(&shape);
        let flags = shape::flags_for(&shape, &strides);
        Ok(Self {
            dtype,
            shape,
            strides,
            flags,
            size,
            slot,
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
        let mut buffer = Buffer::try_zeroed(shape::checked_nbytes(shape, T::DTYPE.size())?)?;
        T::write_bytes(data, buffer.bytes_mut());
        Self::leaf(Ok(buffer), T::DTYPE, shape.into())
    }

    /// An array of the given `shape` with every element set to `value`.
    pub fn full<T: Element>(shape: &[usize], value: T) -> Result<Self> {
        let itemsize = T::DTYPE.size();
        let mut buffer = Buffer::try_zeroed(shape::checked_nbytes(shape, itemsize)?)?;
        let mut element = vec![0u8; itemsize];
        T::write_bytes(&[value], &mut element);
        for chunk in buffer.bytes_mut().chunks_exact_mut(itemsize) {
            chunk.copy_from_slice(&element);
        }
        Self::leaf(Ok(buffer), T::DTYPE, shape.into())
    }

    /// A zero-dimensional array holding `value`.
    pub fn scalar<T: Element>(value: T) -> Self {
        Self::from_slice(&[value], &[]).expect("a scalar always matches its shape")
    }

    /// A node that is already evaluated, holding `data` (or a sticky error).
    fn leaf(data: Result<Buffer>, dtype: DType, shape: Shape) -> Result<Self> {
        let output = Output::new(dtype, shape, OnceLock::from(data))?;
        Ok(Self::from_node(Node {
            stream: default_stream(),
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
        if let Some(err) = inputs.iter().find_map(Array::error) {
            return Err(err);
        }
        let outputs = primitive
            .output_specs(&inputs)?
            .into_iter()
            .map(|spec| Output::new(spec.dtype, spec.shape, OnceLock::new()))
            .collect::<Result<SmallVec<_>>>()?;
        let count = outputs.len() as u32;
        let node = Arc::new(Node {
            stream: default_stream(),
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

    pub(crate) fn out(&self) -> &Output {
        &self.node.outputs[self.output as usize]
    }

    /// The error this array evaluated to, if it has already failed.
    fn error(&self) -> Option<Error> {
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
            State::Done => None,
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
        let buffer = self.data().expect("eval succeeded, so data is set");
        Ok(T::read_bytes(&buffer.bytes()[..self.nbytes()]))
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

    /// The evaluated buffer, if evaluation succeeded.
    pub(crate) fn data(&self) -> Option<&Buffer> {
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
        State::Pending { inputs, .. } => inputs,
        State::Done => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(c.device(), Device::cpu());
        assert_eq!(c.primitive(), Some(Primitive::Add));
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
