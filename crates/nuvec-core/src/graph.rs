//! The graph-level API: primitives, fused and custom nodes, and access to an array's node.
//!
//! This is what graph transformations such as `nuvec-compile` build on; everyday code
//! never needs it. It is not covered by the API's stability pins, and its functions return
//! `Result` (they are not array operations: recording a primitive validates it, see
//! [`from_primitive`]).

use crate::array::State;
use crate::{Array, DType, Result};

pub use crate::custom::CustomCall;
pub use crate::fast::FastOp;
pub use crate::fused::{Fused, FusedRef};
pub use crate::primitive::{
    ArgReduceOp, BinaryOp, Primitive, ReduceOp, ScanOp, ScatterReduce, UnaryOp,
};
pub use crate::shape::Flags;
pub use crate::view::SliceArg;

/// Records `primitive` applied to `inputs`, with the same validation as the operations,
/// and returns one array per output. Failure is sticky: if an input has failed, the
/// outputs fail with its error.
pub fn from_primitive(primitive: Primitive, inputs: Vec<Array>) -> Result<Vec<Array>> {
    Array::from_op(primitive, inputs)
}

/// A placeholder with this dtype and shape, standing for an input while a function is
/// traced (as in MLX): operations can be recorded on it, but evaluating anything that
/// depends on it fails. Placeholders are tracers.
pub fn placeholder(dtype: DType, shape: &[i32]) -> Result<Array> {
    let dims = shape
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                crate::Error::invalid_shape("placeholder", shape.iter().map(|&d| i64::from(d)))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Array::placeholder(dtype, &dims)
}

/// The arrays `a` is computed from, until it has been evaluated. Empty for host data and
/// evaluated arrays.
pub fn inputs(a: &Array) -> Vec<Array> {
    match &*a.node.lock() {
        State::Pending { inputs, .. } | State::Retained { inputs, .. } => inputs.clone(),
        State::Running { .. } | State::Done | State::Failed { .. } => Vec::new(),
    }
}

/// Every output of the operation that produces `a`, in order (`a` is the one at
/// [`output_index`]).
pub fn siblings(a: &Array) -> Vec<Array> {
    (0..a.node.outputs.len() as u32)
        .map(|output| Array {
            node: a.node.clone(),
            output,
        })
        .collect()
}

/// Which output of its operation `a` is.
pub fn output_index(a: &Array) -> usize {
    a.output as usize
}

/// Identifies the operation that produces `a` (shared by its siblings) for as long as the
/// array is alive.
pub fn id(a: &Array) -> usize {
    std::sync::Arc::as_ptr(&a.node) as usize
}

/// Whether `a` depends on a transform's tracer or a placeholder, that is, it is being
/// traced by `grad`, `vjp`, `jvp`, `vmap` or `compile`. Tracer status is transient: an
/// array is a tracer only while the outermost transform it depends on runs (on this
/// thread).
pub fn is_tracer(a: &Array) -> bool {
    a.node.is_tracer()
}

/// The operation that produces `a`, until it has been evaluated. `None` for host data and
/// evaluated arrays.
pub fn primitive(a: &Array) -> Option<Primitive> {
    match &*a.node.lock() {
        State::Pending { primitive, .. } => Some(primitive.clone()),
        _ => None,
    }
}

/// Per-dimension steps of `a`'s layout, in elements.
pub fn strides(a: &Array) -> &[isize] {
    &a.out().strides
}

/// Memory-layout flags of `a`.
pub fn flags(a: &Array) -> Flags {
    a.out().flags
}

/// `a[args]` with one [`SliceArg`] per dimension (Python semantics; a view), the lowered
/// form of basic indexing.
pub fn slice(a: &Array, args: &[SliceArg]) -> Result<Array> {
    crate::ops::imp::slice(a, args)
}
