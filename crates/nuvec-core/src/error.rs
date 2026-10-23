//! The error type: located op errors (record time) and boundary errors (evaluation, I/O).

use std::fmt;
use std::panic::Location;

use crate::{DType, Device};

/// Why an operation failed.
///
/// Array operations never return this directly: an invalid call yields a *failed array*
/// that carries an [`Error::Op`] (the op's name, the caller's source location and the
/// reason), and every array computed from it carries the same error. It surfaces as `Err`
/// at the boundary: [`crate::Array::check`], [`crate::Array::eval`],
/// [`crate::Array::item`], [`crate::Array::to_vec`], I/O and transforms. Errors that
/// happen while evaluating (a GPU failure, memory exhausted) use the other variants.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// An array operation rejected its arguments when it was recorded.
    Op {
        /// The public operation the caller invoked (MLX's name).
        op: &'static str,
        /// Where the caller invoked it.
        at: &'static Location<'static>,
        /// Why it failed.
        kind: OpErrorKind,
    },
    /// Memory for this many bytes could not be allocated (or exceeds the memory limit).
    OutOfMemory(usize),
    /// The GPU backend failed (for example, a command buffer error).
    Backend(String),
    /// No backend is available for the device (the GPU without Metal).
    UnsupportedDevice(Device),
    /// Reading or writing a file failed.
    Io(String),
    /// An argument of a boundary function (transforms, loaders) is invalid.
    InvalidArgument(String),
}

/// The reason an operation rejected its arguments (see [`Error::Op`]).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpErrorKind {
    /// Operand shapes are incompatible.
    ShapeMismatch {
        /// Shape of the left operand (or of the array).
        lhs: Vec<i32>,
        /// Shape of the right operand (or the requested shape).
        rhs: Vec<i32>,
    },
    /// Operand element types are incompatible.
    DTypeMismatch {
        /// Element type of the left operand (or of the array).
        lhs: DType,
        /// Element type of the right operand (or the requested type).
        rhs: DType,
    },
    /// The number of host values does not match the shape.
    SizeMismatch {
        /// Element count implied by the shape.
        expected: usize,
        /// Number of values provided.
        actual: usize,
    },
    /// The operation does not support this element type.
    UnsupportedDType(DType),
    /// An axis is out of range, repeated, or more than one where one is expected.
    InvalidAxis {
        /// The axis as given.
        axis: i32,
        /// Dimensions of the array.
        ndim: usize,
    },
    /// A shape is invalid: a negative or too large dimension, two `-1`, or an element
    /// count that does not match.
    InvalidShape(Vec<i64>),
    /// An index is out of bounds or malformed.
    InvalidIndex(String),
    /// Host memory for this many bytes could not be allocated.
    OutOfMemory(usize),
    /// An argument is invalid for another reason.
    InvalidArgument(String),
}

/// `Result` with [`Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;

/// The location of op errors created inside nuvec, before the public op that serves the
/// caller attaches the caller's location (see [`Error::locate`]).
#[inline(never)]
fn unlocated() -> &'static Location<'static> {
    Location::caller()
}

impl Error {
    /// An op error without a caller location yet: the outermost public op attaches its
    /// name and the caller's location.
    pub(crate) fn op(op: &'static str, kind: OpErrorKind) -> Self {
        Self::Op {
            op,
            at: unlocated(),
            kind,
        }
    }

    /// Incompatible operand shapes.
    pub(crate) fn shape_mismatch(op: &'static str, lhs: &[usize], rhs: &[usize]) -> Self {
        Self::op(
            op,
            OpErrorKind::ShapeMismatch {
                lhs: dims_i32(lhs),
                rhs: dims_i32(rhs),
            },
        )
    }

    /// Incompatible operand element types.
    pub(crate) fn dtype_mismatch(op: &'static str, lhs: DType, rhs: DType) -> Self {
        Self::op(op, OpErrorKind::DTypeMismatch { lhs, rhs })
    }

    /// An unsupported element type.
    pub(crate) fn unsupported_dtype(op: &'static str, dtype: DType) -> Self {
        Self::op(op, OpErrorKind::UnsupportedDType(dtype))
    }

    /// Host data that does not match its shape.
    pub(crate) fn size_mismatch(op: &'static str, expected: usize, actual: usize) -> Self {
        Self::op(op, OpErrorKind::SizeMismatch { expected, actual })
    }

    /// An invalid axis.
    pub(crate) fn invalid_axis(op: &'static str, axis: i64, ndim: usize) -> Self {
        let axis = i32::try_from(axis).unwrap_or(if axis < 0 { i32::MIN } else { i32::MAX });
        Self::op(op, OpErrorKind::InvalidAxis { axis, ndim })
    }

    /// An invalid shape.
    pub(crate) fn invalid_shape(op: &'static str, shape: impl IntoIterator<Item = i64>) -> Self {
        Self::op(op, OpErrorKind::InvalidShape(shape.into_iter().collect()))
    }

    /// An invalid index.
    pub(crate) fn invalid_index(op: &'static str, msg: impl Into<String>) -> Self {
        Self::op(op, OpErrorKind::InvalidIndex(msg.into()))
    }

    /// An op error with the caller's location: `op` (the public operation the caller
    /// invoked) and `at` replace the internal name of an error that has no location yet;
    /// invalid arguments and allocation failures become op errors. Errors that already
    /// carry a location (from a failed input) are returned unchanged: the first error wins.
    pub(crate) fn locate(self, op: &'static str, at: &'static Location<'static>) -> Self {
        match self {
            Self::Op { at: here, kind, .. } if *here == *unlocated() => Self::Op { op, at, kind },
            Self::InvalidArgument(msg) => Self::Op {
                op,
                at,
                kind: OpErrorKind::InvalidArgument(msg),
            },
            Self::OutOfMemory(n) => Self::Op {
                op,
                at,
                kind: OpErrorKind::OutOfMemory(n),
            },
            other => other,
        }
    }

    /// The reason of an op error.
    pub fn kind(&self) -> Option<&OpErrorKind> {
        match self {
            Self::Op { kind, .. } => Some(kind),
            _ => None,
        }
    }

    /// Where the failing op was called, for op errors raised by a public operation.
    pub fn location(&self) -> Option<&'static Location<'static>> {
        match self {
            Self::Op { at, .. } if **at != *unlocated() => Some(at),
            _ => None,
        }
    }

    /// The name of the failing op, for op errors.
    pub fn op_name(&self) -> Option<&'static str> {
        match self {
            Self::Op { op, .. } => Some(op),
            _ => None,
        }
    }
}

/// Dimensions as `i32`, saturating (shapes in errors are reported as MLX does).
pub(crate) fn dims_i32(shape: &[usize]) -> Vec<i32> {
    shape
        .iter()
        .map(|&d| i32::try_from(d).unwrap_or(i32::MAX))
        .collect()
}

impl fmt::Display for OpErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { lhs, rhs } => write!(f, "shape mismatch, {lhs:?} vs {rhs:?}"),
            Self::DTypeMismatch { lhs, rhs } => write!(f, "dtype mismatch, {lhs} vs {rhs}"),
            Self::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "size mismatch, the shape needs {expected} values, got {actual}"
                )
            }
            Self::UnsupportedDType(dtype) => write!(f, "not supported for {dtype}"),
            Self::InvalidAxis { axis, ndim } => {
                write!(f, "invalid axis {axis} for an array of {ndim} dimensions")
            }
            Self::InvalidShape(shape) => write!(f, "invalid shape {shape:?}"),
            Self::InvalidIndex(msg) => write!(f, "invalid index: {msg}"),
            Self::OutOfMemory(n) => write!(f, "out of memory: cannot allocate {n} bytes"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Op { op, at, kind } if **at == *unlocated() => write!(f, "{op}: {kind}"),
            Self::Op { op, at, kind } => write!(
                f,
                "{op} at {}:{}:{}: {kind}",
                at.file(),
                at.line(),
                at.column()
            ),
            Self::OutOfMemory(n) => write!(f, "out of memory: cannot allocate {n} bytes"),
            Self::Backend(msg) => write!(f, "GPU backend error: {msg}"),
            Self::UnsupportedDevice(device) => write!(f, "no backend for device {device}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_with_location() {
        let at = Location::caller();
        let e = Error::shape_mismatch("matmul", &[2, 3], &[4, 5]).locate("matmul", at);
        assert_eq!(
            e.to_string(),
            format!(
                "matmul at {}:{}:{}: shape mismatch, [2, 3] vs [4, 5]",
                at.file(),
                at.line(),
                at.column()
            )
        );
        assert_eq!(e.location(), Some(at));
        assert_eq!(e.op_name(), Some("matmul"));
        assert!(matches!(e.kind(), Some(OpErrorKind::ShapeMismatch { .. })));
    }

    #[test]
    fn unlocated_errors_take_the_outer_op() {
        let inner = Error::unsupported_dtype("exp", DType::Complex64);
        assert_eq!(inner.location(), None);
        assert_eq!(inner.to_string(), "exp: not supported for complex64");
        let at = Location::caller();
        let outer = inner.locate("softmax", at);
        assert_eq!(outer.op_name(), Some("softmax"));
        assert_eq!(outer.location(), Some(at));
        // Located errors are kept: the first error wins.
        let again = outer.clone().locate("sum", Location::caller());
        assert_eq!(again, outer);
        // Invalid arguments become op errors; boundary errors stay as they are.
        let e = Error::InvalidArgument("bad".into()).locate("split", at);
        assert_eq!(e.kind(), Some(&OpErrorKind::InvalidArgument("bad".into())));
        let io = Error::Io("x".into()).locate("split", at);
        assert_eq!(io, Error::Io("x".into()));
    }
}
