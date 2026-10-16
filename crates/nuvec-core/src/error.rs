//! The error type returned by fallible operations.

use std::fmt;

use crate::{DType, Device};

/// Why an operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Operand shapes are incompatible.
    ShapeMismatch {
        /// The operation that rejected the shapes.
        op: &'static str,
        /// Shape of the left operand.
        lhs: Vec<usize>,
        /// Shape of the right operand.
        rhs: Vec<usize>,
    },
    /// Operand element types are incompatible.
    DTypeMismatch {
        /// The operation that rejected the types.
        op: &'static str,
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
    UnsupportedDType {
        /// The operation that rejected the type.
        op: &'static str,
        /// The unsupported element type.
        dtype: DType,
    },
    /// Memory for this many bytes could not be allocated (or exceeds the memory limit).
    OutOfMemory(usize),
    /// The device has no backend yet.
    UnsupportedDevice(Device),
    /// An argument is invalid for another reason.
    InvalidArgument(String),
}

/// `Result` with [`Error`] as the error type.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: shape mismatch, {lhs:?} vs {rhs:?}")
            }
            Self::DTypeMismatch { op, lhs, rhs } => {
                write!(f, "{op}: dtype mismatch, {lhs} vs {rhs}")
            }
            Self::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "size mismatch: shape needs {expected} values, got {actual}"
                )
            }
            Self::UnsupportedDType { op, dtype } => write!(f, "{op}: not supported for {dtype}"),
            Self::OutOfMemory(n) => write!(f, "out of memory: cannot allocate {n} bytes"),
            Self::UnsupportedDevice(device) => write!(f, "no backend for device {device} yet"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
