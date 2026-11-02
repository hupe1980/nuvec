//! Errors of the CUDA runtime.

use std::fmt;

/// Why a CUDA operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaError {
    /// No usable device: no driver, a driver too old, no GPU, or a GPU below sm_80.
    NoDevice(String),
    /// A driver call failed.
    Driver {
        /// The call.
        call: &'static str,
        /// The driver's message.
        message: String,
    },
    /// Memory for this many bytes could not be allocated (or exceeds the memory limit).
    OutOfMemory(usize),
    /// A Tile kernel failed to compile or launch (cutile, `tileiras`).
    Tile(String),
    /// An argument is invalid.
    InvalidArgument(String),
}

/// `Result` with [`CudaError`].
pub type Result<T> = std::result::Result<T, CudaError>;

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice(msg) => write!(f, "no CUDA device: {msg}"),
            Self::Driver { call, message } => write!(f, "CUDA {call}: {message}"),
            Self::OutOfMemory(n) => write!(f, "CUDA out of memory: cannot allocate {n} bytes"),
            Self::Tile(msg) => write!(f, "CUDA Tile kernel: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "CUDA: {msg}"),
        }
    }
}

impl std::error::Error for CudaError {}

/// A driver result as [`CudaError::Driver`], naming `call`.
pub(crate) fn driver(call: &'static str) -> impl Fn(cuda_core::DriverError) -> CudaError {
    move |e| CudaError::Driver {
        call,
        message: e.to_string(),
    }
}

/// A raw `CUresult` as a `Result`.
pub(crate) fn check(call: &'static str, r: cuda_core::sys::CUresult) -> Result<()> {
    if r == cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(driver(call)(cuda_core::DriverError(r)))
    }
}
