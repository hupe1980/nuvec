//! Errors of the Metal runtime.

use std::fmt;

/// Why a GPU operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuError {
    /// The kernel library is unavailable in this build (non-macOS target).
    NoKernels,
    /// The kernel library could not be loaded or compiled.
    LibraryLoad(String),
    /// No kernel with this name exists in the library.
    MissingKernel(String),
    /// A compute pipeline could not be created.
    Pipeline {
        /// The kernel name.
        name: String,
        /// Metal's description.
        message: String,
    },
    /// A dispatch argument is invalid (offset out of range, too many arguments, …).
    InvalidArgument(String),
    /// The buffer is used by submitted GPU work that has not been waited on.
    BufferInUse,
    /// Metal could not allocate a buffer or command object of this many bytes.
    OutOfMemory(usize),
    /// The command buffer failed on the GPU.
    CommandBuffer(String),
}

/// `Result` with [`GpuError`].
pub type Result<T> = std::result::Result<T, GpuError>;

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoKernels => f.write_str("the Metal kernel library is unavailable in this build"),
            Self::LibraryLoad(msg) => write!(f, "cannot load the Metal kernel library: {msg}"),
            Self::MissingKernel(name) => write!(f, "no Metal kernel named `{name}`"),
            Self::Pipeline { name, message } => {
                write!(f, "cannot create pipeline `{name}`: {message}")
            }
            Self::InvalidArgument(msg) => write!(f, "invalid dispatch argument: {msg}"),
            Self::BufferInUse => {
                f.write_str("buffer is in use by submitted GPU work; wait for it first")
            }
            Self::OutOfMemory(n) => write!(f, "Metal could not allocate {n} bytes"),
            Self::CommandBuffer(msg) => write!(f, "GPU command buffer failed: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}
