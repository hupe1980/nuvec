//! Arrays backed by a file's bytes, for loaders (safetensors, npy, ...).

use std::path::Path;
use std::sync::Arc;

use crate::array::Data;
use crate::buffer::Buffer;
use crate::{Array, DType, Error, Result, shape};

/// A file's bytes in array storage: on a Mac with a GPU, the file is memory-mapped and
/// wrapped as GPU-visible memory without a copy, so arrays made from it are views whose
/// pages are read when first touched (by the CPU or the GPU). Elsewhere the file is read
/// into memory. The mapping lives as long as any array made from it.
///
/// ```
/// use nuvec_core::{DType, MappedFile};
///
/// # fn main() -> nuvec_core::Result<()> {
/// # if cfg!(miri) { return Ok(()); } // no file system under Miri's isolation
/// let path = std::env::temp_dir().join("nuvec-mapped-file-doc.bin");
/// let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
/// std::fs::write(&path, &bytes).unwrap();
/// let file = MappedFile::open(&path)?;
/// let a = file.array(4, DType::F32, &[3])?; // bytes 4..16, no copy
/// assert_eq!(a.to_vec::<f32>()?, vec![2.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct MappedFile {
    buffer: Arc<Buffer>,
}

impl std::fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedFile")
            .field("len", &self.len())
            .finish()
    }
}

impl MappedFile {
    /// Maps (or reads) the file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            buffer: Arc::new(Buffer::map_file(path.as_ref())?),
        })
    }

    /// The file's size in bytes.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the file is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The file's bytes.
    pub fn bytes(&self) -> &[u8] {
        self.buffer.bytes()
    }

    /// A row-major array of `dtype` and `shape` stored at byte `offset`: a view of the file
    /// if `offset` is a multiple of the element size, otherwise a copy.
    pub fn array(&self, offset: usize, dtype: DType, shape: &[usize]) -> Result<Array> {
        let nbytes = shape::checked_nbytes(shape, dtype.size())?;
        if offset
            .checked_add(nbytes)
            .is_none_or(|end| end > self.len())
        {
            return Err(Error::Io(format!(
                "{nbytes} bytes at offset {offset} exceed the file's {} bytes",
                self.len()
            )));
        }
        if offset.is_multiple_of(dtype.size()) {
            let data = Data {
                buffer: self.buffer.clone(),
                offset: (offset / dtype.size()) as isize,
            };
            return Array::leaf(Ok(data), dtype, shape.into());
        }
        let mut copy = Buffer::try_alloc(nbytes)?;
        copy.bytes_mut()
            .copy_from_slice(&self.bytes()[offset..offset + nbytes]);
        Array::from_buffer(copy, dtype, shape)
    }
}
