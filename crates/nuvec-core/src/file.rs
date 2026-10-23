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
/// The file must not be modified (written or truncated, by this or another process) while
/// it is mapped: arrays made from it would change underneath their readers (undefined
/// behaviour), and reads past a truncated end raise `SIGBUS`. Map only files nothing
/// writes while they are in use, such as model weights.
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
    /// Maps (or reads) the file at `path`. The file must not be modified while it is
    /// mapped (see [`MappedFile`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            buffer: Arc::new(Buffer::map_file(path.as_ref())?),
        })
    }

    /// The bytes mapping a file of `len` bytes holds in the GPU allocator's `mapped`
    /// statistic: the whole file, rounded up to pages.
    pub fn mapping_size(len: u64) -> u64 {
        nuvec_gpu::mapping_size(len as usize) as u64
    }

    /// The bytes storage of `len` bytes (as [`MappedFile::read_ranges`] allocates) holds in
    /// the GPU allocator's `active` statistic: its size class.
    pub fn copy_size(len: u64) -> u64 {
        nuvec_gpu::allocation_size(len as usize).map_or(u64::MAX, |n| n as u64)
    }

    /// The bytes that may still be mapped or allocated before the GPU's memory limit
    /// (`memory_limit − active − mapped`), or `None` without a GPU.
    pub fn headroom() -> Option<u64> {
        let m = nuvec_gpu::Device::shared()?.memory();
        Some(m.memory_limit.saturating_sub(m.active_with_mapped) as u64)
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

    /// Reads parts of the file at `path` into new array storage of `len` bytes, without
    /// mapping it: each `(file_offset, offset, n)` of `ranges` puts the file's bytes
    /// `file_offset..file_offset + n` at `offset..offset + n`. For files whose arrays are
    /// not aligned to their element sizes (safetensors written without header padding, as
    /// MLX writes them): the ranges are placed so that every array is aligned, and arrays
    /// made from the result are views of it. Bytes outside the ranges are unspecified.
    ///
    /// The ranges must be in increasing order of `offset` and must not overlap. They are
    /// read with positioned reads in chunks of 16 MiB, on up to 8 threads (one per 64 MiB
    /// read), straight into the storage: from the page cache this runs at memory
    /// bandwidth, and nothing but the storage is allocated (the file is not mapped).
    ///
    /// ```
    /// use nuvec_core::{DType, MappedFile};
    ///
    /// # fn main() -> nuvec_core::Result<()> {
    /// # if cfg!(miri) { return Ok(()); }
    /// let path = std::env::temp_dir().join("nuvec-read-ranges-doc.bin");
    /// // One padding byte, then three f32s: misaligned in the file.
    /// let mut bytes = vec![0u8];
    /// bytes.extend([1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()));
    /// std::fs::write(&path, &bytes).unwrap();
    /// let file = MappedFile::read_ranges(&path, &[(1, 0, 12)], 12)?;
    /// assert_eq!(file.array(0, DType::F32, &[3])?.to_vec::<f32>()?, vec![1.0, 2.0, 3.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_ranges(
        path: impl AsRef<Path>,
        ranges: &[(u64, usize, usize)],
        len: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let mut at = 0;
        for &(_, offset, n) in ranges {
            if offset < at || offset.checked_add(n).is_none_or(|end| end > len) {
                return Err(Error::InvalidArgument(format!(
                    "{}: read ranges must be increasing, disjoint and within {len} bytes",
                    path.display()
                )));
            }
            at = offset + n;
        }
        let io = |e: std::io::Error| Error::Io(format!("{}: {e}", path.display()));
        let file = std::fs::File::open(path).map_err(io)?;
        let mut buffer = Buffer::try_alloc(len)?;
        read_parallel(&file, buffer.bytes_mut(), ranges).map_err(io)?;
        Ok(Self {
            buffer: Arc::new(buffer),
        })
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

/// Reads chunk size: large enough that a read's cost is the copy, small enough to spread a
/// file over the threads.
const CHUNK: usize = 16 << 20;
/// Bytes per reading thread (smaller reads use fewer threads).
const PER_THREAD: usize = 64 << 20;
/// At most this many reading threads: more do not read faster from the page cache.
const MAX_THREADS: usize = 8;

/// Fills `dest[offset..offset + n]` with the file's bytes `file_offset..file_offset + n` for
/// each range (increasing and disjoint in `dest`, checked by the caller), in chunks of
/// [`CHUNK`] spread over a few threads.
fn read_parallel(
    file: &std::fs::File,
    dest: &mut [u8],
    ranges: &[(u64, usize, usize)],
) -> std::io::Result<()> {
    // The chunks as disjoint mutable slices of `dest`, with their file offsets.
    let mut chunks: Vec<(u64, &mut [u8])> = Vec::new();
    let mut rest = dest;
    let mut consumed = 0;
    for &(file_offset, offset, n) in ranges {
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(offset - consumed);
        let (mut range, tail) = tail.split_at_mut(n);
        rest = tail;
        consumed = offset + n;
        let mut at = file_offset;
        while !range.is_empty() {
            let (chunk, tail) = range.split_at_mut(range.len().min(CHUNK));
            let n = chunk.len() as u64;
            chunks.push((at, chunk));
            at += n;
            range = tail;
        }
    }
    let total: usize = ranges.iter().map(|r| r.2).sum();
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(MAX_THREADS)
        .min(total.div_ceil(PER_THREAD))
        .max(1);
    if threads == 1 {
        return chunks
            .into_iter()
            .try_for_each(|(at, chunk)| read_exact_at(file, chunk, at));
    }
    let work = std::sync::Mutex::new(chunks.into_iter());
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    loop {
                        let next = work
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .next();
                        let Some((at, chunk)) = next else {
                            return Ok(());
                        };
                        read_exact_at(file, chunk, at)?;
                    }
                })
            })
            .collect();
        workers.into_iter().try_for_each(|w| {
            w.join()
                .unwrap_or_else(|_| Err(std::io::Error::other("a reading thread panicked")))
        })
    })
}

/// Fills `buf` with the file's bytes from `offset` (a positioned read: the file's cursor is
/// neither used nor moved, so threads read one file concurrently).
#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranges of several chunks, gaps and odd offsets land where they are asked to.
    #[test]
    #[cfg_attr(miri, ignore)] // no file system under Miri's isolation
    fn read_ranges_places_every_byte() {
        let path =
            std::env::temp_dir().join(format!("nuvec-read-ranges-{}.bin", std::process::id()));
        let bytes: Vec<u8> = (0..(CHUNK * 2 + 12_345))
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let big = CHUNK * 2 + 1000;
        let ranges = [
            (3u64, 0usize, 10usize),
            (17, 16, big),
            (bytes.len() as u64 - 5, big + 32, 5),
        ];
        let file = MappedFile::read_ranges(&path, &ranges, big + 37).unwrap();
        for &(at, offset, n) in &ranges {
            assert_eq!(
                &file.bytes()[offset..offset + n],
                &bytes[at as usize..at as usize + n]
            );
        }
        // Past the end of the file, out of order, or past the storage: errors.
        assert!(MappedFile::read_ranges(&path, &[(bytes.len() as u64 - 2, 0, 4)], 4).is_err());
        assert!(MappedFile::read_ranges(&path, &[(0, 8, 4), (0, 0, 4)], 12).is_err());
        assert!(MappedFile::read_ranges(&path, &[(0, 0, 8)], 4).is_err());
        let empty = MappedFile::read_ranges(&path, &[], 0).unwrap();
        assert!(empty.is_empty());
        std::fs::remove_file(&path).unwrap();
    }
}
