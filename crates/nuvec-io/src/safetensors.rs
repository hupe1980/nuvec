//! The safetensors format: an 8-byte little-endian header size, a JSON header mapping
//! tensor names to their dtype, shape and byte range, then the tensors' bytes.
//!
//! Loading maps the file without a copy (see [`MappedFile`]): every tensor is a view of the
//! mapped bytes, read from disk (or the page cache) when first used. Files whose tensors are
//! not aligned to their element sizes (MLX writes headers without padding) are not mapped:
//! their tensors are read once, with parallel positioned reads, into one buffer per file
//! where each is aligned ([`MappedFile::read_ranges`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use nuvec_core::{Array, DType, Error, MappedFile, Result};

use crate::json::{self, Json};

fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument(format!("safetensors: {}", msg.into()))
}

/// The tensors and metadata of a safetensors file.
#[derive(Debug, Clone, Default)]
pub struct SafeTensors {
    /// The tensors by name.
    pub tensors: BTreeMap<String, Array>,
    /// The `__metadata__` strings.
    pub metadata: BTreeMap<String, String>,
}

/// safetensors' dtype names, with MLX's `C64` for complex64.
fn dtype_of(name: &str) -> Option<DType> {
    Some(match name {
        "BOOL" => DType::Bool,
        "U8" => DType::U8,
        "U16" => DType::U16,
        "U32" => DType::U32,
        "U64" => DType::U64,
        "I8" => DType::I8,
        "I16" => DType::I16,
        "I32" => DType::I32,
        "I64" => DType::I64,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "F32" => DType::F32,
        "F64" => DType::F64,
        "C64" => DType::Complex64,
        _ => return None,
    })
}

fn name_of(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "BOOL",
        DType::U8 => "U8",
        DType::U16 => "U16",
        DType::U32 => "U32",
        DType::U64 => "U64",
        DType::I8 => "I8",
        DType::I16 => "I16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F64 => "F64",
        _ => "C64",
    }
}

fn usize_of(v: &Json, what: &str) -> Result<usize> {
    match v {
        Json::Int(n) if *n >= 0 => Ok(*n as usize),
        _ => Err(invalid(format!("{what} must be a non-negative integer"))),
    }
}

/// The reference implementation's limit on the header size.
const MAX_HEADER: usize = 100_000_000;

/// A parsed and validated header: the metadata and the tensors' byte ranges (sorted, tiling
/// the data section from `start` to the end of the file).
struct Header {
    metadata: BTreeMap<String, String>,
    /// `(begin, end, name, dtype, shape)`, relative to the data section.
    ranges: Vec<(usize, usize, String, DType, Vec<usize>)>,
    /// Where the data section starts (8 + the header's length).
    start: usize,
}

impl Header {
    /// Whether every tensor is aligned to its element size in the file (else the loader
    /// reads the tensors instead of mapping the file).
    fn aligned(&self) -> bool {
        self.ranges
            .iter()
            .all(|(begin, _, _, dtype, _)| (self.start + begin).is_multiple_of(dtype.size()))
    }

    /// Where the tensors `keep` accepts go in one buffer read from the file: in file order,
    /// each continuing the previous kept tensor's file range if it follows it in the file
    /// and lands aligned to its element size, else starting a new range at the next multiple
    /// of 16. MLX-written files are aligned relative to their data section, so their kept
    /// tensors take one range per gap left by rejected ones.
    fn plan(&self, keep: &dyn Fn(&str) -> bool) -> ReadPlan {
        let mut plan = ReadPlan::default();
        // The file end of the current range, if any.
        let mut file_end = None;
        for (index, (begin, end, name, dtype, _)) in self.ranges.iter().enumerate() {
            if !keep(name) {
                continue;
            }
            let at = (self.start + begin) as u64;
            let n = end - begin;
            let offset = if file_end == Some(at) && plan.len.is_multiple_of(dtype.size()) {
                let last = plan.ranges.last_mut().expect("an open range");
                last.2 += n;
                plan.len
            } else {
                let offset = plan.len.next_multiple_of(16);
                plan.ranges.push((at, offset, n));
                offset
            };
            plan.tensors.push((index, offset));
            plan.len = offset + n;
            file_end = Some(at + n as u64);
        }
        plan
    }
}

/// The reads loading a misaligned file makes ([`Header::plan`]).
#[derive(Default)]
struct ReadPlan {
    /// `(file offset, buffer offset, length)`, increasing and disjoint in the buffer.
    ranges: Vec<(u64, usize, usize)>,
    /// `(index into Header::ranges, buffer offset)` of each kept tensor.
    tensors: Vec<(usize, usize)>,
    /// The buffer's length.
    len: usize,
}

/// The data section's start from the 8-byte header size, for a file of `file_len` bytes.
fn header_end(size: [u8; 8], file_len: usize) -> Result<usize> {
    usize::try_from(u64::from_le_bytes(size))
        .ok()
        .filter(|&n| n <= MAX_HEADER)
        .and_then(|n| n.checked_add(8))
        .filter(|&end| end <= file_len)
        .ok_or_else(|| invalid("header larger than the file or 100 MB"))
}

/// Parses and validates the JSON header (`json`, the bytes after the size) of a file of
/// `file_len` bytes whose data starts at `start`.
fn parse_header(json_bytes: &[u8], start: usize, file_len: usize) -> Result<Header> {
    let Json::Object(entries) = json::parse(json_bytes)? else {
        return Err(invalid("the header must be a JSON object"));
    };
    let mut metadata = BTreeMap::new();
    let mut ranges = Vec::new();
    let mut names = BTreeSet::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            match entry {
                Json::Null => {}
                Json::Object(fields) => {
                    for (k, v) in fields {
                        let Json::Str(s) = v else {
                            return Err(invalid("metadata values must be strings"));
                        };
                        metadata.insert(k, s);
                    }
                }
                _ => return Err(invalid("`__metadata__` must be an object")),
            }
            continue;
        }
        // A repeated name would tile the data and then silently replace the first tensor.
        if !names.insert(name.clone()) {
            return Err(invalid(format!("duplicate tensor `{name}`")));
        }
        let Json::Object(fields) = entry else {
            return Err(invalid(format!("entry `{name}` must be an object")));
        };
        let field = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        let dtype = match field("dtype") {
            Some(Json::Str(d)) => {
                dtype_of(d).ok_or_else(|| invalid(format!("`{name}`: unsupported dtype {d}")))?
            }
            _ => return Err(invalid(format!("`{name}`: missing dtype"))),
        };
        let shape = match field("shape") {
            Some(Json::Array(dims)) => dims
                .iter()
                .map(|d| usize_of(d, "a dimension"))
                .collect::<Result<Vec<_>>>()?,
            _ => return Err(invalid(format!("`{name}`: missing shape"))),
        };
        let (begin, end) = match field("data_offsets") {
            Some(Json::Array(o)) if o.len() == 2 => {
                (usize_of(&o[0], "an offset")?, usize_of(&o[1], "an offset")?)
            }
            _ => return Err(invalid(format!("`{name}`: missing data_offsets"))),
        };
        let expected = shape
            .iter()
            .try_fold(dtype.size(), |n, &d| n.checked_mul(d))
            .ok_or_else(|| invalid(format!("`{name}`: size overflows")))?;
        if end.checked_sub(begin) != Some(expected) {
            return Err(invalid(format!(
                "`{name}`: data_offsets [{begin}, {end}] do not hold {expected} bytes"
            )));
        }
        ranges.push((begin, end, name, dtype, shape));
    }
    // The ranges must tile the data exactly.
    ranges.sort_by_key(|r| (r.0, r.1));
    let mut at = 0;
    for (begin, end, name, ..) in &ranges {
        if *begin != at {
            return Err(invalid(format!(
                "`{name}`: data_offsets overlap or leave a gap"
            )));
        }
        at = *end;
    }
    if start.checked_add(at) != Some(file_len) {
        return Err(invalid(
            "the tensors do not cover the data to the end of the file",
        ));
    }
    Ok(Header {
        metadata,
        ranges,
        start,
    })
}

/// Reads and validates a file's header with ordinary reads (nothing is mapped), returning
/// it and the file's length.
fn read_header(path: &Path) -> Result<(Header, usize)> {
    use std::io::Read;
    let io = |e: std::io::Error| Error::Io(format!("{}: {e}", path.display()));
    let mut file = std::fs::File::open(path).map_err(io)?;
    let file_len = usize::try_from(file.metadata().map_err(io)?.len())
        .map_err(|_| Error::Io(format!("{}: too large", path.display())))?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size).map_err(|_| {
        invalid(format!(
            "{}: file shorter than its header size",
            path.display()
        ))
    })?;
    let start = header_end(size, file_len).map_err(|e| named(path, e))?;
    let mut json_bytes = vec![0u8; start - 8];
    file.read_exact(&mut json_bytes).map_err(io)?;
    let header = parse_header(&json_bytes, start, file_len).map_err(|e| named(path, e))?;
    Ok((header, file_len))
}

/// `e` with the file's name in front of its message.
fn named(path: &Path, e: Error) -> Error {
    match e {
        Error::InvalidArgument(m) => Error::InvalidArgument(format!("{}: {m}", path.display())),
        Error::Io(m) => Error::Io(format!("{}: {m}", path.display())),
        e => e,
    }
}

/// What loading safetensors checkpoints holds in the GPU allocator, in bytes (see
/// [`safetensors_footprint`]). Compare with
/// `nuvec_gpu::MemoryStats`'s `mapped`, `active` and `peak_with_mapped`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    /// Kept mapped: the page-rounded whole files (header included) of aligned files.
    pub mapped: u64,
    /// Read into allocator memory: the size-class-rounded tensors of files whose tensors are
    /// not aligned (as MLX writes them; mlx-community checkpoints). Such files are not
    /// mapped.
    pub copied: u64,
    /// The peak of `mapped + copied` while loading: `mapped + copied` (no file is mapped
    /// and copied at once).
    pub transient: u64,
}

impl Footprint {
    /// The footprint of one file of `file_len` bytes with this header.
    fn of(header: &Header, file_len: usize) -> Self {
        if header.aligned() {
            let mapping = nuvec_core::MappedFile::mapping_size(file_len as u64);
            Footprint {
                mapped: mapping,
                copied: 0,
                transient: mapping,
            }
        } else {
            let copied = nuvec_core::MappedFile::copy_size(header.plan(&|_| true).len as u64);
            Footprint {
                mapped: 0,
                copied,
                transient: copied,
            }
        }
    }
}

/// What loading a safetensors file, or every `*.safetensors` file of a directory, will
/// hold, from their headers only (ordinary reads; nothing is mapped, copied or wired):
/// the bytes kept mapped, the bytes copied, and the transient peak of the load.
///
/// After [`load_safetensors`] (file by file for a directory) the GPU allocator's `mapped`
/// grows by `mapped`, its `active` by `copied`, and `peak_with_mapped` rises by at most
/// `transient`, which is `mapped + copied`: misaligned files are read, not mapped.
/// [`load_safetensors_filtered`] copies at most `copied`.
///
/// ```
/// # fn main() -> nuvec_core::Result<()> {
/// # if cfg!(miri) { return Ok(()); }
/// use nuvec_core::array;
///
/// let path = std::env::temp_dir().join("nuvec-footprint-doc.safetensors");
/// nuvec_io::save_safetensors(&path, &[("w", &array![1.0, 2.0])], &[])?;
/// let fp = nuvec_io::safetensors_footprint(&path)?;
/// assert_eq!(fp.copied, 0); // nuvec writes aligned files: mapped, not copied
/// assert!(fp.mapped >= std::fs::metadata(&path).unwrap().len());
/// # Ok(())
/// # }
/// ```
pub fn safetensors_footprint(path: impl AsRef<Path>) -> Result<Footprint> {
    let path = path.as_ref();
    let files = if path.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(path)
            .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        files
    } else {
        vec![path.to_path_buf()]
    };
    let mut total = Footprint::default();
    for file in &files {
        let (header, len) = read_header(file)?;
        let one = Footprint::of(&header, len);
        total.mapped += one.mapped;
        total.copied += one.copied;
    }
    total.transient = total.mapped + total.copied;
    Ok(total)
}

/// Loads a safetensors file. The tensors are views of the memory-mapped file or, if they
/// are not aligned to their element sizes (MLX writes headers without padding), of one
/// buffer they are read into with parallel positioned reads (the file is then not mapped).
///
/// Validation follows the reference implementation: a header of at most 100 MB, an
/// optional (or `null`) `__metadata__` of strings, and tensors whose byte ranges, sorted
/// by offset, tile the data from its start to the end of the file.
///
/// Before mapping or reading, the file's load transient ([`safetensors_footprint`]) is
/// checked against the GPU's memory limit: a load that would pass it fails with an error
/// naming the file, with nothing mapped or allocated.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<SafeTensors> {
    load_safetensors_filtered(path, |_| true)
}

/// [`load_safetensors`], returning only the tensors whose names `keep` accepts. In files
/// that are read rather than mapped (misaligned ones, as MLX writes them), the others are
/// not read at all: loading a language model from a multimodal checkpoint, say, skips the
/// vision weights.
///
/// ```
/// # fn main() -> nuvec_core::Result<()> {
/// # if cfg!(miri) { return Ok(()); }
/// use nuvec_core::array;
///
/// let path = std::env::temp_dir().join("nuvec-filtered-doc.safetensors");
/// nuvec_io::save_safetensors(&path, &[("a", &array![1.0]), ("b", &array![2.0])], &[])?;
/// let loaded = nuvec_io::load_safetensors_filtered(&path, |name| name == "b")?;
/// assert_eq!(loaded.tensors.keys().collect::<Vec<_>>(), ["b"]);
/// # Ok(())
/// # }
/// ```
pub fn load_safetensors_filtered(
    path: impl AsRef<Path>,
    keep: impl Fn(&str) -> bool,
) -> Result<SafeTensors> {
    let path = path.as_ref();
    let (header, len) = read_header(path)?;
    let aligned = header.aligned();
    let plan = (!aligned).then(|| header.plan(&keep));
    let need = match &plan {
        Some(plan) => nuvec_core::MappedFile::copy_size(plan.len as u64),
        None => Footprint::of(&header, len).transient,
    };
    if let Some(room) = nuvec_core::MappedFile::headroom()
        && need > room
    {
        return Err(Error::Io(format!(
            "{}: out of memory: loading needs {need} bytes (mapped or read), {room} \
             are left under the memory limit",
            path.display()
        )));
    }
    let oom = |e| match e {
        Error::OutOfMemory(n) => Error::Io(format!(
            "{}: out of memory: {n} bytes pass the memory limit",
            path.display()
        )),
        e => e,
    };
    let mut out = SafeTensors {
        metadata: header.metadata,
        ..SafeTensors::default()
    };
    match plan {
        // Read (not mapped): a mapping besides the copy would hold the checkpoint twice.
        Some(plan) => {
            let data = MappedFile::read_ranges(path, &plan.ranges, plan.len).map_err(oom)?;
            let mut ranges: Vec<_> = header.ranges.into_iter().map(Some).collect();
            for (index, offset) in plan.tensors {
                let (_, _, name, dtype, shape) = ranges[index].take().expect("kept once");
                out.tensors.insert(name, data.array(offset, dtype, &shape)?);
            }
        }
        None => {
            let file = MappedFile::open(path).map_err(oom)?;
            // The file is the one whose header was read; it must not change while it is
            // mapped (see `MappedFile`), so a different length is an error.
            if file.len() != len {
                return Err(Error::Io(format!(
                    "{}: the file changed while it was loaded",
                    path.display()
                )));
            }
            for (begin, _, name, dtype, shape) in header.ranges {
                if keep(&name) {
                    let tensor = file.array(header.start + begin, dtype, &shape)?;
                    out.tensors.insert(name, tensor);
                }
            }
        }
    }
    Ok(out)
}

/// The array's elements as little-endian bytes, row-major.
pub(crate) fn array_bytes(a: &Array) -> Result<Vec<u8>> {
    macro_rules! le {
        ($t:ty) => {
            a.to_vec::<$t>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        };
    }
    Ok(match a.dtype() {
        DType::Bool => a.to_vec::<bool>()?.iter().map(|&b| u8::from(b)).collect(),
        DType::U8 => a.to_vec::<u8>()?,
        DType::U16 => le!(u16),
        DType::U32 => le!(u32),
        DType::U64 => le!(u64),
        DType::I8 => le!(i8),
        DType::I16 => le!(i16),
        DType::I32 => le!(i32),
        DType::I64 => le!(i64),
        DType::F16 => le!(nuvec_core::f16),
        DType::BF16 => le!(nuvec_core::bf16),
        DType::F32 => le!(f32),
        DType::F64 => le!(f64),
        _ => a
            .to_vec::<nuvec_core::Complex64>()?
            .iter()
            .flat_map(|z| [z.re.to_le_bytes(), z.im.to_le_bytes()].concat())
            .collect(),
    })
}

/// The file's bytes for `tensors` (evaluating them) and `metadata`.
fn encode(tensors: &[(&str, &Array)], metadata: &[(&str, &str)]) -> Result<Vec<u8>> {
    let mut sorted: Vec<&(&str, &Array)> = tensors.iter().collect();
    sorted.sort_by(|(ln, la), (rn, ra)| ra.dtype().size().cmp(&la.dtype().size()).then(ln.cmp(rn)));
    let mut header = Vec::new();
    if !metadata.is_empty() {
        let map: BTreeMap<String, String> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        header.push(format!("\"__metadata__\":{}", json::object(&map)));
    }
    let mut data = Vec::new();
    for (name, a) in sorted {
        let bytes = array_bytes(a)?;
        let shape: Vec<String> = a.shape().iter().map(i32::to_string).collect();
        header.push(format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            json::quote(name),
            name_of(a.dtype()),
            shape.join(","),
            data.len(),
            data.len() + bytes.len()
        ));
        data.extend(bytes);
    }
    let mut header = format!("{{{}}}", header.join(",")).into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend((header.len() as u64).to_le_bytes());
    out.extend(header);
    out.extend(data);
    Ok(out)
}

/// Writes `bytes` to `path` atomically: to `path.tmp`, renamed over `path` on success.
fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let io = |e: std::io::Error| Error::Io(format!("{}: {e}", path.display()));
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    if let Err(e) = std::fs::write(&tmp, bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(io(e));
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        io(e)
    })
}

/// Saves `tensors` (evaluating them) and `metadata` as a safetensors file, written to
/// `path.tmp` and renamed into place.
///
/// As the reference implementation, tensors are laid out by decreasing element size (then
/// by name) after a header padded to 8 bytes, so every tensor is aligned to its element
/// size and loads without a copy.
pub fn save_safetensors(
    path: impl AsRef<Path>,
    tensors: &[(&str, &Array)],
    metadata: &[(&str, &str)],
) -> Result<()> {
    write_atomically(path.as_ref(), &encode(tensors, metadata)?)
}

/// A background safetensors write ([`save_safetensors_async`]). [`SaveHandle::wait`]
/// returns its result; dropping the handle waits for the write to finish.
#[must_use = "wait() returns the write's result; dropping the handle only waits"]
#[derive(Debug)]
pub struct SaveHandle {
    thread: Option<std::thread::JoinHandle<Result<()>>>,
}

impl SaveHandle {
    /// Waits for the write and returns its result.
    pub fn wait(mut self) -> Result<()> {
        self.join()
    }

    fn join(&mut self) -> Result<()> {
        match self.thread.take() {
            Some(t) => t
                .join()
                .unwrap_or_else(|_| Err(Error::Io("the safetensors writer panicked".into()))),
            None => Ok(()),
        }
    }
}

impl Drop for SaveHandle {
    fn drop(&mut self) {
        let _ = self.join();
    }
}

/// [`save_safetensors`] in the background, for snapshots taken while work continues (a
/// KV cache or recurrent state copied with `copy`, say): the arrays are submitted for
/// evaluation ([`nuvec_core::async_eval`]) and returned at once with a handle; a
/// background thread waits for them and writes `path.tmp`, renamed into place on success.
/// The file is byte-identical to [`save_safetensors`]'s. The handle holds the arrays until
/// the write is done, so an in-place update of one of them copies instead of writing into
/// memory being saved.
///
/// ```
/// # fn main() -> nuvec_core::Result<()> {
/// # if cfg!(miri) { return Ok(()); }
/// use nuvec_core::array;
///
/// let path = std::env::temp_dir().join("nuvec-async-save-doc.safetensors");
/// let state = array![1.0, 2.0, 3.0];
/// let handle = nuvec_io::save_safetensors_async(&path, &[("state", &state.copy())], &[]);
/// // ... more work here ...
/// handle.wait()?;
/// assert_eq!(nuvec_io::load_safetensors(&path)?.tensors["state"].to_vec::<f32>()?, [1.0, 2.0, 3.0]);
/// # Ok(())
/// # }
/// ```
pub fn save_safetensors_async(
    path: impl AsRef<Path>,
    tensors: &[(&str, &Array)],
    metadata: &[(&str, &str)],
) -> SaveHandle {
    let path = path.as_ref().to_path_buf();
    let tensors: Vec<(String, Array)> = tensors
        .iter()
        .map(|(n, a)| ((*n).to_string(), (*a).clone()))
        .collect();
    let metadata: Vec<(String, String)> = metadata
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    let submitted = nuvec_core::async_eval(&tensors.iter().map(|(_, a)| a).collect::<Vec<_>>());
    let thread = std::thread::spawn(move || {
        submitted?;
        let tensors: Vec<(&str, &Array)> = tensors.iter().map(|(n, a)| (n.as_str(), a)).collect();
        let metadata: Vec<(&str, &str)> = metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        write_atomically(&path, &encode(&tensors, &metadata)?)
    });
    SaveHandle {
        thread: Some(thread),
    }
}
