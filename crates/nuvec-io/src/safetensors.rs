//! The safetensors format: an 8-byte little-endian header size, a JSON header mapping
//! tensor names to their dtype, shape and byte range, then the tensors' bytes.
//!
//! Loading maps the file without a copy (see [`MappedFile`]): every tensor is a view of the
//! mapped bytes, read from disk (or the page cache) when first used.

use std::collections::BTreeMap;
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

/// Loads a safetensors file. The tensors are views of the memory-mapped file.
///
/// Validation follows the reference implementation: a header of at most 100 MB, an
/// optional (or `null`) `__metadata__` of strings, and tensors whose byte ranges, sorted
/// by offset, tile the data from its start to the end of the file.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<SafeTensors> {
    let file = MappedFile::open(path)?;
    let bytes = file.bytes();
    let size = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or_else(|| invalid("file shorter than its header size"))?;
    let start = usize::try_from(size)
        .ok()
        .filter(|&n| n <= MAX_HEADER)
        .and_then(|n| n.checked_add(8))
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| invalid("header larger than the file or 100 MB"))?;
    let Json::Object(entries) = json::parse(&bytes[8..start])? else {
        return Err(invalid("the header must be a JSON object"));
    };
    let mut out = SafeTensors::default();
    let mut ranges = Vec::new();
    for (name, entry) in entries {
        if name == "__metadata__" {
            match entry {
                Json::Null => {}
                Json::Object(fields) => {
                    for (k, v) in fields {
                        let Json::Str(s) = v else {
                            return Err(invalid("metadata values must be strings"));
                        };
                        out.metadata.insert(k, s);
                    }
                }
                _ => return Err(invalid("`__metadata__` must be an object")),
            }
            continue;
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
    if start + at != bytes.len() {
        return Err(invalid(
            "the tensors do not cover the data to the end of the file",
        ));
    }
    for (begin, _, name, dtype, shape) in ranges {
        out.tensors
            .insert(name, file.array(start + begin, dtype, &shape)?);
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

/// Saves `tensors` (evaluating them) and `metadata` as a safetensors file.
///
/// As the reference implementation, tensors are laid out by decreasing element size (then
/// by name) after a header padded to 8 bytes, so every tensor is aligned to its element
/// size and loads without a copy.
pub fn save_safetensors(
    path: impl AsRef<Path>,
    tensors: &[(&str, &Array)],
    metadata: &[(&str, &str)],
) -> Result<()> {
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
        let shape: Vec<String> = a.shape().iter().map(usize::to_string).collect();
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
    std::fs::write(path.as_ref(), out)
        .map_err(|e| Error::Io(format!("{}: {e}", path.as_ref().display())))
}
