//! NumPy's `.npy` format: a magic string, a version, a Python-literal header (`descr`,
//! `fortran_order`, `shape`), then the elements.
//!
//! Loading maps the file without a copy (see [`MappedFile`]); Fortran-order files load as
//! transposed views.

use std::path::Path;

use nuvec_core::{Array, DType, Error, MappedFile, Result, transpose};

use crate::safetensors::array_bytes;

fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument(format!("npy: {}", msg.into()))
}

/// NumPy's type string (little-endian) for a dtype; bfloat16 has none.
fn descr_of(dtype: DType) -> Option<&'static str> {
    Some(match dtype {
        DType::Bool => "|b1",
        DType::U8 => "|u1",
        DType::U16 => "<u2",
        DType::U32 => "<u4",
        DType::U64 => "<u8",
        DType::I8 => "|i1",
        DType::I16 => "<i2",
        DType::I32 => "<i4",
        DType::I64 => "<i8",
        DType::F16 => "<f2",
        DType::F32 => "<f4",
        DType::F64 => "<f8",
        DType::Complex64 => "<c8",
        _ => return None,
    })
}

fn dtype_of(descr: &str) -> Option<DType> {
    // Single-byte types carry no byte order; `=` is native (little-endian on Apple Silicon).
    let canonical = match descr.as_bytes() {
        [b'=' | b'<' | b'|', rest @ ..] => format!("<{}", std::str::from_utf8(rest).ok()?),
        _ => return None,
    };
    [
        DType::Bool,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Complex64,
    ]
    .into_iter()
    .find(|&d| descr_of(d).is_some_and(|s| s[1..] == canonical[1..]))
}

/// The text after `'key':` in the header dict.
fn field<'a>(header: &'a str, key: &str) -> Result<&'a str> {
    let pat = format!("'{key}':");
    let at = header
        .find(&pat)
        .ok_or_else(|| invalid(format!("header has no `{key}`")))?;
    Ok(header[at + pat.len()..].trim_start())
}

/// Loads a `.npy` file as a view of the memory-mapped file.
pub fn load_npy(path: impl AsRef<Path>) -> Result<Array> {
    let file = MappedFile::open(path)?;
    let bytes = file.bytes();
    if bytes.get(..6) != Some(b"\x93NUMPY") || bytes.len() < 10 {
        return Err(invalid("not an npy file"));
    }
    let (len, start) = match bytes[6] {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 | 3 => (
            bytes
                .get(8..12)
                .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize)
                .ok_or_else(|| invalid("truncated header"))?,
            12,
        ),
        v => return Err(invalid(format!("unsupported version {v}"))),
    };
    let header = bytes
        .get(start..start + len)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(|| invalid("truncated or non-text header"))?;
    let descr = field(header, "descr")?;
    let descr = descr
        .strip_prefix('\'')
        .and_then(|d| d.split('\'').next())
        .ok_or_else(|| invalid("bad descr"))?;
    let dtype = dtype_of(descr).ok_or_else(|| invalid(format!("unsupported type {descr}")))?;
    let fortran = field(header, "fortran_order")?.starts_with("True");
    let shape_text = field(header, "shape")?;
    let close = shape_text.find(')').ok_or_else(|| invalid("bad shape"))?;
    let shape = shape_text[1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| invalid("bad shape")))
        .collect::<Result<Vec<_>>>()?;
    let data = start + len;
    if fortran && shape.len() > 1 {
        let reversed: Vec<usize> = shape.iter().rev().copied().collect();
        let a = file.array(data, dtype, &reversed)?;
        let axes: Vec<usize> = (0..shape.len()).rev().collect();
        transpose(&a, &axes)
    } else {
        file.array(data, dtype, &shape)
    }
}

/// Saves `a` (evaluating it) as a C-order `.npy` file (version 1). bfloat16 has no NumPy
/// type and is rejected.
pub fn save_npy(path: impl AsRef<Path>, a: &Array) -> Result<()> {
    let descr =
        descr_of(a.dtype()).ok_or_else(|| invalid(format!("{} has no NumPy type", a.dtype())))?;
    let shape = match a.shape() {
        [n] => format!("({n},)"),
        s => format!(
            "({})",
            s.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    // Pad so the data starts 64-byte aligned; the header ends with a newline.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header.into_bytes());
    out.extend(array_bytes(a)?);
    std::fs::write(path.as_ref(), out)
        .map_err(|e| Error::Io(format!("{}: {e}", path.as_ref().display())))
}
