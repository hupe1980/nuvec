//! The IDX format (MNIST and Fashion-MNIST): a big-endian header (two zero bytes, a type
//! code, the number of dimensions, then one `u32` per dimension) followed by row-major
//! big-endian data.

use std::path::Path;

use nuvec_core::{Array, Error, Result};

fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument(format!("idx: {}", msg.into()))
}

/// Parses an IDX file's bytes. Element types map to `uint8`, `int8`, `int16`, `int32`,
/// `float32` and `float64`.
pub fn parse_idx(bytes: &[u8]) -> Result<Array> {
    let header = bytes
        .get(..4)
        .ok_or_else(|| invalid("file shorter than its header"))?;
    if header[..2] != [0, 0] {
        return Err(invalid("bad magic number"));
    }
    let ndim = usize::from(header[3]);
    let dims_end = 4 + 4 * ndim;
    let shape: Vec<usize> = bytes
        .get(4..dims_end)
        .ok_or_else(|| invalid("truncated dimensions"))?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&d| u32::from_be_bytes(d) as usize)
        .collect();
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| invalid("dimensions overflow"))?;
    let data = &bytes[dims_end..];
    macro_rules! decode {
        ($t:ty) => {{
            let size = std::mem::size_of::<$t>();
            if data.len() != count * size {
                return Err(invalid(format!(
                    "expected {} data bytes for shape {shape:?}, got {}",
                    count * size,
                    data.len()
                )));
            }
            let values: Vec<$t> = data
                .chunks_exact(size)
                .map(|c| <$t>::from_be_bytes(c.try_into().expect("element bytes")))
                .collect();
            Array::from_slice(&values, &shape)
        }};
    }
    match header[2] {
        0x08 => decode!(u8),
        0x09 => decode!(i8),
        0x0B => decode!(i16),
        0x0C => decode!(i32),
        0x0D => decode!(f32),
        0x0E => decode!(f64),
        code => Err(invalid(format!("unknown type code {code:#04x}"))),
    }
}

/// Reads an (uncompressed) IDX file.
pub fn read_idx(path: impl AsRef<Path>) -> Result<Array> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    parse_idx(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(code: u8, shape: &[u32], data: &[u8]) -> Vec<u8> {
        let mut b = vec![0, 0, code, shape.len() as u8];
        for d in shape {
            b.extend(d.to_be_bytes());
        }
        b.extend(data);
        b
    }

    #[test]
    fn images_and_labels() {
        let images = parse_idx(&file(
            0x08,
            &[2, 2, 3],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255],
        ))
        .unwrap();
        assert_eq!(images.shape(), &[2, 2, 3]);
        assert_eq!(images.to_vec::<u8>().unwrap()[11], 255);
        let labels = parse_idx(&file(0x08, &[3], &[7, 2, 1])).unwrap();
        assert_eq!(labels.to_vec::<u8>().unwrap(), [7, 2, 1]);
    }

    #[test]
    fn big_endian_wider_types() {
        let mut data = Vec::new();
        data.extend((-2i32).to_be_bytes());
        data.extend(300i32.to_be_bytes());
        assert_eq!(
            parse_idx(&file(0x0C, &[2], &data))
                .unwrap()
                .to_vec::<i32>()
                .unwrap(),
            [-2, 300]
        );
        let data: Vec<u8> = [1.5f32, -0.25]
            .iter()
            .flat_map(|v| v.to_be_bytes())
            .collect();
        assert_eq!(
            parse_idx(&file(0x0D, &[2], &data))
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            [1.5, -0.25]
        );
    }

    #[test]
    fn malformed_files_are_errors() {
        assert!(parse_idx(&[0, 0]).is_err());
        assert!(parse_idx(&[1, 0, 8, 1, 0, 0, 0, 1, 5]).is_err());
        assert!(parse_idx(&file(0x08, &[3], &[1, 2])).is_err());
        assert!(parse_idx(&file(0x42, &[1], &[1])).is_err());
        assert!(matches!(
            read_idx("/nonexistent/train-images-idx3-ubyte"),
            Err(Error::Io(_))
        ));
    }
}
