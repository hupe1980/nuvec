//! Fuzz targets for every parser of untrusted model files: JSON, safetensors, npy
//! and idx, each run on `NUVEC_FUZZ_CASES` (default 10 000) mutated inputs seeded with the
//! fixtures and every crash input found so far. A parser may return `Err` on any
//! input; it must never panic.

mod fuzz_harness;

use std::path::PathBuf;

use fuzz_harness::fuzz;
use nuvec_io::{json, load_npy, load_safetensors, parse_idx};

fn fixtures(extension: &str) -> Vec<Vec<u8>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == extension))
        .collect();
    paths.sort();
    paths.iter().map(|p| std::fs::read(p).unwrap()).collect()
}

/// A per-target scratch file: the file loaders take paths.
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nuvec-io-fuzz-{}-{name}", std::process::id()))
}

fn safetensors_bytes(header: &[u8], data: usize) -> Vec<u8> {
    [
        (header.len() as u64).to_le_bytes().to_vec(),
        header.to_vec(),
        vec![0; data],
    ]
    .concat()
}

fn npy_bytes(header: &[u8], data: usize) -> Vec<u8> {
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((header.len() as u16).to_le_bytes());
    b.extend(header);
    b.extend(vec![0; data]);
    b
}

fn idx_bytes(code: u8, shape: &[u32], data: usize) -> Vec<u8> {
    let mut b = vec![0, 0, code, shape.len() as u8];
    for d in shape {
        b.extend(d.to_be_bytes());
    }
    b.extend(vec![1; data]);
    b
}

#[test]
fn fuzz_json() {
    let mut corpus: Vec<Vec<u8>> = vec![
        br#"{"a": [1, -2.5e3, true, false, null], "b": {"c": "\u00e9\n\ud83d\ude00"}}"#.to_vec(),
        br#"{"model_type": "llama", "hidden_size": 64, "rope_theta": 10000.0, "eos_token_id": [1, 2]}"#
            .to_vec(),
        // Crash and strictness inputs found so far.
        [vec![b'['; 100_000], vec![b']'; 100_000]].concat(),
        b"{\"a\":".repeat(10_000),
        br#""\ud800""#.to_vec(),
        br#""\udc00\u+abc""#.to_vec(),
        b"[01, -01, 1., .5, 1e, +1]".to_vec(),
    ];
    // safetensors headers.
    for file in fixtures("safetensors") {
        let n = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
        corpus.push(file[8..8 + n].to_vec());
    }
    fuzz("json", 0x6A73_6F6E, &corpus, |input| {
        let _ = json::parse(input);
    });
}

#[test]
fn fuzz_safetensors() {
    let mut corpus = fixtures("safetensors");
    corpus.extend([
        safetensors_bytes(
            br#"{"__metadata__":{"k":"v"},"a":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"b":{"dtype":"F32","shape":[1],"data_offsets":[3,7]}}"#,
            7,
        ),
        safetensors_bytes(
            br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[9223372036854775806,9223372036854775807]}}"#,
            0,
        ),
        safetensors_bytes(
            br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#,
            2,
        ),
        safetensors_bytes(
            br#"{"x":{"dtype":"F64","shape":[2147483648,1073741824],"data_offsets":[0,0]}}"#,
            0,
        ),
        [u64::MAX.to_le_bytes().to_vec(), b"{}".to_vec()].concat(),
    ]);
    let path = scratch("st.safetensors");
    fuzz("safetensors", 0x5354, &corpus, |input| {
        std::fs::write(&path, input).unwrap();
        if let Ok(st) = load_safetensors(&path) {
            for a in st.tensors.values() {
                let _ = (a.shape(), a.dtype());
            }
        }
    });
    std::fs::remove_file(&path).ok();
}

#[test]
fn fuzz_npy() {
    let mut corpus = fixtures("npy");
    corpus.extend([
        npy_bytes(
            b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
            24,
        ),
        npy_bytes(
            b"{'descr': '<i2', 'fortran_order': True, 'shape': (3, 4), }",
            24,
        ),
        npy_bytes(b"{'descr': '<f4', 'fortran_order': False, 'shape': )}", 0),
        npy_bytes(
            "{'descr': '<f4', 'fortran_order': False, 'shape': \u{e9})}".as_bytes(),
            0,
        ),
        npy_bytes(
            b"{'descr': '<f8', 'fortran_order': False, 'shape': (2147483648, 1073741824), }",
            0,
        ),
    ]);
    let path = scratch("a.npy");
    fuzz("npy", 0x4E5059, &corpus, |input| {
        std::fs::write(&path, input).unwrap();
        if let Ok(a) = load_npy(&path) {
            let _ = (a.shape(), a.dtype());
        }
    });
    std::fs::remove_file(&path).ok();
}

#[test]
fn fuzz_idx() {
    let corpus = vec![
        idx_bytes(0x08, &[2, 2, 3], 12),
        idx_bytes(0x0C, &[2], 8),
        idx_bytes(0x0D, &[1, 2], 8),
        idx_bytes(0x0E, &[1 << 31, 1 << 30], 0),
        idx_bytes(0x0B, &[u32::MAX, u32::MAX, u32::MAX], 0),
    ];
    fuzz("idx", 0x1D8, &corpus, |input| {
        if let Ok(a) = parse_idx(input) {
            let _ = a.shape();
        }
    });
}
