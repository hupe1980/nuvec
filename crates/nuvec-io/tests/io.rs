//! safetensors and npy against files written by MLX 0.32.4 and NumPy
//! (`scripts/io_fixtures.py`), and round trips through nuvec's writers.

use std::path::PathBuf;

use nuvec_core::{Array, Complex64, DType, array, ops};
use nuvec_io::{load_npy, load_safetensors, save_npy, save_safetensors};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nuvec-io-{}-{name}", std::process::id()))
}

fn f64s(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

fn ramp(n: usize, step: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * step).collect()
}

#[test]
fn loads_mlx_safetensors() {
    let st = load_safetensors(fixture("mlx.safetensors")).unwrap();
    assert_eq!(st.metadata["format"], "mlx");
    assert_eq!(st.metadata["note"], "ü \"q\"");
    let t = &st.tensors;
    assert_eq!(t["bool"].dtype(), DType::Bool);
    assert_eq!(t["bool"].shape(), [2, 3]);
    assert_eq!(
        t["bool"].to_vec::<bool>().unwrap(),
        vec![true, false, true, false, true, false]
    );
    assert_eq!(t["u8"].to_vec::<u8>().unwrap(), vec![0, 1, 2, 3, 4]);
    assert_eq!(t["u32"].to_vec::<u32>().unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(t["i16"].to_vec::<i16>().unwrap(), vec![-3, -2, -1, 0, 1, 2]);
    assert_eq!(t["i64"].to_vec::<i64>().unwrap(), vec![-3, -2, -1, 0, 1, 2]);
    assert_eq!(t["f16"].dtype(), DType::F16);
    assert_eq!(f64s(&t["f16"]), ramp(8, 0.25));
    assert_eq!(t["bf16"].dtype(), DType::BF16);
    assert_eq!(f64s(&t["bf16"]), ramp(8, 0.5));
    assert_eq!(t["f32"].shape(), [3, 4]);
    assert_eq!(f64s(&t["f32"]), ramp(12, 0.125));
    assert_eq!(t["scalar"].shape(), &[] as &[i32]);
    assert_eq!(t["scalar"].item::<f32>().unwrap(), 2.5);
    assert_eq!(t["empty"].shape(), [0, 3]);
    // Loaded tensors take part in computation like any array.
    let doubled = (&t["f32"] * 2.0).sum_all();
    assert_eq!(doubled, 2.0 * ramp(12, 0.125).iter().sum::<f64>());
}

trait SumAll {
    fn sum_all(&self) -> f64;
}

impl SumAll for Array {
    fn sum_all(&self) -> f64 {
        f64s(self).iter().sum()
    }
}

#[test]
fn loads_numpy_and_mlx_npy() {
    let c = load_npy(fixture("c_f32.npy")).unwrap();
    assert_eq!((c.dtype(), c.shape()), (DType::F32, &[2, 3, 4][..]));
    assert_eq!(f64s(&c), ramp(24, 0.5));
    // Fortran order loads as a transposed view with C-order values.
    let f = load_npy(fixture("fortran_i16.npy")).unwrap();
    assert_eq!(f.shape(), [3, 4]);
    assert_eq!(f.to_vec::<i16>().unwrap(), (0..12).collect::<Vec<i16>>());
    assert_eq!(
        load_npy(fixture("bool.npy"))
            .unwrap()
            .to_vec::<bool>()
            .unwrap(),
        vec![false, true, false, true, false]
    );
    let z = load_npy(fixture("c64.npy"))
        .unwrap()
        .to_vec::<Complex64>()
        .unwrap();
    assert_eq!(
        z,
        (0..4)
            .map(|i| Complex64::new(i as f32, 2.0 * i as f32))
            .collect::<Vec<_>>()
    );
    let s = load_npy(fixture("scalar_f64.npy")).unwrap();
    assert_eq!((s.shape(), s.item::<f64>().unwrap()), (&[] as &[i32], -1.5));
    // MLX does not pad the header: the data is misaligned and copied.
    let m = load_npy(fixture("mlx_f16.npy")).unwrap();
    assert_eq!((m.dtype(), m.shape()), (DType::F16, &[2, 3][..]));
    assert_eq!(f64s(&m), ramp(6, 0.5));
}

#[test]
fn round_trips_every_dtype() {
    let base = ops::arange(0.0, 6.0, 1.0);
    let mut arrays: Vec<(String, Array)> = [
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
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::Complex64,
    ]
    .into_iter()
    .map(|d| (format!("t_{d}"), base.reshape([2, 3]).astype(d)))
    .collect();
    // Odd-sized byte tensors: the writer orders by element size, so `b` stays aligned.
    arrays.push(("a".into(), array![1u8, 2, 3]));
    arrays.push(("b".into(), array![0.5, -2.0]));
    // A transposed (strided) input is written in logical order.
    arrays.push(("t".into(), base.reshape([2, 3]).t()));
    let refs: Vec<(&str, &Array)> = arrays.iter().map(|(n, a)| (n.as_str(), a)).collect();
    let path = temp("roundtrip.safetensors");
    save_safetensors(&path, &refs, &[("k", "v")]).unwrap();
    let loaded = load_safetensors(&path).unwrap();
    assert_eq!(loaded.metadata["k"], "v");
    for (name, a) in &arrays {
        let b = &loaded.tensors[name];
        assert_eq!((b.dtype(), b.shape()), (a.dtype(), a.shape()), "{name}");
        if a.dtype() == DType::Complex64 {
            assert_eq!(
                b.to_vec::<Complex64>().unwrap(),
                a.to_vec::<Complex64>().unwrap()
            );
        } else {
            assert_eq!(f64s(b), f64s(a), "{name}");
        }
    }
    for (name, a) in &arrays {
        if a.dtype() == DType::BF16 {
            assert!(save_npy(temp("bf16.npy"), a).is_err());
            continue;
        }
        let p = temp(&format!("{name}.npy"));
        save_npy(&p, a).unwrap();
        let b = load_npy(&p).unwrap();
        assert_eq!((b.dtype(), b.shape()), (a.dtype(), a.shape()), "{name}");
        if a.dtype() == DType::Complex64 {
            assert_eq!(
                b.to_vec::<Complex64>().unwrap(),
                a.to_vec::<Complex64>().unwrap()
            );
        } else {
            assert_eq!(f64s(&b), f64s(a), "{name}");
        }
        std::fs::remove_file(p).ok();
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn malformed_files_are_rejected() {
    let path = temp("bad.safetensors");
    for bytes in [
        vec![1u8, 2, 3],
        [100u64.to_le_bytes().to_vec(), b"{}".to_vec()].concat(),
        [11u64.to_le_bytes().to_vec(), b"{\"x\": 1}   ".to_vec()].concat(),
        {
            let h = b"{\"x\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}";
            [
                (h.len() as u64).to_le_bytes().to_vec(),
                h.to_vec(),
                vec![0; 8],
            ]
            .concat()
        },
        {
            let h = b"{\"x\":{\"dtype\":\"F8_E4M3\",\"shape\":[1],\"data_offsets\":[0,1]}}";
            [
                (h.len() as u64).to_le_bytes().to_vec(),
                h.to_vec(),
                vec![0; 1],
            ]
            .concat()
        },
        // A gap between tensors, and trailing bytes after the last one.
        {
            let h = b"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
            [
                (h.len() as u64).to_le_bytes().to_vec(),
                h.to_vec(),
                vec![0; 2],
            ]
            .concat()
        },
        {
            let h = b"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
            [
                (h.len() as u64).to_le_bytes().to_vec(),
                h.to_vec(),
                vec![0; 3],
            ]
            .concat()
        },
    ] {
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_safetensors(&path).is_err(), "{bytes:?}");
    }
    std::fs::write(&path, b"\x93NUMPY\x01\x00\x10\x00{'descr': '>f4'").unwrap();
    assert!(load_npy(&path).is_err());
    assert!(load_safetensors(temp("missing.safetensors")).is_err());
    std::fs::remove_file(path).ok();
}

#[test]
fn reads_null_metadata_and_copies_misaligned_tensors() {
    // As written by MLX without metadata, and by older writers in name order (a 3-byte
    // tensor before a float32 one leaves the latter misaligned: it is copied).
    let h = b"{\"__metadata__\":null,\"a\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[3,7]}}";
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(h);
    bytes.extend([7, 8, 9]);
    bytes.extend(1.5f32.to_le_bytes());
    let path = temp("null-metadata.safetensors");
    std::fs::write(&path, &bytes).unwrap();
    let st = load_safetensors(&path).unwrap();
    assert!(st.metadata.is_empty());
    assert_eq!(st.tensors["a"].to_vec::<u8>().unwrap(), vec![7, 8, 9]);
    assert_eq!(st.tensors["b"].to_vec::<f32>().unwrap(), vec![1.5]);
    std::fs::remove_file(path).ok();
}

#[test]
fn unpadded_headers_are_copied_once() {
    // An unpadded header (as MLX writes) leaves the whole data section misaligned: it is
    // copied once, and every tensor reads its values from the copy.
    let h = b"{ \"w\":{\"dtype\":\"U32\",\"shape\":[2],\"data_offsets\":[0,8]},\"s\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[8,12]},\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[12,16]}}";
    assert_ne!(
        (8 + h.len()) % 4,
        0,
        "the data section must start misaligned"
    );
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(h);
    bytes.extend(7u32.to_le_bytes());
    bytes.extend(u32::MAX.to_le_bytes());
    bytes.extend([0x80, 0x3f, 0x00, 0x40]); // bfloat16 1.0 and 2.0
    bytes.extend((-0.5f32).to_le_bytes());
    let path = temp("unpadded.safetensors");
    std::fs::write(&path, &bytes).unwrap();
    let st = load_safetensors(&path).unwrap();
    assert_eq!(st.tensors["w"].to_vec::<u32>().unwrap(), vec![7, u32::MAX]);
    assert_eq!(f64s(&st.tensors["s"]), vec![1.0, 2.0]);
    assert_eq!(st.tensors["x"].to_vec::<f32>().unwrap(), vec![-0.5]);
    std::fs::remove_file(path).ok();
}

#[test]
fn deeply_nested_json_is_an_error_not_a_stack_overflow() {
    // On a thread with the default 2 MB stack: 100 000 levels used to overflow it.
    let handle = std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(|| {
            let deep = [vec![b'['; 100_000], vec![b']'; 100_000]].concat();
            let arrays = nuvec_io::json::parse(&deep);
            let objects = nuvec_io::json::parse(&b"{\"a\":".repeat(100_000));
            (arrays.is_err(), objects.is_err())
        })
        .unwrap();
    assert_eq!(handle.join().unwrap(), (true, true));
    // 128 levels are accepted, 129 are not.
    let nested = |n: usize| [vec![b'['; n], vec![b']'; n]].concat();
    assert!(nuvec_io::json::parse(&nested(128)).is_ok());
    assert!(nuvec_io::json::parse(&nested(129)).is_err());
}

#[test]
fn json_is_strict() {
    use nuvec_io::json::{Json, parse};
    for bad in [
        &br#""\ud800""#[..],
        br#""\ud800x""#,
        br#""\ud800\u0041""#,
        br#""\udc00""#,
        br#""\u+abc""#,
        br#""\u-abc""#,
        b"01",
        b"-01",
        b"00",
        b"1.",
        b".5",
        b"-",
        b"1e",
        b"1e+",
        b"+1",
        b"1.e5",
    ] {
        assert!(parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
    }
    assert_eq!(
        parse(br#""\ud83d\ude00""#).unwrap(),
        Json::Str("\u{1F600}".into())
    );
    assert_eq!(parse(b"0").unwrap(), Json::Int(0));
    assert_eq!(parse(b"-0").unwrap(), Json::Int(0));
    assert_eq!(parse(b"10").unwrap(), Json::Int(10));
    assert_eq!(parse(b"-0.5e-1").unwrap(), Json::Float(-0.05));
    assert_eq!(parse(b"1E+2").unwrap(), Json::Float(100.0));
    assert_eq!(parse(b"99999999999999999999").unwrap(), Json::Float(1e20));
}

/// A safetensors file with header `h` and `data` zero bytes after it.
fn safetensors_bytes(h: &[u8], data: usize) -> Vec<u8> {
    [
        (h.len() as u64).to_le_bytes().to_vec(),
        h.to_vec(),
        vec![0; data],
    ]
    .concat()
}

#[test]
fn hostile_headers_are_errors() {
    let path = temp("hostile.npy");
    for header in [
        &b"{'descr': '<f4', 'fortran_order': False, 'shape': )}"[..],
        "{'descr': '<f4', 'fortran_order': False, 'shape': \u{e9})}".as_bytes(),
    ] {
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header);
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            load_npy(&path).is_err(),
            "{}",
            String::from_utf8_lossy(header)
        );
    }
    std::fs::remove_file(&path).ok();

    let path = temp("hostile.safetensors");
    let offsets = |begin: u64, end: u64| {
        format!("{{\"x\":{{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[{begin},{end}]}}}}")
    };
    for bytes in [
        // Offsets near `usize::MAX` (the parser's integers are `i64`) and beyond.
        safetensors_bytes(offsets(i64::MAX as u64 - 1, i64::MAX as u64).as_bytes(), 0),
        safetensors_bytes(offsets(u64::MAX - 1, u64::MAX).as_bytes(), 0),
        // A duplicate name: two tensors that tile the data, one of them silently lost.
        safetensors_bytes(
            b"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]},\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}",
            2,
        ),
        // A header length above the 100 MB limit, rejected before reading or allocating it.
        [100_000_001u64.to_le_bytes().to_vec(), b"{}".to_vec()].concat(),
        [u64::MAX.to_le_bytes().to_vec(), b"{}".to_vec()].concat(),
    ] {
        std::fs::write(&path, &bytes).unwrap();
        assert!(
            load_safetensors(&path).is_err(),
            "{}",
            String::from_utf8_lossy(&bytes)
        );
    }
    std::fs::remove_file(&path).ok();

    // idx: 2^31 × 2^30 float64 elements need 2^64 bytes.
    let mut idx = vec![0, 0, 0x0E, 2];
    idx.extend((1u32 << 31).to_be_bytes());
    idx.extend((1u32 << 30).to_be_bytes());
    assert!(nuvec_io::parse_idx(&idx).is_err());
}
