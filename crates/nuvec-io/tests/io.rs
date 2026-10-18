//! safetensors and npy against files written by MLX 0.32.4 and NumPy
//! (`scripts/io_fixtures.py`), and round trips through nuvec's writers.

use std::path::PathBuf;

use nuvec_core::{Array, Complex64, DType, arange, reshape};
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
    a.astype(DType::F64).unwrap().to_vec::<f64>().unwrap()
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
    assert_eq!(t["bool"].shape(), &[2, 3]);
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
    assert_eq!(t["f32"].shape(), &[3, 4]);
    assert_eq!(f64s(&t["f32"]), ramp(12, 0.125));
    assert_eq!(t["scalar"].shape(), &[] as &[usize]);
    assert_eq!(t["scalar"].item::<f32>().unwrap(), 2.5);
    assert_eq!(t["empty"].shape(), &[0, 3]);
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
    assert_eq!(f.shape(), &[3, 4]);
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
    assert_eq!(
        (s.shape(), s.item::<f64>().unwrap()),
        (&[] as &[usize], -1.5)
    );
    // MLX does not pad the header: the data is misaligned and copied.
    let m = load_npy(fixture("mlx_f16.npy")).unwrap();
    assert_eq!((m.dtype(), m.shape()), (DType::F16, &[2, 3][..]));
    assert_eq!(f64s(&m), ramp(6, 0.5));
}

#[test]
fn round_trips_every_dtype() {
    let base = arange(0.0, 6.0, 1.0, DType::F32).unwrap();
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
    .map(|d| {
        (
            format!("t_{d}"),
            reshape(&base, &[2, 3]).unwrap().astype(d).unwrap(),
        )
    })
    .collect();
    // Odd-sized byte tensors: the writer orders by element size, so `b` stays aligned.
    arrays.push(("a".into(), Array::from_slice(&[1u8, 2, 3], &[3]).unwrap()));
    arrays.push((
        "b".into(),
        Array::from_slice(&[0.5f32, -2.0], &[2]).unwrap(),
    ));
    // A transposed (strided) input is written in logical order.
    arrays.push(("t".into(), reshape(&base, &[2, 3]).unwrap().t().unwrap()));
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
