//! Any tree of arrays as a safetensors file by path: `save` / `load_into`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nuvec_core::{Array, DType, Tree, array, bf16, f16};
use nuvec_io::{load_into, load_safetensors, save};

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nuvec-io-tree-{}-{name}", std::process::id()))
}

fn bits(a: &Array) -> (DType, Vec<i32>, Vec<u8>) {
    let bytes = match a.dtype() {
        DType::F16 => a
            .to_vec::<f16>()
            .unwrap()
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        DType::BF16 => a
            .to_vec::<bf16>()
            .unwrap()
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        DType::I32 => a
            .to_vec::<i32>()
            .unwrap()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        DType::U64 => a
            .to_vec::<u64>()
            .unwrap()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        _ => a
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
    };
    (a.dtype(), a.shape().to_vec(), bytes)
}

fn same<T: Tree>(a: &T, b: &T) {
    assert_eq!(a.paths(), b.paths());
    for (x, y) in a.leaves().iter().zip(b.leaves()) {
        assert_eq!(bits(x), bits(&y));
    }
}

#[test]
fn round_trips_trees_bit_for_bit() {
    let tuple = (
        array![1.5, -2.0],
        Array::from_slice(&[f16::from_f32(0.1), f16::from_f32(-3.0)], [2]),
        Array::from_slice(&[bf16::from_f32(7.25)], [1]),
        Array::from_slice(&[1i32, 2, 3, 4], [2, 2]),
        Array::from(0.5f32),
    );
    let path = temp("tuple.safetensors");
    save(&path, &tuple).unwrap();
    let mut fresh = (
        array![0.0, 0.0],
        Array::from_slice(&[f16::ZERO, f16::ZERO], [2]),
        Array::from_slice(&[bf16::ZERO], [1]),
        Array::from_slice(&[0i32; 4], [2, 2]),
        Array::from(0.0f32),
    );
    load_into(&path, &mut fresh).unwrap();
    same(&tuple, &fresh);

    let map: BTreeMap<String, Array> = [
        ("w".to_owned(), array![[1.0, 2.0]]),
        ("b".to_owned(), array![3.0]),
    ]
    .into();
    let path = temp("map.safetensors");
    save(&path, &map).unwrap();
    let mut fresh = map.map(|_, a| a.zeros_like());
    load_into(&path, &mut fresh).unwrap();
    same(&map, &fresh);

    let nested: Vec<Option<Array>> = vec![Some(array![1.0]), None, Some(array![2.0, 3.0])];
    let path = temp("nested.safetensors");
    save(&path, &nested).unwrap();
    let file = load_safetensors(&path).unwrap();
    assert_eq!(file.tensors.keys().collect::<Vec<_>>(), ["0", "2"]);
    assert_eq!(file.metadata["format"], "nuvec-tree");
    assert!(file.metadata["root"].contains("Vec"), "{:?}", file.metadata);
    let mut fresh = nested.map(|_, a| a.zeros_like());
    load_into(&path, &mut fresh).unwrap();
    same(&nested, &fresh);
}

#[test]
fn records_step_and_learning_rate() {
    let state: BTreeMap<String, Array> = [
        ("step".to_owned(), Array::from(3u64)),
        ("learning_rate".to_owned(), Array::from(0.5f32)),
    ]
    .into();
    let path = temp("state.safetensors");
    save(&path, &(array![1.0], state)).unwrap();
    let file = load_safetensors(&path).unwrap();
    assert_eq!(file.metadata["step"], "3");
    assert_eq!(file.metadata["learning_rate"], "0.5");
    assert_eq!(file.tensors["1.step"].dtype(), DType::U64);
}

#[test]
fn load_into_reports_every_problem_and_keeps_the_tree() {
    let saved: BTreeMap<String, Array> = [
        ("a".to_owned(), array![1.0, 2.0, 3.0]),
        ("c".to_owned(), array![4.0]),
        ("z".to_owned(), array![5.0]),
    ]
    .into();
    let path = temp("bad.safetensors");
    save(&path, &saved).unwrap();
    let mut tree: BTreeMap<String, Array> = [
        ("a".to_owned(), array![0.0, 0.0]),
        ("b".to_owned(), array![0.0]),
        ("c".to_owned(), array![0.0]),
    ]
    .into();
    let err = load_into(&path, &mut tree).unwrap_err().to_string();
    assert!(err.contains("load_into: 3 problems"), "{err}");
    for part in ["missing: b", "unknown: z", "shape: a [3], expected [2]"] {
        assert!(err.contains(part), "{err}");
    }
    assert_eq!(tree["c"].to_vec::<f32>().unwrap(), [0.0], "unchanged");
}
