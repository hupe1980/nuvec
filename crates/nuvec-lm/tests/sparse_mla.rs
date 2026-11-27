//! Offline differential and state-isolation checks for a single V3.2 attention layer.
use std::collections::BTreeMap;
use std::path::Path;

use nuvec::{Array, DType, Device, Result, grad, io, metal, ops, with_stream};
use nuvec_lm::sparse_mla::{SparseMla, SparseMlaConfig};
use nuvec_nn::Yarn;

fn config() -> SparseMlaConfig {
    SparseMlaConfig {
        hidden_size: 16,
        num_attention_heads: 2,
        q_lora_rank: 8,
        kv_lora_rank: 8,
        qk_nope_head_dim: 6,
        qk_rope_head_dim: 4,
        v_head_dim: 6,
        index_n_heads: 4,
        index_head_dim: 8,
        index_topk: 3,
        max_position_embeddings: 32,
        yarn: Some(Yarn {
            factor: 4.0,
            original_max_position_embeddings: 16.0,
            beta_fast: 8.0,
            mscale_all_dim: 1.0,
            ..Yarn::default()
        }),
        ..SparseMlaConfig::default()
    }
}

fn tensors(name: &str) -> BTreeMap<String, Array> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/sparse_mla/{name}.safetensors"));
    io::load_safetensors(path).unwrap().tensors
}

fn close(a: &Array, b: &Array, tol: f32) {
    assert_eq!(a.shape(), b.shape());
    for (i, (a, b)) in a
        .astype(DType::F32)
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .zip(b.astype(DType::F32).to_vec::<f32>().unwrap())
        .enumerate()
    {
        assert!(
            (a - b).abs() <= tol * (1.0 + b.abs()),
            "element {i}: {a} vs {b} (tol {tol})"
        );
    }
}

fn oracle_and_state(name: &str) -> Result<()> {
    let arrays = tensors(name);
    let layer = SparseMla::new(config(), arrays.clone())?;
    let x = &arrays["input"];
    let tol = if name == "f32" { 2e-4 } else { 0.03 };
    let empty = layer.new_session();
    let (full, state, ids) = layer.append_with_selection(x, &empty)?;
    close(&full, &arrays["expected"], tol);
    let picks = ids.to_vec::<i32>()?;
    let expected = arrays["picks"].to_vec::<i32>()?;
    for (a, b) in picks.chunks(3).zip(expected.chunks(3)) {
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }
    assert_eq!(state.nbytes(), 8 * (8 + 4 + 8) * x.itemsize());
    // Cross the sparse boundary in either direction using an immutable prefix.
    for split in [0, 1, 2, 3, 4, 7, 8] {
        let (left, prefix) = layer.append(&x.i((.., ..split)), &empty)?;
        let (right, end) = layer.append(&x.i((.., split..)), &prefix)?;
        close(
            &ops::concatenate(&[left, right], 1),
            &arrays["expected"],
            tol,
        );
        assert_eq!(end.offset(), 8);
        let trimmed = state.trim(8 - split)?;
        let alternate = x.i((.., split..)).negative();
        let (resumed, _) = layer.append(&alternate, &trimmed)?;
        let (fresh, _) = layer.append(&alternate, &prefix)?;
        close(&resumed, &fresh, tol);
    }
    // Identical rows without positional differences tie index scores if the indexer
    // projection is zero. Ensure selection is reproducible, unique and causal.
    let mut tied = arrays.clone();
    let w = &tied["indexer.weights_proj.weight"];
    tied.insert(
        "indexer.weights_proj.weight".into(),
        ops::zeros(w.shape(), w.dtype()),
    );
    let ties = SparseMla::new(config(), tied)?;
    let (y, _, a) = ties.append_with_selection(x, &ties.new_session())?;
    let (_, _, b) = ties.append_with_selection(x, &ties.new_session())?;
    let a = a.to_vec::<i32>()?;
    assert_eq!(a, b.to_vec::<i32>()?);
    for (row, ids) in a.chunks(3).enumerate() {
        let valid = ids
            .iter()
            .copied()
            .filter(|&i| i >= 0)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(valid.len(), (row + 1).min(3));
        assert!(valid.iter().all(|&i| i <= row as i32));
    }
    let altered = ops::concatenate(&[x.i((.., ..4)), x.i((.., 4..)).negative()], 1);
    close(
        &ties.append(&altered, &ties.new_session())?.0.i((.., ..4)),
        &y.i((.., ..4)),
        tol,
    );
    let mut wider = config();
    wider.index_topk = 66;
    wider.max_position_embeddings = 96;
    let wider = SparseMla::new(wider, arrays.clone())?;
    let (out, _, picks) =
        wider.append_with_selection(&arrays["long_input"], &wider.new_session())?;
    close(&out, &arrays["long_expected"], tol);
    assert_eq!(picks.shape(), [67, 66]);
    let first = picks.i(0).to_vec::<i32>()?;
    assert_eq!(first[0], 0);
    assert!(first[1..].iter().all(|&p| p == -1));
    // A view with a non-unit hidden stride.
    let strided = ops::stack(&[x.clone(), x.negative()], -1).i((.., .., .., 0));
    close(&layer.append(&strided, &empty)?.0, &full, tol);
    Ok(())
}

#[test]
fn release_oracle_and_session_on_cpu() {
    with_stream(Device::cpu(), || {
        oracle_and_state("f32").unwrap();
        oracle_and_state("bf16").unwrap();
    });
}

#[test]
fn release_oracle_and_session_on_metal() {
    if !metal::is_available() {
        eprintln!("skipping physical Metal checks: Metal unavailable");
        return;
    }
    with_stream(Device::gpu(0), || {
        oracle_and_state("f32").unwrap();
        oracle_and_state("bf16").unwrap();
    });
}

#[test]
fn rejects_bad_input_without_changing_state() {
    // 119-sparse-mla/SR-001: session ownership. 119-sparse-mla/SR-002: extent checks.
    let mut weights = tensors("f32");
    let x = weights["input"].clone();
    let layer = SparseMla::new(config(), weights.clone()).unwrap();
    let session = layer.new_session();
    let (_, state) = layer.append(&x, &session).unwrap();
    assert!(layer.append(&x.astype(DType::F16), &state).is_err());
    assert!(
        layer
            .append(&ops::zeros([1, 25, 16], x.dtype()), &state)
            .is_err()
    );
    assert!(layer.append(&x.i(0), &state).is_err());
    assert!(state.trim(-1).is_err());
    assert!(state.trim(9).is_err());
    let other = SparseMla::new(config(), weights.clone()).unwrap();
    assert!(other.append(&x, &state).is_err());
    close(
        &layer.append(&x, &state.trim(8).unwrap()).unwrap().0,
        &weights["expected"],
        2e-4,
    );
    weights.remove("indexer.wk.weight");
    assert!(
        SparseMla::new(config(), weights)
            .err()
            .unwrap()
            .to_string()
            .contains("indexer.wk.weight")
    );
    let mut cfg = config();
    cfg.qk_rope_head_dim = 3;
    assert!(SparseMla::new(cfg, tensors("f32")).is_err());
}

#[test]
fn differentiates_fixed_selections() {
    with_stream(Device::cpu(), || {
        let arrays = tensors("f32");
        let layer = SparseMla::new(config(), arrays.clone()).unwrap();
        let f =
            |x: &Array| -> Result<Array> { Ok(layer.append(x, &layer.new_session())?.0.sum(..)) };
        let gradient = grad(f)(&arrays["input"]).unwrap();
        let actual = gradient.multiply(&arrays["direction"]).sum(..);
        close(&actual, &arrays["directional"], 2e-3);
    });
}
