//! Replays outputs recorded from real MLX (`scripts/mlx_fixtures.py`) without Python.

#[path = "common/fixture.rs"]
mod fixture;

use fixture::{compare, from_array, load, run};

#[test]
fn fixture_meta() {
    let (meta, cases) = load();
    assert!(
        meta.contains("mlx=0.32.4") && meta.contains("device=cpu"),
        "{meta}"
    );
    assert!(cases.len() >= 150, "only {} cases", cases.len());
}

#[test]
#[cfg_attr(
    miri,
    ignore = "hundreds of cases: too slow under Miri, and covered natively"
)]
fn mlx_fixtures() {
    let (_, cases) = load();
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|case| {
            let got = run(case).and_then(|a| from_array(&a));
            match got {
                Ok(t) => compare(case, &t, &case.outputs[0]).err(),
                Err(e) => Some(format!("{} ({}): nuvec error: {e}", case.name, case.op)),
            }
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} MLX fixture cases failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn comparator_detects_corruption() {
    let (_, cases) = load();
    let case = cases
        .iter()
        .find(|c| c.name == "add-float32")
        .expect("case exists");
    let mut corrupted = case.outputs[0].clone();
    // Elements 0-3 are NaN/inf/-0.0 edge values; element 4 is an ordinary number.
    corrupted.data[4 * 4 + 1] ^= 0x40; // flip a mantissa bit of element 4
    let got = from_array(&run(case).unwrap()).unwrap();
    let err = compare(case, &got, &corrupted).unwrap_err();
    assert!(
        err.contains("add-float32") && err.contains("element 4"),
        "{err}"
    );
}
