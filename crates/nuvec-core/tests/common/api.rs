//! Helpers of the array API tests: the API fixtures recorded from MLX
//! (`fixtures/mlx_api.bin`, written by `scripts/mlx_fixtures.py`) and a gradient checker
//! in the new style.

#![allow(dead_code, unreachable_pub)]

use nuvec_core::{Array, DType, Result, jvp_flat, vjp_flat};

use super::fixture::{Case, Tensor, from_array, parse, to_array, values};

/// The API fixtures.
pub fn load_api() -> Vec<Case> {
    let (meta, cases) = parse(include_bytes!("../fixtures/mlx_api.bin"));
    assert!(meta.contains("mlx=0.32.4"), "{meta}");
    cases
}

/// The cases of `kind` (`index`, `set`, `add`, `reduce`).
pub fn cases(kind: &str) -> Vec<Case> {
    load_api().into_iter().filter(|c| c.op == kind).collect()
}

/// The case's inputs as arrays, the first one with its `pre` view applied (`t`: reversed
/// axes, `r`: the last axis reversed), as the recorder did.
pub fn inputs(case: &Case) -> Vec<Array> {
    let mut xs: Vec<Array> = case.inputs.iter().map(to_array).collect();
    let x = &xs[0];
    xs[0] = match case.params.get("pre").map(String::as_str) {
        Some("t") => x.t(),
        Some("r") => {
            let mut idx: Vec<nuvec_core::graph::SliceArg> =
                vec![nuvec_core::graph::SliceArg::full(); x.ndim()];
            *idx.last_mut().unwrap() = nuvec_core::graph::SliceArg::new(None, None, -1);
            nuvec_core::graph::slice(x, &idx).unwrap()
        }
        _ => x.clone(),
    };
    xs
}

fn epsilon(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 9.77e-4,
        DType::BF16 => 7.82e-3,
        DType::F32 | DType::Complex64 => 1.2e-7,
        _ => 2.3e-16,
    }
}

/// Compares `got` with the case's (single) output: dtype and shape exactly, integers
/// exactly, floats within `ulps` units of the output dtype's epsilon (relative to the
/// magnitude, at least 1).
pub fn compare(case: &Case, got: &Array, ulps: f64) -> std::result::Result<(), String> {
    let ctx = format!("{} [{:?}]", case.name, case.params);
    let expected = &case.outputs[0];
    let got: Tensor = from_array(got).map_err(|e| format!("{ctx}: nuvec error: {e}"))?;
    if got.dtype != expected.dtype {
        return Err(format!(
            "{ctx}: dtype {} vs MLX {}",
            got.dtype, expected.dtype
        ));
    }
    if got.shape != expected.shape {
        return Err(format!(
            "{ctx}: shape {:?} vs MLX {:?}",
            got.shape, expected.shape
        ));
    }
    let eps = epsilon(expected.dtype);
    for (i, (g, e)) in values(&got).iter().zip(values(expected)).enumerate() {
        let same = |a: f64, b: f64| {
            a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= ulps * eps * b.abs().max(1.0)
        };
        if !(same(g.0, e.0) && same(g.1, e.1)) {
            return Err(format!("{ctx}: element {i}: {g:?} vs MLX {e:?}"));
        }
    }
    Ok(())
}

/// Deterministic pseudo-random numbers in `[lo, hi)`.
pub fn uniform(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn vec64(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

/// Checks the derivatives of `f` at float64 inputs of `shapes` (values in `[lo, hi)`):
/// `vjp` against central finite differences of `⟨c, f(x)⟩`, and `⟨c, jvp_flat(t)⟩ = ⟨vjp_flat(c), t⟩`.
pub fn check_grad(
    name: &str,
    f: &dyn Fn(&[Array]) -> Array,
    shapes: &[&[usize]],
    lo: f64,
    hi: f64,
) {
    let seed = name.bytes().map(u64::from).sum::<u64>();
    let data: Vec<Vec<f64>> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| uniform(seed + i as u64, s.iter().product(), lo, hi))
        .collect();
    let build = |data: &[Vec<f64>]| -> Vec<Array> {
        data.iter()
            .zip(shapes)
            .map(|(d, s)| Array::from_slice(d, *s))
            .collect()
    };
    let primals = build(&data);
    let out = f(&primals);
    let c = uniform(seed + 99, out.size(), -1.0, 1.0);
    let cot = Array::from_slice(&c, out.shape());
    let fr = |xs: &[Array]| -> Result<Vec<Array>> { Ok(vec![f(xs)]) };
    let (_, grads) = vjp_flat(fr, &primals, &[cot]).unwrap();
    let h = 1e-6;
    for (i, g) in grads.iter().enumerate() {
        let want: Vec<i32> = shapes[i].iter().map(|&d| d as i32).collect();
        assert_eq!(g.shape(), want.as_slice(), "{name}: gradient {i} shape");
        let g = vec64(g);
        for j in 0..data[i].len() {
            let at = |delta: f64| {
                let mut d = data.clone();
                d[i][j] += delta;
                dot(&c, &vec64(&f(&build(&d))))
            };
            let numeric = (at(h) - at(-h)) / (2.0 * h);
            assert!(
                (g[j] - numeric).abs() <= 1e-6 * numeric.abs().max(1.0),
                "{name}: d/dx{i}[{j}] = {} vs finite difference {numeric}",
                g[j]
            );
        }
    }
    let t: Vec<Vec<f64>> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| uniform(seed + 200 + i as u64, s.iter().product(), -1.0, 1.0))
        .collect();
    let (_, jt) = jvp_flat(fr, &primals, &build(&t)).unwrap();
    let lhs = dot(&c, &vec64(&jt[0]));
    let rhs: f64 = grads.iter().zip(&t).map(|(g, t)| dot(&vec64(g), t)).sum();
    assert!(
        (lhs - rhs).abs() <= 1e-9 * lhs.abs().max(1.0),
        "{name}: <c, jvp_flat(t)> = {lhs} vs <vjp_flat(c), t> = {rhs}"
    );
}

/// The case's inputs, plus the expected output as an array.
pub fn expected(case: &Case) -> Array {
    to_array(&case.outputs[0])
}
