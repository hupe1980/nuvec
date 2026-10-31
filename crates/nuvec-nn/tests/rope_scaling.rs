//! RoPE scalings, the YaRN mscale² fold and the compiled activations against mlx-lm 0.32.2,
//! recorded on MLX's GPU by `scripts/lm_fixtures.py --only ops` (`mlx_lm_ops.safetensors`).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use nuvec::fast::{self, AttentionOptions};
use nuvec::{Array, DType};
use nuvec_nn::{Llama3, LongRope, Rope, RopeScaling, Yarn};

fn fixtures() -> &'static BTreeMap<String, Array> {
    static F: OnceLock<BTreeMap<String, Array>> = OnceLock::new();
    F.get_or_init(|| {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/mlx_lm_ops.safetensors"
        );
        nuvec::io::load_safetensors(path)
            .unwrap()
            .tensors
            .into_iter()
            .collect()
    })
}

fn get(name: &str) -> &'static Array {
    fixtures()
        .get(name)
        .unwrap_or_else(|| panic!("no fixture {name}"))
}

fn values(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

/// Elements within `tol` of the expected, relative to the larger of 1 and its magnitude.
fn close(name: &str, got: &Array, want: &Array, tol: f64) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    for (i, (a, b)) in values(got).iter().zip(values(want)).enumerate() {
        let ok = a == &b || (a - b).abs() <= tol * b.abs().max(1.0);
        assert!(ok, "{name}: element {i}: {a} vs {b}");
    }
}

fn scalings() -> [(&'static str, i32, f32, RopeScaling); 4] {
    let long_factor = values(get("longrope.factor"))
        .into_iter()
        .map(|v| v as f32)
        .collect();
    [
        (
            "yarn",
            64,
            150_000.0,
            RopeScaling::Yarn(Yarn {
                factor: 32.0,
                original_max_position_embeddings: 4096.0,
                ..Yarn::default()
            }),
        ),
        (
            "llama3",
            128,
            500_000.0,
            RopeScaling::Llama3(Llama3 {
                factor: 8.0,
                ..Llama3::default()
            }),
        ),
        (
            "longrope",
            96,
            10_000.0,
            RopeScaling::LongRope(LongRope {
                long_factor,
                ..LongRope::default()
            }),
        ),
        (
            "proportional",
            512,
            1e6,
            RopeScaling::Proportional {
                rotated_dims: 128,
                factor: 1.0,
            },
        ),
    ]
}

#[test]
fn frequencies_mscale_and_rotations_equal_mlx_lm() {
    for (name, dims, base, scaling) in scalings() {
        let freqs = scaling.frequencies(dims, base).expect("scaled frequencies");
        close(
            &format!("{name} freqs"),
            &freqs,
            get(&format!("{name}.freqs")),
            1e-6,
        );
        let want = get(&format!("{name}.mscale")).item::<f32>().unwrap();
        assert!(
            (scaling.mscale() - want).abs() <= 1e-6 * want,
            "{name} mscale {} vs {want}",
            scaling.mscale()
        );
        let rope = Rope::scaled(dims, base, &scaling);
        let got = rope.forward(get(&format!("{name}.x")), 3);
        close(
            &format!("{name} rotated"),
            &got,
            get(&format!("{name}.out")),
            1e-5,
        );
    }
    // `.base(..)` after `scaled` recomputes the frequencies (no stale frequencies).
    let (_, dims, base, scaling) = scalings().into_iter().next().unwrap();
    let rebased = Rope::scaled(dims, 1.0, &scaling).base(base);
    let fresh = Rope::scaled(dims, base, &scaling);
    let x = get("yarn.x");
    assert_eq!(values(&rebased.forward(x, 3)), values(&fresh.forward(x, 3)));
}

/// gpt-oss's attention: mlx-lm multiplies q and k by YaRN's mscale before rotating; nuvec
/// rotates them unscaled (one fused kernel each) and multiplies the attention scale by
/// mscale², the same up to rounding.
#[test]
fn yarn_mscale_folds_into_the_attention_scale() {
    let (_, dims, base, scaling) = scalings().into_iter().next().unwrap();
    let rope = Rope::scaled(dims, base, &scaling);
    let r = rope.rotation();
    let rotate = |x: &Array, offset: i32| {
        fast::rope(
            x,
            r.dims,
            r.traditional,
            r.base,
            r.scale,
            &Array::from(offset),
            r.freqs,
        )
    };
    let (q, k) = (rotate(get("yarn_attn.q"), 3), rotate(get("yarn_attn.k"), 0));
    let opts = AttentionOptions::causal().sinks(Some(get("yarn_attn.sinks")));
    let scale = 0.125 * rope.mscale().powi(2);
    let got = fast::scaled_dot_product_attention(&q, &k, get("yarn_attn.v"), scale, &opts);
    close("yarn attention", &got, get("yarn_attn.out"), 1e-5);
}

#[test]
fn activations_equal_mlx_lm() {
    for (label, tol) in [("f32", 1e-6), ("bf16", 1e-2)] {
        let (gate, x) = (
            get(&format!("act_{label}.gate")),
            get(&format!("act_{label}.x")),
        );
        let swiglu = nuvec_nn::swiglu_oai(gate, x, 1.702, 7.0);
        close(
            &format!("{label} swiglu_oai"),
            &swiglu,
            get(&format!("act_{label}.swiglu_oai")),
            tol,
        );
        let geglu = nuvec_nn::geglu(gate, x);
        close(
            &format!("{label} geglu"),
            &geglu,
            get(&format!("act_{label}.geglu")),
            tol,
        );
        let capped = nuvec_nn::logit_softcap(&gate.multiply(10), 30.0);
        close(
            &format!("{label} softcap"),
            &capped,
            get(&format!("act_{label}.softcap")),
            tol,
        );
        assert_eq!(swiglu.dtype(), gate.dtype());
    }
    for cap in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            nuvec_nn::logit_softcap(get("act_f32.gate"), cap)
                .check()
                .is_err(),
            "cap {cap}"
        );
    }
}

/// The soft-capped attention recipe of `logit_softcap`'s documentation, with an additive
/// mask and sinks appended uncapped, equals `softmax(c · tanh(s / c) + mask) v` computed
/// directly in float64.
#[test]
fn softcap_recipe_equals_the_formula() {
    let (h, lq, lk, d, cap, scale) = (2usize, 3usize, 5usize, 4usize, 2.0f32, 0.7f32);
    let data = |n: usize, seed: u64| -> Vec<f32> {
        (0..n)
            .map(|i| (((i as u64 * 2654435761 + seed) % 1000) as f32 / 500.0 - 1.0) * 2.0)
            .collect()
    };
    let (qd, kd, vd) = (
        data(h * lq * d, 1),
        data(h * lk * d, 2),
        data(h * lk * d, 3),
    );
    let mask: Vec<f32> = (0..lq * lk)
        .map(|i| {
            if i % lk > i / lk + lk - lq {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let sinks = [0.3f32, -0.4];
    let q = Array::from_slice(&qd, [1, h, lq, d]);
    let k = Array::from_slice(&kd, [1, h, lk, d]);
    let v = Array::from_slice(&vd, [1, h, lk, d]);
    let m = Array::from_slice(&mask, [lq, lk]);
    let s = q.matmul(&k.transpose([0, 1, 3, 2])).multiply(scale);
    let s = nuvec_nn::logit_softcap(&s, cap).add(&m);
    let sink = Array::from_slice(&sinks, [1, h, 1, 1]).broadcast_to([1, h, lq, 1]);
    let p = nuvec::ops::concatenate(&[s, sink], -1).softmax(-1);
    let got = p.i((.., .., .., ..lk as i32)).matmul(&v);
    let mut want = vec![0f64; h * lq * d];
    for hh in 0..h {
        for i in 0..lq {
            let mut logits: Vec<f64> = (0..lk)
                .map(|j| {
                    let dot: f64 = (0..d)
                        .map(|t| {
                            f64::from(qd[(hh * lq + i) * d + t])
                                * f64::from(kd[(hh * lk + j) * d + t])
                        })
                        .sum();
                    let s = dot * f64::from(scale);
                    f64::from(cap) * (s / f64::from(cap)).tanh() + f64::from(mask[i * lk + j])
                })
                .collect();
            logits.push(f64::from(sinks[hh]));
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
            let z: f64 = e.iter().sum();
            for t in 0..d {
                want[(hh * lq + i) * d + t] = (0..lk)
                    .map(|j| e[j] / z * f64::from(vd[(hh * lk + j) * d + t]))
                    .sum();
            }
        }
    }
    for (i, (g, w)) in values(&got).iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-5, "element {i}: {g} vs {w}");
    }
}
