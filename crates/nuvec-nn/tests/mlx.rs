//! `nuvec-nn` against `mlx.nn` and `mlx.optimizers` 0.32.4: inputs and expected outputs
//! recorded by `scripts/nn_fixtures.py` (`just nn-fixtures`) on MLX's CPU device.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use nuvec::{Array, DType, random};
use nuvec_nn::layers::LoraBase;
use nuvec_nn::optim::{Adam, LearningRate, Lion, Muon, Optimizer, Sgd, clip_grad_norm, schedules};
use nuvec_nn::{
    Embedding, LayerNorm, Linear, LoraLinear, Module, RmsNorm, Rope, gelu_approx, gelu_fast_approx,
    relu, silu,
};

fn fixtures() -> &'static BTreeMap<String, Array> {
    static F: OnceLock<BTreeMap<String, Array>> = OnceLock::new();
    F.get_or_init(|| {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/mlx_nn.safetensors"
        );
        nuvec::io::load_safetensors(path).unwrap().tensors
    })
}

fn get(name: &str) -> Array {
    fixtures()
        .get(name)
        .unwrap_or_else(|| panic!("no fixture {name}"))
        .clone()
}

fn f32s(a: &Array) -> Vec<f32> {
    a.astype(DType::F32).unwrap().to_vec::<f32>().unwrap()
}

/// Equal to MLX within `rel` (relative to max(|expected|, 1)).
fn close(got: &Array, expected: &str, rel: f32) {
    let (g, e) = (f32s(got), f32s(&get(expected)));
    assert_eq!(got.shape(), get(expected).shape(), "{expected}: shape");
    for (i, (x, y)) in g.iter().zip(&e).enumerate() {
        assert!(
            (x - y).abs() <= rel * y.abs().max(1.0),
            "{expected}[{i}]: {x} vs MLX {y}"
        );
    }
}

#[test]
fn initialisation_draws_mlx_values() {
    random::seed(1);
    let e = Embedding::new(10, 4).unwrap();
    assert_eq!(
        f32s(&e.weight),
        f32s(&get("init_embedding.weight")),
        "bit for bit"
    );
    random::seed(2);
    let base = Linear::new(6, 5, true).unwrap();
    let lora = LoraLinear::from_base(LoraBase::Linear(base), 3, 20.0).unwrap();
    assert_eq!(
        f32s(&lora.lora_a),
        f32s(&get("init_lora.lora_a")),
        "bit for bit"
    );
    assert_eq!(f32s(&lora.lora_b), vec![0.0; 15]);
}

#[test]
fn layers_equal_mlx() {
    let emb = Embedding {
        weight: get("embedding.weight"),
    };
    close(
        &emb.forward(&get("embedding.ids")).unwrap(),
        "embedding.out",
        0.0,
    );
    close(
        &emb.as_linear(&get("embedding.x")).unwrap(),
        "embedding.as_linear",
        1e-6,
    );

    let x = get("rms_norm.x");
    let mut rms = RmsNorm::new(8, 1e-5).unwrap();
    rms.weight = get("rms_norm.weight");
    close(&rms.forward(&x).unwrap(), "rms_norm.out", 1e-6);

    let mut ln = LayerNorm::new(8, 1e-5, true, true).unwrap();
    ln.weight = Some(get("layer_norm.weight"));
    ln.bias = Some(get("layer_norm.bias"));
    close(
        &ln.forward(&get("layer_norm.x")).unwrap(),
        "layer_norm.out",
        1e-6,
    );
    let plain = LayerNorm::new(8, 1e-5, false, false).unwrap();
    close(
        &plain.forward(&get("layer_norm.x")).unwrap(),
        "layer_norm.plain",
        1e-6,
    );

    let q = get("rope.x");
    close(&Rope::new(8).forward(&q, 3).unwrap(), "rope.out", 1e-6);
    let traditional = Rope {
        traditional: true,
        ..Rope::new(12)
    };
    close(
        &traditional.forward(&q, 0).unwrap(),
        "rope.traditional",
        1e-6,
    );

    let a = get("activations.x");
    close(&silu(&a).unwrap(), "activations.silu", 1e-6);
    close(&gelu_approx(&a).unwrap(), "activations.gelu_approx", 1e-6);
    close(
        &gelu_fast_approx(&a).unwrap(),
        "activations.gelu_fast_approx",
        1e-6,
    );
    close(&relu(&a).unwrap(), "activations.relu", 0.0);

    let lora = LoraLinear {
        base: LoraBase::Linear(Linear {
            weight: get("lora.weight"),
            bias: Some(get("lora.bias")),
        }),
        lora_a: get("lora.lora_a"),
        lora_b: get("lora.lora_b"),
        scale: 20.0,
    };
    close(&lora.forward(&get("lora.x")).unwrap(), "lora.out", 1e-5);
    close(&lora.fuse().unwrap().weight, "lora.fused_weight", 1e-6);
}

#[test]
fn lora_trains_only_the_adapter_with_mlx_lm_names() {
    random::seed(0);
    let base = Linear::new(4, 3, true).unwrap();
    let lora = LoraLinear::from_base(LoraBase::Linear(base), 2, 20.0).unwrap();
    let names = |ps: Vec<(String, Array)>| ps.into_iter().map(|(n, _)| n).collect::<Vec<_>>();
    assert_eq!(
        names(lora.parameters()),
        ["linear.weight", "linear.bias", "lora_a", "lora_b"]
    );
    assert_eq!(names(lora.trainable_parameters()), ["lora_a", "lora_b"]);
}

fn run_optimizer(name: &str, mut opt: impl Optimizer, rel: f32) {
    let mut params = vec![get("optim.matrix"), get("optim.vector")];
    for step in 0..3 {
        let grads = [
            get(&format!("optim.g{step}_matrix")),
            get(&format!("optim.g{step}_vector")),
        ];
        params = opt.apply(&params, &grads).unwrap();
        close(&params[0], &format!("optim_{name}.s{step}_matrix"), rel);
        close(&params[1], &format!("optim_{name}.s{step}_vector"), rel);
    }
}

#[test]
fn optimizers_equal_mlx() {
    run_optimizer(
        "sgd",
        Sgd::new(0.1)
            .momentum(0.9)
            .weight_decay(0.01)
            .nesterov(true),
        1e-6,
    );
    run_optimizer("adam", Adam::new(0.1).bias_correction(true), 1e-6);
    run_optimizer("adamw", Adam::adamw(0.1).weight_decay(0.1), 1e-6);
    run_optimizer("lion", Lion::new(0.1).weight_decay(0.1), 1e-6);
    // Five chained matrix products per step.
    run_optimizer("muon", Muon::new(0.1), 1e-5);
    run_optimizer(
        "adamw_cosine",
        Adam::adamw(schedules::cosine_decay(0.1, 4, 0.01)),
        1e-6,
    );
}

#[test]
fn schedules_equal_mlx() {
    let check = |name: &str, s: LearningRate| {
        // MLX's CPU cosine is its own polynomial: within an ulp of the platform's.
        let got = Array::from_slice(&(0..20).map(|t| s.at(t)).collect::<Vec<f32>>(), &[20]);
        close(&got.unwrap(), &format!("schedule.{name}"), 1e-6);
    };
    check("exponential", schedules::exponential_decay(0.1, 0.9));
    check("step", schedules::step_decay(0.1, 0.5, 3));
    check("cosine", schedules::cosine_decay(0.1, 10, 0.01));
    check("linear", schedules::linear_schedule(0.0, 0.1, 5));
    check(
        "joined",
        schedules::join_schedules(
            vec![
                schedules::linear_schedule(0.0, 0.1, 5),
                schedules::cosine_decay(0.1, 10, 0.0),
            ],
            vec![5],
        ),
    );
}

#[test]
fn clip_grad_norm_equals_mlx() {
    let (clipped, norm) = clip_grad_norm(&[get("clip.g0"), get("clip.g1")], 1.0).unwrap();
    close(&norm, "clip.norm", 1e-6);
    close(&clipped[0], "clip.c0", 1e-6);
    assert_eq!(clipped[1].dtype(), DType::F16);
    close(&clipped[1], "clip.c1", 1e-3);
}

#[test]
fn weights_load_by_name() {
    random::seed(0);
    let mut model = vec![
        Linear::new(2, 3, true).unwrap(),
        Linear::new(3, 1, false).unwrap(),
    ];
    let w = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]).unwrap();
    model
        .load_weights([("1.weight".to_owned(), w.clone())], false)
        .unwrap();
    assert_eq!(f32s(&model[1].weight), [1.0, 2.0, 3.0]);
    // Strict: everything must be given, nothing unknown, shapes must match.
    assert!(
        model
            .load_weights([("1.weight".to_owned(), w.clone())], true)
            .is_err()
    );
    let wrong = Array::from_slice(&[1.0f32, 2.0], &[1, 2]).unwrap();
    assert!(
        model
            .load_weights([("1.weight".to_owned(), wrong)], false)
            .is_err()
    );
    let mut all: Vec<(String, Array)> = model.parameters();
    all.push(("2.weight".to_owned(), w));
    assert!(model.load_weights(all, true).is_err(), "unknown name");
}
