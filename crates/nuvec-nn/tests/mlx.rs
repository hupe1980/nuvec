//! `nuvec-nn` against `mlx.nn` and `mlx.optimizers` 0.32.4: inputs and expected outputs
//! recorded by `scripts/nn_fixtures.py` (`just nn-fixtures`) on MLX's CPU device.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use nuvec::io::json::{self, Json};
use nuvec::{Array, DType, Param, Tree, array, random};
use nuvec_nn::layers::LoraBase;
use nuvec_nn::optim::{
    Adam, AdamW, LearningRate, Lion, MultiOptimizer, Muon, Optimizer, Sgd, clip_grad_norm,
    schedules,
};
use nuvec_nn::{
    Embedding, LayerNorm, Linear, LoraLinear, Module, QuantizedEmbedding, QuantizedLinear, RmsNorm,
    Rope, gelu_approx, gelu_fast_approx, relu, silu,
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
    a.astype(DType::F32).to_vec::<f32>().unwrap()
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
    let e = Embedding::new(10, 4);
    assert_eq!(
        f32s(&e.weight),
        f32s(&get("init_embedding.weight")),
        "bit for bit"
    );
    random::seed(2);
    let base = Linear::new(6, 5);
    let lora = LoraLinear::from_base(base, 3);
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
        weight: Param::new(get("embedding.weight")),
    };
    close(&emb.forward(&get("embedding.ids")), "embedding.out", 0.0);
    close(
        &emb.as_linear(&get("embedding.x")),
        "embedding.as_linear",
        1e-6,
    );

    let x = get("rms_norm.x");
    let mut rms = RmsNorm::new(8).eps(1e-5);
    rms.weight = Param::new(get("rms_norm.weight"));
    close(&rms.forward(&x), "rms_norm.out", 1e-6);

    let mut ln = LayerNorm::new(8);
    ln.weight = Some(Param::new(get("layer_norm.weight")));
    ln.bias = Some(Param::new(get("layer_norm.bias")));
    close(&ln.forward(&get("layer_norm.x")), "layer_norm.out", 1e-6);
    let plain = LayerNorm::new(8).affine(false);
    close(
        &plain.forward(&get("layer_norm.x")),
        "layer_norm.plain",
        1e-6,
    );

    let q = get("rope.x");
    close(&Rope::new(8).forward(&q, 3), "rope.out", 1e-6);
    let traditional = Rope::new(12).traditional();
    close(&traditional.forward(&q, 0), "rope.traditional", 1e-6);

    let a = get("activations.x");
    close(&silu(&a), "activations.silu", 1e-6);
    close(&gelu_approx(&a), "activations.gelu_approx", 1e-6);
    close(&gelu_fast_approx(&a), "activations.gelu_fast_approx", 1e-6);
    close(&relu(&a), "activations.relu", 0.0);

    let mut lora = LoraLinear::from_base(
        Linear {
            weight: Param::new(get("lora.weight")),
            bias: Some(Param::new(get("lora.bias"))),
        },
        3,
    );
    lora.lora_a = Param::new(get("lora.lora_a"));
    lora.lora_b = Param::new(get("lora.lora_b"));
    close(&lora.forward(&get("lora.x")), "lora.out", 1e-5);
    close(&lora.fuse().weight, "lora.fused_weight", 1e-6);
    assert!(matches!(lora.base, LoraBase::Linear(_)));
}

#[test]
fn lora_trains_only_the_adapter_with_mlx_lm_names() {
    random::seed(0);
    let base = Linear::new(4, 3);
    let lora = LoraLinear::from_base(base, 2);
    let names = |ps: Vec<(String, Array)>| ps.into_iter().map(|(n, _)| n).collect::<Vec<_>>();
    assert_eq!(
        names(lora.parameters()),
        ["linear.weight", "linear.bias", "lora_a", "lora_b"]
    );
    assert_eq!(names(lora.trainable_parameters()), ["lora_a", "lora_b"]);
}

/// The fixtures' parameters as a tree, and its three gradient trees.
fn optim_tree() -> (BTreeMap<String, Array>, Vec<BTreeMap<String, Array>>) {
    let params: BTreeMap<String, Array> = [
        ("matrix".to_owned(), get("optim.matrix")),
        ("vector".to_owned(), get("optim.vector")),
    ]
    .into();
    let grads = (0..3)
        .map(|step| {
            [
                ("matrix".to_owned(), get(&format!("optim.g{step}_matrix"))),
                ("vector".to_owned(), get(&format!("optim.g{step}_vector"))),
            ]
            .into()
        })
        .collect();
    (params, grads)
}

fn run_optimizer(name: &str, mut opt: impl Optimizer, rel: f32) {
    let (mut params, grads) = optim_tree();
    for (step, g) in grads.iter().enumerate() {
        opt.update(&mut params, g).unwrap();
        close(
            &params["matrix"],
            &format!("optim_{name}.s{step}_matrix"),
            rel,
        );
        close(
            &params["vector"],
            &format!("optim_{name}.s{step}_vector"),
            rel,
        );
    }
}

#[test]
fn optimizers_equal_mlx() {
    run_optimizer(
        "sgd",
        Sgd::new(0.1).momentum(0.9).weight_decay(0.01).nesterov(),
        1e-6,
    );
    run_optimizer("adam", Adam::new(0.1).bias_correction(), 1e-6);
    run_optimizer("adamw", AdamW::new(0.1).weight_decay(0.1), 1e-6);
    run_optimizer("lion", Lion::new(0.1).weight_decay(0.1), 1e-6);
    // Five chained matrix products per step.
    run_optimizer("muon", Muon::new(0.1), 1e-5);
    run_optimizer(
        "adamw_cosine",
        AdamW::new(schedules::cosine_decay(0.1, 4, 0.01)),
        1e-6,
    );
    // MLX's AdamW defaults (weight decay 0.01).
    run_optimizer("adamw_default", AdamW::new(1e-3), 1e-6);
    // A boxed optimizer behaves as the concrete one.
    let boxed: Box<dyn Optimizer> = Box::new(AdamW::new(1e-3));
    run_optimizer("adamw_default", boxed, 1e-6);
}

/// The learning rate is a 0-D float32 array in the optimizer's state, equal to MLX's
/// `state["learning_rate"]` at every step (constant and scheduled), next to `step`.
#[test]
fn learning_rate_and_step_are_state_arrays() {
    let cases: [(&str, Box<dyn Optimizer>); 3] = [
        (
            "sgd",
            Box::new(Sgd::new(0.1).momentum(0.9).weight_decay(0.01).nesterov()),
        ),
        (
            "adamw_cosine",
            Box::new(AdamW::new(schedules::cosine_decay(0.1, 4, 0.01))),
        ),
        (
            "sgd_joined",
            Box::new(Sgd::new(
                schedules::join_schedules(
                    vec![
                        schedules::linear_schedule(0.0, 0.1, 2),
                        schedules::cosine_decay(0.1, 3, 0.0),
                    ],
                    vec![2],
                )
                .unwrap(),
            )),
        ),
    ];
    for (name, mut opt) in cases {
        let (mut params, grads) = optim_tree();
        let want_lr = f32s(&get(&format!("state_{name}.learning_rate")));
        let want_step = get(&format!("state_{name}.step")).to_vec::<u64>().unwrap();
        for (i, g) in grads.iter().chain(&grads[..2]).enumerate() {
            opt.update(&mut params, g).unwrap();
            let lr = opt.learning_rate();
            assert_eq!((lr.dtype(), lr.ndim()), (DType::F32, 0), "{name}");
            let got = lr.item::<f32>().unwrap();
            assert!(
                (got - want_lr[i]).abs() <= 1e-6 * want_lr[i].abs().max(1.0),
                "{name} step {i}: lr {got} vs MLX {}",
                want_lr[i]
            );
            let state = opt.flatten();
            let step = state.iter().find(|(p, _)| p == "step").unwrap();
            assert_eq!(step.1.item::<u64>().unwrap(), want_step[i], "{name} step");
        }
    }
}

/// `MultiOptimizer`: the first matching group wins, the default takes the rest (MLX's
/// `MultiOptimizer` with the same filters).
#[test]
fn multi_optimizer_equals_mlx() {
    let names = ["layer.weight", "layer.lora_a", "layer.lora_b", "out.weight"];
    let tree = |prefix: &str| -> BTreeMap<String, Array> {
        names
            .iter()
            .map(|n| (n.to_string(), get(&format!("multi.{prefix}{n}"))))
            .collect()
    };
    let mut params = tree("p_");
    let mut opt = MultiOptimizer::new(Sgd::new(1e-2).weight_decay(1e-4))
        .group(|p| p.contains("lora_a"), Adam::new(1e-3));
    for step in 0..3 {
        let g = tree(&format!("g{step}_"));
        opt.update(&mut params, &g).unwrap();
        for n in names {
            close(&params[n], &format!("multi.s{step}_{n}"), 1e-6);
        }
    }
    // Each group keeps the state of the parameters it selected.
    let paths = opt.paths();
    assert!(
        paths.iter().any(|p| p == "groups.0.slots.layer.lora_a.m"),
        "{paths:?}"
    );
    assert!(
        paths.iter().any(|p| p == "default.slots.out.weight.v"),
        "{paths:?}"
    );
    assert!(
        !paths.iter().any(|p| p.starts_with("groups.0.slots.out")),
        "{paths:?}"
    );
}

/// Adam over two layers with `a` frozen for two steps keeps `a`'s moments and matches MLX
/// for the whole sequence.
#[test]
fn freezing_keeps_optimizer_state_as_mlx() {
    #[derive(Clone, Module)]
    struct Pair {
        a: Linear,
        b: Linear,
    }
    let layer = |n: &str| Linear {
        weight: Param::new(get(&format!("freeze_init.{n}.weight"))),
        bias: Some(Param::new(get(&format!("freeze_init.{n}.bias")))),
    };
    let mut model = Pair {
        a: layer("a"),
        b: layer("b"),
    };
    let scale = |path: &str| match path {
        "a.weight" => 0.1f32,
        "a.bias" => -0.2,
        "b.weight" => 0.3,
        _ => -0.4,
    };
    let mut opt = Adam::new(0.01);
    let mut after_step_3: Vec<(String, Array)> = Vec::new();
    for step in 0..6 {
        if step == 3 {
            model.freeze(|p| p.starts_with("a.")).unwrap();
            after_step_3 = opt.flatten();
        }
        if step == 5 {
            model.unfreeze(|p| p.starts_with("a.")).unwrap();
            // The unfrozen layer's moments are those it had after step 3.
            let now = opt.flatten();
            for (p, a) in after_step_3
                .iter()
                .filter(|(p, _)| p.starts_with("slots.a."))
            {
                let (_, b) = now.iter().find(|(q, _)| q == p).unwrap();
                assert_eq!(f32s(a), f32s(b), "{p}");
            }
        }
        let grads = model.map(|path, a| nuvec::ops::full(a.shape(), scale(path)));
        opt.update(&mut model, &grads).unwrap();
        for (path, a) in model.parameters() {
            close(&a, &format!("freeze.s{step}_{path}"), 1e-6);
        }
    }
    for (path, a) in opt.flatten() {
        if let Some(stripped) = path.strip_prefix("slots.") {
            close(&a, &format!("freeze_state.{stripped}"), 1e-6);
        }
    }
}

#[test]
fn schedules_equal_mlx() {
    let check = |name: &str, s: LearningRate| {
        // MLX's CPU cosine is its own polynomial: within an ulp of the platform's.
        let rates: Vec<Array> = (0..20u64).map(|t| s.at(&Array::from(t))).collect();
        let got = nuvec::ops::stack(&rates, 0);
        assert_eq!(got.dtype(), DType::F32);
        close(&got, &format!("schedule.{name}"), 1e-6);
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
        )
        .unwrap(),
    );
}

#[test]
fn clip_grad_norm_equals_mlx() {
    let grads = (get("clip.g0"), get("clip.g1"));
    let ((c0, c1), norm) = clip_grad_norm(&grads, 1.0);
    close(&norm, "clip.norm", 1e-6);
    close(&c0, "clip.c0", 1e-6);
    assert_eq!(c1.dtype(), DType::F16);
    close(&c1, "clip.c1", 1e-3);
    let (_, bad) = clip_grad_norm(&grads, -1.0);
    assert!(
        bad.check().is_err(),
        "a negative max_norm is a failed array"
    );
}

#[test]
fn weights_load_by_name() {
    random::seed(0);
    let mut model = vec![Linear::new(2, 3), Linear::new(3, 1).without_bias()];
    let w = array![[1.0, 2.0, 3.0]];
    model
        .load_weights_partial(vec![("1.weight".to_owned(), w.clone())])
        .unwrap();
    assert_eq!(f32s(&model[1].weight), [1.0, 2.0, 3.0]);
    // Strict: everything must be given, nothing unknown, shapes must match.
    assert!(
        model
            .load_weights(vec![("1.weight".to_owned(), w.clone())])
            .is_err()
    );
    let wrong = array![[1.0, 2.0]];
    assert!(
        model
            .load_weights_partial(vec![("1.weight".to_owned(), wrong)])
            .is_err()
    );
    let mut all: Vec<(String, Array)> = model.parameters();
    all.push(("2.weight".to_owned(), w));
    assert!(model.load_weights(all).is_err(), "unknown name");
}

fn names_fixture() -> Json {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mlx_nn_names.json"
    );
    json::parse(&std::fs::read(path).unwrap()).unwrap()
}

fn fixture_names(key: &str) -> Vec<String> {
    names_fixture()
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("no fixture names {key}"))
        .iter()
        .map(|n| n.as_str().unwrap().to_owned())
        .collect()
}

fn names(m: &impl Module) -> Vec<String> {
    m.parameters().into_iter().map(|(n, _)| n).collect()
}

/// Every layer's parameter names equal MLX 0.32.4's `tree_flatten` names.
#[test]
fn parameter_names_equal_mlx() {
    random::seed(0);
    let linear = Linear::new(64, 8);
    assert_eq!(names(&Linear::new(4, 3)), fixture_names("Linear"));
    assert_eq!(
        names(&Linear::new(4, 3).without_bias()),
        fixture_names("Linear_no_bias")
    );
    assert_eq!(
        names(&QuantizedLinear::from_linear(&linear)),
        fixture_names("QuantizedLinear")
    );
    assert_eq!(names(&Embedding::new(10, 4)), fixture_names("Embedding"));
    assert_eq!(
        names(&QuantizedEmbedding::from_embedding(&Embedding::new(10, 64))),
        fixture_names("QuantizedEmbedding")
    );
    assert_eq!(names(&RmsNorm::new(8)), fixture_names("RMSNorm"));
    assert_eq!(names(&LayerNorm::new(8)), fixture_names("LayerNorm"));
    assert_eq!(
        names(&LayerNorm::new(8).without_bias()),
        fixture_names("LayerNorm_no_bias")
    );
    assert_eq!(
        names(&LayerNorm::new(8).affine(false)),
        fixture_names("LayerNorm_no_affine")
    );
    assert_eq!(names(&Rope::new(8)), fixture_names("RoPE"));
    let lora = LoraLinear::from_base(Linear::new(6, 5), 3);
    assert_eq!(names(&lora), fixture_names("LoRALinear"));
    let trainable: Vec<String> = lora
        .trainable_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(trainable, fixture_names("LoRALinear_trainable"));
    // MLX's `QuantizedLinear.unfreeze()` makes the packed weight, scales and biases
    // trainable; nuvec keeps them buffers (never trained), so only the bias trains.
    let mut q = QuantizedLinear::from_linear(&linear);
    q.unfreeze(|_| true).unwrap();
    let q_trainable: Vec<String> = q
        .trainable_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(q_trainable, ["bias"]);
    assert_eq!(
        fixture_names("QuantizedLinear_trainable_after_unfreeze"),
        ["weight", "scales", "biases", "bias"],
        "the recorded MLX behaviour"
    );
    // MLX's MNIST model: names and totals.
    #[derive(Clone, Module)]
    struct Mlp {
        layers: Vec<Linear>,
    }
    let mlp = Mlp {
        layers: vec![
            Linear::new(784, 32),
            Linear::new(32, 32),
            Linear::new(32, 10),
        ],
    };
    assert_eq!(names(&mlp), fixture_names("MLP"));
    let total = names_fixture()
        .get("MLP_total")
        .and_then(Json::as_i64)
        .unwrap();
    assert_eq!(mlp.num_params() as i64, total);
    assert_eq!(mlp.num_trainable() as i64, total);
}

/// The model of `mlx_save_weights.safetensors` (MLX's `save_weights`).
#[derive(Clone, Module)]
struct Small {
    dense: Linear,
    quantized: QuantizedLinear,
    norm: LayerNorm,
}

fn small() -> Small {
    random::seed(9);
    Small {
        dense: Linear::new(8, 4),
        quantized: QuantizedLinear::from_linear(&Linear::new(64, 4)),
        norm: LayerNorm::new(4),
    }
}

fn mlx_file() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mlx_save_weights.safetensors")
}

/// nuvec reads MLX's `save_weights` files, and writes files with the same names, dtypes and
/// values.
#[test]
fn save_weights_round_trips_with_mlx() {
    let mlx = nuvec::io::load_safetensors(mlx_file()).unwrap();
    let mut model = small();
    model.load_weights(mlx_file()).unwrap();
    for (name, a) in model.parameters() {
        assert_eq!(f32s(&a), f32s(&mlx.tensors[&name]), "{name}");
        assert_eq!(a.dtype(), mlx.tensors[&name].dtype(), "{name}");
    }
    let path =
        std::env::temp_dir().join(format!("nuvec-nn-{}-small.safetensors", std::process::id()));
    model.save_weights(&path).unwrap();
    let ours = nuvec::io::load_safetensors(&path).unwrap();
    assert_eq!(
        ours.tensors.keys().collect::<Vec<_>>(),
        mlx.tensors.keys().collect::<Vec<_>>()
    );
    for (name, a) in &ours.tensors {
        let b = &mlx.tensors[name];
        assert_eq!((a.dtype(), a.shape()), (b.dtype(), b.shape()), "{name}");
        assert_eq!(f32s(a), f32s(b), "{name}");
    }
}

/// MLX loads nuvec's `save_weights` output (needs `uv` and MLX; run with `--ignored`).
#[test]
#[ignore = "runs MLX through uv"]
fn mlx_loads_nuvec_save_weights() {
    let mut model = small();
    model.load_weights(mlx_file()).unwrap();
    let path = std::env::temp_dir().join(format!(
        "nuvec-nn-{}-verify.safetensors",
        std::process::id()
    ));
    model.save_weights(&path).unwrap();
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/nn_fixtures.py");
    let out = std::process::Command::new("uv")
        .args(["run", script, "--verify"])
        .arg(&path)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("verified"),
        "{stdout}"
    );
}

/// `Dropout` equals MLX's `nn.Dropout` bit for bit for seeds 0 and 1 and p ∈ {0.1, 0.5}.
#[test]
fn dropout_equals_mlx() {
    use nuvec_nn::Dropout;
    let x = get("dropout.x");
    for seed in [0u64, 1] {
        for (p, tag) in [(0.1f32, 1), (0.5, 5)] {
            random::seed(seed);
            let got = Dropout::new(p).forward(&x);
            assert_eq!(
                f32s(&got),
                f32s(&get(&format!("dropout.s{seed}_p{tag}"))),
                "seed {seed}, p {p}: bit for bit"
            );
        }
    }
}

#[test]
fn dropout_modes_keys_and_errors() {
    use nuvec_nn::Dropout;
    #[derive(Clone, Module)]
    struct Block {
        dense: Linear,
        drop: Dropout,
        inner: Vec<Dropout>,
    }
    random::seed(3);
    let mut block = Block {
        dense: Linear::new(4, 4),
        drop: Dropout::new(0.5),
        inner: vec![Dropout::new(0.3)],
    };
    let x = get("dropout.x").i((.., ..4));
    block.train(false);
    assert!(!block.training());
    assert_eq!(
        f32s(&block.drop.forward(&x)),
        f32s(&x),
        "identity in eval mode"
    );
    assert_eq!(f32s(&block.inner[0].forward(&x)), f32s(&x));
    block.train(true);
    assert!(block.training() && block.inner[0].training);
    // An explicit key gives the same mask every time.
    let key = random::key(42);
    let a = block.drop.forward_with_key(&x, &key);
    let b = block.drop.forward_with_key(&x, &key);
    assert_eq!(f32s(&a), f32s(&b));
    // An invalid probability is a failed array naming it (no panic).
    let err = Dropout::new(1.0)
        .forward(&x)
        .check()
        .unwrap_err()
        .to_string();
    assert!(err.contains("dropout") && err.contains("[0, 1)"), "{err}");
    assert!(Dropout::new(-0.1).forward(&x).is_failed());
}

/// Dropout inside a checkpointed block redraws the same mask when the block is recomputed,
/// so the gradients equal the uncheckpointed ones.
#[test]
fn checkpointed_dropout_keeps_the_gradients() {
    use nuvec_nn::Dropout;
    nuvec::set_default_stream(nuvec::Stream::default_for(nuvec::Device::cpu()));
    let x = get("dropout.x");
    let drop = Dropout::new(0.5);
    let d = drop.clone();
    let block = move |x: &Array| d.forward(x).tanh();
    random::seed(21);
    let plain = nuvec::grad(|x: &Array| block(x).sum(..))(&x).unwrap();
    let block = nuvec::checkpoint(move |x: &Array| drop.forward(x).tanh());
    random::seed(21);
    let through = nuvec::grad(move |x: &Array| Ok(block(x)?.sum(..)))(&x).unwrap();
    assert_eq!(f32s(&plain), f32s(&through));
}

/// `LoraLinear::rslora()` scales by `α/√r` (`scale · √r`), and `.dropout(p)` drops only the
/// update's input (the base sees `x` unchanged), with MLX's masks for a seed.
#[test]
fn lora_dropout_and_rslora() {
    use nuvec_nn::Dropout;
    random::seed(30);
    let lora = LoraLinear::from_base(Linear::new(8, 4), 4);
    assert_eq!(lora.clone().rslora().scale, 40.0);
    let mut lora = lora.dropout(0.5);
    lora.lora_b = Param::new(random::normal([4, 4], DType::F32, 0.0, 1.0, None));
    let x = get("dropout.x");
    random::seed(31);
    let got = lora.forward(&x);
    let LoraBase::Linear(base) = lora.base.clone() else {
        unreachable!()
    };
    random::seed(31);
    let dropped = Dropout::new(0.5).forward(&x);
    let z = dropped.matmul(&lora.lora_a).matmul(&lora.lora_b);
    let want = base.forward(&x).add(z.multiply(20.0).astype(x.dtype()));
    assert_eq!(f32s(&got), f32s(&want));
    // In evaluation mode the update sees `x` itself.
    lora.train(false);
    let eval = lora.forward(&x);
    let full = base
        .forward(&x)
        .add(x.matmul(&lora.lora_a).matmul(&lora.lora_b).multiply(20.0));
    assert_eq!(f32s(&eval), f32s(&full));
}
