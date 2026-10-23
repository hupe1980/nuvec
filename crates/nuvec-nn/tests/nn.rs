//! `nuvec-nn` behaviour: derived modules, introspection, freezing, train/eval mode, loading
//! and saving weights, optimizers' path-keyed state and the layers against
//! values recorded from MLX 0.32.4.

use std::collections::{BTreeMap, HashMap};

use nuvec::{
    Array, DType, Error, OpErrorKind, Param, Tree, array, ops, random, value_and_grad,
    value_and_grad_flat,
};
use nuvec_nn::losses::{Reduction, cross_entropy};
use nuvec_nn::optim::{Adam, AdamW, Optimizer, Sgd};
use nuvec_nn::{
    Dropout, LayerNorm, Linear, LoraLinear, Module, QuantizedLinear, RmsNorm, Rope, Summary, relu,
};

fn values(a: &Array) -> Vec<f32> {
    a.astype(DType::F32).to_vec::<f32>().unwrap()
}

fn names(ps: Vec<(String, Array)>) -> Vec<String> {
    ps.into_iter().map(|(n, _)| n).collect()
}

fn close(got: &[f32], expected: &[f64], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}");
    for (g, e) in got.iter().zip(expected) {
        assert!(
            (f64::from(*g) - e).abs() <= 1e-6 * e.abs().max(1.0),
            "{what}: {g} vs MLX {e}"
        );
    }
}

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("nuvec-nn-{}-{name}", std::process::id()))
}

#[test]
fn linear_initialisation_equals_mlx() {
    random::seed(0);
    let l = Linear::new(3, 2);
    let w: Vec<f32> = [
        -0.4937573969364166,
        -0.5538788437843323,
        -0.4882282614707947,
        0.4372609853744507,
        -0.387317955493927,
        -0.4599017798900604,
    ]
    .map(|v: f64| v as f32)
    .to_vec();
    assert_eq!(values(&l.weight), w, "bit for bit");
    let b: Vec<f32> = [0.004011034965515137f64, -0.022510409355163574]
        .map(|v| v as f32)
        .to_vec();
    assert_eq!(values(l.bias.as_ref().unwrap()), b);
    assert_eq!(l.weight.shape(), [2, 3]);
}

/// `Linear::new(i, o).without_bias()` draws exactly what MLX's `nn.Linear(i, o,
/// bias=False)` draws: the next layer's initialisation is unchanged.
#[test]
fn without_bias_undoes_the_bias_draw() {
    random::seed(4);
    let a = Linear::new(4, 3).without_bias();
    let next = Linear::new(2, 2);
    random::seed(4);
    let scale = (1.0f64 / 4.0).sqrt();
    let w = random::uniform(-scale, scale, [3, 4], DType::F32, None);
    let expected = Linear::new(2, 2);
    assert!(a.bias.is_none());
    assert_eq!(values(&a.weight), values(&w));
    assert_eq!(values(&next.weight), values(&expected.weight));
}

#[test]
fn cross_entropy_equals_mlx() {
    let logits = array![[2.0, -1.0], [-1.0, 2.0]];
    let targets = array![0, 1];
    let none = cross_entropy(&logits, &targets, Reduction::None);
    close(
        &values(&none),
        &[0.04858732968568802, 0.04858732968568802],
        "none",
    );
    let mean = cross_entropy(&logits, &targets, Reduction::Mean);
    close(&values(&mean), &[0.04858732968568802], "mean");
    let logits = Array::from_slice(&[0.5f32, 1.5, -0.3, 2.0, 0.1, 0.4], [2, 3]);
    let targets = array![2u32, 0];
    let loss = cross_entropy(&logits, &targets, Reduction::None);
    close(
        &values(&loss),
        &[2.2273428440093994, 0.30118927359580994],
        "three classes",
    );
    let floats = array![0.0, 1.0];
    assert!(
        cross_entropy(&logits, &floats, Reduction::None)
            .check()
            .is_err()
    );
    let wrong = array![0u32, 1, 2];
    assert!(
        cross_entropy(&logits, &wrong, Reduction::None)
            .check()
            .is_err()
    );
}

/// Two steps from `w = [1, 2, 3]` with the constant gradient `[0.1, -0.2, 0.3]`.
fn steps(opt: &mut impl Optimizer) -> Vec<Vec<f32>> {
    let mut w = Param::new(array![1.0, 2.0, 3.0]);
    let g = Param::new(array![0.1, -0.2, 0.3]);
    let mut out = Vec::new();
    for _ in 0..2 {
        opt.update(&mut w, &g).unwrap();
        out.push(values(&w));
    }
    out
}

/// An optimizer's name, its two steps, and MLX's values for them.
type Case = (&'static str, Vec<Vec<f32>>, [[f64; 3]; 2]);

#[test]
fn optimizers_equal_mlx() {
    let cases: [Case; 5] = [
        (
            "sgd",
            steps(&mut Sgd::new(0.1)),
            [[0.99, 2.02, 2.97], [0.98, 2.04, 2.94]],
        ),
        (
            "sgd momentum + weight decay",
            steps(&mut Sgd::new(0.1).momentum(0.9).weight_decay(0.01)),
            [
                [0.9890000224113464, 2.0179998874664307, 2.9670000076293945],
                [0.9681110382080078, 2.0521819591522217, 2.9043331146240234],
            ],
        ),
        (
            "sgd nesterov",
            steps(&mut Sgd::new(0.1).momentum(0.9).nesterov()),
            [
                [0.9810000061988831, 2.0380001068115234, 2.943000078201294],
                [0.9538999795913696, 2.0922000408172607, 2.8617000579833984],
            ],
        ),
        (
            "adam",
            steps(&mut Adam::new(0.01)),
            [
                [0.9683773517608643, 2.0316226482391357, 2.9683773517608643],
                [0.9258815050125122, 2.0741186141967773, 2.9258813858032227],
            ],
        ),
        (
            "adam bias correction",
            steps(&mut Adam::new(0.01).bias_correction()),
            [
                [0.9900000691413879, 2.009999990463257, 2.990000009536743],
                [0.9800001978874207, 2.0199999809265137, 2.9800000190734863],
            ],
        ),
    ];
    for (name, got, expected) in cases {
        for (step, (g, e)) in got.iter().zip(&expected).enumerate() {
            close(g, e, &format!("{name} step {}", step + 1));
        }
    }
}

#[derive(Clone, Module)]
struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(sizes: &[i32]) -> Self {
        Self {
            layers: sizes.windows(2).map(|w| Linear::new(w[0], w[1])).collect(),
        }
    }

    fn forward(&self, x: &Array) -> Array {
        let mut h = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            h = l.forward(&h);
            if i + 1 < self.layers.len() {
                h = relu(&h);
            }
        }
        h
    }
}

#[test]
fn value_and_grad_of_a_model_equals_the_flat_transform_and_training_learns() {
    random::seed(3);
    let mut model = Mlp::new(&[4, 8, 3]);
    let x = random::normal([16, 4], DType::F32, 0.0, 1.0, None);
    let labels: Vec<u32> = (0..16).map(|i| i % 3).collect();
    let y = Array::from_slice(&labels, [16]);
    // Class-dependent shift so the problem is learnable.
    let shift = Array::from_slice(
        &labels.iter().map(|&l| l as f32).collect::<Vec<_>>(),
        [16, 1],
    );
    let x = x.add(&shift);
    let loss = |m: &Mlp| cross_entropy(&m.forward(&x), &y, Reduction::Mean);

    let (value, grads) = value_and_grad(loss)(&model).unwrap();
    assert_eq!(
        grads.paths(),
        model.paths(),
        "gradients have the model's type"
    );
    assert_eq!(grads.layers[0].weight.shape(), [8, 4]);
    let leaves = model.leaves();
    let flat = |ps: &[Array]| loss(&model.unflatten(ps.iter().cloned())?).check();
    let (core_value, core_grads) = value_and_grad_flat(flat, &[0, 1, 2, 3])(&leaves).unwrap();
    assert_eq!(values(&value), values(&core_value));
    for (a, b) in grads.leaves().iter().zip(&core_grads) {
        assert_eq!(values(a), values(b));
    }

    let mut opt = Adam::new(0.05);
    let start = values(&value)[0];
    for _ in 0..60 {
        let (_, grads) = value_and_grad(loss)(&model).unwrap();
        opt.update(&mut model, &grads).unwrap();
        nuvec::eval(&model).unwrap();
    }
    let end = values(&value_and_grad(loss)(&model).unwrap().0)[0];
    assert!(end < 0.3 * start, "loss {start} -> {end}");
    // Gradients with other paths are an error.
    let other = Mlp::new(&[4, 8, 3, 2]);
    assert!(opt.update(&mut model, &other).is_err());
}

#[test]
fn quantized_linear_equals_mlx() {
    random::seed(0);
    let q = QuantizedLinear::from_linear_with(&Linear::new(64, 3), 32, 4);
    assert_eq!(q.weight.shape(), [3, 8]);
    assert_eq!(q.scales.shape(), [3, 2]);
    let x = ops::arange(0.0, 128.0, 1.0);
    let x = (x.reshape([2, 64]) / 64.0) - 0.5;
    // mlx.nn.QuantizedLinear.from_linear(nn.Linear(64, 3), group_size=32, bits=4) after
    // mx.random.seed(0), on the CPU.
    let expected = [
        0.3646765351295471,
        -0.15546883642673492,
        0.23967400193214417,
        -0.7248094081878662,
        -0.07878664135932922,
        -0.29476773738861084,
    ];
    close(&values(&q.forward(&x)), &expected, "QuantizedLinear");
    // Only the bias trains; gradients reach it through the layer.
    assert_eq!(
        names(q.parameters()),
        ["weight", "scales", "biases", "bias"]
    );
    assert_eq!(names(q.trainable_parameters()), ["bias"]);
    let (_, grads) = value_and_grad(|m: &QuantizedLinear| m.forward(&x).sum(..))(&q).unwrap();
    assert_eq!(values(grads.bias.as_ref().unwrap()), vec![2.0; 3]);
    // Quantized tensors are buffers: unfreezing everything never makes them trainable.
    let mut q2 = q.clone();
    assert_eq!(
        q2.unfreeze(|_| true).unwrap(),
        0,
        "the bias was trainable already"
    );
    assert_eq!(names(q2.trainable_parameters()), ["bias"]);
    // An integer parameter is not unfrozen (counted as unchanged).
    let mut ints = (Param::frozen(Array::from_slice(&[1u32, 2], [2])),);
    assert_eq!(ints.unfreeze(|_| true).unwrap(), 0);
    assert_eq!(ints.num_trainable(), 0);
}

/// A model exercising every attribute: a hashed head count, a computed `Array` constant,
/// a renamed field, an absent optional norm, a buffer, an enum, nested modules.
#[derive(Clone, Module)]
struct Attention {
    q_proj: Linear,
    #[tree(rename = "q_norm")]
    qn: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    #[tree(skip)]
    heads: i32,
    #[tree(skip)]
    freqs: Array,
    #[tree(buffer)]
    running_mean: Array,
    gate: Param,
}

/// The same struct with the `#[module(..)]` alias.
#[derive(Clone, Module)]
struct AttentionAlias {
    q_proj: Linear,
    #[module(rename = "q_norm")]
    qn: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    #[module(skip)]
    heads: i32,
    #[module(skip)]
    freqs: Array,
    #[module(buffer)]
    running_mean: Array,
    gate: Param,
}

#[derive(Clone, Module)]
enum Mixer {
    #[tree(rename = "self_attn")]
    Attention(Attention),
    Dense(Linear),
}

fn attention() -> Attention {
    Attention {
        q_proj: Linear::new(4, 4),
        qn: Some(RmsNorm::new(4)),
        k_norm: None,
        heads: 2,
        freqs: array![1.0, 2.0],
        running_mean: ops::zeros([4], DType::F32),
        gate: Param::new(array![0.5]),
    }
}

#[test]
fn derived_modules_use_mlx_names() {
    random::seed(1);
    let a = attention();
    let expected = [
        "q_proj.weight",
        "q_proj.bias",
        "q_norm.weight",
        "running_mean",
        "gate",
    ];
    assert_eq!(names(a.parameters()), expected);
    assert_eq!(
        names(a.trainable_parameters()),
        ["q_proj.weight", "q_proj.bias", "q_norm.weight", "gate"],
        "a buffer is never trainable"
    );
    let alias = AttentionAlias {
        q_proj: a.q_proj.clone(),
        qn: a.qn.clone(),
        k_norm: None,
        heads: 2,
        freqs: a.freqs.clone(),
        running_mean: a.running_mean.clone(),
        gate: a.gate.clone(),
    };
    assert_eq!(
        names(alias.parameters()),
        expected,
        "`#[module(..)]` is an alias"
    );
    let _ = (alias.heads, &alias.freqs);
    // A strict load of a file without the computed constant succeeds.
    let mut loaded = attention();
    loaded.load_weights(a.parameters()).unwrap();
    let _ = (loaded.heads, &loaded.freqs);
    // Enum variants: renamed (adds a segment) and transparent.
    let m = Mixer::Attention(attention());
    assert_eq!(names(m.parameters())[0], "self_attn.q_proj.weight");
    let d = Mixer::Dense(Linear::new(2, 2));
    assert_eq!(names(d.parameters()), ["weight", "bias"]);
}

#[test]
fn introspection() {
    random::seed(0);
    let mlp = Mlp::new(&[784, 32, 32, 10]);
    // MLX's tree_flatten totals (crates/nuvec-nn/tests/fixtures/mlx_nn_names.json).
    assert_eq!(mlp.num_params(), 26506);
    assert_eq!(mlp.num_trainable(), 26506);
    let summary: Summary = mlp.summary();
    assert_eq!(summary.rows.len(), 6);
    assert_eq!(summary.rows[0].path, "layers.0.weight");
    assert_eq!(summary.rows[0].shape, [32, 784]);
    assert_eq!(summary.rows[0].bytes, 32 * 784 * 4);
    assert_eq!(summary.num_params(), 26506);
    let text = summary.to_string();
    assert!(text.starts_with("path"), "{text}");
    assert!(text.contains("layers.2.bias"), "{text}");
    assert!(
        text.ends_with("26,506 params (26,506 trainable), 106,024 bytes"),
        "{text}"
    );
    assert_eq!(
        mlp.display().to_string(),
        "Mlp (26,506 params, 26,506 trainable)\n  layers.0: Linear (25,120)\n  layers.1: Linear (1,056)\n  layers.2: Linear (330)"
    );
    // Nested modules, transparent options and enums, a frozen submodule.
    random::seed(1);
    let mut a = attention();
    a.q_proj.freeze(|_| true).unwrap();
    assert_eq!(
        a.display().to_string(),
        "Attention (29 params, 5 trainable)\n  q_proj: Linear (20, frozen)\n  q_norm: RmsNorm (4)"
    );
    let m = Mixer::Attention(attention());
    assert!(
        m.display()
            .to_string()
            .contains("\n  self_attn: Attention (29)\n    self_attn.q_proj: Linear (20)"),
        "{}",
        m.display()
    );
}

#[test]
fn strict_loading_reports_every_problem() {
    random::seed(2);
    let mut model = Mlp::new(&[4, 3, 2]);
    let before: Vec<Vec<f32>> = model.leaves().iter().map(values).collect();
    let mut weights: Vec<(String, Array)> = model
        .parameters()
        .into_iter()
        .filter(|(n, _)| n != "layers.1.bias") // missing
        .map(|(n, a)| match n.as_str() {
            "layers.0.weight" => (n, ops::zeros([3, 5], DType::F32)), // shape
            "layers.1.weight" => (n, ops::zeros([1, 3], DType::F32)), // shape
            _ => (n, a),
        })
        .collect();
    weights.push(("layers.9.weight".to_owned(), array![1.0])); // unknown
    let err = model.load_weights(weights.clone()).unwrap_err().to_string();
    assert!(err.contains("load_weights: 4 problems"), "{err}");
    let lines: Vec<&str> = err.lines().skip(1).map(str::trim).collect();
    assert_eq!(
        lines,
        [
            "missing: layers.1.bias",
            "unknown: layers.9.weight",
            "shape: layers.0.weight [3, 5], expected [3, 4]",
            "shape: layers.1.weight [1, 3], expected [2, 3]",
        ]
    );
    let after: Vec<Vec<f32>> = model.leaves().iter().map(values).collect();
    assert_eq!(before, after, "unchanged");
    // Partial: assigns matches, ignores unknown names, still rejects shape mismatches.
    let err = model.load_weights_partial(weights).unwrap_err().to_string();
    assert!(err.contains("load_weights_partial: 2 problems"), "{err}");
    let ones = vec![("layers.1.bias".to_owned(), array![1.0, 1.0])];
    model.load_weights_partial(ones).unwrap();
    assert_eq!(values(model.layers[1].bias.as_ref().unwrap()), [1.0, 1.0]);
    // Sources: a `HashMap`, a `BTreeMap` and a safetensors path.
    let all: HashMap<String, Array> = model.parameters().into_iter().collect();
    model.load_weights(all).unwrap();
    let ordered: BTreeMap<String, Array> = model.parameters().into_iter().collect();
    model.load_weights(ordered).unwrap();
    let path = temp("strict.safetensors");
    model.save_weights(&path).unwrap();
    let mut fresh = Mlp::new(&[4, 3, 2]);
    fresh.load_weights(path.as_path()).unwrap();
    fresh.load_weights(path.to_str().unwrap()).unwrap();
    assert_eq!(values(&fresh.layers[1].bias.clone().unwrap()), [1.0, 1.0]);
}

#[test]
fn saving_weights() {
    random::seed(3);
    let mut model = Mlp::new(&[3, 2]);
    model.layers[0].weight.set_frozen(true);
    let path = temp("trainable.safetensors");
    model.save_trainable(&path).unwrap();
    let file = nuvec::io::load_safetensors(&path).unwrap();
    assert_eq!(file.tensors.keys().collect::<Vec<_>>(), ["layers.0.bias"]);
    // An unloaded placeholder is an error naming it.
    model.layers[0].weight = Param::new(
        Array::failure("load", OpErrorKind::InvalidArgument("not loaded".into()))
            .broadcast_to([2, 3]),
    );
    let err = model.save_weights(temp("bad.safetensors")).unwrap_err();
    assert!(err.to_string().contains("layers.0.weight"), "{err}");
}

/// A generic container with a plain array next to a module.
#[derive(Clone, Module)]
struct WithPlain {
    layer: Linear,
    extra: (Param, Vec<Linear>),
}

#[test]
fn freezing_by_path() {
    random::seed(4);
    let mut m = WithPlain {
        layer: Linear::new(2, 2),
        extra: (Param::new(array![1.0]), vec![Linear::new(2, 1)]),
    };
    assert_eq!(m.freeze(|_| true).unwrap(), 5);
    assert_eq!(m.num_trainable(), 0);
    assert_eq!(m.freeze(|_| true).unwrap(), 0, "already frozen");
    assert_eq!(m.unfreeze(|p| p.contains("bias")).unwrap(), 2);
    assert_eq!(
        names(m.trainable_parameters()),
        ["layer.bias", "extra.1.0.bias"]
    );
    // A plain `Array` leaf cannot be frozen: an error naming it, nothing changed.
    let mut plain = (Param::new(array![1.0]), array![2.0]);
    let err = plain.freeze(|_| true).unwrap_err().to_string();
    assert!(err.contains("\"1\"") || err.contains(": 1"), "{err}");
    assert!(!plain.0.is_frozen(), "nothing changed");
}

#[derive(Clone, Module)]
struct Regularised {
    first: Linear,
    drop: Dropout,
    inner: Vec<(Linear, Dropout)>,
}

#[test]
fn train_and_eval_mode() {
    random::seed(5);
    let mut m = Regularised {
        first: Linear::new(2, 2),
        drop: Dropout::new(0.5),
        inner: vec![(Linear::new(2, 2), Dropout::new(0.1))],
    };
    assert!(m.training(), "modules start in training mode, as MLX");
    let train_def = m.def();
    m.train(false);
    assert!(!m.training());
    assert!(!m.drop.training && !m.inner[0].1.training);
    assert_ne!(m.def(), train_def, "the mode is part of the structure");
    let x = array![[1.0, 2.0]];
    assert_eq!(
        values(&m.drop.forward(&x)),
        [1.0, 2.0],
        "identity in eval mode"
    );
    m.train(true);
    assert!(m.training() && m.inner[0].1.training);
    assert_eq!(m.def(), train_def);
}

#[test]
fn layer_builders() {
    let ln = LayerNorm::new(8).eps(1e-6).affine(false);
    assert!(ln.weight.is_none() && ln.bias.is_none());
    assert_eq!(ln.eps, 1e-6);
    let nb = LayerNorm::new(8).without_bias();
    assert_eq!(names(nb.parameters()), ["weight"]);
    let rms = RmsNorm::new(4).eps(1e-6);
    assert_eq!(rms.eps, 1e-6);
    let rope = Rope::new(8).traditional().base(500_000.0).scale(0.5);
    assert!(rope.traditional && rope.base == 500_000.0 && rope.scale == 0.5);
    assert!(rope.parameters().is_empty());
    // Changing a hashed configuration changes the structure.
    assert_ne!(RmsNorm::new(4).def(), RmsNorm::new(4).eps(1e-6).def());
}

/// Two layers, the first frozen at times.
#[derive(Clone, Module)]
struct Pair {
    a: Linear,
    b: Linear,
}

/// A constant gradient for each trainable parameter, chosen by its path (so a parameter
/// keeps its gradient when others freeze); zero for the others.
fn constant_gradients(m: &Pair) -> Pair {
    let trainable: Vec<String> = names(m.trainable_parameters());
    m.map(|path, p| {
        let scale = match path {
            "a.weight" => 0.1f32,
            "a.bias" => -0.2,
            "b.weight" => 0.3,
            _ => -0.4,
        };
        if trainable.iter().any(|t| t == path) {
            ops::full(p.shape(), scale)
        } else {
            p.zeros_like()
        }
    })
}

/// Optimizer state follows each parameter's path: freezing a layer keeps the others'
/// state and its own, so unfreezing it continues from its moments, as in MLX.
#[test]
fn optimizer_state_follows_the_parameter_path() {
    random::seed(5);
    let mut model = Pair {
        a: Linear::new(3, 2),
        b: Linear::new(2, 2),
    };
    // The reference keeps one optimizer per parameter.
    let mut reference: Vec<(String, Param, Adam)> = model
        .trainable_parameters()
        .into_iter()
        .map(|(n, p)| (n, Param::new(p), Adam::new(0.01)))
        .collect();
    let step_reference = |reference: &mut Vec<(String, Param, Adam)>, m: &Pair| {
        let grads = constant_gradients(m);
        let g: HashMap<String, Array> = grads.flatten().into_iter().collect();
        for (name, _) in m.trainable_parameters() {
            let (_, p, opt) = reference.iter_mut().find(|(n, ..)| *n == name).unwrap();
            opt.update(p, &Param::new(g[&name].clone())).unwrap();
        }
    };
    let mut opt = Adam::new(0.01);
    // Step 1, everything trainable.
    step_reference(&mut reference, &model);
    let grads = constant_gradients(&model);
    opt.update(&mut model, &grads).unwrap();
    // Steps 2 and 3, `a` frozen: `b` continues from its state.
    model.freeze(|p| p.starts_with("a.")).unwrap();
    for _ in 0..2 {
        step_reference(&mut reference, &model);
        let grads = constant_gradients(&model);
        opt.update(&mut model, &grads).unwrap();
    }
    // Step 4, `a` unfrozen: it continues from its own moments (its reference optimizer
    // was not stepped while frozen, so its bias-free Adam update matches).
    model.unfreeze(|p| p.starts_with("a.")).unwrap();
    step_reference(&mut reference, &model);
    let grads = constant_gradients(&model);
    opt.update(&mut model, &grads).unwrap();
    for (name, p) in model.parameters() {
        let (_, want, _) = reference.iter().find(|(n, ..)| *n == name).unwrap();
        assert_eq!(values(&p), values(want), "{name}");
    }
    // Every parameter's slots are still there.
    let paths = opt.paths();
    for name in ["a.weight", "a.bias", "b.weight", "b.bias"] {
        assert!(paths.contains(&format!("slots.{name}.m")), "{paths:?}");
    }
}

/// `Optimizer` is object-safe: a boxed optimizer steps as the concrete type and
/// visits the same state paths.
#[test]
fn boxed_optimizers_step_as_the_concrete_type() {
    let w = Param::new(array![1.0, 2.0, 3.0]);
    let g = Param::new(array![0.1, -0.2, 0.3]);
    let mut boxed: Vec<Box<dyn Optimizer>> = vec![
        Box::new(Adam::new(0.01)),
        Box::new(Sgd::new(0.1)),
        Box::new(AdamW::new(1e-3)),
    ];
    let step = |opt: &mut dyn Optimizer| {
        let out = opt.apply(&[("w", w.value())], &[g.value()]).unwrap();
        values(&out[0])
    };
    let concrete = [
        step(&mut Adam::new(0.01)),
        step(&mut Sgd::new(0.1)),
        step(&mut AdamW::new(1e-3)),
    ];
    for (opt, want) in boxed.iter_mut().zip(&concrete) {
        assert_eq!(&step(opt.as_mut()), want);
    }
    let mut concrete = Adam::new(0.01);
    let mut model = w.clone();
    concrete.update(&mut model, &g).unwrap();
    let mut model2 = w.clone();
    let mut b: Box<dyn Optimizer> = Box::new(Adam::new(0.01));
    b.update(&mut model2, &g).unwrap();
    assert_eq!(values(&model), values(&model2));
    assert_eq!(b.paths(), concrete.paths());
    assert_eq!(
        concrete.paths(),
        ["step", "learning_rate", "slots.m", "slots.v"]
    );
}

/// The optimizer state is a tree: `step`, `learning_rate` (a 0-D float32 array) and the
/// slots by parameter path; hyperparameters and the optimizer type are part of its
/// structure.
#[test]
fn optimizer_state_is_a_tree() {
    random::seed(6);
    let model = Pair {
        a: Linear::new(3, 2),
        b: Linear::new(2, 2),
    };
    let mut opt = Adam::new(1e-3);
    assert_eq!(opt.paths(), ["step", "learning_rate"]);
    opt.init(&model);
    assert_eq!(
        opt.paths(),
        [
            "step",
            "learning_rate",
            "slots.a.bias.m",
            "slots.a.bias.v",
            "slots.a.weight.m",
            "slots.a.weight.v",
            "slots.b.bias.m",
            "slots.b.bias.v",
            "slots.b.weight.m",
            "slots.b.weight.v",
        ]
    );
    let lr = opt.learning_rate();
    assert_eq!(
        (lr.dtype(), lr.ndim(), lr.item::<f32>().unwrap()),
        (DType::F32, 0, 1e-3)
    );
    // `set_learning_rate` writes the array, and the next update uses it.
    let mut a = Sgd::new(0.5);
    a.set_learning_rate(0.25);
    assert_eq!(a.learning_rate().item::<f32>().unwrap(), 0.25);
    let mut w = Param::new(array![1.0]);
    a.update(&mut w, &Param::new(array![1.0])).unwrap();
    assert_eq!(values(&w), [0.75]);
    // Hyperparameters and the boxed type are part of the structure.
    assert_ne!(Adam::new(1e-3).def(), Adam::new(1e-3).eps(1e-6).def());
    let boxed_adam: Box<dyn Optimizer> = Box::new(Adam::new(1e-3));
    let boxed_sgd: Box<dyn Optimizer> = Box::new(Sgd::new(1e-3));
    assert_ne!(boxed_adam.def(), boxed_sgd.def());
    // Gradients must have the model's paths.
    let mut m = model.clone();
    assert!(
        opt.update(
            &mut m,
            &Pair {
                a: Linear::new(3, 2),
                b: Linear::new(2, 2)
            }
        )
        .is_ok()
    );
    let other = (model.a.clone(),);
    assert!(
        Adam::new(1e-3)
            .update(
                &mut (model.a.clone(), model.b.clone()),
                &(other.0.clone(), other.0)
            )
            .is_err()
    );
}

/// Checkpoint and resume a run in two lines: `nuvec::save(path, &(model, opt))` and
/// `nuvec::load_into(path, &mut (model, opt))`.
#[test]
fn checkpoint_and_resume_are_bit_identical() {
    random::seed(7);
    let mut state = (Mlp::new(&[4, 8, 3]), Adam::new(1e-2).bias_correction());
    let x = random::normal([6, 4], DType::F32, 0.0, 1.0, None);
    let y = Array::from_slice(&[0u32, 1, 2, 0, 1, 2], [6]);
    let step = |(model, opt): &mut (Mlp, Adam)| {
        let loss = |m: &Mlp| cross_entropy(&m.forward(&x), &y, Reduction::Mean);
        let (_, grads) = value_and_grad(loss)(&*model).unwrap();
        opt.update(model, &grads).unwrap();
        nuvec::eval(&*model).unwrap();
    };
    for _ in 0..3 {
        step(&mut state);
    }
    let path = temp("resume.safetensors");
    nuvec::save(&path, &state).unwrap();
    let mut resumed = (Mlp::new(&[4, 8, 3]), Adam::new(1e-2).bias_correction());
    resumed.1.init(&resumed.0);
    nuvec::load_into(&path, &mut resumed).unwrap();
    for _ in 0..2 {
        step(&mut state);
        step(&mut resumed);
    }
    for ((p, a), (q, b)) in state.flatten().iter().zip(resumed.flatten()) {
        assert_eq!(p, &q);
        assert_eq!(
            a.astype(DType::F64).to_vec::<f64>().unwrap(),
            b.astype(DType::F64).to_vec::<f64>().unwrap(),
            "{p}"
        );
    }
    let file = nuvec::io::load_safetensors(&path).unwrap();
    assert_eq!(file.metadata["step"], "3");
}

/// `join_schedules` with a boundary too many or too few is an error, not a panic.
#[test]
fn join_schedules_rejects_mismatched_boundaries() {
    use nuvec_nn::optim::{LearningRate, schedules};
    let two = || vec![LearningRate::from(0.1f32), LearningRate::from(0.01f32)];
    assert!(schedules::join_schedules(two(), vec![]).is_err());
    assert!(schedules::join_schedules(two(), vec![1, 2]).is_err());
    assert!(schedules::join_schedules(vec![], vec![]).is_err());
    let joined = schedules::join_schedules(two(), vec![3]).unwrap();
    let at = |t: u64| joined.at(&Array::from(t)).item::<f32>().unwrap();
    assert_eq!((at(2), at(3)), (0.1, 0.01));
}

/// A layer called with an input of the wrong shape gives a failed array whose error
/// points at the user's call of `forward` (`#[track_caller]` through the layer), not at
/// the layer's code.
#[test]
fn layer_errors_point_at_the_forward_call() {
    random::seed(0);
    let l = Linear::new(3, 2);
    let x = ops::zeros([4, 5], DType::F32);
    let line = line!() + 1;
    let y = l.forward(&x);
    let err = y.check().unwrap_err();
    assert!(
        matches!(
            err,
            Error::Op {
                kind: OpErrorKind::ShapeMismatch { .. },
                ..
            }
        ),
        "{err:?}"
    );
    let at = err.location().expect("an op error has a location");
    assert!(at.file().ends_with("tests/nn.rs"), "{at}");
    assert_eq!(at.line(), line, "{at}");
    // Without a bias (a plain matmul) and through an activation, too.
    let no_bias = Linear {
        weight: l.weight.clone(),
        bias: None,
    };
    let line = line!() + 1;
    let y = relu(&no_bias.forward(&x));
    let at = y.check().unwrap_err().location().unwrap();
    assert!(at.file().ends_with("tests/nn.rs"), "{at}");
    assert_eq!(at.line(), line, "{at}");
}

/// A compiled training step (`compile_mut` over `(model, optimizer)`) equals the
/// uncompiled one, traces once, retraces exactly once for each structure change (eval mode,
/// a frozen layer, a hashed hyperparameter), and a new learning rate needs no retrace.
#[test]
fn compiled_training_steps_follow_the_structure() {
    use std::cell::Cell;
    nuvec::set_default_stream(nuvec::Stream::default_for(nuvec::Device::cpu()));
    random::seed(8);
    let model = Regularised {
        first: Linear::new(2, 2),
        drop: Dropout::new(0.25),
        inner: vec![(Linear::new(2, 2), Dropout::new(0.0))],
    };
    let forward = |m: &Regularised, x: &Array| {
        let h = m.drop.forward(&relu(&m.first.forward(x)));
        m.inner[0].0.forward(&h)
    };
    let traces = Cell::new(0);
    let step = nuvec::compile_mut(
        |(m, opt): &mut (Regularised, Adam), (x, y): &(Array, Array)| {
            traces.set(traces.get() + 1);
            let loss = |m: &Regularised| forward(m, x).subtract(y).multiply(2.0).sum(..);
            let (l, g) = value_and_grad(loss)(m)?;
            opt.update(m, &g)?;
            Ok(l)
        },
    );
    let plain = |(m, opt): &mut (Regularised, Adam), (x, y): &(Array, Array)| {
        let loss = |m: &Regularised| forward(m, x).subtract(y).multiply(2.0).sum(..);
        let (l, g) = value_and_grad(loss)(&*m).unwrap();
        opt.update(m, &g).unwrap();
        l
    };
    let batch = (
        array![[1.0, -2.0], [0.5, 3.0]],
        array![[0.0, 1.0], [1.0, 0.0]],
    );
    let mut a = (model.clone(), Adam::new(0.01));
    a.1.init(&a.0);
    let mut b = a.clone();
    let run = |a: &mut (Regularised, Adam), b: &mut (Regularised, Adam), what: &str| {
        // The same random draws in both: one seed before each pair of calls.
        random::seed(9);
        let l1 = step(a, &batch).unwrap();
        random::seed(9);
        let l2 = plain(b, &batch);
        assert_eq!(values(&l1), values(&l2), "{what}: loss");
        for (x, y) in a.leaves().iter().zip(b.leaves()) {
            assert_eq!(values(x), values(&y), "{what}: state");
        }
    };
    for _ in 0..5 {
        run(&mut a, &mut b, "steady");
    }
    assert_eq!(traces.get(), 1, "traces once");
    a.1.set_learning_rate(0.001);
    b.1.set_learning_rate(0.001);
    run(&mut a, &mut b, "new learning rate");
    assert_eq!(traces.get(), 1, "a new rate is an input");
    let mut expect = 1;
    for change in ["eval", "freeze", "eps"] {
        match change {
            "eval" => {
                a.0.train(false);
                b.0.train(false);
            }
            "freeze" => {
                a.0.first.freeze(|_| true).unwrap();
                b.0.first.freeze(|_| true).unwrap();
            }
            _ => {
                a.1 = a.1.clone().eps(1e-6);
                b.1 = b.1.clone().eps(1e-6);
            }
        }
        run(&mut a, &mut b, change);
        expect += 1;
        assert_eq!(traces.get(), expect, "{change}: one retrace");
        run(&mut a, &mut b, change);
        assert_eq!(traces.get(), expect, "{change}: replayed after");
    }
}

/// `map_modules` finds every submodule of a type whose path matches, and may replace it.
#[test]
fn map_modules_edits_matching_submodules() {
    random::seed(10);
    let mut m = Mlp::new(&[2, 3, 3, 1]);
    // Replace the first two layers (by path) with bias-free copies.
    let changed = m
        .map_modules(
            |p| p == "layers.0" || p == "layers.1",
            |_, l: &mut Linear| {
                l.bias = None;
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(changed, 2);
    assert_eq!(
        names(m.parameters()),
        [
            "layers.0.weight",
            "layers.1.weight",
            "layers.2.weight",
            "layers.2.bias"
        ]
    );
    // Other types are skipped.
    assert_eq!(
        m.map_modules(|_| true, |_, _: &mut RmsNorm| Ok(()))
            .unwrap(),
        0
    );
    // The first error stops the walk; earlier changes are kept.
    let err = m.map_modules(
        |_| true,
        |p, l: &mut Linear| {
            if p == "layers.1" {
                return Err(Error::InvalidArgument(format!("stop at {p}")));
            }
            l.weight.set_frozen(true);
            Ok(())
        },
    );
    assert!(err.unwrap_err().to_string().contains("stop at layers.1"));
    assert!(m.layers[0].weight.is_frozen() && !m.layers[1].weight.is_frozen());
}

/// Central finite differences of `f` against `grad(f)` for a few elements of every
/// trainable leaf (`eps` and `tol` per dtype), plus `⟨c, J t⟩ = ⟨Jᵀ c, t⟩`.
fn check_gradients<T: Tree>(name: &str, t: &T, f: impl Fn(&T) -> Array, eps: f64, tol: f64) {
    let g = nuvec::grad(&f)(t).unwrap();
    let trainable: std::collections::HashSet<String> = Module::trainable_parameters(t)
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    let leaves = t.flatten();
    for (k, ((path, leaf), grad)) in leaves.iter().zip(g.leaves()).enumerate() {
        let gv = grad.astype(DType::F64).to_vec::<f64>().unwrap();
        if !trainable.contains(path) {
            assert!(
                gv.iter().all(|&x| x == 0.0),
                "{name}: {path} is not trained"
            );
            continue;
        }
        let base = leaf.astype(DType::F64).to_vec::<f64>().unwrap();
        for i in [0, base.len() / 2, base.len().saturating_sub(1)] {
            let at = |delta: f64| {
                let mut v = base.clone();
                v[i] += delta;
                let replaced = Array::from_slice(&v, leaf.shape()).astype(leaf.dtype());
                let mut all: Vec<Array> = leaves.iter().map(|(_, a)| a.clone()).collect();
                all[k] = replaced;
                f(&t.unflatten(all).unwrap())
                    .astype(DType::F64)
                    .item::<f64>()
                    .unwrap()
            };
            let fd = (at(eps) - at(-eps)) / (2.0 * eps);
            assert!(
                (fd - gv[i]).abs() <= tol * fd.abs().max(1.0),
                "{name}: {path}[{i}]: finite difference {fd} vs gradient {}",
                gv[i]
            );
        }
    }
    // vjp/jvp consistency over the whole tree.
    let tangents = t.map(|_, a| a.ones_like());
    let (_, jt) = nuvec::jvp(&f, t, &tangents).unwrap();
    let lhs = jt.astype(DType::F64).item::<f64>().unwrap();
    let mut rhs = 0.0;
    for (grad, (path, _)) in g.leaves().iter().zip(&leaves) {
        if trainable.contains(path) {
            rhs += grad.astype(DType::F64).sum(..).item::<f64>().unwrap();
        }
    }
    assert!(
        (lhs - rhs).abs() <= tol * lhs.abs().max(1.0),
        "{name}: jvp {lhs} vs vjp {rhs}"
    );
}

#[test]
fn gradients_through_modules_check_out() {
    nuvec::set_default_stream(nuvec::Stream::default_for(nuvec::Device::cpu()));
    random::seed(40);
    let x64 = random::normal([3, 4], DType::F64, 0.0, 1.0, None);
    let to64 = |m: &Mlp| m.map(|_, a| a.astype(DType::F64));
    // A derived module, in float64.
    let mlp = to64(&Mlp::new(&[4, 5, 2]));
    check_gradients(
        "mlp",
        &mlp,
        |m: &Mlp| m.forward(&x64).tanh().sum(..),
        1e-6,
        1e-6,
    );
    // LayerNorm without a bias, and Dropout in evaluation mode (the identity).
    #[derive(Clone, Module)]
    struct Normed {
        dense: Linear,
        norm: LayerNorm,
        drop: Dropout,
    }
    let mut normed = Normed {
        dense: Linear::new(4, 4),
        norm: LayerNorm::new(4).without_bias(),
        drop: Dropout::new(0.5),
    };
    normed.train(false);
    let x = random::normal([3, 4], DType::F32, 0.0, 1.0, None);
    check_gradients(
        "layer norm",
        &normed,
        |m: &Normed| {
            m.drop
                .forward(&m.norm.forward(&m.dense.forward(&x)))
                .multiply(&x)
                .sum(..)
        },
        1e-2,
        2e-2,
    );
    // LoRA over a full-precision and a quantized base: only the adapters are trained.
    for quantized in [false, true] {
        let base = Linear::new(64, 8);
        let mut lora = if quantized {
            LoraLinear::from_base(QuantizedLinear::from_linear(&base), 4)
        } else {
            LoraLinear::from_base(base, 4)
        };
        lora.lora_b = Param::new(random::normal([4, 8], DType::F32, 0.0, 0.1, None));
        let x = random::normal([2, 64], DType::F32, 0.0, 1.0, None);
        check_gradients(
            if quantized { "qlora" } else { "lora" },
            &lora,
            |m: &LoraLinear| m.forward(&x).tanh().sum(..),
            1e-2,
            2e-2,
        );
    }
}
