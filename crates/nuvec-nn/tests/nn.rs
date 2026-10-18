//! `nuvec-nn` against values recorded from MLX 0.32.4 (`mlx.nn`, `mlx.optimizers`).

use nuvec::{Array, Result, random};
use nuvec_nn::losses::{Reduction, cross_entropy};
use nuvec_nn::optim::{Adam, Optimizer, Sgd};
use nuvec_nn::{Linear, Module, Parameters, QuantizedLinear, relu, value_and_grad};

fn values(a: &Array) -> Vec<f32> {
    a.to_vec::<f32>().unwrap()
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

#[test]
fn linear_initialisation_equals_mlx() {
    random::seed(0);
    let l = Linear::new(3, 2, true).unwrap();
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
    assert_eq!(l.weight.shape(), &[2, 3]);
}

#[test]
fn cross_entropy_equals_mlx() {
    let logits = Array::from_slice(&[2.0f32, -1.0, -1.0, 2.0], &[2, 2]).unwrap();
    let targets = Array::from_slice(&[0i32, 1], &[2]).unwrap();
    let none = cross_entropy(&logits, &targets, Reduction::None).unwrap();
    close(
        &values(&none),
        &[0.04858732968568802, 0.04858732968568802],
        "none",
    );
    let mean = cross_entropy(&logits, &targets, Reduction::Mean).unwrap();
    close(&values(&mean), &[0.04858732968568802], "mean");
    let logits = Array::from_slice(&[0.5f32, 1.5, -0.3, 2.0, 0.1, 0.4], &[2, 3]).unwrap();
    let targets = Array::from_slice(&[2u32, 0], &[2]).unwrap();
    let loss = cross_entropy(&logits, &targets, Reduction::None).unwrap();
    close(
        &values(&loss),
        &[2.2273428440093994, 0.30118927359580994],
        "three classes",
    );
    let floats = Array::from_slice(&[0.0f32, 1.0], &[2]).unwrap();
    assert!(cross_entropy(&logits, &floats, Reduction::None).is_err());
    let wrong = Array::from_slice(&[0u32, 1, 2], &[3]).unwrap();
    assert!(cross_entropy(&logits, &wrong, Reduction::None).is_err());
}

/// Two steps from `w = [1, 2, 3]` with the constant gradient `[0.1, -0.2, 0.3]`.
fn steps(opt: &mut impl Optimizer) -> Vec<Vec<f32>> {
    let mut w = vec![Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap()];
    let g = vec![Array::from_slice(&[0.1f32, -0.2, 0.3], &[3]).unwrap()];
    (0..2)
        .map(|_| {
            w = opt.apply(&w, &g).unwrap();
            values(&w[0])
        })
        .collect()
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
            steps(&mut Sgd::new(0.1).momentum(0.9).nesterov(true)),
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
            steps(&mut Adam::new(0.01).bias_correction(true)),
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

#[derive(Clone)]
struct Mlp(Vec<Linear>);

impl Module for Mlp {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("layers", &mut self.0);
    }
}

/// The trainable parameters, without names.
fn arrays(m: &impl Module) -> Vec<Array> {
    m.trainable_parameters()
        .into_iter()
        .map(|(_, a)| a)
        .collect()
}

impl Mlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for (i, l) in self.0.iter().enumerate() {
            h = l.forward(&h)?;
            if i + 1 < self.0.len() {
                h = relu(&h)?;
            }
        }
        Ok(h)
    }
}

#[test]
fn value_and_grad_matches_the_core_transform_and_training_learns() {
    random::seed(3);
    let mut model = Mlp(vec![
        Linear::new(4, 8, true).unwrap(),
        Linear::new(8, 3, true).unwrap(),
    ]);
    let x = random::normal(&[16, 4], nuvec::DType::F32, None, None, None).unwrap();
    let labels: Vec<u32> = (0..16).map(|i| i % 3).collect();
    let y = Array::from_slice(&labels, &[16]).unwrap();
    // Class-dependent shift so the problem is learnable.
    let shift = Array::from_slice(
        &labels.iter().map(|&l| l as f32).collect::<Vec<_>>(),
        &[16, 1],
    )
    .unwrap();
    let x = nuvec::add(&x, &shift).unwrap();
    let loss = |m: &Mlp| cross_entropy(&m.forward(&x)?, &y, Reduction::Mean);

    let (value, grads) = value_and_grad(&model, loss).unwrap();
    assert_eq!(grads.len(), 4);
    let params = arrays(&model);
    let f = |ps: &[Array]| {
        let mut m = model.clone();
        m.update_trainable(ps)?;
        loss(&m)
    };
    let (core_value, core_grads) = nuvec::value_and_grad(f, &[0, 1, 2, 3])(&params).unwrap();
    assert_eq!(values(&value), values(&core_value));
    for (a, b) in grads.iter().zip(&core_grads) {
        assert_eq!(values(a), values(b));
    }

    let mut opt = Adam::new(0.05);
    let start = values(&value)[0];
    for _ in 0..60 {
        let (_, grads) = value_and_grad(&model, loss).unwrap();
        opt.update(&mut model, &grads).unwrap();
        nuvec::eval(&arrays(&model)).unwrap();
    }
    let end = values(&value_and_grad(&model, loss).unwrap().0)[0];
    assert!(end < 0.3 * start, "loss {start} -> {end}");
    assert!(model.update_trainable(&params[..2]).is_err());
}

#[test]
fn quantized_linear_equals_mlx() {
    random::seed(0);
    let q = QuantizedLinear::from_linear(&Linear::new(64, 3, true).unwrap(), 32, 4).unwrap();
    assert_eq!(q.weight.shape(), &[3, 8]);
    assert_eq!(q.scales.shape(), &[3, 2]);
    let x = nuvec::arange(0.0, 128.0, 1.0, nuvec::DType::F32).unwrap();
    let x = (x.reshape(&[2, 64]).unwrap() / 64.0) - 0.5;
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
    close(
        &values(&q.forward(&x).unwrap()),
        &expected,
        "QuantizedLinear",
    );
    // Only the bias trains (as MLX's from_linear); gradients reach it through the layer.
    let names = |ps: Vec<(String, Array)>| ps.into_iter().map(|(n, _)| n).collect::<Vec<_>>();
    assert_eq!(
        names(q.parameters()),
        ["weight", "scales", "biases", "bias"]
    );
    assert_eq!(names(q.trainable_parameters()), ["bias"]);
    let (_, grads) = value_and_grad(&q, |m: &QuantizedLinear| {
        nuvec::sum(&m.forward(&x)?, &[], false)
    })
    .unwrap();
    assert_eq!(values(&grads[0]), vec![2.0; 3]);
}
