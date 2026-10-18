//! Trains MLX's MNIST example model (an MLP with two hidden layers of 32 units, ReLU,
//! SGD with learning rate 0.1, batch 256) on the default device (the GPU when available).
//!
//! ```text
//! cargo run --release -p nuvec-nn --example mnist -- <dir with the four MNIST idx files> [--cpu] [--epochs N] [--async] [--compile]
//! ```
//!
//! With `--async`, each step's parameters are submitted with `async_eval`, so the next batch
//! and graph are built while the GPU runs the step. With `--compile`, the training step
//! (forward, gradients and update) is compiled, as MLX's example does with `mx.compile`.
//!
//! `just mnist` downloads MNIST and compares with MLX.

use std::time::Instant;

use nuvec::{
    Array, DType, Device, Result, Stream, argmax, astype, equal, io, mean, random,
    set_default_stream,
};
use nuvec_nn::losses::{Reduction, cross_entropy};
use nuvec_nn::optim::{Optimizer, Sgd};
use nuvec_nn::{Linear, Module, Parameters, relu, value_and_grad};

#[derive(Clone)]
struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(sizes: &[usize]) -> Result<Self> {
        let layers = sizes
            .windows(2)
            .map(|w| Linear::new(w[0], w[1], true))
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let (last, hidden) = self.layers.split_last().expect("at least one layer");
        let mut h = x.clone();
        for layer in hidden {
            h = relu(&layer.forward(&h)?)?;
        }
        last.forward(&h)
    }
}

impl Module for Mlp {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        p.module("layers", &mut self.layers);
    }
}

/// The model's trainable parameters, without names.
fn arrays(m: &Mlp) -> Vec<Array> {
    m.trainable_parameters()
        .into_iter()
        .map(|(_, a)| a)
        .collect()
}

/// Images as float32 rows in `[0, 1]` and labels as uint32, as in MLX's example.
fn load(dir: &str, prefix: &str) -> Result<(Vec<f32>, Vec<u32>, usize)> {
    let images = io::read_idx(format!("{dir}/{prefix}-images-idx3-ubyte"))?;
    let labels = io::read_idx(format!("{dir}/{prefix}-labels-idx1-ubyte"))?;
    let n = labels.size();
    let x = images
        .to_vec::<u8>()?
        .iter()
        .map(|&p| f32::from(p) / 255.0)
        .collect();
    let y = labels
        .to_vec::<u8>()?
        .iter()
        .map(|&l| u32::from(l))
        .collect();
    Ok((x, y, n))
}

/// A small deterministic generator for shuffling (splitmix64).
struct Shuffle(u64);

impl Shuffle {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            p.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
        p
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "data/mnist".into());
    let epochs: usize = args
        .iter()
        .position(|a| a == "--epochs")
        .and_then(|i| args.get(i + 1))
        .map_or(10, |v| v.parse().expect("--epochs takes a number"));
    let asynchronous = args.iter().any(|a| a == "--async");
    let compiled = args.iter().any(|a| a == "--compile");
    if args.iter().any(|a| a == "--cpu") {
        set_default_stream(Stream::default_for(Device::cpu()));
    }
    let (train_x, train_y, n) = load(&dir, "train").inspect_err(|_| {
        eprintln!("cannot read MNIST from {dir}: run `just mnist` (it downloads the data)");
    })?;
    let (test_x, test_y, n_test) = load(&dir, "t10k")?;
    let pixels = train_x.len() / n;
    let test_x = Array::from_slice(&test_x, &[n_test, pixels])?;
    let test_y = Array::from_slice(&test_y, &[n_test])?;

    let (batch, lr) = (256, 0.1);
    random::seed(0);
    let mut model = Mlp::new(&[pixels, 32, 32, 10])?;
    nuvec::eval(&arrays(&model))?;
    let mut optimizer = Sgd::new(lr);
    // The step as a function of the parameters and the batch, returning new parameters.
    let k = arrays(&model).len();
    let template = model.clone();
    let step = nuvec::compile(move |xs: &[Array]| {
        let mut m = template.clone();
        m.update_trainable(&xs[..k])?;
        let (x, y) = (&xs[k], &xs[k + 1]);
        let (_, grads) = value_and_grad(&m, |m: &Mlp| {
            cross_entropy(&m.forward(x)?, y, Reduction::Mean)
        })?;
        Sgd::new(lr).update(&mut m, &grads)?;
        Ok(arrays(&m))
    });
    let mut shuffle = Shuffle(0);
    let device = nuvec::default_stream().device;
    let mut times = Vec::new();
    let mut accuracy = 0.0;
    for epoch in 0..epochs {
        let tic = Instant::now();
        let perm = shuffle.permutation(n);
        for ids in perm.chunks(batch) {
            let mut xb = Vec::with_capacity(ids.len() * pixels);
            for &i in ids {
                xb.extend_from_slice(&train_x[i * pixels..(i + 1) * pixels]);
            }
            let yb: Vec<u32> = ids.iter().map(|&i| train_y[i]).collect();
            let x = Array::from_slice(&xb, &[ids.len(), pixels])?;
            let y = Array::from_slice(&yb, &[ids.len()])?;
            if compiled {
                let mut args = arrays(&model);
                args.extend([x, y]);
                model.update_trainable(&step(&args)?)?;
            } else {
                let (_, grads) = value_and_grad(&model, |m: &Mlp| {
                    cross_entropy(&m.forward(&x)?, &y, Reduction::Mean)
                })?;
                optimizer.update(&mut model, &grads)?;
            }
            if asynchronous {
                nuvec::async_eval(&arrays(&model))?;
            } else {
                nuvec::eval(&arrays(&model))?;
            }
        }
        let predictions = argmax(&model.forward(&test_x)?, Some(1), false)?;
        let hits = astype(&equal(&predictions, &test_y)?, DType::F32)?;
        accuracy = mean(&hits, &[], false)?.item::<f32>()?;
        let seconds = tic.elapsed().as_secs_f64();
        times.push(seconds);
        println!("Epoch {epoch}: Test accuracy {accuracy:.3}, Time {seconds:.3} (s)");
    }
    let mut steady: Vec<f64> = times.iter().skip(1).copied().collect();
    steady.sort_by(f64::total_cmp);
    let median = steady.get(steady.len() / 2).copied().unwrap_or(times[0]);
    println!(
        "RESULT device={device} async={asynchronous} compiled={compiled} accuracy={accuracy:.4} epoch_seconds={median:.4}"
    );
    Ok(())
}
