//! Neural-network building blocks on nuvec, following MLX's `mlx.nn` and `mlx.optimizers`.
//!
//! - [`Module`]: named parameter trees (`layers.0.weight`), trainable or frozen, loaded
//!   from checkpoints by name ([`Module::load_weights`]).
//! - Layers ([`layers`]): [`Linear`], [`QuantizedLinear`], [`Embedding`],
//!   [`QuantizedEmbedding`], [`RmsNorm`],
//!   [`LayerNorm`], [`Rope`] and [`LoraLinear`]; activations [`relu`], [`silu`],
//!   [`gelu_approx`] and [`gelu_fast_approx`].
//! - [`losses::cross_entropy`], [`value_and_grad`] for a model's loss, and
//!   [`optim`]: SGD, Adam, AdamW, Lion and Muon with learning-rate schedules and gradient
//!   clipping.
//!
//! ```
//! use nuvec::{Array, Result, random};
//! use nuvec_nn::optim::{Optimizer, Sgd};
//! use nuvec_nn::{Linear, Module, Parameters, losses, relu, value_and_grad};
//!
//! #[derive(Clone)]
//! struct Mlp {
//!     hidden: Linear,
//!     out: Linear,
//! }
//!
//! impl Module for Mlp {
//!     fn visit(&mut self, p: &mut Parameters<'_>) {
//!         p.module("hidden", &mut self.hidden);
//!         p.module("out", &mut self.out);
//!     }
//! }
//!
//! # fn main() -> Result<()> {
//! random::seed(0);
//! let mut model = Mlp { hidden: Linear::new(2, 8, true)?, out: Linear::new(8, 2, true)? };
//! let names: Vec<String> = model.parameters().into_iter().map(|(n, _)| n).collect();
//! assert_eq!(names, ["hidden.weight", "hidden.bias", "out.weight", "out.bias"]);
//!
//! let x = Array::from_slice(&[1.0f32, -1.0, 0.5, 2.0], &[2, 2])?;
//! let y = Array::from_slice(&[0u32, 1], &[2])?;
//! let loss = |m: &Mlp| {
//!     let logits = m.out.forward(&relu(&m.hidden.forward(&x)?)?)?;
//!     losses::cross_entropy(&logits, &y, losses::Reduction::Mean)
//! };
//! let mut opt = Sgd::new(0.5);
//! let (before, _) = value_and_grad(&model, loss)?;
//! for _ in 0..20 {
//!     let (_, grads) = value_and_grad(&model, loss)?;
//!     opt.update(&mut model, &grads)?;
//! }
//! let (after, _) = value_and_grad(&model, loss)?;
//! assert!(after.item::<f32>()? < before.item::<f32>()?);
//! # Ok(())
//! # }
//! ```
//!
//! **Status:** pre-alpha.

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};

use nuvec::{Array, Error, Result, add, maximum, multiply, sigmoid, tanh};

pub mod layers;
pub mod losses;
pub mod optim;

pub use layers::{
    Embedding, LayerNorm, Linear, LoraLinear, QuantizedEmbedding, QuantizedLinear, RmsNorm, Rope,
};

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A model: a tree of named parameters, as MLX's `nn.Module`.
///
/// A module implements [`Module::visit`], reporting each parameter and submodule by name
/// in a fixed order; everything else (named parameters, trainable parameters, updates,
/// loading weights) is built on it. Models are cheap to clone: arrays are shared.
pub trait Module: Clone {
    /// Reports every parameter and submodule to `p`, frozen ones included.
    fn visit(&mut self, p: &mut Parameters<'_>);

    /// Every parameter with its path (`layers.0.weight`), in visiting order.
    fn parameters(&self) -> Vec<(String, Array)> {
        let mut out = Vec::new();
        self.clone().visit(&mut Parameters::new(&mut |name, a, _| {
            out.push((name.to_owned(), a.clone()));
        }));
        out
    }

    /// The parameters that are trained (not frozen), with their paths.
    fn trainable_parameters(&self) -> Vec<(String, Array)> {
        let mut out = Vec::new();
        self.clone()
            .visit(&mut Parameters::new(&mut |name, a, trainable| {
                if trainable {
                    out.push((name.to_owned(), a.clone()));
                }
            }));
        out
    }

    /// Replaces the trainable parameters, given in the order of
    /// [`Module::trainable_parameters`] (as optimizers return them).
    fn update_trainable(&mut self, values: &[Array]) -> Result<()> {
        let expected = self.trainable_parameters().len();
        if values.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "{} values for {expected} trainable parameters",
                values.len()
            )));
        }
        let mut next = values.iter();
        self.visit(&mut Parameters::new(&mut |_, a, trainable| {
            if trainable {
                *a = next.next().expect("counted").clone();
            }
        }));
        Ok(())
    }

    /// Replaces parameters by path, as MLX's `load_weights`: each weight must have its
    /// parameter's shape (the dtype is kept as loaded). With `strict`, every parameter
    /// must be given and no unknown names are allowed.
    fn load_weights(
        &mut self,
        weights: impl IntoIterator<Item = (String, Array)>,
        strict: bool,
    ) -> Result<()> {
        let mut weights: HashMap<String, Array> = weights.into_iter().collect();
        let mut problem: Option<String> = None;
        let mut seen = HashSet::new();
        self.visit(&mut Parameters::new(&mut |name, a, _| {
            seen.insert(name.to_owned());
            match weights.remove(name) {
                Some(w) if w.shape() == a.shape() => *a = w,
                Some(w) => {
                    problem.get_or_insert(format!(
                        "{name}: shape {:?}, expected {:?}",
                        w.shape(),
                        a.shape()
                    ));
                }
                None if strict => {
                    problem.get_or_insert(format!("{name}: missing"));
                }
                None => {}
            }
        }));
        if let Some(p) = problem {
            return Err(Error::InvalidArgument(format!("load_weights: {p}")));
        }
        if strict && !weights.is_empty() {
            let mut unknown: Vec<&String> = weights.keys().collect();
            unknown.sort();
            return Err(Error::InvalidArgument(format!(
                "load_weights: unknown parameters {unknown:?}"
            )));
        }
        Ok(())
    }
}

/// Receives a module's parameters during [`Module::visit`], with their paths.
pub struct Parameters<'a> {
    prefix: String,
    frozen: bool,
    f: &'a mut dyn FnMut(&str, &mut Array, bool),
}

impl<'a> Parameters<'a> {
    /// A visitor calling `f(path, parameter, trainable)` for each parameter.
    pub fn new(f: &'a mut dyn FnMut(&str, &mut Array, bool)) -> Self {
        Self {
            prefix: String::new(),
            frozen: false,
            f,
        }
    }

    fn path(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{name}", self.prefix)
        }
    }

    /// A trainable parameter.
    pub fn param(&mut self, name: &str, a: &mut Array) {
        let path = self.path(name);
        (self.f)(&path, a, !self.frozen);
    }

    /// A trainable parameter that may be absent (an optional bias).
    pub fn optional(&mut self, name: &str, a: &mut Option<Array>) {
        if let Some(a) = a {
            self.param(name, a);
        }
    }

    /// A parameter that is never trained (a quantized weight, a LoRA base).
    pub fn frozen(&mut self, name: &str, a: &mut Array) {
        let path = self.path(name);
        (self.f)(&path, a, false);
    }

    /// A submodule, whose parameters are named `name.…`.
    pub fn module<M: Module>(&mut self, name: &str, m: &mut M) {
        let outer = std::mem::take(&mut self.prefix);
        self.prefix = if outer.is_empty() {
            name.to_owned()
        } else {
            format!("{outer}.{name}")
        };
        m.visit(self);
        self.prefix = outer;
    }

    /// Visits `f`'s parameters as frozen.
    pub fn freeze(&mut self, f: impl FnOnce(&mut Self)) {
        let was = std::mem::replace(&mut self.frozen, true);
        f(self);
        self.frozen = was;
    }
}

/// A sequence of modules is a module; its parts are named `0`, `1`, … (MLX's lists).
impl<M: Module> Module for Vec<M> {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        for (i, m) in self.iter_mut().enumerate() {
            p.module(&i.to_string(), m);
        }
    }
}

/// A module whose parameters are all frozen (MLX's `freeze`).
#[derive(Debug, Clone)]
pub struct Frozen<M>(pub M);

impl<M: Module> Module for Frozen<M> {
    fn visit(&mut self, p: &mut Parameters<'_>) {
        let inner = &mut self.0;
        p.freeze(|p| inner.visit(p));
    }
}

/// The loss of `model` and its gradient with respect to every trainable parameter (in the
/// order of [`Module::trainable_parameters`]), as MLX's `nn.value_and_grad`.
///
/// `loss` receives a copy of the model whose trainable parameters are being
/// differentiated and must return a 0-D array.
pub fn value_and_grad<M: Module>(
    model: &M,
    loss: impl Fn(&M) -> Result<Array>,
) -> Result<(Array, Vec<Array>)> {
    let parameters: Vec<Array> = model
        .trainable_parameters()
        .into_iter()
        .map(|(_, a)| a)
        .collect();
    let argnums: Vec<usize> = (0..parameters.len()).collect();
    let f = |ps: &[Array]| {
        let mut m = model.clone();
        m.update_trainable(ps)?;
        loss(&m)
    };
    nuvec::value_and_grad(f, &argnums)(&parameters)
}

/// `max(x, 0)`, as MLX's `nn.relu`.
pub fn relu(x: &Array) -> Result<Array> {
    maximum(x, 0.0)
}

/// `x · sigmoid(x)`, as MLX's `nn.silu`.
pub fn silu(x: &Array) -> Result<Array> {
    multiply(x, sigmoid(x)?)
}

/// GELU with the tanh approximation, as MLX's `nn.gelu_approx`:
/// `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`.
pub fn gelu_approx(x: &Array) -> Result<Array> {
    let cube = multiply(multiply(x, x)?, x)?;
    let inner = multiply(
        add(x, multiply(cube, 0.044715)?)?,
        (2.0 / std::f64::consts::PI).sqrt(),
    )?;
    multiply(multiply(x, 0.5)?, add(tanh(&inner)?, 1.0)?)
}

/// `x · sigmoid(1.702 x)`, as MLX's `nn.gelu_fast_approx`.
pub fn gelu_fast_approx(x: &Array) -> Result<Array> {
    multiply(x, sigmoid(&multiply(x, 1.702)?)?)
}
