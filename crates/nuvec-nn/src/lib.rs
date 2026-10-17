//! Neural-network building blocks on nuvec, following MLX's `mlx.nn` and `mlx.optimizers`:
//! a [`Module`] trait (parameters out, parameters in), the [`Linear`] layer, [`relu`],
//! [`losses::cross_entropy`], the [`optim::Sgd`] and [`optim::Adam`] optimizers, and
//! [`value_and_grad`] for a model's loss.
//!
//! ```
//! use nuvec::{Array, Result, random};
//! use nuvec_nn::optim::{Optimizer, Sgd};
//! use nuvec_nn::{Linear, Module, losses, value_and_grad};
//!
//! # fn main() -> Result<()> {
//! random::seed(0);
//! let mut model = Linear::new(2, 2, true)?;
//! let x = Array::from_slice(&[1.0f32, -1.0, 0.5, 2.0], &[2, 2])?;
//! let y = Array::from_slice(&[0u32, 1], &[2])?;
//! let mut opt = Sgd::new(0.5);
//! let loss = |m: &Linear| losses::cross_entropy(&m.forward(&x)?, &y, losses::Reduction::Mean);
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
//! **Status:** pre-alpha. The API covers what an MLP needs; parameter trees, more layers
//! and schedulers follow.

#![forbid(unsafe_code)]

use nuvec::{Array, Error, Result, matmul, maximum, random};

pub mod losses;
pub mod optim;

/// The version of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A model with trainable parameters.
///
/// Parameters are flat, in a fixed order. Composite models concatenate their parts'
/// parameters, and [`Module::set_parameters`] takes them back in the same order. Models
/// are cheap to clone (arrays are shared).
pub trait Module: Clone {
    /// The trainable parameters.
    fn parameters(&self) -> Vec<Array>;

    /// Replaces the parameters, given in the order of [`Module::parameters`].
    fn set_parameters(&mut self, parameters: &[Array]) -> Result<()>;
}

/// A sequence of modules is a module (its parts' parameters, in order).
impl<M: Module> Module for Vec<M> {
    fn parameters(&self) -> Vec<Array> {
        self.iter().flat_map(Module::parameters).collect()
    }

    fn set_parameters(&mut self, parameters: &[Array]) -> Result<()> {
        let total: usize = self.iter().map(|m| m.parameters().len()).sum();
        if parameters.len() != total {
            return Err(count_mismatch(parameters.len(), total));
        }
        let mut rest = parameters;
        for module in self {
            let n = module.parameters().len();
            module.set_parameters(&rest[..n])?;
            rest = &rest[n..];
        }
        Ok(())
    }
}

fn count_mismatch(got: usize, expected: usize) -> Error {
    Error::InvalidArgument(format!("{got} parameters for a module with {expected}"))
}

/// The loss of `model` and its gradient with respect to every parameter (in the order of
/// [`Module::parameters`]), as MLX's `nn.value_and_grad`.
///
/// `loss` receives a copy of the model whose parameters are being differentiated and must
/// return a 0-D array.
pub fn value_and_grad<M: Module>(
    model: &M,
    loss: impl Fn(&M) -> Result<Array>,
) -> Result<(Array, Vec<Array>)> {
    let parameters = model.parameters();
    let argnums: Vec<usize> = (0..parameters.len()).collect();
    let f = |ps: &[Array]| {
        let mut m = model.clone();
        m.set_parameters(ps)?;
        loss(&m)
    };
    nuvec::value_and_grad(f, &argnums)(&parameters)
}

/// `max(x, 0)`, as MLX's `nn.relu`.
pub fn relu(x: &Array) -> Result<Array> {
    maximum(x, 0.0)
}

/// An affine layer `y = x Wᵀ + b`, with `W` of shape `[output, input]` and `b` of shape
/// `[output]`.
#[derive(Debug, Clone)]
pub struct Linear {
    /// The weight, `[output_dims, input_dims]`.
    pub weight: Array,
    /// The bias, `[output_dims]`.
    pub bias: Option<Array>,
}

impl Linear {
    /// A layer initialised as MLX's `nn.Linear`: weight, then bias, drawn uniformly from
    /// `[-1/√input, 1/√input)` with this thread's key sequence (so the same seed gives
    /// MLX's values).
    pub fn new(input_dims: usize, output_dims: usize, bias: bool) -> Result<Self> {
        let scale = (1.0 / input_dims as f64).sqrt();
        let draw = |shape: &[usize]| random::uniform(-scale, scale, shape, nuvec::DType::F32, None);
        let weight = draw(&[output_dims, input_dims])?;
        let bias = if bias {
            Some(draw(&[output_dims])?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// `x Wᵀ + b` over the last axis of `x`; with a bias it is one fused `addmm`, as in
    /// MLX.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match &self.bias {
            Some(b) => nuvec::addmm(b, x, &self.weight.t()?, 1.0, 1.0),
            None => matmul(x, &self.weight.t()?),
        }
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Array> {
        std::iter::once(self.weight.clone())
            .chain(self.bias.clone())
            .collect()
    }

    fn set_parameters(&mut self, parameters: &[Array]) -> Result<()> {
        let expected = 1 + usize::from(self.bias.is_some());
        if parameters.len() != expected {
            return Err(count_mismatch(parameters.len(), expected));
        }
        self.weight = parameters[0].clone();
        if self.bias.is_some() {
            self.bias = Some(parameters[1].clone());
        }
        Ok(())
    }
}
