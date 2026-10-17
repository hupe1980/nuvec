//! Optimizers, with MLX's update formulas (`mlx.optimizers`).

use nuvec::{Array, Result, add, divide, multiply, sqrt, subtract};

use crate::Module;

/// Updates a model's parameters from their gradients.
pub trait Optimizer {
    /// Applies one step: `gradients` are in the order of [`Module::parameters`].
    fn update<M: Module>(&mut self, model: &mut M, gradients: &[Array]) -> Result<()> {
        let parameters = model.parameters();
        if parameters.len() != gradients.len() {
            return Err(nuvec::Error::InvalidArgument(format!(
                "{} gradients for {} parameters",
                gradients.len(),
                parameters.len()
            )));
        }
        let updated = self.apply(&parameters, gradients)?;
        model.set_parameters(&updated)
    }

    /// The updated parameters.
    fn apply(&mut self, parameters: &[Array], gradients: &[Array]) -> Result<Vec<Array>>;
}

/// The learning rate as MLX stores it (a float32 array) cast to the gradient's dtype.
fn rate(lr: f64, like: &Array) -> Result<Array> {
    Array::scalar(lr as f32).astype(like.dtype())
}

/// Stochastic gradient descent with optional momentum, dampening, Nesterov momentum and
/// weight decay, as MLX's `optimizers.SGD`.
#[derive(Debug, Clone)]
pub struct Sgd {
    /// The learning rate.
    pub learning_rate: f64,
    /// The momentum strength (0 disables momentum).
    pub momentum: f64,
    /// L2 weight decay added to the gradient.
    pub weight_decay: f64,
    /// Dampening for momentum.
    pub dampening: f64,
    /// Nesterov momentum.
    pub nesterov: bool,
    velocity: Vec<Array>,
}

impl Sgd {
    /// Plain SGD with `learning_rate`.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: Vec::new(),
        }
    }

    /// Sets the momentum.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the weight decay.
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enables Nesterov momentum.
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for Sgd {
    fn apply(&mut self, parameters: &[Array], gradients: &[Array]) -> Result<Vec<Array>> {
        if self.velocity.len() != parameters.len() {
            self.velocity = parameters
                .iter()
                .map(|p| nuvec::zeros(p.shape(), p.dtype()))
                .collect::<Result<_>>()?;
        }
        let mut out = Vec::with_capacity(parameters.len());
        for ((p, g), v) in parameters.iter().zip(gradients).zip(&mut self.velocity) {
            let mut g = g.clone();
            if self.weight_decay != 0.0 {
                g = add(&g, multiply(p, self.weight_decay)?)?;
            }
            let lr = rate(self.learning_rate, &g)?;
            if self.momentum <= 0.0 {
                out.push(subtract(p, multiply(&lr, &g)?)?);
                continue;
            }
            let mut next = multiply(self.momentum, &*v)?;
            next = if self.dampening > 0.0 {
                add(&next, multiply(1.0 - self.dampening, &g)?)?
            } else {
                add(&next, &g)?
            };
            let update = if self.nesterov {
                add(&g, multiply(self.momentum, &next)?)?
            } else {
                next.clone()
            };
            *v = next;
            out.push(subtract(p, multiply(&lr, &update)?)?);
        }
        Ok(out)
    }
}

/// Adam, as MLX's `optimizers.Adam` (bias correction off unless enabled, as in MLX).
#[derive(Debug, Clone)]
pub struct Adam {
    /// The learning rate.
    pub learning_rate: f64,
    /// Decay rates of the first and second moment estimates.
    pub betas: (f64, f64),
    /// Added to the denominator for numerical stability.
    pub eps: f64,
    /// Whether to correct the moment estimates' bias toward zero.
    pub bias_correction: bool,
    step: u64,
    moments: Vec<(Array, Array)>,
}

impl Adam {
    /// Adam with `learning_rate`, betas `(0.9, 0.999)` and `eps = 1e-8`.
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            betas: (0.9, 0.999),
            eps: 1e-8,
            bias_correction: false,
            step: 0,
            moments: Vec::new(),
        }
    }

    /// Enables bias correction.
    pub fn bias_correction(mut self, enabled: bool) -> Self {
        self.bias_correction = enabled;
        self
    }
}

impl Optimizer for Adam {
    fn apply(&mut self, parameters: &[Array], gradients: &[Array]) -> Result<Vec<Array>> {
        if self.moments.len() != parameters.len() {
            self.moments = parameters
                .iter()
                .map(|p| {
                    Ok((
                        nuvec::zeros(p.shape(), p.dtype())?,
                        nuvec::zeros(p.shape(), p.dtype())?,
                    ))
                })
                .collect::<Result<_>>()?;
        }
        self.step += 1;
        let (b1, b2) = self.betas;
        let mut out = Vec::with_capacity(parameters.len());
        for ((p, g), (m, v)) in parameters.iter().zip(gradients).zip(&mut self.moments) {
            let lr = rate(self.learning_rate, g)?;
            *m = add(multiply(b1, &*m)?, multiply(1.0 - b1, g)?)?;
            *v = add(multiply(b2, &*v)?, multiply(1.0 - b2, multiply(g, g)?)?)?;
            let updated = if self.bias_correction {
                let step = self.step as i32;
                let c1 = divide(&lr, 1.0 - b1.powi(step))?.astype(g.dtype())?;
                let c2 = divide(1.0, sqrt(&Array::scalar((1.0 - b2.powi(step)) as f32))?)?
                    .astype(g.dtype())?;
                let denominator = add(multiply(sqrt(&*v)?, &c2)?, self.eps)?;
                subtract(p, divide(multiply(&c1, &*m)?, &denominator)?)?
            } else {
                subtract(p, divide(multiply(&lr, &*m)?, add(sqrt(&*v)?, self.eps)?)?)?
            };
            out.push(updated);
        }
        Ok(out)
    }
}
