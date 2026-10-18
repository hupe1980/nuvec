//! Optimizers, learning-rate schedules and gradient clipping, with MLX's formulas
//! (`mlx.optimizers`).
//!
//! Every optimizer updates the trainable parameters of a [`Module`] from gradients in the
//! order of [`Module::trainable_parameters`] (as [`crate::value_and_grad`] returns them).
//! The learning rate is a constant or a [`schedules`] function of the step, evaluated
//! before the step is counted and kept in float32, as in MLX.

use std::sync::Arc;

use nuvec::{
    Array, DType, Error, Result, add, addmm, divide, matmul, minimum, multiply, reshape, sign,
    sqrt, subtract, sum,
};

use crate::Module;

/// Updates a model's trainable parameters from their gradients.
pub trait Optimizer {
    /// Applies one step to `model`: `gradients` are in the order of
    /// [`Module::trainable_parameters`].
    fn update<M: Module>(&mut self, model: &mut M, gradients: &[Array]) -> Result<()> {
        let parameters: Vec<Array> = model
            .trainable_parameters()
            .into_iter()
            .map(|(_, a)| a)
            .collect();
        if parameters.len() != gradients.len() {
            return Err(Error::InvalidArgument(format!(
                "{} gradients for {} parameters",
                gradients.len(),
                parameters.len()
            )));
        }
        let updated = self.apply(&parameters, gradients)?;
        model.update_trainable(&updated)
    }

    /// One step: the updated parameters.
    fn apply(&mut self, parameters: &[Array], gradients: &[Array]) -> Result<Vec<Array>>;
}

/// A learning rate: a constant or a function of the step (from 0).
#[derive(Clone)]
pub enum LearningRate {
    /// The same rate at every step.
    Constant(f32),
    /// The rate at each step.
    Schedule(Arc<dyn Fn(u64) -> f32 + Send + Sync>),
}

impl LearningRate {
    /// The rate at `step`.
    pub fn at(&self, step: u64) -> f32 {
        match self {
            Self::Constant(lr) => *lr,
            Self::Schedule(f) => f(step),
        }
    }
}

impl std::fmt::Debug for LearningRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(lr) => write!(f, "Constant({lr})"),
            Self::Schedule(_) => f.write_str("Schedule"),
        }
    }
}

impl From<f64> for LearningRate {
    fn from(lr: f64) -> Self {
        Self::Constant(lr as f32)
    }
}

impl From<f32> for LearningRate {
    fn from(lr: f32) -> Self {
        Self::Constant(lr)
    }
}

/// Learning-rate schedules, as MLX's `mlx.optimizers` schedulers (computed in float32).
pub mod schedules {
    use std::sync::Arc;

    use super::LearningRate;

    fn schedule(f: impl Fn(u64) -> f32 + Send + Sync + 'static) -> LearningRate {
        LearningRate::Schedule(Arc::new(f))
    }

    /// `init · decay_rate^step`.
    pub fn exponential_decay(init: f64, decay_rate: f64) -> LearningRate {
        schedule(move |step| init as f32 * (decay_rate as f32).powf(step as f32))
    }

    /// `init · decay_rate^⌊step / step_size⌋`.
    pub fn step_decay(init: f64, decay_rate: f64, step_size: u64) -> LearningRate {
        let step_size = step_size.max(1);
        schedule(move |step| init as f32 * (decay_rate as f32).powf((step / step_size) as f32))
    }

    /// From `init` to `end` along half a cosine over `decay_steps`, then `end`.
    pub fn cosine_decay(init: f64, decay_steps: u64, end: f64) -> LearningRate {
        let decay_steps = decay_steps.max(1);
        let rate = (std::f64::consts::PI / decay_steps as f64) as f32;
        schedule(move |step| {
            let s = step.min(decay_steps) as f32;
            let decay = 0.5f32 * (1.0 + (rate * s).cos());
            end as f32 + decay * (init - end) as f32
        })
    }

    /// From `init` to `end` linearly over `steps`, then `end`.
    pub fn linear_schedule(init: f64, end: f64, steps: u64) -> LearningRate {
        let steps = steps.max(1);
        let slope = ((end - init) / steps as f64) as f32;
        schedule(move |step| step.min(steps) as f32 * slope + init as f32)
    }

    /// `schedules[0]` until `boundaries[0]`, then `schedules[1]` (counting its steps
    /// from the boundary), and so on (`boundaries` has one entry fewer).
    pub fn join_schedules(schedules: Vec<LearningRate>, boundaries: Vec<u64>) -> LearningRate {
        assert_eq!(
            schedules.len(),
            boundaries.len() + 1,
            "join_schedules: one boundary between consecutive schedules"
        );
        schedule(move |step| {
            let mut out = schedules[0].at(step);
            for (b, s) in boundaries.iter().zip(&schedules[1..]) {
                if step >= *b {
                    out = s.at(step - b);
                }
            }
            out
        })
    }
}

/// The learning rate as MLX stores it (float32) cast to the gradient's dtype.
fn rate(lr: f32, like: &Array) -> Result<Array> {
    Array::scalar(lr).astype(like.dtype())
}

/// Zeros shaped like each parameter, created on first use.
fn zeros_like(state: &mut Vec<Array>, parameters: &[Array]) -> Result<()> {
    if state.len() != parameters.len() {
        *state = parameters
            .iter()
            .map(|p| nuvec::zeros(p.shape(), p.dtype()))
            .collect::<Result<_>>()?;
    }
    Ok(())
}

/// Stochastic gradient descent with optional momentum, dampening, Nesterov momentum and
/// weight decay, as MLX's `optimizers.SGD`.
#[derive(Debug, Clone)]
pub struct Sgd {
    /// The learning rate.
    pub learning_rate: LearningRate,
    /// The momentum strength (0 disables momentum).
    pub momentum: f64,
    /// L2 weight decay added to the gradient.
    pub weight_decay: f64,
    /// Dampening for momentum.
    pub dampening: f64,
    /// Nesterov momentum.
    pub nesterov: bool,
    step: u64,
    velocity: Vec<Array>,
}

impl Sgd {
    /// Plain SGD.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            step: 0,
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
        zeros_like(&mut self.velocity, parameters)?;
        let lr_now = self.learning_rate.at(self.step);
        self.step += 1;
        let mut out = Vec::with_capacity(parameters.len());
        for ((p, g), v) in parameters.iter().zip(gradients).zip(&mut self.velocity) {
            let mut g = g.clone();
            if self.weight_decay != 0.0 {
                g = add(&g, multiply(p, self.weight_decay)?)?;
            }
            let lr = rate(lr_now, &g)?;
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

/// Adam, as MLX's `optimizers.Adam` (bias correction off unless enabled, as in MLX), and
/// AdamW (decoupled weight decay) with [`Adam::weight_decay`].
#[derive(Debug, Clone)]
pub struct Adam {
    /// The learning rate.
    pub learning_rate: LearningRate,
    /// Decay rates of the first and second moment estimates.
    pub betas: (f64, f64),
    /// Added to the denominator for numerical stability.
    pub eps: f64,
    /// Whether to correct the moment estimates' bias toward zero.
    pub bias_correction: bool,
    /// Decoupled weight decay (AdamW); 0 for Adam.
    pub weight_decay: f64,
    step: u64,
    m: Vec<Array>,
    v: Vec<Array>,
}

impl Adam {
    /// Adam with betas `(0.9, 0.999)` and `eps = 1e-8`.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            betas: (0.9, 0.999),
            eps: 1e-8,
            bias_correction: false,
            weight_decay: 0.0,
            step: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// AdamW, as MLX's `optimizers.AdamW`: Adam with decoupled weight decay (default
    /// 0.01).
    pub fn adamw(learning_rate: impl Into<LearningRate>) -> Self {
        Self::new(learning_rate).weight_decay(0.01)
    }

    /// Sets the betas.
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
        self
    }

    /// Sets the decoupled weight decay (AdamW).
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enables bias correction.
    pub fn bias_correction(mut self, enabled: bool) -> Self {
        self.bias_correction = enabled;
        self
    }
}

impl Optimizer for Adam {
    fn apply(&mut self, parameters: &[Array], gradients: &[Array]) -> Result<Vec<Array>> {
        zeros_like(&mut self.m, parameters)?;
        zeros_like(&mut self.v, parameters)?;
        let lr_now = self.learning_rate.at(self.step);
        self.step += 1;
        let (b1, b2) = self.betas;
        let mut out = Vec::with_capacity(parameters.len());
        for (((p, g), m), v) in parameters
            .iter()
            .zip(gradients)
            .zip(&mut self.m)
            .zip(&mut self.v)
        {
            let lr = rate(lr_now, g)?;
            let p = if self.weight_decay != 0.0 {
                // MLX's AdamW: `parameter * (1 - lr * weight_decay)`.
                multiply(p, subtract(1.0, multiply(&lr, self.weight_decay)?)?)?
            } else {
                p.clone()
            };
            *m = add(multiply(b1, &*m)?, multiply(1.0 - b1, g)?)?;
            *v = add(multiply(b2, &*v)?, multiply(1.0 - b2, multiply(g, g)?)?)?;
            let updated = if self.bias_correction {
                // In float32, as MLX (`1 - b ** step` on a float32 array).
                let step = self.step as f32;
                let one_minus = |b: f64| Array::scalar(1.0 - (b as f32).powf(step));
                let c1 = divide(&lr, one_minus(b1))?.astype(g.dtype())?;
                let c2 = divide(1.0f32, sqrt(&one_minus(b2))?)?.astype(g.dtype())?;
                let denominator = add(multiply(sqrt(&*v)?, &c2)?, self.eps)?;
                subtract(&p, divide(multiply(&c1, &*m)?, &denominator)?)?
            } else {
                subtract(&p, divide(multiply(&lr, &*m)?, add(sqrt(&*v)?, self.eps)?)?)?
            };
            out.push(updated);
        }
        Ok(out)
    }
}

/// Lion, as MLX's `optimizers.Lion`: the sign of an interpolated momentum, with optional
/// decoupled weight decay.
#[derive(Debug, Clone)]
pub struct Lion {
    /// The learning rate.
    pub learning_rate: LearningRate,
    /// Interpolation (`b1`) and momentum (`b2`) rates; MLX's default `(0.9, 0.99)`.
    pub betas: (f64, f64),
    /// Decoupled weight decay.
    pub weight_decay: f64,
    step: u64,
    m: Vec<Array>,
}

impl Lion {
    /// Lion with betas `(0.9, 0.99)` and no weight decay.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            betas: (0.9, 0.99),
            weight_decay: 0.0,
            step: 0,
            m: Vec::new(),
        }
    }

    /// Sets the decoupled weight decay.
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Lion {
    fn apply(&mut self, parameters: &[Array], gradients: &[Array]) -> Result<Vec<Array>> {
        zeros_like(&mut self.m, parameters)?;
        let lr_now = self.learning_rate.at(self.step);
        self.step += 1;
        let (b1, b2) = self.betas;
        let mut out = Vec::with_capacity(parameters.len());
        for ((p, g), m) in parameters.iter().zip(gradients).zip(&mut self.m) {
            let lr = rate(lr_now, g)?;
            let c = add(multiply(b1, &*m)?, multiply(1.0 - b1, g)?)?;
            *m = add(multiply(b2, &*m)?, multiply(1.0 - b2, g)?)?;
            let p = if self.weight_decay > 0.0 {
                multiply(subtract(1.0, multiply(&lr, self.weight_decay)?)?, p)?
            } else {
                p.clone()
            };
            out.push(subtract(&p, multiply(&lr, sign(&c)?)?)?);
        }
        Ok(out)
    }
}

/// Muon, as MLX's `optimizers.Muon`: momentum (Nesterov by default) whose update is
/// orthogonalised by Newton–Schulz iterations for parameters of two or more dimensions.
#[derive(Debug, Clone)]
pub struct Muon {
    /// The learning rate.
    pub learning_rate: LearningRate,
    /// The momentum (MLX's default 0.95).
    pub momentum: f64,
    /// Weight decay added to the gradient (MLX's default 0.01).
    pub weight_decay: f64,
    /// Nesterov momentum (MLX's default).
    pub nesterov: bool,
    /// Newton–Schulz iterations (MLX's default 5).
    pub ns_steps: usize,
    step: u64,
    v: Vec<Array>,
}

impl Muon {
    /// Muon with MLX's defaults.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        Self {
            learning_rate: learning_rate.into(),
            momentum: 0.95,
            weight_decay: 0.01,
            nesterov: true,
            ns_steps: 5,
            step: 0,
            v: Vec::new(),
        }
    }
}

impl Optimizer for Muon {
    fn apply(&mut self, parameters: &[Array], gradients: &[Array]) -> Result<Vec<Array>> {
        zeros_like(&mut self.v, parameters)?;
        let lr_now = self.learning_rate.at(self.step);
        self.step += 1;
        let mut out = Vec::with_capacity(parameters.len());
        for ((p, g), v) in parameters.iter().zip(gradients).zip(&mut self.v) {
            let mut g = g.clone();
            if self.weight_decay != 0.0 {
                g = add(&g, multiply(self.weight_decay, p)?)?;
            }
            *v = add(
                multiply(self.momentum, &*v)?,
                multiply(1.0 - self.momentum, &g)?,
            )?;
            let mut update = if self.nesterov {
                add(
                    multiply(&g, 1.0 - self.momentum)?,
                    multiply(&*v, self.momentum)?,
                )?
            } else {
                v.clone()
            };
            let mut lr = rate(lr_now, &g)?;
            if update.ndim() >= 2 {
                let shape = update.shape().to_vec();
                let flat = reshape(&update, &[shape[0] as isize, -1])?;
                let isize_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
                update = reshape(&orthogonalize(&flat, self.ns_steps)?, &isize_shape)?;
                let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
                lr = multiply(&lr, (rows as f64 / cols as f64).max(1.0).sqrt())?;
            }
            out.push(subtract(p, multiply(&lr, &update)?)?);
        }
        Ok(out)
    }
}

/// The nearest semi-orthogonal matrix to `x` (2-D), by MLX's quintic Newton–Schulz
/// iteration.
fn orthogonalize(x: &Array, steps: usize) -> Result<Array> {
    let (a, b, c) = (3.4445, -4.775, 2.0315);
    let transpose = x.shape()[0] > x.shape()[1];
    let mut x = if transpose { x.t()? } else { x.clone() };
    let norm = sqrt(&sum(&multiply(&x, &x)?, &[], true)?)?;
    x = divide(&x, add(&norm, 1e-7)?)?;
    for _ in 0..steps {
        let am = matmul(&x, &x.t()?)?;
        let bm = addmm(&multiply(&am, b)?, &am, &am, c as f32, 1.0)?;
        x = addmm(&multiply(&x, a)?, &bm, &x, 1.0, 1.0)?;
    }
    if transpose { x.t() } else { Ok(x) }
}

/// `gradients` scaled so that their joint L2 norm is at most `max_norm`, and that norm
/// before scaling, as MLX's `optimizers.clip_grad_norm` (float16 norms in float32).
pub fn clip_grad_norm(gradients: &[Array], max_norm: f64) -> Result<(Vec<Array>, Array)> {
    if max_norm < 0.0 {
        return Err(Error::InvalidArgument(format!(
            "clip_grad_norm: max_norm should be >= 0, got {max_norm}"
        )));
    }
    let mut squared: Option<Array> = None;
    for g in gradients {
        let g = if g.dtype() == DType::F16 {
            g.astype(DType::F32)?
        } else {
            g.clone()
        };
        let s = sum(&multiply(&g, &g)?, &[], false)?;
        squared = Some(match squared {
            Some(acc) => add(&acc, &s)?,
            None => s,
        });
    }
    let total = sqrt(&squared.unwrap_or_else(|| Array::scalar(0.0f32)))?;
    let normalizer = minimum(divide(max_norm, add(&total, 1e-6)?)?, 1.0)?;
    let clipped = gradients
        .iter()
        .map(|g| {
            let scaled = multiply(g, &normalizer)?;
            if g.dtype().category() == nuvec::Category::Float {
                scaled.astype(g.dtype())
            } else {
                Ok(scaled)
            }
        })
        .collect::<Result<_>>()?;
    Ok((clipped, total))
}
