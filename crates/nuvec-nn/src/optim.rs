//! Optimizers, learning-rate schedules and gradient clipping, with MLX's formulas
//! (`mlx.optimizers`).
//!
//! An optimizer updates the trainable leaves of a model from a gradient tree of the
//! model's type (as `nuvec::value_and_grad` returns it): `opt.update(&mut model, &grads)`.
//! Its state is a tree (`step`, `learning_rate`, and the slots by parameter path), so an
//! optimizer is part of a compiled training step's state and of a checkpoint
//! (`nuvec::save(path, &(model, opt))`):
//!
//! - state is keyed by parameter path: freezing, unfreezing or replacing parameters never
//!   moves state between them, and the state of a parameter that is not updated is kept;
//! - the step counter and the learning rate are 0-D arrays (as MLX's
//!   `optimizer.state["learning_rate"]`); a schedule writes the rate from the step at every
//!   update (evaluated before the step is counted, in float32), so a compiled step sees
//!   every new step and rate;
//! - hyperparameters are static fields of the tree: changing one retraces a compiled step.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use nuvec::tree::{Leaves, Visit, VisitMut};
use nuvec::{Array, Category, DType, Error, OpErrorKind, Result, Tree, ops};

use crate::Module;

/// Updates a model's trainable leaves from their gradients, as MLX's `Optimizer`.
///
/// The trait is object-safe (`Box<dyn Optimizer>` is an optimizer and a tree too). Its
/// state is reached only through [`Optimizer::visit_state`] and
/// [`Optimizer::visit_state_mut`] (each optimizer has its own slot type).
pub trait Optimizer {
    /// One step over named parameters (the path keys each parameter's state) and their
    /// gradients: the updated parameters, in order. Increments the step once.
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>>;

    /// Creates zero state for `params` that have none (paths with state are kept).
    fn init_params(&mut self, params: &[(&str, &Array)]);

    /// Visits the state (`step`, `learning_rate`, slots) and the hyperparameters.
    fn visit_state(&self, v: &mut Visit<'_>);

    /// Visits the state mutably.
    fn visit_state_mut(&mut self, v: &mut VisitMut<'_>);

    /// The learning rate: the 0-D float32 array in the state.
    fn learning_rate(&self) -> Array;

    /// Sets the learning rate array (in a compiled step it is an input: no retrace).
    fn set_learning_rate(&mut self, lr: f32);

    /// A boxed copy (object-safe `Clone`).
    fn clone_box(&self) -> Box<dyn Optimizer>;

    /// Creates the state of every trainable leaf of `model` now (instead of at the first
    /// update), so a compiled step's state structure is fixed from the start; also needed
    /// before loading a checkpoint into a fresh optimizer.
    fn init<M: Tree>(&mut self, model: &M)
    where
        Self: Sized,
    {
        let params = model.trainable_parameters();
        let refs: Vec<(&str, &Array)> = params.iter().map(|(p, a)| (p.as_str(), a)).collect();
        self.init_params(&refs);
    }

    /// Applies one step to the trainable leaves of `model` from `grads`, a gradient tree of
    /// the model's type with the model's paths (else an error). Frozen parameters and
    /// buffers are left alone; their state is kept.
    fn update<M: Tree>(&mut self, model: &mut M, grads: &M) -> Result<()>
    where
        Self: Sized,
    {
        let (params, gradients) = trainable_with_grads(model, grads)?;
        let p: Vec<(&str, &Array)> = params.iter().map(|(n, a)| (n.as_str(), a)).collect();
        let g: Vec<&Array> = gradients.iter().collect();
        let updated = self.apply(&p, &g)?;
        let mut updated: HashMap<&str, Array> = params
            .iter()
            .map(|(n, _)| n.as_str())
            .zip(updated)
            .collect();
        *model = model.map(|path, a| updated.remove(path).unwrap_or_else(|| a.clone()));
        Ok(())
    }
}

/// Named parameters and their gradients.
type Trainable = (Vec<(String, Array)>, Vec<Array>);

/// The trainable leaves of `model` with their paths, and their gradients from `grads`
/// (which must have the model's leaf paths).
fn trainable_with_grads<M: Tree>(model: &M, grads: &M) -> Result<Trainable> {
    let (paths, gpaths) = (model.paths(), grads.paths());
    if paths != gpaths {
        let first = paths.iter().zip(&gpaths).find(|(a, b)| a != b).map_or_else(
            || "a different number of leaves".to_owned(),
            |(a, b)| format!("{b:?} where the model has {a:?}"),
        );
        return Err(Error::InvalidArgument(format!(
            "update: the gradients do not have the model's paths: {first}"
        )));
    }
    let trainable: HashSet<String> = model
        .trainable_parameters()
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    let mut params = Vec::new();
    let mut gs = Vec::new();
    for ((path, a), g) in model.flatten().into_iter().zip(grads.leaves()) {
        if trainable.contains(&path) {
            params.push((path, a));
            gs.push(g);
        }
    }
    Ok((params, gs))
}

impl Optimizer for Box<dyn Optimizer> {
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>> {
        (**self).apply(params, grads)
    }
    fn init_params(&mut self, params: &[(&str, &Array)]) {
        (**self).init_params(params);
    }
    fn visit_state(&self, v: &mut Visit<'_>) {
        (**self).visit_state(v);
    }
    fn visit_state_mut(&mut self, v: &mut VisitMut<'_>) {
        (**self).visit_state_mut(v);
    }
    fn learning_rate(&self) -> Array {
        (**self).learning_rate()
    }
    fn set_learning_rate(&mut self, lr: f32) {
        (**self).set_learning_rate(lr);
    }
    fn clone_box(&self) -> Box<dyn Optimizer> {
        (**self).clone_box()
    }
}

impl Clone for Box<dyn Optimizer> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

/// A boxed optimizer is a tree through its state: the one hand-written tree impl of
/// nuvec-nn (a trait object cannot be derived).
impl Leaves for Box<dyn Optimizer> {
    fn visit(&self, v: &mut Visit<'_>) {
        (**self).visit_state(v);
    }
}

impl Tree for Box<dyn Optimizer> {
    fn visit_mut(&mut self, v: &mut VisitMut<'_>) {
        (**self).visit_state_mut(v);
    }
}

impl std::fmt::Debug for dyn Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dyn Optimizer")
    }
}

/// A boxed optimizer (a tree through [`Optimizer::visit_state`]).
type Boxed = Box<dyn Optimizer>;

/// An optimizer's state: the step counter and the learning rate as 0-D arrays, and one
/// slot (momenta, moment estimates) per parameter path.
#[derive(Debug, Clone, Tree)]
pub struct OptimizerState<S: Tree> {
    /// Steps taken (a 0-D uint64 array).
    pub step: Array,
    /// The learning rate (a 0-D float32 array).
    pub learning_rate: Array,
    /// The slots by parameter path.
    pub slots: BTreeMap<String, S>,
}

impl<S: Tree> OptimizerState<S> {
    fn new(lr: f32) -> Self {
        Self {
            step: Array::from(0u64),
            learning_rate: Array::from(lr),
            slots: BTreeMap::new(),
        }
    }
}

/// A learning rate: a constant or a function of the step (a 0-D array, from 0) returning a
/// 0-D array, so a compiled step sees every new rate.
#[derive(Clone)]
pub enum LearningRate {
    /// The same rate at every step (it initialises the state's rate array).
    Constant(f32),
    /// The rate at each step, written into the state before each update.
    Schedule(Arc<dyn Fn(&Array) -> Array + Send + Sync>),
}

impl LearningRate {
    /// The rate at `step` (a 0-D array), in float32.
    pub fn at(&self, step: &Array) -> Array {
        match self {
            Self::Constant(lr) => Array::from(*lr),
            Self::Schedule(f) => f(step).astype(DType::F32),
        }
    }

    fn initial(&self) -> Array {
        self.at(&Array::from(0u64))
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

/// Learning-rate schedules, as MLX's `mlx.optimizers` schedulers: float32 array functions of
/// the step.
pub mod schedules {
    use std::sync::Arc;

    use nuvec::{Array, DType};

    use super::LearningRate;

    fn schedule(f: impl Fn(&Array) -> Array + Send + Sync + 'static) -> LearningRate {
        LearningRate::Schedule(Arc::new(f))
    }

    /// The step in float32.
    fn steps(step: &Array) -> Array {
        step.astype(DType::F32)
    }

    /// `b^t` for a float32 array `t` (`exp(t ln b)`, in float32, `b` rounded to float32 as
    /// MLX does).
    fn power(b: f64, t: &Array) -> Array {
        t.multiply(f64::from(b as f32).ln()).exp()
    }

    /// `init · decay_rate^step`.
    pub fn exponential_decay(init: f64, decay_rate: f64) -> LearningRate {
        schedule(move |step| power(decay_rate, &steps(step)).multiply(init))
    }

    /// `init · decay_rate^⌊step / step_size⌋`.
    pub fn step_decay(init: f64, decay_rate: f64, step_size: u64) -> LearningRate {
        let size = step_size.max(1) as f64;
        schedule(move |step| {
            let k = steps(step)
                .divide(size)
                .astype(DType::U32)
                .astype(DType::F32);
            power(decay_rate, &k).multiply(init)
        })
    }

    /// From `init` to `end` along half a cosine over `decay_steps`, then `end`.
    pub fn cosine_decay(init: f64, decay_steps: u64, end: f64) -> LearningRate {
        let decay_steps = decay_steps.max(1) as f64;
        let rate = std::f64::consts::PI / decay_steps;
        schedule(move |step| {
            let s = steps(step).minimum(decay_steps);
            let decay = s.multiply(rate).cos().add(1.0).multiply(0.5);
            decay.multiply(init - end).add(end)
        })
    }

    /// From `init` to `end` linearly over `steps`, then `end`.
    pub fn linear_schedule(init: f64, end: f64, steps_total: u64) -> LearningRate {
        let n = steps_total.max(1) as f64;
        let slope = (end - init) / n;
        schedule(move |step| steps(step).minimum(n).multiply(slope).add(init))
    }

    /// `schedules[0]` until `boundaries[0]`, then `schedules[1]` (counting its steps
    /// from the boundary), and so on. `boundaries` must have one entry fewer than
    /// `schedules` (an error otherwise).
    pub fn join_schedules(
        schedules: Vec<LearningRate>,
        boundaries: Vec<u64>,
    ) -> nuvec::Result<LearningRate> {
        if schedules.len() != boundaries.len() + 1 {
            return Err(nuvec::Error::InvalidArgument(format!(
                "join_schedules: {} schedules need {} boundaries, got {}",
                schedules.len(),
                schedules.len().saturating_sub(1),
                boundaries.len()
            )));
        }
        Ok(schedule(move |step| {
            let t = steps(step);
            let mut out = schedules[0].at(&t);
            for (b, s) in boundaries.iter().zip(&schedules[1..]) {
                let b = *b as f64;
                let later = s.at(&t.subtract(b).maximum(0.0));
                out = t.less(b).select(&out, &later);
            }
            out
        }))
    }
}

/// The learning rate cast to the gradient's dtype.
fn rate(lr: &Array, like: &Array) -> Array {
    lr.astype(like.dtype())
}

/// One step of an optimizer with slot type `S`: checks the arguments, writes the scheduled
/// rate, increments the step, and updates each parameter with `f(p, g, slot, lr, step)`
/// (`step` counted already). The state changes only on success.
fn step<S: Tree>(
    state: &mut OptimizerState<S>,
    schedule: &LearningRate,
    params: &[(&str, &Array)],
    grads: &[&Array],
    new_slot: impl Fn(&Array) -> S,
    mut f: impl FnMut(&Array, &Array, &mut S, &Array, &Array) -> Array,
) -> Result<Vec<Array>> {
    if params.len() != grads.len() {
        return Err(Error::InvalidArgument(format!(
            "{} gradients for {} parameters",
            grads.len(),
            params.len()
        )));
    }
    let mut next = state.clone();
    if let LearningRate::Schedule(s) = schedule {
        next.learning_rate = s(&next.step).astype(DType::F32);
    }
    next.step = next.step.add(1u64);
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(params.len());
    for (&(path, p), &g) in params.iter().zip(grads) {
        if !seen.insert(path) {
            return Err(Error::InvalidArgument(format!(
                "parameter {path:?} given twice"
            )));
        }
        let slot = next
            .slots
            .entry(path.to_owned())
            .or_insert_with(|| new_slot(p));
        let mut mismatch = false;
        slot.for_each(|_, s| mismatch |= s.shape() != p.shape());
        if mismatch {
            return Err(Error::InvalidArgument(format!(
                "parameter {path:?} changed its shape to {:?}",
                p.shape()
            )));
        }
        out.push(f(p, g, slot, &next.learning_rate, &next.step).check()?);
    }
    *state = next;
    Ok(out)
}

/// Implements the object-safe part of [`Optimizer`] for an optimizer with a `state` field.
macro_rules! state_access {
    ($slot:ty) => {
        fn init_params(&mut self, params: &[(&str, &Array)]) {
            for &(path, p) in params {
                self.state
                    .slots
                    .entry(path.to_owned())
                    .or_insert_with(|| <$slot>::zeros(p));
            }
        }
        fn visit_state(&self, v: &mut Visit<'_>) {
            v.static_field("optimizer", &std::any::type_name::<Self>());
            Leaves::visit(self, v);
        }
        fn visit_state_mut(&mut self, v: &mut VisitMut<'_>) {
            Tree::visit_mut(self, v);
        }
        fn learning_rate(&self) -> Array {
            self.state.learning_rate.clone()
        }
        fn set_learning_rate(&mut self, lr: f32) {
            self.state.learning_rate = Array::from(lr);
        }
        fn clone_box(&self) -> Box<dyn Optimizer> {
            Box::new(self.clone())
        }
    };
}

/// One state array like `p` (zeros).
fn zeros(p: &Array) -> Array {
    ops::zeros(p.shape(), p.dtype())
}

/// SGD's state: the momentum buffer.
#[derive(Debug, Clone, Tree)]
pub struct SgdSlot {
    /// The momentum.
    pub v: Array,
}

impl SgdSlot {
    fn zeros(p: &Array) -> Self {
        Self { v: zeros(p) }
    }
}

/// Stochastic gradient descent with optional momentum, dampening, Nesterov momentum and
/// weight decay, as MLX's `optimizers.SGD`.
#[derive(Debug, Clone, Tree)]
pub struct Sgd {
    #[tree(skip, static_unchecked)]
    schedule: LearningRate,
    #[tree(skip)]
    momentum: f64,
    #[tree(skip)]
    weight_decay: f64,
    #[tree(skip)]
    dampening: f64,
    #[tree(skip)]
    nesterov: bool,
    #[tree(rename = "")]
    state: OptimizerState<SgdSlot>,
}

/// A fresh state whose rate array is the schedule's at step 0.
fn fresh<S: Tree>(schedule: &LearningRate) -> OptimizerState<S> {
    let mut state = OptimizerState::new(0.0);
    state.learning_rate = schedule.initial();
    state
}

impl Sgd {
    /// Plain SGD.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        let schedule = learning_rate.into();
        Self {
            state: fresh(&schedule),
            schedule,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
        }
    }

    /// Sets the momentum.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the (L2) weight decay added to the gradient.
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Sets the dampening of the momentum.
    pub fn dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    /// Enables Nesterov momentum.
    pub fn nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }
}

impl Optimizer for Sgd {
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>> {
        let (momentum, wd, damp, nesterov) = (
            self.momentum,
            self.weight_decay,
            self.dampening,
            self.nesterov,
        );
        let sgd = |p: &Array, g: &Array, s: &mut SgdSlot, lr: &Array, _: &Array| {
            let mut g = g.clone();
            if wd != 0.0 {
                g = g.add(p.multiply(wd));
            }
            let lr = rate(lr, &g);
            if momentum <= 0.0 {
                return p.subtract(lr.multiply(&g));
            }
            let mut next = s.v.multiply(momentum);
            next = if damp > 0.0 {
                next.add(g.multiply(1.0 - damp))
            } else {
                next.add(&g)
            };
            let update = if nesterov {
                g.add(next.multiply(momentum))
            } else {
                next.clone()
            };
            s.v = next;
            p.subtract(lr.multiply(&update))
        };
        step(
            &mut self.state,
            &self.schedule,
            params,
            grads,
            SgdSlot::zeros,
            sgd,
        )
    }

    state_access!(SgdSlot);
}

/// Adam's (and AdamW's) state: the first and second moment estimates.
#[derive(Debug, Clone, Tree)]
pub struct AdamSlot {
    /// The first moment.
    pub m: Array,
    /// The second moment.
    pub v: Array,
}

impl AdamSlot {
    fn zeros(p: &Array) -> Self {
        Self {
            m: zeros(p),
            v: zeros(p),
        }
    }
}

/// `1 - b^step` in float32, an array function of the step (MLX's bias correction, with `b`
/// rounded to float32 as MLX does). Computed as `-expm1(step · ln b)` with Kahan's
/// `expm1(x) = (u - 1) · x / log(u)`, `u = exp(x)`, which avoids the cancellation of
/// `1 - exp(x)` for `b` near 1 (0.999).
fn one_minus_power(b: f64, step: &Array) -> Array {
    let ln_b = f64::from(b as f32).ln();
    let x = step.astype(DType::F32).multiply(ln_b);
    let u = x.exp();
    let kahan = u.subtract(1.0).multiply(&x).divide(u.log());
    u.equal(1.0).select(&x, &kahan).negative()
}

/// Adam's hyperparameters.
#[derive(Debug, Clone, Copy)]
struct AdamParams {
    betas: (f64, f64),
    eps: f64,
    bias_correction: bool,
    /// AdamW's decoupled weight decay (0 for Adam).
    decay: f64,
}

/// One Adam step.
fn adam_step(
    p: &Array,
    g: &Array,
    s: &mut AdamSlot,
    lr: &Array,
    step: &Array,
    h: AdamParams,
) -> Array {
    let (b1, b2) = h.betas;
    let lr = rate(lr, g);
    let p = if h.decay != 0.0 {
        // MLX's AdamW: `parameter * (1 - lr * weight_decay)`.
        p.multiply(ops::subtract(1.0, lr.multiply(h.decay)))
    } else {
        p.clone()
    };
    s.m = s.m.multiply(b1).add(g.multiply(1.0 - b1));
    s.v = s.v.multiply(b2).add(g.multiply(g).multiply(1.0 - b2));
    if h.bias_correction {
        // In float32, as MLX (`1 - b ** step` on a float32 array).
        let c1 = lr.divide(one_minus_power(b1, step)).astype(g.dtype());
        let c2 =
            ops::divide(Array::from(1.0f32), one_minus_power(b2, step).sqrt()).astype(g.dtype());
        let denominator = s.v.sqrt().multiply(&c2).add(h.eps);
        p.subtract(c1.multiply(&s.m).divide(&denominator))
    } else {
        p.subtract(lr.multiply(&s.m).divide(s.v.sqrt().add(h.eps)))
    }
}

/// Adam, as MLX's `optimizers.Adam` (no weight decay; bias correction off unless
/// enabled, as in MLX). [`AdamW`] adds decoupled weight decay.
#[derive(Debug, Clone, Tree)]
pub struct Adam {
    #[tree(skip, static_unchecked)]
    schedule: LearningRate,
    #[tree(skip)]
    betas: (f64, f64),
    #[tree(skip)]
    eps: f64,
    #[tree(skip)]
    bias_correction: bool,
    #[tree(rename = "")]
    state: OptimizerState<AdamSlot>,
}

impl Adam {
    /// Adam with betas `(0.9, 0.999)` and `eps = 1e-8`.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        let schedule = learning_rate.into();
        Self {
            state: fresh(&schedule),
            schedule,
            betas: (0.9, 0.999),
            eps: 1e-8,
            bias_correction: false,
        }
    }

    /// Sets the betas.
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
        self
    }

    /// Sets `eps`, added to the denominator.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Enables bias correction.
    pub fn bias_correction(mut self) -> Self {
        self.bias_correction = true;
        self
    }
}

impl Optimizer for Adam {
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>> {
        let h = AdamParams {
            betas: self.betas,
            eps: self.eps,
            bias_correction: self.bias_correction,
            decay: 0.0,
        };
        let f = |p: &Array, g: &Array, s: &mut AdamSlot, lr: &Array, t: &Array| {
            adam_step(p, g, s, lr, t, h)
        };
        step(
            &mut self.state,
            &self.schedule,
            params,
            grads,
            AdamSlot::zeros,
            f,
        )
    }

    state_access!(AdamSlot);
}

/// AdamW, as MLX's `optimizers.AdamW`: Adam with decoupled weight decay (MLX's default
/// 0.01).
#[derive(Debug, Clone, Tree)]
pub struct AdamW {
    #[tree(skip, static_unchecked)]
    schedule: LearningRate,
    #[tree(skip)]
    betas: (f64, f64),
    #[tree(skip)]
    eps: f64,
    #[tree(skip)]
    bias_correction: bool,
    #[tree(skip)]
    weight_decay: f64,
    #[tree(rename = "")]
    state: OptimizerState<AdamSlot>,
}

impl AdamW {
    /// AdamW with MLX's defaults: betas `(0.9, 0.999)`, `eps = 1e-8`, weight decay 0.01.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        let schedule = learning_rate.into();
        Self {
            state: fresh(&schedule),
            schedule,
            betas: (0.9, 0.999),
            eps: 1e-8,
            bias_correction: false,
            weight_decay: 0.01,
        }
    }

    /// Sets the betas.
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
        self
    }

    /// Sets `eps`, added to the denominator.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the decoupled weight decay.
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enables bias correction.
    pub fn bias_correction(mut self) -> Self {
        self.bias_correction = true;
        self
    }
}

impl Optimizer for AdamW {
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>> {
        let h = AdamParams {
            betas: self.betas,
            eps: self.eps,
            bias_correction: self.bias_correction,
            decay: self.weight_decay,
        };
        let f = |p: &Array, g: &Array, s: &mut AdamSlot, lr: &Array, t: &Array| {
            adam_step(p, g, s, lr, t, h)
        };
        step(
            &mut self.state,
            &self.schedule,
            params,
            grads,
            AdamSlot::zeros,
            f,
        )
    }

    state_access!(AdamSlot);
}

/// Lion's (and Muon's) state: one momentum.
#[derive(Debug, Clone, Tree)]
pub struct MomentumSlot {
    /// The momentum.
    pub m: Array,
}

impl MomentumSlot {
    fn zeros(p: &Array) -> Self {
        Self { m: zeros(p) }
    }
}

/// Lion, as MLX's `optimizers.Lion`: the sign of an interpolated momentum, with optional
/// decoupled weight decay.
#[derive(Debug, Clone, Tree)]
pub struct Lion {
    #[tree(skip, static_unchecked)]
    schedule: LearningRate,
    #[tree(skip)]
    betas: (f64, f64),
    #[tree(skip)]
    weight_decay: f64,
    #[tree(rename = "")]
    state: OptimizerState<MomentumSlot>,
}

impl Lion {
    /// Lion with betas `(0.9, 0.99)` and no weight decay.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        let schedule = learning_rate.into();
        Self {
            state: fresh(&schedule),
            schedule,
            betas: (0.9, 0.99),
            weight_decay: 0.0,
        }
    }

    /// Sets the betas (interpolation, momentum).
    pub fn betas(mut self, b1: f64, b2: f64) -> Self {
        self.betas = (b1, b2);
        self
    }

    /// Sets the decoupled weight decay.
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Lion {
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>> {
        let ((b1, b2), wd) = (self.betas, self.weight_decay);
        let lion = |p: &Array, g: &Array, s: &mut MomentumSlot, lr: &Array, _: &Array| {
            let lr = rate(lr, g);
            let c = s.m.multiply(b1).add(g.multiply(1.0 - b1));
            s.m = s.m.multiply(b2).add(g.multiply(1.0 - b2));
            let p = if wd > 0.0 {
                ops::subtract(1.0, lr.multiply(wd)).multiply(p)
            } else {
                p.clone()
            };
            p.subtract(lr.multiply(c.sign()))
        };
        step(
            &mut self.state,
            &self.schedule,
            params,
            grads,
            MomentumSlot::zeros,
            lion,
        )
    }

    state_access!(MomentumSlot);
}

/// Muon, as MLX's `optimizers.Muon`: momentum (Nesterov by default) whose update is
/// orthogonalised by Newton–Schulz iterations for parameters of two or more dimensions.
#[derive(Debug, Clone, Tree)]
pub struct Muon {
    #[tree(skip, static_unchecked)]
    schedule: LearningRate,
    #[tree(skip)]
    momentum: f64,
    #[tree(skip)]
    weight_decay: f64,
    #[tree(skip)]
    nesterov: bool,
    #[tree(skip)]
    ns_steps: usize,
    #[tree(rename = "")]
    state: OptimizerState<MomentumSlot>,
}

impl Muon {
    /// Muon with MLX's defaults: momentum 0.95, weight decay 0.01, Nesterov, 5 steps.
    pub fn new(learning_rate: impl Into<LearningRate>) -> Self {
        let schedule = learning_rate.into();
        Self {
            state: fresh(&schedule),
            schedule,
            momentum: 0.95,
            weight_decay: 0.01,
            nesterov: true,
            ns_steps: 5,
        }
    }

    /// Sets the momentum.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the weight decay added to the gradient.
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Uses plain momentum instead of Nesterov's (MLX's default).
    pub fn without_nesterov(mut self) -> Self {
        self.nesterov = false;
        self
    }

    /// Sets the number of Newton–Schulz iterations.
    pub fn ns_steps(mut self, n: usize) -> Self {
        self.ns_steps = n;
        self
    }
}

impl Optimizer for Muon {
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>> {
        let (momentum, wd, nesterov, ns) = (
            self.momentum,
            self.weight_decay,
            self.nesterov,
            self.ns_steps,
        );
        let muon = |p: &Array, g: &Array, s: &mut MomentumSlot, lr: &Array, _: &Array| {
            let mut g = g.clone();
            if wd != 0.0 {
                g = g.add(p.multiply(wd));
            }
            s.m = s.m.multiply(momentum).add(g.multiply(1.0 - momentum));
            let mut update = if nesterov {
                g.multiply(1.0 - momentum).add(s.m.multiply(momentum))
            } else {
                s.m.clone()
            };
            let mut lr = rate(lr, &g);
            if update.ndim() >= 2 {
                let shape = update.shape().to_vec();
                let flat = update.reshape([shape[0], -1]);
                update = orthogonalize(&flat, ns).reshape(&shape[..]);
                let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
                lr = lr.multiply((f64::from(rows) / f64::from(cols)).max(1.0).sqrt());
            }
            p.subtract(lr.multiply(&update))
        };
        step(
            &mut self.state,
            &self.schedule,
            params,
            grads,
            MomentumSlot::zeros,
            muon,
        )
    }

    state_access!(MomentumSlot);
}

/// The nearest semi-orthogonal matrix to `x` (2-D), by MLX's quintic Newton–Schulz
/// iteration.
fn orthogonalize(x: &Array, steps: usize) -> Array {
    let (a, b, c) = (3.4445, -4.775, 2.0315);
    let transpose = x.shape()[0] > x.shape()[1];
    let mut x = if transpose { x.t() } else { x.clone() };
    let norm = x.multiply(&x).sum_keepdims(..).sqrt();
    x = x.divide(norm.add(1e-7));
    for _ in 0..steps {
        let am = x.matmul(&x.t());
        let bm = am.multiply(b).addmm(&am, &am, c as f32, 1.0);
        x = x.multiply(a).addmm(&bm, &x, 1.0, 1.0);
    }
    if transpose { x.t() } else { x }
}

/// Parameter groups, as MLX's `MultiOptimizer`: each group's filter selects parameters by
/// path (the first matching group wins) and its optimizer updates them; the default
/// optimizer takes the rest. Every group steps once per update. Its state is a tree:
/// `groups.0.…`, …, `default.…`.
///
/// ```
/// use nuvec_nn::optim::{Adam, MultiOptimizer, Sgd};
///
/// let opt = MultiOptimizer::new(Sgd::new(1e-2).weight_decay(1e-4))
///     .group(|path| path.contains("lora_a"), Adam::new(1e-3));
/// # let _ = opt;
/// ```
#[derive(Debug, Clone, Tree)]
pub struct MultiOptimizer {
    groups: Vec<Boxed>,
    default: Boxed,
    #[tree(skip, static_unchecked)]
    filters: Vec<Filter>,
}

/// A group's path filter.
#[derive(Clone)]
struct Filter(Arc<dyn Fn(&str) -> bool + Send + Sync>);

impl std::fmt::Debug for Filter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Filter")
    }
}

impl MultiOptimizer {
    /// Groups with `default` for every parameter no group selects.
    pub fn new(default: impl Optimizer + 'static) -> Self {
        Self {
            groups: Vec::new(),
            default: Box::new(default),
            filters: Vec::new(),
        }
    }

    /// Adds a group: `opt` updates the parameters whose path matches `filter` (and no
    /// earlier group's).
    pub fn group(
        mut self,
        filter: impl Fn(&str) -> bool + Send + Sync + 'static,
        opt: impl Optimizer + 'static,
    ) -> Self {
        self.groups.push(Box::new(opt));
        self.filters.push(Filter(Arc::new(filter)));
        self
    }

    /// The group of `path`: an index into `groups`, or `groups.len()` for the default.
    fn group_of(&self, path: &str) -> usize {
        self.filters
            .iter()
            .position(|f| (f.0)(path))
            .unwrap_or(self.groups.len())
    }
}

impl Optimizer for MultiOptimizer {
    fn apply(&mut self, params: &[(&str, &Array)], grads: &[&Array]) -> Result<Vec<Array>> {
        if params.len() != grads.len() {
            return Err(Error::InvalidArgument(format!(
                "{} gradients for {} parameters",
                grads.len(),
                params.len()
            )));
        }
        let n = self.groups.len() + 1;
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, (path, _)) in params.iter().enumerate() {
            members[self.group_of(path)].push(i);
        }
        let mut next = self.clone();
        let mut out: Vec<Option<Array>> = vec![None; params.len()];
        for (k, idx) in members.iter().enumerate() {
            let p: Vec<(&str, &Array)> = idx.iter().map(|&i| params[i]).collect();
            let g: Vec<&Array> = idx.iter().map(|&i| grads[i]).collect();
            let opt = if k < next.groups.len() {
                &mut next.groups[k]
            } else {
                &mut next.default
            };
            for (&i, a) in idx.iter().zip(opt.apply(&p, &g)?) {
                out[i] = Some(a);
            }
        }
        *self = next;
        Ok(out
            .into_iter()
            .map(|a| a.expect("every parameter has a group"))
            .collect())
    }

    fn init_params(&mut self, params: &[(&str, &Array)]) {
        let n = self.groups.len();
        for k in 0..=n {
            let p: Vec<(&str, &Array)> = params
                .iter()
                .copied()
                .filter(|(path, _)| self.group_of(path) == k)
                .collect();
            if k < n {
                self.groups[k].init_params(&p);
            } else {
                self.default.init_params(&p);
            }
        }
    }

    fn visit_state(&self, v: &mut Visit<'_>) {
        v.static_field("optimizer", &std::any::type_name::<Self>());
        Leaves::visit(self, v);
    }

    fn visit_state_mut(&mut self, v: &mut VisitMut<'_>) {
        Tree::visit_mut(self, v);
    }

    fn learning_rate(&self) -> Array {
        self.default.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        for g in &mut self.groups {
            g.set_learning_rate(lr);
        }
        self.default.set_learning_rate(lr);
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

/// `grads` scaled so that their joint L2 norm is at most `max_norm`, and that norm before
/// scaling, as MLX's `optimizers.clip_grad_norm` (float16 norms in float32). A negative
/// `max_norm` gives failed arrays.
#[track_caller]
pub fn clip_grad_norm<T: Tree>(grads: &T, max_norm: f32) -> (T, Array) {
    if max_norm < 0.0 {
        let failed = Array::failure(
            "clip_grad_norm",
            OpErrorKind::InvalidArgument(format!("max_norm should be >= 0, got {max_norm}")),
        );
        let clipped = grads.map(|_, g| failed.astype(g.dtype()).broadcast_to(g.shape()));
        return (clipped, failed);
    }
    let mut squared: Option<Array> = None;
    grads.for_each(|_, g| {
        let g = if g.dtype() == DType::F16 {
            g.astype(DType::F32)
        } else {
            g.clone()
        };
        let s = g.multiply(&g).sum(..);
        squared = Some(match squared.take() {
            Some(acc) => acc.add(&s),
            None => s,
        });
    });
    let total = squared.unwrap_or_else(|| Array::from(0.0f32)).sqrt();
    let normalizer = ops::divide(f64::from(max_norm), total.add(1e-6)).minimum(1.0);
    let clipped = grads.map(|_, g| {
        let scaled = g.multiply(&normalizer);
        if g.dtype().category() == Category::Float {
            scaled.astype(g.dtype())
        } else {
            scaled
        }
    });
    (clipped, total)
}
