//! Token sampling, as mlx-lm's `sample_utils`: filters on log-probabilities (top-p,
//! min-p, top-k; masked tokens get `-inf`) and categorical sampling at a temperature.

use std::sync::LazyLock;

use nuvec::{
    Array, DType, Error, Result, add, argmax, argpartition, cumsum, exp, greater_equal, less, log,
    max, multiply, negative, put_along_axis, random, slice, sort, sum, take_along_axis, where_,
};

fn invalid(msg: String) -> Error {
    Error::InvalidArgument(msg)
}

/// The last axis from position `start` on.
fn tail(a: &Array, start: isize) -> Result<Array> {
    let mut args = vec![nuvec::SliceArg::full(); a.ndim()];
    *args.last_mut().expect("at least one axis") = nuvec::SliceArg::new(Some(start), None, 1);
    slice(a, &args)
}

/// Keeps the most likely tokens whose probabilities sum to at least `top_p` (nucleus
/// sampling, mlx-lm's `apply_top_p`).
pub fn apply_top_p(logprobs: &Array, top_p: f32) -> Result<Array> {
    let sorted = sort(logprobs, Some(-1))?;
    let probs = exp(&sorted.astype(DType::F32)?)?;
    let above = cumsum(&probs, Some(-1), true, false)?;
    let first = |a: &Array| -> Result<Array> {
        let mut args = vec![nuvec::SliceArg::full(); a.ndim()];
        *args.last_mut().expect("at least one axis") = nuvec::SliceArg::new(Some(0), Some(1), 1);
        slice(a, &args)
    };
    let total = add(first(&above)?, first(&probs)?)?;
    let dropped = sum(
        &greater_equal(&above, multiply(&total, f64::from(top_p))?)?,
        &[-1],
        true,
    )?;
    let threshold = take_along_axis(&sorted, &dropped, -1)?;
    where_(&less(logprobs, &threshold)?, f64::NEG_INFINITY, logprobs)
}

/// Keeps tokens at least `min_p` times as likely as the most likely one, and at least
/// `min_tokens_to_keep` tokens (mlx-lm's `apply_min_p`).
pub fn apply_min_p(logprobs: &Array, min_p: f32, min_tokens_to_keep: usize) -> Result<Array> {
    if !(0.0..=1.0).contains(&min_p) {
        return Err(invalid(format!("min_p must be in [0, 1], got {min_p}")));
    }
    if min_tokens_to_keep < 1 {
        return Err(invalid("min_tokens_to_keep must be at least 1".into()));
    }
    let top = max(logprobs, &[-1], true)?;
    let scaled = add(&top, f64::from(min_p).ln())?;
    let mut remove = less(logprobs, &scaled)?;
    if min_tokens_to_keep > 1 {
        let keep = -(min_tokens_to_keep as isize);
        let top_indices = tail(&argpartition(logprobs, keep, Some(-1))?, keep)?;
        remove = put_along_axis(&remove, &top_indices, &Array::scalar(false), -1)?;
    }
    where_(&remove, f64::NEG_INFINITY, logprobs)
}

/// Keeps the `top_k` most likely tokens (mlx-lm's `apply_top_k`).
pub fn apply_top_k(logprobs: &Array, top_k: usize) -> Result<Array> {
    let vocab = logprobs.shape().last().copied().unwrap_or(0);
    if !(1..vocab).contains(&top_k) {
        return Err(invalid(format!(
            "top_k must be in (0, {vocab}), got {top_k}"
        )));
    }
    let order = argpartition(&negative(logprobs)?, top_k as isize - 1, Some(-1))?;
    let masked = tail(&order, top_k as isize)?;
    let inf = Array::scalar(f32::NEG_INFINITY).astype(logprobs.dtype())?;
    put_along_axis(logprobs, &masked, &inf, -1)
}

type Compiled = Box<dyn Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync>;

/// A token drawn from `logits / temperature` (MLX's `random.categorical`, Gumbel-max), with
/// the next key of this thread's sequence. Compiled, as mlx-lm's `categorical_sampling`:
/// the noise, the scaling and the sum run as one kernel before the argmax.
pub fn categorical(logits: &Array, temperature: f32) -> Result<Array> {
    categorical_with(logits, temperature, None)
}

/// [`categorical`] with `key`, or the next key of this thread's sequence without.
pub fn categorical_with(logits: &Array, temperature: f32, key: Option<&Array>) -> Result<Array> {
    static DRAW: LazyLock<Compiled> = LazyLock::new(|| {
        Box::new(nuvec::compile(|a: &[Array]| {
            let (key, logits, inverse) = (&a[0], &a[1], &a[2]);
            let u = random::uniform(1e-30, 1.0, logits.shape(), DType::F32, Some(key))?;
            let gumbel = negative(&log(&negative(&log(&u)?)?)?)?;
            let scaled = multiply(&logits.astype(DType::F32)?, inverse)?;
            Ok(vec![argmax(&add(&scaled, &gumbel)?, Some(-1), false)?])
        }))
    });
    // `1 / temperature` rounded to float32 once, as the weak scalar it replaces.
    let inverse = Array::scalar((1.0 / f64::from(temperature)) as f32);
    let key = match key {
        Some(k) => k.clone(),
        None => random::next_key()?,
    };
    Ok(DRAW(&[key, logits.clone(), inverse])?.swap_remove(0))
}

/// How the next token is chosen (mlx-lm's `make_sampler`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampler {
    /// 0 picks the most likely token (the filters are then unused).
    pub temperature: f32,
    /// Nucleus sampling: keep the most likely tokens up to this probability mass
    /// (off unless in (0, 1)).
    pub top_p: f32,
    /// Keep tokens at least this fraction as likely as the most likely one (0 is off).
    pub min_p: f32,
    /// Tokens min-p always keeps.
    pub min_tokens_to_keep: usize,
    /// Keep only this many most likely tokens (0 is off).
    pub top_k: usize,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 0.0,
            min_p: 0.0,
            min_tokens_to_keep: 1,
            top_k: 0,
        }
    }
}

impl Sampler {
    /// The next token for log-probabilities `[1, vocab]`: the filters in mlx-lm's order
    /// (top-p, min-p, top-k), then a categorical draw; greedy at temperature 0.
    pub fn sample(&self, logprobs: &Array) -> Result<Array> {
        self.sample_with(logprobs, None)
    }

    /// [`Sampler::sample`] drawing with `key` (or, without, the next key of this thread's
    /// sequence).
    pub fn sample_with(&self, logprobs: &Array, key: Option<&Array>) -> Result<Array> {
        if self.temperature == 0.0 {
            return argmax(logprobs, Some(-1), false);
        }
        let mut lp = logprobs.clone();
        if 0.0 < self.top_p && self.top_p < 1.0 {
            lp = apply_top_p(&lp, self.top_p)?;
        }
        if self.min_p != 0.0 {
            lp = apply_min_p(&lp, self.min_p, self.min_tokens_to_keep)?;
        }
        if self.top_k > 0 {
            lp = apply_top_k(&lp, self.top_k)?;
        }
        categorical_with(&lp, self.temperature, key)
    }
}
