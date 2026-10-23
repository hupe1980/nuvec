//! Token sampling, as mlx-lm's `sample_utils`: filters on log-probabilities (top-p,
//! min-p, top-k; masked tokens get `-inf`) and categorical sampling at a temperature.

use std::sync::LazyLock;

use nuvec::{Array, DType, Error, Result, ScanOptions, random};

fn invalid(msg: String) -> Error {
    Error::InvalidArgument(msg)
}

/// Keeps the most likely tokens whose probabilities sum to at least `top_p` (nucleus
/// sampling, mlx-lm's `apply_top_p`).
pub fn apply_top_p(logprobs: &Array, top_p: f32) -> Result<Array> {
    let sorted = logprobs.sort(-1);
    let probs = sorted.astype(DType::F32).exp();
    // The mass of the tokens after each one (a reversed, exclusive running sum).
    let after = ScanOptions {
        reverse: true,
        exclusive: true,
    };
    let above = probs.cumsum_with(-1, after);
    let total = above
        .i((nuvec::Ellipsis, ..1))
        .add(probs.i((nuvec::Ellipsis, ..1)));
    let dropped = above.greater_equal(total.multiply(f64::from(top_p)));
    let threshold = sorted.take_along_axis(&dropped.sum_keepdims(-1), -1);
    logprobs
        .less(&threshold)
        .select(f64::NEG_INFINITY, logprobs)
        .check()
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
    let scaled = logprobs.max_keepdims(-1).add(f64::from(min_p).ln());
    let mut remove = logprobs.less(&scaled);
    if min_tokens_to_keep > 1 {
        let keep = i32::try_from(min_tokens_to_keep)
            .map_err(|_| invalid(format!("min_tokens_to_keep {min_tokens_to_keep} too large")))?;
        let order = logprobs.argpartition(-keep, -1);
        let top_indices = order.i((nuvec::Ellipsis, -keep..));
        remove = remove.put_along_axis(&top_indices, Array::from(false), -1);
    }
    remove.select(f64::NEG_INFINITY, logprobs).check()
}

/// Keeps the `top_k` most likely tokens (mlx-lm's `apply_top_k`).
pub fn apply_top_k(logprobs: &Array, top_k: usize) -> Result<Array> {
    let vocab = logprobs.shape().last().copied().unwrap_or(0);
    let k = i32::try_from(top_k).ok().filter(|k| (1..vocab).contains(k));
    let Some(k) = k else {
        return Err(invalid(format!(
            "top_k must be in (0, {vocab}), got {top_k}"
        )));
    };
    let order = logprobs.negative().argpartition(k - 1, -1);
    let masked = order.i((nuvec::Ellipsis, k..));
    let inf = Array::from(f32::NEG_INFINITY).astype(logprobs.dtype());
    logprobs.put_along_axis(&masked, &inf, -1).check()
}

/// A compiled draw: `(key, logits, 1 / temperature)` to the token.
type Compiled = Box<dyn Fn(&(Array, Array, Array)) -> Result<Array> + Send + Sync>;

/// A token drawn from `logits / temperature` (MLX's `random.categorical`, Gumbel-max), with
/// the next key of this thread's sequence. Compiled, as mlx-lm's `categorical_sampling`:
/// the noise, the scaling and the sum run as one kernel before the argmax.
pub fn categorical(logits: &Array, temperature: f32) -> Result<Array> {
    categorical_with(logits, temperature, None)
}

/// [`categorical`] with `key`, or the next key of this thread's sequence without.
pub fn categorical_with(logits: &Array, temperature: f32, key: Option<&Array>) -> Result<Array> {
    static DRAW: LazyLock<Compiled> = LazyLock::new(|| {
        Box::new(nuvec::compile(
            |(key, logits, inverse): &(Array, Array, Array)| {
                let u = random::uniform(1e-30, 1.0, logits.shape(), DType::F32, key);
                let gumbel = u.log().negative().log().negative();
                let scaled = logits.astype(DType::F32).multiply(inverse);
                scaled.add(&gumbel).argmax(-1)
            },
        ))
    });
    // `1 / temperature` rounded to float32 once, as the weak scalar it replaces.
    let inverse = Array::from((1.0 / f64::from(temperature)) as f32);
    let key = match key {
        Some(k) => k.clone(),
        None => random::next_key(),
    };
    DRAW(&(key.check()?, logits.clone().check()?, inverse))
}

/// How the next token is chosen (mlx-lm's `make_sampler`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampler {
    /// At least 0 and finite; 0 picks the most likely token (the filters are then
    /// unused).
    pub temperature: f32,
    /// Nucleus sampling: keep the most likely tokens up to this probability mass, in
    /// [0, 1] (0 and 1 are off).
    pub top_p: f32,
    /// Keep tokens at least this fraction as likely as the most likely one, in [0, 1] (0
    /// is off).
    pub min_p: f32,
    /// Tokens min-p always keeps (at least 1).
    pub min_tokens_to_keep: usize,
    /// Keep only this many most likely tokens (0, or the vocabulary size or more, is off).
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
    /// Checks the settings: a finite temperature of at least 0, `top_p` and `min_p` in
    /// [0, 1], and `min_tokens_to_keep` at least 1 (NaN is never valid). Generation calls
    /// it before it starts ([`generate`](crate::generate()), [`crate::speculative_generate`],
    /// [`crate::Batcher::submit`]).
    pub fn validate(&self) -> Result<()> {
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return Err(invalid(format!(
                "sampler: temperature must be finite and at least 0, got {}",
                self.temperature
            )));
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(invalid(format!(
                "sampler: top_p must be in [0, 1], got {}",
                self.top_p
            )));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(invalid(format!(
                "sampler: min_p must be in [0, 1], got {}",
                self.min_p
            )));
        }
        if self.min_tokens_to_keep < 1 {
            return Err(invalid(
                "sampler: min_tokens_to_keep must be at least 1".into(),
            ));
        }
        Ok(())
    }

    /// The next token for log-probabilities `[1, vocab]`: the filters in mlx-lm's order
    /// (top-p, min-p, top-k, as `make_sampler` chains them), then a categorical draw;
    /// greedy at temperature 0.
    pub fn sample(&self, logprobs: &Array) -> Result<Array> {
        self.sample_with(logprobs, None)
    }

    /// [`Sampler::sample`] drawing with `key` (or, without, the next key of this thread's
    /// sequence).
    pub fn sample_with(&self, logprobs: &Array, key: Option<&Array>) -> Result<Array> {
        self.validate()?;
        if self.temperature == 0.0 {
            return logprobs.argmax(-1).check();
        }
        let mut lp = logprobs.clone();
        if 0.0 < self.top_p && self.top_p < 1.0 {
            lp = apply_top_p(&lp, self.top_p)?;
        }
        if self.min_p != 0.0 {
            lp = apply_min_p(&lp, self.min_p, self.min_tokens_to_keep)?;
        }
        let vocab = logprobs.shape().last().map_or(0, |&v| v as usize);
        // Keeping the whole vocabulary (or more) filters nothing.
        if self.top_k > 0 && self.top_k < vocab {
            lp = apply_top_k(&lp, self.top_k)?;
        }
        categorical_with(&lp, self.temperature, key)
    }
}
