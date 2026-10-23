//! Token generation, as mlx-lm's `generate_step`: the prompt is prefilled in chunks, then
//! each step is built and submitted while the GPU runs the previous one, and the previous
//! token is read afterwards; the KV cache is updated in place. While it runs,
//! the GPU keeps memory resident up to its recommended working set (mlx-lm's
//! `wired_limit`). Generation ends at the model's end-of-sequence tokens, as mlx-lm's
//! `stream_generate` (they are not yielded).

use std::sync::{Mutex, MutexGuard, PoisonError};

use nuvec::{Array, DeviceKind, Result, async_eval, default_device, eval};

use crate::model::{KvCache, Model};
use crate::sample::Sampler;

/// How tokens are chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerateOptions {
    /// Tokens to generate at most.
    pub max_tokens: usize,
    /// How each token is chosen (greedy by default).
    pub sampler: Sampler,
    /// Prompt tokens per prefill step (mlx-lm's 2048).
    pub prefill_step: usize,
    /// Sampling keys from this seed (one per token), so that the same seed gives the same
    /// tokens, alone or batched; without, the thread's key sequence ([`nuvec::random`]).
    pub seed: Option<u64>,
}

/// The sampling keys of one generation: split from its seed, one per token, or none (the
/// thread's sequence is used).
#[derive(Debug, Clone)]
pub(crate) struct Keys(Option<Array>);

impl Keys {
    pub(crate) fn new(seed: Option<u64>) -> Self {
        Keys(seed.map(nuvec::random::key))
    }

    /// The key for the next token.
    pub(crate) fn next(&mut self) -> Result<Option<Array>> {
        let Some(key) = self.0.take() else {
            return Ok(None);
        };
        let [keep, out] = nuvec::random::split_n::<2>(&key);
        self.0 = Some(keep);
        Ok(Some(out.check()?))
    }
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampler: Sampler::default(),
            prefill_step: 2048,
            seed: None,
        }
    }
}

/// The tokens generated after a prompt, one per `next` (an iterator).
#[derive(Debug)]
pub struct Generation<'a> {
    model: &'a Model,
    cache: Vec<KvCache>,
    options: GenerateOptions,
    /// The next token, submitted but maybe not computed yet.
    next: Option<Array>,
    produced: usize,
    wired: WiredLimit,
    keys: Keys,
}

/// mlx-lm's `wired_limit`: while it lives, the GPU keeps memory (the model's weights
/// first) resident up to its recommended working set; the previous limit returns when it is
/// dropped. [`generate`] holds one; holding one around it moves the cost of wiring out of
/// the generation (as mlx-lm's benchmarks do). Does nothing off the GPU, or when the limit
/// is already as high.
///
/// The limit is shared by the process: overlapping `WiredLimit`s (concurrent generations)
/// raise it once, and the previous limit returns when the last one is dropped.
#[derive(Debug)]
pub struct WiredLimit {
    /// Whether this one counts as a holder (on the GPU).
    pub(crate) active: bool,
}

/// The live [`WiredLimit`]s of the process, and the limit to restore after the last.
#[derive(Debug)]
struct Wired {
    holders: usize,
    previous: Option<usize>,
}

impl Wired {
    /// A new holder; the first raises the limit (`raise` returns the limit to restore).
    fn acquire(&mut self, raise: impl FnOnce() -> Option<usize>) {
        if self.holders == 0 {
            self.previous = raise();
        }
        self.holders += 1;
    }

    /// A holder ends; after the last, the previous limit is restored.
    fn release(&mut self, restore: impl FnOnce(usize)) {
        self.holders -= 1;
        if self.holders == 0
            && let Some(previous) = self.previous.take()
        {
            restore(previous);
        }
    }
}

static WIRED: Mutex<Wired> = Mutex::new(Wired {
    holders: 0,
    previous: None,
});

fn wired() -> MutexGuard<'static, Wired> {
    WIRED.lock().unwrap_or_else(PoisonError::into_inner)
}

impl WiredLimit {
    /// Raises the wired limit to the recommended working set.
    pub fn new() -> Self {
        let device =
            nuvec::gpu::Device::shared().filter(|_| default_device().kind == DeviceKind::Gpu);
        let Some(d) = device else {
            return WiredLimit { active: false };
        };
        wired().acquire(|| {
            let limit = d.info().recommended_working_set as usize;
            // Already as high (set by the caller): nothing to restore.
            d.set_wired_limit(limit)
                .ok()
                .filter(|&previous| previous < limit)
        });
        WiredLimit { active: true }
    }
}

impl Default for WiredLimit {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WiredLimit {
    fn drop(&mut self) {
        if self.active {
            wired().release(|previous| {
                if let Some(d) = nuvec::gpu::Device::shared() {
                    let _ = d.set_wired_limit(previous);
                }
            });
        }
    }
}

impl Drop for Generation<'_> {
    /// Waits for the submitted step before its memory is unwired (errors were or would
    /// have been reported by `next`).
    fn drop(&mut self) {
        if let (true, Some(y)) = (self.wired.active, self.next.take()) {
            let _ = eval(std::slice::from_ref(&y));
        }
    }
}

/// Starts generating after `prompt` (at least one token): prefills it and submits the
/// first step. Tokens are produced by iterating, up to `max_tokens`; an end-of-sequence
/// token of the model ([`crate::Config::eos_token_ids`]) ends the generation and is not
/// yielded. The sampler's settings are checked first ([`crate::Sampler::validate`]).
pub fn generate<'a>(
    model: &'a Model,
    prompt: &[u32],
    options: GenerateOptions,
) -> Result<Generation<'a>> {
    if prompt.is_empty() {
        return Err(nuvec::Error::InvalidArgument(
            "generate: empty prompt".into(),
        ));
    }
    options.sampler.validate()?;
    let wired = WiredLimit::new();
    let mut cache = model.new_cache();
    let (rest, last) = prompt.split_at(prompt.len() - 1);
    prefill(model, &mut cache, rest, options.prefill_step)?;
    let mut g = Generation {
        model,
        cache,
        options,
        next: None,
        produced: 0,
        wired,
        keys: Keys::new(options.seed),
    };
    if options.max_tokens > 0 {
        let first = g.step(&Array::from_slice(last, [1]))?;
        async_eval(std::slice::from_ref(&first))?;
        g.next = Some(first);
    }
    Ok(g)
}

/// Runs `tokens` through the model in chunks of `step`, extending `cache` (the logits are
/// never computed).
pub(crate) fn prefill(
    model: &Model,
    cache: &mut [KvCache],
    tokens: &[u32],
    step: usize,
) -> Result<()> {
    prefill_rows(model, cache, &[tokens], step)
}

/// [`prefill`] for several sequences of the same length at once (`cache` holds a batch of
/// `rows.len()`): each chunk is one `[B, n]` forward pass, chunked as `prefill` chunks one
/// sequence.
pub(crate) fn prefill_rows(
    model: &Model,
    cache: &mut [KvCache],
    rows: &[&[u32]],
    step: usize,
) -> Result<()> {
    let len = rows.first().map_or(0, |r| r.len());
    if rows.iter().any(|r| r.len() != len) {
        return Err(nuvec::Error::InvalidArgument(
            "prefill: sequences of different lengths".into(),
        ));
    }
    let b = i32::try_from(rows.len())
        .map_err(|_| nuvec::Error::InvalidArgument("prefill: too many sequences".into()))?;
    let mut start = 0;
    while start < len {
        let n = (len - start).min(step.max(1));
        let chunk: Vec<u32> = rows
            .iter()
            .flat_map(|r| &r[start..start + n])
            .copied()
            .collect();
        let chunk = Array::from_slice(&chunk, [b, n as i32]);
        let _ = model.forward(&chunk, cache).check()?;
        eval(&cache.iter().flat_map(KvCache::state).collect::<Vec<_>>())?;
        start += n;
    }
    Ok(())
}

impl Generation<'_> {
    /// The token following `input` (`[L]` ids), not evaluated.
    fn step(&mut self, input: &Array) -> Result<Array> {
        let logits = self.model.forward(&input.reshape([1, -1]), &mut self.cache);
        let last = logits.i((.., -1..)).reshape([1, -1]);
        let key = self.keys.next()?;
        let sampler = &self.options.sampler;
        if sampler.temperature == 0.0 {
            return sampler.sample(&last);
        }
        // Log-probabilities, as mlx-lm samples from.
        let logprobs = last.subtract(last.logsumexp_keepdims(-1));
        sampler.sample_with(&logprobs, key.as_ref())
    }

    /// The cache (one per layer), for inspection (its length, its memory): the positions
    /// the model has been fed, that is the prompt and the tokens yielded so far, the last
    /// one only if the step after it was submitted (it is unless the generation ended).
    /// Nothing in this crate continues a generation from it.
    pub fn cache(&self) -> &[KvCache] {
        &self.cache
    }
}

impl Iterator for Generation<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Result<u32>> {
        let y = self.next.take()?;
        self.produced += 1;
        let mut next = None;
        if self.produced < self.options.max_tokens {
            // Submit the following step, then wait for this token (mlx-lm's order): the GPU
            // never waits for the host between steps. The cache the step writes is still in
            // flight; nothing else refers to it, so it is updated in place.
            let step = match self.step(&y) {
                Ok(step) => step,
                Err(e) => return Some(Err(e)),
            };
            if let Err(e) = async_eval(std::slice::from_ref(&step)) {
                return Some(Err(e));
            }
            next = Some(step);
        }
        let token = match y.item::<u32>() {
            Ok(token) => token,
            Err(e) => return Some(Err(e)),
        };
        if self.model.config.eos_token_ids.contains(&token) {
            // The end: the step submitted after it is waited for and dropped.
            if let Some(step) = next {
                let _ = eval(std::slice::from_ref(&step));
            }
            return None;
        }
        self.next = next;
        Some(Ok(token))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::Wired;

    /// Two overlapping holders raise the limit once and restore the original only after
    /// both ended, whatever order they end in.
    #[test]
    fn overlapping_wired_limits_restore_after_the_last() {
        let limit = Cell::new(100usize);
        let raise = || Some(limit.replace(1000)).filter(|&previous| previous < 1000);
        let restore = |previous| limit.set(previous);
        let mut wired = Wired {
            holders: 0,
            previous: None,
        };
        wired.acquire(raise);
        wired.acquire(|| unreachable!("raised once"));
        assert_eq!(limit.get(), 1000);
        wired.release(restore);
        assert_eq!(limit.get(), 1000, "the other generation still runs");
        wired.release(restore);
        assert_eq!(limit.get(), 100, "restored after the last");
        // A limit already as high is left alone.
        limit.set(1000);
        wired.acquire(raise);
        wired.release(|_| unreachable!("nothing to restore"));
        assert_eq!(limit.get(), 1000);
    }
}
