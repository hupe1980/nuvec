//! Token generation, as mlx-lm's `generate_step`: the prompt is prefilled in chunks, then
//! each step is built and submitted while the GPU runs the previous one, and the previous
//! token is read afterwards; the KV cache is updated in place. While it runs,
//! the GPU keeps memory resident up to its recommended working set (mlx-lm's
//! `wired_limit`).

use nuvec::{Array, DeviceKind, Result, async_eval, default_device, eval, logsumexp, subtract};

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
        let (keep, out) = nuvec::random::split_pair(&key)?;
        self.0 = Some(keep);
        Ok(Some(out))
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
#[derive(Debug)]
pub struct WiredLimit(Option<usize>);

impl WiredLimit {
    /// Raises the wired limit to the recommended working set.
    pub fn new() -> Self {
        let device =
            nuvec::gpu::Device::shared().filter(|_| default_device().kind == DeviceKind::Gpu);
        WiredLimit(device.and_then(|d| {
            let limit = d.info().recommended_working_set as usize;
            // Already as high (an outer `WiredLimit`): nothing to restore.
            d.set_wired_limit(limit)
                .ok()
                .filter(|&previous| previous < limit)
        }))
    }
}

impl Default for WiredLimit {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for WiredLimit {
    fn drop(&mut self) {
        if let (Some(previous), Some(d)) = (self.0, nuvec::gpu::Device::shared()) {
            let _ = d.set_wired_limit(previous);
        }
    }
}

impl Drop for Generation<'_> {
    /// Waits for the submitted step before its memory is unwired (errors were or would
    /// have been reported by `next`).
    fn drop(&mut self) {
        if let (Some(_), Some(y)) = (self.wired.0, self.next.take()) {
            let _ = eval(std::slice::from_ref(&y));
        }
    }
}

/// Starts generating after `prompt` (at least one token): prefills it and submits the
/// first step. Tokens are produced by iterating.
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
        let first = g.step(&Array::from_slice(last, &[1])?)?;
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
    mut tokens: &[u32],
    step: usize,
) -> Result<()> {
    while !tokens.is_empty() {
        let n = tokens.len().min(step.max(1));
        let chunk = Array::from_slice(&tokens[..n], &[1, n])?;
        model.forward(&chunk, cache)?;
        eval(&cache.iter().flat_map(KvCache::state).collect::<Vec<_>>())?;
        tokens = &tokens[n..];
    }
    Ok(())
}

impl Generation<'_> {
    /// The token following `input` (`[L]` ids), not evaluated.
    fn step(&mut self, input: &Array) -> Result<Array> {
        let l = input.shape()[0] as isize;
        let logits = self
            .model
            .forward(&input.reshape(&[1, l])?, &mut self.cache)?;
        let last = nuvec::slice(
            &logits,
            &[
                nuvec::SliceArg::full(),
                nuvec::SliceArg::new(Some(l - 1), None, 1),
                nuvec::SliceArg::full(),
            ],
        )?
        .reshape(&[1, -1])?;
        let key = self.keys.next()?;
        let sampler = &self.options.sampler;
        if sampler.temperature == 0.0 {
            return sampler.sample(&last);
        }
        // Log-probabilities, as mlx-lm samples from.
        let logprobs = subtract(&last, logsumexp(&last, &[-1], true)?)?;
        sampler.sample_with(&logprobs, key.as_ref())
    }

    /// The cache (one per layer), for instance to continue a conversation.
    pub fn cache(&self) -> &[KvCache] {
        &self.cache
    }
}

impl Iterator for Generation<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Result<u32>> {
        let y = self.next.take()?;
        self.produced += 1;
        if self.produced < self.options.max_tokens {
            // Submit the following step, then wait for this token (mlx-lm's order): the GPU
            // never waits for the host between steps. The cache the step writes is still in
            // flight; nothing else refers to it, so it is updated in place.
            let next = match self.step(&y) {
                Ok(next) => next,
                Err(e) => return Some(Err(e)),
            };
            if let Err(e) = async_eval(std::slice::from_ref(&next)) {
                return Some(Err(e));
            }
            self.next = Some(next);
            return Some(y.item::<u32>());
        }
        Some(y.item::<u32>())
    }
}
