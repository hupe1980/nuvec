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
use crate::stop::{Stop, StopState, Stopped};

/// How tokens are chosen and where generation ends.
#[derive(Debug, Clone, PartialEq)]
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
    /// Stop tokens and stop strings ([`crate::stop`]); none by default.
    pub stop: Stop,
    /// Whether the model's end-of-sequence tokens are ordinary tokens, emitted, so that
    /// generation runs to `max_tokens` or a stop (vLLM's `ignore_eos`).
    pub ignore_eos: bool,
    /// The first `min_tokens` tokens are never end-of-sequence or stop tokens (their
    /// probability is zero), and no stop string ends generation before the
    /// `min_tokens`-th token (vLLM's `min_tokens`).
    pub min_tokens: usize,
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
            stop: Stop::default(),
            ignore_eos: false,
            min_tokens: 0,
        }
    }
}

impl GenerateOptions {
    /// Checks the sampler's settings ([`Sampler::validate`]) and that `min_tokens` is at
    /// most `max_tokens`.
    pub fn validate(&self) -> Result<()> {
        self.sampler.validate()?;
        if self.min_tokens > self.max_tokens {
            return Err(nuvec::Error::InvalidArgument(format!(
                "generate: min_tokens ({}) exceeds max_tokens ({})",
                self.min_tokens, self.max_tokens
            )));
        }
        Ok(())
    }

    /// `logits` (`[rows, vocab]`) for the `n`-th generated token (from 1): within the
    /// first `min_tokens`, the tokens that would end generation (`eos` unless ignored, and
    /// the stop tokens) are at `-inf`.
    pub(crate) fn mask(&self, logits: Array, eos: &[u32], n: usize) -> Array {
        if n > self.min_tokens {
            return logits;
        }
        let vocab = logits.shape().last().map_or(0, |&v| v as u32);
        let ids: Vec<u32> = (!self.ignore_eos)
            .then_some(eos)
            .into_iter()
            .flatten()
            .chain(self.stop.stop_tokens())
            .copied()
            .filter(|&t| t < vocab)
            .collect();
        if ids.is_empty() {
            return logits;
        }
        let n = ids.len() as i32;
        logits
            .at((.., &Array::from_slice(&ids, [n])))
            .set(f32::NEG_INFINITY)
    }
}

/// Why a generation finished ([`Generation::finish_reason`], [`crate::Event::Finished`]).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// The model produced an end-of-sequence token (not emitted).
    EndOfSequence,
    /// It produced its `max_tokens`.
    Length,
    /// A stop token (not emitted) or a stop string ([`GenerateOptions::stop`]).
    Stop(Stopped),
    /// Its prompt and tokens no longer fit in the cache, even alone (a batched request).
    CacheFull,
    /// It was cancelled ([`crate::Batcher::cancel`]).
    Cancelled,
    /// Admitting or decoding it failed; a batcher's other requests go on.
    Error(nuvec::Error),
}

/// What `token`, the `n`-th generated (from 1), does: whether it is emitted, and why
/// generation ends with it. Shared by every generation path, so they end alike.
pub(crate) fn judge(
    options: &GenerateOptions,
    eos: &[u32],
    stops: &mut StopState,
    token: u32,
    n: usize,
) -> (bool, Option<FinishReason>) {
    if !options.ignore_eos && eos.contains(&token) {
        return (false, Some(FinishReason::EndOfSequence));
    }
    match stops.push(&options.stop, token, n, options.min_tokens) {
        Some(s @ Stopped::Token(_)) => (false, Some(FinishReason::Stop(s))),
        Some(s) => (true, Some(FinishReason::Stop(s))),
        None if n >= options.max_tokens => (true, Some(FinishReason::Length)),
        None => (true, None),
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
    stops: StopState,
    finished: Option<FinishReason>,
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
        let info = nuvec::metal::device_info().filter(|_| default_device().kind == DeviceKind::Gpu);
        let Some(info) = info else {
            return WiredLimit { active: false };
        };
        wired().acquire(|| {
            let limit = info.recommended_working_set as usize;
            // Already as high (set by the caller): nothing to restore.
            nuvec::memory::set_wired_limit(limit)
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
                let _ = nuvec::memory::set_wired_limit(previous);
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
/// first step. Tokens are produced by iterating, up to `max_tokens`. An end-of-sequence
/// token of the model ([`crate::Config::eos_token_ids`]) or a stop token ends the
/// generation and is not yielded; a stop string ends it after the token that completes it
/// ([`crate::stop`]). [`Generation::finish_reason`] says why it ended. The options are
/// checked first ([`GenerateOptions::validate`]).
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
    options.validate()?;
    let wired = WiredLimit::new();
    let mut cache = model.new_cache();
    let (rest, last) = prompt.split_at(prompt.len() - 1);
    prefill(model, &mut cache, rest, options.prefill_step)?;
    let mut g = Generation {
        model,
        cache,
        next: None,
        produced: 0,
        wired,
        keys: Keys::new(options.seed),
        stops: StopState::default(),
        finished: None,
        options,
    };
    if g.options.max_tokens > 0 {
        let first = g.step(&Array::from_slice(last, [1]))?;
        async_eval(std::slice::from_ref(&first))?;
        g.next = Some(first);
    } else {
        g.finished = Some(FinishReason::Length);
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
        // The chunk's intermediates leave the allocator's cache, as mlx-lm's prefill.
        nuvec::memory::clear_cache();
        start += n;
    }
    Ok(())
}

impl Generation<'_> {
    /// The token following `input` (`[L]` ids), not evaluated.
    fn step(&mut self, input: &Array) -> Result<Array> {
        let logits = self.model.forward(&input.reshape([1, -1]), &mut self.cache);
        let last = logits.i((.., -1..)).reshape([1, -1]);
        let eos = &self.model.config.eos_token_ids;
        let last = self.options.mask(last, eos, self.produced + 1);
        let sampler = &self.options.sampler;
        let token = if sampler.temperature == 0.0 {
            sampler.sample(&last)?
        } else {
            let key = self.keys.next()?;
            // Log-probabilities, as mlx-lm samples from.
            let logprobs = last.subtract(last.logsumexp_keepdims(-1));
            sampler.sample_with(&logprobs, key.as_ref())?
        };
        // A pipeline stage yields rank 0's token (the only rank that runs the head).
        let deterministic = sampler.temperature == 0.0 || self.options.seed.is_some();
        Ok(self.model.share_sampled(&token, deterministic))
    }

    /// The cache (one per layer), for inspection (its length, its memory): the positions
    /// the model has been fed, that is the prompt and the tokens yielded so far, the last
    /// one only if the step after it was submitted (it is unless the generation ended).
    /// Nothing in this crate continues a generation from it.
    pub fn cache(&self) -> &[KvCache] {
        &self.cache
    }

    /// Why the generation ended, once it has (the iterator returned its last item).
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.finished.as_ref()
    }

    /// Ends the generation: the step submitted after the last token, if any, is waited
    /// for and dropped.
    fn end(&mut self, reason: FinishReason, submitted: Option<Array>) {
        if let Some(step) = submitted {
            let _ = eval(std::slice::from_ref(&step));
        }
        self.finished = Some(reason);
    }
}

/// Whether the allocator's cache is emptied when the caller asks for the token after
/// `produced` tokens: after the first and every 256 after it, as mlx-lm's `generate_step`
/// clears after yielding token `n` with `n % 256 == 0` (outside that token's time).
fn clears_before(produced: usize) -> bool {
    produced > 0 && (produced - 1).is_multiple_of(256)
}

impl Iterator for Generation<'_> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Result<u32>> {
        let y = self.next.take()?;
        if clears_before(self.produced) {
            nuvec::memory::clear_cache();
        }
        self.produced += 1;
        let mut next = None;
        if self.produced < self.options.max_tokens {
            // Submit the following step, then wait for this token (mlx-lm's order): the GPU
            // never waits for the host between steps. The cache the step writes is still in
            // flight; nothing else refers to it, so it is updated in place.
            let step = match self.step(&y) {
                Ok(step) => step,
                Err(e) => {
                    self.end(FinishReason::Error(e.clone()), None);
                    return Some(Err(e));
                }
            };
            if let Err(e) = async_eval(std::slice::from_ref(&step)) {
                self.end(FinishReason::Error(e.clone()), None);
                return Some(Err(e));
            }
            next = Some(step);
        }
        let token = match y.item::<u32>() {
            Ok(token) => token,
            Err(e) => {
                self.end(FinishReason::Error(e.clone()), next);
                return Some(Err(e));
            }
        };
        let eos = &self.model.config.eos_token_ids;
        let (emit, reason) = judge(&self.options, eos, &mut self.stops, token, self.produced);
        if let Some(reason) = reason {
            self.end(reason, next);
            return emit.then_some(Ok(token));
        }
        self.next = next;
        Some(Ok(token))
    }
}

#[cfg(test)]
mod tests {
    /// mlx-lm's schedule: token `n` (0-based) is followed by a clear when `n % 256 == 0`.
    #[test]
    fn the_cache_is_cleared_after_mlx_lm_s_tokens() {
        let clears: Vec<usize> = (0..800).filter(|&p| super::clears_before(p)).collect();
        assert_eq!(clears, [1, 257, 513, 769]);
    }

    /// Prefilling prompts of the same length together gives each row's cache the keys and
    /// values of the prompt prefilled alone (the `Batcher`'s grouped prefill), for every
    /// family's attention layers: sliding windows, K = V layers with their own KV-head
    /// count, KV-shared layers, Gated DeltaNet states.
    #[test]
    fn grouped_prefill_rows_equal_each_prompt_alone() {
        nuvec::with_row_exact(|| {
            use super::prefill_rows;
            use nuvec::Array;
            if !nuvec::with_stream(nuvec::Device::gpu(0), || {
                (nuvec::Array::from(1.0f32) + 1.0).eval().is_ok()
            }) {
                return;
            }
            let bits = |a: &Array| {
                a.astype(nuvec::DType::F32)
                    .view(nuvec::DType::U32)
                    .to_vec::<u32>()
                    .unwrap()
            };
            for name in [
                "gemma4_moe_q4",
                "qwen3_5_q4",
                "qwen3_5_moe_q4",
                "gpt_oss_q4",
            ] {
                let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures")
                    .join(name);
                let model = crate::load_model(&dir).unwrap();
                for len in [9u32, 70, 186] {
                    let rows: Vec<Vec<u32>> = (0..4u32)
                        .map(|r| (0..len).map(|t| (t * 7 + r * 3) % 50 + 1).collect())
                        .collect();
                    let refs: Vec<&[u32]> = rows.iter().map(Vec::as_slice).collect();
                    let mut both = model.new_cache();
                    prefill_rows(&model, &mut both, &refs, 2048).unwrap();
                    for (i, row) in rows.iter().enumerate() {
                        let mut alone = model.new_cache();
                        prefill_rows(&model, &mut alone, &[row.as_slice()], 2048).unwrap();
                        for (layer, (b, a)) in both.iter().zip(&alone).enumerate() {
                            let b = b.row(i as i32);
                            let (bs, as_) = (b.state(), a.state());
                            assert_eq!(bs.len(), as_.len(), "{name} layer {layer}");
                            for (x, y) in bs.iter().zip(&as_) {
                                assert_eq!(x.shape(), y.shape(), "{name} layer {layer} row {i}");
                                assert!(
                                    bits(x) == bits(y),
                                    "{name} length {len} layer {layer} row {i}: cache differs"
                                );
                            }
                        }
                    }
                }
            }
        });
    }
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
