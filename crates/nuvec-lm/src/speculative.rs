//! Speculative decoding: a drafter proposes the next tokens, one forward pass verifies them
//! all, and the longest prefix the model agrees with is kept, plus the model's own next
//! token. Verifying several tokens in one pass costs much less than generating them one by
//! one, so every accepted draft token saves most of a step.
//!
//! The output is the model's greedy output, bit for bit: a verify pass computes each row
//! as serial decode would (nuvec's row independence, up to
//! [`nuvec::ROW_INDEPENDENT_ROWS`] rows, under which verification runs:
//! [`nuvec::with_row_exact`]; and its attention planner's serial rule), and the verify
//! width is capped where that holds. Speculation is batch 1 on the contiguous
//! cache; the [`Batcher`](crate::Batcher) and paged caches decode without it.
//!
//! Drafters: [`PromptLookup`] (n-grams of the context, no model), [`MtpDrafter`] (the
//! model's own multi-token-prediction head, Qwen 3.5–3.8) and [`SuffixDecoding`] (a
//! suffix index over the prompt and earlier outputs).

use nuvec::{Array, Result, async_eval, eval, ops};

use crate::generate::{GenerateOptions, Keys, WiredLimit};
use crate::model::{KvCache, Model};

pub use crate::suffix::SuffixDecoding;

/// Draft tokens, `[n]` (uint32): on the device for model drafters, built on the host for
/// the others.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Draft {
    /// The tokens, `[n]`.
    pub tokens: Array,
}

impl Draft {
    /// No draft: the step generates one token as usual.
    pub fn empty() -> Self {
        Self::from_tokens(&[])
    }

    /// A draft of host tokens.
    pub fn from_tokens(tokens: &[u32]) -> Self {
        Self {
            tokens: Array::from_slice(tokens, [tokens.len() as i32]),
        }
    }

    /// A draft of tokens computed on the device (`[n]`, uint32; its length is its shape's,
    /// known without evaluating it).
    pub fn new(tokens: Array) -> Self {
        Self { tokens }
    }

    /// The number of draft tokens.
    pub fn len(&self) -> usize {
        self.tokens.shape().first().map_or(0, |&n| n as usize)
    }

    /// Whether there is no draft.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Proposes the next tokens. The generation calls [`Drafter::observe`] after every step of
/// the model (the prompt's prefill chunks, plain steps and verifications) with the tokens
/// the context grew by, so a drafter with its own state stays aligned with the model.
pub trait Drafter {
    /// Up to `max` tokens to follow `context` (the prompt and the tokens so far); a longer
    /// draft is cut.
    fn draft(&mut self, model: &Model, context: &[u32], max: usize) -> Result<Draft>;

    /// The context grew by `tokens` (`[n]`, uint32, maybe not computed yet); with
    /// [`Drafter::needs_hidden`], `hidden` (`[1, n, hidden_size]`) holds the model's final
    /// hidden state at the position before each token.
    fn observe(&mut self, model: &Model, tokens: &Array, hidden: Option<&Array>) -> Result<()> {
        let _ = (model, tokens, hidden);
        Ok(())
    }

    /// Whether [`Drafter::observe`] needs the model's hidden states (model drafters).
    fn needs_hidden(&self) -> bool {
        false
    }

    /// The draft length and break-even acceptance [`DraftPolicy::Auto`] uses for `model`
    /// (measured on an M4 Pro). The default: prompt lookup's.
    fn auto(&self, model: &Model) -> Auto {
        let _ = model;
        Auto {
            len: 4,
            break_even: 0.5,
        }
    }
}

/// [`DraftPolicy::Auto`]'s setting for a drafter and model: the draft length, and the
/// accepted tokens per verification below which drafting does not pay (the verification's
/// cost in plain steps, minus one).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Auto {
    /// Tokens drafted per step.
    pub len: usize,
    /// The break-even mean of accepted tokens per verification.
    pub break_even: f32,
}

impl<D: Drafter + ?Sized> Drafter for &mut D {
    fn draft(&mut self, model: &Model, context: &[u32], max: usize) -> Result<Draft> {
        (**self).draft(model, context, max)
    }

    fn observe(&mut self, model: &Model, tokens: &Array, hidden: Option<&Array>) -> Result<()> {
        (**self).observe(model, tokens, hidden)
    }

    fn needs_hidden(&self) -> bool {
        (**self).needs_hidden()
    }

    fn auto(&self, model: &Model) -> Auto {
        (**self).auto(model)
    }
}

impl<D: Drafter + ?Sized> Drafter for Box<D> {
    fn draft(&mut self, model: &Model, context: &[u32], max: usize) -> Result<Draft> {
        (**self).draft(model, context, max)
    }

    fn observe(&mut self, model: &Model, tokens: &Array, hidden: Option<&Array>) -> Result<()> {
        (**self).observe(model, tokens, hidden)
    }

    fn needs_hidden(&self) -> bool {
        (**self).needs_hidden()
    }

    fn auto(&self, model: &Model) -> Auto {
        (**self).auto(model)
    }
}

/// Prompt lookup (n-gram) drafting: the tokens that followed the most recent earlier
/// occurrence of the context's last `n` tokens, trying the longest `n` first (by default
/// only 3: shorter suffixes recur everywhere and draft badly). It needs no model, and
/// drafts well when the output repeats its input (code edits, quotes, summaries, tool
/// calls).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptLookup {
    /// The longest suffix matched.
    pub max_ngram: usize,
    /// The shortest suffix matched.
    pub min_ngram: usize,
}

impl Default for PromptLookup {
    fn default() -> Self {
        Self {
            max_ngram: 3,
            min_ngram: 3,
        }
    }
}

impl PromptLookup {
    /// The draft of host tokens.
    pub fn lookup(&self, context: &[u32], max: usize) -> Vec<u32> {
        let len = context.len();
        for n in (self.min_ngram.max(1)..=self.max_ngram).rev() {
            if len <= n {
                continue;
            }
            let suffix = &context[len - n..];
            // The most recent earlier occurrence (not the suffix itself).
            let found = (0..len - n).rev().find(|&i| &context[i..i + n] == suffix);
            if let Some(i) = found {
                let start = i + n;
                return context[start..(start + max).min(len)].to_vec();
            }
        }
        Vec::new()
    }
}

impl Drafter for PromptLookup {
    fn draft(&mut self, _: &Model, context: &[u32], max: usize) -> Result<Draft> {
        Ok(Draft::from_tokens(&self.lookup(context, max)))
    }
}

/// Drafts with the model's multi-token-prediction head (Qwen 3.5–3.8,
/// [`Model::attach_mtp`]), as mlx-vlm's `qwen3_5_mtp` drafter: the first draft token comes
/// from the head on the last verified position, each further one from the head on its own
/// previous hidden state. After a verification the drafted positions are dropped from the
/// head's cache and the verified ones fed with the model's hidden states, so its cache
/// always holds the model's positions.
#[derive(Debug)]
pub struct MtpDrafter {
    cache: Vec<KvCache>,
    /// The next draft token (`[1]`) and the head's hidden state it came from.
    seed: Option<(Array, Array)>,
    /// Positions the drafting added to the head's cache since the last observation.
    drafted: usize,
}

impl MtpDrafter {
    /// A drafter for `model`; an error if it has no MTP head.
    pub fn new(model: &Model) -> Result<Self> {
        if !model.has_mtp() {
            return Err(nuvec::Error::InvalidArgument(
                "MtpDrafter: the model has no MTP weights (Model::attach_mtp)".into(),
            ));
        }
        Ok(Self {
            cache: model.new_mtp_cache(),
            seed: None,
            drafted: 0,
        })
    }
}

impl Drafter for MtpDrafter {
    fn draft(&mut self, model: &Model, _: &[u32], max: usize) -> Result<Draft> {
        let Some((first, hidden)) = self.seed.clone().filter(|_| max > 0) else {
            return Ok(Draft::empty());
        };
        let (mut token, mut h) = (first.clone(), hidden);
        let mut tokens = vec![first];
        while tokens.len() < max {
            h = model.mtp_forward(&token.reshape([1, 1]), &h, &mut self.cache);
            self.drafted += 1;
            token = model.logits(&h).argmax(-1).reshape([1]);
            tokens.push(token.clone());
        }
        Ok(Draft::new(ops::concatenate(&tokens, 0)))
    }

    fn observe(&mut self, model: &Model, tokens: &Array, hidden: Option<&Array>) -> Result<()> {
        let Some(hidden) = hidden else {
            return Err(nuvec::Error::InvalidArgument(
                "MtpDrafter: observed without hidden states".into(),
            ));
        };
        for c in &mut self.cache {
            c.trim(self.drafted)?;
        }
        self.drafted = 0;
        let n = tokens.shape()[0];
        if n == 0 {
            return Ok(());
        }
        let h = model.mtp_forward(&tokens.reshape([1, n]), hidden, &mut self.cache);
        let last = h.i((.., n - 1.., ..));
        let token = model.logits(&last).argmax(-1).reshape([1]);
        self.seed = Some((token, last));
        Ok(())
    }

    fn needs_hidden(&self) -> bool {
        true
    }

    /// One draft token (further depths reuse the head's own hidden state, and are rarely
    /// accepted). Measured on an M4 Pro (natural text, 256 tokens): one draft token gives
    /// Qwen3.5-9B 1.36× and Qwen3.8-27B 1.66× plain decode (a verification costs 1.24 and
    /// 1.07 plain steps); on Qwen3.6-35B-A3B, whose two rows read different experts, 1.01×
    /// (1.67 plain steps).
    fn auto(&self, model: &Model) -> Auto {
        // Below the measured break-even (1.67 − 1) for MoE models: a window of 16
        // verifications scatters around the mean, and pausing at the mean itself flaps.
        let break_even = match &model.config.moe {
            Some(_) => 0.6,
            None => 0.24,
        };
        Auto { len: 1, break_even }
    }
}

/// How many tokens each step drafts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DraftPolicy {
    /// The drafter's measured length for the model ([`Drafter::auto`]) with a kill
    /// switch: when the mean of accepted tokens over the last 16 verifications falls below
    /// the break-even, the next 64 tokens are decoded plainly (pipelined), then drafting
    /// resumes. So drafting never costs much on text it cannot predict.
    #[default]
    Auto,
    /// Up to this many, every step (A/B measurements).
    Fixed(usize),
    /// Up to this many, pausing after rejected drafts: after a draft with nothing
    /// accepted, drafting pauses for 1, 2, 4, … (at most 16) steps, until a draft is
    /// accepted again.
    Adaptive(usize),
}

/// The verifications [`DraftPolicy::Auto`] averages, and the tokens it then pauses for.
const WINDOW: usize = 16;
const PAUSE: usize = 64;

/// [`DraftPolicy::Auto`]'s kill switch: the accepted tokens of the last [`WINDOW`]
/// verifications.
#[derive(Debug, Default)]
struct KillSwitch {
    window: std::collections::VecDeque<usize>,
}

impl KillSwitch {
    /// Records a verification; whether drafting should pause (the window's mean is below
    /// `break_even`), which starts a new window.
    fn record(&mut self, accepted: usize, break_even: f32) -> bool {
        self.window.push_back(accepted);
        if self.window.len() > WINDOW {
            self.window.pop_front();
        }
        let mean = self.window.iter().sum::<usize>() as f32 / self.window.len() as f32;
        let pause = self.window.len() == WINDOW && mean < break_even;
        if pause {
            self.window.clear();
        }
        pause
    }
}

/// Counts of a speculative generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpeculativeStats {
    /// Forward passes after the prompt (plain steps and verifications).
    pub steps: usize,
    /// Verification passes (steps with a draft).
    pub verifies: usize,
    /// Draft tokens proposed.
    pub drafted: usize,
    /// Draft tokens accepted.
    pub accepted: usize,
    /// Times [`DraftPolicy::Auto`]'s kill switch paused drafting.
    pub pauses: usize,
}

/// The most rows a verification has, so that each is computed as serial decode computes
/// it: nuvec's row bound and, for head sizes 192 and 256 (whose float32 rows need the
/// one-key-per-step kernel, and compose beyond one vector threadgroup), the rows of one
/// threadgroup (32 SIMD-groups over the grouped query heads).
pub fn verify_rows(model: &Model) -> usize {
    let c = &model.config;
    let gqa = (c.num_attention_heads / c.num_key_value_heads.max(1)).max(1);
    match c.head_dim {
        192 | 256 => nuvec::ROW_INDEPENDENT_ROWS.min(32 / gqa).max(1),
        _ => nuvec::ROW_INDEPENDENT_ROWS,
    }
}

/// The tokens generated after a prompt with speculative decoding, one per `next`.
pub struct SpeculativeGeneration<'a, D: Drafter> {
    model: &'a Model,
    drafter: D,
    cache: Vec<KvCache>,
    options: GenerateOptions,
    policy: DraftPolicy,
    /// The draft length and break-even of [`DraftPolicy::Auto`].
    auto: Auto,
    /// The kill switch of [`DraftPolicy::Auto`].
    switch: KillSwitch,
    /// The prompt and every token generated so far.
    context: Vec<u32>,
    /// Tokens verified but not yet returned.
    pending: std::collections::VecDeque<u32>,
    produced: usize,
    stats: SpeculativeStats,
    /// Steps left without drafting, and the next pause after a rejected draft.
    pause: (usize, usize),
    /// Without a draft, steps are pipelined as in [`crate::generate`]: the next token,
    /// submitted but not read yet.
    inflight: Option<Array>,
    /// Whether the drafter had a draft after the last token (the pipeline then drains,
    /// and the next step verifies a draft).
    draft_due: bool,
    /// The draft a model-free drafter gave for the current context when asked whether it
    /// has one: used by the next step instead of drafting again.
    probed: Option<Draft>,
    /// Sampling keys (one per step) when the temperature is above 0.
    keys: Keys,
    _wired: WiredLimit,
}

impl<D: Drafter> std::fmt::Debug for SpeculativeGeneration<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeculativeGeneration")
            .field("produced", &self.produced)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// Starts generating after `prompt` (at least one token) with speculative decoding:
/// `drafter` proposes tokens as `policy` says (at most [`verify_rows`]` − 1` per step).
/// Gated DeltaNet layers keep every position's state while a draft is verified, and roll
/// back to the last accepted one ([`KvCache::keep_steps`]).
///
/// Greedy (temperature 0) output is plain greedy decoding's, bit for bit. With a
/// temperature, a verification accepts by the lossless rule
/// ([`nuvec::speculative::accept_sampled`]) against the sampler's filtered distribution
/// ([`crate::Sampler::probabilities`]), so tokens are distributed as plain sampling's (not
/// the same tokens for the same seed: the keys are used differently); a seed still repeats
/// a run.
pub fn speculative_generate<'a, D: Drafter>(
    model: &'a Model,
    prompt: &[u32],
    options: GenerateOptions,
    mut drafter: D,
    policy: DraftPolicy,
) -> Result<SpeculativeGeneration<'a, D>> {
    let invalid = |m: &str| Err(nuvec::Error::InvalidArgument(format!("speculative: {m}")));
    if prompt.is_empty() {
        return invalid("empty prompt");
    }
    options.sampler.validate()?;
    let auto = drafter.auto(model);
    let wired = WiredLimit::new();
    let mut cache = model.new_cache();
    // The prompt but its last token, in chunks; a drafter that wants them sees each chunk's
    // hidden states (paired with the tokens that follow, all known here).
    let head = &prompt[..prompt.len() - 1];
    let step = options.prefill_step.max(1);
    let mut start = 0;
    while start < head.len() {
        let n = (head.len() - start).min(step);
        let chunk = Array::from_slice(&head[start..start + n], [1, n as i32]);
        let mut states: Vec<Array> = Vec::new();
        if drafter.needs_hidden() {
            let (_, hidden) = model.forward_hidden(&chunk, &mut cache);
            let next = Array::from_slice(&prompt[start + 1..start + n + 1], [n as i32]);
            drafter.observe(model, &next, Some(&hidden))?;
            states.push(hidden);
        } else {
            let _ = model.forward(&chunk, &mut cache).check()?;
        }
        states.extend(cache.iter().flat_map(KvCache::state));
        eval(&states)?;
        start += n;
    }
    Ok(SpeculativeGeneration {
        model,
        drafter,
        cache,
        options,
        policy,
        auto,
        switch: KillSwitch::default(),
        context: prompt.to_vec(),
        pending: Default::default(),
        produced: 0,
        stats: SpeculativeStats::default(),
        pause: (0, 1),
        inflight: None,
        draft_due: false,
        probed: None,
        keys: Keys::new(options.seed),
        _wired: wired,
    })
}

impl<D: Drafter> SpeculativeGeneration<'_, D> {
    /// The draft length the policy asks for.
    fn draft_len(&self) -> usize {
        match self.policy {
            DraftPolicy::Auto => self.auto.len,
            DraftPolicy::Fixed(n) | DraftPolicy::Adaptive(n) => n,
        }
    }

    /// The counts so far.
    pub fn stats(&self) -> SpeculativeStats {
        self.stats
    }

    /// The token chosen from `logits` `[1, 1, vocab]`: the argmax, or a sample.
    fn choose(&mut self, logits: &Array) -> Result<Array> {
        let sampler = self.options.sampler;
        if sampler.temperature == 0.0 {
            return Ok(logits.argmax(-1).reshape([1]));
        }
        let last = logits.reshape([1, -1]);
        let logprobs = last.subtract(last.logsumexp_keepdims(-1));
        let key = self.keys.next()?;
        Ok(sampler.sample_with(&logprobs, key.as_ref())?.reshape([1]))
    }

    /// Submits a single-token step after `token` (`[1]`, maybe not computed yet) and
    /// returns its token, not read.
    fn submit(&mut self, token: &Array) -> Result<Array> {
        let input = token.reshape([1, 1]);
        let next = if self.drafter.needs_hidden() {
            let (logits, hidden) = self.model.forward_hidden(&input, &mut self.cache);
            let next = self.choose(&logits)?;
            self.drafter.observe(self.model, &next, Some(&hidden))?;
            next
        } else {
            let logits = self.model.forward(&input, &mut self.cache);
            let next = self.choose(&logits)?;
            self.drafter.observe(self.model, &next, None)?;
            next
        };
        async_eval(std::slice::from_ref(&next))?;
        self.stats.steps += 1;
        Ok(next)
    }

    /// After a token generated without a draft: counts down a pause, or asks whether a
    /// draft is due.
    fn after_plain_token(&mut self) -> Result<()> {
        self.probed = None;
        if self.pause.0 > 0 {
            self.pause.0 -= 1;
            self.draft_due = false;
        } else if self.drafter.needs_hidden() {
            // Model drafters always have a draft.
            self.draft_due = true;
        } else {
            let d = self
                .drafter
                .draft(self.model, &self.context, self.draft_len())?;
            self.draft_due = !d.is_empty();
            self.probed = self.draft_due.then_some(d);
        }
        Ok(())
    }

    /// One verification step: the last token and the draft through the model; the
    /// accepted draft tokens and the model's next token become pending.
    fn verify(&mut self, draft: Draft) -> Result<()> {
        let last = *self.context.last().expect("never empty");
        let k = draft.len();
        let input = ops::concatenate(&[Array::from_slice(&[last], [1]), draft.tokens.clone()], 0)
            .reshape([1, k as i32 + 1]);
        for cache in &mut self.cache {
            cache.keep_steps(true);
        }
        let (logits, hidden) = if self.drafter.needs_hidden() {
            let (logits, hidden) = self.model.forward_hidden(&input, &mut self.cache);
            (logits, Some(hidden))
        } else {
            (self.model.forward(&input, &mut self.cache), None)
        };
        let rows = logits.reshape([k as i32 + 1, -1]);
        let none = Array::from_slice(&[0u32; 0], [0]);
        let sampler = self.options.sampler;
        let decision = match sampler.temperature {
            // The draft tokens equal to the argmax at their position, then the argmax.
            0.0 => nuvec::speculative::accept_greedy(&rows, &draft.tokens, &none),
            _ => {
                let logprobs = rows.subtract(rows.logsumexp_keepdims(-1));
                let probs = sampler.probabilities(&logprobs)?;
                let key = match self.keys.next()? {
                    Some(k) => k,
                    None => nuvec::random::next_key(),
                };
                nuvec::speculative::accept_sampled(&probs, &draft.tokens, None, &none, &key)
            }
        };
        // One host read: the decision and the draft.
        eval(&[&decision, &draft.tokens])?;
        let (accepted, next, _) = nuvec::speculative::read(&decision)?;
        let draft_v = draft.tokens.to_vec::<u32>()?;
        for cache in &mut self.cache {
            cache.trim(k - accepted)?;
            cache.keep_steps(false);
        }
        let mut new: Vec<u32> = draft_v[..accepted].to_vec();
        new.push(next);
        let new = &new[..];
        let hidden = hidden.map(|h| h.i((.., 0..accepted as i32 + 1, ..)));
        let tokens = Array::from_slice(new, [new.len() as i32]);
        self.drafter.observe(self.model, &tokens, hidden.as_ref())?;
        self.stats.steps += 1;
        self.stats.verifies += 1;
        self.stats.drafted += k;
        self.stats.accepted += accepted;
        match self.policy {
            DraftPolicy::Adaptive(_) => {
                self.pause = match accepted {
                    0 => (self.pause.1, (self.pause.1 * 2).min(16)),
                    _ => (0, 1),
                };
            }
            DraftPolicy::Auto => {
                if self.switch.record(accepted, self.auto.break_even) {
                    self.pause.0 = PAUSE;
                    self.stats.pauses += 1;
                }
            }
            DraftPolicy::Fixed(_) => {}
        }
        self.draft_due = self.pause.0 == 0;
        self.context.extend(new);
        self.pending.extend(new);
        Ok(())
    }

    /// The next token: pending from a verification, read from the pipeline, or from a new
    /// step.
    fn advance(&mut self) -> Result<u32> {
        if let Some(t) = self.pending.pop_front() {
            return Ok(t);
        }
        let more = self.produced + 1 < self.options.max_tokens;
        if let Some(y) = self.inflight.take() {
            // Keep the pipeline going (submit the following step, then read this token)
            // unless a draft is due.
            if !self.draft_due && more {
                self.inflight = Some(self.submit(&y)?);
            }
            let t = y.item::<u32>()?;
            self.context.push(t);
            self.after_plain_token()?;
            return Ok(t);
        }
        let left = self.options.max_tokens - self.produced;
        let max = self
            .draft_len()
            .min(verify_rows(self.model) - 1)
            .min(left.saturating_sub(1));
        let probed = self.probed.take();
        let draft = match self.pause.0 {
            0 if max > 0 => {
                let d = match probed {
                    Some(d) => d,
                    None => self.drafter.draft(self.model, &self.context, max)?,
                };
                match d.len() > max {
                    true => Draft::new(d.tokens.i(0..max as i32)),
                    false => d,
                }
            }
            _ => Draft::empty(),
        };
        if draft.is_empty() {
            let last = Array::from_slice(&[*self.context.last().expect("never empty")], [1]);
            self.inflight = Some(self.submit(&last)?);
            self.draft_due = false;
            return self.advance();
        }
        self.verify(draft)?;
        Ok(self.pending.pop_front().expect("verification adds a token"))
    }
}

impl<D: Drafter> Drop for SpeculativeGeneration<'_, D> {
    /// Waits for the submitted step before its memory is unwired (errors were or would
    /// have been reported by `next`), as [`crate::Generation`] does.
    fn drop(&mut self) {
        if let (true, Some(y)) = (self._wired.active, self.inflight.take()) {
            let _ = nuvec::with_row_exact(|| eval(std::slice::from_ref(&y)));
        }
    }
}

impl<D: Drafter> Iterator for SpeculativeGeneration<'_, D> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Result<u32>> {
        if self.produced >= self.options.max_tokens {
            return None;
        }
        // Verification rows keep the bits of one-row decode steps (the prompt is prefilled
        // as `generate` prefills it), so the tokens are plain decoding's (D-103).
        let token = nuvec::with_row_exact(|| self.advance());
        self.produced = match token {
            Ok(_) => self.produced + 1,
            Err(_) => self.options.max_tokens,
        };
        // An end-of-sequence token ends the generation (not yielded), as in `generate`.
        if let Ok(t) = token
            && self.model.config.eos_token_ids.contains(&t)
        {
            self.produced = self.options.max_tokens;
            return None;
        }
        Some(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kill_switch_pauses_when_drafts_stop_paying() {
        let mut k = KillSwitch::default();
        // Fifteen verifications decide nothing; the sixteenth compares the mean.
        for _ in 0..15 {
            assert!(!k.record(0, 0.5));
        }
        assert!(k.record(0, 0.5));
        // A new window after a pause; a mean at or above the break-even keeps drafting.
        for i in 0..32 {
            assert!(!k.record(i % 2, 0.5), "{i}");
        }
        // Acceptance falling off: a pause once the window's mean drops below.
        let paused = (0..32).position(|_| k.record(0, 0.5)).unwrap();
        assert!(paused < 16, "{paused}");
    }

    #[test]
    fn prompt_lookup_drafts_what_followed_the_suffix() {
        // By default, a 3-token suffix: "1 2 3 … 1 2 3" drafts "4 5 9".
        let d = PromptLookup::default();
        assert_eq!(d.lookup(&[1, 2, 3, 4, 5, 9, 7, 1, 2, 3], 3), [4, 5, 9]);
        assert!(d.lookup(&[1, 2, 3, 4, 5, 9, 7, 2, 3], 3).is_empty());
        // Shorter suffixes when allowed, the longest first: "… 2 3" drafts "4 5 9".
        let short = PromptLookup {
            max_ngram: 3,
            min_ngram: 2,
        };
        let context = [1, 2, 3, 4, 5, 9, 7, 2, 3];
        assert_eq!(short.lookup(&context, 3), [4, 5, 9]);
        assert_eq!(short.lookup(&context, 1), [4]);
        // No earlier occurrence of a long enough suffix: nothing.
        assert!(short.lookup(&[1, 2, 3], 4).is_empty());
        assert!(short.lookup(&[5, 1, 5, 2, 5], 1).is_empty());
        // The most recent occurrence wins.
        let single = PromptLookup {
            max_ngram: 3,
            min_ngram: 1,
        };
        assert_eq!(single.lookup(&[5, 1, 5, 2, 5], 1), [2]);
    }
}
