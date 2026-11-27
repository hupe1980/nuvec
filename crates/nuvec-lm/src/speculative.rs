//! Speculative decoding: a drafter proposes the next tokens, one forward pass verifies them
//! all, and the longest prefix the model agrees with is kept, plus the model's own next
//! token. Verifying several tokens in one pass costs much less than generating them one by
//! one, so every accepted draft token saves most of a step. A block drafter's candidates
//! can be verified as a draft tree (DDTree): each position's top tokens, the most likely
//! prefixes in one pass with an ancestor mask.
//!
//! Verification routes products as MLX's, so greedy output can differ from plain greedy
//! decoding's in low bits (and, at a near tie, in a token), as mlx-lm's. With
//! [`GenerateOptions::row_exact`] it is plain greedy decoding's, bit for bit: a verify
//! pass computes each row as serial decode would (nuvec's row independence, up to
//! [`nuvec::ROW_INDEPENDENT_ROWS`] rows: [`nuvec::with_row_exact`]; and its attention
//! planner's serial rule), and the verify width is capped where that holds. Speculation
//! is batch 1 on the contiguous cache; the [`Batcher`](crate::Batcher) and paged caches
//! decode without it.
//!
//! Drafters: [`DFlashDrafter`](crate::DFlashDrafter) (DFlash and DSpark block drafters over
//! the model's hidden states), [`PromptLookup`] (n-grams of the context, no model),
//! [`MtpDrafter`] (the model's own multi-token-prediction head, Qwen 3.5–3.8) and
//! [`SuffixDecoding`] (a suffix index over the prompt and earlier outputs).

use nuvec::{Array, Result, async_eval, eval, ops};

use crate::generate::{FinishReason, GenerateOptions, Keys, WiredLimit, judge};
use crate::model::{KvCache, Model};
use crate::stop::StopState;

pub use crate::suffix::SuffixDecoding;
pub use tree::Candidates;
pub(crate) use tree::top_k;

mod modes;
mod tree;

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

    fn checked(self) -> Result<Self> {
        let tokens = self.tokens.check()?;
        if tokens.ndim() != 1 || tokens.dtype() != nuvec::DType::U32 {
            return Err(nuvec::Error::InvalidArgument(
                "draft tokens must be a uint32 vector".into(),
            ));
        }
        Ok(Self { tokens })
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

    /// With [`Drafter::needs_hidden`]: the decoder layers whose outputs `observe` gets,
    /// concatenated (`[1, n, layers · hidden_size]`, [`Model::forward_layers`]), instead of
    /// the final hidden state (DFlash drafters).
    fn hidden_layers(&self) -> Option<Vec<usize>> {
        None
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

    /// The best `k` tokens at each of `depth` positions after `context`, for a draft tree
    /// ([`DraftPolicy::Auto`], [`DraftPolicy::Tree`]); `None` (the default) drafts a chain.
    fn candidates(
        &mut self,
        model: &Model,
        context: &[u32],
        depth: usize,
        k: usize,
    ) -> Result<Option<Candidates>> {
        let _ = (model, context, depth, k);
        Ok(None)
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

    fn candidates(
        &mut self,
        model: &Model,
        context: &[u32],
        depth: usize,
        k: usize,
    ) -> Result<Option<Candidates>> {
        (**self).candidates(model, context, depth, k)
    }

    fn observe(&mut self, model: &Model, tokens: &Array, hidden: Option<&Array>) -> Result<()> {
        (**self).observe(model, tokens, hidden)
    }

    fn needs_hidden(&self) -> bool {
        (**self).needs_hidden()
    }

    fn hidden_layers(&self) -> Option<Vec<usize>> {
        (**self).hidden_layers()
    }

    fn auto(&self, model: &Model) -> Auto {
        (**self).auto(model)
    }
}

impl<D: Drafter + ?Sized> Drafter for Box<D> {
    fn draft(&mut self, model: &Model, context: &[u32], max: usize) -> Result<Draft> {
        (**self).draft(model, context, max)
    }

    fn candidates(
        &mut self,
        model: &Model,
        context: &[u32],
        depth: usize,
        k: usize,
    ) -> Result<Option<Candidates>> {
        (**self).candidates(model, context, depth, k)
    }

    fn observe(&mut self, model: &Model, tokens: &Array, hidden: Option<&Array>) -> Result<()> {
        (**self).observe(model, tokens, hidden)
    }

    fn needs_hidden(&self) -> bool {
        (**self).needs_hidden()
    }

    fn hidden_layers(&self) -> Option<Vec<usize>> {
        (**self).hidden_layers()
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
    /// The drafter's measured length for the model ([`Drafter::auto`]), never slower
    /// than plain decoding by much:
    ///
    /// - Model drafters ([`Drafter::needs_hidden`]: DFlash, MTP) time plain decoding,
    ///   chains and, where they apply, draft trees, and keep the fastest (probing the
    ///   others now and then, less often the more they lose). Trees apply with a drafter
    ///   that gives [candidates](Drafter::candidates), an attention-only model and without
    ///   [`GenerateOptions::row_exact`]: the most likely prefixes of the drafter's
    ///   positions, as many as maximise the expected tokens per measured time.
    /// - Other drafters use a kill switch: when the mean of accepted tokens over the last
    ///   16 verifications falls below the drafter's break-even, the next 64 tokens are
    ///   decoded plainly (pipelined), then drafting resumes.
    #[default]
    Auto,
    /// A draft tree of up to this many nodes every step, where [`DraftPolicy::Auto`] would
    /// draft a tree; a chain of this many otherwise (A/B measurements).
    Tree(usize),
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

/// A linear draft and the decision taken at its first differing token.
#[derive(Debug)]
#[non_exhaustive]
pub struct VerifyRound {
    /// Zero-based output index of the first proposed token, excluding the prompt.
    pub start: usize,
    /// Proposed token IDs, in order, excluding the input seed.
    pub draft: Vec<u32>,
    /// Number of accepted draft tokens. The target emits these and then `next`.
    pub accepted: usize,
    /// Token chosen by the target after the accepted prefix.
    pub next: u32,
    /// Elapsed time from the fenced draft to the evaluated target decision.
    pub target_time: Option<std::time::Duration>,
}

/// Counts of a speculative generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpeculativeStats {
    /// Forward passes after the prompt (plain steps and verifications).
    pub steps: usize,
    /// Verification passes (steps with a draft).
    pub verifies: usize,
    /// Largest linear verification, including its seed token.
    pub max_verify_rows: usize,
    /// Time spent producing linear drafts with stage timing enabled.
    pub draft_time: std::time::Duration,
    /// Time spent verifying linear drafts with stage timing enabled.
    pub verify_time: std::time::Duration,
    /// Draft tokens proposed (tree nodes for draft trees).
    pub drafted: usize,
    /// Draft tokens accepted.
    pub accepted: usize,
    /// Times [`DraftPolicy::Auto`]'s kill switch paused drafting (drafters without
    /// hidden states).
    pub pauses: usize,
}

/// The logits of `tokens` and the hidden states `drafter` observes: its layers' outputs
/// ([`Drafter::hidden_layers`]) or the final hidden state.
fn forward_for<D: Drafter + ?Sized>(
    model: &Model,
    drafter: &D,
    tokens: &Array,
    cache: &mut [KvCache],
) -> (Array, Array) {
    match drafter.hidden_layers() {
        Some(ids) => model.forward_layers(tokens, cache, &ids),
        None => model.forward_hidden(tokens, cache),
    }
}

/// The most rows a verification has. Row-exact ([`GenerateOptions::row_exact`]), each row
/// is computed as serial decode computes it: nuvec's row bound and, for head sizes 192
/// and 256 (whose float32 rows need the one-key-per-step kernel, and compose beyond one
/// vector threadgroup), the rows of one threadgroup (32 SIMD-groups over the grouped
/// query heads). Otherwise 16 (a DFlash block; MLX's few-row quantized kernel).
pub fn verify_rows(model: &Model, row_exact: bool) -> usize {
    if !row_exact {
        return 16;
    }
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
    time_stages: bool,
    rounds: Option<Vec<VerifyRound>>,
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
    /// The prompt's length (`context` starts with it).
    prompt: usize,
    /// Measured verification times of draft trees per row count.
    costs: tree::Costs,
    /// [`DraftPolicy::Auto`]'s timed choice for model drafters: plain, chains or trees.
    modes: modes::Modes,
    /// Moving averages of a tree's accepted tokens and of its expected ones (the drafter's
    /// probabilities), whose ratio calibrates the expectation.
    calibration: (f64, f64),
    stops: StopState,
    finished: Option<FinishReason>,
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
/// Recurrent and convolution layers (Gated DeltaNet, Mamba-2, short convolutions) keep every
/// position's state while a draft is verified, and roll back to the last accepted one
/// ([`KvCache::keep_steps`]).
///
/// Greedy (temperature 0) output is plain greedy decoding's, bit for bit, with
/// [`GenerateOptions::row_exact`] (see the [module documentation](self)). With a
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
    if model.stage().is_some() {
        return invalid("not supported on a pipelined model");
    }
    options.validate()?;
    let auto = drafter.auto(model);
    let wired = WiredLimit::new();
    let mut cache = model.new_cache_for(&options)?;
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
            let (_, hidden) = forward_for(model, &drafter, &chunk, &mut cache);
            let next = Array::from_slice(&prompt[start + 1..start + n + 1], [n as i32]);
            drafter.observe(model, &next, Some(&hidden))?;
            states.push(hidden);
        } else {
            let _ = model.forward(&chunk, &mut cache).check()?;
        }
        states.extend(cache.iter().flat_map(KvCache::state));
        eval(&states)?;
        nuvec::memory::clear_cache();
        start += n;
    }
    let keys = Keys::new(options.seed);
    let modes = modes::Modes::new(
        policy == DraftPolicy::Auto && !options.row_exact && model.attention_only(),
    );
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
        time_stages: false,
        rounds: None,
        pause: (0, 1),
        inflight: None,
        draft_due: false,
        probed: None,
        keys,
        prompt: prompt.len(),
        costs: tree::Costs::default(),
        modes,
        calibration: (1.0, 1.0),
        stops: StopState::default(),
        finished: None,
        _wired: wired,
    })
}

impl<D: Drafter> SpeculativeGeneration<'_, D> {
    /// The draft length the policy asks for.
    fn draft_len(&self) -> usize {
        match self.policy {
            DraftPolicy::Auto => self.auto.len,
            DraftPolicy::Fixed(n) | DraftPolicy::Adaptive(n) | DraftPolicy::Tree(n) => n,
        }
    }

    /// Runs `f` row-exact when the options ask for it ([`GenerateOptions::row_exact`]).
    fn row_exact<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        match self.options.row_exact {
            true => nuvec::with_row_exact(|| f(self)),
            false => f(self),
        }
    }

    /// The counts so far.
    pub fn stats(&self) -> SpeculativeStats {
        self.stats
    }

    /// Record linear verification decisions. With `time_stages`, fence drafts before
    /// timing the target. Use an unfenced run to measure end-to-end throughput.
    pub fn record_rounds(&mut self, time_stages: bool) {
        self.rounds.get_or_insert_with(Vec::new);
        self.time_stages = time_stages;
    }

    /// Drain the decisions recorded since the previous call. Periodically drain long
    /// generations to bound the memory used for tracing.
    pub fn take_rounds(&mut self) -> Vec<VerifyRound> {
        self.rounds.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Why the generation ended, once it has (the iterator returned its last item).
    pub fn finish_reason(&self) -> Option<&FinishReason> {
        self.finished.as_ref()
    }

    /// The tokens generated so far (pending ones included).
    fn generated(&self) -> usize {
        self.context.len() - self.prompt
    }

    /// The token chosen from `logits` `[1, 1, vocab]` as the `n`-th generated: the
    /// argmax, or a sample.
    fn choose(&mut self, logits: &Array, n: usize) -> Result<Array> {
        let sampler = self.options.sampler;
        let eos = &self.model.config.eos_token_ids;
        let last = self.options.mask(logits.reshape([1, -1]), eos, n);
        if sampler.temperature == 0.0 {
            return Ok(last.argmax(-1).reshape([1]));
        }
        let logprobs = last.subtract(last.logsumexp_keepdims(-1));
        let key = self.keys.next()?;
        Ok(sampler.sample_with(&logprobs, key.as_ref())?.reshape([1]))
    }

    /// Submits a single-token step after `token` (`[1]`, maybe not computed yet) and
    /// returns its token, the `n`-th generated, not read.
    fn submit(&mut self, token: &Array, n: usize) -> Result<Array> {
        let input = token.reshape([1, 1]);
        let next = if self.drafter.needs_hidden() {
            let (logits, hidden) = forward_for(self.model, &self.drafter, &input, &mut self.cache);
            let next = self.choose(&logits, n)?;
            self.drafter.observe(self.model, &next, Some(&hidden))?;
            next
        } else {
            let logits = self.model.forward(&input, &mut self.cache);
            let next = self.choose(&logits, n)?;
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
            // Model drafters always have a draft, unless plain decoding is timed faster.
            self.draft_due = !(self.timed() && self.modes.mode() == modes::Mode::Plain);
        } else {
            let d = self
                .drafter
                .draft(self.model, &self.context, self.draft_len())?
                .checked()?;
            self.draft_due = !d.is_empty();
            self.probed = self.draft_due.then_some(d);
        }
        Ok(())
    }

    /// One verification step: the last token and the draft through the model; the
    /// accepted draft tokens and the model's next token become pending.
    fn verify(&mut self, draft: Draft, start: Option<std::time::Instant>) -> Result<()> {
        let last = *self.context.last().expect("never empty");
        let k = draft.len();
        let input = ops::concatenate(&[Array::from_slice(&[last], [1]), draft.tokens.clone()], 0)
            .reshape([1, k as i32 + 1]);
        for cache in &mut self.cache {
            cache.keep_steps(true);
        }
        let (logits, hidden) = if self.drafter.needs_hidden() {
            let (logits, hidden) = forward_for(self.model, &self.drafter, &input, &mut self.cache);
            (logits, Some(hidden))
        } else {
            (self.model.forward(&input, &mut self.cache), None)
        };
        let rows = logits.reshape([k as i32 + 1, -1]);
        // Row r chooses the token after the last and r draft tokens.
        let first = self.generated() + 1;
        let rows = match first <= self.options.min_tokens {
            false => rows,
            true => {
                let eos = &self.model.config.eos_token_ids;
                let masked: Vec<Array> = (0..=k as i32)
                    .map(|r| self.options.mask(rows.i(r..r + 1), eos, first + r as usize))
                    .collect();
                ops::concatenate(&masked, 0)
            }
        };
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
        let target_time = start.map(|t| t.elapsed());
        if let Some(time) = target_time {
            self.stats.verify_time += time;
        }
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
        self.stats.max_verify_rows = self.stats.max_verify_rows.max(k + 1);
        self.stats.drafted += k;
        self.stats.accepted += accepted;
        match self.policy {
            DraftPolicy::Adaptive(_) => {
                self.pause = match accepted {
                    0 => (self.pause.1, (self.pause.1 * 2).min(16)),
                    _ => (0, 1),
                };
            }
            DraftPolicy::Auto if !self.timed() => {
                if self.switch.record(accepted, self.auto.break_even) {
                    self.pause.0 = PAUSE;
                    self.stats.pauses += 1;
                }
            }
            DraftPolicy::Auto => {}
            DraftPolicy::Fixed(_) | DraftPolicy::Tree(_) => {}
        }
        self.draft_due = self.pause.0 == 0;
        if let Some(rounds) = &mut self.rounds {
            rounds.push(VerifyRound {
                start: first - 1,
                draft: draft_v,
                accepted,
                next,
                target_time,
            });
        }
        self.context.extend(new);
        self.pending.extend(new);
        Ok(())
    }

    /// Whether [`DraftPolicy::Auto`] times plain decoding, chains and trees (model drafters)
    /// instead of its kill switch.
    fn timed(&self) -> bool {
        self.policy == DraftPolicy::Auto && self.drafter.needs_hidden()
    }

    /// Whether this step drafts a tree: by policy, without `row_exact` (masked keys change
    /// the attention's reduction order), on attention-only models.
    fn tree_due(&self) -> bool {
        matches!(self.policy, DraftPolicy::Auto | DraftPolicy::Tree(_))
            && !self.options.row_exact
            && self.model.attention_only()
    }

    /// The candidates for a tree of at most `nodes` nodes `depth` deep, if the drafter
    /// gives them.
    fn tree_draft(&mut self, depth: usize, nodes: usize) -> Result<Option<Vec<tree::Node>>> {
        let k = nodes.min(16);
        let start = std::time::Instant::now();
        let Some(c) = self
            .drafter
            .candidates(self.model, &self.context, depth, k)?
        else {
            return Ok(None);
        };
        let [d, k] = c.tokens.shape_n::<2>()?;
        eval(&[&c.tokens, &c.logprobs])?;
        self.costs.record_draft(start.elapsed().as_secs_f64());
        let tokens = c.tokens.to_vec::<u32>()?;
        let logprobs = c.logprobs.astype(nuvec::DType::F32).to_vec::<f32>()?;
        let mut nodes = tree::build(&tokens, &logprobs, d as usize, k as usize, nodes);
        if self.policy == DraftPolicy::Auto {
            let calibration = (self.calibration.0 / self.calibration.1).clamp(0.25, 2.0);
            nodes.truncate(tree::size(&nodes, calibration, &self.costs));
        }
        Ok(Some(nodes))
    }

    /// Each tree row's token from `logits` (`[1, L, vocab]`), row `r` the
    /// `first + depth[r]`-th generated: the argmax, or a sample of the filtered distribution
    /// (lossless: every emitted token is a sample given its path).
    fn tree_choices(&mut self, logits: &Array, depth: &[i32]) -> Result<Array> {
        let l = depth.len();
        let rows = logits.reshape([l as i32, -1]);
        let first = self.generated() + 1;
        let rows = match first <= self.options.min_tokens {
            false => rows,
            true => {
                let eos = &self.model.config.eos_token_ids;
                let masked: Vec<Array> = (0..l)
                    .map(|r| {
                        let n = first + depth[r] as usize;
                        self.options.mask(rows.i(r as i32..r as i32 + 1), eos, n)
                    })
                    .collect();
                ops::concatenate(&masked, 0)
            }
        };
        let sampler = self.options.sampler;
        let chosen = match sampler.temperature {
            0.0 => rows.argmax(-1),
            _ => {
                let logprobs = rows.subtract(rows.logsumexp_keepdims(-1));
                let key = self.keys.next()?;
                sampler.sample_with(&logprobs, key.as_ref())?
            }
        };
        Ok(chosen.astype(nuvec::DType::U32))
    }

    /// One verification of a draft tree: the last token and the nodes through the model with
    /// the tree's positions and mask; the accepted path and the model's next token become
    /// pending, and the caches keep the path.
    fn verify_tree(&mut self, nodes: &[tree::Node]) -> Result<()> {
        let last = *self.context.last().expect("never empty");
        let l = nodes.len() + 1;
        let mut tokens = vec![last];
        tokens.extend(nodes.iter().map(|n| n.token));
        let mut depth = vec![0i32];
        depth.extend(nodes.iter().map(|n| n.depth as i32));
        // Row i: itself and its ancestors (node j is row j).
        let mut ancestors = vec![false; l * l];
        for i in 0..l {
            let mut a = i;
            loop {
                ancestors[i * l + a] = true;
                if a == 0 {
                    break;
                }
                a = nodes[a - 1].parent;
            }
        }
        let base = self.cache.first().map_or(0, KvCache::offset);
        let mask = crate::model::TreeMask::new(
            Array::from_slice(&depth, [l as i32]),
            Array::from_slice(&ancestors, [l as i32, l as i32]),
            base,
        );
        for cache in &mut self.cache {
            cache.set_tree(Some(mask.clone()));
        }
        let start = std::time::Instant::now();
        let input = Array::from_slice(&tokens, [1, l as i32]);
        let (logits, hidden) = forward_for(self.model, &self.drafter, &input, &mut self.cache);
        for cache in &mut self.cache {
            cache.set_tree(None);
        }
        let chosen = self.tree_choices(&logits, &depth)?;
        drop(logits);
        // The hidden states with the choices, so that no graph holds the caches when the
        // path is written into them (in place).
        eval(&[&chosen, &hidden])?;
        self.costs.record(l, start.elapsed().as_secs_f64());
        let chosen = chosen.to_vec::<u32>()?;
        // The walk: from the root, into the child holding the model's token.
        let mut path = vec![0usize];
        loop {
            let at = *path.last().expect("the root");
            let next = (1..l).find(|&j| nodes[j - 1].parent == at && tokens[j] == chosen[at]);
            match next {
                Some(j) => path.push(j),
                None => break,
            }
        }
        let accepted = path.len() - 1;
        let next = chosen[*path.last().expect("the root")];
        let index = Array::from_slice(
            &path.iter().map(|&j| j as i32).collect::<Vec<_>>(),
            [path.len() as i32],
        );
        let hidden = ops::take(&hidden, &index, 1);
        let rows = self
            .cache
            .iter()
            .map(|c| c.path_rows(base, &path))
            .collect::<Result<Vec<_>>>()?;
        eval(&rows.iter().flatten().chain([&hidden]).collect::<Vec<_>>())?;
        for (cache, rows) in self.cache.iter_mut().zip(rows) {
            cache.keep_path(base, &path, rows);
        }
        let mut new: Vec<u32> = path[1..].iter().map(|&j| tokens[j]).collect();
        new.push(next);
        let observed = Array::from_slice(&new, [new.len() as i32]);
        self.drafter.observe(self.model, &observed, Some(&hidden))?;
        let expected: f64 = nodes.iter().map(|n| f64::from(n.score).exp()).sum();
        let ema = |old: f64, new: f64| 0.9 * old + 0.1 * new;
        self.calibration = (
            ema(self.calibration.0, accepted as f64),
            ema(self.calibration.1, expected.max(1e-3)),
        );
        self.stats.steps += 1;
        self.stats.verifies += 1;
        self.stats.drafted += nodes.len();
        self.stats.accepted += accepted;
        if self.policy == DraftPolicy::Auto
            && !self.timed()
            && self.switch.record(accepted, self.auto.break_even)
        {
            self.pause.0 = PAUSE;
            self.stats.pauses += 1;
        }
        self.draft_due = self.pause.0 == 0;
        self.context.extend(&new);
        self.pending.extend(&new);
        Ok(())
    }

    /// A step without a draft (pipelined), then its token.
    fn plain_step(&mut self) -> Result<u32> {
        let last = Array::from_slice(&[*self.context.last().expect("never empty")], [1]);
        let n = self.generated() + 1;
        self.inflight = Some(self.submit(&last, n)?);
        self.draft_due = false;
        self.advance()
    }

    /// The next token: pending from a verification, read from the pipeline, or from a new
    /// step.
    fn advance(&mut self) -> Result<u32> {
        if let Some(t) = self.pending.pop_front() {
            return Ok(t);
        }
        let more = self.produced + 1 < self.options.max_tokens;
        if let Some(y) = self.inflight.take() {
            let start = std::time::Instant::now();
            // Keep the pipeline going (submit the following step, then read this token)
            // unless a draft is due.
            if !self.draft_due && more {
                // `y` is the next generated token; the step after it samples the one after.
                let n = self.generated() + 2;
                self.inflight = Some(self.submit(&y, n)?);
            }
            let t = y.item::<u32>()?;
            self.context.push(t);
            if self.timed() && self.modes.mode() == modes::Mode::Plain {
                self.modes.record(1, start.elapsed().as_secs_f64());
            }
            self.after_plain_token()?;
            return Ok(t);
        }
        let left = self.options.max_tokens - self.produced;
        let max = self
            .draft_len()
            .min(verify_rows(self.model, self.options.row_exact) - 1)
            .min(left.saturating_sub(1));
        let probed = self.probed.take();
        let start = std::time::Instant::now();
        let timed = self.pause.0 == 0 && max > 0 && probed.is_none() && self.timed();
        if timed && self.modes.mode() == modes::Mode::Plain {
            return self.plain_step();
        }
        let tree_due = match timed {
            true => self.modes.mode() == modes::Mode::Tree,
            false => self.tree_due(),
        };
        if self.pause.0 == 0 && max > 0 && probed.is_none() && tree_due {
            // Depth is capped by the tokens left (`max`); a node count is not.
            let nodes = verify_rows(self.model, false) - 1;
            let depth = match self.policy {
                DraftPolicy::Tree(_) => max,
                _ => self.auto.len.min(max).max(1),
            };
            let nodes = match self.policy {
                DraftPolicy::Tree(n) => n.min(nodes),
                _ => nodes,
            };
            match self.tree_draft(depth, nodes)? {
                Some(tree) if tree.is_empty() => return self.plain_step(),
                Some(tree) => {
                    self.verify_tree(&tree)?;
                    if timed {
                        self.modes
                            .record(self.pending.len(), start.elapsed().as_secs_f64());
                    }
                    return Ok(self.pending.pop_front().expect("verification adds a token"));
                }
                None if timed => self.modes.without_trees(),
                None => {}
            }
        }
        let draft = match self.pause.0 {
            0 if max > 0 => {
                let d = match probed {
                    Some(d) => d,
                    None => self
                        .drafter
                        .draft(self.model, &self.context, max)?
                        .checked()?,
                };
                match d.len() > max {
                    true => Draft::new(d.tokens.i(0..max as i32)),
                    false => d,
                }
            }
            _ => Draft::empty(),
        };
        if draft.is_empty() {
            return self.plain_step();
        }
        let verify_start = if self.time_stages {
            eval(&[&draft.tokens])?;
            self.stats.draft_time += start.elapsed();
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.verify(draft, verify_start)?;
        if timed {
            self.modes
                .record(self.pending.len(), start.elapsed().as_secs_f64());
        }
        Ok(self.pending.pop_front().expect("verification adds a token"))
    }
}

impl<D: Drafter> Drop for SpeculativeGeneration<'_, D> {
    /// Waits for the submitted step before its memory is unwired (errors were or would
    /// have been reported by `next`), as [`crate::Generation`] does.
    fn drop(&mut self) {
        if let (true, Some(y)) = (self._wired.active, self.inflight.take()) {
            let _ = self.row_exact(|_| eval(std::slice::from_ref(&y)));
        }
    }
}

impl<D: Drafter> Iterator for SpeculativeGeneration<'_, D> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Result<u32>> {
        if self.finished.is_some() {
            return None;
        }
        if self.produced >= self.options.max_tokens {
            self.finished = Some(FinishReason::Length);
            return None;
        }
        // Row-exact, verification rows keep the bits of one-row decode steps (the prompt is
        // prefilled as `generate` prefills it), so the tokens are plain decoding's.
        let token = match self.row_exact(|s| s.advance()) {
            Ok(t) => t,
            Err(e) => {
                self.finished = Some(FinishReason::Error(e.clone()));
                return Some(Err(e));
            }
        };
        self.produced += 1;
        // The same end rules as `generate`; tokens verified after the end are dropped.
        let eos = &self.model.config.eos_token_ids;
        let (emit, reason) = judge(&self.options, eos, &mut self.stops, token, self.produced);
        if reason.is_some() {
            self.finished = reason;
            self.pending.clear();
        }
        emit.then_some(Ok(token))
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

#[cfg(all(test, target_os = "macos"))]
mod tree_verification {
    use std::path::PathBuf;

    use nuvec::Array;

    use crate::model::{KvCache, Model, TreeMask};

    fn gpu() -> bool {
        !cfg!(miri) && (&Array::from(1.0f32) + 1.0).eval().is_ok()
    }

    /// A fixture model and its prompt.
    fn fixture(name: &str) -> (Model, Vec<u32>) {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        let expected = nuvec::io::load_safetensors(dir.join("expected.safetensors"))
            .unwrap()
            .tensors;
        let prompt = expected["prompt"].to_vec::<u32>().unwrap();
        (crate::load_model(&dir).unwrap(), prompt)
    }

    /// Within 1e-4 of the largest logit in float32; within bfloat16's rounding otherwise
    /// (the tree's rows reduce over masked keys, in another order).
    fn close(a: &Array, b: &Array, what: &str) {
        let tol = match b.dtype() {
            nuvec::DType::F32 => 1e-4,
            _ => 2e-2,
        };
        let f32 = |x: &Array| x.astype(nuvec::DType::F32).to_vec::<f32>().unwrap();
        let (a, b) = (f32(a), f32(b));
        let top = b.iter().fold(0f32, |m, x| m.max(x.abs()));
        let diff = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(diff <= tol * top.max(1.0), "{what}: {diff} (largest {top})");
    }

    /// Each node's logits equal its path verified as a chain; after keeping a path that is
    /// not the first branch, the next step equals the path decoded as a chain.
    #[test]
    fn tree_verification_equals_chains() {
        if !gpu() {
            return;
        }
        // Root, then: 1 (a) and 2 (b) under it; 3 (c) and 4 (d) under 1; 5 (e) under 2;
        // 6 (f) under 3.
        let parents = [0usize, 0, 1, 1, 2, 3];
        let tokens = [5u32, 9, 17, 23, 31, 40];
        let path_of = |mut i: usize| {
            let mut p = vec![];
            while i > 0 {
                p.push(i);
                i = parents[i - 1];
            }
            p.push(0);
            p.reverse();
            p
        };
        // Windows of 4 (gpt-oss, Gemma 4), sinks (gpt-oss), shared keys (Gemma 4).
        for name in ["qwen3_f32", "llama_bf16", "gpt_oss_f32", "gemma4_e_f32"] {
            let quant = false;
            let (model, prompt) = fixture(name);
            let mut base = model.new_cache();
            let head = &prompt[..prompt.len() - 1];
            let root = *prompt.last().unwrap();
            let _ = model.forward(&Array::from_slice(head, [1, head.len() as i32]), &mut base);
            let all: Vec<u32> = std::iter::once(root).chain(tokens).collect();
            let l = all.len();
            let mut depth = vec![0i32];
            let mut ancestors = vec![false; l * l];
            for i in 0..l {
                if i > 0 {
                    depth.push(path_of(i).len() as i32 - 1);
                }
                for a in path_of(i) {
                    ancestors[i * l + a] = true;
                }
            }
            let mask = TreeMask::new(
                Array::from_slice(&depth, [l as i32]),
                Array::from_slice(&ancestors, [l as i32, l as i32]),
                base[0].offset(),
            );
            let mut tree = base.clone();
            for c in &mut tree {
                c.set_tree(Some(mask.clone()));
            }
            let logits = model
                .forward(&Array::from_slice(&all, [1, l as i32]), &mut tree)
                .reshape([l as i32, -1]);
            for c in &mut tree {
                c.set_tree(None);
            }
            let chain = |path: &[usize]| {
                let mut cache: Vec<KvCache> = base.clone();
                let ids: Vec<u32> = path.iter().map(|&j| all[j]).collect();
                let out =
                    model.forward(&Array::from_slice(&ids, [1, ids.len() as i32]), &mut cache);
                (out.reshape([ids.len() as i32, -1]), cache)
            };
            for i in 0..l {
                let path = path_of(i);
                let (out, _) = chain(&path);
                let last = path.len() as i32 - 1;
                close(
                    &logits.i(i as i32),
                    &out.i(last),
                    &format!("{name} {quant}: node {i}"),
                );
            }
            // Keep root → a → c → f and decode one more token.
            let path = path_of(6);
            let offset = base[0].offset();
            for c in &mut tree {
                let rows = c.path_rows(offset, &path).unwrap();
                c.keep_path(offset, &path, rows);
            }
            let (_, mut kept) = chain(&path);
            let next = Array::from_slice(&[11u32], [1, 1]);
            let a = model.forward(&next, &mut tree);
            let b = model.forward(&next, &mut kept);
            close(
                &a.reshape([-1]),
                &b.reshape([-1]),
                &format!("{name} {quant}: after"),
            );
        }
    }
}
