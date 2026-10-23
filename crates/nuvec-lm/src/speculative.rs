//! Speculative decoding: a drafter proposes the next tokens, one forward pass verifies them
//! all, and the longest prefix the model agrees with is kept, plus the model's own next
//! token. Verifying several tokens in one pass costs much less than generating them one by
//! one, so every accepted draft token saves most of a step. The output is the model's
//! greedy output (up to the rounding differences between one-row and several-row products).

use nuvec::{Array, Result, async_eval, eval};

use crate::generate::{GenerateOptions, WiredLimit, prefill};
use crate::model::{KvCache, Model};

/// Proposes up to `max` tokens to follow `context` (the prompt and the tokens so far).
pub trait Drafter {
    /// The draft (possibly empty: the step then generates one token as usual).
    fn draft(&mut self, context: &[u32], max: usize) -> Vec<u32>;
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

impl<D: Drafter + ?Sized> Drafter for Box<D> {
    fn draft(&mut self, context: &[u32], max: usize) -> Vec<u32> {
        (**self).draft(context, max)
    }
}

impl Default for PromptLookup {
    fn default() -> Self {
        Self {
            max_ngram: 3,
            min_ngram: 3,
        }
    }
}

impl Drafter for PromptLookup {
    fn draft(&mut self, context: &[u32], max: usize) -> Vec<u32> {
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

/// Counts of a speculative generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Forward passes after the prompt.
    pub steps: usize,
    /// Draft tokens proposed.
    pub drafted: usize,
    /// Draft tokens accepted.
    pub accepted: usize,
}

/// The tokens generated after a prompt with speculative decoding, one per `next`.
pub struct SpeculativeGeneration<'a, D: Drafter> {
    model: &'a Model,
    drafter: D,
    cache: Vec<KvCache>,
    options: GenerateOptions,
    draft_tokens: usize,
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
/// `drafter` proposes up to `draft_tokens` tokens per step (a longer draft is cut). Greedy
/// only (the sampler's temperature must be 0, and its settings valid). Gated DeltaNet layers keep every position's state while a draft
/// is verified, and roll back to the last accepted one ([`KvCache::keep_steps`]).
pub fn speculative_generate<'a, D: Drafter>(
    model: &'a Model,
    prompt: &[u32],
    options: GenerateOptions,
    drafter: D,
    draft_tokens: usize,
) -> Result<SpeculativeGeneration<'a, D>> {
    let invalid = |m: &str| Err(nuvec::Error::InvalidArgument(format!("speculative: {m}")));
    if prompt.is_empty() {
        return invalid("empty prompt");
    }
    options.sampler.validate()?;
    if options.sampler.temperature != 0.0 {
        return invalid("only greedy decoding (temperature 0) is supported");
    }
    let wired = WiredLimit::new();
    let mut cache = model.new_cache();
    prefill(
        model,
        &mut cache,
        &prompt[..prompt.len() - 1],
        options.prefill_step,
    )?;
    Ok(SpeculativeGeneration {
        model,
        drafter,
        cache,
        options,
        draft_tokens,
        context: prompt.to_vec(),
        pending: Default::default(),
        produced: 0,
        stats: SpeculativeStats::default(),
        pause: (0, 1),
        inflight: None,
        draft_due: false,
        _wired: wired,
    })
}

impl<D: Drafter> SpeculativeGeneration<'_, D> {
    /// The counts so far.
    pub fn stats(&self) -> SpeculativeStats {
        self.stats
    }

    /// Submits a single-token step after `token` (`[1]`, maybe not computed yet) and
    /// returns its token, not read.
    fn submit(&mut self, token: &Array) -> Result<Array> {
        let logits = self.model.forward(&token.reshape([1, 1]), &mut self.cache);
        let next = logits.argmax(-1).reshape([1]);
        async_eval(std::slice::from_ref(&next))?;
        self.stats.steps += 1;
        Ok(next)
    }

    /// After a token generated without a draft: counts down a pause, or asks the drafter
    /// whether it has a draft now.
    fn after_plain_token(&mut self) {
        if self.pause.0 > 0 {
            self.pause.0 -= 1;
            self.draft_due = false;
        } else {
            self.draft_due = !self.drafter.draft(&self.context, 1).is_empty();
        }
    }

    /// One verification step: the last token and `draft` through the model; the accepted
    /// draft tokens and the model's next token become pending.
    fn verify(&mut self, draft: Vec<u32>) -> Result<()> {
        let last = *self.context.last().expect("never empty");
        let mut input = Vec::with_capacity(draft.len() + 1);
        input.push(last);
        input.extend(&draft);
        let l = input.len();
        for cache in &mut self.cache {
            cache.keep_steps(true);
        }
        let input = Array::from_slice(&input, [1, l]);
        let choice = self.model.forward(&input, &mut self.cache).argmax(-1);
        let choice = choice.to_vec::<u32>()?;
        // The model's tokens after each input position: the drafts it agrees with, then its
        // own next token.
        let accepted = draft
            .iter()
            .zip(&choice)
            .take_while(|(d, c)| d == c)
            .count();
        for cache in &mut self.cache {
            cache.trim(l - 1 - accepted)?;
            cache.keep_steps(false);
        }
        self.stats.steps += 1;
        self.stats.drafted += draft.len();
        self.stats.accepted += accepted;
        // A rejected draft costs a wider step for nothing: after one, drafting pauses for
        // 1, 2, 4, … (at most 16) steps, until a draft is accepted again.
        self.pause = match accepted {
            0 => (self.pause.1, (self.pause.1 * 2).min(16)),
            _ => (0, 1),
        };
        self.draft_due = self.pause.0 == 0;
        let new = &choice[..=accepted];
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
            self.after_plain_token();
            return Ok(t);
        }
        let left = self.options.max_tokens - self.produced;
        let draft = match self.pause.0 {
            0 => {
                let max = self.draft_tokens.min(left.saturating_sub(1));
                let mut draft = self.drafter.draft(&self.context, max);
                // A drafter may propose more than asked: the rest is not verified.
                draft.truncate(max);
                draft
            }
            _ => Vec::new(),
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
            let _ = eval(std::slice::from_ref(&y));
        }
    }
}

impl<D: Drafter> Iterator for SpeculativeGeneration<'_, D> {
    type Item = Result<u32>;

    fn next(&mut self) -> Option<Result<u32>> {
        if self.produced >= self.options.max_tokens {
            return None;
        }
        let token = self.advance();
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
    fn prompt_lookup_drafts_what_followed_the_suffix() {
        // By default, a 3-token suffix: "1 2 3 … 1 2 3" drafts "4 5 9".
        let mut d = PromptLookup::default();
        assert_eq!(d.draft(&[1, 2, 3, 4, 5, 9, 7, 1, 2, 3], 3), [4, 5, 9]);
        assert!(d.draft(&[1, 2, 3, 4, 5, 9, 7, 2, 3], 3).is_empty());
        // Shorter suffixes when allowed, the longest first: "… 2 3" drafts "4 5 9".
        let mut short = PromptLookup {
            max_ngram: 3,
            min_ngram: 2,
        };
        let context = [1, 2, 3, 4, 5, 9, 7, 2, 3];
        assert_eq!(short.draft(&context, 3), [4, 5, 9]);
        assert_eq!(short.draft(&context, 1), [4]);
        // No earlier occurrence of a long enough suffix: nothing.
        assert!(short.draft(&[1, 2, 3], 4).is_empty());
        assert!(short.draft(&[5, 1, 5, 2, 5], 1).is_empty());
        // The most recent occurrence wins.
        let mut single = PromptLookup {
            max_ngram: 3,
            min_ngram: 1,
        };
        assert_eq!(single.draft(&[5, 1, 5, 2, 5], 1), [2]);
    }
}
