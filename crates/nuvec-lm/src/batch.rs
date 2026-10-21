//! Serving many requests at once: a [`Batcher`] admits requests while there is room,
//! prefills them, and decodes every running request in one forward pass per step
//! ([`decode_batch`]), finishing each at an end-of-sequence token or its token budget.
//!
//! When the paged cache runs out of blocks, the most recently admitted request is
//! preempted: its blocks are freed and it waits again with its prompt and the tokens it has
//! produced, to be prefilled anew. A preempted request's later tokens are recomputed by
//! prefill instead of decoding, which can change their low bits.

use std::collections::VecDeque;

use nuvec::{Array, Result, SliceArg, argmax, logsumexp, slice, subtract};

use crate::generate::{GenerateOptions, Keys, prefill};
use crate::model::Model;
use crate::paged::{PagedCache, SeqId, decode_batch};

/// A request of a [`Batcher`].
pub type RequestId = usize;

/// What a [`Batcher::step`] produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// The next token of a request.
    Token {
        /// The request.
        id: RequestId,
        /// The token.
        token: u32,
    },
    /// A request has finished (after its last token).
    Finished {
        /// The request.
        id: RequestId,
        /// Why it finished.
        reason: FinishReason,
    },
}

/// Why a request finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// It produced an end-of-sequence token (the model's).
    EndOfSequence,
    /// It produced its `max_tokens`.
    Length,
}

#[derive(Debug, Clone)]
struct Waiting {
    id: RequestId,
    /// The prompt, followed by the tokens produced so far.
    tokens: Vec<u32>,
    options: GenerateOptions,
    produced: usize,
    /// Its sampling keys (from its seed), continued across preemptions.
    keys: Keys,
}

#[derive(Debug, Clone)]
struct Running {
    request: Waiting,
    sequence: SeqId,
    /// The token to feed next (the last one of the prompt or produced).
    next: u32,
}

/// Requests served together over one [`PagedCache`] (see the [module documentation](self)).
#[derive(Debug)]
pub struct Batcher<'a> {
    model: &'a Model,
    cache: PagedCache,
    block_size: usize,
    max_batch: usize,
    waiting: VecDeque<Waiting>,
    running: Vec<Running>,
    next_id: RequestId,
    preemptions: usize,
}

impl<'a> Batcher<'a> {
    /// A batcher decoding up to `max_batch` requests at a time over `blocks` blocks of
    /// `block_size` positions per attention layer.
    pub fn new(
        model: &'a Model,
        blocks: usize,
        block_size: usize,
        max_batch: usize,
    ) -> Result<Self> {
        Ok(Self {
            model,
            cache: PagedCache::new(model, blocks, block_size, max_batch)?,
            block_size,
            max_batch,
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_id: 0,
            preemptions: 0,
        })
    }

    /// Queues a request (a prompt of at least one token) and returns its id.
    pub fn submit(&mut self, prompt: Vec<u32>, options: GenerateOptions) -> Result<RequestId> {
        if prompt.is_empty() {
            return Err(nuvec::Error::InvalidArgument(
                "batcher: empty prompt".into(),
            ));
        }
        let id = self.next_id;
        self.next_id += 1;
        let request = Waiting {
            id,
            tokens: prompt,
            options,
            produced: 0,
            keys: Keys::new(options.seed),
        };
        if options.max_tokens == 0 {
            // Nothing to produce: finished at the next step.
            self.waiting.push_front(request);
        } else {
            self.waiting.push_back(request);
        }
        Ok(id)
    }

    /// Whether requests are waiting or running.
    pub fn is_busy(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Requests running (decoding) now.
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// Requests preempted so far (each counted once per preemption).
    pub fn preemptions(&self) -> usize {
        self.preemptions
    }

    /// Blocks a sequence of `positions` positions needs.
    fn blocks_for(&self, positions: usize) -> usize {
        positions.div_ceil(self.block_size)
    }

    /// Admits waiting requests while there is room (a batch slot, and blocks for the prompt
    /// and one more position for each running request), prefilling each.
    fn admit(&mut self, events: &mut Vec<Event>) -> Result<()> {
        while let Some(front) = self.waiting.front() {
            if front.options.max_tokens == front.produced {
                let request = self.waiting.pop_front().expect("front");
                events.push(Event::Finished {
                    id: request.id,
                    reason: FinishReason::Length,
                });
                continue;
            }
            let reserve = self.running.len() + 1;
            if self.running.len() == self.max_batch
                || self.blocks_for(front.tokens.len()) + reserve > self.cache.free_blocks()
            {
                break;
            }
            let request = self.waiting.pop_front().expect("front");
            let (head, last) = request.tokens.split_at(request.tokens.len() - 1);
            let mut kv = self.model.new_cache();
            prefill(self.model, &mut kv, head, request.options.prefill_step)?;
            let sequence = self.cache.insert(&kv)?;
            self.running.push(Running {
                next: last[0],
                request,
                sequence,
            });
        }
        Ok(())
    }

    /// Frees the most recently admitted request's blocks and queues it again first (its
    /// tokens include those it has produced).
    fn preempt(&mut self) -> Result<()> {
        let victim = self.running.pop().expect("a running request");
        self.cache.remove(victim.sequence)?;
        self.preemptions += 1;
        self.waiting.push_front(victim.request);
        Ok(())
    }

    /// One step: admits what fits, then decodes one token for every running request.
    /// Returns the tokens produced and the requests finished, in order.
    pub fn step(&mut self) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        self.admit(&mut events)?;
        // Each running sequence needs room for one more position; preempt until it fits.
        loop {
            let needed = self
                .running
                .iter()
                .filter(|r| {
                    let len = self.cache.len(r.sequence).unwrap_or(0);
                    len.is_multiple_of(self.block_size)
                })
                .count();
            if needed <= self.cache.free_blocks() || self.running.len() <= 1 {
                break;
            }
            self.preempt()?;
        }
        if self.running.is_empty() {
            return Ok(events);
        }
        let tokens: Vec<u32> = self.running.iter().map(|r| r.next).collect();
        let ids: Vec<SeqId> = self.running.iter().map(|r| r.sequence).collect();
        let logits = decode_batch(self.model, &tokens, &ids, &mut self.cache)?;
        let next = self.sample(&logits)?;
        let eos = &self.model.config.eos_token_ids;
        let mut finished = Vec::new();
        for (i, (r, &token)) in self.running.iter_mut().zip(&next).enumerate() {
            r.request.produced += 1;
            r.request.tokens.push(token);
            r.next = token;
            events.push(Event::Token {
                id: r.request.id,
                token,
            });
            let reason = if eos.contains(&token) {
                Some(FinishReason::EndOfSequence)
            } else if r.request.produced == r.request.options.max_tokens {
                Some(FinishReason::Length)
            } else {
                None
            };
            if let Some(reason) = reason {
                finished.push((i, reason));
            }
        }
        for &(i, reason) in finished.iter().rev() {
            let r = self.running.remove(i);
            self.cache.remove(r.sequence)?;
            events.push(Event::Finished {
                id: r.request.id,
                reason,
            });
        }
        Ok(events)
    }

    /// The next token of each running request from its row of `logits` (`[B, vocab]`),
    /// with its own sampler and keys: one argmax for greedy rows, the sampler on the
    /// others' (as `generate` samples alone, so a seeded request gets the same tokens).
    fn sample(&mut self, logits: &Array) -> Result<Vec<u32>> {
        let greedy = argmax(logits, Some(-1), false)?.to_vec::<u32>()?;
        let mut next = greedy;
        for (i, r) in self.running.iter_mut().enumerate() {
            let key = r.request.keys.next()?;
            let sampler = &r.request.options.sampler;
            if sampler.temperature == 0.0 {
                continue;
            }
            let row = slice(
                logits,
                &[
                    SliceArg::new(Some(i as isize), Some(i as isize + 1), 1),
                    SliceArg::full(),
                ],
            )?;
            // Log-probabilities, as mlx-lm samples from.
            let logprobs = subtract(&row, logsumexp(&row, &[-1], true)?)?;
            next[i] = sampler
                .sample_with(&logprobs, key.as_ref())?
                .to_vec::<u32>()?[0];
        }
        Ok(next)
    }
}
