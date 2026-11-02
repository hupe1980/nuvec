//! Serving many requests at once: a [`Batcher`] admits requests while there is room,
//! prefills them, and decodes every running request in one forward pass per step
//! ([`decode_batch`](crate::decode_batch)), finishing each at an end-of-sequence token
//! (not emitted) or its token budget.
//!
//! Requests admitted at the same step whose prompts have the same length (more than 64
//! tokens) are prefilled together on quantized models, one batched forward pass per chunk
//! (as mlx-lm's `BatchGenerator`).
//!
//! Products route as MLX routes them, so a request's tokens can depend in their low bits
//! on the requests batched with it, as in mlx-lm. With
//! [`batch_invariant`](BatcherOptions::batch_invariant), every step runs row-exact
//! ([`nuvec::with_row_exact`]): each request gets the tokens it gets alone, at some cost in
//! throughput.
//!
//! Decoding is pipelined as [`generate`](crate::generate()): a step submits the next
//! decoding step, fed with the tokens of the step before, then reads those tokens (one
//! host read per step, every row sampled on the GPU), so the GPU never waits for the host.
//! At most one step is in flight (on a pipeline stage, one micro-batch of the running
//! requests per stage, so every stage works on a different one). A request that ends while
//! its next step is in flight
//! (end-of-sequence, cancellation) emits nothing from it; that step's work is dropped with
//! the request. Admitting requests, preemption and cancellation wait for the step in
//! flight (the following step runs without lookahead).
//!
//! When the paged cache runs out of blocks, the most recently admitted request is
//! preempted: its blocks are freed and it waits again with its prompt and the tokens it has
//! produced, to be prefilled anew. A preempted request's later tokens are recomputed by
//! prefill instead of decoding, which can change their low bits. A request that no longer
//! fits even in an empty cache finishes with [`FinishReason::CacheFull`].
//!
//! A step is atomic per request: the cache commits a step's new positions only after its
//! forward pass succeeded. When a batched forward pass fails, each request is run alone;
//! the requests that fail alone finish with [`FinishReason::Error`] and the others go on
//! (a request may still emit the token that was in flight before the failing step). A
//! request that fails to be admitted (prefilled) finishes the same way.

use std::collections::VecDeque;

use nuvec::{Array, Error, Result, async_eval, eval, ops};

pub use crate::generate::FinishReason;
use crate::generate::{GenerateOptions, Keys, judge, prefill_rows};
use crate::model::Model;
use crate::paged::{PagedCache, Reservation, SeqId, forward_reserved};
use crate::stop::StopState;

/// Tokens to prefill from which prompts of the same length admitted together are prefilled
/// in one batch (more than every quantized product's matrix-vector limit, 33 rows).
const BATCHED_PREFILL_MIN: usize = 64;

/// A request of a [`Batcher`].
pub type RequestId = usize;

/// What a [`Batcher::step`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone)]
struct Waiting {
    id: RequestId,
    /// The prompt, followed by the tokens produced so far.
    tokens: Vec<u32>,
    options: GenerateOptions,
    produced: usize,
    /// Its sampling keys (from its seed), continued across preemptions.
    keys: Keys,
    /// Its progress through its stop strings, continued across preemptions.
    stops: StopState,
}

#[derive(Debug, Clone)]
struct Running {
    request: Waiting,
    sequence: SeqId,
    /// The token to feed next (the last one of the prompt or produced), when no step of
    /// this request is in flight.
    next: u32,
}

/// A submitted step whose tokens have not been read: its positions are committed.
#[derive(Debug)]
struct InFlight {
    /// Its requests, in row order (one finished since is skipped).
    rows: Vec<RequestId>,
    /// Their tokens, `[rows]`, submitted.
    tokens: Array,
}

/// Why submitting a step failed.
enum Failed {
    /// The forward pass: rolled back, the cache as it was.
    Forward(Error),
    /// Sampling, after the positions were committed: these requests cannot go on.
    Sample(Error),
}

/// How a [`Batcher`] is sized and run. Start from the defaults and set what differs:
/// `BatcherOptions { max_batch: 4, ..Default::default() }`.
#[derive(Debug, Clone, PartialEq)]
pub struct BatcherOptions {
    /// Blocks per full-attention layer. `None` (the default) sizes the cache from memory:
    /// `memory_fraction` of what the GPU's recommended working set leaves after the memory
    /// in use (the weights), at least one block per request and at most room for
    /// `max_batch` sequences of 131,072 positions. Without a Metal device, room for
    /// `max_batch` sequences of 4,096 positions.
    pub blocks: Option<usize>,
    /// Positions per block, a power of two (16, as vLLM).
    pub block_size: usize,
    /// Requests decoded together at most (32, as mlx-lm's `BatchGenerator`).
    pub max_batch: usize,
    /// On a pipelined model, the micro-batches the running requests are split into, each
    /// with its own step in flight. `None` (the default): one per stage, so every stage
    /// works on a different one; 1 steps them as one batch. No effect on a whole model.
    pub micro_batches: Option<usize>,
    /// Whether each request gets the tokens it gets alone ([`crate::generate()`]), whatever
    /// it is batched with: every step then runs row-exact ([`nuvec::with_row_exact`]), at
    /// some cost in throughput. Off by default (MLX's routing, as mlx-lm's
    /// `BatchGenerator`).
    pub batch_invariant: bool,
    /// The share of the free memory a default block count takes (0.9).
    pub memory_fraction: f64,
}

/// The positions per request a default block count holds at most.
const MAX_DEFAULT_POSITIONS: usize = 131_072;

impl Default for BatcherOptions {
    fn default() -> Self {
        Self {
            blocks: None,
            block_size: 16,
            max_batch: 32,
            micro_batches: None,
            batch_invariant: false,
            memory_fraction: 0.9,
        }
    }
}

impl BatcherOptions {
    /// Checks every field, naming the one that is invalid.
    pub fn validate(&self) -> Result<()> {
        let invalid = |msg: &str| Err(Error::InvalidArgument(format!("batcher: {msg}")));
        if !self.block_size.is_power_of_two() {
            return invalid(&format!(
                "block_size must be a power of two, not {}",
                self.block_size
            ));
        }
        if self.max_batch == 0 {
            return invalid("max_batch must be at least 1");
        }
        if self.blocks == Some(0) {
            return invalid("blocks must be at least 1");
        }
        if self.micro_batches == Some(0) {
            return invalid("micro_batches must be at least 1");
        }
        if !(self.memory_fraction > 0.0 && self.memory_fraction <= 1.0) {
            return invalid(&format!(
                "memory_fraction must be in (0, 1], not {}",
                self.memory_fraction
            ));
        }
        Ok(())
    }

    /// The block count: `blocks`, or sized from memory (see [`Self::blocks`]).
    fn block_count(&self, model: &Model) -> usize {
        if let Some(blocks) = self.blocks {
            return blocks;
        }
        let per_block = (model.kv_bytes_per_position() * self.block_size).max(1);
        let blocks = match nuvec::metal::device_info() {
            Some(info) => {
                let free = (info.recommended_working_set as usize)
                    .saturating_sub(nuvec::memory::active_memory());
                (free as f64 * self.memory_fraction) as usize / per_block
            }
            None => self.max_batch * 4096 / self.block_size,
        };
        blocks.clamp(
            self.max_batch,
            self.max_batch * MAX_DEFAULT_POSITIONS / self.block_size,
        )
    }
}

/// Requests served together over one [`PagedCache`] (see the [module documentation](self)).
#[derive(Debug)]
pub struct Batcher<'a> {
    model: &'a Model,
    cache: PagedCache,
    max_batch: usize,
    waiting: VecDeque<Waiting>,
    running: Vec<Running>,
    next_id: RequestId,
    preemptions: usize,
    /// Requests to finish as cancelled at the next step.
    cancelled: Vec<RequestId>,
    /// The steps submitted and not read yet, oldest first: one, or on a pipeline stage one
    /// micro-batch per stage.
    in_flight: VecDeque<InFlight>,
    /// A fault hook: the forward pass fails whenever it includes this request.
    #[cfg(test)]
    fault: Option<RequestId>,
    /// Batched prefills so far.
    #[cfg(test)]
    batched: usize,
    /// Whether steps run row-exact ([`BatcherOptions::batch_invariant`]).
    invariant: bool,
    /// Micro-batches kept in flight ([`BatcherOptions::micro_batches`]); `None`: one per
    /// stage.
    micro: Option<usize>,
    /// Steps read so far (the allocator's cache is emptied every 512, as mlx-lm's).
    steps: usize,
}

impl Drop for Batcher<'_> {
    /// Waits for the step in flight (its errors are moot).
    fn drop(&mut self) {
        let invariant = self.invariant;
        for f in std::mem::take(&mut self.in_flight) {
            let wait = || eval(std::slice::from_ref(&f.tokens));
            let _ = if invariant {
                nuvec::with_row_exact(wait)
            } else {
                wait()
            };
        }
    }
}

impl<'a> Batcher<'a> {
    /// A batcher for `model` as `options` say ([`BatcherOptions`]); the options are
    /// checked first.
    pub fn new(model: &'a Model, options: BatcherOptions) -> Result<Self> {
        options.validate()?;
        let blocks = options.block_count(model);
        let max_batch = options.max_batch;
        Ok(Self {
            model,
            cache: PagedCache::new(model, blocks, options.block_size, max_batch)?,
            max_batch,
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_id: 0,
            preemptions: 0,
            cancelled: Vec::new(),
            in_flight: VecDeque::new(),
            #[cfg(test)]
            fault: None,
            #[cfg(test)]
            batched: 0,
            invariant: options.batch_invariant,
            micro: options.micro_batches,
            steps: 0,
        })
    }

    /// Queues a request and returns its id. The prompt must have at least one token and
    /// fit in the empty cache (with room for one token), and the sampler's settings must
    /// be valid ([`crate::Sampler::validate`]).
    pub fn submit(&mut self, prompt: Vec<u32>, options: GenerateOptions) -> Result<RequestId> {
        let invalid = |msg: String| Err(nuvec::Error::InvalidArgument(format!("batcher: {msg}")));
        if prompt.is_empty() {
            return invalid("empty prompt".into());
        }
        options.validate()?;
        if !self.fits(prompt.len()) {
            return invalid(format!(
                "a prompt of {} tokens never fits in {} blocks of {} positions",
                prompt.len(),
                self.cache.total_blocks(),
                self.cache.block_size()
            ));
        }
        let id = self.next_id;
        self.next_id += 1;
        let nothing = options.max_tokens == 0;
        let request = Waiting {
            id,
            tokens: prompt,
            keys: Keys::new(options.seed),
            options,
            produced: 0,
            stops: StopState::default(),
        };
        if nothing {
            // Nothing to produce: finished at the next step.
            self.waiting.push_front(request);
        } else {
            self.waiting.push_back(request);
        }
        Ok(id)
    }

    /// Cancels a waiting or running request: it finishes with
    /// [`FinishReason::Cancelled`] at the next [`Batcher::step`], which frees its blocks
    /// (no token of a step in flight is emitted). Returns `false` (and nothing happens)
    /// for an unknown, finished or already cancelled request.
    pub fn cancel(&mut self, id: RequestId) -> bool {
        let known = self.waiting.iter().any(|w| w.id == id) || self.index(id).is_some();
        if !known || self.cancelled.contains(&id) {
            return false;
        }
        self.cancelled.push(id);
        true
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

    /// The index of running request `id`.
    fn index(&self, id: RequestId) -> Option<usize> {
        self.running.iter().position(|r| r.request.id == id)
    }

    /// Blocks a sequence of `positions` positions needs.
    fn blocks_for(&self, positions: usize) -> usize {
        positions.div_ceil(self.cache.block_size())
    }

    /// Whether a request of `tokens` tokens fits in the empty cache, with room for one more.
    fn fits(&self, tokens: usize) -> bool {
        self.blocks_for(tokens) < self.cache.total_blocks()
    }

    /// Blocks the next step of running requests `rows` allocates.
    fn blocks_needed(&self, rows: impl Iterator<Item = usize>) -> usize {
        let block_size = self.cache.block_size();
        rows.filter(|&i| {
            let len = self.cache.len(self.running[i].sequence).unwrap_or(0);
            len.is_multiple_of(block_size)
        })
        .count()
    }

    /// Finishes the cancelled requests, freeing their blocks.
    fn finish_cancelled(&mut self, events: &mut Vec<Event>) -> Result<()> {
        for id in std::mem::take(&mut self.cancelled) {
            if let Some(i) = self.waiting.iter().position(|w| w.id == id) {
                self.waiting.remove(i);
            } else if let Some(i) = self.index(id) {
                let r = self.running.remove(i);
                self.cache.remove(r.sequence)?;
            } else {
                continue;
            }
            events.push(Event::Finished {
                id,
                reason: FinishReason::Cancelled,
            });
        }
        Ok(())
    }

    /// Whether the first waiting request would be admitted (or finished) by
    /// [`admit`](Self::admit) now.
    fn admissible(&self) -> bool {
        let Some(front) = self.waiting.front() else {
            return false;
        };
        front.options.max_tokens == front.produced
            || !self.fits(front.tokens.len())
            || (self.running.len() < self.max_batch
                && self.blocks_for(front.tokens.len()) + self.running.len()
                    < self.cache.free_blocks())
    }

    /// Admits waiting requests while there is room (a batch slot, and blocks for the prompt
    /// and one more position for each running request), prefilling each (those of the
    /// same length together). A request that fails to be admitted finishes with an error.
    fn admit(&mut self, events: &mut Vec<Event>) {
        let mut admitted: Vec<Waiting> = Vec::new();
        let mut blocks = 0;
        while let Some(front) = self.waiting.front() {
            let finished = if front.options.max_tokens == front.produced {
                Some(FinishReason::Length)
            } else if !self.fits(front.tokens.len()) {
                // Grown by preemption beyond what the cache can hold.
                Some(FinishReason::CacheFull)
            } else {
                None
            };
            if let Some(reason) = finished {
                let request = self.waiting.pop_front().expect("front");
                events.push(Event::Finished {
                    id: request.id,
                    reason,
                });
                continue;
            }
            let running = self.running.len() + admitted.len();
            let needed = self.blocks_for(front.tokens.len());
            if running == self.max_batch || blocks + needed + running + 1 > self.cache.free_blocks()
            {
                break;
            }
            blocks += needed;
            admitted.push(self.waiting.pop_front().expect("front"));
        }
        let n = admitted.len();
        let mut sequences: Vec<Option<Result<SeqId>>> = (0..n).map(|_| None).collect();
        for i in 0..n {
            if sequences[i].is_some() {
                continue;
            }
            let (len, step) = (admitted[i].tokens.len(), admitted[i].options.prefill_step);
            let group: Vec<usize> = (i..n)
                .filter(|&j| {
                    sequences[j].is_none()
                        && admitted[j].tokens.len() == len
                        && admitted[j].options.prefill_step == step
                })
                .collect();
            if group.len() > 1 && self.batched_prefill(len) {
                let requests: Vec<&Waiting> = group.iter().map(|&j| &admitted[j]).collect();
                // A group that fails is prefilled request by request (one bad request
                // fails alone).
                if let Ok(ids) = self.prefill_group(&requests) {
                    for (&j, id) in group.iter().zip(ids) {
                        sequences[j] = Some(Ok(id));
                    }
                    continue;
                }
            }
            for j in group {
                sequences[j] = Some(self.prefill_group(&[&admitted[j]]).map(|ids| ids[0]));
            }
        }
        for (request, sequence) in admitted.into_iter().zip(sequences) {
            match sequence.expect("every admitted request prefilled") {
                Ok(sequence) => {
                    let next = *request.tokens.last().expect("never empty");
                    self.running.push(Running {
                        next,
                        request,
                        sequence,
                    });
                }
                Err(e) => events.push(Event::Finished {
                    id: request.id,
                    reason: FinishReason::Error(e.clone()),
                }),
            }
        }
    }

    /// Whether prompts of `len` tokens admitted together are prefilled in one batch: for
    /// quantized models, from [`BATCHED_PREFILL_MIN`] tokens to prefill on. There every
    /// product runs tiled with its K partitions chosen per sequence, so each sequence gets
    /// the bits it gets alone (fewer rows run matrix-vector kernels alone; unquantized
    /// products choose their tiles from all rows).
    fn batched_prefill(&self, len: usize) -> bool {
        self.model.config.quantization.is_some() && len > BATCHED_PREFILL_MIN
    }

    /// Prefills requests' tokens but the last (prompts of the same length, nothing for
    /// one-token prompts) into the paged cache, together; all are inserted (in one write
    /// per pool) or none.
    fn prefill_group(&mut self, requests: &[&Waiting]) -> Result<Vec<SeqId>> {
        let heads: Vec<&[u32]> = requests
            .iter()
            .map(|r| &r.tokens[..r.tokens.len() - 1])
            .collect();
        let mut kv = self.model.new_cache();
        prefill_rows(
            self.model,
            &mut kv,
            &heads,
            requests[0].options.prefill_step,
        )?;
        #[cfg(test)]
        if requests.len() > 1 {
            self.batched += 1;
        }
        self.cache.insert_many(&kv)
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

    /// Makes room for one more position of each running request: preempts until it fits,
    /// or finishes a request alone and still out of room with `CacheFull`.
    fn make_room(&mut self, events: &mut Vec<Event>) -> Result<()> {
        loop {
            if self.blocks_needed(0..self.running.len()) <= self.cache.free_blocks() {
                return Ok(());
            }
            if self.running.len() > 1 {
                self.preempt()?;
                continue;
            }
            // Alone and still out of room: it can never go on.
            let r = self.running.pop().expect("a running request");
            self.cache.remove(r.sequence)?;
            events.push(Event::Finished {
                id: r.request.id,
                reason: FinishReason::CacheFull,
            });
        }
    }

    /// One step: finishes cancelled requests; with no step in flight, admits what fits
    /// and submits a step for every running request; then submits the next step when
    /// nothing else waits (see the module documentation) and reads the step in flight.
    /// Returns the tokens produced and the requests finished, in order.
    pub fn step(&mut self) -> Result<Vec<Event>> {
        if self.invariant {
            nuvec::with_row_exact(|| self.step_now())
        } else {
            self.step_now()
        }
    }

    fn step_now(&mut self) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        self.finish_cancelled(&mut events)?;
        // Steps whose requests have all finished: only their work is left.
        let (live, done): (Vec<InFlight>, Vec<InFlight>) = std::mem::take(&mut self.in_flight)
            .into_iter()
            .partition(|f| f.rows.iter().any(|&id| self.index(id).is_some()));
        self.in_flight = live.into();
        for f in done {
            let _ = eval(std::slice::from_ref(&f.tokens));
        }
        if self.in_flight.is_empty() {
            self.admit(&mut events);
            self.make_room(&mut events)?;
            if self.running.is_empty() {
                return Ok(events);
            }
            // One micro-batch per pipeline stage (one batch otherwise), by request, each
            // submitted before any is read, so every stage runs a different one.
            let ids: Vec<RequestId> = self.running.iter().map(|r| r.request.id).collect();
            let parts = self.depth().min(ids.len());
            for part in 0..parts {
                let (from, to) = (ids.len() * part / parts, ids.len() * (part + 1) / parts);
                let rows: Vec<usize> = ids[from..to]
                    .iter()
                    .filter_map(|&id| self.index(id))
                    .collect();
                if let Some(f) = self.submit_rows(&rows, &mut events)? {
                    self.in_flight.push_back(f);
                }
            }
        }
        let Some(flight) = self.in_flight.pop_front() else {
            return Ok(events);
        };
        let mut failed = Vec::new();
        let next = self.look_ahead(&flight, &mut failed);
        self.read(flight, next, failed, &mut events)?;
        self.steps += 1;
        if self.steps.is_multiple_of(512) {
            nuvec::memory::clear_cache();
        }
        Ok(events)
    }

    /// The steps kept in flight: one per stage on a pipeline stage (or as set), else one.
    fn depth(&self) -> usize {
        match self.model.stage() {
            Some(s) => self.micro.unwrap_or(s.group().size()).max(1),
            None => 1,
        }
    }

    /// Submits a step for running requests `rows`, fed with their `next` tokens; when the
    /// batched forward pass fails, each request is run alone and those failing alone
    /// finish with an error.
    fn submit_rows(&mut self, rows: &[usize], events: &mut Vec<Event>) -> Result<Option<InFlight>> {
        let all: Vec<usize> = rows.to_vec();
        let tokens = |b: &Self, rows: &[usize]| {
            Array::from(
                rows.iter()
                    .map(|&i| b.running[i].next)
                    .collect::<Vec<u32>>(),
            )
        };
        let mut finished: Vec<(RequestId, FinishReason)> = Vec::new();
        let flight = match self.launch(&all, &tokens(self, &all)) {
            Ok(f) => Some(f),
            Err(Failed::Sample(e)) => {
                finished.extend(
                    all.iter()
                        .map(|&i| (self.running[i].request.id, FinishReason::Error(e.clone()))),
                );
                None
            }
            Err(Failed::Forward(_)) => {
                let mut parts: Vec<InFlight> = Vec::new();
                for i in all {
                    match self.launch(&[i], &tokens(self, &[i])) {
                        Ok(f) => parts.push(f),
                        Err(Failed::Forward(e) | Failed::Sample(e)) => finished
                            .push((self.running[i].request.id, FinishReason::Error(e.clone()))),
                    }
                }
                (!parts.is_empty()).then(|| InFlight {
                    rows: parts.iter().flat_map(|f| f.rows.iter().copied()).collect(),
                    tokens: ops::concatenate(
                        &parts.iter().map(|f| f.tokens.clone()).collect::<Vec<_>>(),
                        0,
                    ),
                })
            }
        };
        self.finish(finished, events)?;
        Ok(flight)
    }

    /// Submits the step after `flight` for its requests that go on past it, fed with its
    /// tokens, unless a request could be admitted or the cache lacks the blocks (then the
    /// next step runs without lookahead). Requests whose step cannot be sampled go to
    /// `failed` (their positions are committed).
    fn look_ahead(
        &mut self,
        flight: &InFlight,
        failed: &mut Vec<(RequestId, Error)>,
    ) -> Option<InFlight> {
        if self.admissible() {
            return None;
        }
        let mut positions = Vec::new();
        let mut rows = Vec::new();
        for (j, &id) in flight.rows.iter().enumerate() {
            if let Some(i) = self.index(id) {
                let r = &self.running[i].request;
                if r.produced + 1 < r.options.max_tokens {
                    positions.push(j as i32);
                    rows.push(i);
                }
            }
        }
        if rows.is_empty() || self.blocks_needed(rows.iter().copied()) > self.cache.free_blocks() {
            return None;
        }
        let tokens = if rows.len() == flight.rows.len() {
            flight.tokens.clone()
        } else {
            ops::take(&flight.tokens, &Array::from(positions), 0)
        };
        match self.launch(&rows, &tokens) {
            Ok(next) => Some(next),
            // The next step runs without lookahead and isolates the failure.
            Err(Failed::Forward(_)) => None,
            Err(Failed::Sample(e)) => {
                failed.extend(
                    rows.iter()
                        .map(|&i| (self.running[i].request.id, e.clone())),
                );
                None
            }
        }
    }

    /// Reads the tokens of `flight` (one host read), emits them and finishes the requests
    /// that are done (and those in `failed`); `next` becomes the step in flight.
    fn read(
        &mut self,
        flight: InFlight,
        next: Option<InFlight>,
        failed: Vec<(RequestId, Error)>,
        events: &mut Vec<Event>,
    ) -> Result<()> {
        let tokens = match flight.tokens.to_vec::<u32>() {
            Ok(tokens) => tokens,
            Err(e) => {
                // No token came out: these requests cannot go on (the step after them
                // depends on this one and is dropped).
                if let Some(next) = next {
                    let _ = eval(std::slice::from_ref(&next.tokens));
                }
                let finished = flight
                    .rows
                    .iter()
                    .filter(|&&id| self.index(id).is_some())
                    .map(|&id| (id, FinishReason::Error(e.clone())))
                    .collect();
                return self.finish(finished, events);
            }
        };
        let eos = &self.model.config.eos_token_ids;
        let mut finished: Vec<(RequestId, FinishReason)> = Vec::new();
        for (&id, &token) in flight.rows.iter().zip(&tokens) {
            let Some(i) = self.running.iter().position(|r| r.request.id == id) else {
                continue;
            };
            let r = &mut self.running[i].request;
            r.produced += 1;
            let (emit, reason) = judge(&r.options, eos, &mut r.stops, token, r.produced);
            if emit {
                r.tokens.push(token);
                self.running[i].next = token;
                events.push(Event::Token { id, token });
            }
            if let Some(reason) = reason {
                finished.push((id, reason));
            }
        }
        for (id, e) in failed {
            if !finished.iter().any(|&(f, _)| f == id) {
                finished.push((id, FinishReason::Error(e)));
            }
        }
        if let Some(next) = next {
            self.in_flight.push_back(next);
        }
        self.finish(finished, events)
    }

    /// Finishes running requests, freeing their blocks, in order.
    fn finish(
        &mut self,
        finished: Vec<(RequestId, FinishReason)>,
        events: &mut Vec<Event>,
    ) -> Result<()> {
        for (id, reason) in finished {
            let Some(i) = self.index(id) else {
                continue;
            };
            let r = self.running.remove(i);
            self.cache.remove(r.sequence)?;
            events.push(Event::Finished { id, reason });
        }
        Ok(())
    }

    /// Submits a step for running requests `rows` fed with `tokens` (`[rows]`): reserves
    /// a position for each, runs the forward pass, samples on the GPU and commits.
    fn launch(&mut self, rows: &[usize], tokens: &Array) -> std::result::Result<InFlight, Failed> {
        let ids: Vec<SeqId> = rows.iter().map(|&i| self.running[i].sequence).collect();
        let reservation = self.cache.reserve(&ids).map_err(Failed::Forward)?;
        let logits = match self.forward(rows, tokens, &reservation) {
            Ok(logits) => logits,
            Err(e) => {
                self.cache.rollback(reservation);
                return Err(Failed::Forward(e));
            }
        };
        self.cache.commit(reservation);
        let tokens = self
            .sample(rows, &logits)
            .and_then(|t| async_eval(std::slice::from_ref(&t)).map(|()| t))
            .map_err(Failed::Sample)?;
        Ok(InFlight {
            rows: rows.iter().map(|&i| self.running[i].request.id).collect(),
            tokens,
        })
    }

    /// The logits `[rows, vocab]` of a reserved step of `rows` (indices of running
    /// requests) fed with `tokens`, not committed.
    fn forward(
        &mut self,
        rows: &[usize],
        tokens: &Array,
        reservation: &Reservation,
    ) -> Result<Array> {
        // The fault hook fails the step at the last layer, after the layers replaced their
        // pools.
        #[cfg(test)]
        let _fault = self
            .fault
            .filter(|f| rows.iter().any(|&i| self.running[i].request.id == *f))
            .map(|_| crate::model::tests::fail_paged_at(self.model.config.num_hidden_layers - 1));
        #[cfg(not(test))]
        let _ = rows;
        forward_reserved(self.model, tokens, reservation, &mut self.cache)
    }

    /// The next token of each of `rows` (running requests) from its row of `logits`
    /// (`[rows, vocab]`), as one array `[rows]`, not evaluated: one argmax for greedy rows,
    /// each other row's sampler with its own keys (as `generate` samples alone, so a seeded
    /// request gets the same tokens).
    fn sample(&mut self, rows: &[usize], logits: &Array) -> Result<Array> {
        let logits = &self.masked(rows, logits);
        let greedy = logits.argmax(-1);
        let sampled = |i: usize| self.running[i].request.options.sampler.temperature != 0.0;
        if !rows.iter().any(|&i| sampled(i)) {
            // A pipeline stage yields rank 0's tokens (the only rank that runs the head).
            return self.model.share_sampled(&greedy, true).check();
        }
        let mut parts = Vec::with_capacity(rows.len());
        for (j, &i) in rows.iter().enumerate() {
            let j = j as i32;
            let r = &mut self.running[i].request;
            if r.options.sampler.temperature == 0.0 {
                parts.push(greedy.i(j..j + 1));
                continue;
            }
            let key = r.keys.next()?;
            let row = logits.i(j..j + 1);
            // Log-probabilities, as mlx-lm samples from.
            let logprobs = row.subtract(row.logsumexp_keepdims(-1));
            parts.push(
                r.options
                    .sampler
                    .sample_with(&logprobs, key.as_ref())?
                    .reshape([1])
                    .astype(greedy.dtype()),
            );
        }
        let seeded = rows.iter().all(|&i| {
            let o = &self.running[i].request.options;
            o.sampler.temperature == 0.0 || o.seed.is_some()
        });
        self.model
            .share_sampled(&ops::concatenate(&parts, 0), seeded)
            .check()
    }

    /// `logits` with each row's end tokens masked while the token it samples is within
    /// the request's `min_tokens` ([`GenerateOptions::min_tokens`]).
    fn masked(&self, rows: &[usize], logits: &Array) -> Array {
        let eos = &self.model.config.eos_token_ids;
        // The token a row samples now: those read, those in flight, then this one.
        let n = |i: usize| {
            let r = &self.running[i].request;
            let pending = self.in_flight.iter().filter(|f| f.rows.contains(&r.id));
            r.produced + pending.count() + 1
        };
        if !rows
            .iter()
            .any(|&i| n(i) <= self.running[i].request.options.min_tokens)
        {
            return logits.clone();
        }
        let parts: Vec<Array> = rows
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                let j = j as i32;
                self.running[i]
                    .request
                    .options
                    .mask(logits.i(j..j + 1), eos, n(i))
            })
            .collect();
        ops::concatenate(&parts, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate;
    use crate::model::tests::fixture_model;

    /// Each invalid field is named; the defaults size the cache from memory and serve.
    #[test]
    fn options_are_checked_and_the_default_cache_fits_memory() {
        let Some(model) = fixture_model("qwen3_f32") else {
            return;
        };
        let d = BatcherOptions::default;
        for (options, field) in [
            (
                BatcherOptions {
                    block_size: 12,
                    ..d()
                },
                "block_size",
            ),
            (
                BatcherOptions {
                    max_batch: 0,
                    ..d()
                },
                "max_batch",
            ),
            (
                BatcherOptions {
                    blocks: Some(0),
                    ..d()
                },
                "blocks",
            ),
            (
                BatcherOptions {
                    micro_batches: Some(0),
                    ..d()
                },
                "micro_batches",
            ),
            (
                BatcherOptions {
                    memory_fraction: 0.0,
                    ..d()
                },
                "memory_fraction",
            ),
            (
                BatcherOptions {
                    memory_fraction: 1.5,
                    ..d()
                },
                "memory_fraction",
            ),
        ] {
            let e = Batcher::new(&model, options).unwrap_err().to_string();
            assert!(e.contains(field), "{field}: {e}");
        }
        let mut b = Batcher::new(&model, d()).unwrap();
        let blocks = b.cache.total_blocks();
        let per_block = model.kv_bytes_per_position() * 16;
        let free = nuvec::metal::device_info().unwrap().recommended_working_set as usize;
        assert!(blocks >= 32 && blocks * per_block <= free, "{blocks}");
        assert!(blocks <= 32 * MAX_DEFAULT_POSITIONS / 16, "{blocks}");
        let id = b.submit(vec![1, 2, 3], greedy(4)).unwrap();
        let mut done = None;
        while b.is_busy() {
            for e in b.step().unwrap() {
                if let Event::Finished { id: f, reason } = e {
                    assert_eq!(f, id);
                    done = Some(reason);
                }
            }
        }
        assert!(done.is_some());
    }

    fn greedy(max_tokens: usize) -> GenerateOptions {
        GenerateOptions {
            max_tokens,
            ..Default::default()
        }
    }

    fn solo(model: &Model, prompt: &[u32], max_tokens: usize) -> Vec<u32> {
        generate(model, prompt, greedy(max_tokens))
            .unwrap()
            .collect::<Result<_>>()
            .unwrap()
    }

    /// Steps until idle (a watchdog of `limit` steps), collecting each request's tokens
    /// and finish reason; `before(step, batcher)` runs before each step.
    fn run(
        batcher: &mut Batcher<'_>,
        requests: usize,
        limit: usize,
        mut before: impl FnMut(usize, &mut Batcher<'_>),
    ) -> (Vec<Vec<u32>>, Vec<Option<FinishReason>>) {
        let mut tokens = vec![Vec::new(); requests];
        let mut finished = vec![None; requests];
        for step in 0..limit {
            if !batcher.is_busy() {
                return (tokens, finished);
            }
            before(step, batcher);
            for event in batcher.step().unwrap() {
                match event {
                    Event::Token { id, token } => tokens[id].push(token),
                    Event::Finished { id, reason } => {
                        assert!(finished[id].is_none(), "request {id} finished twice");
                        finished[id] = Some(reason);
                    }
                }
            }
        }
        panic!("still busy after {limit} steps");
    }

    /// A forward pass that fails for one request finishes that request with an error (after
    /// the token in flight); the others go on with the tokens of a run without the fault, and
    /// the cache's blocks and pools are intact.
    #[test]
    fn a_failing_request_does_not_poison_the_batch() {
        for name in ["qwen3_f32", "qwen3_5_f32"] {
            let Some(model) = fixture_model(name) else {
                return;
            };
            let prompt = [3u32, 17, 92, 5, 64, 33, 101, 8, 77];
            let prompts = [&prompt[..5], &prompt[2..9], &prompt[1..4]];
            let mut batcher = Batcher::new(
                &model,
                BatcherOptions {
                    blocks: Some(32),
                    block_size: 4,
                    max_batch: 3,
                    batch_invariant: true,
                    ..Default::default()
                },
            )
            .unwrap();
            for p in prompts {
                batcher.submit(p.to_vec(), greedy(8)).unwrap();
            }
            // The second request fails from its fourth token on.
            let (tokens, finished) = run(&mut batcher, 3, 50, |step, b| {
                if step == 3 {
                    b.fault = Some(1);
                }
            });
            assert!(
                matches!(&finished[1], Some(FinishReason::Error(m)) if m.to_string().contains("injected")),
                "{name}: {:?}",
                finished[1]
            );
            // Its tokens up to the fault, and the one in flight when it was set.
            assert_eq!(tokens[1].len(), 4, "{name}");
            assert_eq!(
                tokens[1],
                solo(&model, prompts[1], 4),
                "{name}: before the fault"
            );
            // The one-row step after the fault can differ from the batched row in its
            // low bits, so the greedy tokens are compared.
            for i in [0, 2] {
                assert_eq!(
                    tokens[i],
                    solo(&model, prompts[i], 8),
                    "{name}: request {i}"
                );
                assert_eq!(finished[i], Some(FinishReason::Length));
            }
            assert!(batcher.cache.is_empty(), "{name}");
            assert_eq!(batcher.cache.free_blocks(), 32, "{name}: blocks returned");
            // The pools still serve: a new request gets its solo tokens.
            batcher.fault = None;
            batcher.submit(prompts[1].to_vec(), greedy(8)).unwrap();
            let (tokens, _) = run(&mut batcher, 4, 50, |_, _| {});
            assert_eq!(tokens[3], solo(&model, prompts[1], 8), "{name}: after");
        }
    }

    /// A mixed workload (a one-token prompt, a prompt that never fits, requests cancelled
    /// waiting and running, one outgrowing the cache, one failing) runs to completion;
    /// every request that does not fail gets `generate`'s tokens.
    #[test]
    fn a_mixed_workload_runs_to_completion() {
        for name in ["qwen3_f32", "qwen3_5_f32"] {
            let Some(model) = fixture_model(name) else {
                return;
            };
            let base = [3u32, 17, 92, 5, 64, 33, 101, 8, 77, 2, 45, 120, 9, 56];
            let prompt = |n: usize| base.iter().cycle().take(n).copied().collect::<Vec<_>>();
            // 10 blocks of 4 positions, three requests at a time.
            let mut batcher = Batcher::new(
                &model,
                BatcherOptions {
                    blocks: Some(10),
                    block_size: 4,
                    max_batch: 3,
                    batch_invariant: true,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(batcher.submit(prompt(40), greedy(1)).is_err(), "never fits");
            let requests = [
                (prompt(1), 6),  // 0: one token
                (prompt(4), 6),  // 1: cancelled waiting
                (prompt(6), 12), // 2: cancelled running
                (prompt(5), 60), // 3: outgrows the cache
                (prompt(3), 6),  // 4: fails
                (prompt(7), 8),  // 5: plain
            ];
            for (p, max) in &requests {
                batcher.submit(p.clone(), greedy(*max)).unwrap();
            }
            batcher.fault = Some(4);
            assert!(batcher.cancel(1));
            let (tokens, finished) = run(&mut batcher, requests.len(), 500, |step, b| {
                if step == 2 {
                    assert!(b.cancel(2));
                }
            });
            assert!(!batcher.is_busy(), "{name}");
            assert_eq!(finished[1], Some(FinishReason::Cancelled), "{name}");
            assert!(tokens[1].is_empty());
            assert_eq!(finished[2], Some(FinishReason::Cancelled), "{name}");
            assert_eq!(finished[3], Some(FinishReason::CacheFull), "{name}");
            assert!(
                matches!(finished[4], Some(FinishReason::Error(_))),
                "{name}: {:?}",
                finished[4]
            );
            for i in [0, 5] {
                let (p, max) = &requests[i];
                assert_eq!(tokens[i], solo(&model, p, *max), "{name}: request {i}");
                assert_eq!(
                    finished[i],
                    Some(FinishReason::Length),
                    "{name}: request {i}"
                );
            }
            for i in [2, 3] {
                let (p, max) = &requests[i];
                let want = solo(&model, p, *max);
                assert!(
                    want.starts_with(&tokens[i]),
                    "{name}: request {i}: {:?} not a prefix of {want:?}",
                    tokens[i]
                );
            }
            assert!(batcher.cache.is_empty() && batcher.cache.free_blocks() == 10);
        }
    }

    /// Requests admitted together with prompts of the same length are prefilled in one
    /// batch on quantized models (long enough prompts) and still get `generate`'s tokens:
    /// greedy, seeded sampled, and next to a prompt of another length. Other models and
    /// short prompts are prefilled alone.
    #[test]
    fn equal_length_prompts_prefilled_together_get_their_solo_tokens() {
        use crate::Sampler;
        for (name, n, batched) in [
            ("qwen3_5_q4", 70, true),
            ("qwen3_moe_q4", 70, true),
            ("qwen3_5_q4", 12, false),
            ("qwen3_f32", 70, false),
        ] {
            let Some(model) = fixture_model(name) else {
                return;
            };
            let base = [3u32, 17, 92, 5, 64, 33, 101, 8, 77, 2, 45, 120, 9, 56];
            let cycled = |skip: usize, n: usize| -> Vec<u32> {
                base.iter().cycle().skip(skip).take(n).copied().collect()
            };
            let sampled = GenerateOptions {
                max_tokens: 8,
                sampler: Sampler {
                    temperature: 0.8,
                    top_p: 0.9,
                    ..Default::default()
                },
                seed: Some(11),
                ..Default::default()
            };
            let requests = [
                (cycled(0, n), greedy(8)),
                (cycled(1, n), greedy(8)),
                (cycled(2, n), sampled),
                (cycled(3, n - 3), greedy(8)),
                (cycled(4, n), greedy(8)),
            ];
            let mut batcher = Batcher::new(
                &model,
                BatcherOptions {
                    blocks: Some(256),
                    block_size: 4,
                    max_batch: requests.len(),
                    batch_invariant: true,
                    ..Default::default()
                },
            )
            .unwrap();
            for (p, o) in &requests {
                batcher.submit(p.clone(), o.clone()).unwrap();
            }
            let (tokens, _) = run(&mut batcher, requests.len(), 100, |_, _| {});
            assert_eq!(batcher.batched > 0, batched, "{name} ({n} tokens)");
            for (i, (p, o)) in requests.iter().enumerate() {
                let want: Vec<u32> = generate(&model, p, o.clone())
                    .unwrap()
                    .collect::<Result<_>>()
                    .unwrap();
                assert_eq!(tokens[i], want, "{name} ({n} tokens): request {i}");
            }
        }
    }
}
