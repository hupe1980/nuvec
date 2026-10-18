//! Token generation, as mlx-lm's `generate_step`: the prompt is prefilled in chunks, then
//! each step is built while the GPU runs the previous one and submitted once its token is
//! read, so the KV cache is updated in place (the evaluator owns it then).

use nuvec::{Array, DType, Result, argmax, async_eval, eval, log, multiply, negative, random};

use crate::model::{KvCache, Model};

/// How tokens are chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerateOptions {
    /// Tokens to generate at most.
    pub max_tokens: usize,
    /// 0 picks the most likely token; otherwise logits are divided by the temperature
    /// and a token is drawn (MLX's `categorical`).
    pub temperature: f32,
    /// Prompt tokens per prefill step (mlx-lm's 2048).
    pub prefill_step: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            prefill_step: 2048,
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
    let mut cache = model.new_cache();
    let (mut rest, last) = prompt.split_at(prompt.len() - 1);
    while !rest.is_empty() {
        let n = rest.len().min(options.prefill_step.max(1));
        let chunk = Array::from_slice(&rest[..n], &[1, n])?;
        // The logits are never evaluated: only the cache is computed.
        model.forward(&chunk, &mut cache)?;
        eval(&cache.iter().flat_map(KvCache::state).collect::<Vec<_>>())?;
        rest = &rest[n..];
    }
    let mut g = Generation {
        model,
        cache,
        options,
        next: None,
        produced: 0,
    };
    if options.max_tokens > 0 {
        let first = g.step(&Array::from_slice(last, &[1])?)?;
        async_eval(std::slice::from_ref(&first))?;
        g.next = Some(first);
    }
    Ok(g)
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
        let t = self.options.temperature;
        if t == 0.0 {
            return argmax(&last, Some(-1), false);
        }
        // Gumbel-max sampling of `logits / t` (MLX's `random.categorical`).
        let u = random::uniform(1e-30, 1.0, last.shape(), DType::F32, None)?;
        let gumbel = negative(&log(&negative(&log(&u)?)?)?)?;
        let scaled = multiply(last.astype(DType::F32)?, 1.0 / f64::from(t))?;
        argmax(&nuvec::add(&scaled, &gumbel)?, Some(-1), false)
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
            // Build the following step, wait for this token, then submit: the cache the
            // step writes is then finished and owned by the step alone, so it is updated
            // in place. The GPU starts on the first dispatches while the rest is encoded.
            let next = match self.step(&y) {
                Ok(next) => next,
                Err(e) => return Some(Err(e)),
            };
            let id = y.item::<u32>();
            drop(y);
            if let Err(e) = async_eval(std::slice::from_ref(&next)) {
                return Some(Err(e));
            }
            self.next = Some(next);
            return Some(id);
        }
        Some(y.item::<u32>())
    }
}
