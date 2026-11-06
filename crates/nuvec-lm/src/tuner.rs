//! LoRA training, as mlx-lm's `tuner`: `text` datasets ([`Dataset`]), length-sorted padded
//! batches ([`iterate_batches`]), the masked next-token loss ([`default_loss`]) and the
//! training loop ([`train`]): a compiled step over the model and the optimizer, reports
//! of the loss, speed and peak memory, validation, and adapters saved in mlx-lm's layout.
//!
//! ```no_run
//! use nuvec_lm::lora::{self, AdapterConfig, LoraParams};
//! use nuvec_lm::tuner::{self, Dataset, TrainingArgs};
//! use nuvec_nn::Module;
//! use nuvec_nn::optim::Adam;
//!
//! # fn main() -> nuvec::Result<()> {
//! let dir = std::path::Path::new("models/Qwen3-0.6B-4bit");
//! let mut model = nuvec_lm::load_model(dir)?;
//! let tokenizer = nuvec_lm::Tokenizer::from_file(dir.join("tokenizer.json"))?;
//! let eos = model.config.eos_token_ids[0];
//! let train_set = Dataset::load("data/train.jsonl", &tokenizer, eos)?;
//! model.freeze(|_| true)?;
//! let config = AdapterConfig::new(16, LoraParams::default());
//! lora::linear_to_lora_layers(&mut model, config.num_layers, None, config.params)?;
//! let mut optimizer = Adam::new(1e-5);
//! let args = TrainingArgs::default();
//! tuner::train(&mut model, &mut optimizer, &train_set, None, &args, &config, |e| println!("{e:?}"))?;
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use std::time::Instant;

use nuvec::io::json::{self, Json};
use nuvec::{Array, DType, Error, Result, Tree, ops};
use nuvec_nn::losses::{Reduction, cross_entropy};
use nuvec_nn::optim::{Adam, Optimizer};

use crate::Tokenizer;
use crate::lora::{self, AdapterConfig};
use crate::model::Model;

/// A dataset: one token sequence per row, with the position its trained span starts at.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    rows: Vec<(Vec<u32>, usize)>,
}

impl Dataset {
    /// mlx-lm's `TextDataset` from a JSONL file of `{"text": ...}` rows: each text encoded
    /// (with the tokenizer's special tokens) and ended with `eos` unless it already is.
    pub fn load(path: impl AsRef<Path>, tokenizer: &Tokenizer, eos: u32) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let mut rows = Vec::new();
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let at = || format!("{}:{}", path.display(), n + 1);
            let row =
                json::parse(line.as_bytes()).map_err(|e| Error::Io(format!("{}: {e}", at())))?;
            let Some(Json::Str(text)) = row.get("text") else {
                return Err(Error::InvalidArgument(format!(
                    "{}: a row needs a \"text\" string (chat and completion rows need a chat \
                     template, not supported yet)",
                    at()
                )));
            };
            let mut ids = tokenizer.encode(text);
            if ids.last() != Some(&eos) {
                ids.push(eos);
            }
            rows.push((ids, 0));
        }
        Ok(Self { rows })
    }

    /// A dataset of already tokenized rows (trained from position 0).
    pub fn from_tokens(rows: Vec<Vec<u32>>) -> Self {
        Self {
            rows: rows.into_iter().map(|r| (r, 0)).collect(),
        }
    }

    /// Rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether it has no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// A batch: tokens `[B, T]` (int32, zero-padded) and per row its trained span's start and
/// its length, `[B, 2]` (int32), as mlx-lm's `iterate_batches` yields.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Tokens `[B, T]`.
    pub tokens: Array,
    /// Per row, its trained span's start and its length `[B, 2]`.
    pub lengths: Array,
}

/// mlx-lm's `iterate_batches`: rows sorted by length, cut into consecutive batches of
/// `batch_size`, visited in a seeded random order (again each pass with `repeat`), each
/// padded to one plus a multiple of 32 tokens at most `max_seq_length` (longer rows are
/// truncated).
pub fn iterate_batches(
    dataset: &Dataset,
    batch_size: usize,
    max_seq_length: usize,
    repeat: bool,
    seed: u64,
) -> Result<impl Iterator<Item = Batch> + '_> {
    if batch_size == 0 || dataset.len() < batch_size {
        return Err(Error::InvalidArgument(format!(
            "the dataset must have at least batch_size={batch_size} examples but only has {}",
            dataset.len()
        )));
    }
    let mut order: Vec<usize> = (0..dataset.len()).collect();
    order.sort_by_key(|&i| dataset.rows[i].0.len());
    let batches: Vec<Vec<usize>> = order
        .chunks_exact(batch_size)
        .map(<[usize]>::to_vec)
        .collect();
    let mut rng = SplitMix(seed);
    let mut pass: Vec<usize> = Vec::new();
    let mut done = false;
    Ok(std::iter::from_fn(move || {
        loop {
            if let Some(b) = pass.pop() {
                return Some(make_batch(dataset, &batches[b], max_seq_length));
            }
            if done {
                return None;
            }
            pass = rng.permutation(batches.len());
            done = !repeat;
        }
    }))
}

fn make_batch(dataset: &Dataset, rows: &[usize], max_seq_length: usize) -> Batch {
    const PAD_TO: usize = 32;
    let longest = rows
        .iter()
        .map(|&i| dataset.rows[i].0.len())
        .max()
        .unwrap_or(0);
    let width = (1 + PAD_TO * longest.div_ceil(PAD_TO)).min(max_seq_length);
    let mut tokens = vec![0i32; rows.len() * width];
    let mut lengths = Vec::with_capacity(rows.len() * 2);
    for (r, &i) in rows.iter().enumerate() {
        let (ids, offset) = &dataset.rows[i];
        let n = ids.len().min(max_seq_length);
        for (t, &id) in ids[..n].iter().enumerate() {
            tokens[r * width + t] = id as i32;
        }
        lengths.extend([*offset as i32, n as i32]);
    }
    Batch {
        tokens: Array::from_slice(&tokens, [rows.len() as i32, width as i32]),
        lengths: Array::from_slice(&lengths, [rows.len() as i32, 2]),
    }
}

/// mlx-lm's `default_loss`: the next-token cross-entropy over `tokens[:, 1:]`, counted
/// where `offset ≤ step ≤ length` (steps from 1), summed in float32 and divided by the
/// counted tokens. Returns the loss and the token count (both 0-D).
pub fn default_loss(model: &Model, tokens: &Array, lengths: &Array) -> (Array, Array) {
    let width = tokens.shape().get(1).copied().unwrap_or(1);
    let inputs = tokens.i((.., 0..width - 1));
    let targets = tokens.i((.., 1..width));
    let logits = model.forward_uncached(&inputs);
    let steps = ops::arange(1, width, 1).reshape([1, -1]);
    let mask = steps
        .greater_equal(lengths.i((.., 0..1)))
        .astype(DType::F32)
        .multiply(steps.less_equal(lengths.i((.., 1..2))).astype(DType::F32));
    let ce = cross_entropy(&logits, &targets, Reduction::None) * mask.astype(logits.dtype());
    let ntoks = mask.sum(..);
    (ce.astype(DType::F32).sum(..) / &ntoks, ntoks)
}

/// mlx-lm's `TrainingArgs` (its `mlx_lm.lora` defaults).
#[derive(Debug, Clone)]
pub struct TrainingArgs {
    /// Rows per batch (4).
    pub batch_size: usize,
    /// Training steps (1,000).
    pub iters: usize,
    /// Validation batches per evaluation (`None`: the whole set).
    pub val_batches: Option<usize>,
    /// Steps between training reports (10).
    pub steps_per_report: usize,
    /// Steps between validations (200).
    pub steps_per_eval: usize,
    /// Steps between adapter saves (100).
    pub steps_per_save: usize,
    /// Tokens per row at most; longer rows are truncated (2,048).
    pub max_seq_length: usize,
    /// Where `adapters.safetensors` and `adapter_config.json` are saved.
    pub adapter_path: std::path::PathBuf,
    /// Batches whose gradients make one update (1).
    pub grad_accumulation_steps: usize,
    /// The batch order's seed (0).
    pub seed: u64,
}

impl Default for TrainingArgs {
    fn default() -> Self {
        Self {
            batch_size: 4,
            iters: 1000,
            val_batches: Some(25),
            steps_per_report: 10,
            steps_per_eval: 200,
            steps_per_save: 100,
            max_seq_length: 2048,
            adapter_path: "adapters".into(),
            grad_accumulation_steps: 1,
            seed: 0,
        }
    }
}

/// What [`train`] reports, as mlx-lm's training callback receives.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Every `steps_per_report` iterations: the mean loss and the speed over them, the
    /// tokens trained so far and the peak memory (bytes, [`nuvec::memory::peak_memory`]).
    Train {
        /// The iteration reported after.
        iteration: usize,
        /// The mean training loss since the last report.
        loss: f32,
        /// Steps per second since the last report.
        iterations_per_second: f64,
        /// Counted tokens per second since the last report.
        tokens_per_second: f64,
        /// Counted tokens trained so far.
        trained_tokens: u64,
        /// Peak memory, bytes.
        peak_memory: usize,
    },
    /// At the first iteration, every `steps_per_eval` and at the last.
    Validation {
        /// The iteration it ran before.
        iteration: usize,
        /// The validation loss.
        loss: f32,
        /// Seconds it took.
        seconds: f64,
    },
    /// Adapters saved.
    Saved {
        /// The iteration saved after.
        iteration: usize,
    },
}

/// The mean loss over `num_batches` batches of `dataset` (all with `None`), weighted by
/// tokens, as mlx-lm's `evaluate`.
pub fn evaluate(
    model: &Model,
    dataset: &Dataset,
    batch_size: usize,
    num_batches: Option<usize>,
    max_seq_length: usize,
) -> Result<f32> {
    let (mut sum, mut count) = (0.0f64, 0.0f64);
    let batches = iterate_batches(dataset, batch_size, max_seq_length, false, 0)?;
    for batch in batches.take(num_batches.unwrap_or(usize::MAX)) {
        let (loss, ntoks) = default_loss(model, &batch.tokens, &batch.lengths);
        let n = f64::from(ntoks.item::<f32>()?);
        sum += f64::from(loss.item::<f32>()?) * n;
        count += n;
    }
    Ok((sum / count.max(1.0)) as f32)
}

/// mlx-lm's `train`: `args.iters` steps over `train_set`, each one compiled forward,
/// backward and update of the trainable parameters, with gradients accumulated over
/// `grad_accumulation_steps` batches. Reports go to `on_event`; adapters are saved to
/// `args.adapter_path` every `steps_per_save` iterations and at the end.
pub fn train(
    model: &mut Model,
    optimizer: &mut Adam,
    train_set: &Dataset,
    valid_set: Option<&Dataset>,
    args: &TrainingArgs,
    adapter_config: &AdapterConfig,
    mut on_event: impl FnMut(&Event),
) -> Result<()> {
    if args.grad_accumulation_steps == 0 {
        return Err(Error::InvalidArgument(
            "grad_accumulation_steps must be at least 1".into(),
        ));
    }
    let _wired = crate::WiredLimit::new();
    let loss_fn = |m: &Model, b: &(Array, Array)| -> (Array, Array) { default_loss(m, &b.0, &b.1) };
    let step = nuvec::compile_mut(|(model, opt): &mut (Model, Adam), batch: &(Array, Array)| {
        let ((loss, ntoks), grads) = nuvec::value_and_grad(|m: &Model| loss_fn(m, batch))(model)?;
        opt.update(model, &grads)?;
        Ok::<_, Error>((loss, ntoks))
    });
    let mut state = (model.clone(), optimizer.clone());
    let mut accumulated: Option<Model> = None;
    let (mut losses, mut tokens, mut steps, mut trained, mut seconds) =
        (0.0f64, 0.0f64, 0, 0u64, 0.0f64);
    let batches = iterate_batches(
        train_set,
        args.batch_size,
        args.max_seq_length,
        true,
        args.seed,
    )?;
    for (it, batch) in (1..=args.iters).zip(batches) {
        if let Some(valid) = valid_set
            && (it == 1 || it % args.steps_per_eval == 0 || it == args.iters)
        {
            let start = Instant::now();
            let loss = evaluate(
                &state.0,
                valid,
                args.batch_size,
                args.val_batches,
                args.max_seq_length,
            )?;
            on_event(&Event::Validation {
                iteration: it,
                loss,
                seconds: start.elapsed().as_secs_f64(),
            });
        }
        let start = Instant::now();
        let input = (batch.tokens, batch.lengths);
        let (loss, ntoks) = if args.grad_accumulation_steps == 1 {
            step(&mut state, &input)?
        } else {
            // Gradients summed over the batches of one update, then averaged.
            let ((loss, ntoks), grads) =
                nuvec::value_and_grad(|m: &Model| loss_fn(m, &input))(&state.0)?;
            let sum = match accumulated.take() {
                Some(acc) => {
                    let g = grads.leaves();
                    let mut k = 0;
                    acc.map(|_, a| {
                        k += 1;
                        a + &g[k - 1]
                    })
                }
                None => grads,
            };
            if it % args.grad_accumulation_steps == 0 {
                let n = args.grad_accumulation_steps as f32;
                let mean = sum.map(|_, g| g / n);
                state.1.update(&mut state.0, &mean)?;
            } else {
                accumulated = Some(sum);
            }
            (loss, ntoks)
        };
        nuvec::eval(&[&loss, &ntoks])?;
        seconds += start.elapsed().as_secs_f64();
        losses += f64::from(loss.item::<f32>()?);
        let n = ntoks.item::<f32>()? as f64;
        tokens += n;
        steps += 1;
        if it % args.steps_per_report == 0 || it == args.iters {
            trained += tokens as u64;
            on_event(&Event::Train {
                iteration: it,
                loss: (losses / steps as f64) as f32,
                iterations_per_second: steps as f64 / seconds,
                tokens_per_second: tokens / seconds,
                trained_tokens: trained,
                peak_memory: nuvec::memory::peak_memory(),
            });
            (losses, tokens, steps, seconds) = (0.0, 0.0, 0, 0.0);
        }
        if it % args.steps_per_save == 0 || it == args.iters {
            lora::save_adapters(&args.adapter_path, &state.0, adapter_config)?;
            on_event(&Event::Saved { iteration: it });
        }
    }
    (*model, *optimizer) = state;
    Ok(())
}

/// A small seeded generator for the batch order (SplitMix64).
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A random permutation of `0..n` (Fisher–Yates).
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use nuvec_nn::Module;

    use super::*;
    use crate::lora::LoraParams;
    use crate::model::tests::fixture_model;

    fn rows(lengths: &[usize]) -> Dataset {
        Dataset::from_tokens(lengths.iter().map(|&n| (1..=n as u32).collect()).collect())
    }

    /// mlx-lm's batches: sorted by length, consecutive, each row once per pass, padded to
    /// one plus a multiple of 32 (capped), truncated with its length, `(0, length)` rows.
    #[test]
    fn batches_follow_mlx_lm() {
        let data = rows(&[5, 40, 3, 70, 33, 9, 100, 2]);
        let batches: Vec<Batch> = iterate_batches(&data, 2, 64, false, 7).unwrap().collect();
        assert_eq!(batches.len(), 4);
        let mut seen: Vec<(i32, i32)> = Vec::new();
        for b in &batches {
            let lengths = b.lengths.to_vec::<i32>().unwrap();
            let pairs: Vec<(i32, i32)> = lengths.chunks(2).map(|c| (c[0], c[1])).collect();
            let longest = pairs.iter().map(|p| p.1).max().unwrap() as usize;
            let width = b.tokens.shape()[1] as usize;
            assert!(
                width == 64 || width == 1 + 32 * longest.div_ceil(32),
                "{width} for {longest}"
            );
            assert!(pairs.iter().all(|p| p.0 == 0 && p.1 as usize <= width));
            // Rows hold their tokens, then zeros.
            let t = b.tokens.to_vec::<i32>().unwrap();
            for (r, p) in pairs.iter().enumerate() {
                let row = &t[r * width..(r + 1) * width];
                assert!(
                    row[..p.1 as usize]
                        .iter()
                        .enumerate()
                        .all(|(i, &x)| x == i as i32 + 1)
                );
                assert!(row[p.1 as usize..].iter().all(|&x| x == 0));
            }
            seen.extend(pairs);
        }
        // Sorted into consecutive pairs: (2, 3), (5, 9), (33, 40), (70 → 64, 100 → 64).
        let mut lens: Vec<i32> = seen.iter().map(|p| p.1).collect();
        lens.sort();
        assert_eq!(lens, [2, 3, 5, 9, 33, 40, 64, 64]);
        // The order is seeded; repeating passes yields every batch again.
        let again: Vec<Batch> = iterate_batches(&data, 2, 64, false, 7).unwrap().collect();
        assert_eq!(
            batches[0].lengths.to_vec::<i32>().unwrap(),
            again[0].lengths.to_vec::<i32>().unwrap()
        );
        assert_eq!(
            iterate_batches(&data, 2, 64, true, 7)
                .unwrap()
                .take(9)
                .count(),
            9
        );
        assert!(iterate_batches(&rows(&[3]), 2, 64, false, 0).is_err());
    }

    /// The loss equals the mean negative log-likelihood of the counted targets, computed
    /// on the host from the logits.
    #[test]
    fn the_loss_is_the_masked_mean_nll() {
        let Some(model) = fixture_model("qwen3_f32") else {
            return;
        };
        let tokens = Array::from_slice(&[3i32, 7, 1, 9, 4, 0, 5, 2, 8, 6, 0, 0], [2, 6]);
        let lengths = Array::from_slice(&[0i32, 5, 2, 4], [2, 2]);
        let (loss, ntoks) = default_loss(&model, &tokens, &lengths);
        let logits = model
            .forward_uncached(&tokens.i((.., 0..5)))
            .astype(DType::F32);
        let v = logits.shape()[2] as usize;
        let l = logits.to_vec::<f32>().unwrap();
        let t = tokens.to_vec::<i32>().unwrap();
        let (mut sum, mut n) = (0.0f64, 0.0f64);
        for (r, (offset, length)) in [(0usize, 5usize), (2, 4)].into_iter().enumerate() {
            for s in 1..=5usize {
                if s < offset || s > length {
                    continue;
                }
                let row = &l[(r * 5 + s - 1) * v..(r * 5 + s) * v];
                let max = row.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
                let lse = max
                    + row
                        .iter()
                        .map(|&x| (x as f64 - max).exp())
                        .sum::<f64>()
                        .ln();
                sum += lse - row[t[r * 6 + s] as usize] as f64;
                n += 1.0;
            }
        }
        assert_eq!(ntoks.item::<f32>().unwrap(), n as f32);
        let want = sum / n;
        let got = f64::from(loss.item::<f32>().unwrap());
        assert!((got - want).abs() <= 1e-4 * want.abs(), "{got} vs {want}");
    }

    /// A LoRA parameter's gradient matches central differences of the loss.
    #[test]
    fn lora_gradients_match_finite_differences() {
        let Some(mut model) = fixture_model("qwen3_f32") else {
            return;
        };
        model.freeze(|_| true).unwrap();
        // A fixed LoRA initialisation, so the check is the same every run.
        nuvec::random::seed(0);
        let params = LoraParams {
            rank: 2,
            scale: 1.0,
            dropout: 0.0,
        };
        lora::linear_to_lora_layers(&mut model, 1, Some(&["self_attn.q_proj"]), params).unwrap();
        let tokens = Array::from_slice(&[3i32, 7, 1, 9, 4, 2], [1, 6]);
        let lengths = Array::from_slice(&[0i32, 5], [1, 2]);
        let loss = |m: &Model| default_loss(m, &tokens, &lengths).0;
        let (_, grads) = nuvec::value_and_grad(loss)(&model).unwrap();
        let path = model
            .trainable_parameters()
            .into_iter()
            .map(|(p, _)| p)
            .find(|p| p.contains("lora_b"))
            .expect("a LoRA b matrix");
        let g = grads.leaves()[model.paths().iter().position(|p| *p == path).unwrap()]
            .reshape([-1])
            .to_vec::<f32>()
            .unwrap()[0];
        let shifted = |eps: f32| {
            model.map(|p, a| {
                if p == path {
                    let mut v = a.reshape([-1]).to_vec::<f32>().unwrap();
                    v[0] += eps;
                    Array::from_slice(&v, a.shape())
                } else {
                    a.clone()
                }
            })
        };
        let eps = 1e-1;
        let up = loss(&shifted(eps)).item::<f32>().unwrap();
        let down = loss(&shifted(-eps)).item::<f32>().unwrap();
        let fd = (up - down) / (2.0 * eps);
        // Plus the float32 loss's rounding over 2·eps.
        let rounding = 4.0 * f32::EPSILON * up.abs().max(down.abs()) / (2.0 * eps);
        assert!((fd - g).abs() <= 2e-2 * g.abs() + rounding, "{g} vs {fd}");
    }
}
