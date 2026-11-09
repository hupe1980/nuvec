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

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

mod options;

pub use options::{AnyOptimizer, build_optimizer, build_schedule, read_config};

use nuvec::io::json::{self, Json};
use nuvec::{Array, DType, Error, Result, Tree, ops};
use nuvec_nn::Module;
use nuvec_nn::optim::Optimizer;

use crate::Tokenizer;
#[cfg(feature = "chat")]
use crate::chat::ChatTemplate;
use crate::lora::{self, AdapterConfig};
use crate::model::Model;

/// A dataset: one token sequence per row, with the position its trained span starts at.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    rows: Vec<Row>,
}

#[derive(Debug, Clone)]
struct Row {
    tokens: Vec<u32>,
    offset: usize,
    /// What [`iterate_batches`] sorts by: mlx-lm sorts a loaded dataset (its
    /// `CacheDataset`) by `len()` of the raw row, the number of its JSON keys, so rows of
    /// one format keep their file order; tokenized rows by their length.
    sort_key: usize,
}

/// mlx-lm's data options (`create_dataset`'s configuration keys).
#[derive(Debug, Clone)]
pub struct DataOptions {
    /// Trains only the last message of a chat row, or the completion (`mask_prompt`).
    pub mask_prompt: bool,
    /// The key of a text row (`text_feature`, default `"text"`).
    pub text_feature: String,
    /// The key of a chat row's messages (`chat_feature`, default `"messages"`).
    pub chat_feature: String,
    /// The key of a completion row's prompt (`prompt_feature`, default `"prompt"`).
    pub prompt_feature: String,
    /// The key of a completion row's completion (`completion_feature`, default
    /// `"completion"`).
    pub completion_feature: String,
    /// The time templates' `strftime_now` reads; the current time when unset.
    pub now: Option<std::time::SystemTime>,
}

impl Default for DataOptions {
    fn default() -> Self {
        Self {
            mask_prompt: false,
            text_feature: "text".into(),
            chat_feature: "messages".into(),
            prompt_feature: "prompt".into(),
            completion_feature: "completion".into(),
            now: None,
        }
    }
}

/// A row format, as mlx-lm's `create_dataset` picks it from the first row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Text,
    Chat,
    Completions,
}

impl Dataset {
    /// mlx-lm's `TextDataset` from a JSONL file of `{"text": ...}` rows: each text encoded
    /// (with the tokenizer's special tokens) and ended with `eos` unless it already is.
    pub fn load(path: impl AsRef<Path>, tokenizer: &Tokenizer, eos: u32) -> Result<Self> {
        let path = path.as_ref();
        let rows = read_jsonl(path)?;
        Self::build(&rows, tokenizer, eos, &DataOptions::default(), None)
            .map_err(|e| at_path(path, e))
    }

    /// mlx-lm's `load_dataset` of a local JSONL file: [`Dataset::from_rows`] over its
    /// rows.
    #[cfg(feature = "chat")]
    pub fn load_with(
        path: impl AsRef<Path>,
        tokenizer: &Tokenizer,
        template: Option<&ChatTemplate>,
        eos: u32,
        options: &DataOptions,
    ) -> Result<Self> {
        let path = path.as_ref();
        let rows = read_jsonl(path)?;
        Self::from_rows(&rows, tokenizer, template, eos, options).map_err(|e| at_path(path, e))
    }

    /// mlx-lm's `create_dataset`: the format from the first row (a prompt and a
    /// completion; else messages; else text, under [`DataOptions`]'s keys), then per row:
    /// - **text:** the text encoded with special tokens, ended with `eos`;
    /// - **chat:** the messages (and the row's `tools`) rendered by `template` and encoded
    ///   without special tokens; with `mask_prompt`, the trained span starts after all
    ///   messages but the last, rendered with a generation prompt when the last is the
    ///   assistant's;
    /// - **completions:** the same over a user message (the prompt) and an assistant
    ///   message (the completion).
    ///
    /// Renderings pass mlx-lm's default `enable_thinking` ([`crate::chat::has_thinking`]).
    #[cfg(feature = "chat")]
    pub fn from_rows(
        rows: &[Json],
        tokenizer: &Tokenizer,
        template: Option<&ChatTemplate>,
        eos: u32,
        options: &DataOptions,
    ) -> Result<Self> {
        Self::build(rows, tokenizer, eos, options, template)
    }

    #[cfg(feature = "chat")]
    fn build(
        rows: &[Json],
        tokenizer: &Tokenizer,
        eos: u32,
        options: &DataOptions,
        template: Option<&ChatTemplate>,
    ) -> Result<Self> {
        let Some(format) = rows.first().map(|r| format_of(r, options)).transpose()? else {
            return Ok(Self::default());
        };
        if format == Format::Text {
            return text_rows(rows, tokenizer, eos, options);
        }
        let template = template.ok_or_else(|| {
            Error::InvalidArgument(
                "chat and completion data need a chat template, and the checkpoint has none".into(),
            )
        })?;
        let mut render = crate::chat::mlx_lm_options(tokenizer, crate::chat::RenderOptions::new());
        render.now = options.now;
        let mut out = Vec::with_capacity(rows.len());
        for (n, row) in rows.iter().enumerate() {
            let at = |e: Error| Error::InvalidArgument(format!("row {}: {e}", n + 1));
            let field = |key: &str| {
                row.get(key)
                    .ok_or_else(|| at(Error::InvalidArgument(format!("no {key:?}"))))
            };
            let messages: Vec<Json> = match format {
                Format::Chat => field(&options.chat_feature)?
                    .as_array()
                    .ok_or_else(|| {
                        at(Error::InvalidArgument("the messages are not a list".into()))
                    })?
                    .to_vec(),
                _ => vec![
                    message("user", field(&options.prompt_feature)?.clone()),
                    message("assistant", field(&options.completion_feature)?.clone()),
                ],
            };
            let mut opts = render.clone();
            if let Some(tools) = row.get("tools").filter(|t| !matches!(t, Json::Null)) {
                opts = opts.tools(tools.clone());
            }
            let tokens = template.encode(tokenizer, &messages, &opts).map_err(at)?;
            let offset = if options.mask_prompt {
                let last_is_assistant = format == Format::Completions
                    || messages
                        .last()
                        .and_then(|m| m.get("role"))
                        .and_then(Json::as_str)
                        == Some("assistant");
                let prompt = template
                    .encode(
                        tokenizer,
                        &messages[..messages.len().saturating_sub(1)],
                        &opts.clone().generation_prompt(last_is_assistant),
                    )
                    .map_err(at)?;
                prompt.len()
            } else {
                0
            };
            out.push(Row {
                tokens,
                offset,
                sort_key: keys(row),
            });
        }
        Ok(Self { rows: out })
    }

    #[cfg(not(feature = "chat"))]
    fn build(
        rows: &[Json],
        tokenizer: &Tokenizer,
        eos: u32,
        options: &DataOptions,
        _template: Option<&()>,
    ) -> Result<Self> {
        match rows.first().map(|r| format_of(r, options)).transpose()? {
            None => Ok(Self::default()),
            Some(Format::Text) => text_rows(rows, tokenizer, eos, options),
            Some(_) => Err(Error::InvalidArgument(
                "chat and completion data need nuvec-lm's `chat` feature".into(),
            )),
        }
    }

    /// A dataset of already tokenized rows (trained from position 0, batched by length).
    pub fn from_tokens(rows: Vec<Vec<u32>>) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(|tokens| Row {
                    sort_key: tokens.len(),
                    tokens,
                    offset: 0,
                })
                .collect(),
        }
    }

    /// The rows: their tokens and the start of their trained span.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = (&[u32], usize)> {
        self.rows.iter().map(|r| (r.tokens.as_slice(), r.offset))
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

/// The tokenizer's EOS id, as mlx-lm's datasets end text rows: `tokenizer_config.json`'s
/// `eos_token`, else the model configuration's first EOS id.
pub fn eos_token_id(
    tokenizer: &Tokenizer,
    config: &crate::TokenizerConfig,
    model: &crate::Config,
) -> Option<u32> {
    config
        .special_token("eos_token")
        .and_then(|t| tokenizer.token_id(t))
        .or_else(|| model.eos_token_ids.first().copied())
}

fn read_jsonl(path: &Path) -> Result<Vec<Json>> {
    let text =
        std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, line)| {
            json::parse(line.as_bytes())
                .map_err(|e| Error::Io(format!("{}:{}: {e}", path.display(), n + 1)))
        })
        .collect()
}

fn at_path(path: &Path, e: Error) -> Error {
    match e {
        Error::InvalidArgument(m) => Error::InvalidArgument(format!("{}: {m}", path.display())),
        e => e,
    }
}

fn format_of(row: &Json, o: &DataOptions) -> Result<Format> {
    let has = |k: &str| row.get(k).is_some();
    if has(&o.prompt_feature) && has(&o.completion_feature) {
        Ok(Format::Completions)
    } else if has(&o.chat_feature) {
        Ok(Format::Chat)
    } else if has(&o.text_feature) {
        if o.mask_prompt {
            return Err(Error::InvalidArgument(
                "Prompt masking not supported for text dataset.".into(),
            ));
        }
        Ok(Format::Text)
    } else {
        Err(Error::InvalidArgument(format!(
            "unsupported data format: the first row has none of {:?}, {:?} with {:?}, or {:?}",
            o.chat_feature, o.prompt_feature, o.completion_feature, o.text_feature
        )))
    }
}

fn text_rows(rows: &[Json], tokenizer: &Tokenizer, eos: u32, o: &DataOptions) -> Result<Dataset> {
    let mut out = Vec::with_capacity(rows.len());
    for (n, row) in rows.iter().enumerate() {
        let Some(Json::Str(text)) = row.get(&o.text_feature) else {
            return Err(Error::InvalidArgument(format!(
                "row {}: a text row needs a {:?} string",
                n + 1,
                o.text_feature
            )));
        };
        let mut tokens = tokenizer.encode(text);
        if tokens.last() != Some(&eos) {
            tokens.push(eos);
        }
        out.push(Row {
            tokens,
            offset: 0,
            sort_key: keys(row),
        });
    }
    Ok(Dataset { rows: out })
}

/// `len()` of a raw row: its number of keys.
fn keys(row: &Json) -> usize {
    row.as_object().map_or(1, <[_]>::len)
}

#[cfg(feature = "chat")]
fn message(role: &str, content: Json) -> Json {
    Json::Object(vec![
        ("role".into(), Json::Str(role.into())),
        ("content".into(), content),
    ])
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

/// mlx-lm's `iterate_batches`: rows sorted (stably) by length (for a loaded dataset, as
/// `mlx_lm.lora`'s `CacheDataset` measures it: the raw row's number of keys), cut into
/// consecutive batches of `batch_size`, visited in the order of numpy's
/// `np.random.permutation` after `np.random.seed(seed)` (a new permutation each pass with
/// `repeat`), each padded to one plus a multiple of 32 tokens at most `max_seq_length`
/// (longer rows are truncated).
pub fn iterate_batches(
    dataset: &Dataset,
    batch_size: usize,
    max_seq_length: usize,
    repeat: bool,
    seed: u32,
) -> Result<impl Iterator<Item = Batch> + '_> {
    batches(
        dataset,
        batch_size,
        max_seq_length,
        repeat,
        Rc::new(RefCell::new(NumpyRandom::new(seed))),
    )
}

/// [`iterate_batches`] drawing from a shared generator: mlx-lm's training and validation
/// batches all draw from numpy's one global state.
fn batches(
    dataset: &Dataset,
    batch_size: usize,
    max_seq_length: usize,
    repeat: bool,
    rng: Rc<RefCell<NumpyRandom>>,
) -> Result<impl Iterator<Item = Batch> + '_> {
    if batch_size == 0 || dataset.len() < batch_size {
        return Err(Error::InvalidArgument(format!(
            "the dataset must have at least batch_size={batch_size} examples but only has {}",
            dataset.len()
        )));
    }
    let mut order: Vec<usize> = (0..dataset.len()).collect();
    order.sort_by_key(|&i| dataset.rows[i].sort_key);
    let batches: Vec<Vec<usize>> = order
        .chunks_exact(batch_size)
        .map(<[usize]>::to_vec)
        .collect();
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
            // Visited front to back, as mlx-lm; popped from the end.
            pass = rng.borrow_mut().permutation(batches.len());
            pass.reverse();
            done = !repeat;
        }
    }))
}

fn make_batch(dataset: &Dataset, rows: &[usize], max_seq_length: usize) -> Batch {
    const PAD_TO: usize = 32;
    let longest = rows
        .iter()
        .map(|&i| dataset.rows[i].tokens.len())
        .max()
        .unwrap_or(0);
    let width = (1 + PAD_TO * longest.div_ceil(PAD_TO)).min(max_seq_length);
    let mut tokens = vec![0i32; rows.len() * width];
    let mut lengths = Vec::with_capacity(rows.len() * 2);
    for (r, &i) in rows.iter().enumerate() {
        let Row {
            tokens: ids,
            offset,
            ..
        } = &dataset.rows[i];
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
    // Per-token losses without the full logits (`Model::token_losses`).
    let losses = model.token_losses(&model.forward_hidden_uncached(&inputs), &targets);
    let steps = ops::arange(1, width, 1).reshape([1, -1]);
    let mask = steps
        .greater_equal(lengths.i((.., 0..1)))
        .astype(DType::F32)
        .multiply(steps.less_equal(lengths.i((.., 1..2))).astype(DType::F32));
    let ce = &losses * mask.astype(losses.dtype());
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
    /// The batch order's seed (0): numpy's `np.random.seed`, as mlx-lm.
    pub seed: u32,
    /// Recompute each block's activations in the backward pass instead of keeping them
    /// (mlx-lm's `grad_checkpoint`; off).
    pub grad_checkpoint: bool,
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
            grad_checkpoint: false,
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
///
/// The batch order is numpy's after `np.random.seed(0)`, as mlx-lm's `--test` without
/// `--train`.
pub fn evaluate(
    model: &Model,
    dataset: &Dataset,
    batch_size: usize,
    num_batches: Option<usize>,
    max_seq_length: usize,
) -> Result<f32> {
    let rng = Rc::new(RefCell::new(NumpyRandom::new(0)));
    evaluate_with(model, dataset, batch_size, num_batches, max_seq_length, rng)
}

fn evaluate_with(
    model: &Model,
    dataset: &Dataset,
    batch_size: usize,
    num_batches: Option<usize>,
    max_seq_length: usize,
    rng: Rc<RefCell<NumpyRandom>>,
) -> Result<f32> {
    let (mut sum, mut count) = (0.0f64, 0.0f64);
    let batches = batches(dataset, batch_size, max_seq_length, false, rng)?;
    for batch in batches.take(num_batches.unwrap_or(usize::MAX)) {
        let (loss, ntoks) = default_loss(model, &batch.tokens, &batch.lengths);
        let n = f64::from(ntoks.item::<f32>()?);
        sum += f64::from(loss.item::<f32>()?) * n;
        count += n;
    }
    Ok((sum / count.max(1.0)) as f32)
}

/// mlx-lm's `train`: `args.iters` steps over `train_set`, each one compiled forward,
/// backward and update of the trainable parameters with any optimizer, with gradients
/// accumulated over `grad_accumulation_steps` batches (also compiled, as mlx-lm's
/// `step(batch, prev_grad, do_update)`). Reports go to `on_event`. The trainable weights
/// are saved to `args.adapter_path` every `steps_per_save` iterations (also as
/// `NNNNNNN_adapters.safetensors`) and at the end.
pub fn train<O>(
    model: &mut Model,
    optimizer: &mut O,
    train_set: &Dataset,
    valid_set: Option<&Dataset>,
    args: &TrainingArgs,
    adapter_config: &AdapterConfig,
    mut on_event: impl FnMut(&Event),
) -> Result<()>
where
    O: Optimizer + Tree + Clone,
{
    let k = args.grad_accumulation_steps;
    if k == 0 {
        return Err(Error::InvalidArgument(
            "grad_accumulation_steps must be at least 1".into(),
        ));
    }
    let _wired = crate::WiredLimit::new();
    let mut state = (model.clone(), optimizer.clone());
    state.0.set_grad_checkpoint(args.grad_checkpoint);
    optimizer_init(&mut state);
    // The trainable leaves' positions among the model's leaves: only they are accumulated.
    let trainable: std::collections::HashSet<String> = state
        .0
        .trainable_parameters()
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    let index: std::collections::HashMap<String, usize> = state
        .0
        .paths()
        .into_iter()
        .filter(|p| trainable.contains(p))
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();
    let selected = |grads: &Model| -> Vec<Array> {
        let mut out = vec![Array::from(0.0f32); index.len()];
        grads.for_each(|p, a| {
            if let Some(&i) = index.get(p) {
                out[i] = a.clone();
            }
        });
        out
    };
    type Input = ((Array, Array), Vec<Array>);
    let value_and_grad = |m: &Model,
                          b: &(Array, Array),
                          prev: &[Array]|
     -> Result<(Array, Array, Model, Vec<Array>)> {
        let ((loss, ntoks), grads) =
            nuvec::value_and_grad(|m: &Model| default_loss(m, &b.0, &b.1))(m)?;
        let mut sum = selected(&grads);
        if !prev.is_empty() {
            sum = sum.iter().zip(prev).map(|(g, p)| g + p).collect();
        }
        Ok((loss, ntoks, grads, sum))
    };
    // Sums the gradients into `prev` (no update).
    let accumulate = nuvec::compile_mut(|(m, _): &mut (Model, O), (b, prev): &Input| {
        let (loss, ntoks, _, sum) = value_and_grad(m, b, prev)?;
        Ok::<_, Error>((loss, ntoks, sum))
    });
    // Adds `prev`, averages over the `k` batches and updates.
    let update = nuvec::compile_mut(|(m, o): &mut (Model, O), (b, prev): &Input| {
        let (loss, ntoks, grads, sum) = value_and_grad(m, b, prev)?;
        let grads = match k {
            1 => grads,
            _ => grads.map(|p, a| match index.get(p) {
                Some(&i) => sum[i].divide(k as f32),
                None => a.clone(),
            }),
        };
        o.update(m, &grads)?;
        Ok::<_, Error>((loss, ntoks))
    });
    let mut accumulated: Vec<Array> = Vec::new();
    let (mut losses, mut tokens, mut steps, mut trained, mut seconds) =
        (0.0f64, 0.0f64, 0, 0u64, 0.0f64);
    // mlx-lm seeds numpy once (`np.random.seed(args.seed)`); training and validation
    // batches draw from that one state.
    let rng = Rc::new(RefCell::new(NumpyRandom::new(args.seed)));
    let batches = batches(
        train_set,
        args.batch_size,
        args.max_seq_length,
        true,
        rng.clone(),
    )?;
    for (it, batch) in (1..=args.iters).zip(batches) {
        if let Some(valid) = valid_set
            && (it == 1 || it % args.steps_per_eval == 0 || it == args.iters)
        {
            let start = Instant::now();
            let loss = evaluate_with(
                &state.0,
                valid,
                args.batch_size,
                args.val_batches,
                args.max_seq_length,
                rng.clone(),
            )?;
            on_event(&Event::Validation {
                iteration: it,
                loss,
                seconds: start.elapsed().as_secs_f64(),
            });
        }
        let start = Instant::now();
        let input = (
            (batch.tokens, batch.lengths),
            std::mem::take(&mut accumulated),
        );
        let (loss, ntoks) = if it % k == 0 {
            update(&mut state, &input)?
        } else {
            let (loss, ntoks, sum) = accumulate(&mut state, &input)?;
            nuvec::eval(&sum)?;
            accumulated = sum;
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
            if it % args.steps_per_save == 0 {
                let numbered = args
                    .adapter_path
                    .join(format!("{it:07}_adapters.safetensors"));
                lora::save_weights(numbered, &state.0)?;
            }
            on_event(&Event::Saved { iteration: it });
        }
    }
    state.0.set_grad_checkpoint(false);
    (*model, *optimizer) = state;
    Ok(())
}

/// Creates the optimizer's state for every trainable leaf before the first step, so the
/// compiled step's state has one structure throughout.
fn optimizer_init<O: Optimizer>(state: &mut (Model, O)) {
    let (model, opt) = state;
    opt.init(model);
}

/// numpy's `RandomState` (MT19937, not its newer `Generator`), as far as `np.random.seed(n)` and
/// `np.random.permutation(n)` use it: `init_genrand` seeding, 32-bit draws, the masked
/// rejection of `random_interval` and `shuffle`'s Fisher–Yates order.
struct NumpyRandom {
    mt: [u32; 624],
    index: usize,
}

impl NumpyRandom {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                self.mt[i] = self.mt[(i + 397) % 624] ^ (y >> 1) ^ ((y & 1) * 0x9908_b0df);
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random_interval(max)`: uniform in `0..=max` by masked rejection.
    fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mask = u32::MAX >> max.leading_zeros();
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `permutation(n)`.
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u32) as usize;
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

    #[test]
    fn batch_order_is_numpys_permutation() {
        // numpy 2.5.4: `np.random.seed(0)`, then `permutation(10)`, `permutation(50)`, ...
        let mut r = NumpyRandom::new(0);
        assert_eq!(r.permutation(10), [2, 8, 4, 9, 1, 6, 7, 3, 0, 5]);
        assert_eq!(
            r.permutation(50)[..12],
            [26, 35, 44, 28, 11, 2, 34, 47, 40, 22, 4, 10]
        );
        assert_eq!(r.permutation(3), [2, 1, 0]);
        assert_eq!(NumpyRandom::new(7).permutation(5), [0, 3, 2, 1, 4]);
        assert_eq!(NumpyRandom::new(0).next_u32(), 2_357_136_044);
    }

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

    /// A LoRA model on the float32 fixture, its adapters seeded.
    fn lora_model(kind: crate::lora::FineTuneType) -> Option<(Model, AdapterConfig)> {
        let mut model = fixture_model("qwen3_f32")?;
        nuvec::random::seed(0);
        let params = LoraParams {
            rank: 2,
            scale: 1.0,
            dropout: 0.0,
        };
        let config = AdapterConfig::new(2, params).fine_tune_type(kind);
        lora::apply(&mut model, &config).unwrap();
        // Non-zero `lora_b`, so every adapter's gradient is.
        let model = model.map(|p, a| match p.ends_with("lora_b") {
            true => ops::full(a.shape(), 0.05f32),
            false => a.clone(),
        });
        Some((model, config))
    }

    fn adapter_grads(model: &Model, tokens: &Array, lengths: &Array) -> (f32, Vec<Vec<f32>>) {
        let ((loss, _), grads) =
            nuvec::value_and_grad(|m: &Model| default_loss(m, tokens, lengths))(model).unwrap();
        let trainable: std::collections::HashSet<String> = model
            .trainable_parameters()
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        let mut out = Vec::new();
        grads.for_each(|p, a| {
            if trainable.contains(p) {
                out.push(a.reshape([-1]).to_vec::<f32>().unwrap());
            }
        });
        (loss.item::<f32>().unwrap(), out)
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        let scale = b.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol * scale,
                "{what} element {i}: {x} vs {y}"
            );
        }
    }

    /// With gradient checkpointing, the loss and the adapters' gradients are the plain
    /// step's, inside and outside `compile`.
    #[test]
    fn checkpointed_blocks_give_the_same_gradients() {
        let Some((model, _)) = lora_model(crate::lora::FineTuneType::Lora) else {
            return;
        };
        let tokens = Array::from_slice(&[3i32, 7, 1, 9, 4, 2, 5, 8, 6, 1, 0, 0], [2, 6]);
        let lengths = Array::from_slice(&[0i32, 6, 0, 4], [2, 2]);
        let (loss, grads) = adapter_grads(&model, &tokens, &lengths);
        let mut ckpt = model.clone();
        ckpt.set_grad_checkpoint(true);
        let (l2, g2) = adapter_grads(&ckpt, &tokens, &lengths);
        assert!((loss - l2).abs() <= 1e-5 * loss.abs());
        for (a, b) in g2.iter().zip(&grads) {
            close(a, b, 1e-5, "checkpointed gradient");
        }
        let step = nuvec::compile(|m: &Model| {
            let ((loss, _), _) =
                nuvec::value_and_grad(|m: &Model| default_loss(m, &tokens, &lengths))(m)?;
            Ok::<_, Error>(loss)
        });
        let l3 = step(&ckpt).unwrap().item::<f32>().unwrap();
        assert!((loss - l3).abs() <= 1e-5 * loss.abs());
    }

    /// Checkpointed blocks over a quantized base (QLoRA): the packed weights are constants of
    /// the recomputation, and the loss and gradients are the plain step's.
    #[test]
    fn checkpointed_quantized_blocks_give_the_same_gradients() {
        let Some(mut model) = fixture_model("qwen2_q4") else {
            return;
        };
        nuvec::random::seed(0);
        let config = AdapterConfig::new(
            2,
            LoraParams {
                rank: 2,
                scale: 1.0,
                dropout: 0.0,
            },
        );
        lora::apply(&mut model, &config).unwrap();
        let model = model.map(|p, a| match p.ends_with("lora_b") {
            true => ops::full(a.shape(), 0.05f32),
            false => a.clone(),
        });
        let tokens = Array::from_slice(&[3i32, 7, 1, 9, 4, 2], [1, 6]);
        let lengths = Array::from_slice(&[0i32, 6], [1, 2]);
        let (loss, grads) = adapter_grads(&model, &tokens, &lengths);
        let mut ckpt = model.clone();
        ckpt.set_grad_checkpoint(true);
        let (l2, g2) = adapter_grads(&ckpt, &tokens, &lengths);
        assert!((loss - l2).abs() <= 1e-3 * loss.abs(), "{loss} vs {l2}");
        for (a, b) in g2.iter().zip(&grads) {
            close(a, b, 2e-2, "checkpointed quantized gradient");
        }
    }

    /// Two batches of 2 accumulated make the update one batch of 4 makes (equal lengths,
    /// so the per-batch means average to the batch's), with the compiled steps.
    #[test]
    fn accumulated_batches_equal_one_larger_batch() {
        let Some((model, config)) = lora_model(crate::lora::FineTuneType::Lora) else {
            return;
        };
        let data = Dataset::from_tokens(
            (0..4)
                .map(|r| (0..9).map(|t| (r * 7 + t * 3) % 50 + 1).collect())
                .collect(),
        );
        let dir = std::env::temp_dir().join(format!("nuvec-tuner-{}-accum", std::process::id()));
        let run = |batch_size: usize, k: usize| {
            let mut m = model.clone();
            let mut opt = nuvec_nn::optim::Sgd::new(0.5);
            let args = TrainingArgs {
                batch_size,
                iters: k,
                grad_accumulation_steps: k,
                steps_per_report: 1,
                steps_per_save: 1000,
                adapter_path: dir.clone(),
                ..TrainingArgs::default()
            };
            train(&mut m, &mut opt, &data, None, &args, &config, |_| {}).unwrap();
            m.trainable_parameters()
                .into_iter()
                .map(|(_, a)| a.reshape([-1]).to_vec::<f32>().unwrap())
                .collect::<Vec<_>>()
        };
        let (one, two) = (run(4, 1), run(2, 2));
        for (a, b) in two.iter().zip(&one) {
            close(a, b, 1e-5, "accumulated update");
        }
        assert!(dir.join("adapters.safetensors").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Full fine-tuning trains every parameter of the last blocks and nothing else; a
    /// quantized base is an error.
    #[test]
    fn full_fine_tuning_unfreezes_the_last_blocks() {
        let Some(mut model) = fixture_model("qwen3_f32") else {
            return;
        };
        let config = AdapterConfig::new(1, LoraParams::default())
            .fine_tune_type(crate::lora::FineTuneType::Full);
        let n = lora::apply(&mut model, &config).unwrap();
        let last = model.config.num_hidden_layers - 1;
        let trainable = model.trainable_parameters();
        assert_eq!(n, trainable.len());
        assert!(!trainable.is_empty());
        assert!(
            trainable
                .iter()
                .all(|(p, _)| p.starts_with(&format!("model.layers.{last}.")))
        );
        let Some(mut quantized) = fixture_model("qwen2_q4") else {
            return;
        };
        let err = lora::apply(&mut quantized, &config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("full-precision"), "{err}");
    }

    /// Trained weights saved with `save_weights` load back (`resume_adapter_file`); DoRA
    /// adapters (with `m`) round-trip through the adapter directory; a foreign tensor is an
    /// error.
    #[test]
    fn weights_and_dora_adapters_round_trip() {
        let Some((model, config)) = lora_model(crate::lora::FineTuneType::Dora) else {
            return;
        };
        assert!(
            model
                .trainable_parameters()
                .iter()
                .any(|(p, _)| p.ends_with(".m"))
        );
        let dir = std::env::temp_dir().join(format!("nuvec-tuner-{}-dora", std::process::id()));
        lora::save_adapters(&dir, &model, &config).unwrap();
        let text = std::fs::read_to_string(dir.join("adapter_config.json")).unwrap();
        assert!(text.contains("\"fine_tune_type\": \"dora\""), "{text}");
        let mut again = fixture_model("qwen3_f32").unwrap();
        assert_eq!(lora::load_adapters(&dir, &mut again).unwrap(), config);
        let tokens = Array::from_slice(&[3i32, 7, 1, 9], [1, 4]);
        let (a, b) = (
            model.forward_uncached(&tokens),
            again.forward_uncached(&tokens),
        );
        close(
            &b.reshape([-1]).to_vec::<f32>().unwrap(),
            &a.reshape([-1]).to_vec::<f32>().unwrap(),
            1e-6,
            "dora logits",
        );
        // Resume: the weights file onto a freshly adapted model.
        let file = dir.join("resume.safetensors");
        lora::save_weights(&file, &model).unwrap();
        let (mut fresh, _) = lora_model(crate::lora::FineTuneType::Dora).unwrap();
        let fresh_ab = fresh.map(|p, a| match p.ends_with("lora_a") {
            true => a.zeros_like(),
            false => a.clone(),
        });
        fresh = fresh_ab;
        lora::load_weights(&file, &mut fresh).unwrap();
        let c = fresh.forward_uncached(&tokens);
        close(
            &c.reshape([-1]).to_vec::<f32>().unwrap(),
            &a.reshape([-1]).to_vec::<f32>().unwrap(),
            1e-6,
            "resumed logits",
        );
        let mut other = fixture_model("qwen3_f32").unwrap();
        assert!(lora::load_weights(&file, &mut other).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `train` runs with every optimizer of `mlx_lm.lora` and a schedule, compiled.
    #[test]
    fn every_optimizer_trains() {
        let Some((model, config)) = lora_model(crate::lora::FineTuneType::Lora) else {
            return;
        };
        let data = rows(&[5, 9, 7, 6]);
        let dir = std::env::temp_dir().join(format!("nuvec-tuner-{}-opt", std::process::id()));
        for name in ["adam", "adamw", "sgd", "muon", "adafactor"] {
            let schedule =
                options::parse_yaml("name: cosine_decay\nwarmup: 1\narguments: [1e-3, 3]\n")
                    .unwrap();
            let mut opt =
                build_optimizer(name, &Json::Null, build_schedule(&schedule).unwrap()).unwrap();
            let mut m = model.clone();
            let args = TrainingArgs {
                batch_size: 2,
                iters: 3,
                steps_per_report: 3,
                steps_per_save: 1000,
                adapter_path: dir.clone(),
                ..TrainingArgs::default()
            };
            let mut losses = Vec::new();
            train(&mut m, &mut opt, &data, None, &args, &config, |e| {
                if let Event::Train { loss, .. } = e {
                    losses.push(*loss);
                }
            })
            .unwrap();
            assert!(losses.iter().all(|l| l.is_finite()), "{name}: {losses:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
