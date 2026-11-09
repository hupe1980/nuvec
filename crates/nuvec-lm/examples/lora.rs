//! Fine-tuning, as `mlx_lm.lora` (its options, defaults and configuration files):
//! `train.jsonl`, `valid.jsonl` and `test.jsonl` in `--data`, of `{"text": ...}`,
//! `{"messages": [...], "tools": [...]}` or `{"prompt": ..., "completion": ...}` rows
//! (chat and completion rows rendered by the checkpoint's chat template); the trained
//! weights saved in mlx-lm's layout to `--adapter-path`.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example lora -- --model DIR --data DIR --train
//!     [--fine-tune-type lora|dora|full] [--optimizer adam|adamw|muon|sgd|adafactor]
//!     [--num-layers 16] [--batch-size 4] [--iters 1000] [--val-batches 25]
//!     [--learning-rate 1e-5] [--steps-per-report 10] [--steps-per-eval 200]
//!     [--resume-adapter-file FILE] [--adapter-path adapters] [--save-every 100]
//!     [--test] [--test-batches 500] [--max-seq-length 2048] [--config FILE.yaml]
//!     [--grad-checkpoint] [--grad-accumulation-steps 1] [--seed 0] [--mask-prompt]
//! ```
//!
//! As in mlx-lm, a flag overrides the `--config` file, which overrides the defaults; the
//! file also holds `lora_parameters`, `optimizer_config`, `lr_schedule` and the data keys
//! (`text_feature`, `chat_feature`, `prompt_feature`, `completion_feature`). Each report is
//! a line, and a `REPORT {json}` line with mlx-lm's callback fields
//! (`scripts/bench_lora.py` reads it).

use std::path::PathBuf;

use nuvec::io::json::Json;
use nuvec::{Device, Stream, set_default_stream};
use nuvec_lm::lora::{self, AdapterConfig, FineTuneType, LoraParams};
use nuvec_lm::tuner::{self, DataOptions, Dataset, Event, TrainingArgs};
use nuvec_lm::{ChatTemplate, Tokenizer, TokenizerConfig, load_model};
use nuvec_nn::Module;
use nuvec_nn::optim::LearningRate;

/// mlx-lm's options: the command line, then the `--config` file.
struct Options {
    args: Vec<String>,
    config: Json,
}

impl Options {
    fn flag(&self, name: &str) -> bool {
        self.args
            .iter()
            .any(|a| a == &format!("--{}", name.replace('_', "-")))
            || matches!(self.config.get(name), Some(Json::Bool(true)))
    }

    fn text(&self, name: &str) -> Option<String> {
        let flag = format!("--{}", name.replace('_', "-"));
        let given = self
            .args
            .iter()
            .position(|a| *a == flag)
            .and_then(|i| self.args.get(i + 1));
        match given {
            Some(v) => Some(v.clone()),
            None => match self.config.get(name) {
                Some(Json::Str(s)) => Some(s.clone()),
                Some(Json::Int(i)) => Some(i.to_string()),
                Some(Json::Float(f)) => Some(f.to_string()),
                _ => None,
            },
        }
    }

    fn number<T: std::str::FromStr>(&self, name: &str, default: T) -> T {
        self.text(name).map_or(default, |v| {
            v.parse()
                .unwrap_or_else(|_| panic!("--{name}: {v:?} is not a number"))
        })
    }
}

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut o = Options {
        args,
        config: Json::Null,
    };
    if let Some(path) = o.text("config") {
        println!("Loading configuration file {path}");
        o.config = tuner::read_config(path)?;
    }
    let model_dir = PathBuf::from(o.text("model").expect("--model DIR"));
    let data = PathBuf::from(o.text("data").expect("--data DIR"));
    let d = TrainingArgs::default();
    let train_args = TrainingArgs {
        batch_size: o.number("batch_size", d.batch_size),
        iters: o.number("iters", d.iters),
        val_batches: match o.number::<i64>("val_batches", 25) {
            n if n < 0 => None,
            n => Some(n as usize),
        },
        steps_per_report: o.number("steps_per_report", d.steps_per_report),
        steps_per_eval: o.number("steps_per_eval", d.steps_per_eval),
        steps_per_save: o.number("save_every", d.steps_per_save),
        max_seq_length: o.number("max_seq_length", d.max_seq_length),
        adapter_path: o.text("adapter_path").map_or(d.adapter_path, PathBuf::from),
        grad_accumulation_steps: o.number("grad_accumulation_steps", 1),
        seed: o.number("seed", 0),
        grad_checkpoint: o.flag("grad_checkpoint"),
    };
    let kind = FineTuneType::from_name(&o.text("fine_tune_type").unwrap_or_else(|| "lora".into()))?;
    let lp = o.config.get("lora_parameters");
    let lora_number = |key: &str, default: f64| {
        lp.and_then(|l| l.get(key))
            .and_then(Json::as_f64)
            .unwrap_or(default)
    };
    let d = LoraParams::default();
    let params = LoraParams {
        rank: lora_number("rank", f64::from(d.rank)) as i32,
        scale: lora_number("scale", f64::from(d.scale)) as f32,
        dropout: lora_number("dropout", f64::from(d.dropout)) as f32,
    };
    let mut config = AdapterConfig::new(o.number("num_layers", 16), params).fine_tune_type(kind);
    if let Some(keys) = lp.and_then(|l| l.get("keys")).and_then(Json::as_array) {
        config = config.keys(keys.iter().filter_map(Json::as_str));
    }

    set_default_stream(Stream::default_for(Device::gpu(0)));
    let mut model = load_model(&model_dir)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
    let eos = tuner::eos_token_id(
        &tokenizer,
        &TokenizerConfig::from_dir(&model_dir)?,
        &model.config,
    )
    .ok_or_else(|| nuvec::Error::InvalidArgument("the checkpoint names no EOS token".into()))?;
    let template = ChatTemplate::from_dir(&model_dir).ok();
    let d = DataOptions::default();
    let data_options = DataOptions {
        mask_prompt: o.flag("mask_prompt"),
        text_feature: o.text("text_feature").unwrap_or(d.text_feature),
        chat_feature: o.text("chat_feature").unwrap_or(d.chat_feature),
        prompt_feature: o.text("prompt_feature").unwrap_or(d.prompt_feature),
        completion_feature: o.text("completion_feature").unwrap_or(d.completion_feature),
        ..d
    };
    let load = |split: &str| -> nuvec::Result<Option<Dataset>> {
        let path = data.join(format!("{split}.jsonl"));
        path.exists()
            .then(|| Dataset::load_with(&path, &tokenizer, template.as_ref(), eos, &data_options))
            .transpose()
    };
    if model.config.num_hidden_layers < config.num_layers {
        return Err(nuvec::Error::InvalidArgument(format!(
            "requested to train {} layers but the model only has {} layers",
            config.num_layers, model.config.num_hidden_layers
        )));
    }
    // mlx-lm seeds MLX's generator before the adapters are drawn (`mx.random.seed`).
    nuvec::random::seed(u64::from(train_args.seed));
    lora::apply(&mut model, &config)?;
    if let Some(file) = o.text("resume_adapter_file") {
        println!("Loading fine-tuned weights from {file}");
        lora::load_weights(&file, &mut model)?;
    }
    // As mlx-lm counts: a packed quantized weight by the values it holds.
    let bits = model
        .config
        .quantization
        .as_ref()
        .map_or(32, |q| q.default.bits as usize);
    let count = |ps: Vec<(String, nuvec::Array)>| {
        ps.iter()
            .map(|(_, a)| match a.dtype() {
                nuvec::DType::U32 => a.size() * 32 / bits,
                _ => a.size(),
            })
            .sum::<usize>()
    };
    let (trainable, total) = (
        count(model.trainable_parameters()),
        count(model.parameters()),
    );
    println!(
        "Trainable parameters: {:.3}% ({:.3}M/{:.3}M)",
        100.0 * trainable as f64 / total as f64,
        trainable as f64 / 1e6,
        total as f64 / 1e6
    );

    if o.flag("train") {
        let train_set = load("train")?.ok_or_else(|| {
            nuvec::Error::InvalidArgument(format!("{}: no train.jsonl", data.display()))
        })?;
        let valid_set = load("valid")?;
        let learning_rate = match o.config.get("lr_schedule") {
            Some(s) if !matches!(s, Json::Null) => tuner::build_schedule(s)?,
            _ => LearningRate::Constant(o.number("learning_rate", 1e-5)),
        };
        let name = o.text("optimizer").unwrap_or_else(|| "adam".into());
        let optimizer_config = o
            .config
            .get("optimizer_config")
            .and_then(|c| c.get(&name.to_ascii_lowercase()))
            .cloned()
            .unwrap_or(Json::Null);
        let mut optimizer = tuner::build_optimizer(&name, &optimizer_config, learning_rate)?;
        tuner::train(
            &mut model,
            &mut optimizer,
            &train_set,
            valid_set.as_ref(),
            &train_args,
            &config,
            |event| report(event, &train_args),
        )?;
    }
    if o.flag("test") {
        let test_set = load("test")?.ok_or_else(|| {
            nuvec::Error::InvalidArgument(format!("{}: no test.jsonl", data.display()))
        })?;
        let batches = match o.number::<i64>("test_batches", 500) {
            n if n < 0 => None,
            n => Some(n as usize),
        };
        model.train(false);
        let loss = tuner::evaluate(
            &model,
            &test_set,
            train_args.batch_size,
            batches,
            train_args.max_seq_length,
        )?;
        println!(
            "Test loss {loss:.3}, Test ppl {:.3}.",
            f64::from(loss).exp()
        );
    }
    if nuvec::metal::profile::enabled() {
        let rows = nuvec::metal::profile::report();
        let total: f64 = rows.iter().map(|r| r.2).sum();
        eprintln!("GPU time: {:.1} ms", total * 1e3);
        for (kernel, n, seconds) in rows.iter().take(30) {
            eprintln!(
                "{:>9.2} ms {:>5.1}% {n:>7}  {kernel}",
                seconds * 1e3,
                100.0 * seconds / total
            );
        }
    }
    Ok(())
}

/// Prints a training event as `mlx_lm.lora` does, with a `REPORT {json}` line per report.
fn report(event: &Event, args: &TrainingArgs) {
    match event {
        Event::Train {
            iteration,
            loss,
            iterations_per_second,
            tokens_per_second,
            trained_tokens,
            peak_memory,
        } => {
            let peak = *peak_memory as f64 / 1e9;
            println!(
                "Iter {iteration}: Train loss {loss:.3}, It/sec {iterations_per_second:.3}, \
                 Tokens/sec {tokens_per_second:.3}, Trained Tokens {trained_tokens}, Peak mem {peak:.3} GB"
            );
            println!(
                "REPORT {{\"iteration\": {iteration}, \"train_loss\": {loss}, \
                 \"iterations_per_second\": {iterations_per_second}, \"tokens_per_second\": \
                 {tokens_per_second}, \"trained_tokens\": {trained_tokens}, \"peak_memory\": {peak}}}"
            );
        }
        Event::Validation {
            iteration,
            loss,
            seconds,
        } => {
            println!("Iter {iteration}: Val loss {loss:.3}, Val took {seconds:.3}s");
        }
        Event::Saved { iteration } => {
            println!(
                "Iter {iteration}: Saved adapter weights to {}",
                args.adapter_path.display()
            );
        }
    }
}
