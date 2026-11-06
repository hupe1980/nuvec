//! LoRA fine-tuning, as `mlx_lm.lora --train` (its options and defaults): `train.jsonl`
//! and, if present, `valid.jsonl` of `{"text": ...}` rows in `--data`; adapters saved in
//! mlx-lm's layout to `--adapter-path`.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example lora -- --model DIR --data DIR [--iters 1000]
//!     [--batch-size 4] [--num-layers 16] [--learning-rate 1e-5] [--max-seq-length 2048]
//!     [--steps-per-report 10] [--steps-per-eval 200] [--val-batches 25] [--save-every 100]
//!     [--grad-accumulation-steps 1] [--seed 0] [--adapter-path adapters]
//! ```
//!
//! Each report is a line, and a `REPORT {json}` line with mlx-lm's callback fields
//! (`scripts/bench_lora.py` reads it).

use std::path::PathBuf;

use nuvec::{Device, Stream, set_default_stream};
use nuvec_lm::lora::{self, AdapterConfig, LoraParams};
use nuvec_lm::tuner::{self, Dataset, Event, TrainingArgs};
use nuvec_lm::{Tokenizer, load_model};
use nuvec_nn::Module;
use nuvec_nn::optim::Adam;

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let number =
        |flag: &str, default: usize| value(flag).map_or(default, |v| v.parse().expect(flag));
    let model_dir = PathBuf::from(value("--model").expect("--model DIR"));
    let data = PathBuf::from(value("--data").expect("--data DIR"));
    let defaults = TrainingArgs::default();
    let train_args = TrainingArgs {
        batch_size: number("--batch-size", defaults.batch_size),
        iters: number("--iters", defaults.iters),
        val_batches: match number("--val-batches", 25) {
            0 => None,
            n => Some(n),
        },
        steps_per_report: number("--steps-per-report", defaults.steps_per_report),
        steps_per_eval: number("--steps-per-eval", defaults.steps_per_eval),
        steps_per_save: number("--save-every", defaults.steps_per_save),
        max_seq_length: number("--max-seq-length", defaults.max_seq_length),
        adapter_path: value("--adapter-path").map_or(defaults.adapter_path, PathBuf::from),
        grad_accumulation_steps: number("--grad-accumulation-steps", 1),
        seed: number("--seed", 0) as u64,
    };
    let learning_rate: f32 =
        value("--learning-rate").map_or(1e-5, |v| v.parse().expect("a number"));
    let num_layers = number("--num-layers", 16);

    set_default_stream(Stream::default_for(Device::gpu(0)));
    let mut model = load_model(&model_dir)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))?;
    let eos = model.config.eos_token_ids[0];
    let train_set = Dataset::load(data.join("train.jsonl"), &tokenizer, eos)?;
    let valid_path = data.join("valid.jsonl");
    let valid_set = valid_path
        .exists()
        .then(|| Dataset::load(&valid_path, &tokenizer, eos))
        .transpose()?;
    model.freeze(|_| true)?;
    let config = AdapterConfig::new(num_layers, LoraParams::default());
    lora::linear_to_lora_layers(&mut model, num_layers, None, config.params)?;
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
    let mut optimizer = Adam::new(learning_rate);
    tuner::train(
        &mut model,
        &mut optimizer,
        &train_set,
        valid_set.as_ref(),
        &train_args,
        &config,
        |event| match event {
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
                    train_args.adapter_path.display()
                );
            }
        },
    )?;
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
