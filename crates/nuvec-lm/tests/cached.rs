//! Real checkpoints from the local Hugging Face cache against mlx-lm 0.32.2 (ignored: run
//! with `cargo test --release -p nuvec-lm --test cached -- --ignored`). Each test skips with
//! a message when its model or its fixture (`scripts/lm_fixtures.py --cached`) is absent.
//!
//! Teacher-forced over mlx-lm's 32 greedy tokens: at every step mlx-lm's token is among
//! nuvec's 5 most likely, with a log-probability within 0.1 of mlx-lm's (half-precision
//! hybrid and mixture-of-experts models do not reproduce every argmax exactly).

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use nuvec::prelude::*;
use nuvec_lm::config::LayerType;
use nuvec_lm::{load_model, resolve};

struct Fixture {
    prompt: Vec<u32>,
    greedy: Vec<u32>,
    greedy_logprob: Vec<f32>,
}

fn setup(repo: &str) -> Option<(PathBuf, Fixture)> {
    let Ok(dir) = resolve(repo) else {
        eprintln!("skipped: {repo} is not in the Hugging Face cache");
        return None;
    };
    let name = repo.split('/').next_back().unwrap_or(repo);
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cached")
        .join(format!("{name}.safetensors"));
    let Ok(file) = nuvec::io::load_safetensors(&path) else {
        eprintln!(
            "skipped: no fixture {} (scripts/lm_fixtures.py --cached)",
            path.display()
        );
        return None;
    };
    let t = &file.tensors;
    Some((
        dir,
        Fixture {
            prompt: t["prompt"].to_vec::<u32>().unwrap(),
            greedy: t["greedy"].to_vec::<u32>().unwrap(),
            greedy_logprob: t["greedy_logprob"].to_vec::<f32>().unwrap(),
        },
    ))
}

fn agrees_with_mlx_lm(repo: &str) {
    let Some((dir, f)) = setup(repo) else {
        return;
    };
    let model = load_model(&dir).unwrap();
    let n = f.greedy.len();
    let tokens: Vec<u32> = f.prompt.iter().chain(&f.greedy[..n - 1]).copied().collect();
    let len = i32::try_from(tokens.len()).unwrap();
    let input = Array::from_slice(&tokens, [1, len]);
    let mut cache = model.new_cache();
    let start = i32::try_from(f.prompt.len()).unwrap() - 1;
    let logits = model
        .forward(&input, &mut cache)
        .i((0, start..))
        .astype(DType::F32);
    let vocab = usize::try_from(logits.shape()[1]).unwrap();
    let logits = logits.to_vec::<f32>().unwrap();
    let mut exact = 0;
    for (step, row) in logits.chunks(vocab).enumerate() {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum = max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
        let mut order: Vec<usize> = (0..vocab).collect();
        order.select_nth_unstable_by(4, |&a, &b| row[b].total_cmp(&row[a]));
        let top5 = &order[..5];
        let want = f.greedy[step] as usize;
        let best = *top5
            .iter()
            .max_by(|&&a, &&b| row[a].total_cmp(&row[b]))
            .unwrap();
        exact += usize::from(best == want);
        assert!(
            top5.contains(&want),
            "{repo} step {step}: mlx-lm's token {want} is not in nuvec's top 5 {top5:?}"
        );
        let diff = (row[want] - log_sum - f.greedy_logprob[step]).abs();
        assert!(
            diff <= 0.1,
            "{repo} step {step}: log-probability differs by {diff}"
        );
    }
    println!("{repo}: nuvec's greedy token equals mlx-lm's at {exact} of {n} steps");
}

#[test]
#[ignore = "needs mlx-community/Qwen3.8-27B-4bit in the Hugging Face cache"]
fn qwen3_8_27b_loads_on_the_qwen3_5_path() {
    let Ok(dir) = resolve("mlx-community/Qwen3.8-27B-4bit") else {
        eprintln!("skipped: mlx-community/Qwen3.8-27B-4bit is not in the Hugging Face cache");
        return;
    };
    let model = load_model(&dir).unwrap();
    let kinds = &model.config.layer_types;
    let linear = kinds.iter().filter(|k| **k == LayerType::Linear).count();
    let attention = kinds.iter().filter(|k| **k == LayerType::Attention).count();
    assert_eq!((linear, attention), (48, 16));
}

#[test]
#[ignore = "needs mlx-community/Qwen3.8-27B-4bit and its fixture"]
fn qwen3_8_27b_agrees_with_mlx_lm() {
    agrees_with_mlx_lm("mlx-community/Qwen3.8-27B-4bit");
}

#[test]
#[ignore = "needs mlx-community/Meta-Llama-3.1-8B-Instruct-4bit and its fixture"]
fn llama_3_1_8b_agrees_with_mlx_lm() {
    agrees_with_mlx_lm("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit");
}
