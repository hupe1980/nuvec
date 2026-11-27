//! Greedy decoding against a reference from mlx-lm: the same prompt ids, prefilled as
//! mlx-lm's `generate_step` does (every token but the last in chunks of `prefill_step_size`,
//! the last alone), the argmax of each step's log-probabilities.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example greedy_parity -- --model mlx-community/Qwen3.5-9B-4bit \
//!     --reference ref.json [--limit 22] [--tokens 256] [--dump logits.f32]
//! ```
//!
//! The reference holds `cases`: `prompt` (ids), `tokens` (mlx-lm's greedy ids) and `top2`
//! (per step, mlx-lm's two best `[id, logprob]`), as mrun's `scripts/gen_reference.py` writes
//! it. Prints one line per case: identical, or the first differing step with mlx-lm's
//! margin between its two best tokens and nuvec's; then a summary. `--dump` writes each
//! step's logits (float32) for comparison.

use nuvec::Array;
use nuvec::io::json::{self, Json};
use nuvec_lm::load;

fn invalid(message: &str) -> nuvec::Error {
    nuvec::Error::InvalidArgument(message.into())
}

fn ids(case: &Json, field: &str, vocab: usize) -> nuvec::Result<Vec<u32>> {
    case.get(field)
        .and_then(Json::as_array)
        .filter(|a| !a.is_empty() && a.len() <= i32::MAX as usize)
        .ok_or_else(|| invalid(field))?
        .iter()
        .map(|v| {
            v.as_i64()
                .and_then(|n| u32::try_from(n).ok())
                .filter(|&n| (n as usize) < vocab)
                .ok_or_else(|| invalid("token id outside the model vocabulary"))
        })
        .collect()
}

fn count(arg: Option<String>, available: usize) -> nuvec::Result<usize> {
    let n = match arg {
        Some(s) => s
            .parse()
            .map_err(|_| invalid("expected a positive count"))?,
        None => available,
    };
    if (1..=available).contains(&n) {
        Ok(n)
    } else {
        Err(invalid("count outside reference bounds"))
    }
}

fn margins(case: &Json, tokens: &[u32], vocab: usize) -> nuvec::Result<Vec<f64>> {
    let rows = case
        .get("top2")
        .and_then(Json::as_array)
        .filter(|a| a.len() == tokens.len())
        .ok_or_else(|| invalid("missing top-two rows"))?;
    rows.iter()
        .zip(tokens)
        .map(|(row, &token)| {
            let pair = row
                .as_array()
                .filter(|v| v.len() == 2)
                .ok_or_else(|| invalid("bad top-two row"))?;
            let decode = |entry: &Json| {
                let e = entry
                    .as_array()
                    .filter(|p| p.len() == 2)
                    .ok_or_else(|| invalid("bad top-two entry"))?;
                let id = e[0]
                    .as_i64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| (n as usize) < vocab)
                    .ok_or_else(|| invalid("bad top-two id"))?;
                let lp = e[1]
                    .as_f64()
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| invalid("bad log probability"))?;
                Ok((id, lp))
            };
            let (first, a) = decode(&pair[0])?;
            let (second, b) = decode(&pair[1])?;
            if first == second || a < b || (first != token && !(a == b && second == token)) {
                return Err(invalid("top-two row disagrees with reference token"));
            }
            Ok(a - b)
        })
        .collect()
}

fn main() -> nuvec::Result<()> {
    nuvec::set_default_stream(nuvec::Stream::default_for(nuvec::Device::gpu(0)));
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model_id = value("--model").unwrap_or_else(|| "mlx-community/Qwen3.5-9B-4bit".into());
    let path = value("--reference").ok_or_else(|| invalid("--reference is required"))?;
    let bytes = std::fs::read(&path).map_err(|e| nuvec::Error::Io(format!("{path}: {e}")))?;
    let reference = json::parse(&bytes)?;
    let step = reference
        .get("prefill_step_size")
        .and_then(Json::as_i64)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid("invalid prefill_step_size"))?;
    let cases = reference
        .get("cases")
        .and_then(Json::as_array)
        .ok_or_else(|| invalid("missing cases"))?;
    let limit = count(value("--limit"), cases.len())?;
    let dir = nuvec_lm::resolve(&model_id)?;
    if let Some(snapshot) = reference.get("snapshot").and_then(Json::as_str) {
        if dir.file_name().and_then(|v| v.to_str()) != Some(snapshot) {
            return Err(invalid(
                "reference snapshot differs from the selected checkpoint",
            ));
        }
    } else if reference.get("model").and_then(Json::as_str) != Some(model_id.as_str()) {
        return Err(invalid(
            "reference model differs from the selected checkpoint",
        ));
    }
    let (model, _) = load(&model_id)?;
    let mut dump = value("--dump")
        .map(std::fs::File::create)
        .transpose()
        .map_err(|e| nuvec::Error::Io(e.to_string()))?;
    let json_lines = args.iter().any(|arg| arg == "--jsonl");
    let (mut same, mut ties) = (0, 0);
    for (i, case) in cases.iter().take(limit).enumerate() {
        let vocab = model.config.vocab_size;
        let prompt = ids(case, "prompt", vocab)?;
        let expected = ids(case, "tokens", vocab)?;
        let top2 = margins(case, &expected, vocab)?;
        let n = count(value("--tokens"), expected.len())?;
        let mut cache = model.new_cache();
        let (head, last) = prompt.split_at(prompt.len() - 1);
        for part in head.chunks(step) {
            let tokens = Array::from_slice(part, [1, part.len() as i32]);
            let _ = model.forward(&tokens, &mut cache).check()?;
            nuvec::eval(
                &cache
                    .iter()
                    .flat_map(nuvec_lm::KvCache::state)
                    .collect::<Vec<_>>(),
            )?;
        }
        let mut token = last[0];
        let mut differs = None;
        for (t, want) in expected.iter().take(n).enumerate() {
            let logits = model.forward(&Array::from_slice(&[token], [1, 1]), &mut cache);
            let last = logits.reshape([-1]);
            if let Some(dump) = &mut dump {
                use std::io::Write;
                let row = last.astype(nuvec::DType::F32).to_vec::<f32>()?;
                dump.write_all(
                    &row.iter()
                        .flat_map(|f| f.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )
                .map_err(|e| nuvec::Error::Io(e.to_string()))?;
            }
            let logprobs = last.subtract(last.logsumexp_keepdims(-1));
            token = logprobs.argmax(-1).item::<u32>()?;
            if token != *want {
                let values = logprobs.astype(nuvec::DType::F32).to_vec::<f32>()?;
                let winner = token as usize;
                let runner_up = values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != winner)
                    .map(|(_, value)| *value)
                    .fold(f32::NEG_INFINITY, f32::max);
                differs = Some((t, *want, token, top2[t], values[winner] - runner_up));
                break;
            }
        }
        match differs {
            None => {
                same += 1;
                if json_lines {
                    println!(
                        r#"{{"prompt_tokens":{},"matched_tokens":{n},"status":"pass"}}"#,
                        prompt.len()
                    );
                } else {
                    println!("case {i:2} ({:5} tokens): identical", prompt.len());
                }
            }
            Some((t, want, got, mlx_margin, margin)) => {
                if mlx_margin <= 0.125 + 1e-6 {
                    ties += 1;
                }
                if json_lines {
                    println!(
                        r#"{{"prompt_tokens":{},"matched_tokens":{t},"status":"diverged","step":{t},"expected":{want},"observed":{got},"mlx_margin":{mlx_margin},"nuvec_margin":{margin}}}"#,
                        prompt.len()
                    );
                } else {
                    println!(
                        "case {i:2} ({:5} tokens): step {t}: mlx-lm {want}, nuvec {got}; margins mlx-lm {mlx_margin:.4}, nuvec {margin:.4}",
                        prompt.len()
                    );
                }
            }
        }
    }
    if !json_lines {
        println!(
            "{same} of {limit} identical; {ties} of the others part where mlx-lm's two best are within one bf16 step"
        );
    }
    if same != limit {
        return Err(invalid("greedy token divergence"));
    }
    Ok(())
}
