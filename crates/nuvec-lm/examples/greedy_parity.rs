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

fn ids(value: &Json) -> Vec<u32> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|v| v.as_i64().expect("an id") as u32)
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
    let path = value("--reference").expect("--reference");
    let bytes = std::fs::read(&path).map_err(|e| nuvec::Error::Io(format!("{path}: {e}")))?;
    let reference = json::parse(&bytes)?;
    let step = reference
        .get("prefill_step_size")
        .and_then(Json::as_i64)
        .unwrap_or(2048) as usize;
    let cases = reference
        .get("cases")
        .and_then(Json::as_array)
        .expect("cases");
    let limit = value("--limit").map_or(cases.len(), |v| v.parse().expect("a number"));
    let (model, _) = load(&model_id)?;
    let dump = value("--dump").map(|p| {
        std::cell::RefCell::new(std::fs::File::create(p).expect("--dump: a writable path"))
    });

    let (mut same, mut ties) = (0, 0);
    for (i, case) in cases.iter().take(limit).enumerate() {
        let prompt = ids(case.get("prompt").expect("prompt"));
        let expected = ids(case.get("tokens").expect("tokens"));
        let top2 = case.get("top2").and_then(Json::as_array).expect("top2");
        let n = value("--tokens").map_or(expected.len(), |v| v.parse().expect("a number"));
        let mut cache = model.new_cache();
        let (head, last) = prompt.split_at(prompt.len() - 1);
        for part in head.chunks(step) {
            let tokens = Array::from_slice(part, [1, part.len() as i32]);
            let _ = model.forward(&tokens, &mut cache);
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
            if let Some(dump) = &dump {
                use std::io::Write;
                let row = last.astype(nuvec::DType::F32).to_vec::<f32>()?;
                dump.borrow_mut()
                    .write_all(
                        &row.iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect::<Vec<u8>>(),
                    )
                    .map_err(|e| nuvec::Error::Io(e.to_string()))?;
            }
            let logprobs = last.subtract(last.logsumexp_keepdims(-1));
            let values = logprobs.astype(nuvec::DType::F32).to_vec::<f32>()?;
            let mut best = [0usize, 1];
            if values[1] > values[0] {
                best = [1, 0];
            }
            for (k, &v) in values.iter().enumerate().skip(2) {
                if v > values[best[0]] {
                    best = [k, best[0]];
                } else if v > values[best[1]] {
                    best[1] = k;
                }
            }
            let values = [values[best[0]], values[best[1]]];
            token = best[0] as u32;
            if token != *want {
                let pair = top2[t].as_array().expect("top2 step");
                let lp = |k: usize| pair[k].as_array().expect("pair")[1].as_f64().expect("lp");
                differs = Some((t, *want, token, lp(0) - lp(1), values[0] - values[1]));
                break;
            }
        }
        match differs {
            None => {
                same += 1;
                println!("case {i:2} ({:5} tokens): identical", prompt.len());
            }
            Some((t, want, got, mlx_margin, margin)) => {
                if mlx_margin <= 0.125 + 1e-6 {
                    ties += 1;
                }
                println!(
                    "case {i:2} ({:5} tokens): step {t}: mlx-lm {want}, nuvec {got}; margins mlx-lm {mlx_margin:.4}, nuvec {margin:.4}",
                    prompt.len()
                );
            }
        }
    }
    println!(
        "{same} of {limit} identical; {ties} of the others part where mlx-lm's two best are within one bf16 step"
    );
    Ok(())
}
