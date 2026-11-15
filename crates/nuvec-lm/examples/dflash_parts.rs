//! The parts of a DFlash speculative step, timed apart (after a prompt): a one-row decode
//! step, an 8-row verification (row-exact and not), and one draft.
//!
//! cargo run --release -p nuvec-lm --example dflash_parts -- --model mlx-community/Qwen3-8B-4bit \
//!     --dflash-dir DIR [--dflash-bits 4] [--only draft|verify] [--rows 8,16] [--draft 7]
//!
//! With `NUVEC_PROFILE=1` it prints the GPU time per kernel at the end.

use std::time::Instant;

use nuvec::{Array, eval};
use nuvec_lm::{DFlashDrafter, Drafter, load};

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model_id = value("--model").unwrap_or_else(|| "mlx-community/Qwen3-8B-4bit".into());
    let dir = value("--dflash-dir").expect("--dflash-dir");
    let (model, tokenizer) = load(&model_id)?;
    let mut drafter = match value("--dflash-bits") {
        Some(b) => DFlashDrafter::load_quantized(
            &dir,
            &model,
            nuvec::Quant::affine(64, b.parse().unwrap()),
        )?,
        None => DFlashDrafter::load(&dir, &model)?,
    };
    let ids = drafter.hidden_layers().unwrap();
    let prompt = tokenizer.encode("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?");
    let mut cache = model.new_cache();
    let tokens = Array::from_slice(&prompt, [1, prompt.len() as i32]);
    let (_, features) = model.forward_layers(&tokens, &mut cache, &ids);
    drafter.observe(&model, &tokens, Some(&features))?;
    eval(&cache.iter().flat_map(|c| c.state()).collect::<Vec<_>>())?;
    let time = |name: &str, f: &mut dyn FnMut() -> nuvec::Result<()>| -> nuvec::Result<()> {
        f()?;
        let n = 10;
        let t = Instant::now();
        for _ in 0..n {
            f()?;
        }
        println!(
            "{name}: {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        );
        Ok(())
    };
    let step =
        |rows: usize, exact: bool, cache: &mut Vec<nuvec_lm::KvCache>| -> nuvec::Result<()> {
            let input = Array::from_slice(&vec![prompt[0]; rows], [1, rows as i32]);
            let run = |c: &mut Vec<nuvec_lm::KvCache>| {
                let (logits, f) = model.forward_layers(&input, c, &ids);
                eval(&[&logits, &f])
            };
            match exact {
                true => nuvec::with_row_exact(|| run(cache))?,
                false => run(cache)?,
            }
            for c in cache.iter_mut() {
                c.trim(rows)?;
            }
            Ok(())
        };
    let only = value("--only").unwrap_or_default();
    if only.is_empty() {
        time("decode step (1 row)", &mut || step(1, false, &mut cache))?;
        time("verify (8 rows, row-exact)", &mut || {
            step(8, true, &mut cache)
        })?;
    }
    if only.is_empty() || only == "verify" {
        let rows = value("--rows").unwrap_or_else(|| "8,16".into());
        for r in rows.split(',').map(|r| r.parse::<usize>().expect("--rows")) {
            let name = format!("verify ({r} rows, MLX routing)");
            time(&name, &mut || step(r, false, &mut cache))?;
        }
        if only == "verify" {
            return report();
        }
    }
    let context = prompt.clone();
    let n = value("--draft").map_or(7, |d| d.parse().expect("--draft"));
    time(&format!("draft ({n} tokens)"), &mut || {
        let d = drafter.draft(&model, &context, n)?;
        eval(&[&d.tokens])
    })?;
    report()
}

/// Prints the GPU time per kernel (with `NUVEC_PROFILE=1`), most expensive first.
fn report() -> nuvec::Result<()> {
    let mut report = nuvec::metal::profile::report();
    report.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (name, calls, s) in report.iter().take(15) {
        println!("{:9.2} ms {calls:6}  {name}", s * 1e3);
    }
    Ok(())
}
