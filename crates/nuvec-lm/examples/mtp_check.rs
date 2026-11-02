//! Checks an attached MTP head on a real checkpoint:
//!
//! 1. agreement: over a prompt, how often the head's prediction of token `i + 2` (from
//!    token `i + 1` and the model's hidden state at `i`) equals the model's own prediction
//!    at `i + 1`;
//! 2. exactness: greedy decoding with MTP drafts of 1–3 tokens gives plain greedy
//!    decoding's tokens, at the prompt's length and padded to about 4K tokens.
//!
//! `cargo run --release -p nuvec-lm --example mtp_check -- <model> <mtp.safetensors> <prompt-file>`

use nuvec_lm::{DraftPolicy, GenerateOptions, MtpDrafter, generate, speculative_generate};

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (mut model, tokenizer) = nuvec_lm::load(&args[1])?;
    model.attach_mtp(&args[2])?;
    model.config.eos_token_ids.clear();
    let text = std::fs::read_to_string(&args[3]).expect("prompt file");
    let tokens = tokenizer.encode(&text);
    let l = tokens.len() as i32;
    let input = nuvec::Array::from_slice(&tokens, [1, l]);
    let (logits, hidden) = model.forward_hidden(&input, &mut model.new_cache());
    let target = logits.argmax(-1).to_vec::<u32>()?;
    let pairs = nuvec::Array::from_slice(&tokens[1..], [1, l - 1]);
    let h = hidden.i((.., 0..l - 1, ..));
    let mtp = model.logits(&model.mtp_forward(&pairs, &h, &mut model.new_mtp_cache()));
    let mtp = mtp.argmax(-1).to_vec::<u32>()?;
    let n = tokens.len() - 2;
    let agree = (0..n).filter(|&i| mtp[i] == target[i + 1]).count();
    println!(
        "agreement: {agree}/{n} ({:.0}%)",
        100.0 * agree as f64 / n as f64
    );
    let options = GenerateOptions {
        max_tokens: 256,
        ..Default::default()
    };
    let long: Vec<u32> = tokens.iter().copied().cycle().take(4000).collect();
    for prompt in [&tokens, &long] {
        let plain: Vec<u32> =
            generate(&model, prompt, options.clone())?.collect::<nuvec::Result<_>>()?;
        for k in 1..=3 {
            let g = speculative_generate(
                &model,
                prompt,
                options.clone(),
                MtpDrafter::new(&model)?,
                DraftPolicy::Fixed(k),
            )?;
            let spec: Vec<u32> = g.collect::<nuvec::Result<_>>()?;
            let same = plain.iter().zip(&spec).take_while(|(a, b)| a == b).count();
            println!(
                "{} prompt tokens, drafts of {k}: {}",
                prompt.len(),
                match same == plain.len() && spec.len() == plain.len() {
                    true => "identical to plain greedy".to_owned(),
                    false => format!("DIFFERS at token {same} of {}", plain.len()),
                }
            );
        }
    }
    Ok(())
}
