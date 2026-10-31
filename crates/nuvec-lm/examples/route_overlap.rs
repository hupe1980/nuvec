//! Expert overlap of speculative verify rows in a mixture-of-experts model: over one
//! forward pass of a prompt, the distinct experts of every window of `rows` consecutive
//! positions (a verification's rows), per layer, against random routing's expectation
//! (`n · (1 − (1 − k/n)^rows)`). Prints one JSON object per window size.
//!
//! ```text
//! cargo run --release -p nuvec-lm --features diagnostics --example route_overlap -- \
//!     --model mlx-community/Qwen3.6-35B-A3B-4bit --text-file prompt.txt
//! ```

use nuvec::Array;
use nuvec_lm::load;

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model_id = value("--model").expect("--model");
    let text = std::fs::read_to_string(value("--text-file").expect("--text-file"))
        .expect("readable text file");
    let (model, tokenizer) = load(&model_id)?;
    let Some(moe) = model.config.moe.clone() else {
        eprintln!("{model_id}: not a mixture-of-experts model");
        std::process::exit(2);
    };
    let ids = tokenizer.encode(&text);
    let tokens = Array::from_slice(&ids, [1, ids.len() as i32]);
    let mut cache = model.new_cache();
    let (logits, routes) = nuvec_lm::model::record_routes(|| model.forward(&tokens, &mut cache));
    logits.eval()?;
    let (n, k) = (moe.experts, moe.experts_per_token);
    let layers: Vec<Vec<u32>> = routes
        .iter()
        .map(|r| r.astype(nuvec::DType::U32).to_vec::<u32>())
        .collect::<nuvec::Result<_>>()?;
    for rows in [2usize, 5, 8] {
        let (mut distinct, mut windows) = (0usize, 0usize);
        for layer in &layers {
            let positions = layer.len() / k;
            for start in 0..positions.saturating_sub(rows - 1) {
                let mut seen: Vec<u32> = layer[start * k..(start + rows) * k].to_vec();
                seen.sort_unstable();
                seen.dedup();
                distinct += seen.len();
                windows += 1;
            }
        }
        let mean = distinct as f64 / windows.max(1) as f64;
        let random = n as f64 * (1.0 - (1.0 - k as f64 / n as f64).powi(rows as i32));
        println!(
            "{{\"rows\": {rows}, \"pairs\": {}, \"distinct_experts\": {mean:.2}, \
             \"random\": {random:.2}, \"layers\": {}, \"positions\": {}}}",
            rows * k,
            layers.len(),
            ids.len()
        );
    }
    Ok(())
}
