//! A long greedy decode loop on a non-main thread, for leak checks
//! (`scripts/leak_check.sh`): a thread other than the main one has no autorelease pool of
//! its own, so Objective-C objects that Metal autoreleases there pile up unless nuvec
//! drains them.
//!
//! ```text
//! cargo run --release -p nuvec-lm --example decode_loop -- [--steps 10000] [--model DIR]
//! ```
//!
//! `--model` defaults to the tiny `qwen3_f32` test fixture. The cache restarts every 256
//! steps, so memory stays flat; prints the step count and the last token.

use std::path::PathBuf;

use nuvec::{Array, Device, Stream, set_default_stream};
use nuvec_lm::load_model;

fn main() -> nuvec::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let steps: usize = value("--steps").map_or(10_000, |v| v.parse().expect("a number"));
    let dir = value("--model").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3_f32"),
        PathBuf::from,
    );
    std::thread::spawn(move || decode(&dir, steps))
        .join()
        .expect("the decode thread does not panic")
}

fn decode(dir: &std::path::Path, steps: usize) -> nuvec::Result<()> {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let model = load_model(dir)?;
    let mut cache = model.new_cache();
    let mut token = 1u32;
    for step in 0..steps {
        if step % 256 == 0 {
            cache = model.new_cache();
        }
        let x = Array::from_slice(&[token], [1, 1]);
        let logits = model.forward(&x, &mut cache);
        token = logits.argmax(-1).item::<u32>()?;
    }
    println!("{steps} steps, last token {token}");
    Ok(())
}
