//! Loading a safetensors checkpoint and using every tensor once on the GPU, compared against
//! MLX by `scripts/bench_load.py` (`just bench-load`). Prints `name<TAB>milliseconds`.
//!
//! The file (written by the script) is in the page cache after the first run, as when a
//! model is loaded again; the time covers opening, parsing and one GPU pass (a sum) over
//! every tensor. Also prints the GPU allocator's peak memory (mapped files are not
//! allocations).

use std::time::Instant;

use nuvec_core::{Device, Stream, eval, set_default_stream, sum};
use nuvec_io::load_safetensors;

fn main() -> nuvec_core::Result<()> {
    // `cargo test --all-targets` runs this without a file: nothing to measure.
    let Some(path) = std::env::args().nth(1).filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: load <file.safetensors> (see scripts/bench_load.py)");
        return Ok(());
    };
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let st = load_safetensors(&path)?;
        let sums = st
            .tensors
            .values()
            .map(|a| sum(a, &[], false))
            .collect::<nuvec_core::Result<Vec<_>>>()?;
        eval(&sums)?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    let peak = nuvec_gpu::Device::shared().map_or(0, |d| d.memory().peak);
    println!("load+sum\t{:.1}\t{}", best * 1e3, peak);
    Ok(())
}
