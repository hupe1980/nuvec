//! Worker for scripts/bench_sparse_mla.py. Input and layer weights are shared with MLX.
use nuvec::{DType, memory};
use nuvec_lm::sparse_mla::{SparseMla, SparseMlaConfig};
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let usage =
        || nuvec::Error::InvalidArgument("usage: sparse_mla_bench WEIGHTS PREFIX ROWS".into());
    if args.len() != 4 {
        return Err(usage().into());
    }
    let prefix: i32 = args[2].parse().map_err(|_| usage())?;
    let rows: i32 = args[3].parse().map_err(|_| usage())?;
    let mut arrays = nuvec::io::load_safetensors(&args[1])?.tensors;
    let hidden = arrays.remove("hidden").ok_or_else(usage)?;
    let end = prefix.checked_add(rows).ok_or_else(usage)?;
    if prefix < 1 || rows < 1 || end > hidden.shape_n::<3>()?[1] {
        return Err(usage().into());
    }
    let layer = SparseMla::new(SparseMlaConfig::default(), arrays)?;
    let (_, state) = layer.append(&hidden.i((.., ..prefix)), &layer.new_session())?;
    state.eval()?;
    let x = hidden.i((.., prefix..end));
    x.eval()?;
    memory::clear_cache();
    println!("READY");
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let command = line?;
        if command == "stages" {
            let start = Instant::now();
            let (out, _, picks) = layer.append_with_selection(&x, &state)?;
            picks.eval()?;
            let index_ms = start.elapsed().as_secs_f64() * 1e3;
            let selected = Instant::now();
            out.eval()?;
            let remainder_ms = selected.elapsed().as_secs_f64() * 1e3;
            println!("{{\"index_ms\":{index_ms},\"remainder_ms\":{remainder_ms}}}");
            io::stdout().flush()?;
            continue;
        }
        if command != "measure" {
            break;
        }
        memory::reset_peak_memory();
        let start = Instant::now();
        let (out, next) = layer.append(&x, &state)?;
        out.eval()?;
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        let peak = memory::peak_memory();
        let sample = out.i((0, -1, ..8)).astype(DType::F32).to_vec::<f32>()?;
        println!(
            "{{\"ms\":{elapsed},\"peak_bytes\":{peak},\"history_bytes\":{},\"sample\":{sample:?}}}",
            next.nbytes()
        );
        io::stdout().flush()?;
    }
    Ok(())
}
