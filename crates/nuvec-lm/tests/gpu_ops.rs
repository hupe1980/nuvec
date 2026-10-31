//! Each operation of a model's forward pass on the GPU against the same operation on the
//! CPU, fed the GPU's inputs: names the kernel behind a wrong logit on a GPU generation
//! the fixtures were not recorded on. Skipped without a Metal device.

#![cfg(target_os = "macos")]

use std::collections::HashSet;
use std::path::PathBuf;

use nuvec::graph::{from_primitive, id, inputs, output_index, primitive};
use nuvec::{Array, DType, Device, Stream, set_default_stream, with_stream};
use nuvec_lm::load_model;

/// The forward pass's nodes, inputs first.
fn nodes(out: &Array) -> Vec<Array> {
    let (mut order, mut seen) = (Vec::new(), HashSet::new());
    let mut stack = vec![(out.clone(), false)];
    while let Some((a, done)) = stack.pop() {
        if done {
            order.push(a);
            continue;
        }
        if !seen.insert(id(&a)) {
            continue;
        }
        stack.push((a.clone(), true));
        for i in inputs(&a) {
            stack.push((i, false));
        }
    }
    order
}

/// The worst relative difference of `name`'s operations from the CPU's, or of its logits
/// from mlx-lm's, and a report of the worst operations.
fn check(name: &str) -> (f32, String) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let prompt = nuvec::io::load_safetensors(dir.join("expected.safetensors"))
        .unwrap()
        .tensors["prompt"]
        .to_vec::<u32>()
        .unwrap();
    let model = load_model(&dir).unwrap();
    let tokens = Array::from_slice(&prompt, [1, prompt.len()]);
    let out = model.forward(&tokens, &mut model.new_cache());
    // Recorded before anything runs: evaluating a node drops its primitive and inputs.
    let graph: Vec<_> = nodes(&out)
        .into_iter()
        .filter_map(|a| Some((primitive(&a)?, inputs(&a), a)))
        .collect();
    let mut report = Vec::new();
    for (p, ins, gpu) in graph {
        gpu.eval().unwrap();
        let cpu = with_stream(Device::cpu(), || from_primitive(p.clone(), ins.clone()));
        let cpu = match cpu.and_then(|o| {
            let c = o[output_index(&gpu)].clone();
            c.eval().map(|()| c)
        }) {
            Ok(c) => c,
            Err(e) => {
                report.push((f32::INFINITY, format!("{p:?}: no CPU result: {e}")));
                continue;
            }
        };
        let g = gpu.astype(DType::F32).to_vec::<f32>().unwrap();
        let c = cpu.astype(DType::F32).to_vec::<f32>().unwrap();
        let scale = c.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
        let err = g
            .iter()
            .zip(&c)
            .map(|(g, c)| if g == c { 0.0 } else { (g - c).abs() })
            .fold(0.0f32, f32::max)
            / scale;
        let shapes: Vec<_> = ins
            .iter()
            .map(|i| (i.dtype(), i.shape().to_vec()))
            .collect();
        let mut what = format!("{p:?}");
        what.truncate(160);
        report.push((
            if err.is_nan() { f32::INFINITY } else { err },
            format!("{what} {shapes:?} -> {:?} {:?}", gpu.dtype(), gpu.shape()),
        ));
    }
    report.sort_by(|a, b| b.0.total_cmp(&a.0));
    let worst: Vec<_> = report
        .iter()
        .take(8)
        .map(|(e, w)| format!("{e:.3e} {w}"))
        .collect();
    let want = nuvec::io::load_safetensors(dir.join("expected.safetensors"))
        .unwrap()
        .tensors["logits"]
        .to_vec::<f32>()
        .unwrap();
    let got = out.astype(DType::F32).to_vec::<f32>().unwrap();
    let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    let logits = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
        / scale;
    let worst = format!(
        "{name}: {} nodes, logits {logits:e} from mlx-lm's; worst:\n{}",
        report.len(),
        worst.join("\n")
    );
    eprintln!("{worst}");
    (report[0].0.max(logits), worst)
}

const MODELS: [&str; 3] = ["gemma4_moe_q4", "gpt_oss_q4", "llama_gptq"];

/// The largest difference of `name`'s logits, evaluated as one graph, from mlx-lm's, over
/// the largest logit; with `hold`, every intermediate keeps a handle (nothing is written
/// in place).
fn whole(name: &str, hold: bool) -> f32 {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let expected = nuvec::io::load_safetensors(dir.join("expected.safetensors"))
        .unwrap()
        .tensors;
    let prompt = expected["prompt"].to_vec::<u32>().unwrap();
    let want = expected["logits"].to_vec::<f32>().unwrap();
    let model = load_model(&dir).unwrap();
    let tokens = Array::from_slice(&prompt, [1, prompt.len()]);
    let out = model.forward(&tokens, &mut model.new_cache());
    let held = if hold { nodes(&out) } else { Vec::new() };
    let got = out.astype(DType::F32).to_vec::<f32>().unwrap();
    drop(held);
    let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    got.iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(
            0.0f32,
            |m, d| if d.is_nan() { f32::INFINITY } else { m.max(d) },
        )
        / scale
}

/// Run by [`whole_graphs_match_mlx_lm`] in a child process with `NUVEC_PROFILE=1` (one
/// dispatch per command buffer, each waited for).
#[test]
#[ignore = "run by whole_graphs_match_mlx_lm"]
fn serialized_child() {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    for name in MODELS {
        println!("serialized {name} {:e}", whole(name, false));
    }
}

/// The forward pass as one graph, with and without in-place writes and serialized, against
/// mlx-lm's logits: which of them goes wrong where single operations are right.
#[test]
fn whole_graphs_match_mlx_lm() {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    if cfg!(miri) || (&Array::from(1.0f32) + 1.0).eval().is_err() {
        eprintln!("skipped: no Metal device");
        return;
    }
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "serialized_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("NUVEC_PROFILE", "1")
        .output()
        .unwrap();
    let serialized = String::from_utf8_lossy(&child.stdout);
    let mut lines: Vec<String> = serialized
        .lines()
        .filter(|l| l.starts_with("serialized "))
        .map(String::from)
        .collect();
    if lines.is_empty() {
        lines.push(format!(
            "serialized: no result\n{}",
            String::from_utf8_lossy(&child.stderr)
        ));
    }
    let mut bad = false;
    for name in MODELS {
        let (plain, held) = (whole(name, false), whole(name, true));
        bad |= plain > 2e-2 || held > 2e-2;
        lines.push(format!("plain {name} {plain:e}; held {name} {held:e}"));
    }
    let bad = bad
        || lines.iter().any(|l| {
            l.split_whitespace()
                .last()
                .and_then(|v| v.parse::<f32>().ok())
                .is_none_or(|v| v > 2e-2)
        });
    eprintln!("{}", lines.join("\n"));
    assert!(
        !bad,
        "whole graphs differ from mlx-lm:\n{}",
        lines.join("\n")
    );
}

#[test]
fn quantized_models_match_the_cpu_operation_by_operation() {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    if cfg!(miri) || (&Array::from(1.0f32) + 1.0).eval().is_err() {
        eprintln!("skipped: no Metal device");
        return;
    }
    // Every model before failing: one run shows each one's worst operations.
    let reports: Vec<_> = MODELS.into_iter().map(check).collect();
    // bf16 rounds to 0.4% of a value; sums of a few rounded products stay within 1%.
    let bad: Vec<_> = reports
        .iter()
        .filter(|(e, _)| *e > 2e-2)
        .map(|(_, w)| w.as_str())
        .collect();
    assert!(
        bad.is_empty(),
        "GPU operations differ from the CPU:\n{}",
        bad.join("\n")
    );
}
