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

/// The worst relative differences of `name`'s operations, largest first.
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
    let worst = format!(
        "{name}: {} nodes; worst:\n{}",
        report.len(),
        worst.join("\n")
    );
    eprintln!("{worst}");
    (report[0].0, worst)
}

#[test]
fn quantized_models_match_the_cpu_operation_by_operation() {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    if cfg!(miri) || (&Array::from(1.0f32) + 1.0).eval().is_err() {
        eprintln!("skipped: no Metal device");
        return;
    }
    // Every model before failing: one run shows each one's worst operations.
    let reports: Vec<_> = ["gemma4_moe_q4", "gpt_oss_q4", "llama_gptq"]
        .into_iter()
        .map(check)
        .collect();
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
