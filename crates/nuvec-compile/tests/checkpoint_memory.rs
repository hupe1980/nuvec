//! `checkpoint` saves memory under `compile`: its backward recomputation depends on the
//! cotangents, so common-subexpression elimination cannot merge it with the forward pass.
//! A separate test binary, so that no other test allocates while peak memory is measured.

#![cfg(target_os = "macos")]

use nuvec_compile::compile_flat;
use nuvec_core::{
    Array, Device, Result, Stream, checkpoint_flat, eval, ops, set_default_stream,
    value_and_grad_flat,
};

/// One block: a wide two-layer MLP with a residual, `[B, D] → [B, 4D] → [B, D]`.
fn block(x: &Array, w1: &Array, w2: &Array) -> Result<Array> {
    let h = x.matmul(w1).tanh();
    let h = h.multiply(&h).tanh();
    x.add(h.matmul(w2).tanh()).check()
}

/// Peak GPU memory (bytes above what was resident before) of one compiled
/// `value_and_grad` step of an 8-block MLP, with or without checkpointed blocks.
fn peak(checkpointed: bool) -> usize {
    let device = nuvec_gpu::Device::shared().expect("checked");
    // Hidden activations of 16 MiB: large outputs return to the allocator as soon as
    // their consumers are encoded (smaller ones when their command buffer is committed).
    let (b, d) = (4096, 256);
    let filled = |shape: &[usize], seed: f64| -> Array {
        let n: usize = shape.iter().product();
        let t = ops::arange(0.0, n as f64, 1.0);
        ((&t * seed).sin() * 0.05f32).reshape(shape)
    };
    let x = filled(&[b, d], 0.37);
    let weights: Vec<Array> = (0..8)
        .flat_map(|i| {
            [
                filled(&[d, 4 * d], 0.11 + i as f64),
                filled(&[4 * d, d], 0.23 + i as f64),
            ]
        })
        .collect();
    let mut all = weights.clone();
    all.push(x);
    eval(&all).unwrap();
    type Block = Box<dyn Fn(&[Array]) -> Result<Vec<Array>> + Send + Sync>;
    let blocks: Vec<Block> = (0..8)
        .map(|_| {
            let f = |xs: &[Array]| Ok(vec![block(&xs[0], &xs[1], &xs[2])?]);
            if checkpointed {
                Box::new(checkpoint_flat(f)) as Box<_>
            } else {
                Box::new(f) as Box<_>
            }
        })
        .collect();
    let loss = move |xs: &[Array]| -> Result<Array> {
        let n = xs.len() - 1;
        let mut h = xs[n].clone();
        for (i, f) in blocks.iter().enumerate() {
            h = f(&[h, xs[2 * i].clone(), xs[2 * i + 1].clone()])?.remove(0);
        }
        h.sum(..).check()
    };
    // The gradient of the input: the backward pass needs the forward activations (or, with
    // checkpoints, only the blocks' inputs). (Weight gradients would add their own
    // cotangents, which the evaluation order may keep longer, whatever the checkpoints.)
    let step = compile_flat(move |xs: &[Array]| {
        let (value, mut grads) = value_and_grad_flat(&loss, &[16])(xs)?;
        grads.push(value);
        Ok(grads)
    });
    let resident = device.memory().active;
    device.reset_peak();
    let outputs = step(&all).unwrap();
    eval(&outputs).unwrap();
    let peak = device.memory().peak;
    drop(outputs);
    peak.saturating_sub(resident)
}

#[test]
fn checkpointed_blocks_lower_peak_memory_under_compile() {
    let ok = !cfg!(miri) && {
        set_default_stream(Stream::default_for(Device::gpu(0)));
        (&Array::from(1.0f32) + 1.0).eval().is_ok()
    };
    if !ok {
        assert!(
            std::env::var("NUVEC_REQUIRE_GPU").as_deref() != Ok("1"),
            "NUVEC_REQUIRE_GPU=1 but no Metal device is available"
        );
        eprintln!("skipped: no Metal device");
        return;
    }
    let plain = peak(false);
    let checkpointed = peak(true);
    assert!(
        checkpointed * 10 <= plain * 6,
        "checkpointed peak {checkpointed} bytes is not at most 60% of {plain}"
    );
}
