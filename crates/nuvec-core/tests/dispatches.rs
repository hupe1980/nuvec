//! GPU dispatches of the fused paged-cache operations: one test, so no other evaluation
//! encodes while it counts; skipped without a Metal device.

mod common;

use nuvec_core::{Array, DType, Device, Stream, fast, ops, set_default_stream};

/// The dispatches evaluating `f`'s result encodes (its inputs evaluated first).
fn dispatches(inputs: &[&Array], f: impl FnOnce() -> Array) -> u64 {
    for a in inputs {
        a.eval().unwrap();
    }
    let before = nuvec_core::__gpu_dispatches();
    f().eval().unwrap();
    nuvec_core::__gpu_dispatches() - before
}

#[test]
fn fused_paged_operations_are_one_dispatch() {
    if common::require_gpu().is_none() {
        return;
    }
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let rotation = fast::Rotation {
        dims: 64,
        traditional: false,
        base: Some(10000.0),
        scale: 1.0,
        freqs: None,
    };
    let (b, h, d) = (4, 2, 64);
    // The projection's layout `[B, 1, H, D]` as `[B, H, 1, D]`.
    let x = ops::ones([b, 1, h, d], DType::BF16).transpose([0, 2, 1, 3]);
    let w = ops::ones([d], DType::BF16);
    let positions = Array::from([3i32, 9, 0, 17]);
    let slots = Array::from_slice(&[1i32, 6, 11, 30], [b, 1]);
    let pool = ops::zeros([8, h, 4, d], DType::BF16);
    let ins = [&x, &w, &positions, &slots];
    let n = dispatches(&ins, || {
        fast::norm_rope_at(&x, Some((&w, 1e-6)), rotation, &positions, None)
    });
    assert_eq!(n, 1, "norm_rope_at");
    let n = dispatches(&[&ins[..], &[&pool]].concat(), || {
        fast::norm_rope_at(&x, Some((&w, 1e-6)), rotation, &positions, (&pool, &slots))
    });
    // A pool another array holds is copied first.
    assert_eq!(n, 2, "norm_rope_at into a held pool");
    // A batch-3 prefill cache's positions 1..4 as rows `[B, T, H, D]`: no packing copy.
    let prefill = ops::ones([3, h, 6, d], DType::BF16);
    let rows = prefill.i((.., .., 1..4)).transpose([0, 2, 1, 3]);
    let slots = Array::from_slice(&(0..9).collect::<Vec<i32>>(), [3, 3]);
    let n = dispatches(&[&prefill, &slots, &pool], || {
        fast::paged_write(&pool, &rows, &slots)
    });
    assert_eq!(n, 2, "paged_write into a held pool");
    // Owned pools are written in place: one dispatch each.
    let n = {
        let owned = ops::zeros([8, h, 4, d], DType::BF16);
        owned.eval().unwrap();
        for a in [&prefill, &slots] {
            a.eval().unwrap();
        }
        let before = nuvec_core::__gpu_dispatches();
        let out = fast::paged_write(&owned, &rows, &slots);
        drop(owned);
        out.eval().unwrap();
        nuvec_core::__gpu_dispatches() - before
    };
    assert_eq!(n, 1, "paged_write in place");
    // A Qwen 3 MoE layer's glue at one token (128 experts, 8 per token, bfloat16): the
    // routing and the weighting are one kernel each, where the composition takes 7.
    let probs = ops::ones([1, 1, 128], DType::BF16).softmax(-1);
    let experts = ops::ones([1, 1, 8, 2048], DType::BF16);
    let w = ops::ones([1, 1, 8], DType::BF16);
    let n = dispatches(&[&probs], || fast::moe_route(&probs, 8, true).1);
    assert_eq!(n, 1, "moe_route");
    let n = dispatches(&[&probs], || {
        let top = probs.argpartition(-8, -1).i((.., .., -8..));
        let scores = probs.take_along_axis(&top, -1);
        scores.divide(scores.sum_keepdims(-1))
    });
    assert_eq!(
        n, 5,
        "routing composed: argsort, take, sum and its rounding, divide"
    );
    let n = dispatches(&[&experts, &w], || fast::weighted_sum(&experts, &w));
    assert_eq!(n, 1, "weighted_sum");
    let n = dispatches(&[&experts, &w], || {
        experts.multiply(w.expand_dims(-1)).sum(-2)
    });
    assert_eq!(n, 3, "weighting composed: multiply, sum and its rounding");
    // Gemma 4's post-MLP chain (norm, residual add, layer scalar, next norm): one kernel,
    // where the composition takes 4.
    let h = ops::ones([1, 1, 2816], DType::BF16);
    let f = ops::ones([1, 1, 2816], DType::BF16);
    let w = ops::ones([2816], DType::BF16);
    let scale = ops::ones([1], DType::BF16);
    let ins = [&h, &f, &w, &scale];
    let n = dispatches(&ins, || {
        let (s, n) = fast::residual_rms_norm(&h, &f, (&w, 1e-6), &scale, Some((&w, 1e-6)));
        s.add(n.unwrap())
    });
    assert_eq!(n, 2, "residual_rms_norm and the add of its outputs");
    let n = dispatches(&ins, || {
        let s = h.add(fast::rms_norm(&f, &w, 1e-6)).multiply(&scale);
        s.add(fast::rms_norm(&s, &w, 1e-6))
    });
    assert_eq!(
        n, 5,
        "composed: rms_norm, add, multiply, rms_norm, and the add"
    );
    // Gated DeltaNet's convolution at pool slots, in place: one kernel.
    let x = ops::ones([4, 1, 64], DType::BF16);
    let w = ops::ones([64, 4, 1], DType::BF16);
    let slots = Array::from([3i32, 0, 5, 1]);
    for a in [&x, &w, &slots] {
        a.eval().unwrap();
    }
    let pool = ops::zeros([6, 3, 64], DType::BF16);
    pool.eval().unwrap();
    let before = nuvec_core::__gpu_dispatches();
    let (y, next) = fast::causal_conv1d_slots(&x, &w, &pool, &slots, true);
    drop(pool);
    nuvec_core::eval(&[y, next]).unwrap();
    assert_eq!(
        nuvec_core::__gpu_dispatches() - before,
        1,
        "causal_conv1d_slots in place"
    );
}
