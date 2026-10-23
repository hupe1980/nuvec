//! The row-independence contract: a product or row-wise operation over B ≤ R rows
//! (`ROW_INDEPENDENT_ROWS`) gives each row the bits of the same row computed alone, and a
//! vector-attention call with 2–8 query rows gives each row the bits of the one-query call
//! over its causal key prefix (the serial rule).
//!
//! One module per covered family; other features (speculative decoding, FP formats) add
//! modules here.

#![allow(clippy::type_complexity)]
#![cfg(target_os = "macos")]

mod common;

use nuvec_core::ops;
use nuvec_core::*;

/// Fixed inputs in about [-1, 1], one pattern per seed.
fn pattern(shape: &[i32], dtype: DType, seed: u64) -> Array {
    let n: i32 = shape.iter().product();
    let t = ops::arange(0, n, 1).astype(DType::F32);
    let x = (&t * (0.7381 + seed as f64 * 0.0137)).sin();
    let x = &x * (&t * (0.0071 * seed as f64 + 0.013) + 0.3).cos();
    x.reshape(shape).astype(dtype)
}

/// The bits of every element (floats widened exactly to float32, integers to int64).
fn bits(a: &Array) -> Vec<u64> {
    match a.dtype().category() {
        Category::Float => a
            .astype(DType::F32)
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        _ => a
            .astype(DType::I64)
            .to_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|v| v as u64)
            .collect(),
    }
}

/// Asserts each row (along `axis`) of `batched` has the bits of `solo(row)`.
fn assert_rows(name: &str, batched: &Array, axis: i32, solo: impl Fn(i32) -> Array) {
    let n = batched.shape()[axis as usize];
    for r in 0..n {
        let got = batched.take(&Array::from(r), axis);
        let want = solo(r);
        let want = want.reshape(got.shape());
        assert_eq!(got.dtype(), want.dtype(), "{name}: dtype");
        assert!(
            bits(&got) == bits(&want),
            "{name}: row {r} of {n} differs from the row alone"
        );
    }
}

fn gpu() -> bool {
    let ok = common::require_gpu().is_some();
    set_default_stream(Stream::default_for(Device::gpu(0)));
    ok
}

mod verify_rows {
    use super::*;
    use nuvec_core::fast::{AttentionKernel, AttentionOptions, plan_attention};

    /// For `lq` causal verify rows over `lk` keys, every row equals the one-query call over
    /// its causal prefix, bit for bit. Returns the plan's dispatch count.
    fn check(hq: i32, hkv: i32, d: i32, lq: i32, lk: i32, dtype: DType, sinks: bool) -> i32 {
        let q = pattern(&[1, hq, lq, d], dtype, 1);
        // Keys and values as a prefix of a larger cache (a view with an offset stride).
        let cap = lk + 13;
        let kc = pattern(&[1, hkv, cap, d], dtype, 2);
        let vc = pattern(&[1, hkv, cap, d], dtype, 3);
        let k = kc.i((.., .., ..lk));
        let v = vc.i((.., .., ..lk));
        let s = pattern(&[hq], dtype, 4);
        let opts = AttentionOptions::causal().sinks(sinks.then_some(&s));
        let composed = matches!(
            plan_attention(
                &[1, hq, lq, d],
                &[1, hkv, lk, d],
                &[1, hkv, lk, d],
                dtype,
                &opts
            )
            .unwrap()
            .kernel,
            AttentionKernel::Composed
        );
        let out = fast::scaled_dot_product_attention(&q, &k, &v, 0.0625, &opts);
        out.eval().unwrap();
        let name = format!("{dtype} hq={hq} hkv={hkv} d={d} lq={lq} lk={lk} sinks={sinks}");
        if !composed {
            assert_rows(&name, &out, 2, |r| {
                let keys = lk - (lq - 1 - r);
                fast::scaled_dot_product_attention(
                    &q.i((.., .., r..r + 1)),
                    &k.i((.., .., ..keys)),
                    &v.i((.., .., ..keys)),
                    0.0625,
                    &AttentionOptions::causal().sinks(sinks.then_some(&s)),
                )
            });
        }
        let plan = plan_attention(
            &[1, hq, lq, d],
            &[1, hkv, lk, d],
            &[1, hkv, lk, d],
            dtype,
            &opts,
        )
        .unwrap();
        match plan.kernel {
            AttentionKernel::Vector { dispatches, .. } => dispatches,
            // Rows needing the one-key-per-step variant (float32, head size 256) beyond one
            // vector threadgroup compose, without the serial rule.
            AttentionKernel::Composed => {
                assert!(
                    dtype == DType::F32 && d == 256 && lq * hq / hkv > 32,
                    "{name}"
                );
                0
            }
            other => panic!("{name}: verify rows should run the vector kernels, got {other:?}"),
        }
    }

    #[test]
    fn verify_rows_equal_serial_decode() {
        if !gpu() {
            return;
        }
        let mut split = 0;
        for &(hq, hkv) in &[(8, 1), (24, 2), (16, 1)] {
            for d in [64, 128, 256] {
                for (e, edge) in [1024, 8192, 32768, 65536].into_iter().enumerate() {
                    for lq in [2, 3, 5, 8] {
                        // Rows on both sides of the edge, and all past it; the dtypes
                        // rotate over the grid.
                        for (i, lk) in [edge + lq / 2, edge + lq].into_iter().enumerate() {
                            let dtype = [DType::F16, DType::BF16, DType::F32]
                                [(d as usize / 64 + lq as usize + e + i) % 3];
                            let sinks = (d + lq) % 3 == 0;
                            if check(hq, hkv, d, lq, lk, dtype, sinks) > 1 {
                                split += 1;
                            }
                        }
                    }
                }
            }
        }
        // Rows straddling the one/two-pass switch or a key-block bucket edge run in
        // separate dispatches.
        assert!(split > 0, "no call needed several dispatches");
    }

    #[test]
    fn straddling_rows_split_dispatches() {
        if !gpu() {
            return;
        }
        // Rows on both sides of the 1,024-key one/two-pass switch (M-series classes `d`
        // and `s`) or of the 8,192-key bucket.
        let class = nuvec_gpu::Device::shared()
            .and_then(|d| d.info().class)
            .unwrap_or('g');
        if matches!(class, 'd' | 's') {
            assert!(check(16, 2, 128, 4, 1025, DType::F16, false) > 1);
        }
        assert!(check(16, 2, 128, 4, 8194, DType::BF16, false) > 1);
        // Equal configurations share one dispatch.
        assert_eq!(check(16, 2, 128, 4, 3000, DType::BF16, false), 1);
    }
}

mod dense {
    use super::*;

    /// Rows `B = 2..=R` of `x` (contiguous, a transposed view, or a slice with an
    /// offset) times `w` (`[N, K]` used transposed, as a linear layer, or `[K, N]`).
    #[test]
    fn dense_rows_equal_one_row() {
        if !gpu() {
            return;
        }
        let r = ROW_INDEPENDENT_ROWS as i32;
        assert!(r >= 8);
        // (K, N): layer shapes, a long thin head, and odd sizes (tails of every tile).
        let shapes = [
            (1024, 3072),
            (4096, 1024),
            (64, 151_936),
            (1000, 77),
            (1, 33),
            (4097, 513),
        ];
        for (i, &(k, n)) in shapes.iter().enumerate() {
            let dtype = [DType::F32, DType::F16, DType::BF16][i % 3];
            let w = pattern(&[n, k], dtype, 10 + i as u64);
            let wt = pattern(&[k, n], dtype, 20 + i as u64);
            for b in 2..=r {
                // Three layouts of the rows.
                let base = pattern(&[b + 3, k], dtype, b as u64);
                let layouts = [
                    base.i(..b),
                    base.i(3..3 + b),
                    pattern(&[k, b], dtype, 7).t(),
                ];
                for (l, x) in layouts.iter().enumerate() {
                    let name = format!("{dtype} K={k} N={n} B={b} layout {l}");
                    super::check_one_by_one(&name, x, |x| x.matmul(&w.t()));
                    super::check_one_by_one(&format!("{name} (w [K, N])"), x, |x| x.matmul(&wt));
                }
            }
        }
        // Batched rows: `[2, B, K] · [K, N]`.
        let w = pattern(&[256, 96], DType::BF16, 3);
        let x = pattern(&[2, 5, 256], DType::BF16, 4);
        let y = x.matmul(&w);
        for i in 0..2 {
            super::check_one_by_one("batched", &x.i(i), |x| x.matmul(&w));
            assert_eq!(bits(&y.i(i)), bits(&x.i(i).matmul(&w)));
        }
    }

    /// With a bias (`addmm`), rows are independent too.
    #[test]
    fn addmm_rows_equal_one_row() {
        if !gpu() {
            return;
        }
        let w = pattern(&[512, 384], DType::F16, 5);
        let bias = pattern(&[384], DType::F16, 6);
        for b in 2..=ROW_INDEPENDENT_ROWS as i32 {
            let x = pattern(&[b, 512], DType::F16, 7 + b as u64);
            super::check_one_by_one(&format!("addmm B={b}"), &x, |x| {
                ops::addmm(&bias, x, &w, 0.5, 2.0)
            });
        }
    }
}

/// Each row (along axis 0 of a 2-D `x`) of `product(x)` has the bits of `product` of that
/// row alone.
fn check_one_by_one(name: &str, x: &Array, product: impl Fn(&Array) -> Array) {
    let batched = product(x);
    batched.eval().unwrap();
    let n = x.shape()[0];
    for r in 0..n {
        let one = product(&x.i(r..r + 1));
        assert!(
            bits(&batched.i(r..r + 1)) == bits(&one),
            "{name}: row {r} of {n} differs from the row alone"
        );
    }
}

mod affine {
    use super::*;

    /// Quantized products over B = 2..=R rows: every bit width and group size, half
    /// activations, aligned and unaligned K, transposed and untransposed weights, rows
    /// contiguous, sliced (an offset) and transposed (non-contiguous).
    #[test]
    fn quantized_rows_equal_one_row() {
        if !gpu() {
            return;
        }
        let mut case = 0u64;
        for bits in [1, 2, 3, 4, 5, 6, 8] {
            for gs in [32, 64, 128] {
                case += 1;
                let dtype = if case.is_multiple_of(2) {
                    DType::F16
                } else {
                    DType::BF16
                };
                // K aligned to the fast kernels (2,048) or not (a multiple of the group
                // size only); N with and without a partial last block.
                let k = if case.is_multiple_of(3) {
                    2048 + gs
                } else {
                    2048
                };
                let n = if case.is_multiple_of(2) { 520 } else { 512 };
                let w = pattern(&[n, k], dtype, case);
                let (wq, s, b) = w.quantize(gs, bits);
                // Untransposed weights are quantized along N.
                let wn = pattern(&[k, 640], dtype, case + 100);
                let (wqn, sn, bn) = wn.quantize(gs, bits);
                for rows in (2..=ROW_INDEPENDENT_ROWS as i32)
                    .filter(|r| [2, 3, 4, 7, 8, 13, 16].contains(r))
                {
                    let base = pattern(&[rows + 2, k], dtype, rows as u64);
                    let layouts = [
                        base.i(..rows),
                        base.i(2..2 + rows),
                        pattern(&[k, rows], dtype, 9).t(),
                    ];
                    for (l, x) in layouts.iter().enumerate() {
                        let name =
                            format!("{dtype} bits {bits} gs {gs} K={k} N={n} B={rows} layout {l}");
                        check_one_by_one(&name, x, |x| {
                            x.quantized_matmul(&wq, &s, &b, true, gs, bits)
                        });
                        if l == 0 {
                            check_one_by_one(&format!("{name} untransposed"), x, |x| {
                                x.quantized_matmul(&wqn, &sn, &bn, false, gs, bits)
                            });
                        }
                    }
                }
            }
        }
    }

    /// A tiled quantized product over a batch of items (`[B, L, K]`, as a batched prefill
    /// of same-length prompts) gives each item the bits of its product alone (`[1, L, K]`):
    /// the split-K partitions are counted per item. Shapes with few output tiles (split
    /// alone), many (no split) and between.
    #[test]
    fn quantized_batch_items_equal_one_item() {
        if !gpu() {
            return;
        }
        for (case, (l, n, k)) in [
            (40, 256, 4096),
            (194, 1024, 4096),
            (100, 512, 2048),
            (300, 4096, 4096),
        ]
        .into_iter()
        .enumerate()
        {
            let w = pattern(&[n, k], DType::BF16, case as u64 + 30);
            let (wq, s, b) = w.quantize(64, 4);
            for items in [2, 4, 8] {
                let x = pattern(&[items, l, k], DType::BF16, case as u64 + 40);
                let y = x.quantized_matmul(&wq, &s, &b, true, 64, 4);
                assert_rows(&format!("B={items} L={l} N={n} K={k}"), &y, 0, |r| {
                    x.i(r..r + 1).quantized_matmul(&wq, &s, &b, true, 64, 4)
                });
            }
        }
    }
}

mod gather {
    use super::*;

    /// Gathered (mixture-of-experts) products over B = 2..=R tokens with top-2 and top-8
    /// routing: each output row (one token and expert) has the bits of the same row run
    /// alone, with unsorted indices (`[B, top-k]`) and with rows sorted by expert.
    #[test]
    fn gathered_rows_equal_one_row() {
        if !gpu() {
            return;
        }
        let (e, n, k) = (16, 96, 512);
        let wf = pattern(&[e, n, k], DType::BF16, 1);
        let (q4, s4, b4) = wf.quantize(64, 4);
        let (q8, s8, b8) = wf.quantize(32, 8);
        let dense = pattern(&[e, k, n], DType::F16, 2);
        let products: [(&str, Box<dyn Fn(&Array, &Array, bool) -> Array>); 3] = [
            (
                "gather_qmm 4-bit",
                Box::new(|x: &Array, ids: &Array, sorted| {
                    ops::gather_qmm(x, &q4, &s4, &b4, None, Some(ids), true, 64, 4, sorted)
                }),
            ),
            (
                "gather_qmm 8-bit",
                Box::new(|x: &Array, ids: &Array, sorted| {
                    ops::gather_qmm(x, &q8, &s8, &b8, None, Some(ids), true, 32, 8, sorted)
                }),
            ),
            (
                "gather_mm",
                Box::new(|x: &Array, ids: &Array, sorted| {
                    ops::gather_mm(&x.astype(DType::F16), &dense, None, Some(ids), sorted)
                }),
            ),
        ];
        for (name, product) in &products {
            for topk in [2, 8] {
                for b in [2, 5, ROW_INDEPENDENT_ROWS as i32] {
                    // Unsorted: tokens `[B, 1, 1, K]` against `[B, top-k]` experts.
                    let base = pattern(&[b + 1, 1, 1, k], DType::BF16, b as u64);
                    let x = base.i(1..);
                    let ids: Vec<u32> = (0..b * topk)
                        .map(|i| ((i * 7 + i / topk) % e) as u32)
                        .collect();
                    let ids = Array::from_slice(&ids, [b, topk]);
                    let out = product(&x, &ids, false);
                    out.eval().unwrap();
                    for t in 0..b {
                        let one = product(&x.i(t..t + 1), &ids.i(t..t + 1), false);
                        assert!(
                            bits(&out.i(t..t + 1)) == bits(&one),
                            "{name} top-{topk} B={b}: token {t} differs"
                        );
                    }
                    // Sorted: one row per (token, expert), sorted by expert.
                    let rows = b * topk;
                    let mut sorted: Vec<u32> = (0..b * topk).map(|i| (i % e) as u32).collect();
                    sorted.sort_unstable();
                    let sorted = Array::from(sorted);
                    let xs = pattern(&[rows, 1, k], DType::BF16, 3 + b as u64);
                    let out = product(&xs, &sorted, true);
                    out.eval().unwrap();
                    for r in 0..rows {
                        let one = product(&xs.i(r..r + 1), &sorted.i(r..r + 1), true);
                        assert!(
                            bits(&out.i(r..r + 1)) == bits(&one),
                            "{name} sorted, {rows} rows: row {r} differs"
                        );
                    }
                }
            }
        }
    }
}

mod rowwise {
    use super::*;

    /// The row-wise operations of a decode step, over B = 2..=R sequences: each
    /// sequence's result has the bits of the same sequence run alone.
    #[test]
    fn decode_step_ops_are_row_independent() {
        if !gpu() {
            return;
        }
        let d = 128;
        for b in [2, 3, ROW_INDEPENDENT_ROWS as i32] {
            let dt = if b % 2 == 0 { DType::BF16 } else { DType::F16 };
            let x = pattern(&[b, d], dt, 1);
            let r = pattern(&[b, d], dt, 2);
            let w = pattern(&[d], dt, 3);
            check_one_by_one(&format!("rms_norm B={b}"), &x, |xs| {
                fast::rms_norm(xs, &w, 1e-6)
            });
            for row in 0..b {
                let (h, n) = fast::add_rms_norm(&x, &r, &w, 1e-6);
                let (h1, n1) = fast::add_rms_norm(&x.i(row..row + 1), &r.i(row..row + 1), &w, 1e-6);
                assert!(
                    bits(&h.i(row..row + 1)) == bits(&h1),
                    "add_rms_norm h B={b} row {row}"
                );
                assert!(
                    bits(&n.i(row..row + 1)) == bits(&n1),
                    "add_rms_norm B={b} row {row}"
                );
            }
            check_one_by_one(&format!("element-wise B={b}"), &x, |xs| {
                (xs * 2.0 + 1.0).exp().multiply(xs.sigmoid())
            });
            // RoPE with one offset per sequence, on `[B, H, 1, D]`.
            let q = pattern(&[b, 8, 1, d], dt, 4);
            let offsets = Array::from((0..b).map(|i| 100 + 37 * i).collect::<Vec<i32>>());
            let rope = fast::rope(&q, d, false, Some(1e6), 1.0, &offsets, None);
            for row in 0..b {
                let one = fast::rope(
                    &q.i(row..row + 1),
                    d,
                    false,
                    Some(1e6),
                    1.0,
                    &offsets.i(row..row + 1),
                    None,
                );
                assert!(
                    bits(&rope.i(row..row + 1)) == bits(&one),
                    "rope B={b} row {row}"
                );
            }
            // norm_rope (norm, RoPE and a cache write) at one position.
            let rot = fast::Rotation {
                dims: d,
                traditional: false,
                base: Some(1e6),
                scale: 1.0,
                freqs: None,
            };
            let normed = fast::norm_rope(&q, Some((&w, 1e-6)), rot, 7, None);
            for row in 0..b {
                let one = fast::norm_rope(&q.i(row..row + 1), Some((&w, 1e-6)), rot, 7, None);
                assert!(
                    bits(&normed.i(row..row + 1)) == bits(&one),
                    "norm_rope B={b} row {row}"
                );
            }
            // Vector attention over equal key lengths.
            let k = pattern(&[b, 2, 3000, d], dt, 5);
            let v = pattern(&[b, 2, 3000, d], dt, 6);
            let opts = fast::AttentionOptions::causal();
            let att = fast::scaled_dot_product_attention(&q, &k, &v, 0.088, &opts);
            for row in 0..b {
                let i = row..row + 1;
                let one = fast::scaled_dot_product_attention(
                    &q.i(i.clone()),
                    &k.i(i.clone()),
                    &v.i(i),
                    0.088,
                    &opts,
                );
                assert!(
                    bits(&att.i(row..row + 1)) == bits(&one),
                    "attention B={b} row {row}"
                );
            }
            // The gated delta rule over per-sequence states, and through a state pool.
            let (hk, hv, dk, dv) = (2, 4, 64, 32);
            let gq = pattern(&[b, 1, hk, dk], dt, 7) * 0.1;
            let gk = pattern(&[b, 1, hk, dk], dt, 8) * 0.1;
            let gv = pattern(&[b, 1, hv, dv], dt, 9);
            let g = (pattern(&[b, 1, hv], DType::F32, 10) * 0.1).sigmoid();
            let beta = pattern(&[b, 1, hv], DType::F32, 11).sigmoid();
            let state = pattern(&[b, hv, dv, dk], DType::F32, 12) * 0.1;
            let (y, s) = fast::gated_delta(&gq, &gk, &gv, &g, &beta, &state);
            let pool = pattern(&[b + 3, hv, dv, dk], DType::F32, 13) * 0.1;
            let slots = Array::from((0..b).map(|i| (i * 2 + 1) % (b + 3)).collect::<Vec<i32>>());
            let (ys, _) = fast::gated_delta_slots(&gq, &gk, &gv, &g, &beta, &pool, &slots);
            for row in 0..b {
                let i = row..row + 1;
                let (y1, s1) = fast::gated_delta(
                    &gq.i(i.clone()),
                    &gk.i(i.clone()),
                    &gv.i(i.clone()),
                    &g.i(i.clone()),
                    &beta.i(i.clone()),
                    &state.i(i.clone()),
                );
                assert!(
                    bits(&y.i(i.clone())) == bits(&y1),
                    "gated_delta y B={b} row {row}"
                );
                assert!(
                    bits(&s.i(i.clone())) == bits(&s1),
                    "gated_delta state B={b} row {row}"
                );
                let (ys1, _) = fast::gated_delta_slots(
                    &gq.i(i.clone()),
                    &gk.i(i.clone()),
                    &gv.i(i.clone()),
                    &g.i(i.clone()),
                    &beta.i(i.clone()),
                    &pool,
                    &slots.i(i.clone()),
                );
                assert!(
                    bits(&ys.i(i)) == bits(&ys1),
                    "gated_delta_slots B={b} row {row}"
                );
            }
        }
    }
}
