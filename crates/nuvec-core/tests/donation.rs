//! In-place `at(..).set(..)` (a slice update) on the GPU (buffer donation): effective when the evaluation
//! owns the base, invisible otherwise. A separate test binary, so that no other test
//! allocates while peak memory is measured. Every test skips when no Metal device is
//! available.

#![cfg(target_os = "macos")]

mod common;

use nuvec_core::ops;
use nuvec_core::*;

/// Serialises the tests that measure peak memory (tests run on parallel threads).
static PEAK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Bytes the GPU allocator has held for arrays at most since the last call.
fn gpu_peak_since_reset() -> usize {
    nuvec_gpu::Device::shared().map_or(0, |d| {
        let peak = d.memory().peak;
        d.reset_peak();
        peak
    })
}

#[test]
fn at_set_writes_in_place_when_it_owns_its_base() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let _serial = PEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    with_stream(Device::gpu(0), || {
        let n = 4 << 20; // 16 MiB of float32
        let mut x = ops::zeros([64, n / 64], DType::F32);
        x.eval().unwrap();
        gpu_peak_since_reset();
        let resident = nuvec_gpu::Device::shared().map_or(0, |d| d.memory().active);
        for i in 0..64 {
            let v = ops::full([1, n / 64], i as f32);
            // `at` takes `x` by value: the update owns its base and writes into it.
            x = x.at((i..i + 1, ..)).set(&v);
            x.eval().unwrap();
        }
        // Each step reused the buffer: a copy would have held two 16 MiB buffers at once.
        let peak = gpu_peak_since_reset();
        assert!(
            peak < resident + n * 4 / 2,
            "peak {peak}, before {resident}"
        );
        let v = x.to_vec::<f32>().unwrap();
        assert!((0..64).all(|i| v[i * n / 64] == i as f32 && v[(i + 1) * n / 64 - 1] == i as f32));
    });
}

/// `x.clone().at(..).set(..)` updates a copy: `x` keeps its values, on a held base and on a
/// chain of updates.
#[test]
fn at_set_on_a_clone_leaves_the_original_unchanged() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    with_stream(Device::gpu(0), || {
        let x = ops::zeros([4, 8], DType::F32);
        x.eval().unwrap();
        let y = x.clone().at((.., 0..4)).set(1.0);
        assert_eq!(x.to_vec::<f32>().unwrap(), [0.0; 32]);
        let v = y.to_vec::<f32>().unwrap();
        assert!((0..32).all(|i| v[i] == if i % 8 < 4 { 1.0 } else { 0.0 }));
        // Unevaluated, and updated again through the clone: still unchanged.
        let x = ops::arange(0, 32, 1).astype(DType::F32).reshape([4, 8]);
        let mut y = x.clone();
        for i in 0..4 {
            y = y.at((i, ..)).set(-1.0);
        }
        assert_eq!(y.to_vec::<f32>().unwrap(), [-1.0; 32]);
        let expected: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(x.to_vec::<f32>().unwrap(), expected);
    });
}

/// The decode order (mlx-lm's): each step is submitted while the previous one is still in
/// flight, and a small result of the previous step is read afterwards. The base is then
/// in flight, held only by pending work, and is still written in place.
#[test]
fn slice_update_writes_in_place_into_bases_still_in_flight() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let _serial = PEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    with_stream(Device::gpu(0), || {
        let n = 4 << 20; // 16 MiB of float32
        let row = |i: usize| (i..i + 1, ..);
        let mut a = ops::zeros([64, n / 64], DType::F32);
        a.eval().unwrap();
        // Work that keeps each step's command buffer running while the next is built,
        // without allocating anything sizeable (a reduction of a resident 64 MiB array).
        // (Computed on the GPU, so that it is resident in the GPU allocator when measured.)
        let big = ops::zeros([16 << 20], DType::F32) + 1.0f32 / (16 << 20) as f32;
        big.eval().unwrap();
        gpu_peak_since_reset();
        let resident = nuvec_gpu::Device::shared().map_or(0, |d| d.memory().active);
        let mut token: Option<Array> = None;
        for i in 0..64 {
            let update = ops::full([1, n / 64], i as f32);
            a = a.at(row(i)).set(&update);
            // A small result of this step, read only after the next step is submitted.
            let busy = big.sum(..);
            let next = a.i(row(i)).sum(..) + busy;
            async_eval(&[a.clone(), next.clone()]).unwrap();
            if let Some(t) = token.replace(next) {
                let want = (i - 1) as f32 * (n / 64) as f32 + 1.0;
                assert!((t.item::<f32>().unwrap() - want).abs() < 1.0);
            }
        }
        // A copy per step would have held two 16 MiB buffers at once.
        let peak = gpu_peak_since_reset();
        assert!(
            peak < resident + n * 4 / 2,
            "peak {peak}, before {resident}"
        );
        let v = a.to_vec::<f32>().unwrap();
        assert!((0..64).all(|i| v[i * n / 64] == i as f32 && v[(i + 1) * n / 64 - 1] == i as f32));
    });
}

#[test]
fn slice_update_never_changes_arrays_that_are_still_held() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    with_stream(Device::gpu(0), || {
        let base = Array::from([0.0f32; 8]);
        base.eval().unwrap();
        let view = base.i(0..4);
        let one = Array::from([1.0f32; 2]);
        // The base is held (and viewed): the update copies.
        let b = base.clone().at(2..4).set(&one);
        assert_eq!(b.to_vec::<f32>().unwrap(), [0., 0., 1., 1., 0., 0., 0., 0.]);
        assert_eq!(base.to_vec::<f32>().unwrap(), [0.0; 8]);
        assert_eq!(view.to_vec::<f32>().unwrap(), [0.0; 4]);
        // A chain submitted with async_eval (bases in flight) gives the same values as
        // one evaluated step by step.
        let mut chained = ops::zeros([16], DType::F32);
        for i in 0..16 {
            let v = ops::full([1], i as f32);
            chained = chained.at(i..i + 1).set(&v);
            async_eval(std::slice::from_ref(&chained)).unwrap();
        }
        let expected: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(chained.to_vec::<f32>().unwrap(), expected);
    });
}

#[test]
fn slice_update_differentiates_on_the_gpu() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    with_stream(Device::gpu(0), || {
        // d/dx sum(slice_update(x, 3·x[0:2], [2:4])²) at x = [1, 2, 3, 4].
        let f = |x: &Array| -> Result<Array> {
            let part = x.i(0..2) * 3.0;
            let y = x.clone().at(2..4).set(&part);
            Ok((&y * &y).sum(..))
        };
        let x = array![1.0, 2.0, 3.0, 4.0];
        let g = grad(f)(&x).unwrap().to_vec::<f32>().unwrap();
        // y = [x0, x1, 3x0, 3x1]: dy²/dx0 = 2x0 + 18x0, dy²/dx1 = 2x1 + 18x1, x2 and x3
        // are overwritten.
        assert_eq!(g, [20.0, 40.0, 0.0, 0.0]);
    });
}

/// On the CPU, a strided update (a transpose) is read in place: the update allocates only
/// its result, not a contiguous copy of the update.
#[test]
fn cpu_slice_update_reads_strided_updates_in_place() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let _serial = PEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    with_stream(Device::cpu(), || {
        let (rows, cols) = (32, 4096);
        let base = ops::zeros([64, cols], DType::F32);
        let source = ops::arange(0.0, (rows * cols) as f64, 1.0).reshape([cols, rows]);
        base.eval().unwrap();
        source.eval().unwrap();
        let update = source.t(); // [rows, cols], strided
        gpu_peak_since_reset();
        let resident = nuvec_gpu::Device::shared().map_or(0, |d| d.memory().active);
        let out = base.slice_update(&update, (0..rows, ..));
        out.eval().unwrap();
        // The result (a copy of the base: the CPU does not donate) and nothing else; a
        // contiguous copy of the update would add another 512 KiB.
        let peak = gpu_peak_since_reset();
        let (base_bytes, update_bytes) = (64 * cols * 4, rows * cols * 4);
        assert!(
            peak < resident + base_bytes + update_bytes / 2,
            "peak {peak}, before {resident}"
        );
        let v = out.to_vec::<f32>().unwrap();
        assert_eq!((v[0], v[1], v[cols]), (0.0, 32.0, 1.0));
        assert_eq!(v[rows * cols], 0.0);
    });
}

/// Bases produced earlier in the same evaluation are written in place too: growing a cache
/// by `zeros`, then the old part, then a new segment, in one evaluation, holds one cache.
#[test]
fn slice_update_writes_in_place_into_bases_of_the_same_evaluation() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let _serial = PEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    with_stream(Device::gpu(0), || {
        let (heads, d, old_len, seg_len, cap) = (8, 128, 1024, 1024, 2304);
        let part = |len: usize, start: f64| {
            let n = (heads * len * d) as f64;
            ops::arange(start, start + n, 1.0).reshape([1, heads, len, d])
        };
        let (old, seg) = (part(old_len, 0.0), part(seg_len, 1e7));
        old.eval().unwrap();
        seg.eval().unwrap();
        let at = |a: usize, b: usize| (.., .., a..b, ..);
        gpu_peak_since_reset();
        let resident = nuvec_gpu::Device::shared().map_or(0, |dv| dv.memory().active);
        // (The intermediate is not bound to a name: a shadowed binding would keep it alive.)
        let grown = ops::zeros([1, heads, cap, d], DType::F32)
            .at(at(0, old_len))
            .set(&old)
            .at(at(old_len, old_len + seg_len))
            .set(&seg);
        grown.eval().unwrap();
        let cache = heads * cap * d * 4;
        let peak = gpu_peak_since_reset();
        assert!(
            peak < resident + cache * 3 / 2,
            "peak {peak}, before {resident}, cache {cache}"
        );
        let want = ops::concatenate(
            &[
                old.clone(),
                seg.clone(),
                ops::zeros([1, heads, cap - old_len - seg_len, d], DType::F32),
            ],
            2,
        );
        assert_eq!(
            grown.to_vec::<f32>().unwrap(),
            want.to_vec::<f32>().unwrap()
        );

        // A chain long enough to span several command buffers (committed ones included).
        let n = 4 << 20;
        gpu_peak_since_reset();
        let resident = nuvec_gpu::Device::shared().map_or(0, |dv| dv.memory().active);
        let mut a = ops::zeros([64, n / 64], DType::F32);
        for i in 0..64 {
            // A scalar update (broadcast), so that only the cache takes memory.
            a = a.at((i..i + 1, ..)).set(Array::from(i as f32));
        }
        a.eval().unwrap();
        let peak = gpu_peak_since_reset();
        assert!(
            peak < resident + n * 4 * 3 / 2,
            "chain: peak {peak}, before {resident}"
        );
        let v = a.to_vec::<f32>().unwrap();
        assert!((0..64).all(|i| v[i * n / 64] == i as f32 && v[(i + 1) * n / 64 - 1] == i as f32));
    });
}

/// Within one evaluation, a base another array still holds is never written, and a reader
/// encoded before an in-place write sees the values before it.
#[test]
fn same_evaluation_donation_respects_holders_and_earlier_readers() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    with_stream(Device::gpu(0), || {
        let row = || (0..1, ..);
        let update = ops::full([1, 1024], 7.0f32);
        // Held: `a` must keep its values.
        let a = ops::zeros([16, 1024], DType::F32) + 1.0;
        let b = a.clone().at(row()).set(&update);
        eval(&[b.clone(), a.clone()]).unwrap();
        assert!(a.to_vec::<f32>().unwrap().iter().all(|&x| x == 1.0));
        assert_eq!(b.to_vec::<f32>().unwrap()[0], 7.0);
        // Read before the write in the same evaluation: `sum` sees the ones.
        for _ in 0..20 {
            let a = ops::zeros([16, 1024], DType::F32) + 1.0;
            let total = a.sum(..);
            let b = a.at(row()).set(&update);
            eval(&[total.clone(), b.clone()]).unwrap();
            assert_eq!(total.item::<f32>().unwrap(), 16.0 * 1024.0);
            let v = b.to_vec::<f32>().unwrap();
            assert!(v[..1024].iter().all(|&x| x == 7.0) && v[1024..].iter().all(|&x| x == 1.0));
        }
    });
}

/// The elements as float64 (exact for float32).
fn values(a: &Array) -> Vec<f64> {
    a.astype(DType::F64).to_vec::<f64>().unwrap()
}

/// Element-wise operations and softmax write into an input they own (MLX's donation), never
/// into one another array holds, with the same results either way.
#[test]
fn element_wise_operations_write_into_owned_inputs() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let _serial = PEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    with_stream(Device::gpu(0), || {
        let n = 4 << 20; // 16 MiB of float32
        let make = || ops::arange(0.0, n as f64, 1.0).sin().reshape([64, n / 64]);
        // References computed with the input held (no donation possible).
        let held = make();
        held.eval().unwrap();
        let reference = |f: &dyn Fn(&Array) -> Array| values(&f(&held));
        type Op = (&'static str, Box<dyn Fn(&Array) -> Array>);
        let ops: [Op; 4] = [
            ("exp", Box::new(|a: &Array| a.exp())),
            ("a * 3 + 1", Box::new(|a: &Array| a * 3.0 + 1.0)),
            (
                "where",
                Box::new(|a: &Array| {
                    let m = a.greater(0.0);
                    m.select(a, 0.0)
                }),
            ),
            ("softmax", Box::new(|a: &Array| a.softmax(-1))),
        ];
        for (name, f) in &ops {
            let want = reference(f.as_ref());
            assert_eq!(
                values(&held),
                values(&make()),
                "{name}: the held input changed"
            );
            let a = make();
            a.eval().unwrap();
            gpu_peak_since_reset();
            let resident = nuvec_gpu::Device::shared().map_or(0, |d| d.memory().active);
            let y = f(&a);
            drop(a);
            y.eval().unwrap();
            let peak = gpu_peak_since_reset();
            assert!(
                peak < resident + n * 4 / 2,
                "{name}: peak {peak}, before {resident}"
            );
            assert_eq!(values(&y), want, "{name}: in place differs");
        }
        // One array as both operands: never written while it is still read.
        let a = make();
        a.eval().unwrap();
        let want: Vec<f64> = values(&a).iter().map(|x| x * x).collect();
        let sq = &a * &a;
        drop(a);
        let got = values(&sq);
        assert!(
            got.iter()
                .zip(&want)
                .all(|(g, w)| (g - w).abs() <= 1e-6 * w.abs().max(1.0))
        );
    });
}

/// Attention without a fused kernel (head size 256 over a longer cache, as Qwen 3.5 prefill
/// chunks) holds one score matrix, as MLX: the mask and the softmax write into it.
#[test]
fn composed_attention_holds_one_score_matrix() {
    if crate::common::require_gpu().is_none() {
        return;
    }
    let _serial = PEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    with_stream(Device::gpu(0), || {
        let (hq, hkv, lq, lk, d) = (16, 2, 256, 4096, 256);
        let q = random::normal([1, hq, lq, d], DType::BF16, 0.0, 1.0, Some(&random::key(1)));
        let k = random::normal(
            [1, hkv, lk, d],
            DType::BF16,
            0.0,
            1.0,
            Some(&random::key(2)),
        );
        eval(&[q.clone(), k.clone()]).unwrap();
        gpu_peak_since_reset();
        let resident = nuvec_gpu::Device::shared().map_or(0, |dv| dv.memory().active);
        let o = fast::scaled_dot_product_attention(
            &q,
            &k,
            &k,
            0.0625,
            &fast::AttentionOptions::causal(),
        );
        o.eval().unwrap();
        let scores = hq * lq * lk * 2;
        let peak = gpu_peak_since_reset();
        assert!(
            peak < resident + scores * 3 / 2,
            "peak {peak}, before {resident}, scores {scores}"
        );
    });
}
