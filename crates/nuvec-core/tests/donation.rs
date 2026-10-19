//! In-place `slice_update` on the GPU (buffer donation): effective when the evaluation
//! owns the base, invisible otherwise. A separate test binary, so that no other test
//! allocates while peak memory is measured. Every test skips when no Metal device is
//! available.

#![cfg(target_os = "macos")]

use nuvec_core::*;

fn gpu_available() -> bool {
    let ok = !cfg!(miri) && on(Device::gpu(0), || (&Array::scalar(1.0f32) + 1.0).eval()).is_ok();
    if !ok {
        eprintln!("skipped: no Metal device");
    }
    ok
}

/// Runs `f` with `device` as this thread's default, restoring the previous default.
fn on<R>(device: Device, f: impl FnOnce() -> R) -> R {
    let previous = default_stream();
    set_default_stream(Stream::default_for(device));
    let r = f();
    set_default_stream(previous);
    r
}

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
fn slice_update_writes_in_place_when_it_owns_its_base() {
    if !gpu_available() {
        return;
    }
    let _serial = PEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    on(Device::gpu(0), || {
        let n = 4 << 20; // 16 MiB of float32
        let row = |i: usize| [SliceArg::from(i as isize..i as isize + 1), SliceArg::full()];
        let mut a = zeros(&[64, n / 64], DType::F32).unwrap();
        a.eval().unwrap();
        gpu_peak_since_reset();
        let resident = nuvec_gpu::Device::shared().map_or(0, |d| d.memory().active);
        for i in 0..64 {
            let update = Array::full(&[1, n / 64], i as f32).unwrap();
            a = slice_update(&a, &update, &row(i)).unwrap();
            a.eval().unwrap();
        }
        // Each step reused the buffer: a copy would have held two 16 MiB buffers at once.
        let peak = gpu_peak_since_reset();
        assert!(
            peak < resident + n * 4 / 2,
            "peak {peak}, before {resident}"
        );
        let v = a.to_vec::<f32>().unwrap();
        assert!((0..64).all(|i| v[i * n / 64] == i as f32 && v[(i + 1) * n / 64 - 1] == i as f32));
    });
}

/// The decode order (mlx-lm's): each step is submitted while the previous one is still in
/// flight, and a small result of the previous step is read afterwards. The base is then
/// in flight, held only by pending work, and is still written in place.
#[test]
fn slice_update_writes_in_place_into_bases_still_in_flight() {
    if !gpu_available() {
        return;
    }
    let _serial = PEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    on(Device::gpu(0), || {
        let n = 4 << 20; // 16 MiB of float32
        let row = |i: usize| [SliceArg::from(i as isize..i as isize + 1), SliceArg::full()];
        let mut a = zeros(&[64, n / 64], DType::F32).unwrap();
        a.eval().unwrap();
        // Work that keeps each step's command buffer running while the next is built,
        // without allocating anything sizeable (a reduction of a resident 64 MiB array).
        let big = Array::full(&[16 << 20], 1.0f32 / (16 << 20) as f32).unwrap();
        big.eval().unwrap();
        gpu_peak_since_reset();
        let resident = nuvec_gpu::Device::shared().map_or(0, |d| d.memory().active);
        let mut token: Option<Array> = None;
        for i in 0..64 {
            let update = Array::full(&[1, n / 64], i as f32).unwrap();
            a = slice_update(&a, &update, &row(i)).unwrap();
            // A small result of this step, read only after the next step is submitted.
            let busy = sum(&big, &[], false).unwrap();
            let next = add(sum(&slice(&a, &row(i)).unwrap(), &[], false).unwrap(), busy).unwrap();
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
    if !gpu_available() {
        return;
    }
    on(Device::gpu(0), || {
        let base = Array::from_slice(&[0.0f32; 8], &[8]).unwrap();
        base.eval().unwrap();
        let view = slice(&base, &[SliceArg::from(0..4)]).unwrap();
        let one = Array::from_slice(&[1.0f32; 2], &[2]).unwrap();
        // The base is held (and viewed): the update copies.
        let b = slice_update(&base, &one, &[SliceArg::from(2..4)]).unwrap();
        assert_eq!(b.to_vec::<f32>().unwrap(), [0., 0., 1., 1., 0., 0., 0., 0.]);
        assert_eq!(base.to_vec::<f32>().unwrap(), [0.0; 8]);
        assert_eq!(view.to_vec::<f32>().unwrap(), [0.0; 4]);
        // A chain submitted with async_eval (bases in flight) gives the same values as
        // one evaluated step by step.
        let mut chained = zeros(&[16], DType::F32).unwrap();
        for i in 0..16 {
            let v = Array::full(&[1], i as f32).unwrap();
            chained =
                slice_update(&chained, &v, &[SliceArg::from(i as isize..i as isize + 1)]).unwrap();
            async_eval(std::slice::from_ref(&chained)).unwrap();
        }
        let expected: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(chained.to_vec::<f32>().unwrap(), expected);
    });
}

#[test]
fn slice_update_differentiates_on_the_gpu() {
    if !gpu_available() {
        return;
    }
    on(Device::gpu(0), || {
        // d/dx sum(slice_update(x, 3·x[0:2], [2:4])²) at x = [1, 2, 3, 4].
        let f = |x: &Array| -> Result<Array> {
            let part = multiply(&slice(x, &[SliceArg::from(0..2)])?, 3.0)?;
            let y = slice_update(x, &part, &[SliceArg::from(2..4)])?;
            sum(&multiply(&y, &y)?, &[], false)
        };
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let g = grad(f)(&x).unwrap().to_vec::<f32>().unwrap();
        // y = [x0, x1, 3x0, 3x1]: dy²/dx0 = 2x0 + 18x0, dy²/dx1 = 2x1 + 18x1, x2 and x3
        // are overwritten.
        assert_eq!(g, [20.0, 40.0, 0.0, 0.0]);
    });
}
