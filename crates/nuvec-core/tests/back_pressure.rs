//! The in-flight counter is device-wide within a process. Keep this integration test
//! in its own binary so the number of submitting threads is known.
#![cfg(target_os = "macos")]

mod common;

use nuvec_core::{Device, Stream, async_eval, ops, set_default_stream};

#[test]
fn async_eval_applies_back_pressure() {
    if common::require_gpu().is_none() {
        return;
    }
    let device = nuvec_gpu::Device::shared().unwrap();
    for submitters in [1, 4] {
        device.wait_for_capacity(1);
        let limit = 10 + (submitters - 1);
        std::thread::scope(|scope| {
            for _ in 0..submitters {
                scope.spawn(|| {
                    set_default_stream(Stream::default_for(Device::gpu(0)));
                    let big = ops::full([1 << 20], 1.0f32);
                    let mut outs = Vec::new();
                    for _ in 0..60 / submitters {
                        let y = (&big * 2.0 + 1.0).sum(..);
                        async_eval(std::slice::from_ref(&y)).unwrap();
                        // Each submitter can pass the threshold check with one pending commit.
                        let seen = device.in_flight();
                        assert!(
                            seen <= limit,
                            "{seen} in flight with {submitters} submitters"
                        );
                        outs.push(y);
                    }
                    for y in outs {
                        assert_eq!(y.item::<f32>().unwrap(), 3.0 * (1 << 20) as f32);
                    }
                });
            }
        });
    }
}
