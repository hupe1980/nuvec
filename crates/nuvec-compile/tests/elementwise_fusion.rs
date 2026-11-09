//! Fusion of MLX's element-wise functions (alone in its binary: it counts the process's GPU
//! dispatches).

use nuvec_compile::compile_flat;
use nuvec_core::{Array, DType, Device, Result, Stream, set_default_stream};

/// Makes the GPU this thread's default stream; false (and skip) if there is none.
fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = !cfg!(miri) && (&Array::from(1.0f32) + 1.0).eval().is_ok();
    if !ok {
        assert!(
            std::env::var("NUVEC_REQUIRE_GPU").as_deref() != Ok("1"),
            "NUVEC_REQUIRE_GPU=1 but no Metal device is available"
        );
        eprintln!("skipped: no Metal device");
    }
    ok
}

/// MLX's element-wise functions each fuse with their neighbours into one kernel, whose
/// results equal the uncompiled kernels' (floats within a few ulps of the precise ones;
/// integers exactly).
#[test]
fn the_new_element_wise_functions_fuse() {
    if !gpu() {
        return;
    }
    type Op = fn(&Array, &Array) -> Array;
    let floats: [(&str, Op); 20] = [
        ("erf", |x, _| x.erf()),
        ("expm1", |x, _| x.expm1()),
        ("log1p", |x, _| x.log1p()),
        ("log2", |_, y| y.log2()),
        ("log10", |_, y| y.log10()),
        ("tan", |x, _| x.tan()),
        ("sinh", |x, _| x.sinh()),
        ("cosh", |x, _| x.cosh()),
        ("arccos", |x, _| x.arccos()),
        ("arccosh", |_, y| y.arccosh()),
        ("arcsin", |x, _| x.arcsin()),
        ("arcsinh", |x, _| x.arcsinh()),
        ("arctan", |x, _| x.arctan()),
        ("arctanh", |x, _| x.arctanh()),
        ("rsqrt", |_, y| y.rsqrt()),
        ("floor", |_, y| y.multiply(3.7).floor()),
        ("ceil", |_, y| y.multiply(2.3).ceil()),
        ("round", |_, y| y.multiply(1.5).round(0)),
        ("remainder", |x, y| y.remainder(x)),
        ("arctan2", |x, y| x.arctan2(y)),
    ];
    let x = nuvec_core::random::uniform(0.1, 0.9, [3, 4099], DType::F32, None);
    let y = nuvec_core::random::uniform(1.0, 3.0, [3, 4099], DType::F32, None);
    let xs = [x, y];
    xs.iter().for_each(|a| a.eval().unwrap());
    // The dispatch counter is the process's: this test is alone in its binary.
    let dispatches = |compiled: &dyn Fn(&[Array]) -> Result<Vec<Array>>, xs: &[Array]| {
        compiled(xs).unwrap()[0].eval().unwrap();
        let before = nuvec_core::__gpu_dispatches();
        let out = compiled(xs).unwrap().remove(0);
        out.eval().unwrap();
        (nuvec_core::__gpu_dispatches() - before, out)
    };
    for (name, op) in floats {
        let f = move |xs: &[Array]| -> Result<Vec<Array>> {
            Ok(vec![
                op(&xs[0].multiply(0.5).add(0.2), &xs[1].add(0.5))
                    .multiply(2.0)
                    .add(1.0),
            ])
        };
        let (n, fused) = dispatches(&compile_flat(f), &xs);
        assert_eq!(n, 1, "{name}: {n} kernels");
        let plain = f(&xs).unwrap().remove(0).to_vec::<f32>().unwrap();
        for (a, b) in fused.to_vec::<f32>().unwrap().iter().zip(&plain) {
            assert!(
                (a - b).abs() <= 8.0 * f32::EPSILON * b.abs().max(1.0),
                "{name}: {a} vs {b}"
            );
        }
    }
    let ints: [(&str, Op); 7] = [
        ("bitwise_and", |i, s| i.bitwise_and(s)),
        ("bitwise_or", |i, s| i.bitwise_or(s)),
        ("bitwise_xor", |i, s| i.bitwise_xor(s)),
        ("left_shift", |i, s| i.left_shift(s)),
        ("right_shift", |i, s| i.right_shift(s)),
        ("bitwise_invert", |i, _| i.bitwise_invert()),
        ("remainder", |i, s| i.remainder(s.add(1))),
    ];
    let xs = [
        nuvec_core::random::randint(-1000, 1000, [4099], DType::I32, None),
        nuvec_core::random::randint(0, 31, [4099], DType::I32, None),
    ];
    xs.iter().for_each(|a| a.eval().unwrap());
    for (name, op) in ints {
        let f = move |xs: &[Array]| -> Result<Vec<Array>> {
            Ok(vec![op(&xs[0].add(3), &xs[1]).multiply(3).add(1)])
        };
        let (n, fused) = dispatches(&compile_flat(f), &xs);
        assert_eq!(n, 1, "{name}: {n} kernels");
        assert_eq!(
            fused.to_vec::<i32>().unwrap(),
            f(&xs).unwrap()[0].to_vec::<i32>().unwrap(),
            "{name}"
        );
    }
}
