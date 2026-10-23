//! `compile` on the GPU: compiled results equal uncompiled ones bit for bit (random shapes,
//! layouts and dtypes), traces are cached per signature, and transforms compose. Every
//! test skips when no Metal device is available.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicUsize, Ordering};

use nuvec_compile::compile_flat;
use nuvec_core::graph::{self, Primitive};
use nuvec_core::{
    Array, DType, Device, Result, Step, Stream, array, grad, set_default_stream, vmap_flat,
};

/// Makes the GPU this thread's default stream; false (and skip) if there is none.
fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = !cfg!(miri) && (&Array::from(1.0f32) + 1.0).eval().is_ok();
    if !ok {
        // CI on a GPU machine must not pass by skipping (as `require_gpu` in nuvec-core).
        assert!(
            std::env::var("NUVEC_REQUIRE_GPU").as_deref() != Ok("1"),
            "NUVEC_REQUIRE_GPU=1 but no Metal device is available"
        );
        eprintln!("skipped: no Metal device");
    }
    ok
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn values(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| (self.below(1601) as f32 - 800.0) / 200.0)
            .collect()
    }
}

/// Random values of `shape` and `dtype` in a random layout: contiguous, transposed, a
/// strided (possibly reversed) slice, or broadcast.
fn input(rng: &mut Rng, shape: &[usize], dtype: DType) -> Array {
    let n: usize = shape.iter().product();
    let make = |rng: &mut Rng, shape: &[usize]| {
        let n = shape.iter().product();
        Array::from_slice(&rng.values(n), shape).astype(dtype)
    };
    let a = match if n == 0 { 0 } else { rng.below(4) } {
        0 => make(rng, shape),
        1 => {
            let reversed: Vec<usize> = shape.iter().rev().copied().collect();
            make(rng, &reversed).t()
        }
        2 => {
            let big: Vec<usize> = shape.iter().map(|&d| 2 * d).collect();
            let steps: Vec<i32> = shape
                .iter()
                .map(|_| if rng.below(2) == 0 { 2 } else { -2 })
                .collect();
            strided(&make(rng, &big), &steps)
        }
        _ => {
            let small: Vec<usize> = shape
                .iter()
                .map(|&d| if rng.below(2) == 0 { 1 } else { d })
                .collect();
            make(rng, &small).broadcast_to(shape)
        }
    };
    a.eval().unwrap();
    a
}

/// `a` with every axis strided by `steps` (`(..).step(s)`: a negative step walks backwards).
fn strided(a: &Array, steps: &[i32]) -> Array {
    let s = |k: usize| (..).step(steps[k]);
    match steps.len() {
        0 => a.clone(),
        1 => a.i(s(0)),
        2 => a.i((s(0), s(1))),
        3 => a.i((s(0), s(1), s(2))),
        4 => a.i((s(0), s(1), s(2), s(3))),
        n => unreachable!("{n} dimensions"),
    }
}

/// The outputs' bits (as float32: exact for these values and dtypes).
fn bits(outputs: &[Array]) -> Vec<Vec<u32>> {
    outputs
        .iter()
        .map(|o| {
            o.astype(DType::F32)
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|x| if x.is_nan() { u32::MAX } else { x.to_bits() })
                .collect()
        })
        .collect()
}

/// A mix of element-wise operations over three float inputs: arithmetic, scalars,
/// comparisons, select, transcendentals and casts.
fn float_mix(xs: &[Array]) -> Result<Vec<Array>> {
    let (x, y, z) = (&xs[0], &xs[1], &xs[2]);
    let a = x.multiply(y).add(0.5);
    let b = a
        .greater(z)
        .select(a.multiply(&a).negative().exp(), z.sigmoid().subtract(&a));
    let c = b.maximum(x.tanh()).divide(y.abs().add(1.0));
    let d = c.astype(DType::F32).multiply(2.0).astype(x.dtype());
    let e = d.less_equal(z.abs().sqrt());
    Ok(vec![c, e])
}

/// The same for integers: wrapping arithmetic, comparisons, select, casts.
fn int_mix(xs: &[Array]) -> Result<Vec<Array>> {
    let (x, y, z) = (&xs[0], &xs[1], &xs[2]);
    let a = x.multiply(y).add(3);
    let b = a.greater_equal(z).select(a.subtract(z), a.negative());
    let c = b.maximum(x.abs()).minimum(100);
    let ne = c.not_equal(y).astype(DType::F16);
    Ok(vec![c, ne])
}

#[test]
fn compiled_results_equal_uncompiled_ones() {
    if !gpu() {
        return;
    }
    let mut rng = Rng(0x00C0_FFEE);
    let float = compile_flat(float_mix);
    let int = compile_flat(int_mix);
    for case in 0..60 {
        let ndim = rng.below(5);
        let shape: Vec<usize> = (0..ndim).map(|_| 1 + rng.below(6)).collect();
        let dtypes = [DType::F32, DType::F16, DType::BF16, DType::I32, DType::U8];
        let dtype = dtypes[case % dtypes.len()];
        let xs: Vec<Array> = (0..3).map(|_| input(&mut rng, &shape, dtype)).collect();
        let is_float = matches!(dtype, DType::F32 | DType::F16 | DType::BF16);
        let (out, expected) = if is_float {
            (float(&xs).unwrap(), float_mix(&xs).unwrap())
        } else {
            (int(&xs).unwrap(), int_mix(&xs).unwrap())
        };
        assert!(
            matches!(graph::primitive(&out[0]), Some(Primitive::Compiled(_))),
            "case {case}: fused"
        );
        assert_eq!(
            bits(&out),
            bits(&expected),
            "case {case}: {dtype} {shape:?}"
        );
    }
}

#[test]
fn traces_once_per_signature() {
    if !gpu() {
        return;
    }
    let traces = AtomicUsize::new(0);
    let f = compile_flat(|xs: &[Array]| {
        traces.fetch_add(1, Ordering::Relaxed);
        Ok(vec![xs[0].multiply(2.0).exp()])
    });
    let x = |n: usize| Array::from_slice(&vec![0.5f32; n], [n]);
    for _ in 0..3 {
        f(&[x(4)]).unwrap()[0].eval().unwrap();
    }
    assert_eq!(traces.load(Ordering::Relaxed), 1);
    f(&[x(5)]).unwrap()[0].eval().unwrap();
    f(&[x(4).astype(DType::F16)]).unwrap()[0].eval().unwrap();
    assert_eq!(traces.load(Ordering::Relaxed), 3, "new shape, new dtype");
    // A new layout of the same signature replays (the kernel reads it strided).
    let y = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2]);
    let g = compile_flat(|xs: &[Array]| Ok(vec![xs[0].add(1.0).exp()]));
    let a = g(std::slice::from_ref(&y))
        .unwrap()
        .remove(0)
        .to_vec::<f32>()
        .unwrap();
    let b = g(&[y.t()]).unwrap().remove(0).to_vec::<f32>().unwrap();
    assert_eq!([a[1], a[2]], [b[2], b[1]]);
}

#[test]
fn outputs_may_be_inputs_constants_or_shared() {
    if !gpu() {
        return;
    }
    let c = array![7.0, 8.0];
    let f = compile_flat(|xs: &[Array]| {
        let y = xs[0].add(1.0);
        let yy = y.multiply(&y);
        Ok(vec![xs[0].clone(), c.clone(), y, yy])
    });
    let x = array![1.0, 2.0];
    let out = f(std::slice::from_ref(&x)).unwrap();
    let v: Vec<Vec<f32>> = out.iter().map(|a| a.to_vec::<f32>().unwrap()).collect();
    assert_eq!(
        v,
        [
            vec![1.0, 2.0],
            vec![7.0, 8.0],
            vec![2.0, 3.0],
            vec![4.0, 9.0]
        ]
    );
    assert_eq!(
        graph::id(&out[0]),
        graph::id(&x),
        "an input is returned as is"
    );
}

#[test]
fn one_kernel_computes_several_outputs() {
    if !gpu() {
        return;
    }
    // An Adam step: the new moments are outputs and feed the parameter update.
    let adam = |xs: &[Array]| {
        let (p, m, v, g) = (&xs[0], &xs[1], &xs[2], &xs[3]);
        let m = m.multiply(0.9).add(g.multiply(0.1));
        let v = v.multiply(0.999).add(g.multiply(g).multiply(0.001));
        let step = m.divide(v.sqrt().add(1e-8));
        Ok(vec![p.subtract(step.multiply(1e-3)), m, v])
    };
    let mut rng = Rng(11);
    let compiled = compile_flat(adam);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        // Random layouts, then contiguous inputs, whose kernel computes several elements
        // per thread (the size is not a multiple of their count).
        let shape = [3, 70_001];
        let random: Vec<Array> = (0..4).map(|_| input(&mut rng, &shape, dtype)).collect();
        let n = 3 * 70_001;
        let contiguous: Vec<Array> = (0..4)
            .map(|_| {
                let v = rng.values(n);
                Array::from_slice(&v, shape).astype(dtype)
            })
            .collect();
        for xs in [random, contiguous] {
            let out = compiled(&xs).unwrap();
            assert!(
                out.iter().all(|o| graph::id(o) == graph::id(&out[0])),
                "one operation"
            );
            assert_eq!(bits(&out), bits(&adam(&xs).unwrap()), "{dtype}");
        }
    }
}

#[test]
fn evaluating_inside_a_compiled_function_fails() {
    if !gpu() {
        return;
    }
    let f = compile_flat(|xs: &[Array]| {
        let y = xs[0].add(1.0);
        y.eval()?;
        Ok(vec![y])
    });
    assert!(f(&[Array::from(1.0f32)]).is_err());
}

#[test]
fn transforms_compose_with_compile() {
    if !gpu() {
        return;
    }
    // loss(w) = sum(sigmoid(w * x) * w)
    let x = array![0.5, -1.0, 2.0];
    let loss = |w: &Array| -> Result<Array> { Ok(w.multiply(&x).sigmoid().multiply(w).sum(..)) };
    let compiled = compile_flat(move |xs: &[Array]| Ok(vec![loss(&xs[0])?]));
    let w = array![1.0, 2.0, -0.5];
    let expected = grad(loss)(&w).unwrap().to_vec::<f32>().unwrap();
    // grad of a compiled function: traced through the original operations.
    let g = grad(|w: &Array| Ok(compiled(std::slice::from_ref(w))?.remove(0)))(&w).unwrap();
    assert_eq!(g.to_vec::<f32>().unwrap(), expected);
    // a compiled gradient: fused, and equal.
    let compiled_grad = compile_flat(move |xs: &[Array]| Ok(vec![grad(loss)(&xs[0])?]));
    let g = compiled_grad(std::slice::from_ref(&w)).unwrap().remove(0);
    assert_eq!(g.to_vec::<f32>().unwrap(), expected);
    // vmap of a compiled function.
    let per_row = vmap_flat(
        |xs| Ok(vec![compiled(&[xs[0].clone()])?.remove(0)]),
        &[Some(0)],
        &[],
    );
    let ws = stack_rows(&w);
    let rows = per_row(&[ws]).unwrap().remove(0).to_vec::<f32>().unwrap();
    let one = loss(&w).unwrap().item::<f32>().unwrap();
    assert_eq!(rows, [one, one]);
}

/// `[a, a]` as a 2-row matrix.
fn stack_rows(a: &Array) -> Array {
    let n = a.shape()[0];
    a.reshape([1, n]).broadcast_to([2, n])
}

#[test]
fn long_chains_and_many_inputs_split_into_several_kernels() {
    if !gpu() {
        return;
    }
    // 100 operations: more than one kernel holds.
    let chain = |xs: &[Array]| {
        let mut y = xs[0].clone();
        for i in 0..100 {
            y = if i % 2 == 0 {
                y.add(0.25)
            } else {
                y.multiply(0.5)
            };
        }
        Ok(vec![y])
    };
    let x = array![1.0, -3.0, 8.0];
    assert_eq!(
        bits(&compile_flat(chain)(std::slice::from_ref(&x)).unwrap()),
        bits(&chain(std::slice::from_ref(&x)).unwrap())
    );
    // 40 inputs summed: more inputs than one kernel binds.
    let many = |xs: &[Array]| {
        let mut y = xs[0].clone();
        for x in &xs[1..] {
            y = y.add(x.multiply(3.0));
        }
        Ok(vec![y])
    };
    let mut rng = Rng(7);
    let xs: Vec<Array> = (0..40)
        .map(|_| input(&mut rng, &[4, 5], DType::F32))
        .collect();
    assert_eq!(
        bits(&compile_flat(many)(&xs).unwrap()),
        bits(&many(&xs).unwrap())
    );
}

#[test]
fn empty_and_zero_dimensional_inputs() {
    if !gpu() {
        return;
    }
    let f = compile_flat(|xs: &[Array]| Ok(vec![xs[0].add(&xs[1]).exp()]));
    let e = Array::from_slice::<f32>(&[], [0, 3]);
    assert_eq!(f(&[e.clone(), e]).unwrap()[0].shape(), [0, 3]);
    let s = Array::from(0.0f32);
    assert_eq!(f(&[s.clone(), s]).unwrap()[0].item::<f32>().unwrap(), 1.0);
}

/// Parameters updated with gradients (`w - lr * g`) are plain arrays once the transform
/// has returned: a compiled function of them traces once, not on every call.
#[test]
fn parameters_after_an_optimizer_step_replay_the_trace() {
    if !gpu() {
        return;
    }
    let traces = AtomicUsize::new(0);
    let f = compile_flat(|xs: &[Array]| {
        traces.fetch_add(1, Ordering::Relaxed);
        Ok(vec![xs[0].multiply(&xs[1]).sum(..)])
    });
    let x = array![0.5, -1.0, 2.0];
    let loss = |w: &Array| Ok(w.multiply(w).sum(..));
    let mut w = array![1.0, 2.0, 3.0];
    for _ in 0..3 {
        let g = grad(loss)(&w).unwrap();
        w = &w - &(&g * 0.1f32);
        f(&[w.clone(), x.clone()]).unwrap()[0].eval().unwrap();
    }
    assert_eq!(traces.load(Ordering::Relaxed), 1, "three calls");
    for _ in 0..100 {
        let g = grad(loss)(&w).unwrap();
        w = &w - &(&g * 0.01f32);
        f(&[w.clone(), x.clone()]).unwrap()[0].eval().unwrap();
    }
    assert_eq!(traces.load(Ordering::Relaxed), 1, "a 100-step SGD loop");
}

/// A function compiled inside a transform that captures one of its tracers is traced on
/// every call (its tape would hold the tracer), and the compile cache is bounded.
#[test]
fn tapes_with_live_tracers_are_not_cached_and_the_cache_is_bounded() {
    if !gpu() {
        return;
    }
    let traces = std::sync::Arc::new(AtomicUsize::new(0));
    let counted = traces.clone();
    let c = array![1.0, 2.0];
    let g = grad(|x: &Array| {
        let captured = x.clone();
        let counted = counted.clone();
        let f = compile_flat(move |ys: &[Array]| {
            counted.fetch_add(1, Ordering::Relaxed);
            Ok(vec![ys[0].multiply(&captured)])
        });
        let a = f(std::slice::from_ref(&c))?.remove(0);
        let b = f(std::slice::from_ref(&c))?.remove(0);
        Ok(a.add(&b).sum(..))
    })(&array![3.0, 4.0])
    .unwrap();
    assert_eq!(
        g.to_vec::<f32>().unwrap(),
        [2.0, 4.0],
        "d/dx sum(2 c x) = 2 c"
    );
    assert_eq!(traces.load(Ordering::Relaxed), 2, "traced on each call");

    // 200 signatures: the least recently used tapes are evicted.
    let traces = AtomicUsize::new(0);
    let f = compile_flat(|xs: &[Array]| {
        traces.fetch_add(1, Ordering::Relaxed);
        Ok(vec![xs[0].exp()])
    });
    let x = |n: usize| Array::from_slice(&vec![0.5f32; n], [n]);
    for n in 1..=200 {
        f(&[x(n)]).unwrap()[0].eval().unwrap();
    }
    assert_eq!(traces.load(Ordering::Relaxed), 200);
    f(&[x(200)]).unwrap()[0].eval().unwrap();
    assert_eq!(traces.load(Ordering::Relaxed), 200, "recent: cached");
    f(&[x(1)]).unwrap()[0].eval().unwrap();
    assert_eq!(traces.load(Ordering::Relaxed), 201, "oldest: evicted");
}

/// Inside `compile`, `copy` is neither simplified away nor aliased: a snapshot taken before
/// an in-place update of the same array keeps the old values, and the compiled tape keeps
/// the `Copy` node.
#[test]
fn copy_survives_compile_and_in_place_updates() {
    if !gpu() {
        return;
    }
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::U32] {
        let f = compile_flat(|xs: &[Array]| {
            let state = xs[0].clone() + Array::from(0u32).astype(xs[0].dtype());
            let snapshot = state.copy();
            let updated = state.at(0).set(Array::from(7u32).astype(xs[0].dtype()));
            Ok(vec![snapshot, updated])
        });
        let x = nuvec_core::ops::arange(0, 6, 1).astype(dtype);
        for _ in 0..3 {
            let out = f(std::slice::from_ref(&x)).unwrap();
            assert_eq!(
                out[0].astype(DType::F32).to_vec::<f32>().unwrap(),
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "{dtype}: the snapshot"
            );
            assert_eq!(
                out[1].astype(DType::F32).to_vec::<f32>().unwrap(),
                [7.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "{dtype}: the update"
            );
        }
    }
    // The copy of a same-dtype `astype` "copy" would be removed; `copy` is not.
    let f = compile_flat(|xs: &[Array]| Ok(vec![xs[0].copy()]));
    let out = f(&[array![1.0, 2.0]]).unwrap();
    assert!(matches!(
        graph::primitive(&out[0]),
        Some(Primitive::Copy) | None
    ));
}
