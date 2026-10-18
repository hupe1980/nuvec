//! `compile` on the GPU: compiled results equal uncompiled ones bit for bit (random shapes,
//! layouts and dtypes), traces are cached per signature, and transforms compose. Every
//! test skips when no Metal device is available.

#![cfg(target_os = "macos")]

use std::sync::atomic::{AtomicUsize, Ordering};

use nuvec_compile::compile;
use nuvec_core::*;

/// Makes the GPU this thread's default stream; false (and skip) if there is none.
fn gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = !cfg!(miri) && (&Array::scalar(1.0f32) + 1.0).eval().is_ok();
    if !ok {
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
        Array::from_slice(&rng.values(n), shape)
            .unwrap()
            .astype(dtype)
            .unwrap()
    };
    let a = match if n == 0 { 0 } else { rng.below(4) } {
        0 => make(rng, shape),
        1 => {
            let reversed: Vec<usize> = shape.iter().rev().copied().collect();
            make(rng, &reversed).t().unwrap()
        }
        2 => {
            let big: Vec<usize> = shape.iter().map(|&d| 2 * d).collect();
            let args: Vec<SliceArg> = shape
                .iter()
                .map(|_| {
                    let step = if rng.below(2) == 0 { 2 } else { -2 };
                    SliceArg::new(None, None, step)
                })
                .collect();
            slice(&make(rng, &big), &args).unwrap()
        }
        _ => {
            let small: Vec<usize> = shape
                .iter()
                .map(|&d| if rng.below(2) == 0 { 1 } else { d })
                .collect();
            broadcast_to(&make(rng, &small), shape).unwrap()
        }
    };
    a.eval().unwrap();
    a
}

/// The outputs' bits (as float32: exact for these values and dtypes).
fn bits(outputs: &[Array]) -> Vec<Vec<u32>> {
    outputs
        .iter()
        .map(|o| {
            o.astype(DType::F32)
                .unwrap()
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
    let a = add(&multiply(x, y)?, 0.5)?;
    let b = where_(
        &greater(&a, z)?,
        &exp(&negative(&multiply(&a, &a)?)?)?,
        &subtract(&sigmoid(z)?, &a)?,
    )?;
    let c = divide(&maximum(&b, &tanh(x)?)?, &add(&abs(y)?, 1.0)?)?;
    let d = multiply(&c.astype(DType::F32)?, 2.0)?.astype(x.dtype())?;
    Ok(vec![c, less_equal(&d, &sqrt(&abs(z)?)?)?])
}

/// The same for integers: wrapping arithmetic, comparisons, select, casts.
fn int_mix(xs: &[Array]) -> Result<Vec<Array>> {
    let (x, y, z) = (&xs[0], &xs[1], &xs[2]);
    let a = add(&multiply(x, y)?, 3)?;
    let b = where_(&greater_equal(&a, z)?, &subtract(&a, z)?, &negative(&a)?)?;
    let c = minimum(&maximum(&b, &abs(x)?)?, 100)?;
    Ok(vec![c.clone(), not_equal(&c, y)?.astype(DType::F16)?])
}

#[test]
fn compiled_results_equal_uncompiled_ones() {
    if !gpu() {
        return;
    }
    let mut rng = Rng(0x00C0_FFEE);
    let float = compile(float_mix);
    let int = compile(int_mix);
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
            matches!(out[0].primitive(), Some(Primitive::Compiled(_))),
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
    let f = compile(|xs: &[Array]| {
        traces.fetch_add(1, Ordering::Relaxed);
        Ok(vec![exp(&multiply(&xs[0], 2.0)?)?])
    });
    let x = |n: usize| Array::from_slice(&vec![0.5f32; n], &[n]).unwrap();
    for _ in 0..3 {
        f(&[x(4)]).unwrap()[0].eval().unwrap();
    }
    assert_eq!(traces.load(Ordering::Relaxed), 1);
    f(&[x(5)]).unwrap()[0].eval().unwrap();
    f(&[x(4).astype(DType::F16).unwrap()]).unwrap()[0]
        .eval()
        .unwrap();
    assert_eq!(traces.load(Ordering::Relaxed), 3, "new shape, new dtype");
    // A new layout of the same signature replays (the kernel reads it strided).
    let y = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let g = compile(|xs: &[Array]| Ok(vec![exp(&add(&xs[0], 1.0)?)?]));
    let a = g(std::slice::from_ref(&y))
        .unwrap()
        .remove(0)
        .to_vec::<f32>()
        .unwrap();
    let b = g(&[y.t().unwrap()])
        .unwrap()
        .remove(0)
        .to_vec::<f32>()
        .unwrap();
    assert_eq!([a[1], a[2]], [b[2], b[1]]);
}

#[test]
fn outputs_may_be_inputs_constants_or_shared() {
    if !gpu() {
        return;
    }
    let c = Array::from_slice(&[7.0f32, 8.0], &[2]).unwrap();
    let f = compile(|xs: &[Array]| {
        let y = add(&xs[0], 1.0)?;
        Ok(vec![xs[0].clone(), c.clone(), y.clone(), multiply(&y, &y)?])
    });
    let x = Array::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
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
    assert_eq!(out[0].id(), x.id(), "an input is returned as is");
}

#[test]
fn one_kernel_computes_several_outputs() {
    if !gpu() {
        return;
    }
    // An Adam step: the new moments are outputs and feed the parameter update.
    let adam = |xs: &[Array]| {
        let (p, m, v, g) = (&xs[0], &xs[1], &xs[2], &xs[3]);
        let m = add(&multiply(m, 0.9)?, &multiply(g, 0.1)?)?;
        let v = add(&multiply(v, 0.999)?, &multiply(&multiply(g, g)?, 0.001)?)?;
        let step = divide(&m, &add(&sqrt(&v)?, 1e-8)?)?;
        Ok(vec![subtract(p, &multiply(&step, 1e-3)?)?, m, v])
    };
    let mut rng = Rng(11);
    let compiled = compile(adam);
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        // Random layouts, then contiguous inputs, whose kernel computes several elements
        // per thread (the size is not a multiple of their count).
        let shape = [3, 70_001];
        let random: Vec<Array> = (0..4).map(|_| input(&mut rng, &shape, dtype)).collect();
        let n = 3 * 70_001;
        let contiguous: Vec<Array> = (0..4)
            .map(|_| {
                let v = rng.values(n);
                Array::from_slice(&v, &shape)
                    .unwrap()
                    .astype(dtype)
                    .unwrap()
            })
            .collect();
        for xs in [random, contiguous] {
            let out = compiled(&xs).unwrap();
            assert!(out.iter().all(|o| o.id() == out[0].id()), "one operation");
            assert_eq!(bits(&out), bits(&adam(&xs).unwrap()), "{dtype}");
        }
    }
}

#[test]
fn evaluating_inside_a_compiled_function_fails() {
    if !gpu() {
        return;
    }
    let f = compile(|xs: &[Array]| {
        let y = add(&xs[0], 1.0)?;
        y.eval()?;
        Ok(vec![y])
    });
    assert!(f(&[Array::scalar(1.0f32)]).is_err());
}

#[test]
fn transforms_compose_with_compile() {
    if !gpu() {
        return;
    }
    // loss(w) = sum(sigmoid(w * x) * w)
    let x = Array::from_slice(&[0.5f32, -1.0, 2.0], &[3]).unwrap();
    let loss = |w: &Array| -> Result<Array> {
        sum(&multiply(&sigmoid(&multiply(w, &x)?)?, w)?, &[], false)
    };
    let compiled = compile(move |xs: &[Array]| Ok(vec![loss(&xs[0])?]));
    let w = Array::from_slice(&[1.0f32, 2.0, -0.5], &[3]).unwrap();
    let expected = grad(loss)(&w).unwrap().to_vec::<f32>().unwrap();
    // grad of a compiled function: traced through the original operations.
    let g = grad(|w: &Array| Ok(compiled(std::slice::from_ref(w))?.remove(0)))(&w).unwrap();
    assert_eq!(g.to_vec::<f32>().unwrap(), expected);
    // a compiled gradient: fused, and equal.
    let compiled_grad = compile(move |xs: &[Array]| Ok(vec![grad(loss)(&xs[0])?]));
    let g = compiled_grad(std::slice::from_ref(&w)).unwrap().remove(0);
    assert_eq!(g.to_vec::<f32>().unwrap(), expected);
    // vmap of a compiled function.
    let per_row = vmap(
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
    broadcast_to(&a.reshape(&[1, n as isize]).unwrap(), &[2, n]).unwrap()
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
                add(&y, 0.25)?
            } else {
                multiply(&y, 0.5)?
            };
        }
        Ok(vec![y])
    };
    let x = Array::from_slice(&[1.0f32, -3.0, 8.0], &[3]).unwrap();
    assert_eq!(
        bits(&compile(chain)(std::slice::from_ref(&x)).unwrap()),
        bits(&chain(std::slice::from_ref(&x)).unwrap())
    );
    // 40 inputs summed: more inputs than one kernel binds.
    let many = |xs: &[Array]| {
        let mut y = xs[0].clone();
        for x in &xs[1..] {
            y = add(&y, &multiply(x, 3.0)?)?;
        }
        Ok(vec![y])
    };
    let mut rng = Rng(7);
    let xs: Vec<Array> = (0..40)
        .map(|_| input(&mut rng, &[4, 5], DType::F32))
        .collect();
    assert_eq!(
        bits(&compile(many)(&xs).unwrap()),
        bits(&many(&xs).unwrap())
    );
}

#[test]
fn empty_and_zero_dimensional_inputs() {
    if !gpu() {
        return;
    }
    let f = compile(|xs: &[Array]| Ok(vec![exp(&add(&xs[0], &xs[1])?)?]));
    let e = Array::from_slice::<f32>(&[], &[0, 3]).unwrap();
    assert_eq!(f(&[e.clone(), e]).unwrap()[0].shape(), [0, 3]);
    let s = Array::scalar(0.0f32);
    assert_eq!(f(&[s.clone(), s]).unwrap()[0].item::<f32>().unwrap(), 1.0);
}
