//! `compile` over trees and `compile_mut`: structure-keyed caches with a
//! bound, tracer constants, non-`'static` closures, retracing on structure changes, stateful
//! steps with template write-back, structure errors and random-key threading. On the CPU,
//! and on the GPU where one exists.

use std::cell::Cell;
use std::collections::BTreeMap;

use nuvec_compile::{Options, compile, compile_flat, compile_mut};
use nuvec_core::tree::{Param, Tree};
use nuvec_core::{
    Array, DType, Device, Error, Result, Stream, array, grad, random, set_default_stream,
    value_and_grad, vmap,
};
use nuvec_macros::Tree;

fn on_cpu() {
    set_default_stream(Stream::default_for(Device::cpu()));
}

/// Whether a GPU is available (sets it as the default stream).
fn on_gpu() -> bool {
    set_default_stream(Stream::default_for(Device::gpu(0)));
    let ok = (&Array::from(1.0f32) + 1.0).eval().is_ok();
    if !ok {
        on_cpu();
    }
    ok
}

fn bits(a: &Array) -> Vec<u32> {
    a.astype(DType::F32)
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn same<T: Tree>(a: &T, b: &T) {
    assert_eq!(a.paths(), b.paths());
    for (x, y) in a.leaves().iter().zip(b.leaves()) {
        assert_eq!(bits(x), bits(&y));
    }
}

#[derive(Clone, Tree)]
#[tree(crate = "nuvec_core")]
struct Pair {
    a: Array,
    b: Array,
}

#[test]
fn tree_compile_equals_uncompiled() {
    on_cpu();
    let f = |(x, y): &(Array, Array)| (x.multiply(y).add(1.0), x.exp());
    let c = compile(f);
    let input = (array![1.0, 2.0], array![3.0, 4.0]);
    let (got, want) = (c(&input).unwrap(), f(&input));
    same(&got, &want);
    let g = |p: &Pair| p.a.subtract(&p.b).tanh();
    let c = compile(g);
    let p = Pair {
        a: array![1.0, -1.0],
        b: array![0.5, 0.5],
    };
    assert_eq!(bits(&c(&p).unwrap()), bits(&g(&p)));
    let h = |m: &BTreeMap<String, Array>| {
        m.values()
            .fold(Array::from(0.0f32), |acc, v| acc.add(v.sum(..)))
    };
    let c = compile(h);
    let m: BTreeMap<String, Array> = [
        ("x".to_owned(), array![1.0, 2.0]),
        ("y".to_owned(), array![3.0]),
    ]
    .into();
    assert_eq!(bits(&c(&m).unwrap()), bits(&h(&m)));
    // A function with no array inputs.
    let k = compile(|_: &()| array![1.0, 2.0]);
    assert_eq!(bits(&k(&()).unwrap()), bits(&array![1.0, 2.0]));
}

#[test]
fn structure_is_part_of_the_cache_key() {
    on_cpu();
    let traces = Cell::new(0);
    let f = compile(|t: &(Array, Array)| {
        traces.set(traces.get() + 1);
        t.0.add(&t.1)
    });
    let g = compile(|p: &Pair| {
        traces.set(traces.get() + 1);
        p.a.add(&p.b)
    });
    let (x, y) = (array![1.0], array![2.0]);
    for _ in 0..3 {
        let _ = f(&(x.clone(), y.clone())).unwrap();
        let _ = g(&Pair {
            a: x.clone(),
            b: y.clone(),
        })
        .unwrap();
    }
    assert_eq!(traces.get(), 2, "one trace per structure");
    // Different paths with the same leaves: a map with other keys.
    let traces = Cell::new(0);
    let m = compile(|m: &BTreeMap<String, Array>| {
        traces.set(traces.get() + 1);
        m.values().fold(Array::from(0.0f32), |acc, v| acc.add(v))
    });
    let one: BTreeMap<String, Array> = [("a".to_owned(), array![1.0])].into();
    let other: BTreeMap<String, Array> = [("b".to_owned(), array![1.0])].into();
    let _ = m(&one).unwrap();
    let _ = m(&other).unwrap();
    let _ = m(&one).unwrap();
    assert_eq!(traces.get(), 2);
}

#[test]
fn the_cache_is_bounded() {
    on_cpu();
    let traces = Cell::new(0);
    let f = compile(|x: &Array| {
        traces.set(traces.get() + 1);
        x.add(1.0)
    });
    for n in 1..=200 {
        let x = nuvec_core::ops::zeros([n], DType::F32);
        assert_eq!(
            f(&x).unwrap().to_vec::<f32>().unwrap(),
            vec![1.0; n as usize]
        );
    }
    assert_eq!(traces.get(), 200);
    // The most recent 64 shapes are cached; an evicted one traces again.
    let _ = f(&nuvec_core::ops::zeros([200], DType::F32)).unwrap();
    assert_eq!(traces.get(), 200, "recent: replayed");
    let _ = f(&nuvec_core::ops::zeros([1], DType::F32)).unwrap();
    assert_eq!(traces.get(), 201, "evicted: traced again");
    // Capacity 0 never caches.
    let traces = Cell::new(0);
    let never = Options::default().cache_capacity(0).compile(|x: &Array| {
        traces.set(traces.get() + 1);
        x.add(1.0)
    });
    let _ = never(&array![1.0]).unwrap();
    let _ = never(&array![1.0]).unwrap();
    assert_eq!(traces.get(), 2);
}

/// A tape that captures an outer tracer is correct (also on the CPU, where an equal
/// evaluated scalar constant used to replace the tracer) and not cached.
#[test]
fn tapes_that_capture_a_tracer_are_not_cached() {
    on_cpu();
    let traces = Cell::new(0);
    let x = array![2.0];
    let g = grad(|w: &Array| {
        // `w` is a tracer captured by the compiled function (its inputs are not tracers).
        let f = compile(|y: &Array| {
            traces.set(traces.get() + 1);
            y.multiply(w)
        });
        Ok(f(&x)?.sum(..).add(f(&x)?.sum(..)))
    })(&array![3.0])
    .unwrap();
    assert_eq!(g.to_vec::<f32>().unwrap(), [4.0], "d/dw (2xw) = 2x");
    assert_eq!(traces.get(), 2, "not cached");
}

#[test]
fn closures_may_borrow_and_compose() {
    on_cpu();
    let w = array![[1.0, 2.0], [3.0, 4.0]];
    let f = compile(|x: &Array| x.matmul(&w)); // borrows `w`: no `'static`
    let x = array![[1.0, 0.0]];
    assert_eq!(f(&x).unwrap().to_vec::<f32>().unwrap(), [1.0, 2.0]);
    // A configured vmap composes with compile.
    let batched = vmap(|(m, v): &(Array, Array)| m.matmul(v)).in_axes((None, 0));
    let vs = array![[1.0, 1.0], [2.0, 3.0]];
    let want = batched.call(&(w.clone(), vs.clone())).unwrap();
    let compiled = compile(batched.into_fn());
    assert_eq!(bits(&compiled(&(w.clone(), vs)).unwrap()), bits(&want));
    // The flat form is unchanged.
    let flat = compile_flat(|xs: &[Array]| Ok(vec![xs[0].add(&xs[1])]));
    assert_eq!(
        flat(&[array![1.0], array![2.0]]).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [3.0]
    );
}

#[test]
fn failed_inputs_give_failed_outputs() {
    on_cpu();
    let f = compile(|(a, b): &(Array, Array)| a.add(b).exp());
    let bad = array![1.0, 2.0].matmul(&array![1.0]); // a failed array
    assert!(bad.is_failed());
    let out = f(&(bad.clone(), array![1.0])).unwrap();
    assert!(
        out.is_failed(),
        "sticky: the error surfaces at the boundary"
    );
    assert_eq!(out.check().unwrap_err().op_name(), Some("matmul"));
}

/// A layer with a mode flag, an enum with a leaf-less variant and a hashed config.
#[derive(Clone, Tree)]
#[tree(crate = "nuvec_core")]
enum Gate {
    Scale(Param),
    Off,
}

#[derive(Clone, Tree)]
#[tree(crate = "nuvec_core")]
struct Layer {
    w: Param,
    gate: Gate,
    #[tree(skip)]
    eps: f32,
    #[tree(mode)]
    training: bool,
}

impl Layer {
    fn forward(&self, x: &Array) -> Array {
        let mut y = x.multiply(&self.w).add(f64::from(self.eps));
        if let Gate::Scale(s) = &self.gate {
            y = y.multiply(s);
        }
        if self.training { y.multiply(2.0) } else { y }
    }
}

#[test]
fn structure_changes_retrace() {
    on_cpu();
    let traces = Cell::new(0);
    let f = compile(|(l, x): &(Layer, Array)| {
        traces.set(traces.get() + 1);
        l.forward(x)
    });
    let x = array![1.0, 2.0];
    let mut l = Layer {
        w: Param::new(array![3.0, 4.0]),
        gate: Gate::Scale(Param::new(array![0.5])),
        eps: 0.0,
        training: true,
    };
    for _ in 0..100 {
        let _ = f(&(l.clone(), x.clone())).unwrap();
    }
    assert_eq!(traces.get(), 1, "100 unchanged calls trace once");
    let mut expect = 1;
    let mut check = |l: &Layer, what: &str| {
        let got = f(&(l.clone(), x.clone())).unwrap();
        expect += 1;
        assert_eq!(traces.get(), expect, "{what}: one retrace");
        assert_eq!(
            bits(&got),
            bits(&l.forward(&x)),
            "{what}: the uncompiled result"
        );
        let _ = f(&(l.clone(), x.clone())).unwrap();
        assert_eq!(traces.get(), expect, "{what}: replayed after");
    };
    l.training = false;
    check(&l, "eval mode");
    l.w.set_frozen(true);
    check(&l, "freeze");
    l.gate = Gate::Off;
    check(&l, "leaf-less variant");
    l.eps = 0.25;
    check(&l, "hashed f32");
}

// ── compile_mut ─────────────────────────────────────────────────────────────

#[derive(Clone, Tree)]
#[tree(crate = "nuvec_core")]
struct Mlp {
    w1: Param,
    w2: Param,
}

impl Mlp {
    fn new() -> Self {
        Self {
            w1: Param::new(array![[0.5, -0.25], [0.125, 1.0]]),
            w2: Param::new(array![[1.0], [-0.5]]),
        }
    }

    fn forward(&self, x: &Array) -> Array {
        x.matmul(&self.w1).tanh().matmul(&self.w2)
    }
}

/// SGD with momentum whose slots are created lazily at the first update, and a learning
/// rate array (a traced input).
#[derive(Clone, Tree)]
#[tree(crate = "nuvec_core")]
struct Sgd {
    lr: Array,
    slots: BTreeMap<String, Array>,
    #[tree(skip)]
    momentum: f32,
}

impl Sgd {
    fn new(lr: f32) -> Self {
        Self {
            lr: Array::from(lr),
            slots: BTreeMap::new(),
            momentum: 0.5,
        }
    }

    fn init(&mut self, m: &Mlp) {
        for (p, a) in m.flatten() {
            self.slots.insert(p, a.zeros_like());
        }
    }

    fn update(&mut self, m: &mut Mlp, g: &Mlp) {
        let grads: BTreeMap<String, Array> = g.flatten().into_iter().collect();
        *m = m.map(|p, a| {
            let v = match self.slots.get(p) {
                Some(v) => v.multiply(f64::from(self.momentum)).add(&grads[p]),
                None => grads[p].clone(),
            };
            self.slots.insert(p.to_owned(), v.clone());
            a.subtract(self.lr.multiply(&v))
        });
    }
}

fn loss(m: &Mlp, x: &Array, y: &Array) -> Array {
    let d = m.forward(x).subtract(y);
    d.multiply(&d).mean(..)
}

fn uncompiled_step((m, opt): &mut (Mlp, Sgd), (x, y): &(Array, Array)) -> Result<Array> {
    let (l, g) = value_and_grad(|m: &Mlp| loss(m, x, y))(m)?;
    opt.update(m, &g);
    Ok(l)
}

#[test]
fn compiled_steps_equal_uncompiled_steps() {
    for gpu in [false, true] {
        if gpu && !on_gpu() {
            continue;
        }
        if !gpu {
            on_cpu();
        }
        let batch = (array![[1.0, 2.0], [0.5, -1.0]], array![[1.0], [0.0]]);
        let traces = Cell::new(0);
        let step = compile_mut(|(m, opt): &mut (Mlp, Sgd), (x, y): &(Array, Array)| {
            traces.set(traces.get() + 1);
            let (l, g) = value_and_grad(|m: &Mlp| loss(m, x, y))(m)?;
            opt.update(m, &g);
            Ok(l)
        });
        let mut compiled = (Mlp::new(), Sgd::new(0.1));
        let mut plain = (Mlp::new(), Sgd::new(0.1));
        for _ in 0..5 {
            let a = step(&mut compiled, &batch).unwrap();
            let b = uncompiled_step(&mut plain, &batch).unwrap();
            assert_eq!(bits(&a), bits(&b), "loss (gpu: {gpu})");
        }
        same(&compiled, &plain);
        assert_eq!(traces.get(), 2, "lazy slots: one retrace (gpu: {gpu})");
    }
    on_cpu();
}

#[test]
fn initialised_state_traces_once() {
    on_cpu();
    let traces = Cell::new(0);
    let step = compile_mut(|(m, opt): &mut (Mlp, Sgd), (x, y): &(Array, Array)| {
        traces.set(traces.get() + 1);
        let (l, g) = value_and_grad(|m: &Mlp| loss(m, x, y))(m)?;
        opt.update(m, &g);
        Ok(l)
    });
    let mut state = (Mlp::new(), Sgd::new(0.1));
    state.1.init(&state.0);
    let batch = (array![[1.0, 2.0]], array![[1.0]]);
    for _ in 0..100 {
        let _ = step(&mut state, &batch).unwrap();
    }
    assert_eq!(traces.get(), 1);
    // A new learning rate is an input: no retrace, and it is used.
    let mut plain = state.clone();
    state.1.lr = Array::from(0.01f32);
    plain.1.lr = Array::from(0.01f32);
    let a = step(&mut state, &batch).unwrap();
    let b = uncompiled_step(&mut plain, &batch).unwrap();
    assert_eq!(traces.get(), 1);
    assert_eq!(bits(&a), bits(&b));
    same(&state, &plain);
}

/// A step that counts calls in a hashed host field changes its structure every call.
#[derive(Clone, Tree)]
#[tree(crate = "nuvec_core")]
struct Counted {
    w: Array,
    #[tree(skip)]
    calls: u64,
}

#[test]
fn a_structure_that_keeps_changing_is_an_error() {
    on_cpu();
    let step = compile_mut(|s: &mut Counted, x: &Array| {
        s.calls += 1;
        s.w = s.w.add(x);
        s.w.sum(..)
    });
    let mut s = Counted {
        w: array![1.0],
        calls: 0,
    };
    let _ = step(&mut s, &array![1.0]).unwrap();
    assert_eq!((s.calls, s.w.to_vec::<f32>().unwrap()), (1, vec![2.0]));
    let err = step(&mut s, &array![1.0]).unwrap_err();
    assert!(
        err.to_string()
            .contains("state structure keeps changing; call opt.init(&model)"),
        "{err}"
    );
    assert_eq!(
        (s.calls, s.w.to_vec::<f32>().unwrap()),
        (1, vec![2.0]),
        "unchanged"
    );
}

#[test]
fn a_failing_step_leaves_the_state_unchanged() {
    on_cpu();
    let step = compile_mut(|s: &mut Array, x: &Array| -> Result<Array> {
        *s = s.add(x);
        Err(Error::InvalidArgument("no".into()))
    });
    let mut s = array![1.0];
    assert!(step(&mut s, &array![1.0]).is_err());
    assert_eq!(s.to_vec::<f32>().unwrap(), [1.0]);
}

#[test]
fn random_keys_are_threaded_through_compiled_steps() {
    on_cpu();
    let draw = |s: &mut Array, x: &Array| {
        let mask = random::bernoulli(0.5, x.shape(), None).astype(DType::F32);
        *s = s.add(x.multiply(&mask));
        mask
    };
    let step = compile_mut(draw);
    random::seed(11);
    let mut s = Array::from_slice(&[0.0f32; 16], [16]);
    let x = Array::from_slice(&[1.0f32; 16], [16]);
    let compiled: Vec<Vec<u32>> = (0..4).map(|_| bits(&step(&mut s, &x).unwrap())).collect();
    let after_compiled = random::next_key().to_vec::<u32>().unwrap();
    random::seed(11);
    let mut t = Array::from_slice(&[0.0f32; 16], [16]);
    let plain: Vec<Vec<u32>> = (0..4).map(|_| bits(&draw(&mut t, &x))).collect();
    let after_plain = random::next_key().to_vec::<u32>().unwrap();
    assert_eq!(compiled, plain, "the uncompiled masks, also after replays");
    assert_ne!(
        compiled[0], compiled[1],
        "different masks on consecutive calls"
    );
    assert_eq!(
        after_compiled, after_plain,
        "the sequence advances as uncompiled"
    );
    assert_eq!(bits(&s), bits(&t));
    // A step that draws nothing leaves the thread's key where it was.
    let quiet = compile_mut(|s: &mut Array, x: &Array| {
        *s = s.add(x);
        s.sum(..)
    });
    random::seed(12);
    let mut u = array![0.0];
    let _ = quiet(&mut u, &array![1.0]).unwrap();
    let _ = quiet(&mut u, &array![1.0]).unwrap();
    let after = random::next_key().to_vec::<u32>().unwrap();
    random::seed(12);
    assert_eq!(after, random::next_key().to_vec::<u32>().unwrap());
}

#[test]
fn compile_mut_inside_a_transform_runs_directly() {
    on_cpu();
    let traces = Cell::new(0);
    let step = compile_mut(|s: &mut Array, x: &Array| {
        traces.set(traces.get() + 1);
        *s = s.multiply(x);
        s.sum(..)
    });
    let g = grad(|w: &Array| {
        let mut s = w.clone();
        step(&mut s, &array![3.0])
    })(&array![2.0])
    .unwrap();
    assert_eq!(g.to_vec::<f32>().unwrap(), [3.0]);
    let g2 = grad(|w: &Array| {
        let mut s = w.clone();
        step(&mut s, &array![3.0])
    })(&array![5.0])
    .unwrap();
    assert_eq!(g2.to_vec::<f32>().unwrap(), [3.0]);
    assert_eq!(
        traces.get(),
        2,
        "runs `f` each time, never caching tracer tapes"
    );
}
