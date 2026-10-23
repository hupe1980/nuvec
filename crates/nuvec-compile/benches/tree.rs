//! The host cost of trees: `value_and_grad` over a derived module of 100
//! leaves with hashed static fields against `value_and_grad_flat` on the same leaves, and
//! `def()` (the structure hash a compiled call computes) alone. Only the graph is built
//! (nothing is evaluated), so the numbers are the transforms' bookkeeping. Prints
//! `name<TAB>us_per_call`.

use std::hint::black_box;
use std::time::Instant;

use nuvec_core::tree::{Param, Tree};
use nuvec_core::{
    Array, DType, Device, Stream, ops, set_default_stream, value_and_grad, value_and_grad_flat,
};
use nuvec_macros::Tree;

#[derive(Clone, Tree)]
#[tree(crate = "nuvec_core")]
struct Layer {
    weight: Param,
    bias: Param,
    #[tree(skip)]
    eps: f32,
    #[tree(skip)]
    heads: i32,
}

#[derive(Clone, Tree)]
#[tree(crate = "nuvec_core")]
struct Model {
    layers: Vec<Layer>,
    #[tree(skip)]
    name: String,
}

fn loss(leaves: &[&Array]) -> Array {
    leaves
        .iter()
        .fold(Array::from(0.0f32), |acc, a| acc.add(a.sum(..)))
}

/// Microseconds per call of `f`, the best of 5 runs of `n` calls.
fn time(n: usize, f: &dyn Fn()) -> f64 {
    f();
    (0..5)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..n {
                f();
            }
            t.elapsed().as_secs_f64() / n as f64 * 1e6
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    // `cargo test --all-targets` runs bench binaries too: measure only under `cargo bench`.
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    set_default_stream(Stream::default_for(Device::cpu()));
    let model = Model {
        layers: (0..50)
            .map(|i| Layer {
                weight: Param::new(ops::zeros([4, 4], DType::F32)),
                bias: Param::new(ops::zeros([4], DType::F32)),
                eps: 1e-5,
                heads: i,
            })
            .collect(),
        name: "bench".into(),
    };
    assert_eq!(model.len(), 100);
    let leaves = model.leaves();
    let argnums: Vec<usize> = (0..leaves.len()).collect();
    let tree = value_and_grad(|m: &Model| {
        let all: Vec<&Array> = m
            .layers
            .iter()
            .flat_map(|l| [l.weight.value(), l.bias.value()])
            .collect();
        loss(&all)
    });
    let flat = value_and_grad_flat(
        |xs: &[Array]| Ok(loss(&xs.iter().collect::<Vec<_>>())),
        &argnums,
    );
    let n = 200;
    let t_tree = time(n, &|| {
        let _ = black_box(tree(&model).unwrap());
    });
    let t_flat = time(n, &|| {
        let _ = black_box(flat(&leaves).unwrap());
    });
    let t_def = time(n, &|| {
        black_box(model.def());
    });
    println!("value_and_grad_tree_100\t{t_tree:.2}");
    println!("value_and_grad_flat_100\t{t_flat:.2}");
    println!("def_100\t{t_def:.2}");
    println!("tree_over_flat\t{:.3}", t_tree / t_flat);
}
