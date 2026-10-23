//! Bounds only where a type parameter is used: `T` visited, `C` hashed, `M` only in
//! `PhantomData` (no bound beyond `'static`).
use std::marker::PhantomData;

use nuvec::{Array, Static, Tree};

#[derive(Clone, Tree)]
struct Generic<T, C, M> {
    item: T,
    #[tree(skip)]
    config: C,
    marker: PhantomData<M>,
}

#[derive(Clone, Static)]
enum Act {
    Relu,
    Gelu { approx: bool },
}

#[derive(Clone)]
struct NotATree;

fn main() {
    let g: Generic<Vec<Array>, Act, NotATree> = Generic {
        item: vec![nuvec::array![1.0]],
        config: Act::Gelu { approx: true },
        marker: PhantomData,
    };
    assert_eq!(g.paths(), ["item.0"]);
    let _ = Act::Relu;
}
