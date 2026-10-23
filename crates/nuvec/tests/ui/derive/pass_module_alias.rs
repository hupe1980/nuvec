//! The bare-`Array` lint is syntactic: a type alias of `Array` is not caught (documented),
//! and `#[module(..)]` is an exact alias of `#[tree(..)]`.
use nuvec::macros::Module;
use nuvec::{Param, Tree};

type Weights = nuvec::Array;

#[derive(Clone, Module)]
struct Layer {
    weight: Param,
    aliased: Weights,
    #[module(skip)]
    freqs: nuvec::Array,
    #[module(buffer)]
    running: nuvec::Array,
}

fn main() {
    let l = Layer {
        weight: Param::new(nuvec::array![1.0]),
        aliased: nuvec::array![2.0],
        freqs: nuvec::array![3.0],
        running: nuvec::array![4.0],
    };
    assert_eq!(l.paths(), ["weight", "aliased", "running"]);
    let _ = &l.freqs;
}
