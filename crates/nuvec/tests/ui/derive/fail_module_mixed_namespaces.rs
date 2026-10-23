use nuvec::Array;
use nuvec::macros::Module;

#[derive(Clone, Module)]
struct Bad {
    #[tree(skip)]
    #[module(skip)]
    freqs: Array,
}

fn main() {}
