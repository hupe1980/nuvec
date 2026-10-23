use nuvec::Array;
use nuvec::macros::Module;

#[derive(Clone, Module)]
struct Bad {
    weight: Array,
}

fn main() {}
