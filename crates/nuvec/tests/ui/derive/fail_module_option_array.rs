use nuvec::Array;
use nuvec::macros::Module;

#[derive(Clone, Module)]
struct Bad {
    bias: Option<Array>,
}

fn main() {}
