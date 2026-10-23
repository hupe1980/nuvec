use nuvec::macros::Module;

#[derive(Clone, Module)]
struct Bad {
    weights: Vec<nuvec::Array>,
}

fn main() {}
