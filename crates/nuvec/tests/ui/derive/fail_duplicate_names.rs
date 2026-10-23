use nuvec::{Array, Tree};

#[derive(Clone, Tree)]
struct Twice {
    a: Array,
    #[tree(rename = "a")]
    b: Array,
}

fn main() {}
