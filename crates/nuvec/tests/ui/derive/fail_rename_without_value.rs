use nuvec::{Array, Tree};

#[derive(Clone, Tree)]
struct Bad {
    #[tree(rename)]
    a: Array,
}

fn main() {}
