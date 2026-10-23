use nuvec::{Array, Tree};

#[derive(Clone)]
struct Host;

#[derive(Clone, Tree)]
struct Bad {
    a: Array,
    #[tree(skip)]
    host: Host,
}

fn main() {}
