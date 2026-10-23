use nuvec::{Array, Tree};

trait Layer {}

#[derive(Tree)]
struct Bad {
    a: Array,
    layer: Box<dyn Layer>,
}

fn main() {}
